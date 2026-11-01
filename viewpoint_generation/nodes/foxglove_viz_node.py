"""Foxglove visualization of the viewpoint-generation pipeline.

An alternative to the Open3D GUI's 3D view for remote use: it publishes each
pipeline stage's results as ROS topics that Foxglove's 3D panel renders
natively, so the plan can be inspected from anywhere the Foxglove bridge
reaches.

Like the GUI, it does not sit inside the pipeline. It follows the
`viewpoint_generation` node's `results.file`, `model.mesh.file` and
`model.mesh.units` parameters (every stage ends by setting `results.file`),
reads the files they name, and republishes whenever they change. Setting
this node's own `results_file` parameter instead pins it to one plan file,
for browsing old plans without touching the pipeline.

Topics (all latched, under ~/ = the `topic_prefix` parameter, /viewpoint_generation/viz):
  ~/mesh         foxglove_msgs/SceneUpdate  part, one entity per region (+ noise, unassigned)
  ~/points       sensor_msgs/PointCloud2    region samples; fields rgb, region, cluster
  ~/blind_spots  sensor_msgs/PointCloud2    points no viewpoint can see
  ~/viewpoints   foxglove_msgs/SceneUpdate  per viewpoint: marker, frustum, origin line
  ~/fov          foxglove_msgs/SceneUpdate  FOV coverage cylinders
  ~/path         foxglove_msgs/SceneUpdate  traversal order and joint paths
  ~/labels       foxglove_msgs/SceneUpdate  region/cluster text labels
  ~/selection    foxglove_msgs/SceneUpdate  highlight of ~/select's region/cluster
  ~/selection_points sensor_msgs/PointCloud2 the selected cluster's points
  ~/summary      std_msgs/String            JSON summary for the control panel
Input:
  ~/select       std_msgs/String            {"region": id, "cluster": id} (-1 = none)
TF:
  object_frame -> viewpoint_generation/model, from /tsdf_pose/pose (identity
  until one arrives) -- the same placement the planning scene uses -- and
  viewpoint_generation/model -> viewpoint_generation/selected_viewpoint.
"""

import json
import os
import struct
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from foxglove_msgs.msg import (Color, KeyValuePair, LinePrimitive, SceneEntity,
                               SceneEntityDeletion, SceneUpdate, SpherePrimitive,
                               TextPrimitive, TriangleListPrimitive)
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped, Vector3
from rcl_interfaces.msg import ParameterEvent
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from viewpoint_generation import foxglove_scene as fs

LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

WATCHED = ('results.file', 'model.mesh.file', 'model.mesh.units')
# STEP must be tessellated as the pipeline did it (see foxglove_scene.load_mesh).
TESSELLATION = ('model.mesh.tessellation.linear_deflection',
                'model.mesh.tessellation.angular_deflection',
                'model.mesh.tessellation.relative')


def _color(rgb, a=1.0):
    return Color(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def _points(arr):
    return [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in arr]


def _pack_rgb(rgb):
    r, g, b = (int(round(255 * c)) for c in rgb[:3])
    return struct.unpack('<f', struct.pack('<I', (r << 16) | (g << 8) | b))[0]


def _param_value(v):
    """Python value of an rcl_interfaces/ParameterValue (None when unset)."""
    return {1: v.bool_value, 2: v.integer_value, 3: v.double_value,
            4: v.string_value}.get(v.type)


def _kv(d):
    return [KeyValuePair(key=str(k), value=str(v)) for k, v in d.items()]


class FoxgloveVizNode(Node):

    def __init__(self):
        super().__init__('viewpoint_generation_viz')
        self.declare_parameter('target_node', '/viewpoint_generation')
        self.declare_parameter('topic_prefix', '/viewpoint_generation/viz')
        self.declare_parameter('results_file', '')   # pin to a plan file
        self.declare_parameter('frame_id', 'viewpoint_generation/model')
        self.declare_parameter('parent_frame', 'object_frame')
        self.declare_parameter('placement_topic', '/tsdf_pose/pose')
        self.declare_parameter('traversal_algorithm', '')
        self.declare_parameter('labels', True)

        self.frame = self.get_parameter('frame_id').value
        self.parent = self.get_parameter('parent_frame').value
        self.target = '/' + self.get_parameter('target_node').value.strip('/')
        prefix = self.get_parameter('topic_prefix').value.rstrip('/')

        names = ('mesh', 'viewpoints', 'fov', 'path', 'labels', 'selection')
        self.scene_pubs = {n: self.create_publisher(SceneUpdate, f'{prefix}/{n}', LATCHED)
                           for n in names}
        self.cloud_pubs = {n: self.create_publisher(PointCloud2, f'{prefix}/{n}', LATCHED)
                           for n in ('points', 'blind_spots', 'selection_points')}
        self.summary_pub = self.create_publisher(String, f'{prefix}/summary', LATCHED)
        self.create_subscription(String, f'{prefix}/select', self._on_select, 10)

        self.tf = TransformBroadcaster(self)
        self.placement = Pose()
        self.placement.orientation.w = 1.0
        self.selected_vp = None
        self.create_subscription(PoseStamped, self.get_parameter('placement_topic').value,
                                 self._on_placement, 10)
        self.create_timer(0.1, self._broadcast_tf)

        self.target_params = {'results.file': '', 'model.mesh.file': '', 'model.mesh.units': 'm'}
        self.scene = None
        self.selection = (-1, -1)
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._last_key = None

        self.create_subscription(ParameterEvent, '/parameter_events', self._on_param_event, 50)
        self.add_on_set_parameters_callback(self._on_own_params)
        self.get_client = self.create_client(
            GetParameters, f'{self.target}/get_parameters',
            callback_group=MutuallyExclusiveCallbackGroup())
        self._fetch_timer = self.create_timer(1.0, self._fetch_target_params)

        threading.Thread(target=self._worker, daemon=True, name='foxglove-viz').start()
        self._dirty.set()
        self.get_logger().info(
            f'Foxglove viz following {self.target} '
            f'(frame {self.parent} -> {self.frame}); topics under {prefix}')

    # --- inputs ------------------------------------------------------------------
    def _fetch_target_params(self):
        """Initial read of the pipeline node's parameters (retried until it answers)."""
        if not self.get_client.service_is_ready():
            return
        self._fetch_timer.cancel()
        req = GetParameters.Request(names=list(WATCHED + TESSELLATION))
        fut = self.get_client.call_async(req)

        def done(f):
            res = f.result()
            if res is None:
                self._fetch_timer.reset()
                return
            for name, value in zip(WATCHED + TESSELLATION, res.values):
                self.target_params[name] = _param_value(value)
            self._dirty.set()
        fut.add_done_callback(done)

    def _on_param_event(self, msg):
        if msg.node != self.target:
            return
        changed = False
        for p in list(msg.new_parameters) + list(msg.changed_parameters):
            if p.name in WATCHED + TESSELLATION:
                self.target_params[p.name] = _param_value(p.value)
                changed = True
        if changed:
            self._dirty.set()

    def _on_own_params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        if any(p.name in ('results_file', 'traversal_algorithm', 'labels') for p in params):
            # The worker waits 0.2 s before reading parameters, by which time
            # this change has been stored.
            self._dirty.set()
        return SetParametersResult(successful=True)

    def _on_placement(self, msg):
        self.placement = msg.pose

    def _on_select(self, msg):
        try:
            d = json.loads(msg.data)
            sel = (int(d.get('region', -1)), int(d.get('cluster', -1)))
        except (ValueError, TypeError, AttributeError):
            self.get_logger().warning(f'bad selection {msg.data!r}')
            return
        self.selection = sel
        with self._lock:
            scene = self.scene
        if scene is not None:
            self._publish_selection(scene)

    # --- loading -----------------------------------------------------------------
    def _worker(self):
        while rclpy.ok():
            self._dirty.wait()
            self._dirty.clear()
            time.sleep(0.2)  # coalesce bursts of parameter events
            pinned = self.get_parameter('results_file').value
            results = pinned or self.target_params['results.file']
            mesh = '' if pinned else self.target_params['model.mesh.file']
            units = self.target_params['model.mesh.units'] or 'm'
            algo = self.get_parameter('traversal_algorithm').value or None
            tess = {name.rsplit('.', 1)[1]: self.target_params[name] for name in TESSELLATION
                    if self.target_params.get(name) is not None}
            mtime = os.path.getmtime(fs.resolve_path(results)) if results and os.path.exists(
                fs.resolve_path(results)) else None
            key = (results, mtime, mesh, units, algo, tuple(sorted(tess.items())),
                   self.get_parameter('labels').value)
            if key == self._last_key:
                continue
            self._last_key = key
            t0 = time.monotonic()
            try:
                scene = fs.load_scene(results, mesh, units, algorithm=algo, tessellation=tess)
            except Exception as e:  # noqa: BLE001 -- keep the node alive
                self.get_logger().error(f'failed to load scene: {e}')
                continue
            for w in scene.warnings:
                self.get_logger().warning(w)
            with self._lock:
                self.scene = scene
            self._publish_all(scene)
            s = scene.summary()
            self.get_logger().info(
                f"{s['stage'] or 'empty'}: {len(s['regions'])} regions, {s['num_clusters']} "
                f"clusters, {s['num_viewpoints']} viewpoints, {s['triangles']} triangles "
                f'({time.monotonic() - t0:.1f} s) from {results or mesh or "nothing"}')

    # --- publishing --------------------------------------------------------------
    def _entity(self, eid, stamp, **kw):
        return SceneEntity(timestamp=stamp, frame_id=self.frame, id=eid,
                           lifetime=DurationMsg(), frame_locked=True, **kw)

    def _replace(self, topic, entities, stamp):
        upd = SceneUpdate(deletions=[SceneEntityDeletion(timestamp=stamp, type=1)],
                          entities=entities)
        self.scene_pubs[topic].publish(upd)

    def _cloud(self, points, fields_values, stamp):
        """PointCloud2 with x,y,z plus float32 fields (rgb packed)."""
        names = ['x', 'y', 'z'] + list(fields_values)
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        msg.height, msg.width = 1, len(points)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(names)]
        msg.is_bigendian, msg.point_step, msg.is_dense = False, 4 * len(names), True
        msg.row_step = msg.point_step * len(points)
        cols = [np.asarray(points, dtype=np.float32).reshape(-1, 3)]
        cols += [np.asarray(v, dtype=np.float32).reshape(-1, 1) for v in fields_values.values()]
        msg.data = np.hstack(cols).astype(np.float32).tobytes() if len(points) else b''
        return msg

    def _publish_all(self, scene):
        stamp = self.get_clock().now().to_msg()
        self._publish_mesh(scene, stamp)
        self._publish_points(scene, stamp)
        self._publish_viewpoints(scene, stamp)
        self._publish_path(scene, stamp)
        self._publish_labels(scene, stamp)
        self._publish_selection(scene)
        summary = scene.summary()
        for r, sr in zip(scene.regions, summary['regions']):
            sr['cluster_ids'] = [c.cluster_id for c in r.clusters]
        self.summary_pub.publish(String(data=json.dumps(summary)))

    def _tri_entity(self, eid, scene, faces, rgb, stamp, alpha=1.0, meta=None):
        pts = scene.vertices[scene.triangles[faces]].reshape(-1, 3)
        prim = TriangleListPrimitive(points=_points(pts), color=_color(rgb, alpha))
        prim.pose.orientation.w = 1.0
        return self._entity(eid, stamp, triangles=[prim], metadata=_kv(meta or {}))

    def _publish_mesh(self, scene, stamp):
        ents = []
        if scene.triangles is not None:
            assigned = np.zeros(len(scene.triangles), dtype=bool)
            for r in scene.regions:
                if r.faces.size == 0:
                    continue
                assigned[r.faces] = True
                meta = {'region': r.region_id, 'faces': len(r.faces),
                        'clusters': len(r.clusters), 'blind_spots': len(r.blind_spots),
                        **r.metadata}
                ents.append(self._tri_entity(f'region_{r.region_id}', scene, r.faces,
                                             r.color, stamp, meta=meta))
            if scene.noise_faces.size:
                assigned[scene.noise_faces] = True
                ents.append(self._tri_entity('noise', scene, scene.noise_faces, fs.NOISE_COLOR,
                                             stamp, meta={'noise_faces': len(scene.noise_faces)}))
            rest = np.flatnonzero(~assigned)
            if rest.size:
                ents.append(self._tri_entity(
                    'part' if not scene.regions else 'unassigned', scene, rest, (0.7, 0.7, 0.72),
                    stamp, meta={'file': scene.mesh_file, 'faces': int(rest.size)}))
        self._replace('mesh', ents, stamp)

    def _publish_points(self, scene, stamp):
        pts, rgb, region, cluster = [], [], [], []
        bpts = []
        for r in scene.regions:
            if r.points is None or not len(r.points):
                continue
            n = len(r.points)
            col = np.full(n, _pack_rgb(r.color), dtype=np.float32)
            cl = np.full(n, -1.0, dtype=np.float32)
            for c in r.clusters:
                idx = c.indices[c.indices < n]
                col[idx] = _pack_rgb(c.color)
                cl[idx] = c.position
            pts.append(r.points)
            rgb.append(col)
            region.append(np.full(n, r.region_id, dtype=np.float32))
            cluster.append(cl)
            if r.blind_spots.size:
                bpts.append(r.points[r.blind_spots[r.blind_spots < n]])
        if pts:
            msg = self._cloud(np.vstack(pts), {'rgb': np.concatenate(rgb),
                                               'region': np.concatenate(region),
                                               'cluster': np.concatenate(cluster)}, stamp)
        else:
            msg = self._cloud(np.zeros((0, 3)), {'rgb': [], 'region': [], 'cluster': []}, stamp)
        self.cloud_pubs['points'].publish(msg)
        b = np.vstack(bpts) if bpts else np.zeros((0, 3))
        self.cloud_pubs['blind_spots'].publish(
            self._cloud(b, {'rgb': np.full(len(b), _pack_rgb(fs.BLIND_SPOT_COLOR))}, stamp))

    def _viewpoint_prims(self, scene, vp, rgb, thickness=1.5, alpha=1.0):
        fov_d = scene.camera_config.get('fov_diameter') or 0.03
        depth = float(np.linalg.norm(np.subtract(vp['position'], vp['origin']))) or \
            float(scene.camera_config.get('focal_distance') or 0.3)
        mode_rgb = fs.IMAGING_MODE_COLORS.get(vp.get('imaging_mode', 'photometric'), (1, 1, 1))
        frustum = LinePrimitive(type=2, thickness=thickness, scale_invariant=True,
                                points=_points(fs.frustum_segments(vp, fov_d, fov_d, depth)),
                                color=_color(rgb, alpha))
        frustum.pose.orientation.w = 1.0
        sphere = SpherePrimitive(size=Vector3(x=0.008, y=0.008, z=0.008),
                                 color=_color(mode_rgb, alpha))
        sphere.pose.position = Point(x=float(vp['position'][0]), y=float(vp['position'][1]),
                                     z=float(vp['position'][2]))
        sphere.pose.orientation.w = 1.0
        return frustum, sphere

    def _publish_viewpoints(self, scene, stamp):
        vps, fovs = [], []
        fov_d = scene.camera_config.get('fov_diameter')
        dof = scene.camera_config.get('dof')
        for r in scene.regions:
            for c in r.clusters:
                vp = c.viewpoint
                if not vp:
                    continue
                frustum, sphere = self._viewpoint_prims(scene, vp, c.color)
                meta = {'region': r.region_id, 'cluster': c.cluster_id, 'order': c.position,
                        'imaging_mode': vp.get('imaging_mode', 'photometric'),
                        'points': len(c.indices),
                        'unreachable': c.cluster_id in r.unreachable,
                        'position': ', '.join(f'{v:.4f}' for v in vp['position'])}
                vps.append(self._entity(f'vp_{r.region_id}_{c.cluster_id}', stamp,
                                        lines=[frustum], spheres=[sphere], metadata=_kv(meta)))
                segs = fs.fov_cylinder_segments(vp, fov_d, dof)
                if segs is not None:
                    cyl = LinePrimitive(type=2, thickness=1.0, scale_invariant=True,
                                        points=_points(segs), color=_color(c.color, 0.9))
                    cyl.pose.orientation.w = 1.0
                    fovs.append(self._entity(f'fov_{r.region_id}_{c.cluster_id}', stamp,
                                             lines=[cyl], metadata=_kv(meta)))
        self._replace('viewpoints', vps, stamp)
        self._replace('fov', fovs, stamp)

    def _publish_path(self, scene, stamp):
        ents = []
        for r in scene.regions:
            lines, spheres = [], []
            order_pts = [c.viewpoint['position'] for c in r.clusters if c.viewpoint]
            if len(order_pts) >= 2:
                ln = LinePrimitive(type=0, thickness=2.0, scale_invariant=True,
                                   points=_points(order_pts), color=_color(r.color, 0.8))
                ln.pose.orientation.w = 1.0
                lines.append(ln)
            if r.joint_path is not None and len(r.joint_path) >= 2:
                jp = LinePrimitive(type=0, thickness=3.0, scale_invariant=True,
                                   points=_points(r.joint_path), color=_color((1.0, 0.2, 0.2)))
                jp.pose.orientation.w = 1.0
                lines.append(jp)
            for c in r.clusters:
                if c.cluster_id in r.unreachable and c.viewpoint:
                    s = SpherePrimitive(size=Vector3(x=0.015, y=0.015, z=0.015),
                                        color=_color((1.0, 0.1, 0.1), 0.6))
                    s.pose.position = Point(x=float(c.viewpoint['position'][0]),
                                            y=float(c.viewpoint['position'][1]),
                                            z=float(c.viewpoint['position'][2]))
                    s.pose.orientation.w = 1.0
                    spheres.append(s)
            if lines or spheres:
                ents.append(self._entity(f'path_{r.region_id}', stamp, lines=lines,
                                         spheres=spheres,
                                         metadata=_kv({'region': r.region_id,
                                                       'viewpoints': len(order_pts),
                                                       'unreachable': len(r.unreachable)})))
        self._replace('path', ents, stamp)

    def _publish_labels(self, scene, stamp):
        ents = []
        if self.get_parameter('labels').value and scene.triangles is not None:
            for r in scene.regions:
                if r.faces.size == 0:
                    continue
                centre = scene.vertices[scene.triangles[r.faces]].reshape(-1, 3).mean(axis=0)
                t = TextPrimitive(billboard=True, font_size=14.0, scale_invariant=True,
                                  color=_color((1, 1, 1)), text=f'R{r.region_id}')
                t.pose.position = Point(x=float(centre[0]), y=float(centre[1]), z=float(centre[2]))
                t.pose.orientation.w = 1.0
                ents.append(self._entity(f'label_{r.region_id}', stamp, texts=[t]))
        self._replace('labels', ents, stamp)

    def _publish_selection(self, scene):
        stamp = self.get_clock().now().to_msg()
        rid, cid = self.selection
        region = next((r for r in scene.regions if r.region_id == rid), None)
        ents, sel_pts = [], np.zeros((0, 3))
        self.selected_vp = None
        if region is not None and scene.triangles is not None and region.faces.size:
            tri = scene.vertices[scene.triangles[region.faces]]          # (F,3,3)
            edges = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 1).reshape(-1, 3)
            wire = LinePrimitive(type=2, thickness=1.0, scale_invariant=True,
                                 points=_points(edges), color=_color((0.2, 0.45, 1.0)))
            wire.pose.orientation.w = 1.0
            ents.append(self._entity('selected_region', stamp, lines=[wire],
                                     metadata=_kv({'region': rid})))
            cluster = next((c for c in region.clusters if c.cluster_id == cid), None)
            if cluster is not None:
                if region.points is not None:
                    sel_pts = region.points[cluster.indices[cluster.indices < len(region.points)]]
                if cluster.viewpoint:
                    frustum, sphere = self._viewpoint_prims(scene, cluster.viewpoint,
                                                            (0.2, 0.45, 1.0), thickness=4.0)
                    sphere.size = Vector3(x=0.014, y=0.014, z=0.014)
                    ents.append(self._entity('selected_viewpoint', stamp, lines=[frustum],
                                             spheres=[sphere],
                                             metadata=_kv({'region': rid, 'cluster': cid})))
                    self.selected_vp = cluster.viewpoint
        self._replace('selection', ents, stamp)
        self.cloud_pubs['selection_points'].publish(
            self._cloud(sel_pts, {'rgb': np.full(len(sel_pts), _pack_rgb((0.2, 0.45, 1.0)))},
                        stamp))

    def _broadcast_tf(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.parent
        t.child_frame_id = self.frame
        p = self.placement
        t.transform.translation.x = p.position.x
        t.transform.translation.y = p.position.y
        t.transform.translation.z = p.position.z
        t.transform.rotation = p.orientation
        tfs = [t]
        vp = self.selected_vp
        if vp:
            s = TransformStamped()
            s.header.stamp = now
            s.header.frame_id = self.frame
            s.child_frame_id = 'viewpoint_generation/selected_viewpoint'
            s.transform.translation.x, s.transform.translation.y, s.transform.translation.z = (
                float(v) for v in vp['position'])
            q = vp['orientation']  # xyzw
            s.transform.rotation.x, s.transform.rotation.y = float(q[0]), float(q[1])
            s.transform.rotation.z, s.transform.rotation.w = float(q[2]), float(q[3])
            tfs.append(s)
        self.tf.sendTransform(tfs)


def main(args=None):
    rclpy.init(args=args)
    node = FoxgloveVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
