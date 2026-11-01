"""Scene model for the Foxglove visualization of a viewpoint-generation plan.

Turns a results/plan file (plus its mesh and per-region point clouds) into
plain numpy geometry, ready for `nodes/foxglove_viz_node.py` to pack into
messages. No ROS and no Open3D GUI imports, so it can be tested and used
headless.

Conventions match the Open3D GUI (`visualizer.py`) so the two agree on what a
plan looks like:

- Everything is in the mesh's origin frame (`model_frame`), in metres. The
  part's physical placement is a separate transform, never baked in.
- Region colours come from the same colormap and seeded draw, in region
  order; cluster colours vary the region hue the same way, numbered by the
  cluster's position in the region's traversal order.
- A region's `order` is a list, or after traversal optimisation a dict keyed
  by TSP algorithm (`resolve_order`).
"""

import colorsys
import json
import os
from dataclasses import dataclass, field

import numpy as np
from matplotlib import colormaps

from viewpoint_generation.mesh_utils import STEP_EXTENSIONS, mesh_file_units, read_mesh_file

# Materials.regions_colormap (visualizer's final assignment).
REGIONS_COLORMAP = 'rainbow'

UNIT_SCALE = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'in': 0.0254}

IMAGING_MODE_COLORS = {
    # Materials.viewpoint_marker_material / standard_... / unreachable_...
    'photometric': (0.1, 0.85, 0.2),
    'standard': (1.0, 0.85, 0.1),
    'inaccessible': (1.0, 0.1, 0.1),
}
BLIND_SPOT_COLOR = (1.0, 0.0, 1.0)
NOISE_COLOR = (1.0, 0.0, 0.0)


def resolve_order(order, algorithm=None):
    """A region's traversal order as a flat list of cluster indices.

    Same rules as visualizer._resolve_order_indices: a plain list is the
    order; a dict is keyed by algorithm, each entry either the bare list or
    {'order': [...], ...}; prefer `algorithm`, else the first entry.
    """
    if isinstance(order, dict):
        if not order:
            return []
        entry = order.get(algorithm, next(iter(order.values())))
        if isinstance(entry, dict):
            return list(entry.get('order', []))
        return list(entry)
    return list(order or [])


def _order_entry(order, algorithm=None):
    """The per-algorithm dict of a traversal-optimised order, or None."""
    if not isinstance(order, dict) or not order:
        return None
    entry = order.get(algorithm, next(iter(order.values())))
    return entry if isinstance(entry, dict) else None


def resolve_path(path, catalog_root=None):
    """A results file's recorded path, as seen from this process.

    Plans store absolute paths as they were when written, and the catalog has
    been mounted at different places over time (e.g. an older
    /workspaces/shared_ws/catalog vs today's /catalog). When the path does not
    exist, re-root everything after its last `/catalog/` component onto
    `catalog_root` ($CATALOG_ROOT, default /catalog).
    """
    if not path or os.path.exists(path):
        return path
    root = catalog_root or os.environ.get('CATALOG_ROOT', '/catalog')
    marker = '/catalog/'
    if marker in path:
        candidate = os.path.join(root, path.rsplit(marker, 1)[1])
        if os.path.exists(candidate):
            return candidate
    return path


def region_colors(region_order):
    """{region_id: rgb} exactly as the Open3D GUI draws them."""
    cmap = colormaps[REGIONS_COLORMAP]
    rng = np.random.RandomState(1)  # == np.random.seed(1) then np.random.rand()
    return {rid: tuple(float(c) for c in cmap(rng.rand())[:3]) for rid in region_order}


def cluster_color(region_rgb, position):
    """Cluster colour from its region's hue and its position in the order."""
    h, s, _v = colorsys.rgb_to_hsv(*region_rgb)
    golden = 0.6180339887498949
    frac_v = (position * golden) % 1.0
    frac_h = (position * golden * 2.0) % 1.0
    h_var = (h + 0.15 * (frac_h - 0.5)) % 1.0
    s_var = float(np.clip(max(s, 0.5), 0.0, 1.0))
    v_var = 0.45 + 0.5 * frac_v
    return tuple(float(np.clip(c, 0, 1)) for c in colorsys.hsv_to_rgb(h_var, s_var, v_var))


def read_ply_points(path):
    """(points, normals) from a PLY written by Open3D; normals may be None."""
    import open3d as o3d  # local: keep module import cheap for tests
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    nrm = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
    return pts, nrm


@dataclass
class Cluster:
    cluster_id: int          # index into the region's 'clusters'
    position: int            # index in the region's traversal order
    indices: np.ndarray      # into the region's point cloud
    color: tuple
    viewpoint: dict | None   # origin, position, direction, orientation (xyzw), imaging_mode


@dataclass
class Region:
    region_id: int
    faces: np.ndarray
    color: tuple
    points: np.ndarray | None = None
    clusters: list = field(default_factory=list)
    blind_spots: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    blind_reason: str = ''
    joint_path: np.ndarray | None = None   # cartesian_waypoints (N,3)
    unreachable: list = field(default_factory=list)  # cluster ids
    metadata: dict = field(default_factory=dict)


@dataclass
class PlanScene:
    results_file: str = ''
    mesh_file: str = ''
    vertices: np.ndarray | None = None     # (V,3) metres, model frame
    triangles: np.ndarray | None = None    # (T,3)
    regions: list = field(default_factory=list)
    noise_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    camera_config: dict = field(default_factory=dict)
    stage: str = ''
    warnings: list = field(default_factory=list)

    def summary(self):
        """Compact JSON-able description, for the control panel."""
        n_clusters = sum(len(r.clusters) for r in self.regions)
        n_vps = sum(1 for r in self.regions for c in r.clusters if c.viewpoint)
        modes = {}
        for r in self.regions:
            for c in r.clusters:
                if c.viewpoint:
                    m = c.viewpoint.get('imaging_mode', 'photometric')
                    modes[m] = modes.get(m, 0) + 1
        return {
            'results_file': self.results_file,
            'mesh_file': self.mesh_file,
            'stage': self.stage,
            'triangles': 0 if self.triangles is None else int(len(self.triangles)),
            'regions': [{
                'id': r.region_id,
                'faces': int(len(r.faces)),
                'points': 0 if r.points is None else int(len(r.points)),
                'clusters': len(r.clusters),
                'viewpoints': sum(1 for c in r.clusters if c.viewpoint),
                'blind_spots': int(len(r.blind_spots)),
                'color': '#%02x%02x%02x' % tuple(int(255 * c) for c in r.color),
            } for r in self.regions],
            'num_clusters': n_clusters,
            'num_viewpoints': n_vps,
            'imaging_modes': modes,
            'blind_spots': int(sum(len(r.blind_spots) for r in self.regions)),
            'noise_faces': int(len(self.noise_faces)),
            'camera_config': self.camera_config,
            'warnings': self.warnings,
        }


def infer_stage(mesh_entry):
    regions = mesh_entry.get('regions', [])
    if not regions:
        return 'mesh'
    clusters = [c for r in regions for c in r.get('clusters', [])]
    if not clusters:
        return 'segmented'
    if not any('viewpoint' in c for c in clusters):
        return 'clustered'
    if any(isinstance(r.get('order'), dict) for r in regions):
        return 'optimized'
    return 'projected'


_mesh_cache = {}


def load_mesh(mesh_file, units, tessellation=None):
    """(vertices, triangles, error), metres. Cached by path, units, mtime, tessellation.

    A plan's regions are lists of triangle indices, so the mesh must be
    tessellated exactly as the pipeline tessellated it: STEP goes through
    `cad_loader.step_loader.load_step` with the pipeline's TessellationConfig
    (`tessellation`, the `model.mesh.tessellation.*` values), as
    ViewpointGeneration._load_step_mesh does -- not `read_mesh_file`, whose
    default tessellation is coarser and would scramble every region.
    """
    path = mesh_file
    tess = dict(tessellation or {})
    key = (path, units, os.path.getmtime(path) if os.path.exists(path) else None,
           tuple(sorted(tess.items())))
    if key in _mesh_cache:
        return _mesh_cache[key]
    if os.path.splitext(path)[1].lower() in STEP_EXTENSIONS:
        try:
            from cad_loader.step_loader import TessellationConfig, load_step
            mesh, error = load_step(path, tess_config=TessellationConfig(**tess)).mesh, ''
        except Exception as e:  # noqa: BLE001 -- reported, not raised
            mesh, error = None, f'could not load STEP {path}: {e}'
        scale = 1.0  # load_step returns metres
    else:
        mesh, error = read_mesh_file(path)
        scale = UNIT_SCALE.get(mesh_file_units(path, units), 1.0)
    if mesh is None:
        return None, None, error or f'could not read {path}'
    verts = np.asarray(mesh.vertices, dtype=np.float64) * scale
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    _mesh_cache.clear()  # one part at a time
    _mesh_cache[key] = (verts, tris, '')
    return _mesh_cache[key]


def load_scene(results_file='', mesh_file='', mesh_units='m', algorithm=None,
               catalog_root=None, tessellation=None):
    """Build a PlanScene from a results file, or from a bare mesh.

    Args:
        results_file: plan/results JSON; '' shows the mesh alone.
        mesh_file, mesh_units: used when results_file is empty, or names no mesh.
        algorithm: preferred TSP algorithm for optimised orders.
        catalog_root: see resolve_path().
        tessellation: TessellationConfig fields for STEP; see load_mesh().
    """
    scene = PlanScene()
    entry = {}
    if results_file:
        rf = resolve_path(results_file, catalog_root)
        scene.results_file = rf
        try:
            with open(rf) as f:
                data = json.load(f)
            entry = (data.get('meshes') or [{}])[0]
        except (OSError, ValueError) as e:
            scene.warnings.append(f'cannot read results {results_file}: {e}')

    raw_mesh = entry.get('file') or mesh_file
    units = entry.get('units') or mesh_units
    scene.mesh_file = resolve_path(raw_mesh, catalog_root)
    scene.stage = infer_stage(entry) if entry else ('mesh' if raw_mesh else '')
    scene.camera_config = dict(entry.get('camera_config', {}))
    if not raw_mesh:
        return scene

    verts, tris, error = load_mesh(scene.mesh_file, units, tessellation)
    if verts is None:
        scene.warnings.append(f'cannot load mesh {scene.mesh_file}: {error}')
        return scene
    scene.vertices, scene.triangles = verts, tris

    region_entries = entry.get('regions', [])
    region_order = entry.get('order') or list(range(len(region_entries)))
    colors = region_colors(region_order)
    n_tri = len(tris)

    for rid in region_order:
        if not 0 <= rid < len(region_entries):
            continue
        rd = region_entries[rid]
        faces = np.asarray(rd.get('faces', []), dtype=np.int64)
        if faces.size and faces.max() >= n_tri:
            scene.warnings.append(
                f'region {rid} indexes face {int(faces.max())} but the mesh has {n_tri} '
                'triangles: the mesh was tessellated differently from when the plan was made')
            faces = faces[faces < n_tri]
        region = Region(region_id=rid, faces=faces, color=colors[rid])
        region.metadata = {k: v for k, v in rd.items()
                           if k not in ('faces', 'clusters', 'order', 'blind_spots', 'point_cloud')
                           and not isinstance(v, (list, dict))}

        pc_file = resolve_path(rd.get('point_cloud', {}).get('file', ''), catalog_root)
        if pc_file and os.path.exists(pc_file):
            region.points, _ = read_ply_points(pc_file)
        elif pc_file:
            scene.warnings.append(f'region {rid}: point cloud {pc_file} not found')

        bs = rd.get('blind_spots') or {}
        region.blind_spots = np.asarray(bs.get('points', []), dtype=np.int64)
        region.blind_reason = bs.get('reason', '') or ''

        clusters = rd.get('clusters', [])
        cluster_order = resolve_order(rd.get('order', []), algorithm) or list(range(len(clusters)))
        for pos, cid in enumerate(cluster_order):
            if not 0 <= cid < len(clusters):
                continue
            cd = clusters[cid]
            region.clusters.append(Cluster(
                cluster_id=cid, position=pos,
                indices=np.asarray(cd.get('points', []), dtype=np.int64),
                color=cluster_color(region.color, pos),
                viewpoint=cd.get('viewpoint')))

        oe = _order_entry(rd.get('order'), algorithm)
        jt = (oe or {}).get('joint_trajectory') or {}
        if jt.get('cartesian_waypoints'):
            region.joint_path = np.asarray(jt['cartesian_waypoints'], dtype=np.float64)
        region.unreachable = [u for u in jt.get('unreachable', []) if isinstance(u, int)]
        scene.regions.append(region)

    scene.noise_faces = np.asarray(entry.get('noise_faces', []), dtype=np.int64)
    scene.noise_faces = scene.noise_faces[scene.noise_faces < n_tri]
    return scene


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def frustum_segments(viewpoint, fov_w, fov_h, depth):
    """Line-list endpoints (16,3) of a camera frustum, apex at the camera.

    Camera convention as in visualizer._build_frustum: +Z forward, +X right,
    +Y up. `depth` is metres along the view axis (normally the focal
    distance, so the far rectangle is the imaged patch on the surface).
    """
    pos = np.asarray(viewpoint['position'], dtype=float)
    R = quat_xyzw_to_matrix(viewpoint['orientation'])
    right, up, fwd = R[:, 0], R[:, 1], R[:, 2]
    far = pos + fwd * depth
    hw, hh = fov_w / 2.0, fov_h / 2.0
    tl, tr = far + up * hh - right * hw, far + up * hh + right * hw
    bl, br = far - up * hh - right * hw, far - up * hh + right * hw
    pairs = [(pos, tl), (pos, tr), (pos, bl), (pos, br),
             (tl, tr), (tr, br), (br, bl), (bl, tl)]
    return np.array([p for pair in pairs for p in pair])


def fov_cylinder_segments(viewpoint, fov_diameter, dof, n=24, struts=8):
    """Line-list endpoints of the FOV coverage cylinder at the surface origin."""
    center = np.asarray(viewpoint['origin'], dtype=float)
    axis = np.asarray(viewpoint['direction'], dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or not fov_diameter or not dof:
        return None
    axis = axis / norm
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    r, hh = fov_diameter / 2.0, dof / 2.0
    top, bot = center + hh * axis + r * ring, center - hh * axis + r * ring
    segs = []
    for i in range(n):
        j = (i + 1) % n
        segs += [top[i], top[j], bot[i], bot[j]]
    for i in range(0, n, max(1, n // struts)):
        segs += [top[i], bot[i]]
    return np.array(segs)
