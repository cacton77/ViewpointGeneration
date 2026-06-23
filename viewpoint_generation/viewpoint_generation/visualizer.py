#!/usr/bin/env python3
"""visualizer.py – Scene geometry management and cluster-viewpoint rendering strategies."""

from __future__ import annotations

import colorsys
import copy
import enum
import json
import math
from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from matplotlib import colormaps

from viewpoint_generation.assets.materials import Materials


# ─────────────────────────────────────────────────────────────────────────────
# Rendering Mode Enum
# ─────────────────────────────────────────────────────────────────────────────

class ClusterViewpointMode(enum.Enum):
    """Available rendering strategies for cluster-viewpoint pairs."""
    CONVEX_HULL    = "convex_hull"    # Convex hull of the cluster points (default)
    CLUSTER_CLOUD  = "cluster_cloud"  # Raw point cloud per cluster
    FRUSTUM        = "frustum"        # Camera frustum at the viewpoint
    LINES          = "lines"          # Line from viewpoint to cluster centroid
    VIEWPOINT_ONLY = "viewpoint_only" # Only viewpoint markers, no cluster footprint
    ORIGIN_SPHERE  = "origin_sphere"  # Small sphere at the surface origin of each cluster


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_order_indices(order, selected_algorithm=None):
    """Resolve a region's traversal order into a flat list of cluster indices.

    ``order`` may be a plain list (identity / pre-traversal order) or, after
    traversal optimization, a dict keyed by TSP algorithm name. For a dict,
    prefer ``selected_algorithm`` (the results file's
    ``selected_traversal_algorithm``); otherwise fall back to the first
    available algorithm's path.
    """
    if isinstance(order, dict):
        if not order:
            return []
        if selected_algorithm in order:
            return order[selected_algorithm]
        return next(iter(order.values()))
    return order


def _filter_large_triangles(mesh, method='iqr', threshold_multiplier=1.5):
    """
    Filter out triangles with anomalously large areas from a mesh

    Args:
        mesh: open3d.geometry.TriangleMesh
        method: Method for determining anomalies
                'iqr' - Interquartile range (robust to outliers)
                'std' - Standard deviation
                'percentile' - Keep triangles below a percentile
                'absolute' - Absolute threshold
        threshold_multiplier: For 'iqr': multiplier for IQR (default 1.5, like boxplot)
                             For 'std': number of standard deviations
                             For 'percentile': percentile value (0-100)
                             For 'absolute': maximum allowed area

    Returns:
        open3d.geometry.TriangleMesh with filtered triangles
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(triangles) == 0:
        return mesh

    # Compute area for each triangle
    areas = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        # Compute area using cross product: Area = 0.5 * ||(v1-v0) × (v2-v0)||
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        areas.append(area)

    areas = np.array(areas)

    # Determine threshold based on method
    if method == 'iqr':
        # Interquartile range method (robust)
        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        iqr = q3 - q1
        threshold = q3 + threshold_multiplier * iqr
        print(
            f"IQR method: Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}, Threshold={threshold:.6f}")

    elif method == 'std':
        # Standard deviation method
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        threshold = mean_area + threshold_multiplier * std_area
        print(
            f"STD method: Mean={mean_area:.6f}, Std={std_area:.6f}, Threshold={threshold:.6f}")

    elif method == 'percentile':
        # Percentile method
        threshold = np.percentile(areas, threshold_multiplier)
        print(
            f"Percentile method: {threshold_multiplier}th percentile={threshold:.6f}")

    elif method == 'absolute':
        # Absolute threshold
        threshold = threshold_multiplier
        print(f"Absolute method: Threshold={threshold:.6f}")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Filter triangles
    filtered_triangles = []
    for i, tri in enumerate(triangles):
        if areas[i] <= threshold:
            filtered_triangles.append(tri)

    num_removed = len(triangles) - len(filtered_triangles)
    print(
        f"Removed {num_removed}/{len(triangles)} triangles ({100*num_removed/len(triangles):.1f}%)")
    print(f"Area range: [{np.min(areas):.6f}, {np.max(areas):.6f}]")

    # Create filtered mesh
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = mesh.vertices
    filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)

    # Copy colors and normals if they exist
    if mesh.has_vertex_colors():
        filtered_mesh.vertex_colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        filtered_mesh.vertex_normals = mesh.vertex_normals

    return filtered_mesh


def _build_standard_viewpoint_mesh(viewpoint_data: dict) -> o3d.geometry.TriangleMesh:
    """Build a viewpoint marker (sphere/arrow/axes) from raw viewpoint_data."""
    if Materials.viewpoint_type == "sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=Materials.viewpoint_sphere_size)
    elif Materials.viewpoint_type == "arrow":
        mesh = o3d.geometry.TriangleMesh.create_arrow()
        mesh.scale(Materials.viewpoint_arrow_scale, center=(0, 0, 0))
    else:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=Materials.viewpoint_axis_size, origin=[0, 0, 0])

    orientation = np.array(viewpoint_data['orientation'])  # xyzw
    mesh.rotate(
        o3d.geometry.get_rotation_matrix_from_quaternion(
            np.array([orientation[3], orientation[0],
                      orientation[1], orientation[2]])),
        center=(0, 0, 0),
    )
    position = 1000.0 * np.array(viewpoint_data['position'])
    mesh.translate(position)
    mesh.paint_uniform_color([1.0, 1.0, 1.0])
    return mesh


def _build_frustum(viewpoint_data: dict,
                   fov_w: float = 0.03,
                   fov_h: float = 0.02,
                   depth: float = 30.0) -> o3d.geometry.LineSet:
    """Build a camera-frustum LineSet at the given viewpoint (in mm)."""
    position    = 1000.0 * np.array(viewpoint_data['position'])
    orientation = np.array(viewpoint_data['orientation'])  # xyzw
    R = o3d.geometry.get_rotation_matrix_from_quaternion(
        np.array([orientation[3], orientation[0],
                  orientation[1], orientation[2]]))

    # Camera convention: +Z is forward, +X is right, +Y is up
    right = R[:, 0]
    up    = R[:, 1]
    fwd   = R[:, 2]

    hw = 1000.0 * fov_w / 2.0 * depth
    hh = 1000.0 * fov_h / 2.0 * depth
    far_centre = position + fwd * depth

    tl = far_centre + up * hh - right * hw
    tr = far_centre + up * hh + right * hw
    bl = far_centre - up * hh - right * hw
    br = far_centre - up * hh + right * hw

    pts   = np.array([position, tl, tr, bl, br])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],   # apex → corners
             [1, 2], [2, 4], [4, 3], [3, 1]]    # far rectangle

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    ls.paint_uniform_color([0.9, 0.9, 0.9])
    return ls


# ─────────────────────────────────────────────────────────────────────────────
# Renderer ABC
# ─────────────────────────────────────────────────────────────────────────────

class ClusterViewpointRenderer(ABC):
    """Strategy for building the geometry of a single cluster-viewpoint pair."""

    @abstractmethod
    def build_cluster(
        self,
        cluster_pcd: o3d.geometry.PointCloud,
        cluster_color: np.ndarray,
    ) -> tuple | None:
        """Return (geometry, material) for the cluster footprint, or None."""

    @abstractmethod
    def build_viewpoint(
        self,
        viewpoint_data: dict,
    ) -> tuple | None:
        """Return (geometry, material) for the viewpoint marker, or None."""

    # Default materials — override in subclasses if needed
    def cluster_material(self):
        return Materials.cluster_material

    def selected_cluster_material(self):
        return Materials.selected_cluster_material

    def viewpoint_material(self):
        return Materials.viewpoint_material

    def selected_viewpoint_material(self):
        return Materials.selected_viewpoint_material


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Renderers
# ─────────────────────────────────────────────────────────────────────────────

class ConvexHullRenderer(ClusterViewpointRenderer):
    """Convex hull of the cluster point cloud (original default behaviour)."""

    def build_cluster(self, cluster_pcd, cluster_color):
        pcd, _ = cluster_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        mesh = pcd.compute_convex_hull(joggle_inputs=True)[0]
        mesh.paint_uniform_color(cluster_color)
        mesh.compute_vertex_normals()
        avg_normal = np.mean(np.asarray(pcd.normals), axis=0)
        mesh.translate(avg_normal * 0.20)
        return mesh, Materials.cluster_material

    def build_viewpoint(self, viewpoint_data):
        return _build_standard_viewpoint_mesh(viewpoint_data), Materials.viewpoint_material


class ClusterCloudRenderer(ClusterViewpointRenderer):
    """Raw point cloud coloured by cluster colour."""

    def build_cluster(self, cluster_pcd, cluster_color):
        pcd = copy.deepcopy(cluster_pcd)
        pcd.paint_uniform_color(cluster_color)
        return pcd, Materials.point_cloud_material

    def build_viewpoint(self, viewpoint_data):
        return _build_standard_viewpoint_mesh(viewpoint_data), Materials.viewpoint_material

    def cluster_material(self):
        return Materials.point_cloud_material

    def selected_cluster_material(self):
        return Materials.selected_cluster_material


class FrustumRenderer(ClusterViewpointRenderer):
    """Convex hull cluster footprint + camera frustum at the viewpoint."""

    def __init__(self, fov_w: float = 0.03, fov_h: float = 0.02,
                 depth: float = 30.0):
        self.fov_w = fov_w
        self.fov_h = fov_h
        self.depth = depth

    def build_cluster(self, cluster_pcd, cluster_color):
        pcd, _ = cluster_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        mesh = pcd.compute_convex_hull(joggle_inputs=True)[0]
        mesh.paint_uniform_color(cluster_color)
        mesh.compute_vertex_normals()
        return mesh, Materials.cluster_material

    def build_viewpoint(self, viewpoint_data):
        frustum = _build_frustum(
            viewpoint_data, self.fov_w, self.fov_h, self.depth)
        return frustum, Materials.path_material

    def viewpoint_material(self):
        return Materials.path_material

    def selected_viewpoint_material(self):
        return Materials.selected_path_material


class LinesRenderer(ClusterViewpointRenderer):
    """A line from the viewpoint to the cluster centroid; no hull footprint."""

    def build_cluster(self, _cluster_pcd, _cluster_color):
        return None

    def build_viewpoint(self, viewpoint_data):
        vp_pos   = 1000.0 * np.array(viewpoint_data['position'])
        centroid = np.array(viewpoint_data.get('cluster_centroid', vp_pos),
                            dtype=float)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array([vp_pos, centroid]))
        ls.lines  = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        ls.paint_uniform_color([0.9, 0.9, 0.9])
        return ls, Materials.path_material

    def cluster_material(self):
        return Materials.path_material

    def selected_cluster_material(self):
        return Materials.selected_path_material

    def viewpoint_material(self):
        return Materials.path_material

    def selected_viewpoint_material(self):
        return Materials.selected_path_material


class ViewpointOnlyRenderer(ClusterViewpointRenderer):
    """Only renders viewpoint markers; cluster footprint is suppressed."""

    def build_cluster(self, _cluster_pcd, _cluster_color):
        return None

    def build_viewpoint(self, viewpoint_data):
        return _build_standard_viewpoint_mesh(viewpoint_data), Materials.viewpoint_material


class OriginSphereRenderer(ClusterViewpointRenderer):
    """A small sphere at the surface origin of each cluster, coloured by cluster colour."""

    def __init__(self, radius: float = 2.0):
        self.radius = radius  # mm

    def build_cluster(self, _cluster_pcd, _cluster_color):
        return None

    def build_viewpoint(self, viewpoint_data):
        origin = 1000.0 * np.array(viewpoint_data['origin'])
        color  = viewpoint_data.get('cluster_color', [1.0, 1.0, 1.0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
        sphere.translate(origin)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere, Materials.viewpoint_material


# Map from mode enum to renderer class
_RENDERER_MAP: dict[ClusterViewpointMode, type[ClusterViewpointRenderer]] = {
    ClusterViewpointMode.CONVEX_HULL:    ConvexHullRenderer,
    ClusterViewpointMode.CLUSTER_CLOUD:  ClusterCloudRenderer,
    ClusterViewpointMode.FRUSTUM:        FrustumRenderer,
    ClusterViewpointMode.LINES:          LinesRenderer,
    ClusterViewpointMode.VIEWPOINT_ONLY: ViewpointOnlyRenderer,
    ClusterViewpointMode.ORIGIN_SPHERE:  OriginSphereRenderer,
}


# ─────────────────────────────────────────────────────────────────────────────
# Turntable camera controller
# ─────────────────────────────────────────────────────────────────────────────

class TurntableCameraController:
    """Y-up turntable orbit camera for an Open3D SceneWidget.

    Geometry is imported with a –90° X rotation, so the scene's vertical
    axis is Y.  All orbit math is Y-up (elevation lives in the XZ plane,
    the fixed 'up' vector passed to look_at is [0, 1, 0]).

    Controls:
        Left drag   – orbit (azimuth / elevation, Y axis stays vertical)
        Middle drag – pan  (translate the orbit center)
        Right drag  – dolly (zoom by moving along view axis)
        Scroll      – zoom
    """

    def __init__(self, scene_widget: o3d.visualization.gui.SceneWidget,
                 center=None, eye=None):
        self.scene_widget = scene_widget
        self.center    = np.array([0., 0., 0.]) if center is None else np.asarray(center, dtype=float)
        self.azimuth   = np.pi / 4
        self.elevation = np.pi / 6
        self.distance  = 1.0
        if eye is not None:
            self._compute_from_eye(np.asarray(eye, dtype=float))
        self._drag_start  = None
        self._drag_button = None
        self._update_camera()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_from_eye(self, eye: np.ndarray):
        diff = eye - self.center
        self.distance  = max(float(np.linalg.norm(diff)), 1e-6)
        diff_n = diff / self.distance
        # Y-up: elevation is the angle above the XZ plane (Y component)
        self.elevation = float(np.arcsin(np.clip(diff_n[1], -1.0, 1.0)))
        self.azimuth   = float(np.arctan2(diff_n[2], diff_n[0]))

    def _eye(self) -> np.ndarray:
        el = self.elevation
        # Y-up spherical coordinates
        return self.center + self.distance * np.array([
            np.cos(el) * np.cos(self.azimuth),  # X
            np.sin(el),                           # Y (vertical)
            np.cos(el) * np.sin(self.azimuth),  # Z
        ])

    def _update_camera(self):
        self.scene_widget.look_at(self.center, self._eye(), np.array([0., 1., 0.]))

    # ── Public API ───────────────────────────────────────────────────────────

    def reset_camera(self, center, eye):
        """Full reset: recompute azimuth / elevation / distance from center + eye."""
        self.center = np.asarray(center, dtype=float)
        self._compute_from_eye(np.asarray(eye, dtype=float))
        self._update_camera()

    def set_center(self, center, distance=None):
        """Translate the orbit center; optionally update distance too."""
        self.center = np.asarray(center, dtype=float)
        if distance is not None:
            self.distance = max(float(distance), 1e-6)
        self._update_camera()

    # ── Mouse handler ────────────────────────────────────────────────────────

    def on_mouse(self, event) -> gui.Widget.EventCallbackResult:
        CONSUMED = gui.Widget.EventCallbackResult.CONSUMED
        IGNORED  = gui.Widget.EventCallbackResult.IGNORED

        # ── Scroll: zoom ─────────────────────────────────────────────────────
        # wheel_dy is the reliable cross-version check; SCROLL type may not exist.
        if event.wheel_dy != 0:
            self.distance = max(1e-3, self.distance * (0.9 ** event.wheel_dy))
            self._update_camera()
            return CONSUMED

        # ── Button down: begin drag ─────────────────────────────────────────
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self._drag_start = (event.x, event.y)
            if event.is_button_down(gui.MouseButton.LEFT):
                self._drag_button = 'orbit'
            elif event.is_button_down(gui.MouseButton.MIDDLE):
                self._drag_button = 'pan'
            elif event.is_button_down(gui.MouseButton.RIGHT):
                self._drag_button = 'zoom'
            return CONSUMED

        # ── Drag ────────────────────────────────────────────────────────────
        if event.type == gui.MouseEvent.Type.DRAG and self._drag_start is not None:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._drag_start = (event.x, event.y)

            if self._drag_button == 'orbit':
                self.azimuth   += dx * 0.005
                self.elevation  = np.clip(
                    self.elevation + dy * 0.005, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
                self._update_camera()

            elif self._drag_button == 'pan':
                eye = self._eye()
                fwd = (self.center - eye) / self.distance
                # Y-up: right vector lies in XZ plane
                right = np.cross(fwd, np.array([0., 1., 0.]))
                rlen  = np.linalg.norm(right)
                if rlen > 1e-6:
                    right  /= rlen
                    cam_up  = np.cross(right, fwd)
                    scale   = self.distance * 0.001
                    self.center -= right  * dx * scale
                    self.center += cam_up * dy * scale
                    self._update_camera()

            elif self._drag_button == 'zoom':
                self.distance = max(1e-3, self.distance * (1.0 + dy * 0.005))
                self._update_camera()

            return CONSUMED

        # ── Button up: end drag ─────────────────────────────────────────────
        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            self._drag_start  = None
            self._drag_button = None
            return CONSUMED

        return IGNORED


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class Visualizer:
    """Owns the Open3D scene and all geometry/rendering operations.

    geometry_dict schema::

        {
            "region_N": {
                "cloud":       PointCloud,
                "color":       np.ndarray,          # [r, g, b]
                "clusters": {
                    "region_N_cluster_M": {
                        "pcd":           PointCloud,   # raw cluster cloud
                        "color":         np.ndarray,
                        "viewpoint_data": dict | None, # raw JSON viewpoint data
                    },
                    ...
                },
                "view_mesh": TriangleMesh | None,
                "path":      LineSet | None,
            },
            ...
        }
    """

    def __init__(self, scene_widget: o3d.visualization.gui.SceneWidget):
        self.scene_widget = scene_widget
        self.scene        = scene_widget.scene

        self.ray_casting_scene   = o3d.t.geometry.RaycastingScene()
        self.current_view_matrix = scene_widget.scene.camera.get_view_matrix()

        # Turntable camera — initial look from [1,1,1] toward origin
        self.camera = TurntableCameraController(
            scene_widget,
            center=np.array([0., 0., 0.]),
            eye=np.array([1., 1., 1.]),
        )

        # Active rendering mode
        self._mode:     ClusterViewpointMode     = ClusterViewpointMode.CONVEX_HULL
        self._renderer: ClusterViewpointRenderer = ConvexHullRenderer()

        # Model state
        self.mesh_name        = None
        self.mesh_units       = None
        self.point_cloud_name = None
        self.point_cloud      = None

        # Region / cluster state
        self.mesh_names      : list[str]        = []
        self.region_names    : list[str]        = []
        self.cluster_names   : list[str]        = []
        self.traversal_order : list[list[int]]  = [[0]]
        self.geometries_dict : dict             = {}
        self.results_dict    : dict             = {}
        self.region_number   : int              = 0
        self.cluster_number  : int              = 0

        # Mesh vertex coloring state
        self.meshes             : dict = {}
        self.mesh_vertex_to_pcd : dict = {}
        self.noise_indices      : list = []

        # Selection state
        self.selected_mesh_idx    : int = -1
        self.selected_region_name : str = ''
        self.selected_cluster_name: str = ''

        # Show flags
        self.show_mesh_flag                  = True
        self.show_point_cloud_flag           = False
        self.show_curvatures_flag            = False
        self.show_regions_flag               = False
        self.show_noise_points_flag          = False
        self.show_clusters_flag              = False
        self.show_viewpoints_flag            = False
        self.show_region_view_manifolds_flag = False
        self.show_path_flag                  = False
        self.mesh_has_vertex_colors          = False

    def on_mouse(self, event) -> gui.Widget.EventCallbackResult:
        """Delegate mouse events to the turntable camera controller."""
        return self.camera.on_mouse(event)

    # ── Mode / Re-render ─────────────────────────────────────────────────────

    def set_mode(self, mode: ClusterViewpointMode):
        """Switch cluster-viewpoint rendering mode and rebuild all cluster geometry."""
        self._mode     = mode
        self._renderer = _RENDERER_MAP[mode]()
        self._rerender_clusters()

    def _rerender_clusters(self):
        """Rebuild viewpoint geometry using the current renderer and repaint mesh."""
        for region_data in self.geometries_dict.values():
            if 'clusters' not in region_data:
                continue
            for cluster_name, cluster_data in region_data['clusters'].items():
                # Remove old viewpoint geometry
                self.scene.remove_geometry(f"{cluster_name}_viewpoint")

                # Build viewpoint marker
                vp_data = cluster_data.get('viewpoint_data')
                if vp_data:
                    # Inject centroid for LinesRenderer
                    centroid = np.mean(
                        np.asarray(cluster_data['pcd'].points), axis=0)
                    vp_data_aug = dict(vp_data, cluster_centroid=centroid.tolist())
                    result = self._renderer.build_viewpoint(vp_data_aug)
                    if result:
                        geo, mat = result
                        self.add_geometry(f"{cluster_name}_viewpoint", geo, mat)
        # Repaint mesh if clusters are currently shown
        if self.show_clusters_flag:
            self.show_fov_clusters(True)

    # ── Core scene helper ────────────────────────────────────────────────────

    def _reset_camera_to_bbox(self, bbox: o3d.geometry.AxisAlignedBoundingBox):
        """Reset the turntable camera so the mesh fills the view.

        All geometry is stored in the scene after a -90° X rotation
        ([x,y,z] → [x, z, -y]), so the camera center must be derived from
        the *rotated* bounding-box centre rather than the raw bbox centre.
        The eye is placed one bounding-box diagonal away in the [1,1,1]
        direction of the rotated frame, which keeps the camera well outside
        the mesh regardless of its position or size.
        """
        # -90° X rotation: [x, y, z] → [x, z, -y]
        raw_center = bbox.get_center()
        rotated_center = np.array([raw_center[0], raw_center[2], -raw_center[1]])

        diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        offset_dir = np.array([1., 1., 1.]) / np.sqrt(3)
        eye = rotated_center + offset_dir * diagonal * 1.5

        self.camera.reset_camera(rotated_center, eye)

    def add_geometry(self, name: str, geometry, material):
        """Deep-copy geometry, apply -90° X-axis rotation, then add to scene."""
        geometry = copy.deepcopy(geometry)
        if isinstance(geometry, o3d.geometry.AxisAlignedBoundingBox):
            geometry = geometry.get_oriented_bounding_box()
        geometry.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)),
            center=(0, 0, 0),
        )
        self.scene.add_geometry(name, geometry, material)

    # ── Lighting / ground plane ──────────────────────────────────────────────

    def setup_multi_directional_lighting(self):
        sun_direction = np.array([0.577, -0.577, -0.577], dtype=np.float32)
        self.scene.set_lighting(
            self.scene.LightingProfile.NO_SHADOWS, sun_direction)

    def add_xy_plane(self, bbox):
        if bbox is None:
            return

        x0, y0, _ = bbox.get_min_bound()
        x1, y1, _ = bbox.get_max_bound()
        x0 = 10 * (round(x0 / 10) - 1)
        y0 = 10 * (round(y0 / 10) - 1)
        x1 = 10 * (round(x1 / 10) + 1)
        y1 = 10 * (round(y1 / 10) + 1)

        width = max(abs(x0), abs(x1), abs(y0), abs(y1))

        axes = o3d.geometry.LineSet()
        axes.points = o3d.utility.Vector3dVector(
            np.array([[-width, 0, 0], [width, 0, 0],
                      [0, 0, -width], [0, 0, width]]))
        axes.lines  = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
        axes.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0], [0, 1, 0]]))

        grid = o3d.geometry.LineSet()
        pts, lines, colors = [], [], []
        for x in np.linspace(-width, width,
                              round(2 * width / 10) + 1):
            if abs(x) < 1e-5:
                continue
            pts.extend([[x, 0, -width], [x, 0, width]])
            lines.append([len(pts) - 2, len(pts) - 1])
            colors.append([0.3, 0.3, 0.3])
        for y in np.linspace(-width, width,
                              round(2 * width / 10) + 1):
            if abs(y) < 1e-5:
                continue
            pts.extend([[-width, 0, y], [width, 0, y]])
            lines.append([len(pts) - 2, len(pts) - 1])
            colors.append([0.3, 0.3, 0.3])
        grid.points = o3d.utility.Vector3dVector(np.array(pts))
        grid.lines  = o3d.utility.Vector2iVector(np.array(lines))
        grid.colors = o3d.utility.Vector3dVector(np.array(colors))

        ground = o3d.geometry.TriangleMesh()
        w = width
        ground.vertices = o3d.utility.Vector3dVector(
            np.array([[-w, 0, -w], [w, 0, -w], [w, 0, w], [-w, 0, w]]))
        ground.triangles = o3d.utility.Vector3iVector(
            np.array([[0, 1, 2], [0, 2, 3]]))
        ground.vertex_normals = o3d.utility.Vector3dVector(
            np.array([[0, 0, 1]] * 4))

        self.scene.remove_geometry('ground plane')
        self.scene.remove_geometry('xy axes')
        self.scene.remove_geometry('grid')

        self.scene.add_geometry(
            'ground plane', ground, Materials.ground_plane_material)
        self.scene.add_geometry(
            'xy axes', axes, Materials.axes_line_material)
        self.scene.add_geometry(
            'grid', grid, Materials.grid_line_material)

    # ── Import methods ───────────────────────────────────────────────────────

    def import_mesh(self, file_path: str, mesh_units: str) -> dict | None:
        """Load and display a mesh. Returns {'mesh': mesh, 'bbox': bbox} or None."""
        print(f"Importing mesh from {file_path}")
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            if mesh.is_empty():
                print(f"Warning: Mesh file {file_path} is empty or invalid.")
                return None

            self.mesh_name  = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
            self.mesh_units = mesh_units

            scale_map = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0,
                         'in': 25.4, 'ft': 304.8}
            mesh.scale(scale_map.get(mesh_units, 1.0), center=(0, 0, 0))

            bbox = mesh.get_axis_aligned_bounding_box()

            self.scene.remove_geometry("model_bounding_box")
            self.add_geometry("model_bounding_box", bbox,
                              Materials.bounding_box_material)

            self.scene.remove_geometry("mesh")
            self.scene.remove_geometry("point_cloud")
            self.scene.remove_geometry("curvatures")

            self._clear_result_geometry()

            self.add_geometry("mesh", mesh, Materials.mesh_material)

            self.ray_casting_scene = o3d.t.geometry.RaycastingScene()
            self.ray_casting_scene.add_triangles(
                o3d.t.geometry.TriangleMesh.from_legacy(mesh))

            self.point_cloud = None
            return {'mesh': mesh, 'bbox': bbox}

        except Exception as e:
            print(f"Error loading mesh from {file_path}: {e}")
            return None

    def import_point_cloud(self, file_path: str, pcd_units: str) -> bool:
        """Load and display a point cloud. Returns True on success."""
        if not file_path:
            print("Point cloud path is empty.")
            return False
        print(f"Importing point cloud from {file_path}")
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if pcd.is_empty():
                print(f"Warning: Point cloud {file_path} is empty.")
                return False

            self.point_cloud_name = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]

            scale_map = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0,
                         'in': 25.4, 'ft': 304.8}
            pcd.scale(scale_map.get(pcd_units, 1.0), center=(0, 0, 0))

            self.scene.remove_geometry("point_cloud")
            self.scene.remove_geometry("curvatures")

            self._clear_result_geometry()

            self.add_geometry("point_cloud", pcd, Materials.point_cloud_material)
            self.point_cloud = pcd
            return True

        except Exception as e:
            print(f"Error loading point cloud from {file_path}: {e}")
            return False

    def import_curvature(self, file_path: str) -> bool:
        """Colour-code the stored point cloud by curvature. Returns True on success."""
        if self.point_cloud is None:
            print("No point cloud loaded; cannot import curvature.")
            return False
        print(f"Importing curvature data from {file_path}")
        try:
            curvature = np.load(file_path)
            curvatures_cloud = copy.deepcopy(self.point_cloud)

            normalised = (curvature - curvature.min()) / (
                curvature.max() - curvature.min() + 1e-9)
            cmap = colormaps[Materials.curvature_colormap]
            colors = np.array([cmap(v)[:3] for v in normalised])
            curvatures_cloud.colors = o3d.utility.Vector3dVector(colors)

            self.scene.remove_geometry("curvatures")
            self.scene.remove_geometry("regions")

            self._clear_result_geometry()

            self.add_geometry("curvatures", curvatures_cloud,
                              Materials.point_cloud_material)
            return True

        except Exception as e:
            print(f"Error loading curvature data from {file_path}: {e}")
            return False

    def visualize_results(self, file_path: str) -> None:
        """Parse a regions JSON file and populate the scene.

        Loads the mesh and point cloud referenced in the JSON before building
        region/cluster/viewpoint geometry.

        Sets scene visibility and camera position internally.
        """
        if not file_path:
            print("Regions file path is empty.")
            return

        print(f"Importing regions from {file_path}")

        results_dict = json.load(open(file_path, 'r'))
        self.results_dict = results_dict

        # After traversal optimization each region's 'order' is a dict keyed by
        # algorithm name; this selects which algorithm's path to render.
        selected_algorithm = results_dict.get('selected_traversal_algorithm')

        # ── Load mesh and point cloud from the results dict ───────────────────
        bbox = None
        scale_map = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'in': 25.4, 'ft': 304.8}

        # Loading a new results file invalidates everything from the previous
        # one, so remove all downstream geometry before rebuilding: meshes, the
        # point cloud, curvatures, and every region/cluster/viewpoint/path/view
        # manifold. Without this a stale point cloud (and viewpoints, paths,
        # etc.) from the last file lingers when the new file has none of its own
        # — e.g. when a bare mesh is loaded and its results file has no point
        # cloud or regions yet. _clear_result_geometry must run while the name
        # lists still hold the previous file's names so its removals take effect.
        self.scene.remove_geometry("mesh")
        for name in self.mesh_names:
            self.scene.remove_geometry(name)
        self.scene.remove_geometry("point_cloud")
        self.scene.remove_geometry("curvatures")
        self._clear_result_geometry()
        self.point_cloud            = None
        self.noise_indices          = []
        self.meshes                 = {}
        self.mesh_vertex_to_pcd     = {}
        self.mesh_has_vertex_colors = False
        self.geometries_dict        = {}

        new_mesh_names = []
        new_meshes     = {}
        new_geometries_dict: dict = {}
        show_clusters   = False
        show_viewpoints = False
        show_path       = False
        traversal_order: list[list[int]] = []
        all_noise_points: list = []

        cmap = colormaps[Materials.regions_colormap]
        np.random.seed(1)

        for mesh_idx, mesh_entry in enumerate(results_dict.get('meshes', [])):
            mesh_file  = mesh_entry.get('file', '')
            mesh_units = mesh_entry.get('units', 'mm')
            mesh_name  = f"mesh_{mesh_idx}"

            if mesh_file:
                try:
                    mesh = o3d.io.read_triangle_mesh(mesh_file)
                    if not mesh.is_empty():
                        mesh.scale(scale_map.get(mesh_units, 1.0), center=(0, 0, 0))
                        if mesh_idx == 0:
                            bbox = mesh.get_axis_aligned_bounding_box()
                            self.mesh_name  = mesh_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                            self.mesh_units = mesh_units
                            self.ray_casting_scene = o3d.t.geometry.RaycastingScene()
                            self.ray_casting_scene.add_triangles(
                                o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                            self.scene.remove_geometry("model_bounding_box")
                            self.add_geometry("model_bounding_box", bbox,
                                              Materials.bounding_box_material)
                        new_mesh_names.append(mesh_name)
                        new_meshes[mesh_name] = mesh
                except Exception as e:
                    print(f"Error loading mesh {mesh_file}: {e}")

            pcd_entry = mesh_entry.get('point_cloud', {})
            pcd_file  = pcd_entry.get('file', '')
            pcd_units = pcd_entry.get('units', 'mm')
            if pcd_file:
                self.import_point_cloud(pcd_file, pcd_units)

            if self.point_cloud is None:
                continue

            self.point_cloud.paint_uniform_color((1, 1, 1))

            region_order = mesh_entry.get('order', [])

            # Guard against a stale results file: the region/noise indices may
            # reference a point cloud with a different number of points than the
            # one currently loaded (e.g. the results were generated before the
            # cloud was resampled to a smaller size). Indexing out of bounds
            # would otherwise crash the GUI process.
            n_points = len(self.point_cloud.points)
            max_index = -1
            for region_id in region_order:
                region_points = mesh_entry['regions'][region_id].get('points', [])
                if len(region_points) > 0:
                    max_index = max(max_index, int(np.max(region_points)))
            mesh_noise_points = mesh_entry.get('noise_points', [])
            if len(mesh_noise_points) > 0:
                max_index = max(max_index, int(np.max(mesh_noise_points)))
            if max_index >= n_points:
                print(
                    f"Results file '{file_path}' references point index "
                    f"{max_index} but the loaded point cloud '{pcd_file}' has "
                    f"only {n_points} points. The results were likely generated "
                    "from a different point cloud (e.g. before resampling). "
                    "Skipping visualization.")
                return

            all_noise_points.extend(mesh_noise_points)

            for region_id in region_order:
                region_name = f"mesh_{mesh_idx}_region_{region_id}"
                region_dict = mesh_entry['regions'][region_id]

                region_indices     = region_dict['points']
                region_color       = np.array(list(cmap(np.random.rand())))[0:3]

                # Paint these indices on the single point cloud
                pcd_colors = np.asarray(self.point_cloud.colors)
                pcd_colors[region_indices] = region_color

                new_geometries_dict[region_name] = {
                    'indices': region_indices,
                    'color': region_color,
                }

                region_view_points : list = []
                region_view_normals: list = []
                path_points        : list = []
                path_line = o3d.geometry.LineSet()

                if region_dict.get('clusters'):
                    show_clusters = True
                    cluster_order = _resolve_order_indices(
                        region_dict['order'], selected_algorithm)
                    traversal_order.append(cluster_order)
                    new_geometries_dict[region_name]['clusters'] = {}

                    for i, cluster_id in enumerate(cluster_order):
                        cluster_name = f"{region_name}_cluster_{i}"
                        cluster_dict = region_dict['clusters'][cluster_id]

                        # Derive each cluster's color from the region hue but
                        # spread brightness/saturation widely so clusters within
                        # a region contrast strongly. The golden-ratio sequence
                        # keeps consecutive (spatially adjacent) clusters far
                        # apart in color rather than forming a smooth gradient.
                        h, s, v = colorsys.rgb_to_hsv(*region_color)
                        golden = 0.6180339887498949
                        frac_v = (i * golden) % 1.0
                        frac_h = (i * golden * 2.0) % 1.0
                        h_var = (h + 0.15 * (frac_h - 0.5)) % 1.0   # hue ±0.075
                        s_var = float(np.clip(max(s, 0.5), 0.0, 1.0))
                        v_var = 0.45 + 0.5 * frac_v                 # value 0.45..0.95
                        cluster_color = np.array(
                            colorsys.hsv_to_rgb(h_var, s_var, v_var))
                        cluster_color = np.clip(cluster_color, 0, 1)

                        # Convert cluster sub-indices (relative to region) to global indices.
                        # select_by_index returns points sorted by original index,
                        # so the cluster indices reference the sorted region indices.
                        region_indices_sorted = np.sort(np.array(region_indices))
                        global_cluster_indices = region_indices_sorted[cluster_dict['points']]
                        cluster_pcd = self.point_cloud.select_by_index(global_cluster_indices)

                        vp_data = None
                        if 'viewpoint' in cluster_dict:
                            show_viewpoints = True
                            vp_raw = cluster_dict['viewpoint']
                            # Store centroid for LinesRenderer
                            centroid = np.mean(np.asarray(cluster_pcd.points), axis=0)
                            vp_data = {
                                'origin':           vp_raw['origin'],
                                'position':         vp_raw['position'],
                                'direction':        vp_raw['direction'],
                                'orientation':      vp_raw['orientation'],
                                'cluster_centroid': centroid.tolist(),
                                'cluster_color':    cluster_color.tolist(),
                            }

                            position  = 1000.0 * np.array(vp_raw['position'])
                            direction = np.array(vp_raw['direction'])
                            region_view_points.append(position.tolist())
                            region_view_normals.append(direction.tolist())
                            path_points.append(position)

                            if i != cluster_id:
                                show_path = True

                        new_geometries_dict[region_name]['clusters'][cluster_name] = {
                            'pcd':            cluster_pcd,
                            'indices':        global_cluster_indices,
                            'color':          cluster_color,
                            'viewpoint_data': vp_data,
                        }

                # Region view surface (convex hull of viewpoint positions)
                if show_viewpoints and region_view_points:
                    region_view_cloud = o3d.geometry.PointCloud()
                    region_view_cloud.points = o3d.utility.Vector3dVector(
                        np.array(region_view_points))
                    region_view_cloud.normals = o3d.utility.Vector3dVector(
                        np.array(region_view_normals))

                    if len(region_view_points) > 4:
                        view_mesh = region_view_cloud.compute_convex_hull(
                            joggle_inputs=True)[0]
                        view_mesh = _filter_large_triangles(
                            view_mesh, method='iqr', threshold_multiplier=5.0)
                        view_mesh.compute_vertex_normals()
                        new_geometries_dict[region_name]['view_mesh'] = view_mesh
                    else:
                        new_geometries_dict[region_name]['view_mesh'] = None

                    if len(path_points) > 1:
                        path_line.points = o3d.utility.Vector3dVector(
                            np.array(path_points))
                        path_line.lines = o3d.utility.Vector2iVector(
                            [[i, i + 1] for i in range(len(path_points) - 1)])
                        new_geometries_dict[region_name]['path'] = path_line
                    else:
                        new_geometries_dict[region_name]['path'] = None

        if self.point_cloud is None:
            print("No point cloud loaded; cannot import regions.")
            for mesh_name, mesh in new_meshes.items():
                self.add_geometry(mesh_name, mesh, Materials.mesh_material)
            self.mesh_names = new_mesh_names
            if bbox is not None:
                self._reset_camera_to_bbox(bbox)
                self.add_xy_plane(bbox)
            self.show_mesh(True)
            return

        # ── Paint noise points red on the single point cloud ────────────────
        self.noise_indices = all_noise_points
        if all_noise_points:
            pcd_colors = np.asarray(self.point_cloud.colors)
            pcd_colors[all_noise_points] = [1.0, 0.0, 0.0]

        self.mesh_names      = new_mesh_names
        self.region_names    = []
        self.cluster_names   = []
        self.traversal_order = traversal_order
        self.geometries_dict = new_geometries_dict
        self.region_number   = 0
        self.cluster_number  = 0
        self.selected_mesh_idx     = -1
        self.selected_region_name  = ''
        self.selected_cluster_name = ''

        # ── Add meshes (deferred so vertex colors can be applied) ─────────
        if new_geometries_dict and len(self.point_cloud.points) > 0:
            avg_pcd_spacing = np.mean(
                self.point_cloud.compute_nearest_neighbor_distance())

            for mesh_name, mesh in new_meshes.items():
                # Subdivide until mesh edge length ≈ point cloud spacing
                triangles = np.asarray(mesh.triangles)
                vertices = np.asarray(mesh.vertices)
                edges = np.array([[tri[j], tri[(j + 1) % 3]]
                                  for tri in triangles for j in range(3)])
                avg_edge_length = np.mean(
                    np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]],
                                   axis=1))
                iterations = max(
                    0, math.ceil(math.log2(avg_edge_length / avg_pcd_spacing)))
                if iterations > 0:
                    print(f"{mesh_name}: subdividing {iterations}x "
                          f"(edge {avg_edge_length:.2f} → ~{avg_pcd_spacing:.2f})")
                    mesh = mesh.subdivide_midpoint(iterations)
                    mesh.compute_vertex_normals()
                    new_meshes[mesh_name] = mesh

            pcd_tree = o3d.geometry.KDTreeFlann(self.point_cloud)
            for mesh_name, mesh in new_meshes.items():
                mesh_vertices = np.asarray(mesh.vertices)
                vertex_to_pcd = np.zeros(len(mesh_vertices), dtype=int)
                for i, vertex in enumerate(mesh_vertices):
                    [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
                    vertex_to_pcd[i] = idx[0]
                self.mesh_vertex_to_pcd[mesh_name] = vertex_to_pcd
                self.meshes[mesh_name] = mesh
            self._recolor_meshes()
            self.mesh_has_vertex_colors = True
        else:
            for mesh_name, mesh in new_meshes.items():
                self.add_geometry(mesh_name, mesh, Materials.mesh_material)

        # ── Update single point cloud with region colors ──────────────────
        self._update_point_cloud_scene()

        # ── Add region / cluster geometry ────────────────────────────────
        for region_name, region_data in self.geometries_dict.items():
            self.region_names.append(region_name)

            if show_clusters:
                for cluster_name, cluster_data in region_data['clusters'].items():
                    self.cluster_names.append(cluster_name)

                    if show_viewpoints and cluster_data['viewpoint_data']:
                        result = self._renderer.build_viewpoint(
                            cluster_data['viewpoint_data'])
                        if result:
                            geo, mat = result
                            self.add_geometry(
                                f"{cluster_name}_viewpoint", geo, mat)

                if show_viewpoints:
                    view_mesh = region_data.get('view_mesh')
                    if view_mesh is not None:
                        self.add_geometry(
                            f"{region_name}_view_mesh", view_mesh,
                            Materials.region_view_material)
                    path = region_data.get('path')
                    if path is not None:
                        self.add_geometry(
                            f"{region_name}_path", path,
                            Materials.path_material)

        print(f"Loaded regions from {file_path}")

        # ── Camera + XY plane setup ───────────────────────────────────────────
        if bbox is not None:
            self._reset_camera_to_bbox(bbox)
            self.add_xy_plane(bbox)

        # ── Initial visibility ────────────────────────────────────────────────
        self.show_mesh(True)
        self.show_curvatures(False)
        self.show_noise_points(False)
        self.show_viewpoints(False)

        if show_clusters:
            self.show_regions(False)
            self.show_fov_clusters(True)
            if show_viewpoints:
                self.show_viewpoints(True)
                self.show_region_view_manifolds(True)
                self.show_path(show_path)
                scene_bbox = self.scene_widget.scene.bounding_box
                self.add_xy_plane(scene_bbox)
                bb_size = np.linalg.norm(
                    scene_bbox.get_max_bound() - scene_bbox.get_min_bound())
                self.camera.set_center(scene_bbox.get_center(), distance=bb_size * 1.5)
        elif self.region_names:
            self.show_regions(True)
            self.show_fov_clusters(False)
        else:
            self.show_point_cloud(True)
            self.show_regions(False)
            self.show_fov_clusters(False)


    # ── Clear helpers ─────────────────────────────────────────────────────────

    def _clear_result_geometry(self):
        """Remove all region/cluster/viewpoint/path/view-manifold geometry and
        reset the tracking lists.

        The geometry-removing clears (viewpoints, paths, view manifolds) iterate
        the cluster/region name lists, so they must run *before* clear_clusters
        and clear_regions empty those lists — otherwise nothing is removed and
        stale geometry from the previous file lingers in the scene.
        """
        self.clear_viewpoints()
        self.clear_region_view_manifolds()
        self.clear_paths()
        self.clear_clusters()
        self.clear_regions()

    def clear_regions(self):
        self.region_names = []

    def clear_region_view_manifolds(self):
        for name in self.region_names:
            self.scene.remove_geometry(f"{name}_view_mesh")

    def clear_clusters(self):
        self.cluster_names = []

    def clear_viewpoints(self):
        for name in self.cluster_names:
            self.scene.remove_geometry(f"{name}_viewpoint")

    def clear_paths(self):
        for name in self.region_names:
            self.scene.remove_geometry(f"{name}_path")

    # ── Visibility (pure scene — no menu, no ROS, no cross-calls) ─────────────

    def show_mesh(self, show: bool):
        self.show_mesh_flag = show
        for name in self.mesh_names:
            self.scene.show_geometry(name, show)

    def show_point_cloud(self, show: bool):
        # Never show the point cloud once regions have been generated
        if show and self.region_names:
            show = False
        self.show_point_cloud_flag = show
        self.scene.show_geometry('point_cloud', show)

    def show_curvatures(self, show: bool):
        self.show_curvatures_flag = show
        self.scene.show_geometry('curvatures', show)
        if show:
            self.show_point_cloud(False)
            self.show_regions(False)
            self.show_noise_points(False)

    def show_noise_points(self, show: bool):
        self.show_noise_points_flag = show

    def _update_point_cloud_scene(self):
        """Remove and re-add the point cloud geometry to reflect color changes."""
        self.scene.remove_geometry("point_cloud")
        self.add_geometry("point_cloud", self.point_cloud, Materials.point_cloud_material)
        # Keep point cloud hidden if regions have been generated
        if self.region_names:
            self.scene.show_geometry('point_cloud', False)

    def _recolor_meshes(self):
        """Recolor all meshes from current PCD colors using the stored vertex mapping."""
        pcd_colors = np.asarray(self.point_cloud.colors)
        if self.noise_indices:
            pcd_colors[self.noise_indices] = [1.0, 0.0, 0.0]
        for mesh_name, mesh in self.meshes.items():
            mapping = self.mesh_vertex_to_pcd.get(mesh_name)
            if mapping is None:
                continue
            mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[mapping])
            self.scene.remove_geometry(mesh_name)
            self.add_geometry(mesh_name, mesh, Materials.mesh_material_vertex_colors)

    def show_regions(self, show: bool):
        """Toggle region visibility on the vertex-colored mesh."""
        self.show_regions_flag = show
        if show:
            self.show_point_cloud(False)
            self.show_curvatures(False)
            self.show_noise_points(False)
            self.show_fov_clusters(False)
            # Paint PCD with region colors
            pcd_colors = np.asarray(self.point_cloud.colors)
            pcd_colors[:] = [0.8, 0.8, 0.8]  # base gray
            for region_data in self.geometries_dict.values():
                pcd_colors[region_data['indices']] = region_data['color']
            self._recolor_meshes()
        else:
            mesh_mat = Materials.mesh_material_vertex_colors if self.mesh_has_vertex_colors else Materials.mesh_material
            for name in self.mesh_names:
                self.scene_widget.scene.modify_geometry_material(name, mesh_mat)

    def show_fov_clusters(self, show: bool):
        """Toggle cluster visibility via mesh vertex coloring."""
        self.show_clusters_flag = show
        if show:
            self.show_point_cloud(False)
            self.show_curvatures(False)
            self.show_regions(False)
            # Paint PCD with cluster colors
            pcd_colors = np.asarray(self.point_cloud.colors)
            pcd_colors[:] = [0.8, 0.8, 0.8]  # base gray
            for region_data in self.geometries_dict.values():
                for cluster_data in region_data.get('clusters', {}).values():
                    pcd_colors[cluster_data['indices']] = cluster_data['color']
            self._recolor_meshes()
        else:
            mesh_mat = Materials.mesh_material_vertex_colors if self.mesh_has_vertex_colors else Materials.mesh_material
            for name in self.mesh_names:
                self.scene_widget.scene.modify_geometry_material(name, mesh_mat)

    def show_viewpoints(self, show: bool):
        self.show_viewpoints_flag = show
        for name in self.cluster_names:
            self.scene.show_geometry(f"{name}_viewpoint", show)

    def show_region_view_manifolds(self, show: bool):
        self.show_region_view_manifolds_flag = show
        for name in self.region_names:
            is_selected = (name == self.selected_region_name)
            self.scene.show_geometry(f"{name}_view_mesh", show and is_selected)

    def show_path(self, show: bool):
        self.show_path_flag = show
        for name in self.region_names:
            is_selected = (name == self.selected_region_name)
            self.scene.show_geometry(f"{name}_path", show and is_selected)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _region_names_for_mesh(self, mesh_idx: int) -> list[str]:
        prefix = f"mesh_{mesh_idx}_"
        return [n for n in self.region_names if n.startswith(prefix)]

    def _cluster_names_for_region(self, region_name: str) -> list[str]:
        prefix = f"{region_name}_cluster_"
        return [n for n in self.cluster_names if n.startswith(prefix)]

    def _get_cluster_data(self, cluster_name: str) -> dict | None:
        for region_data in self.geometries_dict.values():
            clusters = region_data.get('clusters', {})
            if cluster_name in clusters:
                return clusters[cluster_name]
        return None

    def select_mesh(self, mesh_idx: int) -> bool:
        """Highlight the selected mesh; dim and hide geometry for all others."""
        if not self.mesh_names:
            return False
        self.selected_mesh_idx = mesh_idx
        selected_name = f"mesh_{mesh_idx}"
        opaque_mat = Materials.mesh_material_vertex_colors if self.mesh_has_vertex_colors else Materials.mesh_material
        transparent_mat = Materials.mesh_material_vertex_colors_transparent if self.mesh_has_vertex_colors else Materials.mesh_material_transparent
        for name in self.mesh_names:
            mat = opaque_mat if name == selected_name else transparent_mat
            self.scene_widget.scene.modify_geometry_material(name, mat)
        for region_name in self.region_names:
            is_selected = region_name.startswith(f"mesh_{mesh_idx}_")
            for cluster_name in self._cluster_names_for_region(region_name):
                self.scene.show_geometry(f"{cluster_name}_viewpoint",
                                         is_selected and self.show_clusters_flag
                                         and self.show_viewpoints_flag)
        return True

    def select_region(self, region_idx: int) -> bool:
        """Highlight the selected region; others keep their colors but dimmed."""
        if region_idx < 0 or region_idx >= len(self.region_names):
            return False
        selected_region_name = self.region_names[region_idx]
        self.selected_region_name = selected_region_name

        # Selected region keeps its full color; all others are significantly
        # dimmed so the selection clearly stands out.
        pcd_colors = np.asarray(self.point_cloud.colors)
        pcd_colors[:] = [0.8, 0.8, 0.8]  # base gray
        for region_name, region_data in self.geometries_dict.items():
            if region_name == selected_region_name:
                color = np.array(region_data['color'])
            else:
                color = np.array(region_data['color']) * 0.15
            pcd_colors[region_data['indices']] = color
        self._recolor_meshes()

        # Show/hide viewpoints, view manifold, and path for the selected region
        for region_name in self.region_names:
            is_selected = (region_name == selected_region_name)
            for cluster_name in self._cluster_names_for_region(region_name):
                self.scene.show_geometry(f"{cluster_name}_viewpoint",
                                         is_selected and self.show_viewpoints_flag)
            self.scene.show_geometry(
                f"{region_name}_view_mesh",
                is_selected and self.show_region_view_manifolds_flag)
            self.scene.show_geometry(
                f"{region_name}_path",
                is_selected and self.show_path_flag)
        return True

    def select_cluster(self, cluster_idx: int) -> bool:
        """Highlight the selected cluster and its viewpoint via mesh vertex coloring."""
        # Scope lookup to the currently selected region so the index is
        # relative to that region rather than the flat list of all clusters.
        if self.selected_region_name:
            region_clusters = self._cluster_names_for_region(self.selected_region_name)
        else:
            region_clusters = self.cluster_names
        if cluster_idx < 0 or cluster_idx >= len(region_clusters):
            return False
        # Reset previous viewpoint material
        if self.selected_cluster_name in self.cluster_names:
            self.scene_widget.scene.modify_geometry_material(
                f"{self.selected_cluster_name}_viewpoint", Materials.viewpoint_material)
        selected_cluster_name = region_clusters[cluster_idx]
        self.selected_cluster_name = selected_cluster_name

        # Color only the clusters within the selected region; leave every
        # other region gray. The selected cluster keeps its full color and the
        # region's other clusters are significantly dimmed.
        pcd_colors = np.asarray(self.point_cloud.colors)
        pcd_colors[:] = [0.8, 0.8, 0.8]  # base gray
        selected_region_data = self.geometries_dict.get(self.selected_region_name)
        if selected_region_data is not None:
            for cname, cdata in selected_region_data.get('clusters', {}).items():
                if cname == selected_cluster_name:
                    color = np.array(cdata['color'])
                else:
                    color = np.array(cdata['color']) * 0.15
                pcd_colors[cdata['indices']] = color
        self._recolor_meshes()

        # Highlight selected viewpoint marker
        cluster_data = self._get_cluster_data(selected_cluster_name)
        if cluster_data and cluster_data.get('viewpoint_data'):
            self.scene_widget.scene.modify_geometry_material(
                f"{selected_cluster_name}_viewpoint", Materials.selected_viewpoint_material)
        return True

    # ── Ray casting ───────────────────────────────────────────────────────────

    def cast_ray_from_center(self) -> dict:
        """Cast a ray from the camera centre toward the scene.

        Returns ``{'hit': True, 'point': np.ndarray, 'distance': float}`` or
        ``{'hit': False}``.  Skips the cast if the view matrix is unchanged.
        """
        camera      = self.scene_widget.scene.camera
        view_matrix = camera.get_view_matrix()

        if np.array_equal(view_matrix, self.current_view_matrix):
            return {'hit': False}

        self.current_view_matrix = view_matrix
        inv = np.linalg.inv(view_matrix)
        cam_pos = inv[:3, 3]
        cam_fwd = -inv[:3, 2]
        cam_fwd = cam_fwd / np.linalg.norm(cam_fwd)

        ray = np.array([[
            cam_pos[0], cam_pos[1], cam_pos[2],
            cam_fwd[0], cam_fwd[1], cam_fwd[2],
        ]], dtype=np.float32)

        rays_tensor = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
        result      = self.ray_casting_scene.cast_rays(rays_tensor)

        if len(result['t_hit']) > 0:
            t = result['t_hit'][0].item()
            if t < np.inf:
                intersection = cam_pos + t * cam_fwd
                return {'hit': True, 'point': intersection, 'distance': t}

        return {'hit': False}
