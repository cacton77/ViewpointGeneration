#!/usr/bin/env python3
"""visualizer.py – Scene geometry management and cluster-viewpoint rendering strategies."""

from __future__ import annotations

import colorsys
import copy
import enum
import json
import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from matplotlib import colormaps

from viewpoint_generation.assets.materials import Materials
from viewpoint_generation.mesh_utils import submesh_from_faces


# ─────────────────────────────────────────────────────────────────────────────
# Rendering Mode Enum
# ─────────────────────────────────────────────────────────────────────────────

class RegionSurfaceMode(enum.Enum):
    """How a region's *surface* is colored. Exclusive (one at a time)."""
    SOLID   = "solid"    # Uniform region color
    CLUSTER = "cluster"  # Vertex/point colored by owning cluster


class OverlayKind(enum.Enum):
    """Per-viewpoint overlay geometries. Inclusive (any combination on)."""
    MARKER        = "marker"         # Sphere/arrow/axes marker at the camera position
    FOV_CYLINDER  = "fov_cylinder"   # Wireframe FOV coverage cylinder at the surface target
    ORIGIN_LINE   = "origin_line"    # Line from the surface origin to the camera position
    FRUSTUM       = "frustum"        # Camera frustum at the viewpoint
    ORIGIN_MARKER = "origin_marker"  # Small white sphere at the surface origin of each cluster


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_order_indices(order, selected_algorithm=None):
    """Resolve a region's traversal order into a flat list of cluster indices.

    ``order`` may be a plain list (identity / pre-traversal order) or, after
    traversal optimization, a dict keyed by TSP algorithm name. Each algorithm
    maps to ``{'order': [...], 'distance': ...}`` (older files stored the bare
    index list — both are handled). For a dict, prefer ``selected_algorithm``
    (the ``selected_traversal_algorithm`` parameter on the task_planning node,
    surfaced here via the GUI); otherwise fall back to the first available
    algorithm's path.
    """
    if isinstance(order, dict):
        if not order:
            return []
        entry = order.get(selected_algorithm, next(iter(order.values())))
        if isinstance(entry, dict):
            return entry.get('order', [])
        return entry
    return order


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


def _build_fov_cylinder(viewpoint_data: dict,
                        n_segments: int = 24,
                        n_struts: int = 8,
                        color=(1.0, 1.0, 1.0)) -> o3d.geometry.LineSet | None:
    """Build a wireframe FOV-cylinder LineSet (in mm) at a cluster's surface target.

    The cylinder is the coverage volume of the field of view: radius
    ``fov_diameter / 2`` (lateral FOV) and height ``dof`` (axial depth of field),
    centred on the cluster's surface ``origin`` and aligned along its averaged view
    ``direction``. It straddles the surface by ±dof/2, matching the greedy-cover
    coverage predicate. Returns None when camera dimensions are unavailable.
    """
    fov_diameter = viewpoint_data.get('fov_diameter')
    dof          = viewpoint_data.get('dof')
    if not fov_diameter or not dof:
        return None

    center = 1000.0 * np.array(viewpoint_data['origin'], dtype=float)
    axis   = np.array(viewpoint_data['direction'], dtype=float)
    norm   = np.linalg.norm(axis)
    if norm < 1e-9:
        return None
    axis = axis / norm

    radius      = 1000.0 * fov_diameter / 2.0
    half_height = 1000.0 * dof / 2.0

    # Orthonormal frame perpendicular to the view axis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    thetas = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    ring = np.array([np.cos(t) * u + np.sin(t) * v for t in thetas])  # (n,3)

    top    = center + half_height * axis + radius * ring
    bottom = center - half_height * axis + radius * ring
    points = np.vstack([top, bottom])

    lines = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        lines.append([i, j])                              # top ring
        lines.append([n_segments + i, n_segments + j])    # bottom ring
    # A handful of vertical struts so the tube reads as a cylinder.
    strut_step = max(1, n_segments // max(1, n_struts))
    for i in range(0, n_segments, strut_step):
        lines.append([i, n_segments + i])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    ls.paint_uniform_color(list(color))
    return ls


# ─────────────────────────────────────────────────────────────────────────────
# Viewpoint overlay builders + registry
#
# Each overlay kind builds one geometry per cluster-viewpoint. Overlays are
# inclusive: any combination can be enabled at once, and each is added to the
# scene under its own name (f"{cluster_name}_vp_{kind.value}") so they coexist.
# The registry maps a kind to (builder, default_material, selected_material);
# the builder takes the cluster's viewpoint_data dict and returns a geometry
# (or None when it cannot be built, e.g. an FOV cylinder with no camera dims).
# ─────────────────────────────────────────────────────────────────────────────

def _build_origin_line(viewpoint_data: dict) -> o3d.geometry.LineSet:
    """A line from the surface origin to the camera position (in mm)."""
    origin   = 1000.0 * np.array(viewpoint_data['origin'], dtype=float)
    position = 1000.0 * np.array(viewpoint_data['position'], dtype=float)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array([origin, position]))
    ls.lines  = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    ls.paint_uniform_color([0.9, 0.9, 0.9])
    return ls


def _build_origin_marker(viewpoint_data: dict,
                         radius: float = Materials.origin_marker_radius) -> o3d.geometry.TriangleMesh:
    """A small sphere at the cluster's surface origin. Always plain white,
    not tinted by cluster color or selection, so it reads as a fixed
    reference point rather than another cluster-colored blob."""
    origin = 1000.0 * np.array(viewpoint_data['origin'], dtype=float)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(origin)
    sphere.paint_uniform_color([1.0, 1.0, 1.0])
    sphere.compute_vertex_normals()
    return sphere


def _build_viewpoint_point_marker(viewpoint_data: dict) -> o3d.geometry.PointCloud:
    """A single fat point at the camera position — the same marker style as the
    unreachable viewpoints. Painted white so the overlay material tints it
    (green when unselected, the selected material when highlighted)."""
    position = 1000.0 * np.array(viewpoint_data['position'], dtype=float)
    marker = o3d.geometry.PointCloud()
    marker.points = o3d.utility.Vector3dVector(position.reshape(1, 3))
    marker.paint_uniform_color([1.0, 1.0, 1.0])
    return marker


# kind -> (builder(viewpoint_data) -> geometry|None, default_material, selected_material)
# default_material may instead be a callable(viewpoint_data) -> MaterialRecord
# for a kind that needs per-cluster material variation (MARKER: green for a
# photometric-tier viewpoint, yellow for standard-tier, red for an
# inaccessible one) — resolve via _resolve_default_material rather than using
# it directly.
def _viewpoint_marker_material(vp):
    mode = (vp or {}).get('imaging_mode')
    if mode == 'inaccessible':
        return Materials.unreachable_marker_material
    if mode == 'standard':
        return Materials.standard_viewpoint_marker_material
    return Materials.viewpoint_marker_material


_OVERLAY_REGISTRY: dict = {
    OverlayKind.MARKER: (
        _build_viewpoint_point_marker,
        _viewpoint_marker_material,
        Materials.selected_viewpoint_marker_material,
    ),
    OverlayKind.FOV_CYLINDER: (
        lambda vp: _build_fov_cylinder(vp, color=(1.0, 1.0, 1.0)),
        Materials.fov_cylinder_material,
        Materials.selected_fov_cylinder_material,
    ),
    OverlayKind.ORIGIN_LINE: (
        _build_origin_line,
        Materials.overlay_line_material,
        Materials.selected_overlay_line_material,
    ),
    OverlayKind.FRUSTUM: (
        _build_frustum,
        Materials.overlay_line_material,
        Materials.selected_overlay_line_material,
    ),
    OverlayKind.ORIGIN_MARKER: (
        _build_origin_marker,
        Materials.origin_marker_material,
        Materials.selected_origin_marker_material,
    ),
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

        # Visualization state: the region surface is colored by one exclusive
        # mode, while any combination of per-viewpoint overlays may be enabled.
        self._surface_mode: RegionSurfaceMode = RegionSurfaceMode.SOLID
        # Overlays shown for non-selected viewpoints vs. the selected viewpoint.
        # Independent sets so the selected viewpoint can show a different subset
        # (e.g. only the selected viewpoint shows its FOV cylinder).
        self._enabled_overlays: set[OverlayKind] = {
            OverlayKind.MARKER, OverlayKind.ORIGIN_MARKER}
        self._selected_overlays: set[OverlayKind] = {
            OverlayKind.MARKER, OverlayKind.FOV_CYLINDER,
            OverlayKind.ORIGIN_LINE, OverlayKind.ORIGIN_MARKER}
        self._built_overlays: set[OverlayKind] = set()

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
        # Which TSP algorithm's path to draw when a region's order is a
        # per-algorithm dict. Set by the GUI from the task_planning parameter
        # (None → first available algorithm). Replaces the old
        # results-dict 'selected_traversal_algorithm' field.
        self.selected_traversal_algorithm = None
        # The algorithm whose cluster order the scene's clusters were numbered
        # in, frozen at load time. The file tree uses this (not the live
        # selection) to order/number clusters so changing the algorithm only
        # redraws the path — it does not renumber clusters or rebuild the tree.
        self._cluster_order_algorithm = None
        self.region_number   : int              = 0
        self.cluster_number  : int              = 0

        # Mesh vertex coloring state
        self.meshes             : dict = {}
        self.mesh_vertex_to_pcd : dict = {}
        self.noise_faces        : list = []
        self.noise_surface_names: list[str] = []

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
        self.mesh_has_vertex_colors          = False
        self.show_joint_path_flag            = True
        self.show_unreachable_flag           = True
        self.show_blind_spots_flag           = True

        # Names of every origin-frame model geometry added via add_geometry
        # (mesh, point cloud, curvatures, regions, clusters, viewpoints, model
        # bounding box). apply_model_placement transforms exactly these to the
        # part's live placement; world-fixed geometry (ground grid, world axes,
        # skybox) is added via self.scene.add_geometry directly and is left put.
        self._placed_geometry_names: set = set()

    def apply_model_placement(self, T_object_from_model: np.ndarray) -> None:
        """Place all origin-frame model geometry at the part's live pose.

        T_object_from_model is the object_frame <- model(origin) frame
        transform (metres) from the tsdf_pose TF. Model geometry was authored
        in the origin frame, scaled to the visualizer's mm working unit, and
        baked with a -90 deg X display rotation (see add_geometry). The
        equivalent display-frame transform is therefore
        Rx . T_mm . Rx^-1, with T_mm = T scaled to mm, applied per geometry
        via Open3DScene.set_geometry_transform."""
        T = np.asarray(T_object_from_model, dtype=float)
        if T.shape != (4, 4):
            return
        Rx = np.eye(4)
        Rx[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
        T_mm = T.copy()
        T_mm[:3, 3] *= 1000.0
        M = Rx @ T_mm @ np.linalg.inv(Rx)
        for name in self._placed_geometry_names:
            if self.scene.has_geometry(name):
                self.scene.set_geometry_transform(name, M)

    def on_mouse(self, event) -> gui.Widget.EventCallbackResult:
        """Delegate mouse events to the turntable camera controller."""
        return self.camera.on_mouse(event)

    # ── Region surface mode (exclusive) ──────────────────────────────────────

    def set_region_surface_mode(self, mode: RegionSurfaceMode):
        """Set how region surfaces are colored (solid vs. by cluster) and repaint."""
        self._surface_mode = mode
        self._render_surfaces()

    def _render_surfaces(self):
        """Apply the active surface mode to every region surface, honoring the
        current selection (selected region opaque/full-bright, others dimmed)."""
        sel = self.selected_region_name
        for region_name in self.geometries_dict:
            selected = (not sel) or (region_name == sel)
            self._apply_surface(region_name, selected)

    # Flat grey applied to non-selected region surfaces so cluster/solid
    # coloring only ever appears on the region currently in focus.
    NONSELECTED_SURFACE_GREY = 0.6

    def _apply_surface(self, region_name: str, selected: bool):
        """Color one region's surface and set its opacity.

        The selected region is colored by the active mode (solid or by cluster);
        every non-selected region is painted a flat grey regardless of the mode,
        so cluster coloring never distracts from the region in focus. Meshes flip
        opaque↔transparent via material; point-cloud surfaces (used when no mesh
        is present) darken instead, since Open3D point clouds do not alpha-blend
        reliably.
        """
        surf = self.geometries_dict.get(region_name, {}).get('surface')
        if not surf:
            return
        name = surf['name']
        if selected:
            base = (surf['colors_solid'] if self._surface_mode == RegionSurfaceMode.SOLID
                    else surf['colors_cluster'])
        else:
            base = np.full_like(surf['colors_solid'], self.NONSELECTED_SURFACE_GREY)
        geom = surf['geometry']
        self.scene.remove_geometry(name)
        if surf['kind'] == 'mesh':
            geom.vertex_colors = o3d.utility.Vector3dVector(base)
            mat = (Materials.mesh_material_vertex_colors if selected
                   else Materials.mesh_material_vertex_colors_transparent)
            self.add_geometry(name, geom, mat)
        else:  # point-cloud surface
            disp = base if selected else base * 0.15
            geom.colors = o3d.utility.Vector3dVector(disp)
            self.add_geometry(name, geom, Materials.point_cloud_material)
        self.scene.show_geometry(name, self.show_mesh_flag)

    # ── Viewpoint overlays (inclusive) ───────────────────────────────────────

    @staticmethod
    def _overlay_geo_name(cluster_name: str, kind: OverlayKind) -> str:
        return f"{cluster_name}_vp_{kind.value}"

    def set_overlay_enabled(self, kind: OverlayKind, on: bool):
        """Enable/disable one overlay kind for the non-selected viewpoints."""
        if on:
            self._enabled_overlays.add(kind)
            if kind not in self._built_overlays:
                self._build_overlay_kind(kind)
        else:
            self._enabled_overlays.discard(kind)
        self._update_overlay_visibility()

    def set_selected_overlay_enabled(self, kind: OverlayKind, on: bool):
        """Enable/disable one overlay kind for the currently-selected viewpoint."""
        if on:
            self._selected_overlays.add(kind)
            if kind not in self._built_overlays:
                self._build_overlay_kind(kind)
        else:
            self._selected_overlays.discard(kind)
        self._update_overlay_visibility()

    @staticmethod
    def _resolve_default_material(default_mat, vp_data):
        """default_mat is either a fixed MaterialRecord (most overlay kinds)
        or a callable(vp_data) -> MaterialRecord (MARKER, whose color varies
        per cluster by imaging_mode) — resolve either form uniformly."""
        return default_mat(vp_data) if callable(default_mat) else default_mat

    def _build_overlay_kind(self, kind: OverlayKind):
        """Build geometry for one overlay kind for every cluster viewpoint."""
        builder, default_mat, _ = _OVERLAY_REGISTRY[kind]
        for region_data in self.geometries_dict.values():
            for cluster_name, cdata in region_data.get('clusters', {}).items():
                vp = cdata.get('viewpoint_data')
                if not vp:
                    continue
                name = self._overlay_geo_name(cluster_name, kind)
                self.scene.remove_geometry(name)
                geo = builder(vp)
                if geo is None:
                    continue
                self.add_geometry(name, geo, self._resolve_default_material(default_mat, vp))
        self._built_overlays.add(kind)

    def _update_overlay_visibility(self):
        """Show/colour overlays for the selected region (all regions when none is
        selected), gated by the master ``show_viewpoints`` flag.

        The selected viewpoint uses its own overlay set (``_selected_overlays``)
        and the highlight materials; every other viewpoint uses the shared
        ``_enabled_overlays`` set and default materials.
        """
        sel_region  = self.selected_region_name
        sel_cluster = self.selected_cluster_name
        for region_name, region_data in self.geometries_dict.items():
            region_visible = (not sel_region) or (region_name == sel_region)
            for cluster_name, cdata in region_data.get('clusters', {}).items():
                is_selected_vp = (cluster_name == sel_cluster)
                kinds = (self._selected_overlays if is_selected_vp
                         else self._enabled_overlays)
                for kind in self._built_overlays:
                    name = self._overlay_geo_name(cluster_name, kind)
                    if not self.scene.has_geometry(name):
                        continue
                    visible = (self.show_viewpoints_flag
                               and region_visible
                               and kind in kinds)
                    self.scene.show_geometry(name, visible)
                    _, default_mat, selected_mat = _OVERLAY_REGISTRY[kind]
                    mat = (selected_mat if is_selected_vp else
                           self._resolve_default_material(default_mat, cdata.get('viewpoint_data')))
                    self.scene_widget.scene.modify_geometry_material(name, mat)

    # ── Core scene helper ────────────────────────────────────────────────────

    # Multiplier on the bounding-box diagonal that sets the camera's distance
    # from the scene center. Larger pulls the camera back; smaller zooms in.
    CAMERA_FRAMING_FACTOR = 1.0

    def _content_bbox(self, mesh_bbox, view_positions):
        """Model-space AABB spanning the loaded content used for framing.

        Unions the mesh bounding box with the viewpoint positions (both in
        correctly-scaled model space). Falls back to the point-cloud bounds when
        no mesh is present. Deliberately excludes ``scene.bounding_box`` so
        hidden/stale geometry (e.g. a mis-scaled point cloud) cannot distort the
        ground grid or camera framing. Returns None when nothing is available.
        """
        bounds = []
        if mesh_bbox is not None:
            bounds.append(np.asarray(mesh_bbox.get_min_bound()))
            bounds.append(np.asarray(mesh_bbox.get_max_bound()))
        if view_positions:
            vp = np.asarray(view_positions)
            bounds.append(vp.min(axis=0))
            bounds.append(vp.max(axis=0))
        if not bounds and self.point_cloud is not None:
            pcd_bbox = self.point_cloud.get_axis_aligned_bounding_box()
            bounds.append(np.asarray(pcd_bbox.get_min_bound()))
            bounds.append(np.asarray(pcd_bbox.get_max_bound()))
        if not bounds:
            return None
        stacked = np.vstack(bounds)
        return o3d.geometry.AxisAlignedBoundingBox(
            min_bound=stacked.min(axis=0), max_bound=stacked.max(axis=0))

    def _reset_camera_to_bbox(self, bbox: o3d.geometry.AxisAlignedBoundingBox | None = None):
        """Reset the turntable camera so the mesh fills the view.

        All geometry is stored in the scene after a -90° X rotation
        ([x,y,z] → [x, z, -y]), so the camera center must be derived from
        the *rotated* bounding-box centre rather than the raw bbox centre.
        The eye is placed one bounding-box diagonal away in the [1,1,1]
        direction of the rotated frame, which keeps the camera well outside
        the mesh regardless of its position or size.

        When ``bbox`` is None (no mesh loaded yet) the default 30 cm bbox is
        used so the camera framing matches the default ground grid.
        """
        if bbox is None:
            bbox = self._default_bbox()

        # -90° X rotation: [x, y, z] → [x, z, -y]
        raw_center = bbox.get_center()
        rotated_center = np.array([raw_center[0], raw_center[2], -raw_center[1]])

        diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        offset_dir = np.array([1., 1., 1.]) / np.sqrt(3)
        eye = rotated_center + offset_dir * diagonal * self.CAMERA_FRAMING_FACTOR

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
        # Track for apply_model_placement (re-adding resets the per-geometry
        # transform to identity, so the tick loop re-applies the placement).
        self._placed_geometry_names.add(name)

    # ── Lighting / ground plane ──────────────────────────────────────────────

    def setup_multi_directional_lighting(self):
        sun_direction = np.array([0.577, -0.577, -0.577], dtype=np.float32)
        self.scene.set_lighting(
            self.scene.LightingProfile.NO_SHADOWS, sun_direction)

    # Default scene size (mm) used for both the ground grid and the camera
    # framing when no mesh/bbox is present. A 300 mm (30 cm) square footprint
    # is centred on the origin so the startup grid and camera distance agree.
    DEFAULT_BBOX_SIZE = 300.0

    def _default_bbox(self) -> o3d.geometry.AxisAlignedBoundingBox:
        """A 30 cm × 30 cm bbox centred on the origin, used on startup before
        any mesh is loaded. Drives both ``add_xy_plane`` and
        ``_reset_camera_to_bbox`` so the grid and camera stay in sync."""
        half = self.DEFAULT_BBOX_SIZE / 2.0
        return o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-half, -half, 0.0]),
            max_bound=np.array([half, half, 0.0]))

    def setup_default_scene(self):
        """Draw the default ground grid and frame the camera to it.

        Used on startup before any mesh/results are loaded. Both the grid and
        the camera derive from ``_default_bbox`` so they stay in sync."""
        self.add_xy_plane()
        self._reset_camera_to_bbox()

    def add_xy_plane(self, bbox=None):
        # No mesh loaded yet — fall back to the default bbox so the grid is
        # sized identically to how the camera is framed.
        if bbox is None:
            bbox = self._default_bbox()

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
        # algorithm name; this selects which algorithm's path to render. The
        # choice comes from the GUI/task_planning parameter, not the file.
        selected_algorithm = self.selected_traversal_algorithm
        # Freeze the cluster-numbering algorithm for this load so the file tree
        # stays stable when the path algorithm is later changed.
        self._cluster_order_algorithm = selected_algorithm

        # ── Load meshes and per-region point clouds from the results dict ──────
        bbox = None
        should_reset_camera = True
        scale_map = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'in': 25.4, 'ft': 304.8}

        # Loading a new results file invalidates everything from the previous
        # one, so remove all downstream geometry before rebuilding: meshes,
        # curvatures, and every region/cluster/viewpoint/path/view manifold.
        # _clear_result_geometry must run while the name lists still hold the
        # previous file's names so its removals take effect.
        self.scene.remove_geometry("mesh")
        for name in self.mesh_names:
            self.scene.remove_geometry(name)
        self.scene.remove_geometry("point_cloud")
        self.scene.remove_geometry("curvatures")
        self._clear_result_geometry()
        self.noise_faces            = []
        self.meshes                 = {}
        self.mesh_vertex_to_pcd     = {}
        self.mesh_has_vertex_colors = False
        self.geometries_dict        = {}

        new_mesh_names = []
        new_meshes     = {}
        new_geometries_dict: dict = {}
        show_clusters   = False
        show_viewpoints = False
        traversal_order: list[list[int]] = []
        noise_faces_by_mesh: dict = {}   # mesh_idx (str) -> list of face indices
        # Viewpoint positions (model-space, mm) accumulated across all regions.
        # Used to frame the grid/camera without relying on scene.bounding_box,
        # which also counts hidden geometry (e.g. a stale, mis-scaled point
        # cloud) and would otherwise blow up the ground grid.
        all_view_positions: list = []

        cmap = colormaps[Materials.regions_colormap]
        np.random.seed(1)

        for mesh_idx, mesh_entry in enumerate(results_dict.get('meshes', [])):
            mesh_file  = mesh_entry.get('file', '')
            mesh_units = mesh_entry.get('units', 'mm')
            mesh_name  = f"mesh_{mesh_idx}"

            mesh = None
            if mesh_file:
                try:
                    mesh = o3d.io.read_triangle_mesh(mesh_file)
                    if mesh.is_empty():
                        mesh = None
                    else:
                        mesh.scale(scale_map.get(mesh_units, 1.0), center=(0, 0, 0))

                        # Mesh, regions, and viewpoints are all authored in the
                        # mesh's origin frame; the part's physical placement is
                        # applied live as a per-geometry display transform from
                        # the object_frame->model_frame TF (see the GUI's
                        # apply_model_placement), never baked in here.

                        if mesh_idx == 0:
                            bbox = mesh.get_axis_aligned_bounding_box()
                            self.mesh_name  = mesh_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                            self.mesh_units = mesh_units

                            # Only re-frame the camera/grid when the mesh's
                            # identity (file + units) actually changes -- a
                            # different part, or a units correction, both of
                            # which can move the content far outside the old
                            # framing. A pose nudge or a same-object results
                            # recompute (regions/clusters/viewpoints) reloads
                            # this same mesh file, so the camera stays put:
                            # resetting it on every pose slider tick made
                            # interactive pose authoring unusable.
                            new_identity = (mesh_file, mesh_units)
                            should_reset_camera = (
                                new_identity != getattr(self, '_loaded_mesh_identity', None))
                            self._loaded_mesh_identity = new_identity
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
                    mesh = None

            if mesh is None:
                continue

            region_order = mesh_entry.get('order', [])

            # FOV dimensions persisted with the results; consumed by the
            # FOVCylinderRenderer. Absent on older results files (cylinder is
            # then skipped per-cluster).
            camera_config = mesh_entry.get('camera_config', {})

            mesh_noise_faces = mesh_entry.get('noise_faces', [])
            noise_faces_by_mesh[str(mesh_idx)] = mesh_noise_faces
            self.noise_faces.extend(mesh_noise_faces)

            for region_id in region_order:
                region_name = f"mesh_{mesh_idx}_region_{region_id}"
                region_dict = mesh_entry['regions'][region_id]

                region_color = np.array(list(cmap(np.random.rand())))[0:3]

                # Each region owns its own sampled point cloud (produced by
                # segment_regions()); load it directly rather than slicing a
                # shared cloud by index.
                region_pcd = None
                pcd_file = region_dict.get('point_cloud', {}).get('file', '')
                if pcd_file and os.path.exists(pcd_file):
                    try:
                        region_pcd = o3d.io.read_point_cloud(pcd_file)
                        # Region point clouds are always sampled/saved in
                        # meters (ViewpointGeneration.set_mesh_file converts
                        # the mesh to meters before segment_regions() samples
                        # each region's cloud), independent of mesh_units
                        # (which only describes the original raw mesh file).
                        # Scale to mm once here so region_pcd-derived geometry
                        # (blind-spot markers, the surface color KDTree below)
                        # matches every other geometry in this mm-scaled scene.
                        region_pcd.scale(1000.0, center=(0, 0, 0))
                    except Exception as e:
                        print(f"Error loading region point cloud {pcd_file}: {e}")

                new_geometries_dict[region_name] = {
                    'faces': region_dict.get('faces', []),
                    'point_cloud': region_pcd,
                    'color': region_color,
                }

                # Blind-spot points (fov_clustering's greedy-cover safety
                # break — surface points no candidate viewpoint can see
                # unoccluded). Independent of clusters/viewpoints: a region
                # can be entirely blind (zero clusters) and still need this
                # marked, so it is not gated on region_dict['clusters'].
                blind_spot_points = region_dict.get('blind_spots', {}).get('points', [])
                blind_spot_markers = None
                if blind_spot_points and region_pcd is not None:
                    blind_spot_markers = region_pcd.select_by_index(blind_spot_points)
                    # select_by_index inherits the region cloud's own (grey)
                    # vertex colors; repaint to match blind_spot_marker_material
                    # explicitly, same convention as _build_unreachable_markers.
                    blind_spot_markers.paint_uniform_color([1.0, 0.0, 1.0])
                new_geometries_dict[region_name]['blind_spot_markers'] = blind_spot_markers

                region_has_viewpoints = False

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

                        # Cluster indices are already local to the region's own
                        # point cloud — no local→global remapping needed.
                        cluster_indices = cluster_dict['points']
                        cluster_pcd = (region_pcd.select_by_index(cluster_indices)
                                       if region_pcd is not None else None)

                        vp_data = None
                        if 'viewpoint' in cluster_dict:
                            show_viewpoints = True
                            region_has_viewpoints = True
                            vp_raw = cluster_dict['viewpoint']
                            # Store centroid for LinesRenderer
                            centroid = (np.mean(np.asarray(cluster_pcd.points), axis=0)
                                        if cluster_pcd is not None
                                        else np.array(vp_raw['origin']))
                            vp_data = {
                                'origin':           vp_raw['origin'],
                                'position':         vp_raw['position'],
                                'direction':        vp_raw['direction'],
                                'orientation':      vp_raw['orientation'],
                                'imaging_mode':     vp_raw.get('imaging_mode', 'photometric'),
                                'cluster_centroid': centroid.tolist(),
                                'cluster_color':    cluster_color.tolist(),
                                'fov_diameter':     camera_config.get('fov_diameter'),
                                'dof':              camera_config.get('dof'),
                            }

                            position  = 1000.0 * np.array(vp_raw['position'])
                            all_view_positions.append(position)

                        new_geometries_dict[region_name]['clusters'][cluster_name] = {
                            'pcd':            cluster_pcd,
                            'indices':        cluster_indices,
                            'color':          cluster_color,
                            'viewpoint_data': vp_data,
                            # Cluster index in the results JSON. The geometry name
                            # is numbered by the frozen load-time traversal order,
                            # but selection is driven by the *live* algorithm order
                            # (see select_cluster), so we keep the stable id here.
                            'cluster_id':     cluster_id,
                        }

                if region_has_viewpoints:
                    joint_path, joint_markers = self._build_joint_path(
                        region_dict, selected_algorithm)
                    new_geometries_dict[region_name]['joint_path'] = joint_path
                    new_geometries_dict[region_name]['joint_markers'] = joint_markers

                    new_geometries_dict[region_name]['unreachable_markers'] = (
                        self._build_unreachable_markers(region_dict, selected_algorithm))

        if not new_geometries_dict:
            print("No regions loaded; showing plain mesh(es).")
            for mesh_name, mesh in new_meshes.items():
                self.add_geometry(mesh_name, mesh, Materials.mesh_material)
            self.mesh_names = new_mesh_names
            if bbox is not None:
                self.add_xy_plane(bbox)
                if should_reset_camera:
                    self._reset_camera_to_bbox(bbox)
            self.show_mesh(True)
            return

        self.mesh_names      = []
        self.region_names    = []
        self.cluster_names   = []
        self.traversal_order = traversal_order
        self.geometries_dict = new_geometries_dict
        self.region_number   = 0
        self.cluster_number  = 0
        self.selected_mesh_idx     = -1
        self.selected_region_name  = ''
        self.selected_cluster_name = ''

        # ── Build per-region surfaces (exact submesh per region) and per-mesh
        #    noise-face surfaces ────────────────────────────────────────────────
        self._build_region_surfaces(new_meshes)
        self._build_noise_surfaces(new_meshes, noise_faces_by_mesh)

        for region_name, region_data in self.geometries_dict.items():
            self.region_names.append(region_name)
            for cluster_name in region_data.get('clusters', {}):
                self.cluster_names.append(cluster_name)

        # Add region surfaces (opaque, current surface mode) and per-region
        # path geometry.
        self._render_surfaces()
        for region_name, region_data in self.geometries_dict.items():
            joint_path = region_data.get('joint_path')
            if joint_path is not None:
                self.add_geometry(f"{region_name}_joint_path", joint_path,
                                  Materials.joint_path_material)
            joint_markers = region_data.get('joint_markers')
            if joint_markers is not None:
                self.add_geometry(f"{region_name}_joint_markers", joint_markers,
                                  Materials.joint_marker_material)
            unreachable_markers = region_data.get('unreachable_markers')
            if unreachable_markers is not None:
                self.add_geometry(f"{region_name}_unreachable_markers",
                                  unreachable_markers,
                                  Materials.unreachable_marker_material)
            blind_spot_markers = region_data.get('blind_spot_markers')
            if blind_spot_markers is not None:
                self.add_geometry(f"{region_name}_blind_spot_markers",
                                  blind_spot_markers,
                                  Materials.blind_spot_marker_material)

        # Build every overlay kind used by either the shared or the
        # selected-viewpoint set, for every cluster.
        self._built_overlays = set()
        if show_viewpoints:
            for kind in self._enabled_overlays | self._selected_overlays:
                self._build_overlay_kind(kind)

        print(f"Loaded regions from {file_path}")

        # ── Camera + XY plane setup ───────────────────────────────────────────
        if bbox is not None:
            self.add_xy_plane(bbox)
            if should_reset_camera:
                self._reset_camera_to_bbox(bbox)

        # ── Initial visibility ────────────────────────────────────────────────
        self.show_mesh(True)
        self.show_curvatures(False)
        self.show_noise_points(False)
        # Independent of clusters/viewpoints — a region can be entirely
        # blind (zero clusters) and still carry blind-spot markers.
        self.show_blind_spots(self.show_blind_spots_flag)

        if show_clusters:
            # Always default to the solid region coloring once viewpoints
            # exist; cluster coloring remains available via the Region
            # Surface menu but is no longer the load-time default.
            self.set_region_surface_mode(RegionSurfaceMode.SOLID)
            self.show_viewpoints(show_viewpoints)
            if show_viewpoints:
                # Cartesian path / unreachable markers only exist after traversal
                # optimization; re-assert their current toggle state so any
                # already-built geometry is scoped correctly on load.
                self.show_joint_path(self.show_joint_path_flag)
                self.show_unreachable(self.show_unreachable_flag)
            # Frame the grid/camera from a model-space content bbox (mesh ∪
            # viewpoint positions) rather than scene.bounding_box. The scene box
            # also counts hidden geometry — notably a stale, mis-scaled point
            # cloud (e.g. after a wrong-units load is corrected) — which would
            # otherwise inflate the ground grid enormously. Both the mesh bbox
            # and the accumulated viewpoint positions are in correctly-scaled
            # model space, and _reset_camera_to_bbox/add_xy_plane expect that.
            content_bbox = self._content_bbox(bbox, all_view_positions)
            if content_bbox is not None:
                self.add_xy_plane(content_bbox)
                if should_reset_camera:
                    self._reset_camera_to_bbox(content_bbox)
        elif self.region_names:
            self.set_region_surface_mode(RegionSurfaceMode.SOLID)
            self.show_viewpoints(False)
        else:
            self.show_point_cloud(True)
            self.show_viewpoints(False)

    # ── Per-region surface construction ──────────────────────────────────────

    def _build_region_surfaces(self, new_meshes):
        """Build each region's exact submesh directly from its face list (no
        subdivision or nearest-vertex projection needed — a region's faces
        *are* the region), precomputing solid and cluster vertex color
        arrays. Populates ``geometries_dict[region]['surface']``."""
        for region_name, rdata in self.geometries_dict.items():
            mesh_name = region_name.split('_region_')[0]
            mesh = new_meshes.get(mesh_name)
            if mesh is None or not rdata['faces']:
                continue

            mesh_triangles = np.asarray(mesh.triangles)
            faces = np.asarray(rdata['faces'])
            sub, _used = submesh_from_faces(mesh, mesh_triangles[faces])

            n_verts = len(sub.vertices)
            colors_solid = np.tile(rdata['color'], (n_verts, 1))
            colors_cluster = colors_solid.copy()

            region_pcd = rdata.get('point_cloud')
            if region_pcd is not None and len(region_pcd.points) > 0:
                # Color each submesh vertex by whichever FOV cluster owns the
                # nearest sampled point, scoped to this region's own (small)
                # point cloud and submesh — the same nearest-point technique
                # as before, but local instead of global.
                pcd_cluster_color = np.tile(
                    rdata['color'], (len(region_pcd.points), 1))
                for cdata in rdata.get('clusters', {}).values():
                    pcd_cluster_color[cdata['indices']] = cdata['color']

                pcd_tree = o3d.geometry.KDTreeFlann(region_pcd)
                verts = np.asarray(sub.vertices)
                v2p = np.zeros(len(verts), dtype=int)
                for i, vertex in enumerate(verts):
                    [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
                    v2p[i] = idx[0]
                colors_cluster = pcd_cluster_color[v2p]

            rdata['surface'] = {
                'kind':           'mesh',
                'geometry':       sub,
                'name':           f"{region_name}_surface",
                'colors_solid':   colors_solid,
                'colors_cluster': colors_cluster,
            }
            self.mesh_has_vertex_colors = True

    def _build_noise_surfaces(self, new_meshes, noise_faces_by_mesh):
        """Build a flat red submesh per mesh for faces no region claimed."""
        for mesh_name, mesh in new_meshes.items():
            mesh_idx = mesh_name.rsplit('_', 1)[-1]
            faces = noise_faces_by_mesh.get(mesh_idx, [])
            if not faces:
                continue

            mesh_triangles = np.asarray(mesh.triangles)
            sub, _used = submesh_from_faces(mesh, mesh_triangles[np.asarray(faces)])
            sub.paint_uniform_color([1.0, 0.0, 0.0])

            name = f"{mesh_name}_noise_surface"
            self.add_geometry(name, sub, Materials.mesh_material)
            self.scene.show_geometry(name, self.show_noise_points_flag)
            self.noise_surface_names.append(name)


    # ── Clear helpers ─────────────────────────────────────────────────────────

    def _clear_result_geometry(self):
        """Remove all region/cluster/viewpoint/joint-path geometry and reset
        the tracking lists.

        The geometry-removing clears (viewpoints, joint paths) iterate the
        cluster/region name lists, so they must run *before* clear_clusters
        and clear_regions empty those lists — otherwise nothing is removed and
        stale geometry from the previous file lingers in the scene.
        """
        self.clear_viewpoints()
        self.clear_joint_paths()
        self.clear_surfaces()
        self.clear_clusters()
        self.clear_regions()

    def clear_regions(self):
        self.region_names = []

    def clear_surfaces(self):
        for name in self.region_names:
            self.scene.remove_geometry(f"{name}_surface")
        for name in self.noise_surface_names:
            self.scene.remove_geometry(name)
        self.noise_surface_names = []

    def clear_clusters(self):
        self.cluster_names = []

    def clear_viewpoints(self):
        for name in self.cluster_names:
            self.scene.remove_geometry(f"{name}_viewpoint")  # legacy name
            for kind in OverlayKind:
                self.scene.remove_geometry(self._overlay_geo_name(name, kind))

    def clear_joint_paths(self):
        for name in self.region_names:
            self.scene.remove_geometry(f"{name}_joint_path")
            self.scene.remove_geometry(f"{name}_joint_markers")
            self.scene.remove_geometry(f"{name}_unreachable_markers")
            self.scene.remove_geometry(f"{name}_blind_spot_markers")

    # ── Visibility (pure scene — no menu, no ROS, no cross-calls) ─────────────

    def show_mesh(self, show: bool):
        self.show_mesh_flag = show
        for name in self.mesh_names:
            self.scene.show_geometry(name, show)
        # Per-region surfaces are the displayed model once results are loaded.
        for region_data in self.geometries_dict.values():
            surf = region_data.get('surface')
            if surf and self.scene.has_geometry(surf['name']):
                self.scene.show_geometry(surf['name'], show)

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
        for name in self.noise_surface_names:
            self.scene.show_geometry(name, show)

    def _update_point_cloud_scene(self):
        """Remove and re-add the point cloud geometry to reflect color changes."""
        self.scene.remove_geometry("point_cloud")
        self.add_geometry("point_cloud", self.point_cloud, Materials.point_cloud_material)
        # Keep point cloud hidden if regions have been generated
        if self.region_names:
            self.scene.show_geometry('point_cloud', False)

    def show_regions(self, show: bool):
        """Back-compat shim: selects the SOLID region-surface mode.

        Region surface coloring is now an exclusive choice (see
        ``set_region_surface_mode``); this keeps the legacy "Show Regions"
        menu item / ROS parameter driving the solid-color view.
        """
        self.show_regions_flag = show
        if show:
            self.show_point_cloud(False)
            self.show_curvatures(False)
            self.show_noise_points(False)
            self.show_clusters_flag = False
            self.set_region_surface_mode(RegionSurfaceMode.SOLID)

    def show_fov_clusters(self, show: bool):
        """Back-compat shim: selects the CLUSTER region-surface mode."""
        self.show_clusters_flag = show
        if show:
            self.show_point_cloud(False)
            self.show_curvatures(False)
            self.show_regions_flag = False
            self.set_region_surface_mode(RegionSurfaceMode.CLUSTER)

    def show_viewpoints(self, show: bool):
        """Master toggle for all viewpoint overlays."""
        self.show_viewpoints_flag = show
        if show:
            # Build any enabled overlay kinds not yet realized (e.g. overlays
            # were enabled while the master toggle was off, or off at load).
            for kind in self._enabled_overlays | self._selected_overlays:
                if kind not in self._built_overlays:
                    self._build_overlay_kind(kind)
        self._update_overlay_visibility()

    def _show_region_scoped(self, suffixes, show: bool):
        """Show/hide per-region geometry named ``{region}_{suffix}``. With no
        region selected every region's geometry is affected; once a region is
        selected only that region's is shown (mirrors the selection scoping used
        for paths)."""
        for name in self.region_names:
            visible = show and (
                not self.selected_region_name or name == self.selected_region_name)
            for suffix in suffixes:
                self._safe_show(f"{name}_{suffix}", visible)

    def show_joint_path(self, show: bool):
        """Toggle the cartesian path the robot's end-effector follows (from the
        pre-computed joint trajectory), plus its waypoint markers."""
        self.show_joint_path_flag = show
        self._show_region_scoped(["joint_path", "joint_markers"], show)

    def show_unreachable(self, show: bool):
        """Toggle markers on viewpoints the arm could not plan a motion to."""
        self.show_unreachable_flag = show
        self._show_region_scoped(["unreachable_markers"], show)

    def show_blind_spots(self, show: bool):
        """Toggle markers on surface points no viewpoint can see with
        unoccluded line-of-sight (fov_clustering's greedy-cover safety
        break) — a genuine inspection coverage gap, distinct from
        show_unreachable's motion-planning failures."""
        self.show_blind_spots_flag = show
        self._show_region_scoped(["blind_spot_markers"], show)

    def _build_joint_path(self, region: dict, algorithm) -> tuple:
        order = region.get('order', {})
        if not isinstance(order, dict):
            return None, None
        algo_entry = (order.get(algorithm)
                      or (next(iter(order.values())) if order else None))
        if not isinstance(algo_entry, dict):
            return None, None
        jt = algo_entry.get('joint_trajectory')
        if not jt or not jt.get('cartesian_waypoints'):
            return None, None

        pts = np.array(jt['cartesian_waypoints']) * 1000.0  # m → mm
        if len(pts) < 2:
            return None, None

        path_ls = o3d.geometry.LineSet()
        path_ls.points = o3d.utility.Vector3dVector(pts)
        path_ls.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(pts) - 1)])

        step = max(1, len(pts) // 30)
        marker_pcd = o3d.geometry.PointCloud()
        marker_pcd.points = o3d.utility.Vector3dVector(pts[::step])
        marker_pcd.paint_uniform_color([1.0, 0.2, 0.2])

        return path_ls, marker_pcd

    def _build_unreachable_markers(self, region: dict, algorithm) -> o3d.geometry.PointCloud | None:
        """Build markers at the positions of viewpoints moveitpy could not
        plan a motion to/through for the given algorithm's path, or None if
        there are none (or no joint trajectory was computed)."""
        order = region.get('order', {})
        if not isinstance(order, dict):
            return None
        algo_entry = (order.get(algorithm)
                      or (next(iter(order.values())) if order else None))
        if not isinstance(algo_entry, dict):
            return None
        jt = algo_entry.get('joint_trajectory')
        if not jt or not jt.get('unreachable'):
            return None

        clusters = region.get('clusters', [])
        pts = []
        for cluster_id in jt['unreachable']:
            if not isinstance(cluster_id, int) or cluster_id < 0 or cluster_id >= len(clusters):
                continue
            viewpoint = clusters[cluster_id].get('viewpoint')
            if not viewpoint:
                continue
            pts.append(1000.0 * np.array(viewpoint['position']))
        if not pts:
            return None

        marker_pcd = o3d.geometry.PointCloud()
        marker_pcd.points = o3d.utility.Vector3dVector(np.array(pts))
        marker_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        return marker_pcd

    def set_traversal_algorithm(self, algorithm):
        """Switch the displayed traversal algorithm and rebuild each region's
        joint-path geometry in place (no full reload). The GUI calls this when
        the task_planning ``selected_traversal_algorithm`` parameter changes."""
        self.selected_traversal_algorithm = algorithm or None
        if not self.results_dict:
            return

        for mesh_idx, mesh_entry in enumerate(self.results_dict.get('meshes', [])):
            regions = mesh_entry.get('regions', [])
            region_order = mesh_entry.get('order', list(range(len(regions))))
            for region_id in region_order:
                if not isinstance(region_id, int) or region_id < 0 or region_id >= len(regions):
                    continue
                region_name = f"mesh_{mesh_idx}_region_{region_id}"
                if region_name not in self.geometries_dict:
                    continue

                joint_path, joint_markers = self._build_joint_path(
                    regions[region_id], self.selected_traversal_algorithm)
                self.scene.remove_geometry(f"{region_name}_joint_path")
                self.scene.remove_geometry(f"{region_name}_joint_markers")
                self.geometries_dict[region_name]['joint_path'] = joint_path
                self.geometries_dict[region_name]['joint_markers'] = joint_markers
                if joint_path is not None:
                    self.add_geometry(f"{region_name}_joint_path", joint_path,
                                      Materials.joint_path_material)
                if joint_markers is not None:
                    self.add_geometry(f"{region_name}_joint_markers", joint_markers,
                                      Materials.joint_marker_material)

                unreachable_markers = self._build_unreachable_markers(
                    regions[region_id], self.selected_traversal_algorithm)
                self.scene.remove_geometry(f"{region_name}_unreachable_markers")
                self.geometries_dict[region_name]['unreachable_markers'] = unreachable_markers
                if unreachable_markers is not None:
                    self.add_geometry(f"{region_name}_unreachable_markers", unreachable_markers,
                                      Materials.unreachable_marker_material)

        # Re-apply each traversal-overlay toggle to the rebuilt geometry.
        self.show_joint_path(self.show_joint_path_flag)
        self.show_unreachable(self.show_unreachable_flag)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _region_names_for_mesh(self, mesh_idx: int) -> list[str]:
        prefix = f"mesh_{mesh_idx}_"
        return [n for n in self.region_names if n.startswith(prefix)]

    def _cluster_names_for_region(self, region_name: str) -> list[str]:
        prefix = f"{region_name}_cluster_"
        return [n for n in self.cluster_names if n.startswith(prefix)]

    def _live_cluster_order(self, region_name: str) -> list:
        """Cluster ids for a region in the *live* traversal-algorithm order.

        The cluster geometry is numbered by the frozen load-time order (so the
        file tree stays stable), but the task_planning selected_viewpoint slider
        walks the currently-selected algorithm's path. Resolve that live order
        from the results dict so selection can follow it.
        """
        if not self.results_dict:
            return []
        parts = region_name.split('_')
        try:
            mesh_idx, region_id = int(parts[1]), int(parts[3])
        except (IndexError, ValueError):
            return []
        meshes = self.results_dict.get('meshes', [])
        if mesh_idx >= len(meshes):
            return []
        regions = meshes[mesh_idx].get('regions', [])
        if region_id >= len(regions):
            return []
        return _resolve_order_indices(
            regions[region_id].get('order', []), self.selected_traversal_algorithm)

    def _region_cluster_names_in_order(self, region_name: str) -> list[str]:
        """Cluster geometry names for a region ordered by the live traversal
        algorithm (so index k is the k-th stop on the drawn path, matching
        task_planning). Falls back to the frozen load order when unavailable."""
        frozen_names = self._cluster_names_for_region(region_name)
        live_order = self._live_cluster_order(region_name)
        if not live_order:
            return frozen_names
        clusters = self.geometries_dict.get(region_name, {}).get('clusters', {})
        id_to_name = {data.get('cluster_id'): name
                      for name, data in clusters.items()}
        ordered = [id_to_name[cid] for cid in live_order if cid in id_to_name]
        # Include any clusters the live order didn't cover, preserving them.
        for name in frozen_names:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _get_cluster_data(self, cluster_name: str) -> dict | None:
        for region_data in self.geometries_dict.values():
            clusters = region_data.get('clusters', {})
            if cluster_name in clusters:
                return clusters[cluster_name]
        return None

    def select_mesh(self, mesh_idx: int) -> bool:
        """Record the selected mesh and refresh per-region surfaces.

        With per-region surfaces, mesh-level emphasis is handled by region
        selection; this simply tracks the index and repaints.
        """
        self.selected_mesh_idx = mesh_idx
        self._render_surfaces()
        return True

    def select_region(self, region_idx: int) -> bool:
        """Select a region: make others semi-transparent, color the selected
        region's surface by the active mode, restrict paths/overlays to it, and
        auto-select its first viewpoint."""
        if region_idx < 0 or region_idx >= len(self.region_names):
            return False
        selected_region_name = self.region_names[region_idx]
        self.selected_region_name = selected_region_name

        # Surfaces: selected opaque/full-bright, all others dimmed/transparent.
        self._render_surfaces()

        # Cartesian path and unreachable markers: only the selected region's,
        # each gated by its own toggle.
        for region_name in self.region_names:
            is_selected = (region_name == selected_region_name)
            self._safe_show(f"{region_name}_joint_path",
                            is_selected and self.show_joint_path_flag)
            self._safe_show(f"{region_name}_joint_markers",
                            is_selected and self.show_joint_path_flag)
            self._safe_show(f"{region_name}_unreachable_markers",
                            is_selected and self.show_unreachable_flag)
            self._safe_show(f"{region_name}_blind_spot_markers",
                            is_selected and self.show_blind_spots_flag)

        # Overlays: every enabled kind, for the selected region only.
        self._update_overlay_visibility()

        # Auto-select viewpoint 0 of this region.
        self.select_cluster(0)
        return True

    def _safe_show(self, name: str, visible: bool):
        if self.scene.has_geometry(name):
            self.scene.show_geometry(name, visible)

    def select_cluster(self, cluster_idx: int) -> bool:
        """Select a viewpoint (cluster) within the active region: restore the
        previously-selected viewpoint's overlays and highlight the new one."""
        # Scope the index to the selected region's clusters, ordered by the live
        # traversal algorithm so index k highlights the k-th viewpoint on the
        # drawn path (matching the task_planning selected_viewpoint slider),
        # rather than the frozen load-time cluster numbering.
        if self.selected_region_name and self.selected_region_name in self.geometries_dict:
            region_clusters = self._region_cluster_names_in_order(self.selected_region_name)
        else:
            region_clusters = self.cluster_names
        if cluster_idx < 0 or cluster_idx >= len(region_clusters):
            return False

        selected_cluster_name = region_clusters[cluster_idx]
        self.selected_cluster_name = selected_cluster_name

        # select_cluster can be driven independently by the task_planning
        # navigation.selected_viewpoint parameter without a matching
        # select_region. Recover the owning region and, if it changed, refresh
        # surfaces so the selection is consistent.
        owning_region = selected_cluster_name.rsplit('_cluster_', 1)[0]
        if owning_region in self.geometries_dict and owning_region != self.selected_region_name:
            self.selected_region_name = owning_region
            self._render_surfaces()

        # Update overlay visibility + highlight materials for the new selection
        # (the selected viewpoint uses its own overlay set and highlight colors).
        self._update_overlay_visibility()
        return True

    @property
    def selected_region_index(self) -> int:
        """Index of the currently selected region within ``region_names``.

        Returns -1 when no region is selected."""
        if self.selected_region_name in self.region_names:
            return self.region_names.index(self.selected_region_name)
        return -1

    @property
    def selected_cluster_index(self) -> int:
        """Index of the currently selected cluster, relative to its region.

        Mirrors how ``select_cluster`` scopes cluster indices to the selected
        region and orders them by the live traversal algorithm. Returns -1 when
        no cluster is selected."""
        if not self.selected_cluster_name:
            return -1
        if self.selected_region_name:
            region_clusters = self._region_cluster_names_in_order(self.selected_region_name)
        else:
            region_clusters = self.cluster_names
        if self.selected_cluster_name in region_clusters:
            return region_clusters.index(self.selected_cluster_name)
        return -1

    # ── File-tree contents ──────────────────────────────────────────────────

    @staticmethod
    def _tree_node(label: str, children: list | None = None,
                   select: dict | None = None, collapsed: bool = False) -> dict:
        """A single file-tree node.

        ``label``    – display text.
        ``children`` – list of child nodes.
        ``select``   – optional action the GUI runs when this node is selected,
                       e.g. ``{'type': 'region', 'region': i}`` or
                       ``{'type': 'cluster', 'region': i, 'cluster': j}``. The
                       indices match ``select_region`` / ``select_cluster``.
        ``collapsed``– hint that this node should start closed. Open3D's
                       TreeView has no collapse API and auto-expands everything,
                       so the GUI honors this by deferring the node's children
                       until it is clicked (lazy population).
        """
        return {
            'label': label,
            'children': children if children is not None else [],
            'select': select,
            'collapsed': collapsed,
        }

    @staticmethod
    def _fmt_vec(vec, precision: int = 4) -> str:
        """Format a numeric vector for display, or fall back to ``str``."""
        if vec is None:
            return ''
        try:
            return "[" + ", ".join(f"{float(v):.{precision}f}" for v in vec) + "]"
        except (TypeError, ValueError):
            return str(vec)

    def _viewpoint_node(self, viewpoint: dict | None) -> dict | None:
        """Build a 'Viewpoint' node from a cluster's viewpoint dict, or None."""
        if not viewpoint:
            return None
        return self._tree_node("Viewpoint", [
            self._tree_node(f"Origin: {self._fmt_vec(viewpoint.get('origin'))}"),
            self._tree_node(f"Position: {self._fmt_vec(viewpoint.get('position'))}"),
            self._tree_node(f"Direction: {self._fmt_vec(viewpoint.get('direction'))}"),
            self._tree_node(f"Orientation: {self._fmt_vec(viewpoint.get('orientation'))}"),
        ])

    def get_file_tree_contents(self) -> list[dict]:
        """Translate the loaded results into a render-agnostic tree.

        Returns a list of top-level nodes, each a dict::

            {'label': <str>, 'children': [<node>, ...], 'select': <dict|None>}

        Regions are expanded into numbered, selectable dropdowns; each region
        holds a 'Clusters' dropdown (numbered, selectable cluster nodes that
        carry viewpoint info) and a 'Paths' dropdown (the traversal order).
        Selecting a region/cluster node drives ``select_region`` /
        ``select_cluster`` via the indices in each node's ``select`` action.

        Point-index lists (region/cluster ``points``) are intentionally omitted.
        The GUI only renders this structure; the results→text mapping lives
        here. Returns an empty list when no results are loaded.
        """
        nodes: list[dict] = []

        if not self.results_dict:
            return nodes

        # Order/number clusters by the *live* traversal algorithm so the tree,
        # the task_planning selected_viewpoint slider, and select_cluster all
        # agree on what "viewpoint k" means (the k-th stop on the drawn path).
        # The tree rebuilds when the algorithm changes, which is expected.
        cluster_order_algorithm = self.selected_traversal_algorithm

        # ── Meshes ────────────────────────────────────────────────────────
        mesh_children = []
        for mesh_idx, mesh_entry in enumerate(self.results_dict.get('meshes', [])):
            mesh_children.append(self._tree_node(str(mesh_idx), [
                self._tree_node(f"File: {mesh_entry.get('file', '')}"),
                self._tree_node(f"Units: {mesh_entry.get('units', '')}"),
                self._tree_node(f"Material: {mesh_entry.get('material', '')}"),
                self._tree_node(f"Dimensions: {mesh_entry.get('dimensions', '')}"),
                self._tree_node(f"Surface Area: {mesh_entry.get('surface_area', '')}"),
                self._build_regions_node(mesh_idx, mesh_entry, cluster_order_algorithm),
            ]))

        nodes.append(self._tree_node("Meshes", mesh_children))
        return nodes

    def _build_regions_node(self, mesh_idx: int, mesh_entry: dict,
                            cluster_order_algorithm) -> dict:
        """Build the 'Regions' dropdown for a mesh: numbered, selectable region
        nodes each containing a 'Clusters' and a 'Paths' dropdown.

        ``cluster_order_algorithm`` orders/numbers the clusters. It is the live
        traversal algorithm so tree/slider/selection indices all mean the same
        viewpoint (the k-th stop on the drawn path)."""
        regions = mesh_entry.get('regions', [])
        region_order = mesh_entry.get('order', list(range(len(regions))))

        region_nodes = []
        for region_id in region_order:
            if not isinstance(region_id, int) or region_id < 0 or region_id >= len(regions):
                continue
            region = regions[region_id]

            # Map to the visualizer's flat region index so selection lines up
            # with select_region. Non-built regions are shown but not selectable.
            region_name = f"mesh_{mesh_idx}_region_{region_id}"
            region_sel_index = (self.region_names.index(region_name)
                                if region_name in self.region_names else None)
            region_select = ({'type': 'region', 'region': region_sel_index}
                             if region_sel_index is not None else None)

            pcd = region.get('point_cloud', {})
            pcd_node = self._tree_node("Point Cloud", [
                self._tree_node(f"File: {pcd.get('file', '')}"),
                self._tree_node(f"Units: {pcd.get('units', '')}"),
                self._tree_node(f"Points: {pcd.get('points', 0)}"),
            ])

            # Clusters (in traversal order) — numbered, selectable, with viewpoint
            cluster_order = _resolve_order_indices(
                region.get('order', []), cluster_order_algorithm)
            cluster_nodes = []
            clusters = region.get('clusters', [])
            for i, cluster_id in enumerate(cluster_order):
                if not isinstance(cluster_id, int) or cluster_id < 0 or cluster_id >= len(clusters):
                    continue
                cluster = clusters[cluster_id]
                cluster_children = []
                vp_node = self._viewpoint_node(cluster.get('viewpoint'))
                if vp_node is not None:
                    cluster_children.append(vp_node)
                cluster_select = ({'type': 'cluster',
                                   'region': region_sel_index, 'cluster': i}
                                  if region_sel_index is not None else None)
                cluster_nodes.append(self._tree_node(
                    f"Cluster {i}", cluster_children, select=cluster_select))
            # Start the Clusters dropdown closed (lazily populated by the GUI).
            clusters_node = self._tree_node("Clusters", cluster_nodes, collapsed=True)

            # Paths — one selectable node per optimized algorithm. Clicking it
            # sets the traversal algorithm (drives the drawn path and execution
            # order). The 'order' index list is omitted; every other metadata
            # key (e.g. 'distance') is shown under the algorithm node.
            order = region.get('order', [])
            path_children = []
            if isinstance(order, dict):
                for algo, info in order.items():
                    algo_children = []
                    if isinstance(info, dict):
                        for key, value in info.items():
                            if key in ('order', 'joint_trajectory'):
                                continue
                            if isinstance(value, float):
                                value = f"{value:.4f}"
                            algo_children.append(
                                self._tree_node(f"{key}: {value}"))
                        jt = info.get('joint_trajectory')
                        if isinstance(jt, dict):
                            for jt_key, jt_val in jt.items():
                                if jt_key == 'cartesian_waypoints':
                                    continue
                                if isinstance(jt_val, float):
                                    jt_val = f"{jt_val:.4f}"
                                algo_children.append(
                                    self._tree_node(f"{jt_key}: {jt_val}"))
                    path_children.append(self._tree_node(
                        algo, algo_children,
                        select={'type': 'algorithm', 'algorithm': algo}))
            elif order:
                path_children.append(self._tree_node(f"Order: {list(order)}"))
            paths_node = self._tree_node("Paths", path_children)

            region_nodes.append(self._tree_node(
                f"Region {region_id}", [pcd_node, clusters_node, paths_node],
                select=region_select))

        return self._tree_node("Regions", region_nodes)

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
