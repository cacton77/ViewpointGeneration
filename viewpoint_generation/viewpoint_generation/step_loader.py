"""STEP file loading via PythonOCC/OpenCascade.

Produces an Open3D `TriangleMesh` that satisfies exactly the same contract as
the STL/OBJ load path -- vertices, triangles, vertex normals, uniform color --
so every downstream stage (region growing, FOV clustering, viewpoint
projection, raycasting, results serialization) works unchanged. Alongside the
mesh it returns the B-rep topology that makes STEP worth ingesting natively:
which triangles came from which analytical face, what kind of surface each
face is, and which faces share an edge.

Two details in here are load-bearing and easy to get wrong:

**Vertex merging.** OpenCascade tessellates every B-rep face independently,
each with its own vertex pool, so two faces meeting at an edge produce
coincident-but-distinct vertices. `RegionGrowing` derives face adjacency from
*shared vertex indices*, so without a merge pass it sees a mesh that falls
apart at every B-rep boundary and can never grow a region across one. The mesh
is therefore welded with `merge_close_vertices`, and the triangle-to-face
mapping is rebuilt afterwards against the surviving triangles.

**Triangle survival.** Merging can also drop degenerate triangles, which
renumbers the triangle array. `tri_to_brep` and every `BRepFace.triangle_indices`
are recomputed from the post-merge geometry rather than carried over, so the
indices in `StepLoadResult` always address the mesh actually returned.

The OCP (`cadquery-ocp`) bindings are imported lazily by `load_step()` so that
importing this module -- which `viewpoint_generation.py` does unconditionally --
never hard-depends on OpenCascade being installed.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

# Distance under which tessellation vertices are welded. OCC emits coincident
# vertices at shared B-rep edges bit-for-bit identically often enough that a
# very tight threshold suffices; 0.01 um is far below any real feature size on
# an inspectable part, so it can never weld genuinely distinct geometry.
MERGE_TOLERANCE = 1e-8

_UNIT_SCALE = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'in': 0.0254}


@dataclass
class TessellationConfig:
    """Controls the quality of the B-rep to triangle conversion.

    Defaults target macro-scale inspection: the chord error is far below a
    camera FOV, while keeping triangle counts manageable. Tighten both for
    CAD-vs-scan comparison, where the mesh is the reference geometry.
    """

    # Maximum chord deviation from the true surface. Interpreted in model
    # units when `relative` is False, and as a fraction of each edge's size
    # when True.
    linear_deflection: float = 0.0005
    # Maximum angular deviation between adjacent facet normals (radians).
    angular_deflection: float = 0.3
    # Scale the linear deflection with the size of each shape.
    relative: bool = False

    def to_dict(self):
        return {
            "linear_deflection": {
                "value": self.linear_deflection,
                "type": "float",
                "description": "Maximum chord deviation of the tessellation from the true B-rep surface (model units)",
                "control": "slider",
                "range": [0.00001, 0.01],
            },
            "angular_deflection": {
                "value": self.angular_deflection,
                "type": "float",
                "description": "Maximum angular deviation between adjacent facets (radians)",
                "control": "slider",
                "range": [0.01, 1.5],
            },
            "relative": {
                "value": self.relative,
                "type": "boolean",
                "description": "Scale linear deflection with shape size instead of using absolute model units",
                "control": "toggle",
            },
        }


@dataclass
class BRepFace:
    """Metadata for a single B-rep face."""

    face_id: int
    surface_type: str            # plane, cylinder, cone, sphere, torus, bspline, ...
    triangle_indices: list       # indices into the mesh's triangle array
    area_m2: float
    centroid: np.ndarray         # face centroid, model coordinates (meters)
    normal_at_centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(3))   # exact outward normal, not tessellated


@dataclass
class StepLoadResult:
    """Result of loading a STEP file."""

    mesh: o3d.geometry.TriangleMesh      # tessellated mesh (same contract as an STL load)
    brep_faces: list                     # one BRepFace per tessellated B-rep face
    face_adjacency: dict                 # face_id -> set of edge-sharing face_ids
    tri_to_brep: np.ndarray              # triangle index -> B-rep face id
    source_file: str
    units: str

    def face_count(self):
        """Number of B-rep faces that produced geometry."""
        return len(self.brep_faces)

    def surface_type_counts(self):
        """How many faces of each analytical surface type the part has."""
        counts = {}
        for face in self.brep_faces:
            counts[face.surface_type] = counts.get(face.surface_type, 0) + 1
        return counts


def _unit_scale(units):
    """The factor converting the given units to meters."""
    return _UNIT_SCALE.get(units, 1.0)


def _classify_surface(geom_type, geom_abs):
    """Map an OCC `GeomAbs_SurfaceType` onto a readable string.

    Args:
        geom_type: The value returned by `BRepAdaptor_Surface.GetType()`.
        geom_abs: The imported `OCP.GeomAbs` module, whose enum members are
            compared against.

    Returns:
        str: One of plane, cylinder, cone, sphere, torus, bezier, bspline,
        revolution, extrusion, offset, or other.
    """
    mapping = {
        geom_abs.GeomAbs_Plane: 'plane',
        geom_abs.GeomAbs_Cylinder: 'cylinder',
        geom_abs.GeomAbs_Cone: 'cone',
        geom_abs.GeomAbs_Sphere: 'sphere',
        geom_abs.GeomAbs_Torus: 'torus',
        geom_abs.GeomAbs_BezierSurface: 'bezier',
        geom_abs.GeomAbs_BSplineSurface: 'bspline',
        geom_abs.GeomAbs_SurfaceOfRevolution: 'revolution',
        geom_abs.GeomAbs_SurfaceOfExtrusion: 'extrusion',
        geom_abs.GeomAbs_OffsetSurface: 'offset',
    }
    return mapping.get(geom_type, 'other')


def _import_occ():
    """Import the OCP modules `load_step()` needs.

    Kept out of module scope so that importing `step_loader` (which the core
    pipeline does unconditionally) does not require OpenCascade to be present.

    Returns:
        dict: The imported OCC symbols.

    Raises:
        ImportError: With installation guidance when OCP is unavailable.
    """
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GProp import GProp_GProps
        from OCP import GeomAbs
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
        from OCP.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,
                                  TopTools_IndexedMapOfShape)
    except ImportError as e:
        raise ImportError(
            'STEP loading requires the OpenCascade Python bindings (OCP). '
            'Install them with `pip install cadquery-ocp --break-system-packages` '
            'inside the container, or `conda install -c conda-forge cadquery-ocp`. '
            f'Original error: {e}') from e
    return {
        'BRep_Tool': BRep_Tool,
        'BRepAdaptor_Surface': BRepAdaptor_Surface,
        'BRepGProp': BRepGProp,
        'BRepMesh_IncrementalMesh': BRepMesh_IncrementalMesh,
        'GProp_GProps': GProp_GProps,
        'GeomAbs': GeomAbs,
        'IFSelect_RetDone': IFSelect_RetDone,
        'STEPControl_Reader': STEPControl_Reader,
        'TopAbs_EDGE': TopAbs_EDGE,
        'TopAbs_FACE': TopAbs_FACE,
        'TopAbs_REVERSED': TopAbs_REVERSED,
        'TopExp': TopExp,
        'TopExp_Explorer': TopExp_Explorer,
        'TopLoc_Location': TopLoc_Location,
        'TopoDS': TopoDS,
        'TopTools_IndexedDataMapOfShapeListOfShape':
            TopTools_IndexedDataMapOfShapeListOfShape,
        'TopTools_IndexedMapOfShape': TopTools_IndexedMapOfShape,
    }


def _build_brep_adjacency(shape, occ, shape_map, index_to_face_id):
    """Build face-to-face adjacency from shared B-rep edges.

    Two faces are adjacent when they share at least one edge. This is exact
    topology from the CAD model -- no distance thresholds, no vertex welding,
    and no dependence on tessellation quality.

    Face identity comes from OCC's own `TopTools_IndexedMapOfShape`, whose
    lookup uses the shape-equality semantics of the kernel. Python-level
    identity does not work here: OCP hands back a fresh wrapper object on every
    access, so `id()` (and `TShape()`'s id) differs between two handles to the
    same face and every lookup misses.

    Args:
        shape: The `TopoDS_Shape` read from the STEP file.
        occ: The imported OCC symbols.
        shape_map: An indexed map of every face in `shape`.
        index_to_face_id: Mapping of shape-map index -> our face_id, covering
            only the faces that produced geometry.

    Returns:
        dict: face_id -> set of adjacent face_ids.
    """
    edge_face_map = occ['TopTools_IndexedDataMapOfShapeListOfShape']()
    occ['TopExp'].MapShapesAndAncestors_s(
        shape, occ['TopAbs_EDGE'], occ['TopAbs_FACE'], edge_face_map)

    adjacency = {face_id: set() for face_id in index_to_face_id.values()}

    for index in range(1, edge_face_map.Extent() + 1):
        # OCP exposes TopTools_ListOfShape as a Python iterable, which spares
        # us the C++ begin/end iterator dance (whose bindings differ between
        # OCC releases).
        face_ids = []
        for face in edge_face_map.FindFromIndex(index):
            map_index = shape_map.FindIndex(face)
            face_id = index_to_face_id.get(map_index)
            if face_id is not None:
                face_ids.append(face_id)
        for first in face_ids:
            for second in face_ids:
                if first != second:
                    adjacency[first].add(second)
    return adjacency


def load_step(filepath, units='m', tess_config=None):
    """Load a STEP file, tessellate its B-rep, and preserve face topology.

    Args:
        filepath: Path to the .stp/.step file.
        units: Units the STEP file is authored in ('m', 'mm', 'cm', 'in').
        tess_config: A `TessellationConfig`; defaults are used when omitted.

    Returns:
        StepLoadResult: The tessellated mesh plus its B-rep metadata, with all
        lengths converted to meters.

    Raises:
        ImportError: OpenCascade (OCP) is not installed.
        IOError: The file could not be read as STEP.
        ValueError: The file contained no tessellatable faces.
    """
    occ = _import_occ()
    if tess_config is None:
        tess_config = TessellationConfig()
    filepath = str(filepath)

    # --- 1. Read the STEP file ---
    reader = occ['STEPControl_Reader']()
    status = reader.ReadFile(filepath)
    if status != occ['IFSelect_RetDone']:
        raise IOError(f'Failed to read STEP file: {filepath} (status={status})')
    reader.TransferRoots()
    shape = reader.OneShape()

    # --- 2. Tessellate the whole shape ---
    occ['BRepMesh_IncrementalMesh'](
        shape,
        tess_config.linear_deflection,
        tess_config.relative,
        tess_config.angular_deflection,
        True,   # parallel
    )

    # --- 3. Walk the faces, collecting triangles and per-face metadata ---
    # An indexed map of every face gives each one a kernel-stable identity, so
    # the adjacency pass below can recognize the same face again.
    shape_map = occ['TopTools_IndexedMapOfShape']()
    occ['TopExp'].MapShapes_s(shape, occ['TopAbs_FACE'], shape_map)

    all_vertices = []
    all_triangles = []
    face_records = []
    index_to_face_id = {}
    vertex_offset = 0
    face_id = 0

    explorer = occ['TopExp_Explorer'](shape, occ['TopAbs_FACE'])
    while explorer.More():
        face = occ['TopoDS'].Face_s(explorer.Current())
        location = occ['TopLoc_Location']()
        triangulation = occ['BRep_Tool'].Triangulation_s(face, location)
        if triangulation is None:
            explorer.Next()
            continue

        node_count = triangulation.NbNodes()
        face_vertices = np.zeros((node_count, 3))
        transformation = location.Transformation()
        for node in range(1, node_count + 1):
            point = triangulation.Node(node)
            if not location.IsIdentity():
                point = point.Transformed(transformation)
            face_vertices[node - 1] = [point.X(), point.Y(), point.Z()]

        triangle_count = triangulation.NbTriangles()
        face_triangles = np.zeros((triangle_count, 3), dtype=np.int64)
        # A face flagged REVERSED stores its triangles wound against the
        # face's true outward normal; swapping two corners restores the
        # outward-facing winding the whole mesh needs for consistent normals.
        reversed_face = face.Orientation() == occ['TopAbs_REVERSED']
        for index in range(1, triangle_count + 1):
            first, second, third = triangulation.Triangle(index).Get()
            if reversed_face:
                first, third = third, first
            face_triangles[index - 1] = [first - 1 + vertex_offset,
                                         second - 1 + vertex_offset,
                                         third - 1 + vertex_offset]

        adaptor = occ['BRepAdaptor_Surface'](face)
        surface_type = _classify_surface(adaptor.GetType(), occ['GeomAbs'])

        # Exact centroid and normal from the analytical surface, not the
        # tessellation: the mid-parameter point is on the true surface even
        # where the facets deviate from it.
        u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2.0
        v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2.0
        centroid = np.zeros(3)
        normal = np.zeros(3)
        try:
            point = adaptor.Value(u_mid, v_mid)
            centroid = np.array([point.X(), point.Y(), point.Z()])
            derivative_u = adaptor.DN(u_mid, v_mid, 1, 0)
            derivative_v = adaptor.DN(u_mid, v_mid, 0, 1)
            normal = np.cross(
                [derivative_u.X(), derivative_u.Y(), derivative_u.Z()],
                [derivative_v.X(), derivative_v.Y(), derivative_v.Z()])
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.zeros(3)
            if reversed_face:
                normal = -normal
        except Exception as e:  # noqa: BLE001 - OCC raises bare Standard_Failure
            logger.debug('Could not evaluate surface at mid-parameters for '
                         'face %d: %s', face_id, e)

        properties = occ['GProp_GProps']()
        occ['BRepGProp'].SurfaceProperties_s(face, properties)
        area = properties.Mass()

        face_records.append({
            'face_id': face_id,
            'surface_type': surface_type,
            'area': area,
            'centroid': centroid,
            'normal': normal,
            'triangle_count': triangle_count,
        })
        index_to_face_id[shape_map.FindIndex(face)] = face_id

        all_vertices.append(face_vertices)
        all_triangles.append(face_triangles)
        vertex_offset += node_count
        face_id += 1
        explorer.Next()

    if not all_vertices:
        raise ValueError(f'No tessellatable faces found in STEP file: {filepath}')

    # --- 4. Assemble the Open3D mesh ---
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)
    # Per-triangle face ownership, in the same order as `triangles`.
    tri_to_brep = np.concatenate([
        np.full(record['triangle_count'], record['face_id'], dtype=np.int64)
        for record in face_records])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    scale = _unit_scale(units)
    if scale != 1.0:
        mesh.scale(scale, center=(0, 0, 0))

    # --- 5. Weld the per-face vertex pools together ---
    # Without this, adjacent B-rep faces share no vertex indices and
    # RegionGrowing's edge-based adjacency cannot cross a face boundary.
    # Merging renumbers (and may drop degenerate) triangles, so the
    # triangle -> face mapping is rebuilt from the surviving triangles by
    # matching them back to their pre-merge vertex positions.
    pre_merge_centroids = _triangle_centroids(vertices * scale, triangles)
    mesh.merge_close_vertices(MERGE_TOLERANCE)
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    merged_vertices = np.asarray(mesh.vertices)
    merged_triangles = np.asarray(mesh.triangles)
    post_merge_centroids = _triangle_centroids(merged_vertices, merged_triangles)
    tri_to_brep = _remap_tri_to_brep(
        pre_merge_centroids, tri_to_brep, post_merge_centroids)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.5, 0.5, 0.5))

    # --- 6. Rebuild per-face triangle lists against the final mesh ---
    brep_faces = []
    for record in face_records:
        owned = np.flatnonzero(tri_to_brep == record['face_id'])
        brep_faces.append(BRepFace(
            face_id=record['face_id'],
            surface_type=record['surface_type'],
            triangle_indices=owned.tolist(),
            area_m2=record['area'] * scale * scale,
            centroid=record['centroid'] * scale,
            normal_at_centroid=record['normal'],
        ))

    face_adjacency = _build_brep_adjacency(shape, occ, shape_map, index_to_face_id)

    logger.info('Loaded STEP %s: %d vertices, %d triangles, %d B-rep faces (%s)',
                filepath, len(merged_vertices), len(merged_triangles),
                len(brep_faces),
                ', '.join(f'{count} {name}' for name, count
                          in sorted(_count_types(brep_faces).items())))

    return StepLoadResult(
        mesh=mesh,
        brep_faces=brep_faces,
        face_adjacency=face_adjacency,
        tri_to_brep=tri_to_brep,
        source_file=filepath,
        units=units,
    )


def _count_types(brep_faces):
    """Count B-rep faces by surface type."""
    counts = {}
    for face in brep_faces:
        counts[face.surface_type] = counts.get(face.surface_type, 0) + 1
    return counts


def _triangle_centroids(vertices, triangles):
    """Centroid of every triangle, as an (N, 3) array."""
    if len(triangles) == 0:
        return np.zeros((0, 3))
    return vertices[triangles].mean(axis=1)


def _remap_tri_to_brep(pre_centroids, pre_tri_to_brep, post_centroids):
    """Carry the triangle-to-face mapping across the vertex merge.

    Vertex welding renumbers triangles and can drop degenerate ones, so the
    pre-merge mapping no longer lines up with the surviving triangle array.
    Triangle centroids are invariant under welding (welded vertices are
    coincident to within MERGE_TOLERANCE), so each surviving triangle is
    matched to its pre-merge twin by nearest centroid.

    Args:
        pre_centroids: Centroids of the pre-merge triangles, in model units.
        pre_tri_to_brep: Face id of each pre-merge triangle.
        post_centroids: Centroids of the surviving triangles.

    Returns:
        np.ndarray: Face id of each surviving triangle.
    """
    if len(post_centroids) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(pre_centroids) == 0:
        return np.full(len(post_centroids), -1, dtype=np.int64)
    if len(post_centroids) == len(pre_centroids) and np.allclose(
            post_centroids, pre_centroids, atol=MERGE_TOLERANCE * 10):
        # Nothing was dropped or reordered -- the common case.
        return pre_tri_to_brep.copy()

    reference = o3d.geometry.PointCloud()
    reference.points = o3d.utility.Vector3dVector(pre_centroids)
    tree = o3d.geometry.KDTreeFlann(reference)

    remapped = np.full(len(post_centroids), -1, dtype=np.int64)
    for index, centroid in enumerate(post_centroids):
        found, indices, _ = tree.search_knn_vector_3d(centroid, 1)
        if found > 0:
            remapped[index] = pre_tri_to_brep[indices[0]]
    return remapped


def load_step_mesh(filepath, units='m', tess_config=None):
    """Load a STEP file and return only its Open3D mesh.

    Convenience wrapper for callers that want the geometry without the
    topology sidecar.

    Returns:
        tuple: (mesh, error) with mesh None on failure.
    """
    try:
        return load_step(filepath, units=units, tess_config=tess_config).mesh, ''
    except (ImportError, IOError, ValueError) as e:
        return None, str(e)
