"""Small shared triangle-mesh helpers used by both segmentation/orchestration
and visualization code."""

import os

import numpy as np
import open3d as o3d

# Tessellated STEP meshes, keyed by (path, mtime, size). Tessellation is far
# more expensive than reading an STL, and the visualizer re-reads the model file
# every time the results change, so a large assembly would otherwise be
# re-tessellated on every recompute. Entries are handed out as copies, since
# callers scale the mesh they receive.
_STEP_CACHE = {}
_STEP_CACHE_LIMIT = 4

STEP_EXTENSIONS = ('.stp', '.step')


def read_mesh_file(file_path: str):
    """Read any supported model file into an Open3D mesh, in the file's own units.

    Open3D reads tessellated formats (STL/OBJ/PLY/OFF) but cannot parse STEP,
    which it fails on silently by returning an empty mesh -- so callers that go
    straight to `o3d.io.read_triangle_mesh` simply display nothing for a
    STEP-loaded part. This dispatches on the extension and tessellates STEP
    through PythonOCC instead.

    The returned geometry is *unscaled*, matching what `read_triangle_mesh`
    would give: the caller applies its own unit conversion. (The STEP loader is
    asked for metres purely because that is its identity scale factor.)

    Args:
        file_path: Path to an .stl/.obj/.ply/.off/.stp/.step file.

    Returns:
        tuple: (mesh, error). mesh is None on failure, with error explaining why.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension in STEP_EXTENSIONS:
        try:
            stat = os.stat(file_path)
        except OSError as e:
            return None, f'Could not stat STEP file {file_path}: {e}'
        key = (os.path.abspath(file_path), stat.st_mtime, stat.st_size)
        cached = _STEP_CACHE.get(key)
        if cached is None:
            from viewpoint_generation.step_loader import load_step_mesh
            mesh, error = load_step_mesh(file_path, units='m')
            if mesh is None:
                return None, error
            if len(_STEP_CACHE) >= _STEP_CACHE_LIMIT:
                _STEP_CACHE.pop(next(iter(_STEP_CACHE)))
            _STEP_CACHE[key] = mesh
            cached = mesh
        # A copy, because callers scale what they are handed and would
        # otherwise compound that scaling into the cached mesh.
        return o3d.geometry.TriangleMesh(cached), ''

    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
    except Exception as e:
        return None, f'Could not load mesh file {file_path}: {e}'
    if mesh.is_empty():
        return None, f'Mesh file {file_path} is empty or invalid.'
    return mesh, ''


def submesh_from_faces(mesh: o3d.geometry.TriangleMesh, tris: np.ndarray):
    """Build a standalone submesh from a subset of a mesh's triangles,
    remapping vertex indices. Returns (submesh, used_vertex_indices)."""
    verts = np.asarray(mesh.vertices)
    used = np.unique(tris)
    remap = np.full(len(verts), -1, dtype=int)
    remap[used] = np.arange(len(used))
    sub = o3d.geometry.TriangleMesh()
    sub.vertices = o3d.utility.Vector3dVector(verts[used])
    sub.triangles = o3d.utility.Vector3iVector(remap[tris])
    sub.compute_vertex_normals()
    return sub, used
