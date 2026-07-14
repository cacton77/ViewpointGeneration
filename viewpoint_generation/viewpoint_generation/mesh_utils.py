"""Small shared triangle-mesh helpers used by both segmentation/orchestration
and visualization code."""

import numpy as np
import open3d as o3d


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
