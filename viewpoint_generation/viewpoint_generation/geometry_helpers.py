import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay


def filter_large_triangles(mesh, method='iqr', threshold_multiplier=1.5):
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
