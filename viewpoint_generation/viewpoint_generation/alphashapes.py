import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import alphashape


def delaunay_to_o3d_mesh(pcd):
    """
    Convert Open3D PointCloud to TriangleMesh using Delaunay triangulation

    Args:
        pcd: open3d.geometry.PointCloud

    Returns:
        open3d.geometry.TriangleMesh
    """
    # Extract points from Open3D PointCloud
    points = np.asarray(pcd.points)

    # Compute Delaunay triangulation
    delaunay = Delaunay(points)

    # Extract surface triangles from tetrahedra
    # Each tetrahedron has 4 faces (triangles)
    triangles = []
    triangle_set = set()

    for simplex in delaunay.simplices:
        # Each simplex is a tetrahedron with 4 vertices
        # Create the 4 triangular faces
        faces = [
            tuple(sorted([simplex[0], simplex[1], simplex[2]])),
            tuple(sorted([simplex[0], simplex[1], simplex[3]])),
            tuple(sorted([simplex[0], simplex[2], simplex[3]])),
            tuple(sorted([simplex[1], simplex[2], simplex[3]]))
        ]

        for face in faces:
            # Keep track of face occurrences
            if face in triangle_set:
                # Interior face (shared by two tetrahedra) - remove it
                triangle_set.remove(face)
            else:
                # Boundary face (only in one tetrahedron) - keep it
                triangle_set.add(face)

    # Convert to list of triangles
    triangles = np.array(list(triangle_set))

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Copy colors if they exist in the point cloud
    if pcd.has_colors():
        mesh.vertex_colors = pcd.colors

    # Copy normals if they exist in the point cloud
    if pcd.has_normals():
        mesh.vertex_normals = pcd.normals

    return mesh


def alphashape_to_o3d_mesh(pcd, alpha=None, fallback_to_convex_hull=True):
    """
    Convert Open3D PointCloud to TriangleMesh using alpha shapes

    Args:
        pcd: open3d.geometry.PointCloud
        alpha: Alpha value for the alpha shape (higher = more convex, lower = more detailed)
               If None, uses automatic alpha value
               Typical values: 0.1 to 10.0
        fallback_to_convex_hull: If True, fall back to convex hull on failure

    Returns:
        open3d.geometry.TriangleMesh or None if computation fails
    """
    # Extract points from Open3D PointCloud
    points = np.asarray(pcd.points)

    # Check if we have enough points
    if len(points) < 4:
        print(
            f"Warning: Not enough points ({len(points)}) for alpha shape. Need at least 4.")
        if fallback_to_convex_hull and len(points) >= 3:
            return pcd.compute_convex_hull()[0]
        return None

    try:
        # Compute alpha shape
        alpha_mesh = alphashape.alphashape(points, alpha)

        # Check if result is valid
        if alpha_mesh is None:
            raise ValueError("alphashape returned None")

        # Check if it's a valid 3D mesh
        if not hasattr(alpha_mesh, 'vertices') or not hasattr(alpha_mesh, 'faces'):
            raise ValueError("alphashape did not return a valid mesh")

        # Check if faces array is 2D
        if alpha_mesh.faces.ndim != 2:
            raise ValueError(
                f"Invalid faces array dimension: {alpha_mesh.faces.ndim}")

        if len(alpha_mesh.faces) == 0:
            raise ValueError("alphashape returned mesh with no faces")

        # Convert to Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(alpha_mesh.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(alpha_mesh.faces)

        # Copy colors if they exist in the point cloud
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            if len(colors) == len(alpha_mesh.vertices):
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        return mesh

    except (ValueError, IndexError, AttributeError) as e:
        print(f"Warning: Alpha shape computation failed: {e}")

        # Try with a fixed alpha value if auto-optimization failed
        if alpha is None:
            print("Retrying with fixed alpha=2.0...")
            try:
                alpha_mesh = alphashape.alphashape(points, 2.0)

                if alpha_mesh and hasattr(alpha_mesh, 'vertices') and hasattr(alpha_mesh, 'faces'):
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(
                        alpha_mesh.vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(
                        alpha_mesh.faces)
                    return mesh
            except Exception as e2:
                print(f"Retry also failed: {e2}")

        # Fall back to convex hull if requested
        if fallback_to_convex_hull:
            print("Falling back to convex hull...")
            try:
                hull, _ = pcd.compute_convex_hull(joggle_inputs=True)

                return hull
            except Exception as e3:
                print(f"Convex hull fallback also failed: {e3}")
                return None

        return None


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
