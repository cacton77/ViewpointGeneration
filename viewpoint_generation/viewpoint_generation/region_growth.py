"""
Optimized Point Cloud Region Growing Library using Open3D

This library provides efficient region growing algorithms for point cloud segmentation
with various optimizations for runtime performance.
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class RegionGrowingConfig:
    """Configuration parameters for region growing algorithms."""

    # Core parameters
    seed_threshold: float = 0.1
    region_threshold: float = 0.2
    min_cluster_size: int = 10
    max_cluster_size: int = 100000

    # Spatial search parameters
    knn_neighbors: int = 30
    radius_search: Optional[float] = None

    # Normal-based parameters
    normal_angle_threshold: float = np.pi / 6  # 30 degrees
    curvature_threshold: float = 0.1

    # Optimization parameters
    use_spatial_hashing: bool = True
    spatial_hash_resolution: float = 0.05
    batch_size: int = 1000

    # Filtering parameters
    remove_statistical_outliers: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0


class SpatialHashGrid:
    """Spatial hash grid for efficient nearest neighbor queries."""

    def __init__(self, points: np.ndarray, resolution: float):
        self.resolution = resolution
        self.grid = {}
        self._build_grid(points)

    def _build_grid(self, points: np.ndarray):
        """Build the spatial hash grid."""
        grid_coords = np.floor(points / self.resolution).astype(int)

        for i, coord in enumerate(grid_coords):
            key = tuple(coord)
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(i)

    def get_neighbors_in_radius(self, point: np.ndarray, radius: float) -> List[int]:
        """Get all point indices within radius of given point."""
        neighbors = []
        grid_radius = int(np.ceil(radius / self.resolution))
        center_coord = np.floor(point / self.resolution).astype(int)

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                for dz in range(-grid_radius, grid_radius + 1):
                    key = tuple(center_coord + np.array([dx, dy, dz]))
                    if key in self.grid:
                        neighbors.extend(self.grid[key])

        return neighbors


class RegionGrowing:
    """
    Optimized region growing implementation for point cloud segmentation.

    Features:
    - Normal-based and curvature-based region growing
    - Spatial indexing for fast neighbor queries
    - Vectorized operations for performance
    - Configurable parameters
    - Statistical outlier removal
    """

    def __init__(self, config: RegionGrowingConfig = None):
        self.config = config or RegionGrowingConfig()
        self.point_cloud = None
        self.points = None
        self.normals = None
        self.curvatures = None
        self.spatial_index = None
        self.spatial_hash = None

    def preprocess_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Preprocess point cloud with filtering and normal estimation."""
        pc = point_cloud

        # Remove statistical outliers
        if self.config.remove_statistical_outliers:
            pc, _ = pc.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio
            )

        # Estimate normals if not present
        if not pc.has_normals():
            pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            pc.orient_normals_consistent_tangent_plane(100)

        self.points = np.asarray(pc.points)
        self.normals = np.asarray(pc.normals)

        return pc

    def compute_curvatures(self, points: np.ndarray, normals: np.ndarray,
                           neighbors_list: List[List[int]]) -> np.ndarray:
        """Compute curvature values for each point using local neighborhood."""
        curvatures = np.zeros(len(points))

        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) < 3:
                curvatures[i] = 1.0  # High curvature for isolated points
                continue

            # Get neighbor points and normals
            neighbor_points = points[neighbors]
            neighbor_normals = normals[neighbors]

            # Compute covariance matrix of neighbor points
            centered_points = neighbor_points - \
                np.mean(neighbor_points, axis=0)
            cov_matrix = np.cov(centered_points.T)

            # Eigenvalues represent principal curvatures
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)

            # Curvature as ratio of smallest to largest eigenvalue
            if eigenvals[-1] > 1e-10:
                curvatures[i] = eigenvals[0] / eigenvals[-1]
            else:
                curvatures[i] = 1.0

        return curvatures

    def build_spatial_structures(self, points: np.ndarray):
        """Build spatial indexing structures for efficient neighbor queries."""
        # Build KDTree for standard queries
        temp_pc = o3d.geometry.PointCloud()
        temp_pc.points = o3d.utility.Vector3dVector(points)
        self.spatial_index = o3d.geometry.KDTreeFlann(temp_pc)

        # Build spatial hash grid if enabled
        if self.config.use_spatial_hashing:
            self.spatial_hash = SpatialHashGrid(
                points, self.config.spatial_hash_resolution)

    def get_neighbors(self, point_idx: int) -> List[int]:
        """Get neighbors for a point using the most appropriate method."""
        if self.config.radius_search is not None:
            # Radius search
            [_, neighbors, _] = self.spatial_index.search_radius_vector_3d(
                self.points[point_idx], self.config.radius_search
            )
            return neighbors
        else:
            # KNN search
            [_, neighbors, _] = self.spatial_index.search_knn_vector_3d(
                self.points[point_idx], self.config.knn_neighbors
            )
            return neighbors[1:]  # Exclude the point itself

    def compute_similarity_metrics(self, point_idx: int, neighbor_idx: int) -> Tuple[float, float]:
        """Compute similarity metrics between two points."""
        # Normal angle difference
        normal_diff = np.arccos(np.clip(
            np.dot(self.normals[point_idx],
                   self.normals[neighbor_idx]), -1.0, 1.0
        ))

        # Curvature difference
        curvature_diff = abs(
            self.curvatures[point_idx] - self.curvatures[neighbor_idx])

        return normal_diff, curvature_diff

    def is_valid_seed(self, point_idx: int) -> bool:
        """Check if a point can be used as a seed for region growing."""
        return self.curvatures[point_idx] < self.config.seed_threshold

    def can_merge_to_region(self, point_idx: int, region_normal: np.ndarray,
                            region_curvature: float) -> bool:
        """Check if a point can be merged to an existing region."""
        # Check normal similarity
        normal_angle = np.arccos(np.clip(
            np.dot(self.normals[point_idx], region_normal), -1.0, 1.0
        ))

        if normal_angle > self.config.normal_angle_threshold:
            return False

        # Check curvature similarity
        curvature_diff = abs(self.curvatures[point_idx] - region_curvature)
        if curvature_diff > self.config.curvature_threshold:
            return False

        return True

    def grow_region_from_seed(self, seed_idx: int, processed: np.ndarray) -> List[int]:
        """Grow a region starting from a seed point using BFS."""
        region = []
        queue = deque([seed_idx])
        processed[seed_idx] = True

        # Initialize region properties with seed
        region_normal = self.normals[seed_idx].copy()
        region_curvature = self.curvatures[seed_idx]

        while queue and len(region) < self.config.max_cluster_size:
            current_idx = queue.popleft()
            region.append(current_idx)

            # Get neighbors of current point
            neighbors = self.get_neighbors(current_idx)

            for neighbor_idx in neighbors:
                if processed[neighbor_idx]:
                    continue

                # Check if neighbor can be merged to region
                if self.can_merge_to_region(neighbor_idx, region_normal, region_curvature):
                    processed[neighbor_idx] = True
                    queue.append(neighbor_idx)

                    # Update region properties (weighted average)
                    weight = 1.0 / (len(region) + 1)
                    region_normal = (1 - weight) * region_normal + \
                        weight * self.normals[neighbor_idx]
                    region_normal /= np.linalg.norm(region_normal)  # Normalize
                    region_curvature = (
                        1 - weight) * region_curvature + weight * self.curvatures[neighbor_idx]

        return region if len(region) >= self.config.min_cluster_size else []

    def segment(self, point_cloud: o3d.geometry.PointCloud) -> Tuple[List[List[int]], List[int]]:
        """
        Perform region growing segmentation on point cloud.

        Args:
            point_cloud: Input point cloud

        Returns:
            Tuple of (list of clusters, list of noise points)
        """
        start_time = time.time()

        # Preprocess point cloud
        self.point_cloud = self.preprocess_point_cloud(point_cloud)
        self.points = np.asarray(self.point_cloud.points)
        self.normals = np.asarray(self.point_cloud.normals)

        print(f"Preprocessing completed in {time.time() - start_time:.2f}s")

        # Build spatial structures
        build_start = time.time()
        self.build_spatial_structures(self.points)
        print(
            f"Spatial indexing completed in {time.time() - build_start:.2f}s")

        if self.curvatures is None:
            # Compute curvatures
            curvature_start = time.time()
            neighbors_list = [self.get_neighbors(i)
                            for i in range(len(self.points))]
            self.curvatures = self.compute_curvatures(
                self.points, self.normals, neighbors_list)
            print(
                f"Curvature computation completed in {time.time() - curvature_start:.2f}s")

        # Find seed points (points with low curvature)
        seed_candidates = [i for i in range(
            len(self.points)) if self.is_valid_seed(i)]
        # Sort by curvature (lowest first)
        seed_candidates.sort(key=lambda x: self.curvatures[x])

        print(f"Found {len(seed_candidates)} seed candidates")

        # Region growing
        clusters = []
        processed = np.zeros(len(self.points), dtype=bool)

        growing_start = time.time()
        for seed_idx in seed_candidates:
            if processed[seed_idx]:
                continue

            region = self.grow_region_from_seed(seed_idx, processed)
            if region:
                clusters.append(region)
                print(f"Cluster {len(clusters)}: {len(region)} points")

        print(
            f"Region growing completed in {time.time() - growing_start:.2f}s")

        # Identify noise points
        noise_points = [i for i in range(len(self.points)) if not processed[i]]

        print(f"Total segmentation time: {time.time() - start_time:.2f}s")
        print(
            f"Found {len(clusters)} clusters and {len(noise_points)} noise points")

        return clusters, noise_points

    def visualize_segmentation(self, clusters: List[List[int]], noise_points: List[int]):
        """Visualize the segmentation results with different colors for each cluster."""
        if self.point_cloud is None:
            raise ValueError(
                "No point cloud data available. Run segment() first.")

        # Create colored point cloud
        colored_pc = o3d.geometry.PointCloud()
        colored_pc.points = self.point_cloud.points

        # Generate colors
        colors = np.zeros((len(self.points), 3))

        # Assign random colors to clusters
        np.random.seed(42)  # For reproducible colors
        for i, cluster in enumerate(clusters):
            color = np.random.rand(3)
            for point_idx in cluster:
                colors[point_idx] = color

        # Assign gray color to noise points
        for point_idx in noise_points:
            colors[point_idx] = [0.5, 0.5, 0.5]

        colored_pc.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        o3d.visualization.draw_geometries([colored_pc])


# Utility functions
def create_sample_point_cloud(n_points: int = 10000) -> o3d.geometry.PointCloud:
    """Create a sample point cloud for testing."""
    points = []

    # Create a few geometric shapes
    # Plane
    x = np.random.uniform(-5, 5, n_points // 3)
    y = np.random.uniform(-5, 5, n_points // 3)
    z = np.random.normal(0, 0.1, n_points // 3)
    plane_points = np.column_stack([x, y, z])
    points.append(plane_points)

    # Sphere
    phi = np.random.uniform(0, 2 * np.pi, n_points // 3)
    theta = np.random.uniform(0, np.pi, n_points // 3)
    r = 2 + np.random.normal(0, 0.1, n_points // 3)
    x = r * np.sin(theta) * np.cos(phi) + 8
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) + 3
    sphere_points = np.column_stack([x, y, z])
    points.append(sphere_points)

    # Cylinder
    theta = np.random.uniform(0, 2 * np.pi, n_points // 3)
    r = 1.5 + np.random.normal(0, 0.05, n_points // 3)
    z = np.random.uniform(-3, 3, n_points // 3)
    x = r * np.cos(theta) - 8
    y = r * np.sin(theta)
    cylinder_points = np.column_stack([x, y, z])
    points.append(cylinder_points)

    all_points = np.vstack(points)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_points)

    return pc


# Example usage
if __name__ == "__main__":
    # Create sample point cloud
    print("Creating sample point cloud...")
    pc = create_sample_point_cloud(5000)

    # Configure region growing
    config = RegionGrowingConfig(
        seed_threshold=0.05,
        region_threshold=0.1,
        min_cluster_size=50,
        normal_angle_threshold=np.pi / 8,  # 22.5 degrees
        curvature_threshold=0.05,
        knn_neighbors=20
    )

    # Perform segmentation
    rg = RegionGrowing(config)
    clusters, noise = rg.segment(pc)

    # Visualize results
    print("\nVisualization (close window to continue):")
    rg.visualize_segmentation(clusters, noise)

    # Print statistics
    print(f"\nSegmentation Results:")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Number of noise points: {len(noise)}")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {len(cluster)} points")
