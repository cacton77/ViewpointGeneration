"""
Mesh Region Growing Library using Open3D

Grows contiguous regions directly over a triangle mesh's face-adjacency
graph, using face-normal similarity as the merge criterion. This replaces a
point-cloud/KNN spatial-search approach with exact surface topology: two
faces are only ever considered neighbors if they share a mesh edge, so
growth can never bridge across a fold, thin wall, or gap the way a spatial
radius/KNN search over a sampled point cloud can.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import open3d as o3d


@dataclass
class RegionGrowingConfig:
    """Configuration parameters for mesh-based region growing."""

    # Core parameters
    seed_threshold: float = 0.1
    curvature_threshold: float = 0.1
    min_cluster_size: int = 10       # faces
    max_cluster_size: int = 100000   # faces

    # Normal-based parameters
    normal_angle_threshold: float = np.pi / 6  # 30 degrees

    def to_dict(self):
        return {
            "seed_threshold": {
                "value": self.seed_threshold,
                "type": "float",
                "description": "Curvature threshold for seed face selection",
                "control": "slider",
                "range": [0.0, 1.0],
            },
            "curvature_threshold": {
                "value": self.curvature_threshold,
                "type": "float",
                "description": "Curvature threshold for region growing",
                "control": "slider",
                "range": [0.0, 1.0],
            },
            "min_cluster_size": {
                "value": self.min_cluster_size,
                "type": "integer",
                "description": "Minimum faces per region",
                "control": "slider",
                "range": [1, self.max_cluster_size],
            },
            "max_cluster_size": {
                "value": self.max_cluster_size,
                "type": "integer",
                "description": "Maximum faces per region",
                "control": "slider",
                "range": [self.min_cluster_size, 1000000],
            },
            "normal_angle_threshold": {
                "value": self.normal_angle_threshold,
                "type": "float",
                "description": "Maximum angle difference in radians for normal-based region growing",
                "control": "slider",
                "range": [0, np.pi],
            },
        }


class RegionGrowing:
    """
    Mesh-native region growing over triangle face-adjacency.

    Features:
    - Normal-based and curvature-based region growing directly on mesh faces
    - Exact face-adjacency neighbor queries (no spatial search, no sampling-
      density dependence)
    - Configurable parameters
    """

    def __init__(self, config: RegionGrowingConfig = None):
        self.config = config or RegionGrowingConfig()
        self.mesh = None
        self.face_normals = None
        self.face_curvatures = None
        self.adjacency = None  # list of np.ndarray, per-face neighbor face indices

    def build_face_adjacency(self, triangles: np.ndarray) -> List[np.ndarray]:
        """Build per-face neighbor lists from shared mesh edges."""
        edge_faces = {}
        for face_idx, tri in enumerate(triangles):
            for j in range(3):
                a, b = int(tri[j]), int(tri[(j + 1) % 3])
                key = (a, b) if a < b else (b, a)
                edge_faces.setdefault(key, []).append(face_idx)

        neighbors = [set() for _ in range(len(triangles))]
        for face_list in edge_faces.values():
            if len(face_list) < 2:
                continue
            for i in face_list:
                neighbors[i].update(f for f in face_list if f != i)

        return [np.fromiter(n, dtype=int) for n in neighbors]

    def compute_face_curvatures(self, face_normals: np.ndarray,
                                 adjacency: List[np.ndarray]) -> np.ndarray:
        """Curvature analog: how much a face's normal deviates from its
        adjacent faces' normals (0 = perfectly flat neighborhood)."""
        curvatures = np.zeros(len(face_normals))

        for i, neighbors in enumerate(adjacency):
            if len(neighbors) == 0:
                curvatures[i] = 1.0  # High curvature for isolated faces
                continue

            dots = np.clip(face_normals[neighbors] @ face_normals[i], -1.0, 1.0)
            curvatures[i] = 1.0 - np.mean(dots)

        return curvatures

    def preprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Ensure triangle normals are present and build face adjacency."""
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()

        self.mesh = mesh
        self.face_normals = np.asarray(mesh.triangle_normals)
        self.adjacency = self.build_face_adjacency(np.asarray(mesh.triangles))

        return mesh

    def is_valid_seed(self, face_idx: int) -> bool:
        """Check if a face can be used as a seed for region growing."""
        return self.face_curvatures[face_idx] < self.config.seed_threshold

    def can_merge_to_region(self, face_idx: int, region_normal: np.ndarray,
                             region_curvature: float) -> bool:
        """Check if a face can be merged into an existing region."""
        normal_angle = np.arccos(np.clip(
            np.dot(self.face_normals[face_idx], region_normal), -1.0, 1.0
        ))
        if normal_angle > self.config.normal_angle_threshold:
            return False

        curvature_diff = abs(self.face_curvatures[face_idx] - region_curvature)
        if curvature_diff > self.config.curvature_threshold:
            return False

        return True

    def grow_region_from_seed(self, seed_idx: int, processed: np.ndarray) -> List[int]:
        """Grow a region starting from a seed face using BFS over face
        adjacency. Faces visited while growing a region that ends up below
        ``min_cluster_size`` are left unmarked in ``processed`` so a
        different, larger region can still claim them."""
        region = []
        local_visited = {seed_idx}
        queue = deque([seed_idx])

        region_normal = self.face_normals[seed_idx].copy()
        region_curvature = self.face_curvatures[seed_idx]

        while queue and len(region) < self.config.max_cluster_size:
            current_idx = queue.popleft()
            region.append(current_idx)

            for neighbor_idx in self.adjacency[current_idx]:
                neighbor_idx = int(neighbor_idx)
                if processed[neighbor_idx] or neighbor_idx in local_visited:
                    continue

                if self.can_merge_to_region(neighbor_idx, region_normal, region_curvature):
                    local_visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

                    # Update region properties (weighted average)
                    weight = 1.0 / (len(region) + 1)
                    region_normal = (1 - weight) * region_normal + \
                        weight * self.face_normals[neighbor_idx]
                    norm = np.linalg.norm(region_normal)
                    if norm > 1e-10:
                        region_normal /= norm
                    region_curvature = (
                        1 - weight) * region_curvature + weight * self.face_curvatures[neighbor_idx]

        if len(region) < self.config.min_cluster_size:
            return []

        for face_idx in region:
            processed[face_idx] = True
        return region

    def segment(self, mesh: o3d.geometry.TriangleMesh) -> Tuple[List[List[int]], List[int]]:
        """
        Perform region growing segmentation directly on a triangle mesh.

        Args:
            mesh: Input triangle mesh.

        Returns:
            Tuple of (list of regions, list of noise face indices). Each
            region is a list of triangle indices into ``mesh.triangles``.
        """
        start_time = time.time()

        self.preprocess_mesh(mesh)
        n_faces = len(self.face_normals)

        curvature_start = time.time()
        self.face_curvatures = self.compute_face_curvatures(
            self.face_normals, self.adjacency)
        print(f"Curvature computation completed in {time.time() - curvature_start:.2f}s")

        # Find seed faces (faces with low curvature), flattest first
        seed_candidates = [i for i in range(n_faces) if self.is_valid_seed(i)]
        seed_candidates.sort(key=lambda x: self.face_curvatures[x])

        print(f"Found {len(seed_candidates)} seed candidates out of {n_faces} faces")

        regions = []
        processed = np.zeros(n_faces, dtype=bool)

        growing_start = time.time()
        for seed_idx in seed_candidates:
            if processed[seed_idx]:
                continue

            region = self.grow_region_from_seed(seed_idx, processed)
            if region:
                regions.append(region)
                print(f"Region {len(regions)}: {len(region)} faces")

        print(f"Region growing completed in {time.time() - growing_start:.2f}s")

        noise_faces = [i for i in range(n_faces) if not processed[i]]

        print(f"Total segmentation time: {time.time() - start_time:.2f}s")
        print(f"Found {len(regions)} regions and {len(noise_faces)} noise faces")

        return regions, noise_faces


# Utility functions
def create_sample_mesh(k: int = 5) -> o3d.geometry.TriangleMesh:
    """Create a sample triangle mesh (a few primitives) for testing."""
    combined_mesh = o3d.geometry.TriangleMesh()
    for i in range(k):
        rand_n = np.random.randint(0, 3)
        if rand_n == 0:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        elif rand_n == 1:
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=3.0, height=3.0, depth=3.0)
        else:
            mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.5, height=4.0)

        mesh.translate(np.random.uniform(-10, 10, 3))
        combined_mesh += mesh

    combined_mesh.compute_triangle_normals()
    return combined_mesh


def visualize_segmentation(mesh: o3d.geometry.TriangleMesh,
                            regions: List[List[int]], noise_faces: List[int]):
    """Visualize the segmentation by painting each region's submesh a
    distinct color and merging noise faces in gray."""
    from viewpoint_generation.mesh_utils import submesh_from_faces

    np.random.seed(42)
    mesh_triangles = np.asarray(mesh.triangles)
    pieces = []
    for region in regions:
        sub, _ = submesh_from_faces(mesh, mesh_triangles[np.asarray(region)])
        sub.paint_uniform_color(np.random.rand(3))
        pieces.append(sub)

    if noise_faces:
        sub, _ = submesh_from_faces(mesh, mesh_triangles[np.asarray(noise_faces)])
        sub.paint_uniform_color([0.5, 0.5, 0.5])
        pieces.append(sub)

    o3d.visualization.draw_geometries(pieces, mesh_show_back_face=True)


# Example usage
if __name__ == "__main__":
    print("Creating sample mesh...")
    mesh = create_sample_mesh(5)

    config = RegionGrowingConfig(
        seed_threshold=0.05,
        curvature_threshold=0.05,
        min_cluster_size=20,
        normal_angle_threshold=np.pi / 8,  # 22.5 degrees
    )

    rg = RegionGrowing(config)
    regions, noise_faces = rg.segment(mesh)

    print("\nVisualization (close window to continue):")
    visualize_segmentation(mesh, regions, noise_faces)

    print(f"\nSegmentation Results:")
    print(f"Number of regions: {len(regions)}")
    print(f"Number of noise faces: {len(noise_faces)}")
    for i, region in enumerate(regions):
        print(f"Region {i + 1}: {len(region)} faces")
