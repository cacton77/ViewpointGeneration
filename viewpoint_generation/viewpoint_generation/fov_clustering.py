import time
import threading
import math
import copy
import numpy as np
import open3d as o3d
from multiprocessing import Queue, Process
from scipy.spatial.distance import euclidean
from itertools import permutations

from sklearn.cluster import KMeans  # . . . . . . . . K-means
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from open3d.geometry import PointCloud, TriangleMesh
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from dataclasses import dataclass


@dataclass
class FOVClusteringConfig:
    """Configuration parameters for region growing algorithms."""

    fov_height: float = 0.02  # Field of view height in m
    fov_width: float = 0.03  # Field of view width in m
    dof: float = 0.02  # Depth of field in m

    # Cost function parameters
    ppsqmm: float = 10.0  # Points per square millimeter for evaluation
    lambda_weight: float = 1.0  # Weight for point out percentage in cost function
    beta_weight: float = 1.0  # Weight for packing efficiency in cost function
    # Maximum allowed point out percentage for a valid cluster
    max_point_out_percentage: float = 0.001

    # Clustering parameters
    point_weight: float = 2.0  # Weight for point locations in clustering
    normal_weight: float = 1.0  # Weight for normals in clustering
    number_of_runs: int = 10  # Number of runs for KMeans
    maximum_iterations: int = 100  # Maximum iterations for KMeans


class FOVClustering:
    def __init__(self, config: FOVClusteringConfig = None):
        self.config = config or FOVClusteringConfig()

    def evaluate_fov_cluster(self, points, normals):
        """ Function to evaluate a cluster based on field of view and depth of field. """
        # Check for minimum number of points
        if points.shape[0] < 4:
            print("Not enough points in the cluster to evaluate.")
            return False, 0, 0, False

        # Convert points and normals to Open3D PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.normals = o3d.utility.Vector3dVector(normals)

        z = np.array([0, 0, 1])
        z_hat = np.average(normals, axis=0)
        x_hat = np.cross(z, z_hat) if np.linalg.norm(
            np.cross(z, z_hat)) != 0 else np.array([1, 0, 0])
        y_hat = np.cross(z_hat, x_hat)
        x_hat = x_hat / np.linalg.norm(x_hat)
        y_hat = y_hat / np.linalg.norm(y_hat)
        z_hat = z_hat / np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))
        t = point_cloud.get_center()

        camera_radius = min(self.config.fov_width, self.config.fov_height) / 2

        # Visualize
        obb = point_cloud.get_minimal_oriented_bounding_box(robust=True)
        camera = o3d.geometry.TriangleMesh.create_cylinder(
            radius=camera_radius, height=2*self.config.dof)
        # Remove top and bottom faces of the cylinder
        camera = camera.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-camera_radius, -camera_radius, -self.config.dof/2],
            max_bound=[camera_radius, camera_radius, self.config.dof/2]))
        camera.paint_uniform_color([1, 0, 0])  # Red color for camera
        obb.color = (0, 1, 0)
        point_cloud.rotate(np.linalg.inv(R), t)
        point_cloud.translate(-t)
        obb.rotate(np.linalg.inv(R), t)
        obb.translate(-t)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([point_cloud, obb, camera, axis], mesh_show_back_face=True)

        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        # Convert to square millimeters
        max_points_in = (np.pi * camera_radius**2) * (self.config.ppsqmm * 1e6)

        # Get height and radial coordinates
        height_coords = points[:, 2]

        # Get indices for the two radial dimensions
        radial_indices = [i for i in range(3) if i != 2]
        radial_coords = points[:, radial_indices]

        # Check constraints
        height_mask = (height_coords >= 0) & (height_coords <= self.config.dof)
        radial_distances_sq = np.sum(radial_coords**2, axis=1)
        extreme_radial_distance = np.max(
            np.sqrt(radial_distances_sq)) if radial_distances_sq.size > 0 else 0
        radius_mask = radial_distances_sq <= camera_radius**2

        points_in = np.sum(height_mask & radius_mask)
        points_in_xy = np.sum(radius_mask)
        points_out = points.shape[0] - points_in

        point_out_percentage = points_out / \
            points.shape[0] if points.shape[0] > 0 else 0
        packing_efficiency = points_in_xy / max_points_in if max_points_in > 0 else 0

        if point_out_percentage > self.config.max_point_out_percentage:
            valid = False
        else:
            valid = True

        borderline = False
        epsilon = 0.05
        if (extreme_radial_distance < (camera_radius*(1+epsilon)) and
                extreme_radial_distance > (camera_radius*(1-epsilon))) or (
                extreme_radial_distance < (camera_radius*(1+epsilon)) and
                extreme_radial_distance > (camera_radius*(1-epsilon))) and valid == True:

            borderline = True

        print(
            f'Points in: {points_in}, Points out: {points_out}, Point out percentage: {point_out_percentage}')
        print(f'Packing efficiency: {packing_efficiency}')

        return valid, point_out_percentage, packing_efficiency, borderline

    def partition(self, points, normals, k) -> list:
        """ K-Means clustering function. """

        if (self.config.normal_weight != 0):
            # Scale point locations to lie between [-1, 1]
            points = 2 * (minmax_scale(points) - 0.5)

        # Combine weighted vertex and normal data
        data = np.concatenate((self.config.point_weight * points,
                               self.config.normal_weight * normals), axis=1)

        # Scikit Learn KMeans
        KM = KMeans(init='k-means++',
                    n_clusters=k,
                    n_init=self.config.number_of_runs,
                    max_iter=self.config.maximum_iterations)
        KM.fit(data)

        labels = KM.labels_
        clusters = [[] for i in range(k)]

        for j in range(len(labels)):
            clusters[labels[j]].append(j)

        return clusters

    def evaluate_k(self, points, normals, k, eval_fun, tries=1):
        """ Run multiple k-means partitions to determine if current k is valid. """
        for i in range(tries):
            clusters = self.partition(points, normals, k)
            k_valid = True
            for j, cluster in enumerate(clusters):
                cluster_valid, cost, _ = eval_fun(cluster)
                k_valid = k_valid and cluster_valid
                if not cluster_valid:
                    break
            if k_valid:
                return True, clusters, cost
        return False, clusters, cost

    def evaluate_k_cost(self, points, normals, k, eval_fun, tries=1):
        """ Evaluate the cost of a given k value for clustering. """
        # this is to make sure that k is always an integer and minimum value of k is 1
        k = max(1, int(k))

        for i in range(tries):
            clusters = self.partition(points, normals, k)
            k_valid = True
            total_cost = 0
            total_count = len(clusters)
            total_point_out_percentage = 0
            total_packing_efficiency = 0
            anyborderline = False

            for j, cluster in enumerate(clusters):
                cluster_points = points[cluster, :]
                cluster_normals = normals[cluster, :]

                cluster_valid, point_out_percentage, packing_efficiency, borderline = eval_fun(
                    copy.deepcopy(cluster_points), copy.deepcopy(cluster_normals))

                total_point_out_percentage += point_out_percentage
                total_packing_efficiency += packing_efficiency
                anyborderline = anyborderline or borderline
                k_valid = k_valid and cluster_valid

            total_point_out_percentage = total_point_out_percentage/total_count
            total_packing_efficiency = total_packing_efficiency/total_count

            if (total_point_out_percentage > 0.001):
                # Remove packing efficiency from cost if point out percentage is greater than 0.001
                s = 0
            else:
                print("total_point_out_percentage is zero")
                s = 1
                if (anyborderline == True):
                    s = 0
                # print(total_packing_eff)

            total_cost = (self.config.lambda_weight)*total_point_out_percentage + \
                s*((1/total_packing_efficiency)**self.config.beta_weight)

        return -total_cost

    def optimize_k_b_opt(self, points, normals, eval_fun) -> int:

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)

        temp_mesh = temp_pcd.compute_convex_hull(joggle_inputs=True)[0]
        # calculate the bounding box of the mesh
        bounding_box = temp_mesh.get_axis_aligned_bounding_box()
        bounding_box.color = (0, 1, 0)

        # Show the temp_pcd, temp_mesh, and bounding_box
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([temp_pcd, temp_mesh, bounding_box, axis])

        # calculate the length and width of the bounding box
        length = bounding_box.get_max_bound(
        )[0]-bounding_box.get_min_bound()[0]
        width = bounding_box.get_max_bound(
        )[1]-bounding_box.get_min_bound()[1]

        # check if the length is greater than the width
        greater_dimension = max(length, width)
        ratio = max(length, width)/min(length, width)

        camera_width = self.config.fov_width
        camera_height = self.config.fov_height
        # camera_area = camera_width*camera_height
        camera_r = min(camera_width, camera_height)/2
        camera_area = np.pi*(camera_r**2)
        surface_area = temp_mesh.get_surface_area()/2

        # Check for long and narrow regions.
        # If the ratio is greater than 100, we assume that the region is long and narrow.
        # It may be worth trying different cases for different bounds.
        if ratio > 100:
            k_min = greater_dimension/(2*camera_r)
            pbounds = {"k": (k_min/2, 3*(k_min/2))}
        else:
            k_min = surface_area/camera_area
            pbounds = {"k": (k_min, 3*k_min)}

        def f(k): return self.evaluate_k_cost(
            points, normals, k=k, eval_fun=eval_fun)

        optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            verbose=2,  # verbose=1 prints only at max, verbose=0 is silent
            random_state=1,
        )
        acq_function = UtilityFunction(kind="ei", kappa=5)
        optimizer.maximize(
            init_points=1,
            n_iter=3,
        )
        y = optimizer.max
        k_opt = max(1, int(y["params"]["k"]))
        valid_clusters = self.partition(points, normals, k_opt)
        non_valid_pcd = 0
        total_count = 0
        total_point_out_percentage = 0
        total_packing_eff = 0

        for j, cluster in enumerate(valid_clusters):
            cluster_valid, point_out_percentage, packing_eff, borderline = eval_fun(
                points, normals)
            total_point_out_percentage += point_out_percentage
            total_packing_eff += packing_eff
            total_count += 1
            # print(f'k-{k} pcd {j}: {cluster_valid}')
            if not cluster_valid:
                non_valid_pcd += 1

        avg_point_out_percentage = total_point_out_percentage/total_count
        avg_packing_eff = total_packing_eff/total_count

        return k_opt, valid_clusters

    def fov_clustering(self, point_cloud):
        """ Partition a region of a point cloud into regions within camera fov and dof. """
        # Get the region point cloud

        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        k_opt, fov_clusters = self.optimize_k_b_opt(
            points, normals, self.evaluate_fov_cluster)

        return fov_clusters


# Utility functions
def create_sample_mesh(k: int = 3, ppsqmm: float = 1000) -> o3d.geometry.PointCloud:
    """Create a sample point cloud for testing."""

    # Randomly create k primitive shapes
    combined_mesh = o3d.geometry.TriangleMesh()
    for i in range(k):
        rand_n = np.random.randint(0, 3)
        if rand_n == 0:
            # Create a sphere
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        elif rand_n == 1:
            # Create a box
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=0.1, height=0.1, depth=0.1)
        elif rand_n == 2:
            # Create a cylinder
            mesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.05, height=0.2)

        mesh.translate(np.random.rand(3))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(
            np.random.rand(3) * np.pi))
        combined_mesh += mesh

    # Set a uniform color for the mesh
    combined_mesh.paint_uniform_color((0.5, 0.5, 0.5))

    return combined_mesh


# Example usage
if __name__ == "__main__":
    # Create sample point cloud
    print("Creating sample point cloud...")
    k = 5
    ppsqmm = 0.5
    mesh = create_sample_mesh(k)

    pc = mesh.sample_points_uniformly(
        int(mesh.get_surface_area() * ppsqmm * 1e6), use_triangle_normal=True)

    # Configure region growing
    config = FOVClusteringConfig()
    config.ppsqmm = ppsqmm  # Points per square millimeter
    config.fov_width = 50*2*0.001*np.sqrt(1/np.pi)
    config.fov_height = 50*2*0.001*np.sqrt(1/np.pi)
    config.dof = 1

    # Perform segmentation
    fc = FOVClustering(config)

    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)

    regions = fc.partition(points, normals, k)
    # Random colors for each region
    region_colors = np.random.rand(len(regions), 3)
    fov_cluster_meshes = [mesh]

    for i, region in enumerate(regions):
        region_pc = pc.select_by_index(regions[i])
        region_points = np.asarray(region_pc.points)
        region_normals = np.asarray(region_pc.normals)

        # FOV clustering
        region_fov_clusters = fc.fov_clustering(region_pc)
        for fov_cluster in region_fov_clusters:
            fov_cluster_pc = region_pc.select_by_index(fov_cluster)
            # Translate slightly by average normal to avoid overlapping
            avg_normal = np.mean(np.asarray(fov_cluster_pc.normals), axis=0)
            fov_cluster_pc.translate(avg_normal * 0.001)  # Translate by
            fov_cluster_mesh = fov_cluster_pc.compute_convex_hull(joggle_inputs=True)[
                0]
            # Slightly adjust color
            color = region_colors[i] + 0.1*(np.random.rand(3) - 0.5)
            # Ensure color values are between 0 and 1
            color = np.clip(color, 0, 1)
            fov_cluster_mesh.paint_uniform_color(
                color)  # Set color for the FOV cluster
            fov_cluster_meshes.append(fov_cluster_mesh)

    # Visualize the clusters
    print("Visualizing clusters...")
    o3d.visualization.draw_geometries(
        fov_cluster_meshes, window_name="FOV Clusters")
