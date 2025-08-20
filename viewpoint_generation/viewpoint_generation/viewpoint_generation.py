import os
import json
import time
import math
import random
import datetime
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from matplotlib import colormaps
from open3d.geometry import PointCloud, TriangleMesh

from viewpoint_generation.curvature import *
from viewpoint_generation.region_growth import *
from viewpoint_generation.fov_clustering import *
from viewpoint_generation.viewpoint_projection import *


class ViewpointGeneration():

    mesh_file = None
    mesh_units = 'm'
    point_cloud_file = None
    point_cloud_units = 'm'
    curvatures_file = None
    regions_file = None

    mesh = None
    point_cloud = None
    curvatures = None  # Will be set after estimating curvature

    ppsqmm = 100

    # Region Growth Parameters

    regions_dict = {}

    fov_height = 0.02
    fov_width = 0.03
    dof = 0.02

    visualize = False
    cuda_enabled = False  # Set to True if using CuPy for GPU acceleration
    mesh_color = (0.5, 0.5, 0.5)
    background_color = (0.1, 0.1, 0.1)
    bb_color = (1., 1., 1.)
    text_color = (1., 1., 1.)
    viewer = o3d.visualization.Visualizer()
    is_running = False

    region_growing_config = RegionGrowingConfig(
        seed_threshold=0.1,
        region_threshold=0.2,
        min_cluster_size=10,
        normal_angle_threshold=np.pi / 6,
        curvature_threshold=0.1,
        knn_neighbors=30
    )
    fc_config = FOVClusteringConfig(
        fov_height=0.02,
        fov_width=0.03,
        dof=0.02,
        ppsqmm=10.0,
        lambda_weight=1.0,
        beta_weight=1.0,
        max_point_out_percentage=0.001,
        point_weight=1.0,
        normal_weight=1.0,
        number_of_runs=10,
        maximum_iterations=100
    )
    vp_config = ViewpointProjectionConfig(
        focal_distance=0.3
    )

    rg = RegionGrowing(region_growing_config)
    fc = FOVClustering(fc_config)
    vp = ViewpointProjection(vp_config)

    def __init__(self):
        pass

    def enable_cuda(self, enabled):
        """
        Set whether to use CuPy for GPU acceleration.
        Args:
            enabled (bool): Whether to enable CuPy for GPU acceleration.
        """
        if enabled:
            try:
                cp.cuda.Device(0).use()
                print("CuPy available. Using GPU acceleration.")
            except cp.cuda.runtime.CUDARuntimeError:
                print("CuPy not available. Using CPU instead.")
                self.cuda_enabled = False
                return False

        self.cuda_enabled = enabled

        return True

    def get_mesh_dimensions(self):
        """
        Get the dimensions of the mesh.
        Returns:
            tuple: (width, height, depth) of the mesh in meters.
        """
        if self.mesh is None:
            return None, None, None

        bbox = self.mesh.get_axis_aligned_bounding_box()
        width = bbox.max_bound[0] - bbox.min_bound[0]
        depth = bbox.max_bound[1] - bbox.min_bound[1]
        height = bbox.max_bound[2] - bbox.min_bound[2]

        return width, depth, height

    def get_mesh_vertices_and_triangles(self):
        """
        Get the vertices and triangles of the mesh.
        Returns:
            tuple: (vertices, triangles) where vertices is a numpy array of shape (N, 3)
            and triangles is a numpy array of shape (M, 3).
        """
        if self.mesh is None:
            return None, None

        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        return vertices, triangles

    def set_mesh_file(self, mesh_file, units):
        if mesh_file == '':
            return False, 'No triangle mesh file provided.'

        if mesh_file is not self.mesh_file or units is not self.mesh_units:
            if mesh_file == '':
                return False, 'No triangle mesh file provided.'

            try:
                mesh = o3d.io.read_triangle_mesh(mesh_file)
            except Exception as e:
                return False, f'Could not load requested triangle mesh file: {e}'

            # Check if the mesh is empty
            if mesh.is_empty():
                return False, 'The loaded triangle mesh is empty.'

            mesh.compute_vertex_normals()
            # Estimate normals if not already present
            if not mesh.has_vertex_normals():
                mesh.estimate_vertex_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=self.region_growing_config.knn_neighbors))

            # Scale the mesh to meters
            if units == 'cm':
                mesh.scale(0.01, center=(0, 0, 0))
            elif units == 'mm':
                mesh.scale(0.001, center=(0, 0, 0))
            elif units == 'in':
                mesh.scale(0.0254, center=(0, 0, 0))
            elif units == 'm':
                # No scaling needed for meters
                pass
            else:
                return False, 'Unknown units. Mesh not scaled.'

            # Check if the mesh has colors
            mesh.paint_uniform_color(self.mesh_color)

            # Update the triangle mesh file
            self.mesh_file = mesh_file
            self.mesh_units = units
            self.mesh = mesh

        return True, f'Triangle mesh file set to \'{mesh_file}\' with units \'{units}\'.'

    def set_point_cloud_file(self, point_cloud_file, point_cloud_units):
        if point_cloud_file is self.point_cloud_file:
            return True, 'Point cloud file already set.'

        if point_cloud_file == '':
            self.point_cloud_file = None
            self.point_cloud = None
            return True, 'Point cloud file cleared.'

        try:
            point_cloud = o3d.io.read_point_cloud(point_cloud_file)
        except Exception as e:
            return False, f'Could not load requested point cloud file: {e}'

        # Check if the point cloud is empty
        if point_cloud.is_empty():
            return False, 'The loaded point cloud is empty.'
        # Check if the point cloud has normals
        if not point_cloud.has_normals():
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=self.region_growing_config.knn_neighbors))

        # Check if the point cloud has colors
        if not point_cloud.has_colors():
            point_cloud.paint_uniform_color((1, 1, 1))

        # Scale the point cloud to meters
        if point_cloud_units == 'cm':
            point_cloud.scale(0.01, center=(0, 0, 0))
        elif point_cloud_units == 'mm':
            point_cloud.scale(0.001, center=(0, 0, 0))
        elif point_cloud_units == 'in':
            point_cloud.scale(0.0254, center=(0, 0, 0))
        elif point_cloud_units == 'm':
            # No scaling needed for meters
            pass
        else:
            return False, f'Unknown units \'{point_cloud_units}\'. Point cloud not scaled.'

        self.point_cloud_file = point_cloud_file
        self.point_cloud = point_cloud
        self.point_cloud_units = point_cloud_units

        return True, f'Point cloud file set to \'{point_cloud_file}\' with units \'{point_cloud_units}\'.'

    def set_ppsqmm(self, ppsqmm):
        if ppsqmm <= 0:
            msg = 'Points per square millimeter must be greater than 0.'
            return False, msg

        self.ppsqmm = ppsqmm

        # Recalculate the number of points to sample based on the new ppsqmm
        if self.mesh is not None:
            N_points = int(self.mesh.get_surface_area() * (self.ppsqmm * 1e6))
            msg = f'Number of points to sample: {N_points}'
            return True, N_points
        else:
            msg = 'No triangle mesh loaded. Cannot set ppsqmm.'
            return False, 0

    def set_sampling_number_of_points(self, N_points):
        if N_points <= 0:
            return False, 'Number of points must be greater than 0.'

        # Update ppsqmm based on the new number of points
        if self.mesh is not None:
            area = self.mesh.get_surface_area()
            ppsqmm = N_points / (area * 1e6)
            self.ppsqmm = ppsqmm
            msg = f'Points per square millimeter set to {self.ppsqmm}.'
            return True, ppsqmm

    def sample_point_cloud(self):
        # Perform poisson disk sampling on the triangle mesh
        # and generate a point cloud
        if self.mesh is None:
            return False, 'No triangle mesh loaded.'
        elif self.is_running:
            return False, 'Point cloud partitioning is running.'

        N_points = int(self.mesh.get_surface_area() * (self.ppsqmm * 1e6))
        if N_points <= 0:
            return False, 'Number of points to sample must be greater than 0.'

        print('Number of points to sample:', N_points)
        print(f'Mesh units: {self.mesh_units}')

        # Save the sampled point cloud to a file under a directory named after the mesh file in the same directory as the mesh file
        mesh_dir = self.mesh_file.rsplit('/', 1)[0]
        mesh_name = self.mesh_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        point_cloud_dir = mesh_dir + '/' + mesh_name + '_point_cloud'
        # Name the point_cloud file after the mesh file name with N_points appended and save as a ply file
        point_cloud_file = point_cloud_dir + '/' + mesh_name + '_' + \
            self.mesh_units + '_point_cloud_' + \
            str(int(N_points)) + 'points.ply'
        # Create the directory if it does not exist
        if not os.path.exists(point_cloud_dir):
            os.makedirs(point_cloud_dir)

        # Check if the point cloud file already exists
        if not os.path.exists(point_cloud_file):
            # Sample the point cloud
            point_cloud = self.mesh.sample_points_poisson_disk(
                number_of_points=int(N_points), init_factor=5, use_triangle_normal=True)
            # Save the point cloud to a file
            o3d.io.write_point_cloud(point_cloud_file, point_cloud)

        message = point_cloud_file

        return True, message

    def set_knn_neighbors(self, k):
        if k <= 0:
            return False, 'Number of neighbors must be greater than 0.'
        self.region_growing_config.knn_neighbors = k
        self.rg.config = self.region_growing_config
        return True, f'KNN neighbors set to {k}.'

    def set_seed_threshold(self, seed_threshold):
        if seed_threshold <= 0:
            return False, 'Seed threshold must be greater than 0.'
        self.region_growing_config.seed_threshold = seed_threshold
        self.rg.config = self.region_growing_config
        return True, f'Seed threshold set to {seed_threshold}.'

    def set_min_cluster_size(self, min_cluster_size):
        if min_cluster_size <= 0:
            return False, 'Minimum cluster size must be greater than 0.'
        self.region_growing_config.min_cluster_size = min_cluster_size
        self.rg.config = self.region_growing_config
        return True, f'Minimum cluster size set to {min_cluster_size}.'

    def set_max_cluster_size(self, max_cluster_size):
        if max_cluster_size <= 0:
            return False, 'Maximum cluster size must be greater than 0.'
        elif max_cluster_size < self.region_growing_config.min_cluster_size:
            return False, 'Maximum cluster size must be greater than or equal to minimum cluster size.'
        self.region_growing_config.max_cluster_size = max_cluster_size
        self.rg.config = self.region_growing_config
        return True, f'Maximum cluster size set to {max_cluster_size}.'

    def set_normal_angle_threshold(self, normal_angle_threshold):
        """
        Set the angle threshold for region growing.
        Args:
            angle_threshold (float): Angle threshold in degrees.
        Returns:
            bool: True if the angle threshold was set successfully, False otherwise.
        """
        if normal_angle_threshold <= 0 or normal_angle_threshold > 180:
            return False, 'Angle threshold must be between 0 and 180 degrees.'

        self.region_growing_config.normal_angle_threshold = normal_angle_threshold * \
            np.pi / 180.0  # Convert to radians
        self.rg.config = self.region_growing_config

        return True, f'Region growth angle threshold set to {normal_angle_threshold} degrees.'

    def set_curvature_threshold(self, curvature_threshold):
        """
        Set the curvature threshold for region growing.
        Args:
            curvature_threshold (float): Curvature threshold in percent.
        Returns:
            bool: True if the curvature threshold was set successfully, False otherwise.
        """
        if curvature_threshold <= 0 or curvature_threshold > 1:
            return False, 'Curvature threshold must be between 0 and 1.'

        self.region_growing_config.curvature_threshold = curvature_threshold
        self.rg.config = self.region_growing_config

        return True, f'Region growth curvature threshold set to {100*curvature_threshold} percent.'

    def set_camera_parameters(self, fov_height, fov_width, dof, focal_distance):
        """
        Set the camera parameters for FOV clustering.
        Args:
            fov_height (float): Height of the field of view in meters.
            fov_width (float): Width of the field of view in meters.
            dof (float): Depth of field in meters.
            focal_distance (float): Focal distance in meters.
        Returns:
            bool: True if the camera parameters were set successfully, False otherwise.
        """
        if fov_height <= 0 or fov_width <= 0 or dof <= 0 or focal_distance <= 0:
            return False, 'FOV height, width, DOF and focal distance must be greater than 0.'

        self.fc_config.fov_height = fov_height
        self.fc_config.fov_width = fov_width
        self.fc_config.dof = dof
        self.vp_config.focal_distance = focal_distance

        self.fc.config = self.fc_config

        # TODO: Set viewpoint generation config parameter focal_distance

        return True, f'Camera parameters set to FOV height: {fov_height} m, FOV width: {fov_width} m, DOF: {dof} m, focal distance: {focal_distance} m.'

    def estimate_curvature(self):
        """ Estimate the curvature of the point cloud using the nearest neighbors. """

        # Check if the point cloud is loaded and has normals
        if self.point_cloud is None:
            print('No point cloud loaded.')
            return False, 'No point cloud loaded.'
        if not self.point_cloud.has_normals():
            print('Estimating normals for the point cloud.')
            self.point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))

        # Check if curvature file already exists
        point_cloud_dir = self.point_cloud_file.rsplit('/', 1)[0]
        point_cloud_name = self.point_cloud_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        curvatures_file = f'{point_cloud_dir}/{point_cloud_name}_{self.region_growing_config.knn_neighbors}nn_curvatures.npy'
        if os.path.exists(curvatures_file):
            print(f'Curvature file already exists: {curvatures_file}')
            # Load the curvature values from the file
            self.curvatures_file = curvatures_file
            self.curvatures = np.load(curvatures_file)
            print('Curvature values loaded from file.')
            return True, curvatures_file

        start_time = time.time()

        # Preprocess point cloud
        point_cloud = self.rg.preprocess_point_cloud(self.point_cloud)
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        print(f"Preprocessing completed in {time.time() - start_time:.2f}s")

        # Build spatial structures
        build_start = time.time()
        self.rg.build_spatial_structures(points)
        print(
            f"Spatial indexing completed in {time.time() - build_start:.2f}s")

        curvature_start = time.time()

        neighbors_list = [self.rg.get_neighbors(i)
                          for i in range(len(points))]
        curvatures = self.rg.compute_curvatures(
            points, normals, neighbors_list)

        print(
            f"Curvature computation completed in {time.time() - curvature_start:.2f}s")

        # Save curvature values to disk named after the point cloud file - .ply
        # Create the directory if it does not exist
        if not os.path.exists(point_cloud_dir):
            os.makedirs(point_cloud_dir)
        # Save the curvature values to a file
        np.save(curvatures_file, curvatures)
        print(f'Curvature values saved to {curvatures_file}.')

        self.curvatures_file = curvatures_file
        self.curvaturess = curvatures

        return True, curvatures_file

    def set_curvature_file(self, curvature_file):
        """
        Set the curvature file to be used for region growing.
        Args:
            curvature_file (str): Path to the curvature file.
        Returns:
            bool: True if the curvature file was set successfully, False otherwise.
        """
        if curvature_file == '':
            self.curvatures_file = None
            self.curvatures = None
            return True, 'Curvature file cleared.'

        if not os.path.exists(curvature_file):
            return False, f'Curvature file does not exist: \'{curvature_file}\'.'

        self.curvatures = np.load(curvature_file)

        return True, f'Curvature file set to \'{curvature_file}\'.'

    def set_regions_file(self, regions_file):
        """
        Set the regions file to be used for region growing.
        Args:
            regions_file (str): Path to the regions file.
        Returns:
            bool: True if the regions file was set successfully, False otherwise.
        """
        if regions_file == '':
            self.regions_file = None
            self.regions_dict = None
            return True, 'Regions file cleared.'

        if not os.path.exists(regions_file):
            return False, f'Regions file does not exist: \'{regions_file}\'.'

        with open(regions_file, 'r') as f:
            self.regions_dict = json.load(f)

        self.regions_file = regions_file

        return True, f'Regions file set to \'{regions_file}\'.'

    def region_growth(self):
        if self.curvatures_file is None:
            return False, 'No curvature file loaded. Please run curvature estimation first.'

        regions_dict = {'regions': {}}

        clusters, noise_points = self.rg.segment(self.point_cloud)

        for i, cluster in enumerate(clusters):
            regions_dict['regions'][i] = {'points': cluster}
            regions_dict['noise_points'] = noise_points

        regions_dict['order'] = list(regions_dict['regions'].keys())

        self.regions_file = self.save_regions_dict(regions_dict)
        self.regions_dict = regions_dict

        return True, self.regions_file

    def save_regions_dict(self, regions_dict):
        # Save the regions to a json file named after the point cloud curvature file stripped of the  .npy extension
        point_cloud_dir = self.point_cloud_file.rsplit('/', 1)[0]
        point_cloud_file = self.point_cloud_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        regions_file = point_cloud_dir + '/' + point_cloud_file + '_' + \
            str(self.region_growing_config.seed_threshold) + '_' + \
            str(self.region_growing_config.min_cluster_size) + '_' + \
            str(self.region_growing_config.max_cluster_size) + '_' + \
            str(self.region_growing_config.curvature_threshold) + '_' + \
            str(self.region_growing_config.curvature_threshold) + '_' + \
            str(self.region_growing_config.normal_angle_threshold) + '_' + \
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + \
            '_viewpoints.json'
        # Create the directory if it does not exist
        if not os.path.exists(point_cloud_dir):
            os.makedirs(point_cloud_dir)
        # Save the region dictionary to a json file
        with open(regions_file, 'w') as f:
            json.dump(regions_dict, f, indent=4)
        print(f'Regions saved to {regions_file}.')
        return regions_file

    def fov_clustering(self):
        """
        Perform FOV clustering on the regions obtained from region growing.
        Returns:
            bool: True if FOV clustering was successful, False otherwise.
        """
        if self.regions_file is None:
            return False, 'No region file loaded. Please run region growth first.'

        # Iterate through regions in the region dictionary
        for region_id, region in self.regions_dict['regions'].items():
            region_point_cloud = self.point_cloud.select_by_index(
                region['points'])
            # Perform FOV clustering
            fov_clusters = self.fc.fov_clustering(region_point_cloud)
            self.regions_dict['regions'][region_id]['clusters'] = {}
            for i, fov_cluster in enumerate(fov_clusters):
                self.regions_dict['regions'][region_id]['clusters'][i] = {
                    'points': fov_cluster}

            # Assign default order
            self.regions_dict['regions'][region_id]['order'] = list(
                self.regions_dict['regions'][region_id]['clusters'].keys())

        self.regions_file = self.save_regions_dict(self.regions_dict)

        return True, self.regions_file

    def project_viewpoints(self):
        """
        Project viewpoints for each FOV cluster in the regions.
        Returns:
            bool: True if viewpoint projection was successful, False otherwise.
        """
        if self.regions_file is None:
            return False, 'No region file loaded. Please run region growth first.'

        # Iterate through regions in the region dictionary
        for region_id, region in self.regions_dict['regions'].items():
            if 'clusters' not in region:
                continue

            region_points = self.point_cloud.select_by_index(
                region['points'])

            for fov_cluster_id, fov_cluster in region['clusters'].items():
                fov_points = region_points.select_by_index(
                    fov_cluster['points'])
                # Project viewpoint for the FOV cluster
                origin, position, direction, orientation = self.vp.project(
                    np.asarray(fov_points.points), np.asarray(fov_points.normals))
                # Store the viewpoint in the region dictionary
                self.regions_dict['regions'][region_id]['clusters'][fov_cluster_id]['viewpoint'] = {
                    'origin': origin.tolist(),
                    'position': position.tolist(),
                    'direction': direction.tolist(),
                    'orientation': orientation.tolist()
                }
                # Visualize the projected viewpoint
                # viewpoint_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                # viewpoint_mesh.translate(viewpoint)
                # viewpoint_mesh.paint_uniform_color((1, 0, 0))  # Red color
                # o3d.visualization.draw_geometries(
                #     [viewpoint_mesh, fov_points],
                #     window_name=f'Region {region_id} FOV Cluster {fov_cluster_id} Viewpoint Projection',
                #     mesh_show_back_face=True
                # )

        # Save the updated regions dictionary with viewpoints
        self.regions_file = self.save_regions_dict(self.regions_dict)

        return True, self.regions_file

    def get_viewpoint(self, region_index, cluster_index):
        """
        Get the viewpoint for a specific region and cluster.
        Args:
            region_index (int): Index of the region.
            cluster_index (int): Index of the cluster.
        Returns:
            dict: Viewpoint information including origin, viewpoint, and direction.
        """
        if self.regions_file is None:
            return None, 'No regions file loaded. Please run region growth first.'

        if self.regions_dict is None or 'regions' not in self.regions_dict:
            return None, 'No regions found in the regions dictionary.'

        if region_index < 0 or region_index >= len(self.regions_dict['regions']):
            return None, f'Invalid region index: {region_index}.'

        if 'clusters' not in self.regions_dict['regions'][str(region_index)]:
            return None, f'No clusters found for region index: {region_index}.'

        if cluster_index < 0 or cluster_index >= len(self.regions_dict['regions'][str(region_index)]['clusters']):
            return None, f'Invalid cluster index: {cluster_index}.'

        viewpoint = self.regions_dict['regions'][str(
            region_index)]['clusters'][str(cluster_index)].get('viewpoint', None)

        if viewpoint is None:
            return None, 'No viewpoint found for the specified region and cluster.'

        return viewpoint, 'Viewpoint retrieved successfully.'

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
