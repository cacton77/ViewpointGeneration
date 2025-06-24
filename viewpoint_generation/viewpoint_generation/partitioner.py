import os
import json
import time
import math
import random
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from matplotlib import colormaps
from open3d.geometry import PointCloud, TriangleMesh

from viewpoint_generation.curvature import *
from viewpoint_generation.region_growth import *


class Partitioner():

    mesh_file = None
    mesh_units = 'm'
    point_cloud_file = None
    point_cloud_units = 'm'
    curvature_file = None
    region_file = None

    mesh = None
    pcd = None
    npcd = None

    ppsqmm = 100

    # Region Growth Parameters
    nn_glob = None
    curvature_cmap = 'RdYlGn'
    curvature = None  # Will be set after estimating curvature
    rg_curvature_threshold = 0.5  # percentile of curvature values
    rg_angle_threshold = 15.0  # in degrees
    rg_min_region_size = 3  # Minimum number of points in a region
    # Number of nearest neighbors to consider for curvature estimation and region growing
    curvature_num_neighbors = 30
    planar_region_cmap = 'plasma'
    region_dict = {}

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

    def set_mesh_file(self, mesh_file, units):
        if mesh_file is None:
            print('No triangle mesh file provided.')
            return False

        if mesh_file is not self.mesh_file:
            if mesh_file == '':
                print('No triangle mesh file provided.')
                return False

            try:
                mesh = o3d.io.read_triangle_mesh(mesh_file)
            except Exception as e:
                print(f'Could not load requested triangle mesh file: {e}')
                return False

            # Check if the mesh is empty
            if mesh.is_empty():
                print('The loaded triangle mesh is empty.')
                return False

            # Update the triangle mesh file
            self.mesh_file = mesh_file
            self.mesh_units = units

            mesh.compute_vertex_normals()
            # Estimate normals if not already present
            if not mesh.has_vertex_normals():
                print('Vertex normals were not present. They have been computed.')
            else:
                print('Vertex normals are present.')

            # Scale the mesh to meters
            if units == 'cm':
                mesh.scale(0.01, center=(0, 0, 0))
                print('Mesh scaled to meters.')
            elif units == 'mm':
                mesh.scale(0.001, center=(0, 0, 0))
                print('Mesh scaled to meters.')
            elif units == 'm':
                print('Mesh is already in meters.')
            elif units == 'in':
                mesh.scale(0.0254, center=(0, 0, 0))
                print('Mesh scaled to meters.')
            else:
                print('Unknown units. Mesh not scaled.')
                return False

            # Check if the mesh has colors
            mesh.paint_uniform_color(self.mesh_color)

            self.mesh = mesh

        return True

    def set_point_cloud_file(self, point_cloud_file, point_cloud_units):
        if point_cloud_file is self.point_cloud_file:
            print('Point cloud file already set.')
            return True

        if point_cloud_file == '':
            print('No point cloud file provided.')
            return False

        try:
            pcd = o3d.io.read_point_cloud(point_cloud_file)
        except:
            print('Could not load requested point cloud file.')
            return False

        # Check if the point cloud is empty
        if pcd.is_empty():
            print('The loaded point cloud is empty.')
            return False
        # Check if the point cloud has normals
        if not pcd.has_normals():
            print('The point cloud does not have normals. Computing normals.')
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))
        else:
            print('The loaded point cloud has normals.')
        # Check if the point cloud has colors
        if not pcd.has_colors():
            print('The point cloud does not have colors. Setting colors to white.')
            pcd.paint_uniform_color((1, 1, 1))
        else:
            print('The loaded point cloud has colors.')

        # Scale the point cloud to meters
        if point_cloud_units == 'cm':
            pcd.scale(0.01, center=(0, 0, 0))
            print('Point cloud scaled to meters.')
        elif point_cloud_units == 'mm':
            pcd.scale(0.001, center=(0, 0, 0))
            print('Point cloud scaled to meters.')
        elif point_cloud_units == 'm':
            print('Point cloud is already in meters.')
        elif point_cloud_units == 'in':
            pcd.scale(0.0254, center=(0, 0, 0))
            print('Point cloud scaled to meters.')
        else:
            print('Unknown units. Point cloud not scaled.')
            return False

        self.point_cloud_file = point_cloud_file
        self.point_cloud = pcd
        self.point_cloud_units = point_cloud_units
        self.npcd = None
        self.nn_glob = None  # Reset nearest neighbors

        return True

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

        # Save the sampled point cloud to a file under a directory named after the mesh file in the same directory as the mesh file
        mesh_dir = self.mesh_file.rsplit('/', 1)[0]
        mesh_name = self.mesh_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        pcd_dir = mesh_dir + '/' + mesh_name + '_pcd'
        # Name the pcd file after the mesh file name with N_points appended and save as a ply file
        pcd_file = pcd_dir + '/' + mesh_name + '_' + \
            self.mesh_units + '_pcd_' + str(int(N_points)) + 'points.ply'
        # Create the directory if it does not exist
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)

        # Check if the point cloud file already exists
        if os.path.exists(pcd_file):
            message = f'Point cloud file already exists. Loaded {pcd_file}.'
        else:
            # Sample the point cloud
            pcd = self.mesh.sample_points_poisson_disk(
                number_of_points=int(N_points), init_factor=5, use_triangle_normal=True)
            # Save the point cloud to a file
            o3d.io.write_point_cloud(pcd_file, pcd)
            message = f'Point cloud file saved to {pcd_file}.'

        message = pcd_file

        return True, message

    def set_curvature_number_of_neighbors(self, k):
        if k <= 0:
            return False, 'Number of neighbors must be greater than 0.'
        self.curvature_num_neighbors = k
        self.nn_glob = None  # Reset nearest neighbors
        return True

    def find_nearest_neighbors(self):
        print('Finding nearest neighbors...')

        # Generate a KDTree object for the point cloud
        if self.point_cloud is None:
            print('No point cloud loaded.')
            return None

        pcd_tree = o3d.geometry.KDTreeFlann(self.point_cloud)

        # Search for nearest neighbors for each point in the point cloud
        search_results = []
        for point in self.point_cloud.points:
            try:
                result = pcd_tree.search_knn_vector_3d(
                    point, self.curvature_num_neighbors)
                search_results.append(result)
            except RuntimeError as e:
                print(f"An error occurred with point {point}: {e}")
                continue

        # Separate the k and index values from the search_results
        k_values = [result[0] for result in search_results]
        self.nn_glob = [result[1] for result in search_results]
        distances = [result[2] for result in search_results]

        print('Nearest neighbors found.')

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

        if self.nn_glob is None:
            self.find_nearest_neighbors()

        # Check if curvature file already exists
        pcd_dir = self.point_cloud_file.rsplit('/', 1)[0]
        pcd_name = self.point_cloud_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        curvature_file = pcd_dir + '/' + pcd_name + '_curvature.npy'
        curvature_file = f'{pcd_dir}/{pcd_name}_{self.curvature_num_neighbors}nn_curvature.npy'
        if os.path.exists(curvature_file):
            print(f'Curvature file already exists: {curvature_file}')
            # Load the curvature values from the file
            self.curvature_file = curvature_file
            self.curvature = np.load(curvature_file)
            print('Curvature values loaded from file.')
            return True, curvature_file

        # Time the curvature estimation
        start_time = time.time()
        if self.cuda_enabled:
            curvature = estimate_curvature_optimized(
                self.point_cloud, nn_glob=self.nn_glob)
        else:
            curvature = estimate_curvature(
                self.point_cloud, nn_glob=self.nn_glob)
        end_time = time.time()
        print(
            f'Curvature estimation took {end_time - start_time:.2f} seconds.')

        # Save curvature values to disk named after the point cloud file - .ply
        pcd_dir = self.point_cloud_file.rsplit('/', 1)[0]
        pcd_name = self.point_cloud_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        curvature_file = pcd_dir + '/' + pcd_name + '_curvature.npy'
        curvature_file = f'{pcd_dir}/{pcd_name}_{self.curvature_num_neighbors}nn_curvature.npy'
        # Create the directory if it does not exist
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)
        # Save the curvature values to a file
        np.save(curvature_file, curvature)
        print(f'Curvature values saved to {curvature_file}.')

        self.curvature_file = curvature_file
        self.curvature = curvature

        return True, curvature_file

    def set_curvature_file(self, curvature_file):
        """
        Set the curvature file to be used for region growing.
        Args:
            curvature_file (str): Path to the curvature file.
        Returns:
            bool: True if the curvature file was set successfully, False otherwise.
        """
        if not os.path.exists(curvature_file):
            print(f'Curvature file does not exist: {curvature_file}')
            return False

        self.curvature = np.load(curvature_file)
        print(f'Curvature values loaded from {curvature_file}.')
        return True

    def set_region_growth_angle_threshold(self, angle_threshold):
        """
        Set the angle threshold for region growing.
        Args:
            angle_threshold (float): Angle threshold in degrees.
        Returns:
            bool: True if the angle threshold was set successfully, False otherwise.
        """
        if angle_threshold <= 0 or angle_threshold > 180:
            message = 'Angle threshold must be between 0 and 180 degrees.'
            return False, message

        self.rg_angle_threshold = angle_threshold * np.pi / 180.0  # Convert to radians
        message = f'Region growth angle threshold set to {angle_threshold} degrees.'
        return True, message

    def set_region_growth_curvature_threshold(self, curvature_threshold):
        """
        Set the curvature threshold for region growing.
        Args:
            curvature_threshold (float): Curvature threshold in percent.
        Returns:
            bool: True if the curvature threshold was set successfully, False otherwise.
        """
        if curvature_threshold <= 0 or curvature_threshold > 1:
            message = 'Curvature threshold must be between 0 and 1.'
            return False, message

        self.rg_curvature_threshold = curvature_threshold
        message = f'Region growth curvature threshold set to {100*curvature_threshold} percent.'
        return True, message

    def region_growth(self):
        region_dict = {'regions': {}, 'order': []}
        config = RegionGrowingConfig(
            seed_threshold=0.05,
            region_threshold=0.1,
            min_cluster_size=50,
            normal_angle_threshold=self.rg_angle_threshold,
            curvature_threshold=self.rg_curvature_threshold,
            knn_neighbors=20
        )

        rg = RegionGrowing(config)
        clusters, noise_points = rg.segment(self.point_cloud)

        for i, cluster in enumerate(clusters):
            region_dict['regions'][i] = {'points': cluster, 'viewpoint': None}
            region_dict['noise_points'] = noise_points

        # Save the regions to a json file named after the point cloud curvature file stripped of the  .npy extension
        curvature_dir = self.curvature_file.rsplit('/', 1)[0]
        curvature_name = self.curvature_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        region_file = curvature_dir + '/' + curvature_name + '_' + \
            str(self.rg_curvature_threshold) + '_' + \
            str(self.rg_angle_threshold) + '_viewpoints.json'
        # Create the directory if it does not exist
        if not os.path.exists(curvature_dir):
            os.makedirs(curvature_dir)
        # Save the region dictionary to a json file
        with open(region_file, 'w') as f:
            json.dump(region_dict, f, indent=4)
        print(f'Regions saved to {region_file}.')

        self.region_file = region_file
        self.region_dict = region_dict

        return True, region_file
