import os
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


class NPCD:
    def __init__(self, points, normals, colors):
        # Numpy array version of Open3D PointCloud
        self.points = points
        self.normals = normals
        self.colors = colors

    @classmethod
    def from_o3d_point_cloud(cls, pcd):
        # Create a new NPCD object from an Open3D PointCloud object
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        colors = np.asarray(pcd.colors)
        return NPCD(points, normals, colors)

    def select_by_index(self, indices):
        # Create a new NPCD object with only the selected indices
        points = self.points[indices]
        normals = self.normals[indices]
        colors = self.colors[indices]
        return NPCD(points, normals, colors)

    def get_o3d_point_cloud(self):
        pcd = PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd


class Partitioner():

    triangle_mesh_file = None
    point_cloud_file = None

    mesh = None
    pcd = None
    npcd = None

    ppsqmm = 100

    # Region Growth Parameters
    nn_glob = None
    curvature_cmap = 'RdYlGn'
    curvature = None  # Will be set after estimating curvature
    rg_curvature_threshold = 50  # percentile of curvature values
    rg_angle_threshold = 15.0  # in degrees
    rg_num_neighbors = 30 # Number of nearest neighbors to consider for curvature estimation and region growing
    planar_region_cmap = 'plasma'

    fov_height = 0.02
    fov_width = 0.03
    dof = 0.02

    visualize = True
    cuda_enabled = False  # Set to True if using CuPy for GPU acceleration
    mesh_color = (0.5, 0.5, 0.5)
    background_color = (0.1, 0.1, 0.1)
    bb_color = (1., 1., 1.)
    text_color = (1., 1., 1.)
    viewer = o3d.visualization.Visualizer()
    is_running = False

    def __init__(self):
        # gui.Application.instance.initialize()
        # self.window = gui.Application.instance.create_window(
        #     "Viewpoint Generation", 800, 600)
        # self.scene_widget = gui.SceneWidget()
        # self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        # self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        # self.scene_widget.set_on_mouse(self._on_mouse_event)
        # self.scene_widget.set_on_key(self._on_key_event)
        # self.window.add_child(self.scene_widget)
        pass

    def set_cuda_enabled(self, enabled):
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

    def set_triangle_mesh_file(self, triangle_mesh_file, units):
        if triangle_mesh_file is None:
            print('No triangle mesh file provided.')
            return False

        if triangle_mesh_file is not self.triangle_mesh_file:
            if triangle_mesh_file == '':
                print('No triangle mesh file provided.')
                return False

            try:
                mesh = o3d.io.read_triangle_mesh(triangle_mesh_file)
            except:
                print('Could not load requested triangle mesh file.')
                return False

            # Check if the mesh is empty
            if mesh.is_empty():
                print('The loaded triangle mesh is empty.')
                return False

            # Update the triangle mesh file
            self.triangle_mesh_file = triangle_mesh_file

            mesh.compute_vertex_normals()
            # Estimate normals if not already present
            if not mesh.has_vertex_normals():
                print('Vertex normals were not present. They have been computed.')
            else:
                print('Vertex normals are present.')

            # Scale the mesh to meters
            if units == 'cm':
                mesh.scale(0.01, center=mesh.get_center())
                print('Triangle mesh scaled to meters.')
            elif units == 'mm':
                mesh.scale(0.001, center=mesh.get_center())
                print('Triangle mesh scaled to meters.')
            elif units == 'm':
                print('Triangle mesh is already in meters.')
            elif units == 'in':
                mesh.scale(0.0254, center=mesh.get_center())
                print('Triangle mesh scaled to meters.')
            else:
                print('Unknown units. Triangle mesh not scaled.')
                return False

            # Check if the mesh has colors
            mesh.paint_uniform_color(self.mesh_color)

            # Visualize if true
            if self.visualize:
                bb = mesh.get_axis_aligned_bounding_box()
                bb.color = self.bb_color
                bb_width = (bb.get_max_bound(
                )[0] - bb.get_min_bound()[0]).round(3)
                bb_depth = (bb.get_max_bound(
                )[1] - bb.get_min_bound()[1]).round(3)
                bb_height = (bb.get_max_bound(
                )[2] - bb.get_min_bound()[2]).round(3)
                bb_bottom_front_left = bb.get_box_points()[0]
                text = o3d.t.geometry.TriangleMesh.create_text(
                    f"width: {bb_width}m, depth: {bb_depth}m, height: {bb_height}m").to_legacy()
                text_bb = text.get_axis_aligned_bounding_box()
                text_height = text_bb.get_max_bound(
                )[1] - text_bb.get_min_bound()[1]
                text_width = text_bb.get_max_bound(
                )[0] - text_bb.get_min_bound()[0]
                text_scale = 2 * bb_width / text_width
                text.scale(text_scale, center=(0, 0, 0))
                text.paint_uniform_color(self.text_color)
                text.translate(bb_bottom_front_left +
                               [0, -2*text_scale*text_height, 0])

                self.viewer.create_window(
                    'Triangle Mesh', width=800, height=600)
                self.viewer.clear_geometries()
                self.viewer.add_geometry(mesh)
                self.viewer.add_geometry(text)
                self.viewer.add_geometry(bb)

                # self.scene_widget.scene.add_geometry(
                #     'TriangleMesh', mesh, rendering.MaterialRecord())
                # self.scene_widget.scene.add_geometry('BoundingBox',
                #                                      bb, rendering.MaterialRecord())
                # self.scene_widget.scene.add_geometry(
                #     'Text', text, rendering.MaterialRecord())
                # self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
                #     size=2.0)
                # self.scene_widget.scene.add_geometry(
                #     'Axes', self.axes, rendering.MaterialRecord())
                # self.scene_widget.scene.camera.look_at(
                #     mesh.get_center() + np.array([0, 0, 1]),
                #     mesh.get_center(),
                #     np.array([0, -1, 0])
                # )

                opt = self.viewer.get_render_option()
                opt.show_coordinate_frame = True
                opt.background_color = self.background_color
                opt.mesh_show_back_face = True
                self.viewer.run()
                self.viewer.destroy_window()

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
            pcd.scale(0.01, center=pcd.get_center())
            print('Point cloud scaled to meters.')
        elif point_cloud_units == 'mm':
            pcd.scale(0.001, center=pcd.get_center())
            print('Point cloud scaled to meters.')
        elif point_cloud_units == 'm':
            print('Point cloud is already in meters.')
        else:
            print('Unknown units. Point cloud not scaled.')
            return False

        # Visualize if true
        if self.visualize:
            bb = pcd.get_axis_aligned_bounding_box()
            bb.color = self.bb_color
            bb_width = (bb.get_max_bound(
            )[0] - bb.get_min_bound()[0]).round(3)
            bb_depth = (bb.get_max_bound(
            )[1] - bb.get_min_bound()[1]).round(3)
            bb_height = (bb.get_max_bound(
            )[2] - bb.get_min_bound()[2]).round(3)
            bb_bottom_front_left = bb.get_box_points()[0]
            # Set text_string to number of points in the point cloud
            text_string = f"N points: {len(pcd.points)}"
            text = o3d.t.geometry.TriangleMesh.create_text(
                text_string).to_legacy()
            text_bb = text.get_axis_aligned_bounding_box()
            text_height = text_bb.get_max_bound(
            )[1] - text_bb.get_min_bound()[1]
            text_width = text_bb.get_max_bound(
            )[0] - text_bb.get_min_bound()[0]
            text_scale = 2 * bb_width / text_width
            text.scale(text_scale, center=(0, 0, 0))
            text.paint_uniform_color(self.text_color)
            text.translate(bb_bottom_front_left +
                           [0, -2*text_scale*text_height, 0])

            self.viewer.create_window(
                'Point Cloud', width=800, height=600)
            self.viewer.clear_geometries()
            self.viewer.add_geometry(pcd)
            self.viewer.add_geometry(text)
            self.viewer.add_geometry(bb)

            opt = self.viewer.get_render_option()
            opt.show_coordinate_frame = True
            opt.background_color = self.background_color
            opt.mesh_show_back_face = True
            # opt.point_show_normal = True
            self.viewer.run()
            self.viewer.destroy_window()
            self.viewer.clear_geometries()

        self.point_cloud_file = point_cloud_file
        self.pcd = pcd
        self.npcd = None
        self.nn_glob = None  # Reset nearest neighbors

        return True

    def set_point_cloud_units(self, units):
        if self.point_cloud_file is None:
            return False, 'No point cloud file loaded.'

        if units == 'cm':
            self.pcd.scale(0.01, center=self.pcd.get_center())
            msg = 'Point cloud scaled to centimeters.'
        elif units == 'mm':
            self.pcd.scale(0.001, center=self.pcd.get_center())
            msg = 'Point cloud scaled to millimeters.'
        elif units == 'm':
            msg = 'Point cloud is already in meters.'
        elif units == 'in':
            self.pcd.scale(0.0254, center=self.pcd.get_center())
            msg = 'Point cloud scaled to meters.'
        else:
            return False, 'Unknown units. Point cloud not scaled.'

        return True, msg

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

    def set_number_of_points(self, N_points):
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
        print('Number of points to sample:', N_points)

        # Save the sampled point cloud to a file under a directory named after the mesh file in the same directory as the mesh file
        mesh_dir = self.triangle_mesh_file.rsplit('/', 1)[0]
        mesh_name = self.triangle_mesh_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        pcd_dir = mesh_dir + '/' + mesh_name + '_pcd'
        # Name the pcd file after the mesh file name with N_points appended and save as a ply file
        pcd_file = pcd_dir + '/' + mesh_name + \
            '_pcd_' + str(int(N_points)) + 'points.ply'
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

        return True, message, pcd_file

    def set_number_of_neighbors(self, k):
        if k <= 0:
            return False, 'Number of neighbors must be greater than 0.'
        self.rg_num_neighbors = k
        self.nn_glob = None  # Reset nearest neighbors
        return True

    def find_nearest_neighbors(self):
        print('Finding nearest neighbors...')

        # Generate a KDTree object for the point cloud
        if self.pcd is None:
            print('No point cloud loaded.')
            return None

        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

        # Search for nearest neighbors for each point in the point cloud
        search_results = []
        for point in self.pcd.points:
            try:
                result = pcd_tree.search_knn_vector_3d(
                    point, self.rg_num_neighbors)
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
        if self.pcd is None:
            print('No point cloud loaded.')
            return None
        if not self.pcd.has_normals():
            print('Estimating normals for the point cloud.')
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))

        if self.nn_glob is None:
            self.find_nearest_neighbors()

        # Check if curvature file already exists
        pcd_dir = self.point_cloud_file.rsplit('/', 1)[0]
        pcd_name = self.point_cloud_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        curvature_file = pcd_dir + '/' + pcd_name + '_curvature.npy'
        curvature_file = f'{pcd_dir}/{pcd_name}_{self.rg_num_neighbors}nn_curvature.npy'
        if os.path.exists(curvature_file):
            print(f'Curvature file already exists: {curvature_file}')
            # Load the curvature values from the file
            self.curvature = np.load(curvature_file)
            print('Curvature values loaded from file.')
            return self.curvature

        # Time the curvature estimation
        start_time = time.time()
        if self.cuda_enabled:
            curvature = estimate_curvature_optimized(
                self.pcd, nn_glob=self.nn_glob)
        else:
            curvature = estimate_curvature(self.pcd, nn_glob=self.nn_glob)
        end_time = time.time()
        print(
            f'Curvature estimation took {end_time - start_time:.2f} seconds.')

        # Print maximum and minimum curvature values
        max_curvature = np.max(curvature)
        min_curvature = np.min(curvature)
        if self.visualize:
            # Normalize the curvature values to the range [0, 1]
            normalized_curvature = (curvature - min_curvature) / \
                (max_curvature - min_curvature)
            # Set the color of the point cloud based on the curvature values
            self.pcd.paint_uniform_color((0, 0, 0))
            cmap = colormaps[self.curvature_cmap]
            for i in range(len(curvature)):
                val = 1 - normalized_curvature[i]
                color = np.array(list(cmap(val)))[0, 0:3]  # Get RGB values
                np.asarray(self.pcd.colors)[i] = color

            bb = self.pcd.get_axis_aligned_bounding_box()
            bb.color = self.bb_color
            bb_width = (bb.get_max_bound(
            )[0] - bb.get_min_bound()[0]).round(3)
            bb_depth = (bb.get_max_bound(
            )[1] - bb.get_min_bound()[1]).round(3)
            bb_height = (bb.get_max_bound(
            )[2] - bb.get_min_bound()[2]).round(3)
            bb_bottom_front_left = bb.get_box_points()[0]
            # Set text_string to number of points in the point cloud
            text_string = f"Number of Neighbors: {self.rg_num_neighbors}, Min Curvature: {min_curvature:.4f}, Max Curvature: {max_curvature:.4f}"
            text = o3d.t.geometry.TriangleMesh.create_text(
                text_string).to_legacy()
            text_bb = text.get_axis_aligned_bounding_box()
            text_height = text_bb.get_max_bound(
            )[1] - text_bb.get_min_bound()[1]
            text_width = text_bb.get_max_bound(
            )[0] - text_bb.get_min_bound()[0]
            text_scale = 2 * bb_width / text_width
            text.scale(text_scale, center=(0, 0, 0))
            text.paint_uniform_color(self.text_color)
            text.translate(bb_bottom_front_left +
                           [0, -2*text_scale*text_height, 0])

            # Visualize the point cloud with the curvature values
            self.viewer.create_window(
                'Curvature', width=800, height=600)
            self.viewer.clear_geometries()
            self.viewer.add_geometry(self.mesh)
            self.viewer.add_geometry(self.pcd)
            # self.viewer.add_geometry(bb)
            self.viewer.add_geometry(text)
            opt = self.viewer.get_render_option()
            opt.show_coordinate_frame = True
            opt.background_color = self.background_color
            opt.mesh_show_back_face = True
            # opt.point_show_normal = True
            self.viewer.run()
            self.viewer.destroy_window()
            self.viewer.clear_geometries()

        # Save curvature values to disk named after the point cloud file - .ply
        pcd_dir = self.point_cloud_file.rsplit('/', 1)[0]
        pcd_name = self.point_cloud_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        curvature_file = pcd_dir + '/' + pcd_name + '_curvature.npy'
        curvature_file = f'{pcd_dir}/{pcd_name}_{self.rg_num_neighbors}nn_curvature.npy'
        # Create the directory if it does not exist
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)
        # Save the curvature values to a file
        np.save(curvature_file, curvature)
        print(f'Curvature values saved to {curvature_file}.')

        self.curvature = curvature

    def region_growth(self, theta_th='auto', cur_th='auto', min_region_size=1):

        if self.npcd is None:
            self.npcd = np.asarray(self.pcd.points)

        # Estimate normals and curvature
        if self.curvature is None:
            self.estimate_curvature()

        normals = np.asarray(self.pcd.normals)

        # return a list of indices that would sort the curvature array, pointcloud
        order = self.curvature[:, 0].argsort().tolist()
        regions = []
        cur_th = 'auto'
        # Set default values for theta_th and cur_th
        if theta_th == 'auto':
            theta_th = self.rg_curvature_threshold / 180.0 * math.pi  # in radians
        if cur_th == 'auto':
            cur_th = np.percentile(self.curvature, self.rg_curvature_threshold)
        # Perform region growing
        # Loop through the points in the point cloud until all points are assigned to a region
        while len(order) > 0:
            region_cur = []
            seed_cur = []
            # Get the curvature value of the first point of minimum curvature
            poi_min = order[0]
            region_cur.append(poi_min)
            seedval = 0
            # Add the first point index which is the index of the point of minimum curvature to the seed_cur list
            seed_cur.append(poi_min)
            # Remove the index point of minimum curvature from the order list
            order.remove(poi_min)
            # Loop through the seed_cur list until all indexes points in the seed_cur list are assigned to a region
            while seedval < len(seed_cur):
                # Get the nearest neighbors of the current seed point
                nn_loc = self.nn_glob[seed_cur[seedval]]
                # Loop through the nearest neighbors
                for j in range(len(nn_loc)):
                    # Get the current nearest neighbor index looped through the list of nearest neighbors
                    nn_cur = nn_loc[j]
                    if nn_cur in order:  # Check if nn_cur is in order
                        # find the angle between the normals of the current seed point and the current nearest neighbor
                        dot_product = np.dot(
                            normals[seed_cur[seedval]], normals[nn_cur])
                        angle = np.arccos(np.abs(dot_product))

                        # check for the angle threshold
                        if angle < theta_th:
                            # add the current nearest neighbor to the region_cur list
                            region_cur.append(nn_cur)
                            # remove the current nearest neighbor from the order list
                            order.remove(nn_cur)
                            # check for the curvature threshold
                            if self.curvature[nn_cur] < cur_th:
                                seed_cur.append(nn_cur)
                # increment the seed value
                seedval += 1

            # Only keep regions above minimum size
            if len(region_cur) >= min_region_size:
                regions.append(region_cur)
            else:
                # Put small regions back in order for potential inclusion in other regions
                for pt in region_cur:
                    if pt not in order:
                        # Insert back in sorted order
                        curvature_val = self.curvature[pt, 0]
                        insert_pos = 0
                        for i, ordered_pt in enumerate(order):
                            if self.curvature[ordered_pt, 0] > curvature_val:
                                insert_pos = i
                                break
                            insert_pos = i + 1
                        order.insert(insert_pos, pt)
            # append the region_cur list to the region list
            regions.append(region_cur)

        # Filter out regions with less than 3 points
        regions = [region for region in regions if len(region) > 2]

        # return the region list which contains the indices of the points in each region

        # Visualize the regions
        if self.visualize:
            for i, region in enumerate(regions):
                print(f'Region {i} has {len(region)} points.')
                # Generate random color from self.planar_region_cmap
                cmap = colormaps[self.planar_region_cmap]
                color = np.array(cmap(random.random()))[:3]
                for point_index in region:
                    np.asarray(self.pcd.colors)[point_index] = color

            bb = self.pcd.get_axis_aligned_bounding_box()
            bb.color = self.bb_color
            bb_width = (bb.get_max_bound(
            )[0] - bb.get_min_bound()[0]).round(3)
            bb_depth = (bb.get_max_bound(
            )[1] - bb.get_min_bound()[1]).round(3)
            bb_height = (bb.get_max_bound(
            )[2] - bb.get_min_bound()[2]).round(3)
            bb_bottom_front_left = bb.get_box_points()[0]
            # Set text_string to number of points in the point cloud
            text_string = f"N points: {len(self.pcd.points)}"
            text = o3d.t.geometry.TriangleMesh.create_text(
                text_string).to_legacy()
            text_bb = text.get_axis_aligned_bounding_box()
            text_height = text_bb.get_max_bound(
            )[1] - text_bb.get_min_bound()[1]
            text_width = text_bb.get_max_bound(
            )[0] - text_bb.get_min_bound()[0]
            text_scale = 2 * bb_width / text_width
            text.scale(text_scale, center=(0, 0, 0))
            text.paint_uniform_color(self.text_color)
            text.translate(bb_bottom_front_left +
                           [0, -2*text_scale*text_height, 0])

            self.viewer.create_window(
                'Regions', width=800, height=600)
            self.viewer.clear_geometries()
            self.viewer.add_geometry(self.pcd)
            self.viewer.add_geometry(text)
            # self.viewer.add_geometry(bb)
            opt = self.viewer.get_render_option()
            opt.show_coordinate_frame = True
            opt.background_color = self.background_color
            opt.mesh_show_back_face = True
            self.viewer.run()
            self.viewer.destroy_window()

        return regions

    def _on_mouse_event(self, event):
        # Force refresh after mouse interaction
        gui.Application.instance.post_to_main_thread(
            self.window, self.refresh_scene)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key_event(self, event):
        # Force refresh after key interaction
        gui.Application.instance.post_to_main_thread(
            self.window, self.refresh_scene)
        return gui.Widget.EventCallbackResult.IGNORED
