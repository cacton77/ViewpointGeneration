import os
import time
import math
import numpy as np
import open3d as o3d

from open3d.geometry import PointCloud, TriangleMesh


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

    num_neighbors = 30

    fov_height = 0.02
    fov_width = 0.03
    dof = 0.02

    visualize = True
    viewer = o3d.visualization.Visualizer()
    is_running = False

    def __init__(self):
        pass

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

            # Visualize if true
            if self.visualize:
                bb = mesh.get_axis_aligned_bounding_box()
                bb.color = (0, 1, 0)
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
                text.paint_uniform_color((0, 1, 0))
                text.translate(bb_bottom_front_left +
                               [0, -2*text_scale*text_height, 0])

                self.viewer.create_window(
                    'Triangle Mesh', width=800, height=600)
                self.viewer.clear_geometries()
                self.viewer.add_geometry(mesh)
                self.viewer.add_geometry(text)
                self.viewer.add_geometry(bb)
                opt = self.viewer.get_render_option()
                opt.show_coordinate_frame = True
                opt.background_color = (0.1, 0.1, 0.1)
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
            bb.color = (0, 1, 0)
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
            text.paint_uniform_color((0, 1, 0))
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
            opt.background_color = (0.1, 0.1, 0.1)
            opt.mesh_show_back_face = True
            opt.point_show_normal = True
            self.viewer.run()
            self.viewer.destroy_window()
            self.viewer.clear_geometries()

        self.point_cloud_file = point_cloud_file
        self.pcd = pcd

        return True

    def sample_point_cloud(self):
        # Perform poisson disk sampling on the triangle mesh
        # and generate a point cloud
        if self.mesh is None:
            return False, 'No triangle mesh loaded.'
        elif self.is_running:
            return False, 'Point cloud partitioning is running.'

        N_points = self.mesh.get_surface_area() * (self.ppsqmm * 1e6)
        print('Number of points to sample:', N_points)

        # Save the sampled point cloud to a file under a directory named after the mesh file in the same directory as the mesh file
        mesh_dir = self.triangle_mesh_file.rsplit('/', 1)[0]
        print(mesh_dir)
        mesh_name = self.triangle_mesh_file.rsplit(
            '/', 1)[-1].rsplit('.', 1)[0]
        pcd_dir = mesh_dir + '/' + mesh_name + '_pcd'
        # Name the pcd file after the mesh file name with N_points appended and save as a ply file
        pcd_file = pcd_dir + '/' + mesh_name + \
            '_pcd_' + str(int(N_points)) + '.ply'
        # Create the directory if it does not exist
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)

        # Check if the point cloud file already exists
        if os.path.exists(pcd_file):
            message = f'Point cloud file already exists. Loaded {pcd_file}.'
        else:
            # Sample the point cloud
            pcd = self.mesh.sample_points_poisson_disk(
                number_of_points=int(N_points), init_factor=5)
            # Save the point cloud to a file
            o3d.io.write_point_cloud(pcd_file, pcd)
            message = f'Point cloud file saved to {pcd_file}.'

        self.set_point_cloud_file(pcd_file, 'm')
        self.run_region_growth()

        return True, message

    def curvature_estimation(self, nn_glob, vp=[0., 0., 0.]):
        # Estimate normals and curvature of the set point cloud
        print('Estimating normals and curvature...')

        if self.npcd is None:
            self.npcd = np.asarray(self.pcd.points)

        viewpoint = np.array(vp)
        # datastructure to store normals and curvature
        normals = np.empty(np.shape(self.npcd), dtype=np.float32)
        curvature = np.empty((len(self.npcd), 1), dtype=np.float32)

        # loop through the point cloud to estimate normals and curvature
        for index in range(len(self.npcd)):
            # access the points in the vicinity of the current point and store in the nn_loc variable
            nn_loc = self.npcd[nn_glob[index]]
            # calculate the covariance matrix of the points in the vicinity
            COV = np.cov(nn_loc, rowvar=False)
            # calculate the eigenvalues and eigenvectors of the covariance matrix
            eigval, eigvec = np.linalg.eig(COV)
            # sort the eigenvalues in ascending order
            idx = np.argsort(eigval)
            # store the normal of the point in the normals variable
            nor = eigvec[:, idx][:, 0]
            # check if the normal is pointing towards the viewpoint
            if nor.dot((viewpoint - self.npcd[index, :])) > 0:
                normals[index] = nor
            else:
                normals[index] = -nor
            # store the curvature of the point in the curv variable
            curvature[index] = eigval[idx][0] / np.sum(eigval)

        # Print maximum and minimum curvature values
        max_curvature = np.max(curvature)
        min_curvature = np.min(curvature)
        if self.visualize:
            # Normalize the curvature values to the range [0, 1]
            normalized_curvature = (curvature - min_curvature) / \
                (max_curvature - min_curvature)
            # Set the color of the point cloud based on the curvature values
            self.pcd.paint_uniform_color((0, 0, 0))
            for i in range(len(self.npcd)):
                np.asarray(self.pcd.colors)[i][0] = normalized_curvature[i]
            # Visualize the point cloud with the curvature values
            self.viewer.create_window(
                'Curvature', width=800, height=600)
            self.viewer.clear_geometries()
            self.viewer.add_geometry(self.pcd)
            opt = self.viewer.get_render_option()
            opt.show_coordinate_frame = True
            opt.background_color = (0.1, 0.1, 0.1)
            opt.mesh_show_back_face = True
            opt.point_show_normal = True
            self.viewer.run()
            self.viewer.destroy_window()
            self.viewer.clear_geometries()

        return normals, curvature

    def run_region_growth(self, theta_th='auto', cur_th='auto'):

        if self.npcd is None:
            self.npcd = np.asarray(self.pcd.points)

        # store point cloud as numpy array
        unique_rows = np.asarray(self.pcd.points)
        # Generate a KDTree object
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

        search_results = []

        # search for nearest neighbors for each point in the point cloud and store the k value, index of the nearby points and their distances them in search_results
        for point in self.pcd.points:
            try:
                result = pcd_tree.search_knn_vector_3d(
                    point, self.num_neighbors)
                search_results.append(result)
            except RuntimeError as e:
                print(f"An error occurred with point {point}: {e}")
                continue

        # separate the k and index values from the search_results

        k_values = [result[0] for result in search_results]
        nn_glob = [result[1] for result in search_results]
        distances = [result[2] for result in search_results]

        # Estimate normals and curvature
        # time and print this operation
        start = time.time()
        normals, curvature = self.curvature_estimation(nn_glob=nn_glob)
        end = time.time()
        print("Time taken to estimate normals and curvature: ", end-start)
        # return a list of indices that would sort the curvature array, pointcloud
        order = curvature[:, 0].argsort().tolist()
        regions = []
        cur_th = 'auto'
        # Set default values for theta_th and cur_th
        if theta_th == 'auto':
            theta_th = 15.0 / 180.0 * math.pi  # in radians
        if cur_th == 'auto':
            cur_th = np.percentile(curvature, 98)
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
                nn_loc = nn_glob[seed_cur[seedval]]
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
                            if curvature[nn_cur] < cur_th:
                                seed_cur.append(nn_cur)
                # increment the seed value
                seedval += 1
            # append the region_cur list to the region list
            regions.append(region_cur)
        # return the region list which contains the indices of the points in each region

        # Visualize the regions
        if self.visualize:
            colors = np.random.rand(len(regions), 3)
            for i, region in enumerate(regions):
                print(f'Region {i} has {len(region)} points.')
                for point_index in region:
                    np.asarray(self.pcd.colors)[point_index] = colors[i]

            bb = self.pcd.get_axis_aligned_bounding_box()
            bb.color = (0, 1, 0)
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
            text.paint_uniform_color((0, 1, 0))
            text.translate(bb_bottom_front_left +
                           [0, -2*text_scale*text_height, 0])

            self.viewer.create_window(
                'Regions', width=800, height=600)
            self.viewer.clear_geometries()
            self.viewer.add_geometry(self.pcd)
            self.viewer.add_geometry(text)
            self.viewer.add_geometry(bb)
            opt = self.viewer.get_render_option()
            opt.show_coordinate_frame = True
            opt.background_color = (0.1, 0.1, 0.1)
            opt.mesh_show_back_face = True
            self.viewer.run()
            self.viewer.destroy_window()

        return regions
