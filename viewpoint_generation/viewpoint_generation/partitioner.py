import os
import open3d as o3d


class Partitioner():

    triangle_mesh_file = None
    point_cloud_file = None

    mesh = None
    pcd = None

    ppsqmm = 100

    fov_height = 0.02
    fov_width = 0.03
    dof = 0.02

    visualize = True
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
            else:
                print('Unknown units. Triangle mesh not scaled.')
                return False

            # Visualize if true
            if self.visualize:
                o3d.visualization.draw_geometries([mesh])

            self.mesh = mesh

        return True

    def set_point_cloud_file(self, point_cloud_file, point_cloud_units):
        if point_cloud_file is not self.point_cloud_file:
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
                o3d.visualization.draw_geometries([pcd])
                return True

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
        # Sample the point cloud

        return True, 'Point cloud sampled successfully.'
