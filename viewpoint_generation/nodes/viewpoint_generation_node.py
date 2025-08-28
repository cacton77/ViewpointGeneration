import os
import time
import rclpy
import rclpy.node
import numpy as np
import open3d as o3d
from rclpy.action import ActionServer
from viewpoint_generation.viewpoint_generation import ViewpointGeneration
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_prefix
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_srvs.srv import Trigger
# from viewpoint_generation_interfaces.action import ViewpointGeneration

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
from viewpoint_generation_interfaces.srv import MoveToPoseStamped, OptimizeViewpointTraversal


class ViewpointGenerationNode(rclpy.node.Node):

    block_next_param_callback = False
    initialized = False
    moving = False

    selected_viewpoint_pose = None
    viewpoint_dict_path = None

    def __init__(self):
        node_name = 'viewpoint_generation'
        super().__init__(node_name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model.mesh.file', ''),
                ('model.mesh.units', 'm'),
                ('model.point_cloud.file', ''),
                ('model.point_cloud.units', 'm'),
                ('model.point_cloud.sampling.ppsqmm', 1.),
                ('model.point_cloud.sampling.number_of_points', 100000),
                ('model.camera.fov.width', 0.02),
                ('model.camera.fov.height', 0.03),
                ('model.camera.dof', 0.02),
                ('model.camera.focal_distance', 0.35),
                ('regions.file', ''),
                ('regions.region_growth.curvature.knn_neighbors', 30),
                ('regions.region_growth.curvature.file', ''),
                ('regions.region_growth.seed_threshold', 15.0),
                ('regions.region_growth.min_cluster_size', 10),
                ('regions.region_growth.max_cluster_size', 100000),
                ('regions.region_growth.curvature_threshold', 0.50),
                ('regions.region_growth.normal_angle_threshold', 15.0),
                ('regions.fov_clustering.lambda_weight', 1.0),
                ('regions.fov_clustering.beta_weight', 1.0),
                ('regions.fov_clustering.max_point_out_percentage', 0.001),
                ('regions.fov_clustering.k-means.point_weight', 1.0),
                ('regions.fov_clustering.k-means.normal_weight', 1.0),
                ('regions.fov_clustering.k-means.number_of_runs', 10),
                ('regions.fov_clustering.k-means.maximum_iterations', 100),
                ('regions.selected_region', 0),
                ('regions.selected_cluster', 0),
                ('viewpoints.traversal', ''),
                ('viewpoints.projection.nothing', ''),
                ('settings.cuda_enabled', False)
            ]
        )

        # Create planning scene publisher
        self.planning_scene_diff_publisher = self.create_publisher(
            PlanningScene, '/planning_scene', 10)
        self.get_logger().info('Planning scene publisher created.')

        # Viewpoint publisher for RViz2 Visualization
        self.viewpoint_publisher = self.create_publisher(
            PoseStamped, f'{node_name}/viewpoint', 10)

        # Sample PCD Service
        services_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, node_name + '/sample_point_cloud',
                            self.sample_point_cloud_callback, callback_group=services_cb_group)
        # Estimate Curvature Service
        self.create_service(Trigger, node_name + '/estimate_curvature',
                            self.estimate_curvature_callback, callback_group=services_cb_group)
        # Region Growth Service
        self.create_service(Trigger, node_name + '/region_growth',
                            self.region_growth_callback, callback_group=services_cb_group)
        # FOV Clustering Service
        self.create_service(Trigger, node_name + '/fov_clustering',
                            self.fov_clustering_callback, callback_group=services_cb_group)
        # Viewpoint Projection Service
        self.create_service(Trigger, node_name + '/viewpoint_projection',
                            self.viewpoint_projection_callback, callback_group=services_cb_group)
        # Optimize Traversal Service
        self.create_service(Trigger, node_name +
                            '/optimize_traversal', self.optimize_traversal, callback_group=services_cb_group)

        # Connect to viewpoint traversal service
        viewpoint_traversal_node_name = 'viewpoint_traversal'
        self.move_to_pose_stamped_client = self.create_client(
            MoveToPoseStamped, f'{viewpoint_traversal_node_name}/move_to_pose_stamped', callback_group=services_cb_group)
        self.optimize_traversal_client = self.create_client(
            OptimizeViewpointTraversal, f'{viewpoint_traversal_node_name}/optimize_traversal', callback_group=services_cb_group)

        # Selected viewpoint publisher timer
        self.create_timer(
            0.1, self.publish_selected_viewpoint)

        # Viewpoint Generation Helpers
        self.viewpoint_generation = ViewpointGeneration()

        self.set_mesh_file(self.get_parameter('model.mesh.file').get_parameter_value().string_value,
                           self.get_parameter('model.mesh.units').get_parameter_value().string_value)
        self.set_point_cloud_file(self.get_parameter('model.point_cloud.file').get_parameter_value().string_value,
                                  self.get_parameter('model.point_cloud.units').get_parameter_value().string_value)
        self.set_sampling_ppsqmm(self.get_parameter(
            'model.point_cloud.sampling.ppsqmm').get_parameter_value().double_value)
        self.set_sampling_number_of_points(
            self.get_parameter(
                'model.point_cloud.sampling.number_of_points').get_parameter_value().integer_value)
        self.set_knn_neighbors(self.get_parameter(
            'regions.region_growth.curvature.knn_neighbors').get_parameter_value().integer_value)
        self.set_curvature_file(self.get_parameter(
            'regions.region_growth.curvature.file').get_parameter_value().string_value)
        self.set_regions_file(self.get_parameter(
            'regions.file').get_parameter_value().string_value)
        self.set_seed_threshold(self.get_parameter(
            'regions.region_growth.seed_threshold').get_parameter_value().double_value)
        self.set_min_cluster_size(self.get_parameter(
            'regions.region_growth.min_cluster_size').get_parameter_value().integer_value)
        self.set_max_cluster_size(self.get_parameter(
            'regions.region_growth.max_cluster_size').get_parameter_value().integer_value)
        self.viewpoint_generation.fov_width = self.get_parameter(
            'model.camera.fov.width').get_parameter_value().double_value
        self.viewpoint_generation.fov_height = self.get_parameter(
            'model.camera.fov.height').get_parameter_value().double_value
        self.viewpoint_generation.dof = self.get_parameter(
            'model.camera.dof').get_parameter_value().double_value
        self.viewpoint_generation.focal_distance = self.get_parameter(
            'model.camera.focal_distance').get_parameter_value().double_value
        self.viewpoint_generation.cuda_enabled = self.get_parameter(
            'settings.cuda_enabled').get_parameter_value().bool_value

        self.select_viewpoint(self.get_parameter('regions.selected_region').get_parameter_value().integer_value,
                              self.get_parameter('regions.selected_cluster').get_parameter_value().integer_value)

        self.add_on_set_parameters_callback(self.parameter_callback)

        # Action Server
        # self._action_server = ActionServer(
        #     self,
        #     ViewpointGeneration,
        #     'viewpoint_generation',
        #     self.execute_callback)

        self.block_next_param_callback = False
        self.initialized = True

    def set_mesh_file(self, mesh_file, mesh_units):
        """
        Helper function to set the triangle mesh file for the partitioner.
        :param mesh_file: The path to the triangle mesh file.
        :return: None
        """

        # If mesh_file begins with "package://package_name", replace it with the path to the package
        if mesh_file.startswith('package://'):
            package_name, relative_path = mesh_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            mesh_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success, message = self.viewpoint_generation.set_mesh_file(
            mesh_file, mesh_units)

        if not success:
            mesh_file_param = rclpy.parameter.Parameter(
                'model.mesh.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            # self.block_next_param_callback = True
            # self.set_parameters([mesh_file_param])
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)

            mesh_file_param = rclpy.parameter.Parameter(
                'model.mesh.file',
                rclpy.Parameter.Type.STRING,
                mesh_file
            )
            mesh_units_param = rclpy.parameter.Parameter(
                'model.mesh.units',
                rclpy.Parameter.Type.STRING,
                mesh_units
            )
            self.block_next_param_callback = True
            self.set_parameters([mesh_file_param])
            self.block_next_param_callback = True
            self.set_parameters([mesh_units_param])

            if self.initialized:
                self.set_point_cloud_file(
                    point_cloud_file='', point_cloud_units='')

            # Object pose relative to 'object_frame'
            # Pose will be changed during registration
            dimensions = self.viewpoint_generation.get_mesh_dimensions()

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            # Set dimensions of the box geometry
            box.dimensions = [dimensions[0], dimensions[1], dimensions[2]]
            box_pose = Pose()
            box_pose.position.z = dimensions[2] / 2.0

            # Create a mesh from the file
            mesh = Mesh()
            mesh.triangles = []
            mesh.vertices = []
            vertices, triangles = self.viewpoint_generation.get_mesh_vertices_and_triangles()

            for vertex in vertices:
                point = Point()
                point.x = vertex[0]
                point.y = vertex[1]
                point.z = vertex[2]
                mesh.vertices.append(point)

            for triangle in triangles:
                mesh_triangle = MeshTriangle()
                mesh_triangle.vertex_indices = triangle.astype(
                    np.uint32).tolist()
                mesh.triangles.append(mesh_triangle)

            # Pose of object relative to 'object_frame'
            # Will be changed by Yusen's point cloud registration
            pose = Pose()

            # Remove object
            remove_object = CollisionObject()
            remove_object.header.frame_id = 'object_frame'
            remove_object.id = 'object'
            remove_object.operation = CollisionObject.REMOVE

            # Update planning scene with the new mesh
            attached_object = AttachedCollisionObject()
            attached_object.link_name = 'object_frame'
            attached_object.object.header.frame_id = 'object_frame'
            attached_object.object.pose = pose
            attached_object.object.id = 'object'
            attached_object.object.meshes = [mesh]
            attached_object.object.mesh_poses = [Pose()]
            attached_object.object.operation = CollisionObject.ADD
            attached_object.touch_links = [
                'turntable_disc_link', 'turntable_base_link', 'planning_volume']

            planning_scene = PlanningScene()
            planning_scene.world.collision_objects.clear()
            planning_scene.world.collision_objects.append(remove_object)
            planning_scene.robot_state.attached_collision_objects.append(
                attached_object)
            planning_scene.robot_state.is_diff = True
            planning_scene.is_diff = True

            self.planning_scene_diff_publisher.publish(planning_scene)
            self.get_logger().info(f'Planning scene updated!')

            return True

    def set_point_cloud_file(self, point_cloud_file, point_cloud_units):
        """
        Helper function to set the point cloud file for the partitioner.
        :param point_cloud_file: The path to the point cloud file.
        :return: None
        """

        # If point_cloud_file begins with "package://package_name", replace it with the path to the package
        if point_cloud_file.startswith('package://'):
            package_name, relative_path = point_cloud_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            point_cloud_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success, message = self.viewpoint_generation.set_point_cloud_file(
            point_cloud_file, point_cloud_units)

        if not success:
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([point_cloud_file_param])
            self.get_logger().error(
                f'Could not load requested point cloud file {point_cloud_file}.'
            )
            return False
        else:
            self.get_logger().info(message)
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                point_cloud_file
            )
            point_cloud_units_param = rclpy.parameter.Parameter(
                'model.point_cloud.units',
                rclpy.Parameter.Type.STRING,
                point_cloud_units
            )
            self.block_next_param_callback = True
            self.set_parameters([point_cloud_file_param])
            self.block_next_param_callback = True
            self.set_parameters([point_cloud_units_param])

            if self.initialized:
                # Clear the curvature file parameter
                self.set_curvature_file(curvature_file='')

            return True

    def set_curvature_file(self, curvature_file):
        """
        Helper function to set the curvature file for the partitioner.
        :param curvature_file: The path to the curvature file.
        :return: None
        """

        # If curvature_file begins with "package://package_name", replace it with the path to the package
        if curvature_file.startswith('package://'):
            package_name, relative_path = curvature_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            curvature_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success, message = self.viewpoint_generation.set_curvature_file(
            curvature_file)

        if not success:
            self.get_logger().error(message)

            curvature_file_param = rclpy.parameter.Parameter(
                'regions.region_growth.curvature.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([curvature_file_param])

            return False
        else:
            self.get_logger().info(message)

            curvature_file_param = rclpy.parameter.Parameter(
                'regions.region_growth.curvature.file',
                rclpy.Parameter.Type.STRING,
                curvature_file
            )
            self.block_next_param_callback = True
            self.set_parameters([curvature_file_param])

            if self.initialized:
                # Clear the regions file parameter
                self.set_regions_file(regions_file='')

            return True

    def set_regions_file(self, regions_file):
        """
        Helper function to set the regions file for the partitioner.
        :param regions_file: The path to the regions file.
        :return: None
        """

        # If regions_file begins with "package://package_name", replace it with the path to the package
        if regions_file.startswith('package://'):
            package_name, relative_path = regions_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            regions_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success, message = self.viewpoint_generation.set_regions_file(
            regions_file)

        if not success:
            self.get_logger().error(message)

            regions_file_param = rclpy.parameter.Parameter(
                'regions.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([regions_file_param])

            return False
        else:
            self.get_logger().info(message)

            regions_file_param = rclpy.parameter.Parameter(
                'regions.file',
                rclpy.Parameter.Type.STRING,
                regions_file
            )
            self.block_next_param_callback = True
            self.set_parameters([regions_file_param])

            return True

    def enable_cuda_callback(self, enable):
        """
        Helper function to enable or disable CUDA for the partitioner.
        :param enable: Boolean indicating whether to enable CUDA.
        :return: True if successful, False otherwise.
        """

        success = self.viewpoint_generation.enable_cuda(enable)

        if not success:
            self.get_logger().error(
                'Failed to enable CUDA. Please check your setup.'
            )
            return False
        else:
            if enable:
                self.get_logger().info('CUDA enabled successfully.')
            else:
                self.get_logger().info('CUDA disabled.')
            return True

    def set_sampling_ppsqmm(self, ppsqmm):
        """
        Helper function to set the points per square millimeter for the partitioner.
        :param ppsqmm: The number of points per square millimeter.
        :return: True if successful, False otherwise.
        """

        if ppsqmm <= 0:
            self.get_logger().error(
                'Points per square millimeter must be greater than 0.'
            )
            return False

        success, N_points = self.viewpoint_generation.set_ppsqmm(ppsqmm)

        if not success:
            self.get_logger().error(
                f'Failed to set points per square millimeter to {ppsqmm}.'
            )
            return False
        else:
            self.get_logger().info(
                f'Points per square millimeter set to {ppsqmm}.')
            number_of_points_param = rclpy.parameter.Parameter(
                'model.point_cloud.sampling.number_of_points',
                rclpy.Parameter.Type.INTEGER,
                N_points
            )
            self.block_next_param_callback = True
            self.set_parameters([number_of_points_param])

        return True

    def set_sampling_number_of_points(self, number_of_points):
        """
        Helper function to set the number of points to sample.
        :param number_of_points: The number of points to sample.
        :return: True if successful, False otherwise.
        """

        if number_of_points <= 0:
            self.get_logger().error(
                'Number of points must be greater than 0.'
            )
            return False

        success, ppsqmm = self.viewpoint_generation.set_sampling_number_of_points(
            number_of_points)

        if not success:
            self.get_logger().error(
                f'Failed to set number of points to {number_of_points}.'
            )
            return False
        else:
            self.get_logger().info(
                f'Number of points set to {number_of_points}.')
            ppsqmm_param = rclpy.parameter.Parameter(
                'model.point_cloud.sampling.ppsqmm',
                rclpy.Parameter.Type.DOUBLE,
                ppsqmm
            )
            self.block_next_param_callback = True
            self.set_parameters([ppsqmm_param])

        return True

    def sample_point_cloud_callback(self, request, response):
        """ Callback for the sample point cloud service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
            success (bool): True if point cloud sampling was successful, False otherwise.
            message (str): Returns the file path of the sampled point cloud if successful, or an error message if not.
        """

        self.get_logger().info('Sampling point cloud...')

        success, message = self.viewpoint_generation.sample_point_cloud()

        if success:
            self.get_logger().info(
                f"Point cloud sampled successfully. File: {message}")

            # Set the point cloud of the partitioner
            pcd_file = message
            success, message = self.viewpoint_generation.set_point_cloud_file(
                pcd_file, 'm')

            # Update the point cloud units to meters
            point_cloud_units_param = rclpy.parameter.Parameter(
                'model.point_cloud.units',
                rclpy.Parameter.Type.STRING,
                'm'
            )
            self.block_next_param_callback = True
            self.set_parameters([point_cloud_units_param])

            # Update the point cloud file parameter with the sampled file
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                pcd_file
            )

            self.set_parameters(
                [point_cloud_file_param])

        else:
            self.get_logger().error(f"Failed to sample point cloud: {message}")

        response.success = success
        response.message = message

        return response

    def set_knn_neighbors(self, number_of_neighbors):
        """
        Helper function to set the number of neighbors for curvature estimation.
        :param number_of_neighbors: The number of neighbors to use for curvature estimation.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_knn_neighbors(
            number_of_neighbors)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            curvature_file_param = rclpy.parameter.Parameter(
                'regions.region_growth.curvature.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([curvature_file_param])

        return True

    def set_seed_threshold(self, seed_threshold):
        """
        Helper function to set the seed threshold for region growth.
        :param seed_threshold: The seed threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_seed_threshold(
            seed_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_min_cluster_size(self, min_cluster_size):
        """
        Helper function to set the minimum cluster size for region growth.
        :param min_cluster_size: The minimum cluster size to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_min_cluster_size(
            min_cluster_size)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_max_cluster_size(self, max_cluster_size):
        """
        Helper function to set the maximum cluster size for region growth.
        :param max_cluster_size: The maximum cluster size to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_max_cluster_size(
            max_cluster_size)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_normal_angle_threshold(self, normal_angle_threshold):
        """
        Helper function to set the angle threshold for region growth.
        :param normal_angle_threshold: The angle threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_normal_angle_threshold(
            normal_angle_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_curvature_threshold(self, curvature_threshold):
        """
        Helper function to set the curvature threshold for region growth.
        :param curvature_threshold: The curvature threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_curvature_threshold(
            curvature_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def estimate_curvature_callback(self, request, response):
        """
        Callback for the estimate curvature service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
            success (bool): True if curvature estimation was successful, False otherwise.
            message (str): Returns the file path of the estimated curvature file if successful, or an error message if not.
        """
        self.get_logger().info('Estimating curvature...')

        success, message = self.viewpoint_generation.estimate_curvature()

        if success:
            self.get_logger().info(
                f"Curvature estimation completed successfully. Curvature file: {message}")
            # Set the curvature file parameter with the estimated curvature file
            curvature_file_param = rclpy.parameter.Parameter(
                'regions.region_growth.curvature.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([curvature_file_param])
        else:
            self.get_logger().error("Curvature estimation failed.")

        response.success = success
        response.message = message

        return response

    def region_growth_callback(self, request, response):
        """
        Callback for the region growth service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
        """
        self.get_logger().info('Performing region growth...')

        success, message = self.viewpoint_generation.region_growth()

        if success:
            self.get_logger().info(
                f"Region growth completed successfully. Regions file: {message}")
            region_file_param = rclpy.parameter.Parameter(
                'regions.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([region_file_param])
        else:
            self.get_logger().error("Region growth failed.")

        response.success = success
        response.message = message

        return response

    def set_camera_parameters(self, fov_width, fov_height, dof, focal_distance):
        success, message = self.viewpoint_generation.set_camera_parameters(
            fov_width, fov_height, dof, focal_distance)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def fov_clustering_callback(self, request, response):
        """
        Callback for the FOV clustering service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
        """
        self.get_logger().info('Performing FOV clustering...')

        success, message = self.viewpoint_generation.fov_clustering()

        if success:
            self.get_logger().info(
                f"FOV clustering completed successfully. Regions file updated: {message}")
            # Set the regions file parameter with the FOV clustering result
            regions_file_param = rclpy.parameter.Parameter(
                'regions.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([regions_file_param])
        else:
            self.get_logger().error(f"FOV clustering failed: {message}")

        response.success = success
        response.message = message

        return response

    def viewpoint_projection_callback(self, request, response):
        """
        Callback for the viewpoint projection service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
            success (bool): True if viewpoint projection was successful, False otherwise.
            message (str): Returns the file path of the projected viewpoints if successful, or an error message if not.
        """
        self.get_logger().info('Projecting viewpoints...')

        success, message = self.viewpoint_generation.project_viewpoints()

        if success:
            self.get_logger().info(
                f"Viewpoint projection completed successfully. Viewpoints file: {message}")
            # Set the viewpoints file parameter with the projected viewpoints file
            # Set the regions file parameter with the FOV clustering result
            regions_file_param = rclpy.parameter.Parameter(
                'regions.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([regions_file_param])
        else:
            self.get_logger().error(f"Viewpoint projection failed: {message}")

        response.success = success
        response.message = message

        return response

    def select_viewpoint(self, region_index, cluster_index):
        """
        Helper function to select a viewpoint based on region and cluster indices.
        :param region_index: The index of the region to select.
        :param cluster_index: The index of the cluster to select.
        :return: None
        """

        viewpoint, message = self.viewpoint_generation.get_viewpoint(
            region_index, cluster_index)

        if viewpoint is None:
            self.get_logger().error(message)
            # Reset the selected region and cluster parameters
            selected_region_param = rclpy.parameter.Parameter(
                'regions.selected_region',
                rclpy.Parameter.Type.INTEGER,
                0
            )
            selected_cluster_param = rclpy.parameter.Parameter(
                'regions.selected_cluster',
                rclpy.Parameter.Type.INTEGER,
                0
            )
            self.block_next_param_callback = True
            self.set_parameters([selected_region_param])
            self.block_next_param_callback = True
            self.set_parameters([selected_cluster_param])
            return False
        else:
            self.get_logger().info(message)
            selected_region_param = rclpy.parameter.Parameter(
                'regions.selected_region',
                rclpy.Parameter.Type.INTEGER,
                region_index
            )
            selected_cluster_param = rclpy.parameter.Parameter(
                'regions.selected_cluster',
                rclpy.Parameter.Type.INTEGER,
                cluster_index
            )
            self.block_next_param_callback = True
            self.set_parameters([selected_region_param])
            self.block_next_param_callback = True
            self.set_parameters([selected_cluster_param])

            # Unpack viewpoint dictionary into PoseStamped message
            viewpoint_pose = PoseStamped()
            viewpoint_pose.header.frame_id = 'object_frame'
            viewpoint_pose.pose.position.x = viewpoint['position'][0]
            viewpoint_pose.pose.position.y = viewpoint['position'][1]
            viewpoint_pose.pose.position.z = viewpoint['position'][2]
            viewpoint_pose.pose.orientation.x = viewpoint['orientation'][0]
            viewpoint_pose.pose.orientation.y = viewpoint['orientation'][1]
            viewpoint_pose.pose.orientation.z = viewpoint['orientation'][2]
            viewpoint_pose.pose.orientation.w = viewpoint['orientation'][3]

            self.selected_viewpoint_pose = viewpoint_pose

            return True

    def publish_selected_viewpoint(self):
        """
        Publish the currently selected viewpoint to the viewpoint topic.
        :return: None
        """
        if self.selected_viewpoint_pose is not None:
            self.viewpoint_publisher.publish(self.selected_viewpoint_pose)

    def move_to_viewpoint_callback(self, request, response):
        """
        Callback for the move to viewpoint service.
        :return: True if the viewpoint was successfully moved to, False otherwise.
        """
        self.move_to_selected_viewpoint()

        return response
        # rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('Successfully moved to viewpoint.')
            else:
                self.get_logger().error(
                    f'Failed to move to viewpoint: {future.result().message}')

            response.success = future.result().success
            response.message = future.result().message
        else:
            response.success = False
            response.message = 'Failed to move to viewpoint. Service call failed.'

        return response

    def move_to_selected_viewpoint(self):
        while self.moving:
            time.sleep(0.1)

        self.moving = True

        region_index = self.get_parameter(
            'regions.selected_region').get_parameter_value().integer_value
        cluster_index = self.get_parameter(
            'regions.selected_cluster').get_parameter_value().integer_value

        viewpoint, message = self.viewpoint_generation.get_viewpoint(
            region_index, cluster_index)

        if viewpoint is None:
            self.get_logger().error(message)
            return False

        # Create a PoseStamped message for the viewpoint
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = 'object_frame'
        pose_goal.pose.position.x = viewpoint['position'][0]
        pose_goal.pose.position.y = viewpoint['position'][1]
        pose_goal.pose.position.z = viewpoint['position'][2]
        pose_goal.pose.orientation.x = viewpoint['orientation'][0]
        pose_goal.pose.orientation.y = viewpoint['orientation'][1]
        pose_goal.pose.orientation.z = viewpoint['orientation'][2]
        pose_goal.pose.orientation.w = viewpoint['orientation'][3]

        self.viewpoint_publisher.publish(pose_goal)

        # Call the MoveToPoseStamped server to move to the viewpoint
        request = MoveToPoseStamped.Request()
        request.pose_goal = pose_goal

        future = self.move_to_pose_stamped_client.call_async(request)
        future.add_done_callback(self.move_to_viewpoint_future_callback)

    def move_to_viewpoint_future_callback(self, future):
        """
        Callback for the MoveToPoseStamped result.
        :param future: The future object containing the result of the action.
        """
        self.moving = False
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('Successfully moved to viewpoint.')
                self.move_to_viewpoint_done_pub.publish(Bool(data=True))
            else:
                self.get_logger().error(
                    f'Failed to move to viewpoint: {result.message}')
                self.move_to_viewpoint_done_pub.publish(Bool(data=False))
        except Exception as e:
            self.get_logger().error(
                f'Exception while moving to viewpoint: {e}')

    def image_region_callback(self, request, response):
        """
        Callback for the image region service.
        :param request: The service request.
        :param response: The service response.
        """
        self.get_logger().info('Image region service called.')

        selected_region = self.get_parameter(
            'regions.selected_region').get_parameter_value().integer_value
        valid = True
        selected_viewpoint = 0
        while valid:
            self.select_viewpoint(selected_region, selected_viewpoint)
            self.move_to_selected_viewpoint()
            while self.moving:
                print("...")
                time.sleep(0.1)
            selected_viewpoint += 1

        response.success = True
        response.message = 'Image region service executed successfully.'
        return response

    def optimize_traversal(self, request, response):
        request = OptimizeViewpointTraversal.Request()
        request.viewpoint_dict_path = self.get_parameter(
            'regions.file').get_parameter_value().string_value

        future = self.optimize_traversal_client.call_async(request)
        future.add_done_callback(self.optimize_traversal_callback)

        response.success = True
        response.message = 'Optimization started...'
        return response

    def optimize_traversal_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('Optimization successful.')
                new_viewpoint_dict_path = result.new_viewpoint_dict_path
                # Update the regions file parameter with the new viewpoint dictionary path
                regions_file_param = rclpy.parameter.Parameter(
                    'regions.file',
                    rclpy.Parameter.Type.STRING,
                    new_viewpoint_dict_path
                )
                self.block_next_param_callback = True
                self.set_parameters([regions_file_param])
                self.get_logger().info(
                    f'New viewpoint dictionary saved at: {new_viewpoint_dict_path}')
            else:
                self.get_logger().error(
                    f'Optimization failed: {result.message}')
                return
        except Exception as e:
            self.get_logger().error(f"Optimization failed: {e}")

    def parameter_callback(self, params):
        """ Callback for parameter changes.
        :param params: List of parameters that have changed.
        :return: SetParametersResult indicating success or failure.
        """

        # If we are blocking the next parameter callback, return success
        if self.block_next_param_callback:
            self.block_next_param_callback = False
            return SetParametersResult(successful=True)

        success = True

        # Iterate through the parameters and set the corresponding values
        # based on the parameter name
        for param in params:
            if param.name == 'model.mesh.file':
                success = self.set_mesh_file(param.value,
                                             self.get_parameter(
                                                 'model.mesh.units').get_parameter_value().string_value)
            elif param.name == 'model.mesh.units':
                success = self.set_mesh_file(
                    self.get_parameter(
                        'model.mesh.file').get_parameter_value().string_value,
                    param.value)
            elif param.name == 'model.point_cloud.file':
                success = self.set_point_cloud_file(param.value, self.get_parameter(
                    'model.point_cloud.units').get_parameter_value().string_value)
            elif param.name == 'model.point_cloud.units':
                success = self.set_point_cloud_file(
                    self.get_parameter(
                        'model.point_cloud.file').get_parameter_value().string_value,
                    param.value)
            elif param.name == 'model.point_cloud.sampling.ppsqmm':
                success = self.set_sampling_ppsqmm(param.value)
            elif param.name == 'model.point_cloud.sampling.number_of_points':
                success = self.set_sampling_number_of_points(param.value)
            elif param.name == 'regions.file':
                success = self.set_regions_file(param.value)
            elif param.name == 'regions.region_growth.curvature.knn_neighbors':
                success = self.set_knn_neighbors(param.value)
            elif param.name == 'regions.region_growth.curvature.file':
                success = self.set_curvature_file(param.value)
            elif param.name == 'regions.region_growth.seed_threshold':
                success = self.set_seed_threshold(param.value)
            elif param.name == 'regions.region_growth.min_cluster_size':
                success = self.set_min_cluster_size(param.value)
            elif param.name == 'regions.region_growth.max_cluster_size':
                success = self.set_max_cluster_size(param.value)
            elif param.name == 'regions.region_growth.curvature_threshold':
                success = self.set_curvature_threshold(
                    param.value)
            elif param.name == 'regions.region_growth.normal_angle_threshold':
                success = self.set_normal_angle_threshold(param.value)
            elif param.name == 'regions.selected_region':
                region_index = param.value
                cluster_index = self.get_parameter(
                    'regions.selected_cluster').get_parameter_value().integer_value
                success = self.select_viewpoint(region_index, cluster_index)
            elif param.name == 'regions.selected_cluster':
                cluster_index = param.value
                region_index = self.get_parameter(
                    'regions.selected_region').get_parameter_value().integer_value
                success = self.select_viewpoint(region_index, cluster_index)
            elif param.name == 'model.camera.fov.height':
                fov_height = param.value
                fov_width = self.get_parameter(
                    'model.camera.fov.width').get_parameter_value().double_value
                dof = self.get_parameter(
                    'model.camera.dof').get_parameter_value().double_value
                focal_distance = self.get_parameter(
                    'model.camera.focal_distance').get_parameter_value().double_value
                self.set_camera_parameters(
                    fov_height=fov_height,
                    fov_width=fov_width,
                    dof=dof,
                    focal_distance=focal_distance
                )
            elif param.name == 'model.camera.fov.width':
                fov_height = self.get_parameter(
                    'model.camera.fov.height').get_parameter_value().double_value
                fov_width = param.value
                dof = self.get_parameter(
                    'model.camera.dof').get_parameter_value().double_value
                focal_distance = self.get_parameter(
                    'model.camera.focal_distance').get_parameter_value().double_value
                self.set_camera_parameters(
                    fov_height=fov_height,
                    fov_width=fov_width,
                    dof=dof,
                    focal_distance=focal_distance
                )
            elif param.name == 'model.camera.dof':
                fov_height = self.get_parameter(
                    'model.camera.fov.height').get_parameter_value().double_value
                fov_width = self.get_parameter(
                    'model.camera.fov.width').get_parameter_value().double_value
                dof = param.value
                focal_distance = self.get_parameter(
                    'model.camera.focal_distance').get_parameter_value().double_value
                self.set_camera_parameters(
                    fov_height=fov_height,
                    fov_width=fov_width,
                    dof=dof,
                    focal_distance=focal_distance
                )
            elif param.name == 'model.camera.focal_distance':
                fov_height = self.get_parameter(
                    'model.camera.fov.height').get_parameter_value().double_value
                fov_width = self.get_parameter(
                    'model.camera.fov.width').get_parameter_value().double_value
                dof = self.get_parameter(
                    'model.camera.dof').get_parameter_value().double_value
                focal_distance = param.value
                self.set_camera_parameters(
                    fov_height=fov_height,
                    fov_width=fov_width,
                    dof=dof,
                    focal_distance=focal_distance
                )
            elif param.name == 'settings.cuda_enabled':
                success = self.enable_cuda_callback(param.value)

        result = SetParametersResult()
        result.successful = success

        return result


def main():
    rclpy.init()
    node = ViewpointGenerationNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
