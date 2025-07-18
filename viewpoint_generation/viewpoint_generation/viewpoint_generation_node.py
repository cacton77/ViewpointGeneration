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

from std_srvs.srv import Trigger
# from viewpoint_generation_interfaces.action import ViewpointGeneration

from moveit.planning import MoveItPy

from geometry_msgs.msg import Pose, Point
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject


class ViewpointGenerationNode(rclpy.node.Node):

    block_next_param_callback = False
    initialized = False

    def __init__(self):
        node_name = 'viewpoint_generation_node'
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
                ('viewpoints.traversal', ''),
                ('viewpoints.projection.nothing', ''),
                ('settings.cuda_enabled', False)
            ]
        )

        # # MoveItPy Initialization
        # self.moveit_py = MoveItPy()
        # self.planning_scene_monitor = self.moveit_py.get_planning_scene_monitor()

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

        self.add_on_set_parameters_callback(self.parameter_callback)

        # Sample PCD Service
        self.create_service(Trigger, node_name + '/sample_point_cloud',
                            self.sample_point_cloud_callback)
        # Estimate Curvature Service
        self.create_service(Trigger, node_name + '/estimate_curvature',
                            self.estimate_curvature_callback)
        # Region Growth Service
        self.create_service(Trigger, node_name + '/region_growth',
                            self.region_growth_callback)
        # FOV Clustering Service
        self.create_service(Trigger, node_name + '/fov_clustering',
                            self.fov_clustering_callback)
        # Viewpoint Projection Service
        self.create_service(Trigger, node_name + '/viewpoint_projection',
                            self.viewpoint_projection_callback)

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
                self.set_point_cloud_file(point_cloud_file='', point_cloud_units='')



            # # Update planning scene with the new mesh
            # with self.planning_scene_monitor.read_write() as scene:
            #     # Load mesh
            #     mesh = o3d.io.read_triangle_mesh(mesh_file)

            #     # Convert mesh data
            #     vertices = np.asarray(mesh.vertices).tolist()
            #     triangles = np.asarray(mesh.triangles).tolist()

            #     # Add collision object (this syntax may vary based on moveit_py version)
            #     scene.add_collision_mesh(
            #         'object_mesh',
            #         'tool0',
            #         vertices,
            #         triangles
            #     )

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

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = ViewpointGeneration.Feedback()
        feedback = [0, 1]
        feedback_msg.feedback = ' '.join(feedback)

        for i in range(1, int(goal_handle.request.goal)):
            feedback.append(feedback[i] + feedback[i-1])
            feedback_msg.feedback = ' '.join(feedback)
            self.get_logger().info('Feedback: {0}'.format(
                feedback_msg.feedback))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()

        result = ViewpointGeneration.Result()
        result.result = feedback_msg.feedback
        return result

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
