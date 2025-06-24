import os
import time
import rclpy
import rclpy.node
import numpy as np
import open3d as o3d
from rclpy.action import ActionServer
from viewpoint_generation.partitioner import Partitioner
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_prefix

from std_srvs.srv import Trigger
from viewpoint_generation_interfaces.action import ViewpointGeneration

from moveit.planning import MoveItPy

from geometry_msgs.msg import Pose, Point
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject


class ViewpointGenerationNode(rclpy.node.Node):

    block_next_param_callback = False

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
                ('regions.region_growth.curvature.file', ''),
                ('regions.region_growth.curvature.number_of_neighbors', 10),
                ('regions.file', ''),
                ('regions.region_growth.curvature_threshold', 0.50),
                ('regions.region_growth.angle_threshold', 15.0),
                ('viewpoints.nothing', ''),
                ('traversal.nothing', ''),
                ('camera.fov.width', 0.02),
                ('camera.fov.height', 0.03),
                ('camera.dof', 0.02),
                ('camera.focal_distance', 0.35),
                ('settings.cuda_enabled', False)
            ]
        )

        # # MoveItPy Initialization
        # self.moveit_py = MoveItPy()
        # self.planning_scene_monitor = self.moveit_py.get_planning_scene_monitor()

        # Viewpoint Generation Helpers
        self.partitioner = Partitioner()

        self.set_mesh_file(self.get_parameter('model.mesh.file').get_parameter_value().string_value,
                           self.get_parameter('model.mesh.units').get_parameter_value().string_value)
        self.set_point_cloud_file(self.get_parameter('model.point_cloud.file').get_parameter_value().string_value,
                                  self.get_parameter('model.point_cloud.units').get_parameter_value().string_value)
        self.set_sampling_ppsqmm(self.get_parameter(
            'model.point_cloud.sampling.ppsqmm').get_parameter_value().double_value)
        self.set_sampling_number_of_points(
            self.get_parameter(
                'model.point_cloud.sampling.number_of_points').get_parameter_value().integer_value)
        self.set_curvature_file(self.get_parameter(
            'regions.region_growth.curvature.file').get_parameter_value().string_value)
        self.partitioner.fov_width = self.get_parameter(
            'camera.fov.width').get_parameter_value().double_value
        self.partitioner.fov_height = self.get_parameter(
            'camera.fov.height').get_parameter_value().double_value
        self.partitioner.dof = self.get_parameter(
            'camera.dof').get_parameter_value().double_value
        self.partitioner.focal_distance = self.get_parameter(
            'camera.focal_distance').get_parameter_value().double_value
        self.partitioner.cuda_enabled = self.get_parameter(
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

        # Action Server
        self._action_server = ActionServer(
            self,
            ViewpointGeneration,
            'viewpoint_generation',
            self.execute_callback)

    def set_mesh_file(self, mesh_file, mesh_units):
        """
        Helper function to set the triangle mesh file for the partitioner.
        :param mesh_file: The path to the triangle mesh file.
        :return: None
        """

        if mesh_file is '' or None:
            self.get_logger().error(
                'No triangle mesh file provided.'
            )
            return False
        elif mesh_units not in ['m', 'cm', 'mm', 'inch']:
            self.get_logger().error(
                f'Invalid mesh units provided: {mesh_units}. Must be one of: m, cm, mm, inch.'
            )
            return False

        # If mesh_file begins with "package://package_name", replace it with the path to the package
        if mesh_file.startswith('package://'):
            package_name, relative_path = mesh_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            mesh_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success = self.partitioner.set_mesh_file(mesh_file, mesh_units)

        if not success:
            mesh_file_param = rclpy.parameter.Parameter(
                'model.mesh.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([mesh_file_param])
            self.get_logger().error(
                f'Could not load requested triangle mesh file {mesh_file}.'
            )
            return False
        if success:
            self.get_logger().info(
                f'Triangle mesh file {mesh_file} loaded successfully.'
            )
            # # Clear the point cloud and curvature file parameters
            # point_cloud_file_param = rclpy.parameter.Parameter(
            #     'model.point_cloud.file',
            #     rclpy.Parameter.Type.STRING,
            #     ''
            # )
            # curvature_file_param = rclpy.parameter.Parameter(
            #     'region_growth.curvature.file',
            #     rclpy.Parameter.Type.STRING,
            #     ''
            # )
            # self.block_next_param_callback = True
            # self.set_parameters([point_cloud_file_param, curvature_file_param])

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

        if point_cloud_file is '' or None:
            self.get_logger().warning(
                'No point cloud file provided.'
            )
            return False
        elif point_cloud_units not in ['m', 'cm', 'mm', 'inch']:
            self.get_logger().error(
                f'Invalid point cloud units provided: {point_cloud_units}. Must be one of: m, cm, mm, inch.'
            )
            return False

        # If point_cloud_file begins with "package://package_name", replace it with the path to the package
        if point_cloud_file.startswith('package://'):
            package_name, relative_path = point_cloud_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            point_cloud_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success = self.partitioner.set_point_cloud_file(
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
            self.get_logger().info(
                f'Point cloud file {point_cloud_file} loaded successfully.'
            )
            return True

    def set_curvature_file(self, curvature_file):
        """
        Helper function to set the curvature file for the partitioner.
        :param curvature_file: The path to the curvature file.
        :return: None
        """

        if curvature_file is '' or None:
            self.get_logger().warning(
                'No curvature file provided.'
            )
            return False

        # If curvature_file begins with "package://package_name", replace it with the path to the package
        if curvature_file.startswith('package://'):
            package_name, relative_path = curvature_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            curvature_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success = self.partitioner.set_curvature_file(curvature_file)

        if not success:
            curvature_file_param = rclpy.parameter.Parameter(
                'regions.region_growth.curvature.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([curvature_file_param])
            self.get_logger().error(
                f'Could not load requested curvature file {curvature_file}.'
            )
            return False
        else:
            self.get_logger().info(
                f'Curvature file {curvature_file} loaded successfully.'
            )
            return True

    def enable_cuda_callback(self, enable):
        """
        Helper function to enable or disable CUDA for the partitioner.
        :param enable: Boolean indicating whether to enable CUDA.
        :return: True if successful, False otherwise.
        """

        success = self.partitioner.enable_cuda(enable)

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

        success, N_points = self.partitioner.set_ppsqmm(ppsqmm)

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

        success, ppsqmm = self.partitioner.set_sampling_number_of_points(
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

        success, message = self.partitioner.sample_point_cloud()

        if success:
            self.get_logger().info(
                f"Point cloud sampled successfully. File: {message}")

            # Set the point cloud of the partitioner
            pcd_file = message
            success = self.partitioner.set_point_cloud_file(pcd_file, 'm')

            # Update the point cloud units to meters
            point_cloud_units_param = rclpy.parameter.Parameter(
                'model.point_cloud.units',
                rclpy.Parameter.Type.STRING,
                'm'
            )

            # Update the point cloud file parameter with the sampled file
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                pcd_file
            )

            self.set_parameters(
                [point_cloud_file_param, point_cloud_units_param])

        else:
            self.get_logger().error(f"Failed to sample point cloud: {message}")

        response.success = success
        response.message = message

        return response

    def set_curvature_number_of_neighbors(self, number_of_neighbors):
        """
        Helper function to set the number of neighbors for curvature estimation.
        :param number_of_neighbors: The number of neighbors to use for curvature estimation.
        :return: True if successful, False otherwise.
        """

        if number_of_neighbors <= 0:
            self.get_logger().error(
                'Number of neighbors must be greater than 0.'
            )
            return False

        success = self.partitioner.set_curvature_number_of_neighbors(
            number_of_neighbors)

        if not success:
            self.get_logger().error(
                f'Failed to set number of neighbors to {number_of_neighbors}.'
            )
            return False
        else:
            self.get_logger().info(
                f'Number of neighbors set to {number_of_neighbors}.')
            curvature_file_param = rclpy.parameter.Parameter(
                'regions.region_growth.curvature.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.block_next_param_callback = True
            self.set_parameters([curvature_file_param])

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

        success, message = self.partitioner.estimate_curvature()

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

    def set_region_growth_curvature_threshold(self, curvature_threshold):
        """
        Helper function to set the curvature threshold for region growth.
        :param curvature_threshold: The curvature threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        if curvature_threshold <= 0:
            self.get_logger().error(
                'Curvature threshold must be greater than 0.'
            )
            return False

        # success = self.partitioner.set_region_growth_curvature_threshold(curvature_threshold)
        self.partitioner.rg_curvature_threshold = curvature_threshold
        success = True  # Simulating success for now

        if not success:
            self.get_logger().error(
                f'Failed to set curvature threshold to {curvature_threshold}.'
            )
            return False
        else:
            self.get_logger().info(
                f'Curvature threshold set to {curvature_threshold}.')
            return True

    def set_region_growth_angle_threshold(self, angle_threshold):
        """
        Helper function to set the angle threshold for region growth.
        :param angle_threshold: The angle threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.partitioner.set_region_growth_angle_threshold(
            angle_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_region_growth_curvature_threshold(self, curvature_threshold):
        """
        Helper function to set the curvature threshold for region growth.
        :param curvature_threshold: The curvature threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.partitioner.set_region_growth_curvature_threshold(
            curvature_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def region_growth_callback(self, request, response):
        """
        Callback for the region growth service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
        """
        self.get_logger().info('Performing region growth...')

        success, message = self.partitioner.region_growth()

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
        if self.block_next_param_callback:
            self.block_next_param_callback = False
            return SetParametersResult(successful=True)

        success = True

        for param in params:
            if param.name == 'model.mesh.file':
                success = self.set_mesh_file(param.value, self.get_parameter(
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
            elif param.name == 'regions.region_growth.curvature.file':
                success = self.set_curvature_file(param.value)
            elif param.name == 'regions.region_growth.curvature.number_of_neighbors':
                success = self.set_curvature_number_of_neighbors(param.value)
            elif param.name == 'regions.region_growth.curvature_threshold':
                success = self.set_region_growth_curvature_threshold(
                    param.value)
            elif param.name == 'regions.region_growth.angle_threshold':
                success = self.set_region_growth_angle_threshold(param.value)
            elif param.name == 'camera.fov.height':
                self.partitioner.fov_height = param.value
            elif param.name == 'camera.fov.width':
                self.partitioner.fov_width = param.value
            elif param.name == 'camera.dof':
                self.partitioner.dof = param.value
            elif param.name == 'camera.focal_distance':
                self.partitioner.focal_distance = param.value
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
