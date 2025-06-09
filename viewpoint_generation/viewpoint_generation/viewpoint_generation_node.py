import os
import time
import rclpy
import rclpy.node
from rclpy.action import ActionServer
from viewpoint_generation.partitioner import Partitioner
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_prefix

from std_srvs.srv import Trigger
from viewpoint_generation_interfaces.action import ViewpointGeneration


class ViewpointGenerationNode(rclpy.node.Node):

    block_next_param_callback = False

    def __init__(self):
        node_name = 'viewpoint_generation_node'
        super().__init__(node_name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('visualize', False),
                ('cuda_enabled', False),
                ('model.triangle_mesh_file', ''),
                ('model.triangle_mesh_units', 'm'),
                ('model.point_cloud_file', ''),
                ('model.point_cloud_units', 'm'),
                ('pcd_sampling.ppsqmm', 1.),
                ('pcd_sampling.number_of_points', 100000),
                ('pcd_sampling.sample_point_cloud', False),
                ('curvature.estimate_curvature', False),
                ('curvature.number_of_neighbors', 30),
                ('regions.region_growth', False),
                ('camera.fov_width', 0.02),
                ('camera.fov_height', 0.03),
                ('camera.dof', 0.02),
                ('camera.focal_distance', 0.35)
            ]
        )

        # Viewpoint Generation Helpers
        self.partitioner = Partitioner()

        self.partitioner.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value
        self.set_triangle_mesh_file(self.get_parameter(
            'model.triangle_mesh_file').get_parameter_value().string_value)
        self.partitioner.triangle_mesh_units = self.get_parameter(
            'model.triangle_mesh_units').get_parameter_value().string_value
        self.set_point_cloud_file(self.get_parameter(
            'model.point_cloud_file').get_parameter_value().string_value)
        self.partitioner.point_cloud_units = self.get_parameter(
            'model.point_cloud_units').get_parameter_value().string_value
        self.partitioner.ppsqmm = self.get_parameter(
            'pcd_sampling.ppsqmm').get_parameter_value().integer_value
        self.partitioner.fov_width = self.get_parameter(
            'camera.fov_width').get_parameter_value().double_value
        self.partitioner.fov_height = self.get_parameter(
            'camera.fov_height').get_parameter_value().double_value
        self.partitioner.dof = self.get_parameter(
            'camera.dof').get_parameter_value().double_value
        self.partitioner.focal_distance = self.get_parameter(
            'camera.focal_distance').get_parameter_value().double_value

        self.add_on_set_parameters_callback(self.parameter_callback)

        # Sample PCD Service
        self.create_service(Trigger, node_name + '/sample_pcd',
                            self.sample_point_cloud_callback)

        # Action Server
        self._action_server = ActionServer(
            self,
            ViewpointGeneration,
            'viewpoint_generation',
            self.execute_callback)

    def set_triangle_mesh_file(self, triangle_mesh_file):
        """
        Helper function to set the triangle mesh file for the partitioner.
        :param triangle_mesh_file: The path to the triangle mesh file.
        :return: None
        """

        if triangle_mesh_file is '' or None:
            self.get_logger().error(
                'No triangle mesh file provided.'
            )
            return

        # If triangle_mesh_file begins with "package://package_name", replace it with the path to the package
        if triangle_mesh_file.startswith('package://'):
            package_name, relative_path = triangle_mesh_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            triangle_mesh_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success = self.partitioner.set_triangle_mesh_file(
            triangle_mesh_file,
            self.get_parameter(
                'model.triangle_mesh_units').get_parameter_value().string_value
        )

        if not success:
            triangle_mesh_file_param = rclpy.parameter.Parameter(
                'model.triangle_mesh_file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([triangle_mesh_file_param])
            self.get_logger().error(
                f'Could not load requested triangle mesh file {triangle_mesh_file}.'
            )
        else:
            self.get_logger().info(
                f'Triangle mesh file {triangle_mesh_file} loaded successfully.'
            )


    def set_point_cloud_file(self, point_cloud_file):
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

        # If point_cloud_file begins with "package://package_name", replace it with the path to the package
        if point_cloud_file.startswith('package://'):
            package_name, relative_path = point_cloud_file.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            point_cloud_file = os.path.join(
                package_path, 'share', package_name, relative_path)

        success = self.partitioner.set_point_cloud_file(
            point_cloud_file,
            self.get_parameter(
                'model.point_cloud_units').get_parameter_value().string_value
        )

        if not success:
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud_file',
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

    def parameter_callback(self, params):
        if self.block_next_param_callback:
            self.block_next_param_callback = False
            return SetParametersResult(successful=True)

        success = True

        for param in params:
            if param.name == 'visualize':
                self.partitioner.visualize = param.value
                self.get_logger().info(
                    f'Visualize set to {param.value}.'
                )
            elif param.name == 'cuda_enabled':
                success = self.partitioner.set_cuda_enabled(param.value)
                if param.value and success:
                    self.get_logger().info(
                        'CUDA enabled successfully.'
                    )
                elif not success:
                    self.get_logger().error(
                        'Failed to enable CUDA. Please check your setup.'
                    )
                else:
                    self.get_logger().info(
                        'CUDA disabled.'
                    )
            elif param.name == 'model.triangle_mesh_file':
                success = self.set_triangle_mesh_file(param.value)
            elif param.name == 'model.triangle_mesh_units':
                success = self.partitioner.triangle_mesh_units = param.value
            elif param.name == 'model.point_cloud_file':
                success = self.set_point_cloud_file(param.value)
            elif param.name == 'model.point_cloud_units':
                success = self.partitioner.point_cloud_units = param.value
            elif param.name == 'pcd_sampling.ppsqmm':
                self.partitioner.ppsqmm = param.value
                success, N_points = self.partitioner.set_ppsqmm(param.value)
                number_of_points_param = rclpy.parameter.Parameter(
                    'pcd_sampling.number_of_points',
                    rclpy.Parameter.Type.INTEGER,
                    N_points
                )
                self.block_next_param_callback = True
                self.set_parameters([number_of_points_param])
            elif param.name == 'pcd_sampling.number_of_points':
                # Set the number of points to sample
                success, ppsqmm = self.partitioner.set_number_of_points(param.value)
                if success:
                    self.get_logger().info(
                        f'Number of points set to {param.value}.'
                    )
                else:
                    self.get_logger().error(
                        'Failed to set number of points.'
                    )
                ppsqmm_param = rclpy.parameter.Parameter(   
                    'pcd_sampling.ppsqmm',
                    rclpy.Parameter.Type.DOUBLE,
                    ppsqmm
                )
                self.block_next_param_callback = True
                self.set_parameters([ppsqmm_param])
            elif param.name == 'pcd_sampling.sample_point_cloud':
                # Sample the point cloud
                success, message, pcd_file = self.sample_point_cloud()
                # Set the point cloud of the partitioner
                self.partitioner.set_point_cloud_file(pcd_file, 'm')

                params = []

                if success:
                    # Update the point cloud units to meters
                    point_cloud_units_param = rclpy.parameter.Parameter(
                        'model.point_cloud_units',
                        rclpy.Parameter.Type.STRING,
                        'm'
                    )
                    # Update the point cloud file parameter with the sampled file
                    point_cloud_file_param = rclpy.parameter.Parameter(
                        'model.point_cloud_file',
                        rclpy.Parameter.Type.STRING,
                        pcd_file
                    )
                    params.append(point_cloud_file_param)
                    params.append(point_cloud_units_param)

                # Set the parameter to False after sampling
                sample_point_cloud_param = rclpy.parameter.Parameter(
                    'sample_point_cloud',
                    rclpy.Parameter.Type.BOOL,
                    False
                )
                params.append(sample_point_cloud_param)

                self.block_next_param_callback = True
                self.set_parameters(params)

            elif param.name == 'curvature.number_of_neighbors':
                success = self.partitioner.set_number_of_neighbors(param.value)
            elif param.name == 'curvature.estimate_curvature':
                self.partitioner.estimate_curvature()
            elif param.name == 'regions.region_growth':
                self.partitioner.region_growth()
            elif param.name == 'camera.fov_height':
                self.partitioner.fov_height = param.value
            elif param.name == 'camera.fov_width':
                self.partitioner.fov_width = param.value
            elif param.name == 'camera.dof':
                self.partitioner.dof = param.value
            elif param.name == 'camera.focal_distance':
                self.partitioner.focal_distance = param.value

        result = SetParametersResult()
        result.successful = True

        return result

    def sample_point_cloud(self):
        """
        Sample the point cloud using the partitioner.
        :return: Tuple (success, message)
        """
        self.get_logger().info('Sampling point cloud...')

        success, message, pcd_file = self.partitioner.sample_point_cloud()

        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)

        return success, message, pcd_file

    def sample_point_cloud_callback(self, request, response):

        success, message = self.sample_point_cloud()

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


def main():
    rclpy.init()
    node = ViewpointGenerationNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
