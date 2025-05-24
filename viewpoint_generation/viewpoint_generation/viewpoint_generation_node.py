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
    def __init__(self):
        node_name = 'viewpoint_generation_node'
        super().__init__(node_name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('visualize', False),
                ('triangle_mesh_file', ''),
                ('triangle_mesh_units', 'm'),
                ('point_cloud_file', ''),
                ('point_cloud_units', 'm'),
                ('ppsqmm', 1),
                ('fov_width', 0.02),
                ('fov_height', 0.03),
                ('dof', 0.02),
                ('focal_distance', 0.35)
            ]
        )

        # Viewpoint Generation Helpers
        self.partitioner = Partitioner()

        self.partitioner.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value
        self.set_triangle_mesh_file(self.get_parameter(
            'triangle_mesh_file').get_parameter_value().string_value)
        self.partitioner.triangle_mesh_units = self.get_parameter(
            'triangle_mesh_units').get_parameter_value().string_value
        self.set_point_cloud_file(self.get_parameter(
            'point_cloud_file').get_parameter_value().string_value)
        self.partitioner.point_cloud_units = self.get_parameter(
            'point_cloud_units').get_parameter_value().string_value
        self.partitioner.ppsqmm = self.get_parameter(
            'ppsqmm').get_parameter_value().integer_value
        self.partitioner.fov_width = self.get_parameter(
            'fov_width').get_parameter_value().double_value
        self.partitioner.fov_height = self.get_parameter(
            'fov_height').get_parameter_value().double_value
        self.partitioner.dof = self.get_parameter(
            'dof').get_parameter_value().double_value
        self.partitioner.focal_distance = self.get_parameter(
            'focal_distance').get_parameter_value().double_value

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
                'triangle_mesh_units').get_parameter_value().string_value
        )

        if not success:
            triangle_mesh_file_param = rclpy.parameter.Parameter(
                'triangle_mesh_file',
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
            self.get_logger().error(
                'No point cloud file provided.'
            )
            return

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
                'point_cloud_units').get_parameter_value().string_value
        )

        if not success:
            point_cloud_file_param = rclpy.parameter.Parameter(
                'point_cloud_file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([point_cloud_file_param])
            self.get_logger().error(
                f'Could not load requested point cloud file {point_cloud_file}.'
            )
        else:
            self.get_logger().info(
                f'Point cloud file {point_cloud_file} loaded successfully.'
            )

    def parameter_callback(self, params):

        success = True

        for param in params:
            if param.name == 'triangle_mesh_file':
                self.set_triangle_mesh_file(param.value)
            elif param.name == 'triangle_mesh_units':
                self.partitioner.triangle_mesh_units = param.value
            elif param.name == 'point_cloud_file':
                self.set_point_cloud_file(param.value)
            elif param.name == 'point_cloud_units':
                self.partitioner.point_cloud_units = param.value
            elif param.name == 'fov_height':
                self.partitioner.fov_height = param.value
            elif param.name == 'fov_width':
                self.partitioner.fov_width = param.value
            elif param.name == 'dof':
                self.partitioner.dof = param.value
            elif param.name == 'focal_distance':
                self.partitioner.focal_distance = param.value
            elif param.name == 'ppsqmm':
                self.partitioner.ppsqmm = param.value

        result = SetParametersResult()
        result.successful = success
        return result

    def sample_point_cloud_callback(self, request, response):
        self.get_logger().info('Sampling point cloud...')

        success, message = self.partitioner.sample_point_cloud()

        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)

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
