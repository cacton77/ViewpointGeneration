#!/usr/bin/env python3
from math import pi
import rclpy
import time
import copy
import json
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped

from std_srvs.srv import Trigger, SetBool
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.srv import ServoCommandType
from viewpoint_generation_interfaces.srv import MoveToPoseStamped


class InspectionTaskPlanningNode(Node):

    block_next_param_callback = False

    node_name = 'inspection_task_planning'
    viewpoint_generation_node_name = 'viewpoint_generation_node'
    viewpoint_traversal_node_name = 'viewpoint_traversal_node'

    selected_viewpoint = None

    def __init__(self):
        super().__init__(self.node_name)
        self.get_logger().info('Task Planning Node Initialized')

        self.state = 'idle'

        # Example of state transitions
        self.transitions = {
            'idle': {
                'start_servo_control': 'servo_control',
                'start_trajectory_control': 'trajectory_control',
            },
            'servo_control': {
                'pause_servo_control': 'idle',
                'start_trajectory_control': 'trajectory_control',
            },
            'trajectory_control': {
                'execute_trajectory': 'executing_trajectory',
                'start_servo_control': 'servo_control'
            },
            'executing_trajectory': {
                'trajectory_done': 'trajectory_control',
            },
            'active': {
                'pause': 'paused',
                'stop': 'stopped'
            },
            'paused': {
                'resume': 'active',
                'stop': 'stopped'
            },
            'stopped': {
                'reset': 'idle'
            }
        }

        self.declare_parameters(
            namespace='',
            parameters=[
                ('servo_controllers', ['ur5e_forward_position_controller',
                                       'turntable_forward_position_controller']),
                ('trajectory_controllers', ['inspection_cell_controller']),
                ('servo_node_name', 'servo_node'),
                ('controller_manager_name', '/controller_manager'),
                ('viewpoints_file', '/workspaces/isaac_ros-dev/src/ViewpointGenerationData/turbine_blade_point_cloud/turbine_blade_mm_point_cloud_100000points_0.1_100_100000_0.1_0.1_0.5235987755982988_2025-08-19_14-14-00_viewpoints_optimized2025-08-19_16-17-34.json'),
                ('selected_region', 0),
                ('selected_viewpoint', 0),
            ]
        )

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.servo_controllers = self.get_parameter(
            'servo_controllers').get_parameter_value().string_array_value
        self.trajectory_controllers = self.get_parameter(
            'trajectory_controllers').get_parameter_value().string_array_value
        self.servo_node_name = self.get_parameter(
            'servo_node_name').get_parameter_value().string_value
        self.controller_manager_name = self.get_parameter(
            'controller_manager_name').get_parameter_value().string_value

        self.load_viewpoints(self.get_parameter(
            'viewpoints_file').get_parameter_value().string_value)

        self.select_viewpoint(self.get_parameter('selected_region').get_parameter_value().integer_value,
                              self.get_parameter('selected_viewpoint').get_parameter_value().integer_value)

        services_cb_group = MutuallyExclusiveCallbackGroup()

        # Switch controller client
        self.controller_manager_client = self.create_client(
            SwitchController,
            f'{self.controller_manager_name}/switch_controller')

        # Start servo client
        self.servo_command_type_client = self.create_client(
            ServoCommandType,
            f'{self.servo_node_name}/switch_command_type'
        )
        # Pause servo client
        self.pause_servo_client = self.create_client(
            SetBool,
            f'{self.servo_node_name}/pause_servo'
        )

        success = self.wait_for_services()
        if not success:
            self.get_logger().error('Failed to wait for services')
            return

        # Selected viewpoint publisher timer
        # Viewpoint publisher for RViz2 Visualization
        self.viewpoint_publisher = self.create_publisher(
            PoseStamped, f'{self.node_name}/selected_viewpoint', 10)

        self.create_timer(
            0.1, self.publish_selected_viewpoint)

        self.move_to_pose_stamped_client = self.create_client(
            MoveToPoseStamped, f'{self.viewpoint_traversal_node_name}/move_to_pose_stamped', callback_group=services_cb_group)

        # Activate servo control by default
        self.set_servo_command_type()
        self.start_servo_control()

    def load_viewpoints(self, filepath):
        self.viewpoints = []
        if filepath == '':
            self.get_logger().warning('Viewpoints file cleared, no viewpoints loaded')
            return False

        with open(filepath, 'r') as f:
            regions_dict = json.load(f)

        region_order = regions_dict['order']

        try:
            for region_id in region_order:
                region_dict = regions_dict['regions'][str(region_id)]
                cluster_order = region_dict['order']
                region_viewpoints = []
                for cluster_id in cluster_order:
                    cluster_dict = region_dict['clusters'][str(cluster_id)]
                    if not 'viewpoint' in cluster_dict:
                        self.get_logger().warning(
                            f'No viewpoint found in cluster {cluster_id}')
                        continue

                    position = cluster_dict['viewpoint']['position']
                    orientation = cluster_dict['viewpoint']['orientation']

                    viewpoint = PoseStamped()
                    viewpoint.header.frame_id = 'object_frame'
                    viewpoint.pose.position.x = position[0]
                    viewpoint.pose.position.y = position[1]
                    viewpoint.pose.position.z = position[2]
                    viewpoint.pose.orientation.x = orientation[0]
                    viewpoint.pose.orientation.y = orientation[1]
                    viewpoint.pose.orientation.z = orientation[2]
                    viewpoint.pose.orientation.w = orientation[3]

                    region_viewpoints.append(viewpoint)

                self.viewpoints.append(region_viewpoints)
        except Exception as e:
            self.get_logger().error(
                f'Failed to load viewpoints from {filepath}: {e}')
            self.viewpoints = []
            return False

        self.get_logger().info(
            f'Loaded {len(self.viewpoints)} regions of viewpoints from {filepath}')
        return True

    def select_viewpoint(self, region_index, viewpoint_index):
        """
        Helper function to select a viewpoint based on region and cluster indices.
        :param region_index: The index of the region to select.
        :param cluster_index: The index of the cluster to select.
        :return: None
        """
        if self.viewpoints is None:
            self.get_logger().error('No viewpoints loaded')
            viewpoint = None
        if self.viewpoints is None or len(self.viewpoints) == 0:
            self.get_logger().error('No viewpoints loaded')
            viewpoint = None

        if region_index < 0 or region_index >= len(self.viewpoints):
            self.get_logger().error(f'Invalid region index: {region_index}')
            viewpoint = None
        elif viewpoint_index < 0 or viewpoint_index >= len(self.viewpoints[region_index]):
            self.get_logger().error(
                f'Invalid viewpoint index: {viewpoint_index}')
            viewpoint = None
        else:
            viewpoint = self.viewpoints[region_index][viewpoint_index]

        if viewpoint is None:
            # Reset the selected region and viewpoint parameters
            selected_region_param = rclpy.parameter.Parameter(
                'selected_region',
                rclpy.Parameter.Type.INTEGER,
                0
            )
            selected_viewpoint_param = rclpy.parameter.Parameter(
                'selected_viewpoint',
                rclpy.Parameter.Type.INTEGER,
                0
            )
            self.block_next_param_callback = True
            self.set_parameters([selected_region_param])
            self.block_next_param_callback = True
            self.set_parameters([selected_viewpoint_param])

            self.selected_viewpoint = None

            return False
        else:
            selected_region_param = rclpy.parameter.Parameter(
                'selected_region',
                rclpy.Parameter.Type.INTEGER,
                region_index
            )
            selected_viewpoint_param = rclpy.parameter.Parameter(
                'selected_viewpoint',
                rclpy.Parameter.Type.INTEGER,
                viewpoint_index
            )
            self.block_next_param_callback = True
            self.set_parameters([selected_region_param])
            self.block_next_param_callback = True
            self.set_parameters([selected_viewpoint_param])

            self.selected_viewpoint = viewpoint
            self.get_logger().info(
                f'Selected viewpoint {viewpoint_index} in region {region_index}')

            return True

    def publish_selected_viewpoint(self):
        """
        Publish the currently selected viewpoint to the viewpoint topic.
        :return: None
        """
        if self.selected_viewpoint is not None:
            self.viewpoint_publisher.publish(self.selected_viewpoint)

    def trigger(self, event):
        if event in self.transitions.get(self.state, {}):
            next_state = self.transitions[self.state][event]
            self.get_logger().info(
                f'State changed from: {self.state} to {next_state} by event {event}')
            self.state = next_state
            return True
        else:
            self.get_logger().warning(
                f'Invalid event "{event}" for current state "{self.state}"')
            return False

    def wait_for_services(self):
        if not self.controller_manager_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                'Controller manager service not available, cannot switch controllers')
            return False
        if not self.pause_servo_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                'Start servo service not available, cannot start servo')
            return False
        if not self.pause_servo_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                'Pause servo service not available, cannot pause servo')
            return False
        self.get_logger().info('All required services are available')
        return True

    def start_servo_control(self):
        self.activate_forward_position_controller()

    def stop_servo_control(self):
        self.pause_servo(pause=True)
        self.activate_joint_trajectory_controller()

    def activate_forward_position_controller(self):
        req = SwitchController.Request()
        req.start_controllers = self.servo_controllers
        req.stop_controllers = self.trajectory_controllers
        self.get_logger().info('Activating forward position controller...')
        future = self.controller_manager_client.call_async(req)
        future.add_done_callback(
            self.activate_forward_position_controller_callback)

    def activate_forward_position_controller_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Activate forward position controller response: %s' % resp.ok)
            self.trigger('start_servo_control')

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))
            self.trigger('idle')

        if resp.ok:
            self.pause_servo(pause=False)

    def set_servo_command_type(self, command_type='TWIST'):
        req = ServoCommandType.Request()

        if command_type == 'JOINT_JOG':
            req.command_type = ServoCommandType.Request.JOINT_JOG
        elif command_type == 'TWIST':
            req.command_type = ServoCommandType.Request.TWIST

        self.get_logger().info(
            f'Setting servo command type to {command_type}...')
        future = self.servo_command_type_client.call_async(req)
        future.add_done_callback(self.set_servo_command_type_callback)

    def set_servo_command_type_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Set servo command type response: %s' % resp.success)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

    def pause_servo(self, pause=True):
        # Call /servo_node/pause_servo service

        req = SetBool.Request()
        req.data = pause

        if pause:
            self.get_logger().info('Pausing servo...')
        else:
            self.get_logger().info('Resuming servo...')

        future = self.pause_servo_client.call_async(req)
        future.add_done_callback(self.pause_servo_callback)

    def pause_servo_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Pause servo response: %s' % resp.success)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        # self.servo_state = ON
        # self.state = IDLE

    def activate_joint_trajectory_controller(self):
        # self.state = STOPPING_SERVO

        req = SwitchController.Request()
        req.start_controllers = self.trajectory_controllers
        req.stop_controllers = self.servo_controllers
        self.get_logger().info('Activating joint trajectory controller...')

        future = self.controller_manager_client.call_async(req)
        future.add_done_callback(
            self.activate_joint_trajectory_controller_callback)

    def activate_joint_trajectory_controller_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Activate joint trajectory controller response: %s' % resp.ok)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))
            self.trigger('idle')

        if resp.ok:
            self.trigger('start_trajectory_control')

    def move_to_viewpoint(self):
        """Move the robot to the selected viewpoint"""
        self.stop_servo_control()
        while not self.state == 'trajectory_control':
            self.get_logger().info('Waiting for trajectory control to be activated...')
            time.sleep(0.1)

        self.trigger('execute_trajectory')

        request = Trigger.Request()
        future = self.move_to_viewpoint_client.call_async(request)
        future.add_done_callback(self.move_to_viewpoint_future_callback)

    def move_to_viewpoint_future_callback(self, future):
        """Callback for the move to viewpoint service future"""
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('Moved to viewpoint triggered successfully')
            else:
                self.get_logger().error(
                    f'Failed to move to viewpoint: {future.result().message}')
        else:
            self.get_logger().error('Failed to call move_to_viewpoint service')

    def move_to_viewpoint_done_callback(self, msg):
        """Callback for the move to viewpoint done topic"""

        self.trigger('trajectory_done')
        self.start_servo_control()

        if msg.data:
            self.get_logger().info('Move to viewpoint completed successfully')
        else:
            self.get_logger().error('Move to viewpoint failed')

    def image_selected_region(self, path):
        """Move to all viewpoints in region"""
        self.path = copy.deepcopy(path)
        self.trigger('start_trajectory_control')

    def process_path(self):
        if not self.path:
            return
        elif self.state == 'executing_trajectory':
            return
        elif self.state != 'trajectory_control':
            self.get_logger().info(
                'Waiting for trajectory control to be activated before processing path...')
            self.stop_servo_control()
            return
        else:
            viewpoint_index = self.path.pop(0)
            self.select_cluster(viewpoint_index)
            # self.move_to_viewpoint()
            self.trigger('execute_trajectory')
            self.move_to_viewpoint_client.call_async(Trigger.Request())
            if not self.path:
                self.get_logger().info('All viewpoints in region processed')
                self.select_cluster(0)
                self.start_servo_control()

    def parameter_callback(self, params):
        """ Callback for parameter changes.
        :param params: List of parameters that have changed.
        :return: SetParametersResult indicating success or failure.
        """
        # If we are blocking the next parameter callback, return success
        if self.block_next_param_callback:
            self.block_next_param_callback = False
            return SetParametersResult(successful=True)

        for param in params:
            # Viewpoints file parameter
            if param.name == 'viewpoints_file' and param.type_ == param.Type.STRING:
                success = self.load_viewpoints(param.value)
                self.get_logger().info(
                    f'Viewpoints file changed to: {param.value}')
            # Viewpoint selection parameters
            elif param.name == 'selected_region':
                region_index = param.value
                viewpoint_index = self.get_parameter(
                    'selected_viewpoint').get_parameter_value().integer_value
                success = self.select_viewpoint(region_index, viewpoint_index)
            elif param.name == 'selected_viewpoint':
                viewpoint_index = param.value
                region_index = self.get_parameter(
                    'selected_region').get_parameter_value().integer_value
                success = self.select_viewpoint(region_index, viewpoint_index)

        result = SetParametersResult()
        result.successful = success

        return result


def main():
    rclpy.init()
    node = InspectionTaskPlanningNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
