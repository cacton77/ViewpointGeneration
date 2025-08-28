#!/usr/bin/env python3
import rclpy
import time
import copy
import json
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.msg import SetParametersResult

from std_srvs.srv import Trigger, SetBool
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.srv import ServoCommandType


class TaskPlanningNode(Node):

    node_name = 'inspection_task_planning'
    viewpoint_generation_node_name = 'viewpoint_generation_node'
    viewpoint_traversal_node_name = 'viewpoint_traversal_node'

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
            namespace='task_planning',
            parameters=[
                ('servo_controllers', ['ur5e_forward_position_controller',
                                       'turntable_forward_position_controller']),
                ('trajectory_controllers', ['inspection_cell_controller']),
                ('servo_node_name', 'servo_node'),
                ('controller_manager_name', '/controller_manager'),
                ('viewpoints_file', '/workspaces/isaac_ros-dev/src/ViewpointGenerationData/turbine_blade_point_cloud/turbine_blade_mm_point_cloud_100000points_0.1_100_100000_0.1_0.1_0.5235987755982988_2025-08-19_14-14-00_viewpoints_optimized2025-08-19_16-17-34.json')
            ]
        )

        self.servo_controllers = self.get_parameter(
            'servo_controllers').get_parameter_value().string_array_value
        self.trajectory_controllers = self.get_parameter(
            'trajectory_controllers').get_parameter_value().string_array_value
        self.servo_node_name = self.get_parameter(
            'servo_node_name').get_parameter_value().string_value
        self.controller_manager_name = self.get_parameter(
            'controller_manager_name').get_parameter_value().string_value
        self.viewpoints = self.load_viewpoints(self.get_parameter(
            'viewpoints_file').get_parameter_value().string_value)

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

        # Activate servo control by default
        self.set_servo_command_type()
        self.start_servo_control()

    def load_viewpoints(self, filepath):
        viewpoints = []
        if filepath == '':
            self.get_logger().warning('Viewpoints file cleared, no viewpoints loaded')
            return viewpoints

        with open(filepath, 'r') as f:
            regions_dict = json.load(f)

        region_order = regions_dict['order']

        for region_id in region_order:
            region_dict = regions_dict['regions'][str(region_id)]
            cluster_order = region_dict['order']
            for cluster_id in cluster_order:
                cluster_dict = region_dict['clusters'][str(cluster_id)]

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
        for param in params:
            if param.name == 'viewpoints_file' and param.type_ == param.Type.STRING:
                success = self.load_viewpoints(param.value)
                self.get_logger().info(
                    f'Viewpoints file changed to: {param.value}')

        result = SetParametersResult()
        result.successful = success

        return result


def main():
    rclpy.init()
    node = TaskPlanningNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
