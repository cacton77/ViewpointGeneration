#!/usr/bin/env python3
from math import pi
import rclpy
import time
import copy
import json
from enum import Enum
from typing import Dict, Callable, Any
from dataclasses import dataclass
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped

from std_srvs.srv import Trigger, SetBool
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.srv import ServoCommandType
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from viewpoint_generation_interfaces.action import InspectRegion


class State(Enum):
    INITIALIZING = 0
    ACTIVATING_SERVO_CONTROL = 1
    IDLE = 2
    DEACTIVATING_SERVO_CONTROL = 3
    TRAJECTORY_CONTROL = 4
    SHUTDOWN = 5


@dataclass
class FLAGS:
    move_to_viewpoint = False
    move_to_viewpoint_request_sent = False
    activate_servo_control_request_sent = False
    servo_control_active = False
    deactivate_servo_control_request_sent = False
    activate_trajectory_control_request_sent = False
    trajectory_control_active = False
    deactivate_trajectory_control_request_sent = False
    verbose = True
    turntable_moving = False
    error = False


class InspectionTaskPlanningNode(Node):

    block_next_param_callback = False

    node_name = 'inspection_task_planning'
    viewpoint_generation_node_name = 'viewpoint_generation'
    viewpoint_traversal_node_name = 'viewpoint_traversal'

    selected_viewpoint = None

    def __init__(self):
        super().__init__(self.node_name)
        self.get_logger().info('Task Planning Node Initialized')

        self.flags = FLAGS()
        self.current_state = State.INITIALIZING

        self.state_functions: Dict[State, Callable] = {
            State.INITIALIZING: self._handle_initializing,
            State.ACTIVATING_SERVO_CONTROL: self._handle_activating_servo_control,
            State.IDLE: self._handle_idle,
            State.DEACTIVATING_SERVO_CONTROL: self._handle_deactivating_servo_control,
            State.TRAJECTORY_CONTROL: self._handle_trajectory_control,
            State.SHUTDOWN: self._handle_shutdown,
        }

        self.valid_transitions = {
            State.INITIALIZING: [State.ACTIVATING_SERVO_CONTROL],
            State.ACTIVATING_SERVO_CONTROL: [State.IDLE, State.SHUTDOWN],
            State.IDLE: [State.DEACTIVATING_SERVO_CONTROL],
            State.DEACTIVATING_SERVO_CONTROL: [State.TRAJECTORY_CONTROL],
            State.TRAJECTORY_CONTROL: [State.ACTIVATING_SERVO_CONTROL],
            State.SHUTDOWN: []
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

        # Callback groups
        subscriber_callback_group = ReentrantCallbackGroup()
        action_callback_group = ReentrantCallbackGroup()
        timer_callback_group = ReentrantCallbackGroup()
        services_cb_group = ReentrantCallbackGroup()

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
            PoseStamped,
            f'{self.node_name}/selected_viewpoint',
            10
        )

        self.create_timer(
            0.1,
            self.publish_selected_viewpoint,
            callback_group=timer_callback_group
        )

        # Turntable status monitor TODO: Implement PID control for turntable to improve response
        self.create_subscription(
            Bool, '/turntable/status', self.turntable_status_callback, 10)  # , callback_group=subscriber_callback_group)

        self.create_service(
            Trigger, f'{self.node_name}/move_to_viewpoint', self.trigger_move_to_viewpoint_callback)
        self.move_to_pose_stamped_client = self.create_client(
            MoveToPoseStamped, f'{self.viewpoint_traversal_node_name}/move_to_pose_stamped', callback_group=services_cb_group)

        self.inspect_region_action_server = ActionServer(
            self,
            InspectRegion,
            f'{self.node_name}/inspect_region',
            self.execute_inspect_region_callback,
            callback_group=action_callback_group
        )

        # Update timer
        self.create_timer(
            0.1,
            self.update,
            callback_group=timer_callback_group
        )

    # ============================================================================
    # INITIALIZATION AND PARAMETER HANDLING
    # ============================================================================

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

    def turntable_status_callback(self, msg):
        self.get_logger().info(f'Turntable moving: {msg.data}')
        self.flags.turntable_moving = msg.data

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

    # ============================================================================
    # STATE HANDLERS
    # ============================================================================

    def _handle_initializing(self):
        self.get_logger().info('Handling INITIALIZING state')
        self.transition_to(State.ACTIVATING_SERVO_CONTROL)

    # ACTIVATING SERVO CONTROL
    def _handle_activating_servo_control(self):
        if self.flags.verbose:
            self.get_logger().info('Handling ACTIVATING_SERVO_CONTROL state')

        if self.flags.servo_control_active:
            self.transition_to(State.IDLE)
        elif not self.flags.activate_servo_control_request_sent:
            self.start_servo_control()
            self.flags.activate_servo_control_request_sent = True

    # IDLE
    def _handle_idle(self):
        if self.flags.verbose:
            self.get_logger().info('Handling IDLE state')

        if self.flags.move_to_viewpoint:
            self.transition_to(State.DEACTIVATING_SERVO_CONTROL)

    # DEACTIVATING SERVO CONTROL
    def _handle_deactivating_servo_control(self):
        if self.flags.verbose:
            self.get_logger().info('Handling DEACTIVATING_SERVO_CONTROL state')

        if self.flags.trajectory_control_active:
            self.transition_to(State.TRAJECTORY_CONTROL)
        elif not self.flags.deactivate_servo_control_request_sent:
            self.stop_servo_control()
            self.flags.deactivate_servo_control_request_sent = True

    # TRAJECTORY CONTROL

    def _handle_trajectory_control(self):
        if self.flags.verbose:
            self.get_logger().info('Handling TRAJECTORY_CONTROL state')

        if self.flags.move_to_viewpoint and not self.flags.move_to_viewpoint_request_sent:
            self.move_to_viewpoint()
            self.flags.move_to_viewpoint_request_sent = True
        elif not self.flags.move_to_viewpoint and not self.flags.turntable_moving:
            self.transition_to(State.ACTIVATING_SERVO_CONTROL)

    # SHUTDOWN
    def _handle_shutdown(self):
        if self.flags.verbose:
            self.get_logger().info('Handling SHUTDOWN state')

        self.stop_servo_control()
        self.get_logger().info('Shutting down node...')
        rclpy.shutdown()

    def transition_to(self, new_state: State) -> bool:
        """Attempt to transition to a new state"""
        if new_state in self.valid_transitions[self.current_state]:
            old_state = self.current_state
            self.current_state = new_state
            if self.flags.verbose:
                self.get_logger().info(
                    f"State transition: {old_state.value} -> {new_state.value}")
            return True
        else:
            if self.flags.verbose:
                self.get_logger().warning(
                    f"Invalid transition: {self.current_state.value} -> {new_state.value}")
            return False

    def update(self):
        if self.current_state in self.state_functions:
            self.state_functions[self.current_state]()

    # ============================================================================
    # MOVE TO VIEWPOINT
    # ============================================================================

    def trigger_move_to_viewpoint_callback(self, request, response):
        self.flags.move_to_viewpoint = True
        return response

    def move_to_viewpoint(self):
        """Move the robot to the selected viewpoint"""
        request = MoveToPoseStamped.Request()

        if self.selected_viewpoint is None:
            self.get_logger().warning('No viewpoint selected, cannot move to viewpoint')

            self.flags.move_to_viewpoint = False
            self.flags.move_to_viewpoint_request_sent = False

            return False

        self.get_logger().info('Moving to selected viewpoint...')

        request.pose_goal = copy.deepcopy(self.selected_viewpoint)

        future = self.move_to_pose_stamped_client.call_async(request)
        future.add_done_callback(self.move_to_pose_stamped_future_callback)

        return True

    def move_to_pose_stamped_future_callback(self, future):
        """Callback for the move to pose stamped service future"""
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('Moved to pose stamped successfully')
            else:
                self.get_logger().error(
                    f'Failed to move to pose stamped: {future.result().message}')

            self.flags.move_to_viewpoint = False
            self.flags.move_to_viewpoint_request_sent = False

        else:
            self.get_logger().error('Failed to call move_to_pose_stamped service')

    # ============================================================================
    # CONTROLLER SWITCHING
    # ============================================================================

    def start_servo_control(self):
        self.get_logger().info('Starting servo control...')
        self.pause_servo(pause=False)
        self.activate_forward_position_controller()

    def stop_servo_control(self):
        self.get_logger().info('Stopping servo control...')
        self.pause_servo(pause=True)
        self.activate_joint_trajectory_controller()

    def activate_forward_position_controller(self):
        self.get_logger().info('Activating forward position controller...')

        req = SwitchController.Request()
        req.activate_controllers = self.servo_controllers
        req.deactivate_controllers = self.trajectory_controllers
        future = self.controller_manager_client.call_async(req)
        future.add_done_callback(
            self.activate_forward_position_controller_callback)

    def activate_forward_position_controller_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Activate forward position controller response: %s' % resp.ok)

            if resp.ok:
                self.flags.servo_control_active = True
                self.flags.trajectory_control_active = False
                self.flags.activate_servo_control_request_sent = False
            else:
                self.flags.servo_control_active = False
                self.flags.trajectory_control_active = False
                self.flags.activate_servo_control_request_sent = False
                self.flags.error = True

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

            self.flags.servo_control_active = False
            self.flags.error = True

    def activate_joint_trajectory_controller(self):
        self.get_logger().info('Activating joint trajectory controller...')

        req = SwitchController.Request()
        req.activate_controllers = self.trajectory_controllers
        req.deactivate_controllers = self.servo_controllers

        future = self.controller_manager_client.call_async(req)
        future.add_done_callback(
            self.activate_joint_trajectory_controller_callback)

    def activate_joint_trajectory_controller_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Activate joint trajectory controller response: %s' % resp.ok)

            if resp.ok:
                self.flags.servo_control_active = False
                self.flags.trajectory_control_active = True
                self.flags.deactivate_servo_control_request_sent = False
            else:
                self.flags.servo_control_active = False
                self.flags.trajectory_control_active = False
                self.flags.deactivate_servo_control_request_sent = False
                self.flags.error = True

        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

            self.flags.trajectory_control_active = False
            self.flags.error = True

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

    # ============================================================================
    # INSPECT REGION ACTION
    # ============================================================================

    def execute_inspect_region_callback(self, goal_handle):
        self.get_logger().info('Executing inspect region...')
        feedback_msg = InspectRegion.Feedback()

        selected_region = self.get_parameter(
            'selected_region').get_parameter_value().integer_value
        for i in range(len(self.viewpoints[selected_region])-1):
            self.select_viewpoint(selected_region, i)
            self.flags.move_to_viewpoint = True
            while self.flags.move_to_viewpoint:
                time.sleep(0.1)

        goal_handle.succeed()

        result = InspectRegion.Result()
        result.success = True
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

    executor = MultiThreadedExecutor(num_threads=4)

    node = InspectionTaskPlanningNode()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
