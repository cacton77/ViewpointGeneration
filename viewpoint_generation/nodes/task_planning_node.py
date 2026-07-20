#!/usr/bin/env python3
import os
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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_prefix

from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType, IntegerRange
from geometry_msgs.msg import PoseStamped

from std_srvs.srv import Trigger, SetBool
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.srv import ServoCommandType
from moveit_msgs.msg import ServoStatus
from viewpoint_generation_interfaces.srv import MoveToPoseStamped, FindNearestViewpoint
from viewpoint_generation_interfaces.action import InspectRegion


def _resolve_order_indices(order, selected_algorithm=None):
    """Resolve a region's traversal order into a flat list of cluster indices.

    ``order`` is a plain list before traversal optimization and a dict keyed by
    TSP algorithm name afterwards. Each algorithm maps to
    ``{'order': [...], 'distance': ...}`` (older files stored the bare index
    list — both are handled). For a dict, prefer ``selected_algorithm`` (this
    node's ``selected_traversal_algorithm`` parameter, set via the GUI);
    otherwise fall back to the first available algorithm's path.
    """
    if isinstance(order, dict):
        if not order:
            return []
        entry = order.get(selected_algorithm, next(iter(order.values())))
        if isinstance(entry, dict):
            return entry.get('order', [])
        return entry
    return order


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
    verbose = False
    turntable_moving = False
    error = False
    # Held True for the duration of the full inspection traversal so the
    # state machine does not cycle back to servo between viewpoints.
    inspect_active = False


class TaskPlanningNode(Node):

    block_next_param_callback = False

    node_name = 'task_planning'
    viewpoint_generation_node_name = 'viewpoint_generation'
    viewpoint_traversal_node_name = 'viewpoint_traversal'

    selected_viewpoint = None
    _viewpoint_region_ids = []  # original region IDs in VRP traversal order

    # Live mesh/region/viewpoint selection indices (visualization + execution
    # scope). selected_mesh is visualization-only; planning/execution always
    # operate on mesh 0 (self.viewpoints).
    _sel_mesh = 0
    _sel_region = 0
    _sel_viewpoint = 0
    results_dict = {}

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

        # Parameters are namespaced (controllers.*, navigation.*, settings.*) so
        # the GUI can render one tab per namespace in the task_planning panel.
        self.declare_parameters(
            namespace='',
            parameters=[
                ('controllers.servo_controllers', ['ur5e_forward_velocity_controller',
                                                   'turntable_forward_position_controller']),
                ('controllers.trajectory_controllers', ['inspection_cell_controller']),
                ('controllers.servo_node_name', 'servo_node'),
                ('controllers.controller_manager_name', '/controller_manager'),
                ('settings.results_file', ''),
                ('navigation.selected_traversal_algorithm', ''),
                ('settings.data_path', '/tmp')
            ]
        )

        # Mesh/region/viewpoint selection parameters. Declared from a config
        # dict so the GUI renders them as sliders (control='slider') and so the
        # IntegerRange descriptors can be resized live as viewpoints load (see
        # _declare_selection_parameters / update_selection). This mirrors the
        # config-dict driven declaration in the viewpoint_generation node.
        self.selection_config = {
            'navigation.selected_mesh': {
                'value': 0, 'type': 'integer', 'range': [0, 0],
                'control': 'slider',
                'description': 'Index of the selected mesh (visualization only)',
            },
            'navigation.selected_region': {
                'value': 0, 'type': 'integer', 'range': [0, 0],
                'control': 'slider',
                'description': 'Index of the selected region',
            },
            'navigation.selected_viewpoint': {
                'value': 0, 'type': 'integer', 'range': [0, 0],
                'control': 'slider',
                'description': 'Index of the selected viewpoint within the region',
            },
        }
        self._declare_selection_parameters()

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.servo_controllers = self.get_parameter(
            'controllers.servo_controllers').get_parameter_value().string_array_value
        self.trajectory_controllers = self.get_parameter(
            'controllers.trajectory_controllers').get_parameter_value().string_array_value
        self.servo_node_name = self.get_parameter(
            'controllers.servo_node_name').get_parameter_value().string_value
        self.controller_manager_name = self.get_parameter(
            'controllers.controller_manager_name').get_parameter_value().string_value

        # Set data path
        self.set_data_path(self.get_parameter(
            'settings.data_path').get_parameter_value().string_value)

        # Which TSP algorithm's order to follow when a region's order is a
        # per-algorithm dict. Source of truth (set via the GUI); replaces the
        # old results-file 'selected_traversal_algorithm' field.
        self.selected_traversal_algorithm = self.get_parameter(
            'navigation.selected_traversal_algorithm').get_parameter_value().string_value

        self.load_viewpoints(self.get_parameter(
            'settings.results_file').get_parameter_value().string_value)

        # Clamp the initial selection to the loaded data and publish the live
        # slider ranges to the GUI.
        self.update_selection(
            self.get_parameter('navigation.selected_mesh').get_parameter_value().integer_value,
            self.get_parameter('navigation.selected_region').get_parameter_value().integer_value,
            self.get_parameter('navigation.selected_viewpoint').get_parameter_value().integer_value)

        # Callback groups
        subscriber_callback_group = ReentrantCallbackGroup()
        action_callback_group = ReentrantCallbackGroup()
        timer_callback_group = ReentrantCallbackGroup()
        viewpoint_callback_group = ReentrantCallbackGroup()
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
            callback_group=viewpoint_callback_group
        )

        # Turntable status monitor TODO: Implement PID control for turntable to improve response
        # Create QoS profile with best effort reliability
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(
            Bool, '/turntable/status', self.turntable_status_callback, qos_profile, callback_group=subscriber_callback_group)

        self.create_service(
            Trigger, f'{self.node_name}/move_to_viewpoint', self.trigger_move_to_viewpoint_callback, callback_group=services_cb_group)
        self.move_to_pose_stamped_client = self.create_client(
            MoveToPoseStamped, f'{self.viewpoint_traversal_node_name}/move_to_pose_stamped', callback_group=services_cb_group)
        self.find_nearest_viewpoint_client = self.create_client(
            FindNearestViewpoint, f'{self.viewpoint_traversal_node_name}/find_nearest_viewpoint', callback_group=services_cb_group)

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

        # Gate the first switch_command_type call on Servo being ready.
        # Servo publishes /servo_node/status only after its internal state
        # machine has a valid robot state to seed the target pose from.
        # Switching command type before that point latches a stale target
        # (FK of initial_positions), causing the robot to drive to a fixed
        # pose on startup regardless of joystick input.
        self._initial_command_type_set = False
        self._servo_status_sub = self.create_subscription(
            ServoStatus,
            f'{self.servo_node_name}/status',
            self._on_servo_status_ready,
            10,
            callback_group=subscriber_callback_group,
        )

    def _on_servo_status_ready(self, _msg: ServoStatus):
        if self._initial_command_type_set:
            return
        self._initial_command_type_set = True
        self.get_logger().info(
            'Servo status received; issuing initial switch_command_type(TWIST).')
        self.set_servo_command_type(command_type='TWIST')
        self.destroy_subscription(self._servo_status_sub)
        self._servo_status_sub = None

    # ============================================================================
    # INITIALIZATION AND PARAMETER HANDLING
    # ============================================================================

    def wait_for_services(self):
        # Timeout must comfortably exceed any TimerAction-delayed servo
        # startup in move_group.launch.py; otherwise __init__ returns
        # before the /servo_node/status gate is created and
        # switch_command_type is never issued.
        if not self.controller_manager_client.wait_for_service(timeout_sec=60.0):
            self.get_logger().error(
                'Controller manager service not available, cannot switch controllers')
            return False
        if not self.pause_servo_client.wait_for_service(timeout_sec=60.0):
            self.get_logger().error(
                'Start servo service not available, cannot start servo')
            return False
        if not self.pause_servo_client.wait_for_service(timeout_sec=60.0):
            self.get_logger().error(
                'Pause servo service not available, cannot pause servo')
            return False
        self.get_logger().info('All required services are available')
        return True

    def set_data_path(self, data_path):
        """
        Helper function to set the data path for the partitioner.
        :param data_path: The path to the data directory.
        """
        data_path = os.path.expandvars(data_path)
        if data_path.startswith('package://'):
            package_name, relative_path = data_path.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            data_path = os.path.join(
                package_path, 'share', package_name, relative_path)

        # If path doesn't exist, create it
        if not os.path.exists(data_path):
            data_path = '/tmp'

        self.data_path = data_path

    def load_viewpoints(self, filepath):
        self.viewpoints = []
        self._viewpoint_region_ids = []
        self.results_dict = {}
        if filepath == '':
            self.get_logger().warning('Viewpoints file cleared, no viewpoints loaded')
            return False

        # If file doesn't exist, look in data path
        if not os.path.exists(filepath):
            filepath = os.path.join(self.data_path, filepath)

        # Expand environment variables in filepath
        filepath = os.path.expandvars(filepath)

        with open(filepath, 'r') as f:
            results_dict = json.load(f)

        # Keep the full results dict so update_selection can size the
        # selected_mesh slider (planning/execution still use mesh 0 below).
        self.results_dict = results_dict

        mesh_dict = results_dict['meshes'][0]
        selected_algorithm = self.selected_traversal_algorithm or None
        vrp_orders = mesh_dict.get('vrp_orders', {})
        # VRP stores its optimized inter-region sequence in vrp_orders; TSP leaves mesh_dict['order'] unchanged.
        region_order = vrp_orders.get(selected_algorithm) or mesh_dict['order']

        try:
            for region_id in region_order:
                region_dict = mesh_dict['regions'][region_id]
                cluster_order = _resolve_order_indices(
                    region_dict['order'], selected_algorithm)
                region_viewpoints = []
                for cluster_id in cluster_order:
                    cluster_dict = region_dict['clusters'][cluster_id]
                    if not 'viewpoint' in cluster_dict:
                        self.get_logger().warning(
                            f'No viewpoint found in cluster {cluster_id}')
                        continue

                    position = cluster_dict['viewpoint']['position']
                    orientation = cluster_dict['viewpoint']['orientation']

                    viewpoint = PoseStamped()
                    # Viewpoints are authored in the mesh origin frame; the
                    # object_frame->model_frame TF (from tsdf_pose) places them.
                    # MoveIt resolves this frame via TF when moving to a goal.
                    viewpoint.header.frame_id = 'model_frame'
                    viewpoint.pose.position.x = position[0]
                    viewpoint.pose.position.y = position[1]
                    viewpoint.pose.position.z = position[2]
                    viewpoint.pose.orientation.x = orientation[0]
                    viewpoint.pose.orientation.y = orientation[1]
                    viewpoint.pose.orientation.z = orientation[2]
                    viewpoint.pose.orientation.w = orientation[3]

                    region_viewpoints.append(viewpoint)

                self.viewpoints.append(region_viewpoints)
                self._viewpoint_region_ids.append(region_id)
        except Exception as e:
            self.get_logger().error(
                f'Failed to load viewpoints from {filepath}: {e}')
            self.viewpoints = []
            self._viewpoint_region_ids = []
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

    def _make_descriptor(self, field_info, with_range=True):
        """Build a ParameterDescriptor from a selection_config entry.

        ``with_range`` controls whether the IntegerRange constraint is applied.
        It must be omitted at declaration time (see _declare_selection_parameters).
        """
        descriptor = ParameterDescriptor()
        descriptor.description = field_info.get('description', '')
        descriptor.additional_constraints = field_info.get('control', '')
        range_val = field_info.get('range')
        if field_info.get('type') == 'integer':
            descriptor.type = ParameterType.PARAMETER_INTEGER
        if with_range and range_val is not None and field_info.get('type') == 'integer':
            ir = IntegerRange()
            ir.from_value = int(range_val[0])
            ir.to_value = int(range_val[1])
            ir.step = 1
            descriptor.integer_range = [ir]
        return descriptor

    def _declare_selection_parameters(self):
        """Declare the selection parameters from self.selection_config.

        The IntegerRange constraint is intentionally NOT applied here. At
        declaration time the results file has not been loaded yet, so the real
        [0, count-1] range is unknown; the config's placeholder range is [0, 0].
        Declaring with that range makes rclpy reject any parameter override
        outside it (e.g. a saved ``navigation.selected_region: 3`` from a loaded
        results config), crashing node startup. The correct range is installed
        by ``update_selection`` -> ``_set_selection_descriptor`` immediately
        after the viewpoints load, which also clamps the override to valid data.
        """
        for field_name, field_info in self.selection_config.items():
            self.declare_parameter(
                field_name, field_info['value'],
                self._make_descriptor(field_info, with_range=False))

    def _set_selection_descriptor(self, name, count):
        """Resize the IntegerRange descriptor of a selection slider to [0, count-1]."""
        descriptor = ParameterDescriptor()
        descriptor.type = ParameterType.PARAMETER_INTEGER
        descriptor.description = self.selection_config[name]['description']
        descriptor.additional_constraints = 'slider'
        ir = IntegerRange()
        ir.from_value = 0
        ir.to_value = max(0, int(count) - 1)
        ir.step = 1
        descriptor.integer_range = [ir]
        self.set_descriptor(name, descriptor)

    def update_selection(self, mesh_index, region_index, viewpoint_index):
        """Clamp and apply the mesh/region/viewpoint selection.

        Mirrors the dynamic-descriptor behaviour the viewpoint_generation node
        used to provide: child indices reset when a parent changes, indices are
        clamped to the loaded data, the active viewpoint pose is resolved for
        execution, and the IntegerRange descriptors are resized to the live
        counts so the GUI sliders track the data. ``selected_mesh`` is
        visualization scope only — planning/execution always use mesh 0
        (``self.viewpoints``).

        :return: True if a valid viewpoint pose is selected, False otherwise.
        """
        # Reset child selections when a parent changes to avoid stale indices.
        if mesh_index != self._sel_mesh:
            region_index = 0
            viewpoint_index = 0
        if region_index != self._sel_region:
            viewpoint_index = 0

        number_of_meshes = len(self.results_dict.get('meshes', [])) \
            if self.results_dict else 0
        number_of_regions = len(self.viewpoints)

        # Clamp mesh/region; reset descendants when out of bounds.
        if mesh_index < 0 or mesh_index >= number_of_meshes:
            mesh_index = 0
        if region_index < 0 or region_index >= number_of_regions:
            region_index = 0
            viewpoint_index = 0

        number_of_viewpoints = (len(self.viewpoints[region_index])
                                if 0 <= region_index < number_of_regions else 0)
        if viewpoint_index < 0 or viewpoint_index >= number_of_viewpoints:
            viewpoint_index = 0

        # Resolve the active viewpoint pose for execution.
        if number_of_viewpoints > 0:
            self.selected_viewpoint = self.viewpoints[region_index][viewpoint_index]
            self.get_logger().info(
                f'Selected viewpoint {viewpoint_index} in region {region_index}')
        else:
            self.selected_viewpoint = None
            if number_of_regions == 0:
                self.get_logger().warning('No viewpoints loaded')

        self._sel_mesh = mesh_index
        self._sel_region = region_index
        self._sel_viewpoint = viewpoint_index

        # Write the clamped values back (blocked so we don't re-enter selection).
        for pname, value in (('navigation.selected_mesh', mesh_index),
                             ('navigation.selected_region', region_index),
                             ('navigation.selected_viewpoint', viewpoint_index)):
            param = rclpy.parameter.Parameter(
                pname, rclpy.Parameter.Type.INTEGER, value)
            self.block_next_param_callback = True
            self.set_parameters([param])

        # Resize the slider ranges to the live counts.
        self._set_selection_descriptor('navigation.selected_mesh', number_of_meshes)
        self._set_selection_descriptor('navigation.selected_region', number_of_regions)
        self._set_selection_descriptor('navigation.selected_viewpoint', number_of_viewpoints)

        return self.selected_viewpoint is not None

    def select_viewpoint(self, region_index, viewpoint_index):
        """Select a viewpoint by region/viewpoint index (keeping the current
        mesh). Thin wrapper over update_selection used by the inspection action.

        :return: True if a valid viewpoint pose is selected, False otherwise.
        """
        return self.update_selection(self._sel_mesh, region_index, viewpoint_index)

    def publish_selected_viewpoint(self):
        """
        Publish the currently selected viewpoint to the viewpoint topic.
        :return: None
        """
        if self.selected_viewpoint is not None:
            self.viewpoint_publisher.publish(self.selected_viewpoint)

    def turntable_status_callback(self, msg):
        self.flags.turntable_moving = msg.data

    # ============================================================================
    # STATE HANDLERS
    # ============================================================================

    # INITIALIZING
    def _handle_initializing(self):
        if self.flags.verbose:
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
        elif not self.flags.move_to_viewpoint and not self.flags.turntable_moving \
                and not self.flags.inspect_active:
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

    def _find_nearest_entry(self, region_id):
        if not self.find_nearest_viewpoint_client.service_is_ready():
            return 0
        req = FindNearestViewpoint.Request()
        req.region_idx = region_id
        future = self.find_nearest_viewpoint_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            return future.result().nearest_viewpoint_idx
        return 0

    def execute_inspect_region_callback(self, goal_handle):
        self.get_logger().info('Executing full region traversal...')
        self.flags.inspect_active = True

        for seq_idx, region_viewpoints in enumerate(self.viewpoints):
            region_id = (self._viewpoint_region_ids[seq_idx]
                         if seq_idx < len(self._viewpoint_region_ids) else seq_idx)
            entry_offset = self._find_nearest_entry(region_id)
            n = len(region_viewpoints)
            ordered = list(range(entry_offset, n)) + list(range(0, entry_offset))

            for i in ordered:
                self.select_viewpoint(seq_idx, i)
                self.flags.move_to_viewpoint = True
                while self.flags.move_to_viewpoint:
                    self.get_logger().info(
                        f'Moving to viewpoint {i} in region {seq_idx}...')
                    time.sleep(0.1)
                while self.flags.turntable_moving:
                    self.get_logger().info('Waiting for turntable to stop moving...')
                    time.sleep(0.1)

        self.flags.inspect_active = False
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

        success = True

        for param in params:
            # Viewpoints file parameter
            if param.name == 'settings.results_file' and param.type_ == param.Type.STRING:
                success = self.load_viewpoints(param.value)
                # Reclamp the selection and refresh the slider ranges to match
                # the newly loaded data.
                self.update_selection(
                    self._sel_mesh, self._sel_region, self._sel_viewpoint)
                self.get_logger().info(
                    f'Viewpoints file changed to: {param.value}')
            # Mesh/region/viewpoint selection — drives both the visualizer
            # (via the GUI) and the active viewpoint pose for execution.
            elif param.name == 'navigation.selected_mesh':
                self.update_selection(
                    param.value, self._sel_region, self._sel_viewpoint)
            elif param.name == 'navigation.selected_region':
                self.update_selection(
                    self._sel_mesh, param.value, self._sel_viewpoint)
            elif param.name == 'navigation.selected_viewpoint':
                self.update_selection(
                    self._sel_mesh, self._sel_region, param.value)
            # Traversal algorithm selection — record the preference and reload
            # viewpoints so the execution order follows the newly chosen path.
            # The parameter set itself always succeeds: an empty/unset
            # results_file just means there is nothing to reload yet.
            elif param.name == 'navigation.selected_traversal_algorithm':
                self.selected_traversal_algorithm = param.value
                results_file = self.get_parameter(
                    'settings.results_file').get_parameter_value().string_value
                if results_file:
                    self.load_viewpoints(results_file)
                    self.update_selection(
                        self._sel_mesh, self._sel_region, self._sel_viewpoint)

        result = SetParametersResult()
        result.successful = success

        return result


def main():
    rclpy.init()

    executor = MultiThreadedExecutor(num_threads=4)
    # executor = rclpy.executors.SingleThreadedExecutor()

    node = TaskPlanningNode()
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
