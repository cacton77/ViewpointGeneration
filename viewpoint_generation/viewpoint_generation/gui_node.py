#!/usr/bin/env python3
import rclpy
import os
import copy
import time
import yaml
import threading
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import ListParameters, DescribeParameters, GetParameters, SetParameters
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType, Log
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_prefix

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

from std_srvs.srv import Trigger, SetBool
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from viewpoint_generation_interfaces.action import InspectRegion
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.srv import ServoCommandType


class ROSThread(Node):

    node_name = 'gui'
    viewpoint_generation_node_name = 'viewpoint_generation'
    traversal_node_name = 'viewpoint_traversal'
    task_planning_node_name = 'task_planning'
    autofocus_node_name = 'autofocus'
    orientation_control_node_name = 'orientation_controller'
    admittance_control_node_name = 'admittance_control'
    teleop_node_name = 'teleop'
    tsdf_node_name = 'tsdf_pose'

    target_nodes = [viewpoint_generation_node_name,
                    traversal_node_name,
                    task_planning_node_name,
                    autofocus_node_name,
                    orientation_control_node_name,
                    admittance_control_node_name,
                    teleop_node_name,
                    tsdf_node_name,
                    node_name]

    flags = {
        'inspect_region_active': False
    }

    robot_moving = False
    path = []  # Move to task planning node eventually

    status_message = "HELLO WORLD"

    def __init__(self, stream_id=0):
        super().__init__(self.node_name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('file_name', 'default'),
                ('data_path', '/tmp'),
                ('show_axes', False),
                ('show_grid', False),
                ('show_model_bounding_box', False),
                ('show_skybox', False),
                ('show_mesh', True),
                ('show_point_cloud', True),
                ('show_curvatures', True),
                ('show_regions', True),
                ('show_noise_points', True),
                ('show_fov_clusters', True),
                ('show_viewpoints', True),
                ('show_joint_path', True),
                ('show_unreachable', True),
                ('show_blind_spots', True),
            ]
        )

        self.file_name = self.get_parameter(
            'file_name').get_parameter_value().string_value
        self.show_axes = self.get_parameter(
            'show_axes').get_parameter_value().bool_value
        self.show_grid = self.get_parameter(
            'show_grid').get_parameter_value().bool_value
        self.show_model_bounding_box = self.get_parameter(
            'show_model_bounding_box').get_parameter_value().bool_value
        self.show_skybox = self.get_parameter(
            'show_skybox').get_parameter_value().bool_value
        self.show_mesh = self.get_parameter(
            'show_mesh').get_parameter_value().bool_value
        self.show_point_cloud = self.get_parameter(
            'show_point_cloud').get_parameter_value().bool_value
        self.show_curvatures = self.get_parameter(
            'show_curvatures').get_parameter_value().bool_value
        self.show_regions = self.get_parameter(
            'show_regions').get_parameter_value().bool_value
        self.show_noise_points = self.get_parameter(
            'show_noise_points').get_parameter_value().bool_value
        self.show_fov_clusters = self.get_parameter(
            'show_fov_clusters').get_parameter_value().bool_value
        self.show_viewpoints = self.get_parameter(
            'show_viewpoints').get_parameter_value().bool_value
        self.show_joint_path = self.get_parameter(
            'show_joint_path').get_parameter_value().bool_value
        self.show_unreachable = self.get_parameter(
            'show_unreachable').get_parameter_value().bool_value
        self.show_blind_spots = self.get_parameter(
            'show_blind_spots').get_parameter_value().bool_value
        self.data_path = self.get_parameter(
            'data_path').get_parameter_value().string_value

        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

        # Connect to parameters of target nodes
        # Create service clients
        self.list_params_clients = {}
        self.describe_params_clients = {}
        self.get_params_clients = {}
        self.set_params_clients = {}
        set_params_cb_group = MutuallyExclusiveCallbackGroup()
        failed_targets = []
        for target_node in self.target_nodes.copy():
            list_params_client = self.create_client(
                ListParameters,
                f'{target_node}/list_parameters'
            )
            describe_params_client = self.create_client(
                DescribeParameters,
                f'{target_node}/describe_parameters'
            )
            get_params_client = self.create_client(
                GetParameters,
                f'{target_node}/get_parameters'
            )
            set_params_client = self.create_client(
                SetParameters,
                f'{target_node}/set_parameters',
                callback_group=set_params_cb_group
            )
            # Wait for the list parameters service to be available as a test of connectivity to the node
            if not list_params_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().warning(
                    f'Failed to connect to {target_node}/list_parameters service')
                failed_targets.append(target_node)
                self.target_nodes.remove(target_node)
            else:
                self.list_params_clients[target_node] = list_params_client
                self.describe_params_clients[target_node] = describe_params_client
                self.get_params_clients[target_node] = get_params_client
                self.set_params_clients[target_node] = set_params_client

        self.get_logger().info(
            f'Parameter Manager Node started for targets: {self.target_nodes}')
        self.get_logger().warning(
            f'Failed to connect to the following targets: {failed_targets}')

        # Dictionary to store parameter information
        self.parameters_dict = {}
        for target_node in self.target_nodes:
            self.parameters_dict[target_node] = {}

        # ------------- Create clients for viewpoint generation services -------------
        services_cb_group = MutuallyExclusiveCallbackGroup()
        self.segment_regions_client = self.create_client(Trigger,
                                                         f'{self.viewpoint_generation_node_name}/segment_regions',
                                                         callback_group=services_cb_group
                                                         )
        self.fov_clustering_client = self.create_client(Trigger,
                                                        f'{self.viewpoint_generation_node_name}/fov_clustering',
                                                        callback_group=services_cb_group
                                                        )
        self.viewpoint_projection_client = self.create_client(Trigger,
                                                              f'{self.viewpoint_generation_node_name}/viewpoint_projection',
                                                              callback_group=services_cb_group
                                                              )

        # Create passthrough client for viewpoint traversal service
        self.move_to_viewpoint_client = self.create_client(Trigger,
                                                           f'{self.task_planning_node_name}/move_to_viewpoint',
                                                           callback_group=services_cb_group
                                                           )

        self.inspect_region_action_client = ActionClient(
            self, InspectRegion, self.task_planning_node_name + '/inspect_region')
        self.optimize_traversal_client = self.create_client(Trigger,
                                                            f'{self.viewpoint_generation_node_name}/optimize_traversal',
                                                            callback_group=services_cb_group
                                                            )

        # ----------- Create clients for task planning services -----------

        # ----------- Create clients for TSDF services -----------

        self.reset_tsdf_client = self.create_client(Trigger,
                                                    f'{self.tsdf_node_name}/reset',
                                                    callback_group=services_cb_group
                                                    )

        self.log = []

        # Wait for services to be available
        self.get_logger().info('Waiting for parameter services...')

        # Wait for viewpoint generation services
        while not self.segment_regions_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/segment_regions service...')
        while not self.fov_clustering_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/fov_clustering service...')
        while not self.viewpoint_projection_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/viewpoint_projection service...')

        # Update internal dict of parameters on timer
        self.create_timer(1.0, self.get_all_parameters)

    def load_config(self, yaml_file):
        # Check if the file exists
        if not os.path.exists(yaml_file):
            self.get_logger().error(f"Config file not found: {yaml_file}")
            return

        self.initialized = False

        with open(yaml_file, 'r') as file:
            config_dict = yaml.safe_load(file)

        for node_name, ros_parameters in config_dict.items():
            params = ros_parameters['ros__parameters']
            for param_name, param_value in params.items():
                try:
                    if node_name == self.node_name:
                        self.set_parameter(param_name, param_value)
                    else:
                        self.set_target_node_parameter(
                            node_name, param_name, param_value)
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to set parameter {param_name} for {node_name}: {e}")

        self.initialized = True

        self.file_name = os.path.basename(yaml_file).replace('.yaml', '')

        self.get_all_parameters()

    def save_parameters_to_file(self, file_path):
        """Save all current and connected inspection node parameters to a file"""
        if os.path.basename(file_path) == 'new.yaml':
            self.get_logger().warn(
                "Cannot save over 'new.yaml'. Use Save As to save with a different name.")
            return

        self.file_name = os.path.basename(file_path).replace('.yaml', '')
        self.set_parameter('file_name', self.file_name)

        output_dict = {
            node_name: {
                'ros__parameters': {
                    name: info['value']
                    for name, info in params.items()
                }
            }
            for node_name, params in self.parameters_dict.items()
            if params
        }

        with open(file_path, 'w') as f:
            yaml.dump(output_dict, f, default_flow_style=False)

        self.get_logger().info(f'Parameters saved to {file_path}')

    # ============================================================================
    # VIEWPOINT GENERATION AND TRAVERSAL OPTIMIZATION
    # ============================================================================

    def segment_regions(self):
        """Trigger the region segmentation service"""
        if not self.segment_regions_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Region segmentation service not available')
            return False

        request = Trigger.Request()
        future = self.segment_regions_client.call_async(request)
        future.add_done_callback(self.segment_regions_future_callback)

    def segment_regions_future_callback(self, future):
        """Callback for the region segmentation service future"""
        if future.result() is not None:
            self.get_logger().info('Region segmentation triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger region segmentation')
            return False

    def fov_clustering(self):
        """Trigger the FOV clustering service"""
        if not self.fov_clustering_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('FOV clustering service not available')
            return False

        request = Trigger.Request()
        future = self.fov_clustering_client.call_async(request)
        future.add_done_callback(self.fov_clustering_future_callback)

    def fov_clustering_future_callback(self, future):
        """Callback for the FOV clustering service future"""
        if future.result() is not None:
            self.get_logger().info('FOV clustering triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger FOV clustering')
            return False

    def project_viewpoints(self):
        """Trigger the viewpoint projection service"""
        if not self.viewpoint_projection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Viewpoint projection service not available')
            return False

        request = Trigger.Request()
        future = self.viewpoint_projection_client.call_async(request)
        future.add_done_callback(self.project_viewpoints_future_callback)

    def project_viewpoints_future_callback(self, future):
        """Callback for the viewpoint projection service future"""
        if future.result() is not None:
            self.get_logger().info('Viewpoint projection triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger viewpoint projection')
            return False

    def select_region(self, region_index):
        self.set_target_node_parameter(self.task_planning_node_name,
                                       'navigation.selected_region', region_index)

    def select_cluster(self, cluster_index):
        """Select a cluster based on cluster index"""
        self.set_target_node_parameter(self.task_planning_node_name,
                                       'navigation.selected_viewpoint', cluster_index)

    def select_traversal_algorithm(self, algorithm):
        """Select which TSP algorithm's path to follow (visualization +
        execution). Sets the parameter on the task_planning node."""
        self.set_target_node_parameter(self.task_planning_node_name,
                                       'navigation.selected_traversal_algorithm', algorithm)

    def set_results_file(self, file_path):
        """Point the task_planning node at the loaded results file so it
        plans/executes over the same viewpoints the GUI is visualizing."""
        self.set_target_node_parameter(self.task_planning_node_name,
                                       'settings.results_file', file_path)

    def optimize_traversal(self):
        """Optimize the viewpoint traversal path"""
        if not self.optimize_traversal_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Optimize traversal service not available')
            return False

        request = Trigger.Request()
        future = self.optimize_traversal_client.call_async(request)
        future.add_done_callback(self.optimize_traversal_future_callback)

    def optimize_traversal_future_callback(self, future):
        """Callback for the optimize traversal service future"""
        if future.result() is not None:
            self.get_logger().info('Optimize traversal triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger optimize traversal')
            return False

    def clear_traversal_paths(self):
        """Set clear_paths=True on the traversal node, then trigger optimize_traversal."""
        node = self.traversal_node_name
        if node not in self.set_params_clients:
            self.get_logger().warning(f'Node {node} not connected')
            return
        param_msg = ParameterMsg()
        param_msg.name = 'clear_paths'
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = True
        param_msg.value = pv
        req = SetParameters.Request()
        req.parameters = [param_msg]
        future = self.set_params_clients[node].call_async(req)
        future.add_done_callback(lambda f: self.optimize_traversal())

    # ============================================================================
    # TASK PLANNING
    # ============================================================================

    def move_to_viewpoint(self):
        future = self.move_to_viewpoint_client.call_async(Trigger.Request())
        future.add_done_callback(self.move_to_viewpoint_future_callback)

    def move_to_viewpoint_future_callback(self, future):
        if future.result() is not None:
            self.get_logger().info('Move to viewpoint triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger move to viewpoint')
            return False

    def inspect_region(self):
        self.flags['inspect_region_active'] = True

        future = self.inspect_region_action_client.send_goal_async(
            InspectRegion.Goal(),
            feedback_callback=self.inspect_region_feedback_callback)
        future.add_done_callback(self.inspect_region_goal_response_callback)

    def inspect_region_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Inspect region goal rejected')
            return

        self.get_logger().info('Inspect region goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(
            self.inspect_region_get_result_callback)

    def inspect_region_get_result_callback(self, future):
        if future.result() is not None:
            self.get_logger().info('Inspect region completed successfully')
            self.get_all_parameters()
        else:
            self.get_logger().error('Failed to trigger inspect region')

    def inspect_region_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Inspect region feedback: {feedback.message}')
        self.get_all_parameters()

    # ============================================================================
    # TSDF SERVICES
    # ============================================================================

    def reset_tsdf(self):
        """Trigger the reset TSDF service"""
        if not self.reset_tsdf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Reset TSDF service not available')
            return False

        request = Trigger.Request()
        future = self.reset_tsdf_client.call_async(request)
        future.add_done_callback(self.reset_tsdf_future_callback)

    def reset_tsdf_future_callback(self, future):
        """Callback for the reset TSDF service future"""
        if future.result() is not None:
            self.get_logger().info('Reset TSDF triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger reset TSDF')
            return False

    # ============================================================================
    # PARAMETER MANAGEMENT
    # ============================================================================

    def expand_params_dict(self):
        expanded = {}
        for target_node, flat_params in self.parameters_dict.items():
            expanded[target_node] = self.expand_dict_keys(flat_params)
        return expanded

    def expand_dict_keys(self, flat_dict):
        """
        Expand a flat dictionary with period-delimited keys into a nested dictionary structure.

        Args:
            flat_dict (dict): Dictionary with keys like 'a.b.c.d'

        Returns:
            dict: Nested dictionary structure
        """
        expanded = {}

        for key, value in flat_dict.items():
            # Split the key by periods
            key_parts = key.split('.')

            # Navigate/create the nested structure
            current_dict = expanded

            # Process all parts except the last one (create nested dicts)
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]

            # Set the final value
            current_dict[key_parts[-1]] = value

        return expanded

    def collapse_dict_keys(self, parameters_dict):
        """
        Collapse a nested dictionary structure into a flat dictionary with period-delimited keys.

        Args:
            parameters_dict (dict): Nested dictionary structure

        Returns:
            dict: Flat dictionary with keys like 'a.b.c.d'
        """
        collapsed = {}

        def _collapse(current_dict, parent_key=''):
            for key, value in current_dict.items():

                new_key = f"{parent_key}.{key}" if parent_key else key

                if 'name' in value and 'type' in value and 'value' in value:
                    collapsed[new_key] = {
                        'name': value['name'],
                        'type': value['type'],
                        'value': value['value'],
                        'update_flag': False
                    }
                elif isinstance(value, dict):
                    _collapse(value, new_key)
                else:
                    collapsed[new_key] = value

        _collapse(parameters_dict)

        self.parameters_dict = collapsed

    def get_all_parameters(self):
        """Get all parameters from the target node"""
        try:
            # First, list all parameter names
            for node_name, list_client in self.list_params_clients.items():
                list_request = ListParameters.Request()
                list_future = list_client.call_async(list_request)
                list_future.add_done_callback(
                    lambda future, node_name=node_name: self.get_all_parameters_future_callback(future, node_name))
        except Exception as e:
            self.get_logger().error(f'Error getting parameters: {str(e)}')

    def get_all_parameters_future_callback(self, list_future, node_name):
        """Callback for the list parameters future"""

        if list_future.result() is not None:
            list_response = list_future.result()

            # Check the correct attribute name for parameter names
            if hasattr(list_response, 'result') and hasattr(list_response.result, 'names'):
                param_names = list_response.result.names
            elif hasattr(list_response, 'names'):
                param_names = list_response.names
            else:
                # Debug: print available attributes
                self.get_logger().info(
                    f'ListParameters response attributes: {dir(list_response)}')
                self.get_logger().error('Could not find parameter names in response')
                return

            if param_names:
                # Describe parameters to retrieve descriptors (ranges, constraints)
                describe_request = DescribeParameters.Request()
                describe_request.names = param_names
                describe_future = self.describe_params_clients[node_name].call_async(
                    describe_request)
                describe_future.add_done_callback(
                    lambda future, node_name=node_name, param_names=param_names: self.get_all_parameter_descriptors_future_callback(future, node_name, param_names))
            else:
                self.get_logger().info('No parameters found in target node')
        else:
            self.get_logger().error('Failed to list parameters')

    def get_all_parameter_descriptors_future_callback(self, describe_future, node_name, param_names):
        """Callback for the describe parameters future; fires GetParameters next."""
        descriptors = {}
        if describe_future.result() is not None:
            for desc in describe_future.result().descriptors:
                descriptors[desc.name] = desc
        else:
            self.get_logger().warning(
                f'Failed to get parameter descriptors for {node_name}')

        get_request = GetParameters.Request()
        get_request.names = param_names
        get_future = self.get_params_clients[node_name].call_async(get_request)
        get_future.add_done_callback(
            lambda future, node_name=node_name, param_names=param_names, descriptors=descriptors:
                self.get_all_parameter_values_future_callback(future, node_name, param_names, descriptors))

    def get_all_parameter_values_future_callback(self, get_future, node_name, param_names, descriptors):
        """Callback for the get parameters future"""
        if get_future.result() is not None:
            get_response = get_future.result()

            param_values = get_response.values

            # Store parameters in dictionary
            for name, value in zip(param_names, param_values):
                if name == 'use_sim_time':
                    # Skip the use_sim_time parameter
                    continue
                elif name == 'start_type_description_service':
                    # Skip the start_type_description_service parameter
                    continue

                param_value = self.extract_parameter_value(value)

                desc = descriptors.get(name)

                # ------- Param info dict structure -------
                if desc and desc.floating_point_range:
                    range = (
                        desc.floating_point_range[0].from_value,
                        desc.floating_point_range[0].to_value,
                    )
                elif desc and desc.integer_range:
                    range = (
                        desc.integer_range[0].from_value,
                        desc.integer_range[0].to_value,
                    )
                else:
                    range = None

                if name in self.parameters_dict[node_name]:
                    value_changed = self.parameters_dict[node_name][name]['value'] != param_value
                    range_changed = self.parameters_dict[node_name][name].get(
                        'range') != range
                    # Preserve an existing True flag — concurrent polls from load_config
                    # can clobber it back to False before the GUI tick has a chance to act.
                    update_flag = self.parameters_dict[node_name][name][
                        'update_flag'] or value_changed or range_changed
                else:
                    update_flag = True
                param_info = {
                    'name': name,
                    'type': self.get_parameter_type_string(value.type),
                    'value': param_value,
                    'update_flag': update_flag,
                    'description': desc.description if desc else '',
                    'control': desc.additional_constraints if desc else '',
                    'range': range,
                    'read_only': desc.read_only if desc else False,
                }
                self.parameters_dict[node_name][name] = param_info
        else:
            self.get_logger().error('Failed to get parameter values')

    def get_parameter_type_string(self, param_type):
        """Convert parameter type enum to string"""
        type_map = {
            ParameterType.PARAMETER_NOT_SET: 'not_set',
            ParameterType.PARAMETER_BOOL: 'bool',
            ParameterType.PARAMETER_INTEGER: 'integer',
            ParameterType.PARAMETER_DOUBLE: 'double',
            ParameterType.PARAMETER_STRING: 'string',
            ParameterType.PARAMETER_BYTE_ARRAY: 'byte_array',
            ParameterType.PARAMETER_BOOL_ARRAY: 'bool_array',
            ParameterType.PARAMETER_INTEGER_ARRAY: 'integer_array',
            ParameterType.PARAMETER_DOUBLE_ARRAY: 'double_array',
            ParameterType.PARAMETER_STRING_ARRAY: 'string_array'
        }
        return type_map.get(param_type, 'unknown')

    def extract_parameter_value(self, param_value):
        """Extract the actual value from ParameterValue message"""
        if param_value.type == ParameterType.PARAMETER_BOOL:
            return param_value.bool_value
        elif param_value.type == ParameterType.PARAMETER_INTEGER:
            return param_value.integer_value
        elif param_value.type == ParameterType.PARAMETER_DOUBLE:
            return param_value.double_value
        elif param_value.type == ParameterType.PARAMETER_STRING:
            return param_value.string_value
        elif param_value.type == ParameterType.PARAMETER_BYTE_ARRAY:
            return list(param_value.byte_array_value)
        elif param_value.type == ParameterType.PARAMETER_BOOL_ARRAY:
            return list(param_value.bool_array_value)
        elif param_value.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            return list(param_value.integer_array_value)
        elif param_value.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return list(param_value.double_array_value)
        elif param_value.type == ParameterType.PARAMETER_STRING_ARRAY:
            return list(param_value.string_array_value)
        else:
            return None

    def print_parameters(self):
        """Print all stored parameters"""
        self.get_logger().info('Retrieved Parameters:')
        self.get_logger().info('-' * 50)
        for node_name, node_dict in self.parameters_dict.items():
            self.get_logger().info(f"Node: {node_name}")
            for name, info in node_dict.items():
                self.get_logger().info(f"Name: {info['name']}")
                self.get_logger().info(f"Type: {info['type']}")
                self.get_logger().info(f"Value: {info['value']}")
                self.get_logger().info(f"Update: {info['update_flag']}")
            self.get_logger().info('-' * 30)

    def set_parameter(self, param_name, new_value):
        """ Set a parameter of this node """
        if type(new_value) is bool:
            param_type = rclpy.Parameter.Type.BOOL
        elif type(new_value) is int:
            param_type = rclpy.Parameter.Type.INTEGER
        elif type(new_value) is float:
            param_type = rclpy.Parameter.Type.DOUBLE
        elif type(new_value) is str:
            param_type = rclpy.Parameter.Type.STRING
        else:
            self.get_logger().error(
                f'Unsupported parameter type: {type(new_value)} for {param_name}')
            return
        param = rclpy.parameter.Parameter(
            param_name,
            param_type,
            new_value
        )
        self.set_parameters([param])

    def set_target_node_parameter(self, target_node, param_name, new_value):
        """Set a parameter on the target node"""
        if target_node not in self.target_nodes and target_node != self.node_name:
            self.get_logger().warning(f'Node {target_node} not connected')
            return False
        if param_name not in self.parameters_dict[target_node]:
            self.get_logger().warning(f'Parameter {param_name} not found')
            return False

        try:
            self.get_logger().info(
                f'Setting parameter {param_name} to {new_value}')

            param_info = self.parameters_dict[target_node][param_name]

            # Create the message objects (NOT rclpy.parameter.Parameter)
            param_msg = ParameterMsg()
            param_msg.name = param_name

            param_value = ParameterValue()

            # Set the value based on type
            if param_info['type'] == 'bool':
                param_value.type = ParameterType.PARAMETER_BOOL
                param_value.bool_value = bool(new_value)
            elif param_info['type'] == 'integer':
                param_value.type = ParameterType.PARAMETER_INTEGER
                param_value.integer_value = int(new_value)
            elif param_info['type'] == 'double':
                param_value.type = ParameterType.PARAMETER_DOUBLE
                param_value.double_value = float(new_value)
            elif param_info['type'] == 'string':
                param_value.type = ParameterType.PARAMETER_STRING
                param_value.string_value = str(new_value)
            elif param_info['type'] == 'double_array':
                param_value.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                param_value.double_array_value = [float(v) for v in new_value]
            elif param_info['type'] == 'integer_array':
                param_value.type = ParameterType.PARAMETER_INTEGER_ARRAY
                param_value.integer_array_value = [int(v) for v in new_value]
            else:
                self.get_logger().error(
                    f'Unsupported parameter type: {param_info["type"]}')
                return False

            param_msg.value = param_value

            # Send set parameter request with the message object
            set_request = SetParameters.Request()
            # Use param_msg, not Parameter object
            set_request.parameters = [param_msg]

            set_future = self.set_params_clients[target_node].call_async(
                set_request)
            # Add done callback lambda with parameter name and new value
            set_future.add_done_callback(
                lambda f, node_name=target_node, param_name=param_name, new_value=new_value: self.set_target_node_parameter_future_callback(f, node_name, param_name, new_value))

        except Exception as e:
            self.get_logger().error(f'Error setting parameter: {str(e)}')
            return False

    def set_target_node_parameter_future_callback(self, future, node_name, param_name, new_value):
        """Callback for the set parameter future"""

        if future.result() is not None:
            results = future.result().results
            if results and results[0].successful:
                self.get_logger().info(
                    f'Successfully set parameter {node_name}/{param_name} to \'{new_value}\'')
                self.get_all_parameters()
                return True
            else:
                reason = results[0].reason if results else 'Unknown error'
                self.get_logger().error(
                    f'Failed to set parameter {node_name}/{param_name}: {reason}')
                return True
        else:
            self.get_logger().error('Service call failed')
            return False

    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame

    def update(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.spin()
