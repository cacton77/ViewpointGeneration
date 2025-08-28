#!/usr/bin/env python3
import rclpy
import os
import copy
import time
import yaml
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters, ListParameters
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType, Log
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_prefix

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

from std_srvs.srv import Trigger, SetBool
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.srv import ServoCommandType


class ROSThread(Node):

    node_name = 'gui'
    viewpoint_generation_node_name = 'viewpoint_generation'
    traversal_node_name = 'viewpoint_traversal'
    task_planning_node_name = 'inspection_task_planning'

    target_nodes = [viewpoint_generation_node_name,
                    traversal_node_name,
                    task_planning_node_name]

    robot_moving = False
    path = []  # Move to task planning node eventually

    def __init__(self, stream_id=0):
        super().__init__(self.node_name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('show_axes', True),
                ('show_grid', True),
                ('show_model_bounding_box', False),
                ('show_reticle', True),
                ('show_skybox', True),
                ('show_mesh', True),
                ('show_point_cloud', True),
                ('show_curvatures', True),
                ('show_regions', True),
                ('show_noise_points', True),
                ('show_fov_clusters', True),
                ('show_viewpoints', True),
                ('show_region_view_manifolds', True),
                ('show_path', False),
            ]
        )

        self.show_axes = self.get_parameter(
            'show_axes').get_parameter_value().bool_value
        self.show_grid = self.get_parameter(
            'show_grid').get_parameter_value().bool_value
        self.show_model_bounding_box = self.get_parameter(
            'show_model_bounding_box').get_parameter_value().bool_value
        self.show_reticle = self.get_parameter(
            'show_reticle').get_parameter_value().bool_value
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
        self.show_region_view_manifolds = self.get_parameter(
            'show_region_view_manifolds').get_parameter_value().bool_value
        self.show_path = self.get_parameter(
            'show_path').get_parameter_value().bool_value

        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

        # Connect to parameters of target nodes
        # Create service clients
        self.list_params_clients = {}
        self.get_params_clients = {}
        self.set_params_clients = {}
        set_params_cb_group = MutuallyExclusiveCallbackGroup()
        target_nodes_copy = self.target_nodes.copy()
        failed_targets = []
        for target_node in target_nodes_copy:
            list_params_client = self.create_client(
                ListParameters,
                f'{target_node}/list_parameters'
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
            if not list_params_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(
                    f'Failed to connect to {target_node}/list_parameters service')
                failed_targets.append(target_node)
                self.target_nodes.remove(target_node)
            elif not get_params_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(
                    f'Failed to connect to {target_node}/get_parameters service')
                failed_targets.append(target_node)
                self.target_nodes.remove(target_node)
            elif not set_params_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(
                    f'Failed to connect to {target_node}/set_parameters service')
                failed_targets.append(target_node)
                self.target_nodes.remove(target_node)
            else:
                self.list_params_clients[target_node] = list_params_client
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
        self.sampling_client = self.create_client(Trigger,
                                                  f'{self.viewpoint_generation_node_name}/sample_point_cloud',
                                                  callback_group=services_cb_group
                                                  )
        self.estimate_curvature_client = self.create_client(Trigger,
                                                            f'{self.viewpoint_generation_node_name}/estimate_curvature',
                                                            callback_group=services_cb_group
                                                            )
        self.region_growth_client = self.create_client(Trigger,
                                                       f'{self.viewpoint_generation_node_name}/region_growth',
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
                                                           f'{self.viewpoint_generation_node_name}/move_to_viewpoint',
                                                           callback_group=services_cb_group
                                                           )
        self.optimize_traversal_client = self.create_client(Trigger,
                                                            f'{self.viewpoint_generation_node_name}/optimize_traversal',
                                                            callback_group=services_cb_group
                                                            )

        # ----------- Create clients for task planning services -----------

        # ROSOUT log subscription
        self.create_subscription(
            Log,
            '/rosout',
            self.rosout_callback,
            10
        )
        self.log = []

        # Wait for services to be available
        self.wait_for_services()

        self.get_all_parameters()

    def rosout_callback(self, msg):
        text = f"[{msg.name}] {msg.msg}" if msg.name else msg.msg
        self.log.append(text)

    def set_param(self, param_name, new_value):
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

    def wait_for_services(self):
        """Wait for all required services to be available"""
        self.get_logger().info('Waiting for parameter services...')

        # Wait for viewpoint generation services
        while not self.sampling_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/sample_point_cloud service...')
        while not self.estimate_curvature_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/estimate_curvature service...')
        while not self.region_growth_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/region_growth service...')
        while not self.fov_clustering_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/fov_clustering service...')
        while not self.viewpoint_projection_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                f'Waiting for {self.viewpoint_generation_node_name}/viewpoint_projection service...')

    def save_parameters_to_file(self, file_path):
        """Save all current and connected inspection node parameters to a file"""
        output_dict = {}

        gui_params = self.get_node_parameters(self.node_name)
        output_dict[self.node_name] = {'ros__parameters': gui_params}
        viewpoint_generation_params = self.get_node_parameters(
            self.viewpoint_generation_node_name)
        output_dict[self.viewpoint_generation_node_name] = {
            'ros__parameters': viewpoint_generation_params}
        # viewpoint_traversal_params = self.get_node_parameters(self.traversal_node_name)
        # output_dict[self.traversal_node_name] = {'ros__parameters': viewpoint_traversal_params}

        # Write to file
        with open(file_path, 'w') as f:
            yaml.dump(output_dict, f, default_flow_style=False)

        self.get_logger().info(f'Parameters saved to {file_path}')

    def get_node_parameters(self, target_node_name):
        # Create parameter client for the target node
        param_client = self.create_client(
            ListParameters, f'/{target_node_name}/list_parameters')
        get_client = self.create_client(
            GetParameters, f'/{target_node_name}/get_parameters')

        if not param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f'Parameter service not available for {target_node_name}')
            return None

        # List all parameters
        list_request = ListParameters.Request()
        list_future = param_client.call_async(list_request)
        rclpy.spin_until_future_complete(self, list_future)

        if list_future.result() is not None:
            param_names = list_future.result().result.names

            # Get parameter values
            get_request = GetParameters.Request()
            get_request.names = param_names
            get_future = get_client.call_async(get_request)
            rclpy.spin_until_future_complete(self, get_future)

            if get_future.result() is not None:
                param_values = get_future.result().values

                # Create parameter dictionary
                params_dict = {}
                for name, value in zip(param_names, param_values):
                    params_dict[name] = self._parameter_value_to_python(value)

                return params_dict

        return None

    def _parameter_value_to_python(self, param_value):
        """Convert ROS parameter value to Python type"""
        if param_value.type == Parameter.Type.BOOL.value:
            return param_value.bool_value
        elif param_value.type == Parameter.Type.INTEGER.value:
            return param_value.integer_value
        elif param_value.type == Parameter.Type.DOUBLE.value:
            return param_value.double_value
        elif param_value.type == Parameter.Type.STRING.value:
            return param_value.string_value
        elif param_value.type == Parameter.Type.BYTE_ARRAY.value:
            return list(param_value.byte_array_value)
        elif param_value.type == Parameter.Type.BOOL_ARRAY.value:
            return list(param_value.bool_array_value)
        elif param_value.type == Parameter.Type.INTEGER_ARRAY.value:
            return list(param_value.integer_array_value)
        elif param_value.type == Parameter.Type.DOUBLE_ARRAY.value:
            return list(param_value.double_array_value)
        elif param_value.type == Parameter.Type.STRING_ARRAY.value:
            return list(param_value.string_array_value)
        else:
            return None

    # ============================================================================
    # VIEWPOINT GENERATION AND TRAVERSAL OPTIMIZATION
    # ============================================================================

    def sample_point_cloud(self):
        """Trigger the sampling service"""
        if not self.sampling_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Sampling service not available')
            return False

        request = Trigger.Request()
        future = self.sampling_client.call_async(request)
        future.add_done_callback(self.sample_point_cloud_future_callback)

    def sample_point_cloud_future_callback(self, future):
        """Callback for the sampling service future"""
        if future.result() is not None:
            self.get_logger().info(
                f'Point cloud sampling triggered successfully. {future.result().message}')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger point cloud sampling')
            return False

    def estimate_curvature(self):
        """Trigger the curvature estimation service"""
        if not self.estimate_curvature_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Curvature estimation service not available')
            return False

        request = Trigger.Request()
        future = self.estimate_curvature_client.call_async(request)
        future.add_done_callback(self.estimate_curvature_future_callback)

    def estimate_curvature_future_callback(self, future):
        """Callback for the curvature estimation service future"""
        if future.result() is not None:
            self.get_logger().info('Curvature estimation triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger curvature estimation')
            return False

    def region_growth(self):
        """Trigger the region growth service"""
        if not self.region_growth_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Region growth service not available')
            return False

        request = Trigger.Request()
        future = self.region_growth_client.call_async(request)
        future.add_done_callback(self.region_growth_future_callback)

    def region_growth_future_callback(self, future):
        """Callback for the region growth service future"""
        if future.result() is not None:
            self.get_logger().info('Region growth triggered successfully')
            self.get_all_parameters()
            return True
        else:
            self.get_logger().error('Failed to trigger region growth')
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
        self.set_parameter(self.viewpoint_generation_node_name,
                           'regions.selected_region', region_index)

    def select_cluster(self, cluster_index):
        """Select a cluster based on cluster index"""
        self.set_parameter(self.viewpoint_generation_node_name,
                           'regions.selected_cluster', cluster_index)

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

    # ============================================================================
    # TASK PLANNING
    # ============================================================================

    def move_to_viewpoint(self):
        pass

    def image_region(self):
        pass

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
                self.get_logger().info(
                    f'Getting parameters for {node_name}...')
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
                # Get parameter values
                get_request = GetParameters.Request()
                get_request.names = param_names
                get_future = self.get_params_clients[node_name].call_async(
                    get_request)
                get_future.add_done_callback(
                    lambda future, node_name=node_name, param_names=param_names: self.get_all_parameter_values_future_callback(future, node_name, param_names))
            else:
                self.get_logger().info('No parameters found in target node')
        else:
            self.get_logger().error('Failed to list parameters')

    def get_all_parameter_values_future_callback(self, get_future, node_name, param_names):
        """Callback for the get parameters future"""
        if get_future.result() is not None:
            get_response = get_future.result()

            # Check the correct attribute name for parameter values
            if hasattr(get_response, 'values'):
                param_values = get_response.values
            else:
                # Debug: print available attributes
                self.get_logger().info(
                    f'GetParameters response attributes: {dir(get_response)}')
                self.get_logger().error('Could not find parameter values in response')
                return

            # Store parameters in dictionary
            for name, value in zip(param_names, param_values):
                if name == 'use_sim_time':
                    # Skip the use_sim_time parameter
                    continue

                param_value = self.extract_parameter_value(value)

                # If param name ends in '.file' and starts with 'package://', replace it with the package path
                if name.endswith('.file') and isinstance(param_value, str) and param_value.startswith('package://'):
                    package_name, relative_path = param_value.split(
                        'package://', 1)[1].split('/', 1)
                    package_prefix = get_package_prefix(package_name)
                    param_value = os.path.join(
                        package_prefix, 'share', package_name, relative_path)

                if name in self.parameters_dict[node_name]:
                    update_flag = self.parameters_dict[node_name][name]['value'] != param_value
                    # self.get_logger().info(
                    # f'Parameter {name} updated: {update_flag} (old: \'{self.parameters_dict[name]["value"]}\', new: \'{param_value}\')')
                else:
                    update_flag = True

                param_info = {
                    'name': name,
                    'type': self.get_parameter_type_string(value.type),
                    'value': param_value,
                    'update_flag': update_flag
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

    def set_parameter(self, node_name, param_name, new_value):
        """Set a parameter on the target node"""
        if param_name not in self.parameters_dict[node_name]:
            self.get_logger().error(f'Parameter {param_name} not found')
            return False

        try:
            self.get_logger().info(
                f'Setting parameter {param_name} to {new_value}')

            param_info = self.parameters_dict[node_name][param_name]

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
            else:
                self.get_logger().error(
                    f'Unsupported parameter type: {param_info["type"]}')
                return False

            param_msg.value = param_value

            # Send set parameter request with the message object
            set_request = SetParameters.Request()
            # Use param_msg, not Parameter object
            set_request.parameters = [param_msg]

            set_future = self.set_params_clients[node_name].call_async(
                set_request)
            # Add done callback lambda with parameter name and new value
            set_future.add_done_callback(
                lambda f, node_name=node_name, param_name=param_name, new_value=new_value: self.set_parameter_future_callback(f, node_name, param_name, new_value))

        except Exception as e:
            self.get_logger().error(f'Error setting parameter: {str(e)}')
            return False

    def set_parameter_future_callback(self, future, node_name, param_name, new_value):
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
