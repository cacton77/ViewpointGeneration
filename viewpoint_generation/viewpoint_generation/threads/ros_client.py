#!/usr/bin/env python3
import rclpy
import os
import time
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters, ListParameters
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_prefix


class ROSThread(Node):
    def __init__(self, stream_id=0):
        super().__init__('gui_node')
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

        # Dictionary to store parameter information
        self.parameters_dict = {}
        self.parameters_dict_collapsed = {}

        # Target node name (change this to your target node)
        # Replace with actual node name
        self.target_node_name = '/viewpoint_generation_node'

        # Create service clients
        self.list_params_client = self.create_client(
            ListParameters,
            f'{self.target_node_name}/list_parameters'
        )

        self.get_params_client = self.create_client(
            GetParameters,
            f'{self.target_node_name}/get_parameters'
        )

        set_params_cb_group = MutuallyExclusiveCallbackGroup()
        self.set_params_client = self.create_client(
            SetParameters,
            f'{self.target_node_name}/set_parameters',
            callback_group=set_params_cb_group
        )

        self.get_logger().info(
            f'Parameter Manager Node started for target: {self.target_node_name}')

        # Create clients for viewpoint generation services
        services_cb_group = MutuallyExclusiveCallbackGroup()
        self.sampling_client = self.create_client(Trigger,
                                                  f'{self.target_node_name}/sample_point_cloud',
                                                  callback_group=services_cb_group
                                                  )
        self.estimate_curvature_client = self.create_client(Trigger,
                                                            f'{self.target_node_name}/estimate_curvature',
                                                            callback_group=services_cb_group
                                                            )
        self.region_growth_client = self.create_client(Trigger,
                                                       f'{self.target_node_name}/region_growth',
                                                       callback_group=services_cb_group
                                                       )
        self.fov_clustering_client = self.create_client(Trigger,
                                                        f'{self.target_node_name}/fov_clustering',
                                                        callback_group=services_cb_group
                                                        )

        # Wait for services to be available
        self.wait_for_services()

        self.get_all_parameters()

    def wait_for_services(self):
        """Wait for all required services to be available"""
        self.get_logger().info('Waiting for parameter services...')

        # Wait for list parameters service
        while not self.list_params_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for {self.target_node_name}/list_parameters service...')

        # Wait for get parameters service
        while not self.get_params_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for {self.target_node_name}/get_parameters service...')

        # Wait for set parameters service
        while not self.set_params_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for {self.target_node_name}/set_parameters service...')

        self.get_logger().info('All parameter services are available!')



    def sample_point_cloud(self):
        """Trigger the sampling service"""
        if not self.sampling_client.wait_for_service(timeout_sec=1.0):
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
        if not self.estimate_curvature_client.wait_for_service(timeout_sec=1.0):
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
        if not self.region_growth_client.wait_for_service(timeout_sec=1.0):
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
        
    def expand_dict_keys(self):
        """
        Expand a flat dictionary with period-delimited keys into a nested dictionary structure.

        Args:
            flat_dict (dict): Dictionary with keys like 'a.b.c.d'

        Returns:
            dict: Nested dictionary structure
        """
        expanded = {}

        for key, value in self.parameters_dict.items():
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
            list_request = ListParameters.Request()
            list_future = self.list_params_client.call_async(list_request)
            list_future.add_done_callback(
                self.get_all_parameters_future_callback)
        except Exception as e:
            self.get_logger().error(f'Error getting parameters: {str(e)}')

    def get_all_parameters_future_callback(self, list_future):
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
                get_future = self.get_params_client.call_async(get_request)
                get_future.add_done_callback(
                    lambda future: self.get_all_parameter_values_future_callback(future, param_names))
            else:
                self.get_logger().info('No parameters found in target node')
        else:
            self.get_logger().error('Failed to list parameters')

    def get_all_parameter_values_future_callback(self, get_future, param_names):
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

                # If name is model.mesh.file and starts with 'package://', replace it with the package path
                if name.endswith('.file') and isinstance(param_value, str) and param_value.startswith('package://'):
                    package_name, relative_path = param_value.split(
                        'package://', 1)[1].split('/', 1)
                    package_prefix = get_package_prefix(package_name)
                    param_value = os.path.join(
                        package_prefix, 'share', package_name, relative_path)

                if name in self.parameters_dict:
                    update_flag = self.parameters_dict[name]['value'] != param_value
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
                self.parameters_dict[name] = param_info

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
        for name, info in self.parameters_dict.items():
            self.get_logger().info(f"Name: {info['name']}")
            self.get_logger().info(f"Type: {info['type']}")
            self.get_logger().info(f"Value: {info['value']}")
            self.get_logger().info('-' * 30)

    def set_parameter(self, param_name, new_value):
        """Set a parameter on the target node"""
        if param_name not in self.parameters_dict:
            self.get_logger().error(f'Parameter {param_name} not found')
            return False

        try:
            self.get_logger().info(
                f'Setting parameter {param_name} to {new_value}')

            param_info = self.parameters_dict[param_name]

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

            set_future = self.set_params_client.call_async(set_request)
            # Add done callback lambda with parameter name and new value
            set_future.add_done_callback(
                lambda f, param_name=param_name, new_value=new_value: self.set_parameter_future_callback(f, param_name, new_value))

        except Exception as e:
            self.get_logger().error(f'Error setting parameter: {str(e)}')
            return False

    def set_parameter_future_callback(self, future, param_name, new_value):
        """Callback for the set parameter future"""

        if future.result() is not None:
            results = future.result().results
            if results and results[0].successful:
                self.get_logger().info(
                    f'Successfully set parameter {param_name} to \'{new_value}\'')
                self.get_all_parameters()
                return True
            else:
                reason = results[0].reason if results else 'Unknown error'
                self.get_logger().error(
                    f'Failed to set parameter {param_name}: {reason}')
                return True
        else:
            self.get_logger().error('Service call failed')
            return False

    def get_parameter_info(self, param_name):
        """Get information about a specific parameter"""
        return self.parameters_dict.get(param_name, None)

    def list_parameter_names(self):
        """Get list of all parameter names"""
        return list(self.parameters_dict.keys())

    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame

    def update(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.spin()
