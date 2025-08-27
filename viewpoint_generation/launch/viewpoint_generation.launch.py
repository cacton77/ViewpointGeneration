import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    object_arg = DeclareLaunchArgument(
        'object',
        default_value='default.yaml',
        description='Name of the config file'
    )

    # Use PathJoinSubstitution to build the path at launch time
    config = PathJoinSubstitution([
        FindPackageShare('viewpoint_generation'),
        'config',
        LaunchConfiguration('object')
    ])

    viewpoint_generation_node = Node(
        package="viewpoint_generation",
        name="viewpoint_generation",
        executable="viewpoint_generation_node",
        parameters=[config],
        output="screen",
        emulate_tty=True
    )

    gui_client_node = Node(
        package='viewpoint_generation',
        executable='gui_client_node',
        name='gui',
        parameters=[config],
        output='screen',)

    ld.add_action(object_arg)  # Don't forget to add the argument!
    ld.add_action(gui_client_node)
    ld.add_action(viewpoint_generation_node)

    return ld
