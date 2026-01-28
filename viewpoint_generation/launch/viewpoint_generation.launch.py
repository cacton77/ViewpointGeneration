import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():

    declared_arguments = [
        DeclareLaunchArgument(
            'object',
            default_value='default.yaml',
            description='Name of the config file'
        ),
        DeclareLaunchArgument(
            'data_path',
            default_value='/data/ViewpointGenerationData',
            description='Path to the data directory.'
        ),
        DeclareLaunchArgument(
            'headless_mode',
            default_value='false',
            description='Run in headless mode (without GUI).'
        ),
    ]

    # Use PathJoinSubstitution to build the path at launch time
    config = PathJoinSubstitution([
        LaunchConfiguration("data_path"),
        LaunchConfiguration("object")
    ])

    viewpoint_generation_node = Node(
        package="viewpoint_generation",
        name="viewpoint_generation",
        executable="viewpoint_generation_node",
        parameters=[config],
        output="screen",
        emulate_tty=True
    )

    gui_node = Node(
        package='viewpoint_generation',
        executable='gui_node',
        name='gui',
        parameters=[config],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless_mode')),
    )

    rqt_node = Node(
        package='rqt_gui',
        executable='rqt_gui',
        name='rqt_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless_mode')),
    )

    register_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=viewpoint_generation_node,
            on_start=[gui_node]
        )
    )

    return LaunchDescription(declared_arguments + [
        viewpoint_generation_node,
        register_event_handler
    ])
