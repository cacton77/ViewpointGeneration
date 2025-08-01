import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='default.yaml',
        description='Name of the config file'
    )

    # Use PathJoinSubstitution to build the path at launch time
    config = PathJoinSubstitution([
        FindPackageShare('viewpoint_generation'),
        'config',
        LaunchConfiguration('config_file')
    ])

    viewpoint_generation_node = Node(
        package="viewpoint_generation",
        name="viewpoint_generation",
        executable="viewpoint_generation_node",
        # parameters=[config],
        parameters=['/workspaces/isaac_ros-dev/install/viewpoint_generation/share/viewpoint_generation/config/default.yaml'],
        output="screen",
        emulate_tty=True
    )

    rqt_configure = Node(
        package='rqt_reconfigure',
        executable='rqt_reconfigure',
        output='screen'
    )

    gui_client_node = Node(
        package='viewpoint_generation',
        executable='gui_client_node',
        name='gui_client',
        output='screen',)

    ld.add_action(config_file_arg)  # Don't forget to add the argument!
    ld.add_action(gui_client_node)
    ld.add_action(rqt_configure)
    ld.add_action(viewpoint_generation_node)

    return ld
