import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('viewpoint_generation'),
        'config',
        'default.yaml'
    )
    # config = os.path.join(

    viewpoint_generation_node = Node(
        package="viewpoint_generation",
        name="viewpoint_generation_node",
        executable="viewpoint_generation_node",
        parameters=[config],
    )

    ld.add_action(viewpoint_generation_node)
    return ld
