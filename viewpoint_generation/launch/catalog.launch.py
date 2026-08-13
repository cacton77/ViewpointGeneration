"""Launch the 3DEXPERIENCE parts catalog node and the part picker UI.

The catalog node needs the DX_* environment variables (supplied by
docker-compose from the host `.env`); it starts and serves whatever is already
cached even when they are absent, logging a warning instead of failing.

    ros2 launch viewpoint_generation catalog.launch.py
    ros2 launch viewpoint_generation catalog.launch.py picker:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    declared_arguments = [
        DeclareLaunchArgument(
            'sync_interval',
            default_value='300',
            description='Seconds between background incremental catalog syncs '
                        '(0 disables the timer).'
        ),
        DeclareLaunchArgument(
            'sync_on_startup',
            default_value='true',
            description='Run a full catalog sync when the node starts.'
        ),
        DeclareLaunchArgument(
            'bookmark_scope',
            default_value='',
            description='3DX search string scoping the catalog. Empty keeps '
                        'the DX_BOOKMARK_SCOPE environment value.'
        ),
        DeclareLaunchArgument(
            'collab_space',
            default_value='',
            description='Restrict the catalog to one collaborative space. '
                        'Empty keeps the DX_COLLAB_SPACE environment value.'
        ),
        DeclareLaunchArgument(
            'mesh_units',
            default_value='mm',
            description='Units used to load STEP files selected from the catalog.'
        ),
        DeclareLaunchArgument(
            'picker',
            default_value='true',
            description='Also start the browser-based part picker UI.'
        ),
        DeclareLaunchArgument(
            'picker_port',
            default_value='5050',
            description='Port the part picker UI listens on.'
        ),
    ]

    # Empty launch arguments must not clobber the environment-derived defaults,
    # so scope/space are passed through only when the operator sets them.
    catalog_parameters = [{
        'catalog.sync_interval': LaunchConfiguration('sync_interval'),
        'catalog.sync_on_startup': LaunchConfiguration('sync_on_startup'),
        'catalog.mesh_units': LaunchConfiguration('mesh_units'),
    }]

    catalog_node = Node(
        package='viewpoint_generation',
        executable='catalog_node',
        name='catalog',
        parameters=catalog_parameters,
        output='screen',
        emulate_tty=True,
    )

    # The picker is configured through its own CLI (it is equally runnable
    # outside ROS), so the port is passed as a process argument rather than a
    # ROS parameter -- as a parameter it would be silently ignored. main()
    # uses parse_known_args, so the --ros-args launch appends are harmless.
    picker_node = Node(
        package='viewpoint_generation',
        executable='picker_node',
        name='picker',
        arguments=['--port', LaunchConfiguration('picker_port'),
                   '--units', LaunchConfiguration('mesh_units')],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('picker')),
    )

    return LaunchDescription(declared_arguments + [catalog_node, picker_node])
