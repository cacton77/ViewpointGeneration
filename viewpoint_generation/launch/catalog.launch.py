"""Launch the PLM parts catalog node, which also serves the part picker UI.

The node hosts the backends named by `plm_backends` (default: the
`PLM_BACKENDS` environment variable, else `dx3`). The dx3 backend needs the
DX_* environment variables (supplied by docker-compose from the host `.env`);
the node starts and serves whatever is already cached even when they are
absent, logging a warning instead of failing.

    ros2 launch viewpoint_generation catalog.launch.py
    ros2 launch viewpoint_generation catalog.launch.py picker:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    declared_arguments = [
        DeclareLaunchArgument(
            'plm_backends',
            default_value=EnvironmentVariable('PLM_BACKENDS', default_value='dx3'),
            description="Comma-separated PLM backends to host (e.g. 'dx3')."
        ),
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
        'catalog.backends': LaunchConfiguration('plm_backends'),
        'catalog.sync_interval': LaunchConfiguration('sync_interval'),
        'catalog.sync_on_startup': LaunchConfiguration('sync_on_startup'),
        'catalog.mesh_units': LaunchConfiguration('mesh_units'),
        # The picker runs inside the catalog node's process, sharing its
        # catalog, sync lock and part_selected publisher.
        'catalog.picker_enabled': LaunchConfiguration('picker'),
        'catalog.picker_port': LaunchConfiguration('picker_port'),
    }]

    catalog_node = Node(
        package='viewpoint_generation',
        executable='catalog_node',
        name='catalog',
        parameters=catalog_parameters,
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(declared_arguments + [catalog_node])
