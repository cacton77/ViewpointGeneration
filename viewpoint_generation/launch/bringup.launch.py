from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("cell", default_value="false"),
        DeclareLaunchArgument("sim", default_value="false"),
        DeclareLaunchArgument("object", default_value="new.yaml",
                              description="Configuration file for viewpoint generation."),
        DeclareLaunchArgument("data_path", default_value="/data/ViewpointGenerationData",
                              description="Path to the data directory."),
        DeclareLaunchArgument("controller", default_value="xbox_controller.yaml",
                              description="Controller configuration file for teleoperation."),
        DeclareLaunchArgument("admittance_config_file", default_value="admittance_control.yaml",
                              description="Configuration file for admittance control."),
        DeclareLaunchArgument("rviz_config",
                              default_value="/config/rviz/inspection_cell.rviz",
                              description="Absolute path to an RViz config file for the whole "
                              "inspection cell. Defaults to the file mounted into the container "
                              "at /config/rviz/inspection_cell.rviz."),
    ]

    cell_enabled = PythonExpression(
        ["'", LaunchConfiguration("cell"), "' != 'false'"])

    macro_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("http_image_publisher"),
                "launch",
                "http_image_publisher.launch.py"
            ])
        ]),
        launch_arguments={
            "stream_url": "http://192.168.0.92:5000/video_feed",
            "base_topic": "camera",
            "frame_id": "eoat_camera_link",
            "publish_rate": "30.0",
            "connection_timeout": "5.0"
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    control_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("inspection_cell_moveit_config"),
                "launch",
                "inspection_cell.launch.py"
            ])
        ]),
        launch_arguments={
            "cell": LaunchConfiguration("cell"),
            "use_fake_hardware": LaunchConfiguration("sim"),
            "rviz_config": LaunchConfiguration("rviz_config"),
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    task_planning_node = Node(
        package="viewpoint_generation",
        executable="task_planning_node",
        name="task_planning",
        output="screen",
        parameters=[
            ParameterFile(
                PathJoinSubstitution([
                    LaunchConfiguration("data_path"),
                    LaunchConfiguration("object")
                ]),
                allow_substs=True
            )
        ],
    )

    viewpoint_traversal_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("viewpoint_generation"),
                "launch",
                "viewpoint_traversal.launch.py"
            ])
        ]),
        launch_arguments={
            "cell": LaunchConfiguration("cell"),
            "use_fake_hardware": LaunchConfiguration("sim"),
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    admittance_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("inspection_control"),
                "launch",
                "admittance_control.launch.py"
            ])
        ]),
        launch_arguments={
            "controller_config_file": LaunchConfiguration("controller"),
            "admittance_config_file": LaunchConfiguration("admittance_config_file")
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    # Foxglove bridge: serves all topics over a WebSocket (ws://<host>:8765)
    # for Foxglove Studio. Always-on (not gated on cell) so visualization is
    # available regardless of cell hardware. Host networking makes 8765
    # directly reachable on the host.
    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        # output="screen",
        parameters=[{
            "port": 8765,
            "address": "0.0.0.0",
        }],
    )

    viewpoint_generation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("viewpoint_generation"),
                "launch",
                "viewpoint_generation.launch.py"
            ])
        ]),
        launch_arguments={
            "data_path": LaunchConfiguration("data_path"),
            "object": LaunchConfiguration("object"),
        }.items()
    )

    return LaunchDescription(declared_arguments + [
        viewpoint_generation_launch,
        # foxglove_bridge_node,
        # task_planning_node,
        control_moveit_launch,
        viewpoint_traversal_launch,
        # admittance_control_launch,
    ])
