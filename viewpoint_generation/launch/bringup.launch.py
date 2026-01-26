from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("cell", default_value="alpha"),
        DeclareLaunchArgument("sim", default_value="false",),
        DeclareLaunchArgument("object", default_value="default.yaml",
                              description="Configuration file for viewpoint generation."),
        DeclareLaunchArgument("data_path", default_value="/data/ViewpointGenerationData",
                              description="Path to the data directory."),
        DeclareLaunchArgument("mock_sensor_commands", default_value="false",
                              description="Enable fake command interfaces for sensors used for simple simulations. "
                              "Used only if 'use_fake_hardware' parameter is true."),
        DeclareLaunchArgument("headless_mode", default_value="false",
                              description="Run in headless mode (without GUI)."),
        DeclareLaunchArgument("safety_limits", default_value="true",
                              description="Enable safety limits controller."),
        DeclareLaunchArgument("safety_pos_margin", default_value="0.15",
                              description="Safety margin for position limits."),
        DeclareLaunchArgument("safety_k_position", default_value="20",
                              description="k-position factor in safety controller."),
        DeclareLaunchArgument("launch_rviz", default_value="true",
                              description="Launch RViz for visualization."),
        DeclareLaunchArgument("launch_moveit", default_value="true",
                              description="Launch MoveIt for motion planning."),
        DeclareLaunchArgument("use_tool_communication", default_value="false",
                              description="Use tool communication for the robot."),

        DeclareLaunchArgument("generation", default_value="true"),
        DeclareLaunchArgument("teleop_config_file",
                              default_value="xbox_controller.yaml", description="Controller configuration file for teleoperation."),
        DeclareLaunchArgument("admittance_config_file", default_value="admittance_control.yaml",
                              description="Configuration file for admittance control."),
    ]

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
        }.items()
    )

    # Depth camera launch moved to inspection_eoat
    # d405_camera_node = Node(
    #     package="realsense2_camera",
    #     executable="realsense2_camera_node",
    #     name="d405_camera",
    #     output="screen",
    #     condition=UnlessCondition(LaunchConfiguration("headless_mode")),
    #     arguments=['--ros-args', '--log-level', 'error']
    # )

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
            "mock_sensor_commands": LaunchConfiguration("mock_sensor_commands"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "safety_limits": LaunchConfiguration("safety_limits"),
            "safety_pos_margin": LaunchConfiguration("safety_pos_margin"),
            "safety_k_position": LaunchConfiguration("safety_k_position"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_moveit": LaunchConfiguration("launch_moveit"),
            "use_tool_communication": LaunchConfiguration("use_tool_communication"),
        }.items()
    )

    task_planning_node = Node(
        package="viewpoint_generation",
        executable="inspection_task_planning_node",
        name="inspection_task_planning",
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
            "mock_sensor_commands": LaunchConfiguration("mock_sensor_commands"),
            "headless_mode": LaunchConfiguration("headless_mode"),
        }.items(),
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
            "teleop_config_file": LaunchConfiguration("teleop_config_file"),
            "admittance_config_file": LaunchConfiguration("admittance_config_file")
        }.items()
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
            "headless_mode": LaunchConfiguration("headless_mode"),
        }.items()
    )

    register_event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=task_planning_node,
            on_start=[viewpoint_generation_launch]
        )
    )

    return LaunchDescription(declared_arguments + [
        macro_camera_launch,
        # d405_camera_node,
        task_planning_node,
        control_moveit_launch,
        register_event_handler,
        viewpoint_traversal_launch,
        admittance_control_launch,
    ])
