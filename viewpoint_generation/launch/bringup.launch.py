from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("sim", default_value="false",),
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        # DeclareLaunchArgument("use_fake_hardware", default_value="true"),
        DeclareLaunchArgument("mock_sensor_commands", default_value="false",
                              description="Enable fake command interfaces for sensors used for simple simulations. "
                              "Used only if 'use_fake_hardware' parameter is true."),
        DeclareLaunchArgument("headless_mode", default_value="false",
                              description="Run in headless mode (without GUI)."),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.102",
                              description="IP address of the robot."),
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
        DeclareLaunchArgument("config_file", default_value="default.yaml",
                              description="Configuration file for viewpoint generation."),
        DeclareLaunchArgument("generation", default_value="true"),
        DeclareLaunchArgument("teleop_config_file",
                              default_value="xbox_controller.yaml", description="Controller configuration file for teleoperation."),
        DeclareLaunchArgument("admittance_config_file", default_value="admittance_control.yaml",
                              description="Configuration file for admittance control."),
    ]

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("inspection_cell_moveit_config"),
                "launch",
                "inspection_cell_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "use_fake_hardware": LaunchConfiguration("sim"),
            "ur_type": LaunchConfiguration("ur_type"),
            "mock_sensor_commands": LaunchConfiguration("mock_sensor_commands"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "safety_limits": LaunchConfiguration("safety_limits"),
            "safety_pos_margin": LaunchConfiguration("safety_pos_margin"),
            "safety_k_position": LaunchConfiguration("safety_k_position"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "launch_moveit": LaunchConfiguration("launch_moveit"),
            "use_tool_communication": LaunchConfiguration("use_tool_communication"),
        }.items()
    )

    # hardware_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare("inspection_cell_moveit_config"),
    #             "launch",
    #             "inspection_cell_hw.launch.py"
    #         ])
    #     ]),
    #     condition=UnlessCondition(LaunchConfiguration("sim")),
    #     launch_arguments={
    #         "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
    #         "mock_sensor_commands": LaunchConfiguration("mock_sensor_commands"),
    #         "headless_mode": LaunchConfiguration("headless_mode"),
    #         "robot_ip": LaunchConfiguration("robot_ip"),
    #         "safety_limits": LaunchConfiguration("safety_limits"),
    #         "safety_pos_margin": LaunchConfiguration("safety_pos_margin"),
    #         "safety_k_position": LaunchConfiguration("safety_k_position")
    #     }.items()
    # )

    viewpoint_generation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("viewpoint_generation"),
                "launch",
                "viewpoint_generation.launch.py"
            ])
        ]),
        launch_arguments={
            "config_file": LaunchConfiguration("config_file")
        }.items()
    )

    viewpoint_traversal_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("viewpoint_generation"),
                "launch",
                "viewpoint_traversal.launch.py"
            ])
        ]),
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

    return LaunchDescription(declared_arguments + [
        simulation_launch,
        # hardware_launch,
        viewpoint_generation_launch,
        viewpoint_traversal_launch,
        admittance_control_launch
    ])
