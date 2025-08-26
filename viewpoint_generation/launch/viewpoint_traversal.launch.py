from moveit_configs_utils.launches import generate_move_group_launch
import time
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, Command
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_moveit_rviz_launch
from ur_moveit_config.launch_common import load_yaml
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction


def launch_setup(context):
    # Get the actual cell value at launch time
    cell = LaunchConfiguration("cell").perform(context)

    xacro_mappings = {
        'cell': LaunchConfiguration("cell").perform(context),
        'use_fake_hardware': LaunchConfiguration("use_fake_hardware").perform(context),
        'mock_sensor_commands': LaunchConfiguration("mock_sensor_commands").perform(context),
        'headless_mode': LaunchConfiguration("headless_mode").perform(context),
    }

    urdf_file_path = PathJoinSubstitution([
        FindPackageShare("inspection_cell_description"),
        "urdf",
        "inspection_cell.urdf.xacro"
    ]).perform(context)

    joint_limits_file = PathJoinSubstitution([
        FindPackageShare("inspection_cell_description"),
        "config",
        cell,
        "joint_limits.yaml"
    ]).perform(context)

    moveit_config = (
        MoveItConfigsBuilder(
            "inspection_cell", package_name="inspection_cell_moveit_config"
        )
        .robot_description(file_path=urdf_file_path,
                           mappings=xacro_mappings)
        .robot_description_semantic(file_path="config/inspection_cell.srdf")
        .moveit_cpp(file_path="config/motion_planning.yaml")
        .joint_limits(file_path=joint_limits_file)
        .planning_scene_monitor(
            publish_planning_scene=False,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl"
        )
        .to_moveit_configs()
    )

    viewpoint_traversal_node = Node(
        name="viewpoint_traversal",
        package="viewpoint_generation",
        executable="viewpoint_traversal_node",
        output="both",
        parameters=[moveit_config.to_dict()],
    )

    return [
        viewpoint_traversal_node
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("cell", default_value="beta", choices=[
                                  "alpha", "beta"], description="Inspection cell type"),
            DeclareLaunchArgument("ur_type", default_value="ur5e",
                                  description="Type of UR robot"),
            DeclareLaunchArgument("use_fake_hardware",
                                  default_value="true", description="Sim mode"),
            DeclareLaunchArgument(
                "mock_sensor_commands", default_value="false", description="Mock sensor commands"),
            DeclareLaunchArgument(
                "headless_mode", default_value="false", description="Disable GUI"),
            OpaqueFunction(function=launch_setup)
        ]
    )
