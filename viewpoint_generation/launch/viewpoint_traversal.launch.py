import time
from launch import LaunchDescription
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from ur_moveit_config.launch_common import load_yaml
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            "inspection_cell",
            package_name="inspection_cell_moveit_config")
        .robot_description_semantic(file_path="config/inspection_cell.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .moveit_cpp(file_path="config/motion_planning.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(default_planning_pipeline="chomp", pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"])
        .trajectory_execution(moveit_manage_controllers=True)
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
        )
        .to_moveit_configs()
    )

    # Your MoveItPy Viewpoint Traversal Node
    viewpoint_traversal_node = Node(
        name="viewpoint_traversal_node",
        package="viewpoint_generation",
        executable="viewpoint_traversal_node",
        output="both",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([viewpoint_traversal_node])
