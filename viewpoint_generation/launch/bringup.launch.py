"""Moved: the system bringup is now `inspection_bringup` in the Inspection_System
repository (https://github.com/cacton77/Inspection_System).

This file only includes it, so `ros2 launch viewpoint_generation bringup.launch.py`
keeps working for now; arguments given on the command line pass through. Use

    ros2 launch inspection_bringup bringup.launch.py rig:=<rig> [argument:=value ...]

instead. It will be removed.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="viewpoint_generation's bringup.launch.py has moved to inspection_bringup "
                    "(Inspection_System repository); including it from there. Use "
                    "`ros2 launch inspection_bringup bringup.launch.py` instead."),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("inspection_bringup"), "launch", "bringup.launch.py"])
        ])),
    ])
