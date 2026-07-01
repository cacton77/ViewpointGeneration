import rclpy
import rclpy.logging
import json
import re
import datetime
import numpy as np
from rclpy.node import Node
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from rcl_interfaces.msg import SetParametersResult
from viewpoint_generation_interfaces.srv import MoveToPoseStamped, OptimizeViewpointTraversal
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from .tsp_solver import TSPSolver


class ViewpointTraversalNode(Node):

    def __init__(self):
        node_name = 'viewpoint_traversal'
        super().__init__(node_name)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('planning_group', 'disc_to_ur5e'),
                ('planner', 'ompl'),
                ('multiplanning', False),
                ('workspace.min_x', -1.0),
                ('workspace.max_x', 1.0),
                ('workspace.min_y', -1.0),
                ('workspace.max_y', 1.0),
                ('workspace.min_z', -1.0),
                ('workspace.max_z', 1.0),
                ('clear_paths', False),
                ('tsp_algorithm', 'greedy'),
            ]
        )

        self.planning_group = self.get_parameter(
            'planning_group').get_parameter_value().string_value
        self.planner = self.get_parameter(
            'planner').get_parameter_value().string_value
        self.multiplanning = self.get_parameter(
            'multiplanning').get_parameter_value().bool_value
        self.clear_paths = self.get_parameter(
            'clear_paths').get_parameter_value().bool_value
        self.tsp_algorithm = self.get_parameter(
            'tsp_algorithm').get_parameter_value().string_value

        self.workspace = {
            'min_x': self.get_parameter('workspace.min_x').get_parameter_value().double_value,
            'max_x': self.get_parameter('workspace.max_x').get_parameter_value().double_value,
            'min_y': self.get_parameter('workspace.min_y').get_parameter_value().double_value,
            'max_y': self.get_parameter('workspace.max_y').get_parameter_value().double_value,
            'min_z': self.get_parameter('workspace.min_z').get_parameter_value().double_value,
            'max_z': self.get_parameter('workspace.max_z').get_parameter_value().double_value
        }

        self.solver = TSPSolver(logger=self.get_logger())

        self.robot = MoveItPy(node_name='moveit_py')
        self.planning_scene_monitor = self.robot.get_planning_scene_monitor()

        try:
            self.get_logger().info("Initializing MoveItPy")
            self.planning_component = self.robot.get_planning_component(
                self.planning_group)
            self.get_logger().info("Planning component 'disc_to_ur5e' initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to get planning component: {e}")
            self.planning_component = None
            return

        services_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_service(
            MoveToPoseStamped,
            'viewpoint_traversal/move_to_pose_stamped',
            self.move_to_pose_stamped_callback,
            callback_group=services_cb_group
        )
        self.get_logger().info("Service 'move_to_pose_stamped' created successfully")

        self.create_service(
            OptimizeViewpointTraversal,
            f'{node_name}/optimize_traversal',
            self.optimize_traversal,
            callback_group=services_cb_group
        )

        self.add_on_set_parameters_callback(self.parameter_callback)

    def init_workspace(self):
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "planning_volume"
            collision_object.id = "workspace"

            box_pose = Pose()
            box_pose.position.x = (
                self.workspace['max_x'] - self.workspace['min_x']) / 2
            box_pose.position.y = (
                self.workspace['max_y'] - self.workspace['min_y']) / 2
            box_pose.position.z = (
                self.workspace['max_z'] - self.workspace['min_z']) / 2

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = (
                self.workspace['max_x'] - self.workspace['min_x'],
                self.workspace['max_y'] - self.workspace['min_y'],
                self.workspace['max_z'] - self.workspace['min_z']
            )

            collision_object.primitives.append(box)
            collision_object.primitive_poses.append(box_pose)
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
            scene.current_state.update()  # Important to ensure the scene is updated

        self.get_logger().info("Workspace initialized successfully")
        self.get_logger().info(f"Workspace boundaries: {self.workspace}")

    def optimize_traversal(self, request, response):
        if not request.viewpoint_dict_path:
            response.success = False
            response.message = "No viewpoint dictionary path provided"
            return response

        self.get_logger().info(
            f'Optimizing traversal using \'{self.tsp_algorithm}\' for {request.viewpoint_dict_path}')
        with open(request.viewpoint_dict_path, 'r') as f:
            viewpoint_dict = json.load(f)

        viewpoint_dict_optimized = self.tsp(viewpoint_dict)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_viewpoint_dict_path = re.sub(
            r'(_optimized.*?)?\.json$',
            f'_optimized{timestamp}.json',
            request.viewpoint_dict_path
        )

        with open(new_viewpoint_dict_path, 'w') as f:
            json.dump(viewpoint_dict_optimized, f, indent=4)

        # Compute total distance across all regions for display
        primary_total = sum(
            v.get('distance', 0.0)
            for v in self.solver.algorithm_results.get(self.tsp_algorithm, {}).values()
        )

        # MST lower bound (summed across all regions of all meshes)
        mst_total = 0.0
        for mesh_entry in viewpoint_dict.get('meshes', []):
            for region in mesh_entry.get('regions', []):
                clusters = region.get('clusters', [])
                vps = [c['viewpoint']['position']
                       for c in clusters if 'viewpoint' in c]
                if len(vps) >= 2:
                    mst_total += self.solver._mst_cost(self.solver.dist_matrix(vps))

        gap_pct = ((primary_total - mst_total) / mst_total * 100.0
                   if mst_total > 1e-9 else 0.0)

        # Per-region summary table
        region_gaps = self.solver._region_gaps
        max_gap = max((v['gap_pct'] for v in region_gaps.values()), default=0.0)
        self.get_logger().info('─' * 95)
        self.get_logger().info(
            f'{"Region":<10} {"N":>4}  {"Distance":>10}  {"LowerBound":>11}  '
            f'{"Gap%":>7}  {"BoundType":>12}  {"ArmTime":>8}  {"ArmDist":>8}  '
            f'{"Unreach":>7}')
        for rname, info in sorted(region_gaps.items()):
            btype = 'Held-Karp' if info['exact'] else 'MST'
            opt_flag = ' ✓' if info['exact'] and info['gap_pct'] < 0.01 else ''
            n_unreachable = len(info.get('unreachable', []))
            self.get_logger().info(
                f'{rname:<10} {info["n"]:>4}  {info["distance"]:>10.4f}'
                f'  {info["lower_bound"]:>11.4f}  {info["gap_pct"]:>6.2f}%'
                f'  {btype:>12}  {info.get("arm_time_s", 0.0):>7.1f}s'
                f'  {info.get("arm_joint_distance", 0.0):>7.2f}r'
                f'  {n_unreachable:>7}{opt_flag}')
        self.get_logger().info('─' * 95)

        completed_str = ','.join(sorted(self.solver.completed_algorithms))
        self.get_logger().info(
            f'Completed: {completed_str}  |  '
            f'MST total: {mst_total:.3f} m  |  '
            f'{self.tsp_algorithm}: {primary_total:.3f} m  |  '
            f'Overall gap: {gap_pct:.2f}%  |  Max region gap: {max_gap:.2f}%')

        # Whole-inspection arm traversal cost/time (moveitpy-computed), summed
        # across all regions for the selected algorithm's chosen path.
        total_arm_time = sum(v.get('arm_time_s', 0.0) for v in region_gaps.values())
        total_arm_distance = sum(
            v.get('arm_joint_distance', 0.0) for v in region_gaps.values())
        total_unreachable = sum(
            len(v.get('unreachable', [])) for v in region_gaps.values())
        regions_with_unreachable = sorted(
            rname for rname, info in region_gaps.items() if info.get('unreachable'))
        self.get_logger().info(
            f'Arm traversal ({self.tsp_algorithm}): {total_arm_time:.1f}s  |  '
            f'{total_arm_distance:.2f} rad joint distance  |  '
            f'{total_unreachable} viewpoint(s) unreachable'
            + (f' across regions {", ".join(regions_with_unreachable)}'
               if regions_with_unreachable else ''))
        for rname in regions_with_unreachable:
            self.get_logger().warning(
                f'  {rname}: unreachable viewpoints '
                f'{region_gaps[rname]["unreachable"]}')

        # Per-algorithm path metrics (distance, etc.) are stored per region in
        # the results JSON under each algorithm's key, not in node parameters.
        msg = f"Algorithm: {self.tsp_algorithm}  Total: {primary_total:.3f} m"
        self.get_logger().info(msg)

        response.success = True
        response.message = msg
        response.new_viewpoint_dict_path = new_viewpoint_dict_path
        return response

    @staticmethod
    def _ensure_order_dict(region):
        """Return the region's algorithm-keyed order dict, converting a legacy
        identity list into an empty dict on first use. Existing algorithm entries
        are preserved so repeated optimize_traversal calls accumulate paths."""
        if not isinstance(region.get('order'), dict):
            region['order'] = {}
        return region['order']

    def tsp(self, viewpoint_dict):
        if self.clear_paths:
            self.solver.algorithm_results.clear()
            self.solver.completed_algorithms.clear()
            # Reset every region's order back to an identity list.
            for mesh_entry in viewpoint_dict.get('meshes', []):
                for region in mesh_entry.get('regions', []):
                    region['order'] = list(range(len(region.get('clusters', []))))
            viewpoint_dict.pop('selected_traversal_algorithm', None)
            return viewpoint_dict

        if self.tsp_algorithm == 'Select Algorithm':
            self.get_logger().warning('No algorithm selected.')
            return viewpoint_dict

        primary = self.tsp_algorithm
        self.solver.algorithm_results.setdefault(primary, {})

        self.solver._region_gaps = {}

        for mesh_idx, mesh_entry in enumerate(viewpoint_dict.get('meshes', [])):
            for region_idx, region in enumerate(mesh_entry.get('regions', [])):
                clusters = region.get('clusters', [])
                order_dict = self._ensure_order_dict(region)
                n_clusters = len(clusters)
                region_key = f"{mesh_idx}:{region_idx}"

                # Regions too small to optimize (or missing viewpoints) keep an
                # identity order so downstream consumers always find a path.
                if n_clusters < 3 or any('viewpoint' not in c for c in clusters):
                    identity = list(range(n_clusters))
                    if n_clusters >= 2 and all('viewpoint' in c for c in clusters):
                        vps = [c['viewpoint']['position'] for c in clusters]
                        identity_distance = self.solver.dist_calc(
                            self.solver.dist_matrix(vps), identity)
                    else:
                        identity_distance = 0.0
                    order_dict[primary] = {
                        'order': identity,
                        'distance': float(identity_distance),
                    }
                    continue

                viewpoints = [c['viewpoint']['position'] for c in clusters]
                n_vp = len(viewpoints)
                dm = self.solver.dist_matrix(viewpoints)

                # Lower bound for this region
                if n_vp <= 18:
                    if n_vp > 12:
                        self.get_logger().info(
                            f'  Running Held-Karp exact solver for N={n_vp} (~30s)...')
                    lower_bound = self.solver._held_karp(dm)
                    bound_label = f'HK(exact)={lower_bound:.4f}'
                else:
                    lower_bound = self.solver._mst_cost(dm)
                    bound_label = f'MST={lower_bound:.4f}'

                self.get_logger().info(
                    f"Running {primary} on region {region_key} (N={n_vp})...")
                path, distance = self.solver._run_algorithm(primary, viewpoints, dm)
                if path is None:
                    continue

                prev_best = self.solver.algorithm_results[primary].get(region_key, {}).get('distance', float('inf'))
                improved = distance < prev_best - 1e-10
                if improved or region_key not in self.solver.algorithm_results[primary]:
                    self.solver.algorithm_results[primary][region_key] = {
                        'path': path.copy(),
                        'distance': distance
                    }
                else:
                    # Keep the previously stored best; restore distance for logging
                    distance = prev_best
                    path = self.solver.algorithm_results[primary][region_key]['path']

                # Store this algorithm's path and metrics under its own key.
                path_to_save = path[:-1] if (len(path) > 0 and path[-1] == path[0]) else path
                order_dict[primary] = {
                    'order': list(path_to_save),
                    'distance': float(distance),
                }

                jt_info = ''
                arm_time_s = 0.0
                arm_joint_distance = 0.0
                unreachable = []
                if self.planning_component and len(path_to_save) > 1:
                    joint_traj = self._compute_joint_trajectory(
                        list(path_to_save), clusters, region_key)
                    order_dict[primary]['joint_trajectory'] = joint_traj
                    arm_time_s = joint_traj['total_time_s']
                    arm_joint_distance = joint_traj['total_joint_distance']
                    unreachable = joint_traj.get('unreachable', [])
                    jt_info = (f' | arm: {arm_time_s:.1f}s'
                               f' / {arm_joint_distance:.2f} rad')
                    if unreachable:
                        jt_info += f' | {len(unreachable)} unreachable'

                gap_pct = ((distance - lower_bound) / lower_bound * 100.0
                        if lower_bound > 1e-9 else 0.0)
                tag = ' [EXACT OPTIMAL]' if (n_vp <= 18 and gap_pct < 0.01) else (
                    ' [best kept]' if not improved else '')
                self.get_logger().info(
                    f'  {primary}: {distance:.4f} m | {bound_label} | '
                    f'Gap={gap_pct:.2f}%{tag}  (N={n_vp}){jt_info}')

                self.solver._region_gaps[region_key] = {
                    'n': n_vp,
                    'distance': distance,
                    'lower_bound': lower_bound,
                    'gap_pct': gap_pct,
                    'exact': n_vp <= 18,
                    'arm_time_s': arm_time_s,
                    'arm_joint_distance': arm_joint_distance,
                    'unreachable': unreachable,
                }

                self.solver.completed_algorithms.add(primary)

        # The selected algorithm is a parameter on the task_planning node (set
        # via the GUI), which drives both the visualized path and the execution
        # order. Each region keeps its per-algorithm 'order' dict here — each
        # value being {'order': [...], 'distance': ...} — so any optimized
        # algorithm can be chosen.
        return viewpoint_dict

    def clear_paths_callback(self, request, response):
        self.solver.algorithm_results.clear()
        self.solver.completed_algorithms.clear()
        response.success = True
        response.message = "Paths cleared"
        return response

    def move_to_pose_stamped_callback(self, request, response):
        if not self.planning_component:
            self.get_logger().error("Planning component is not initialized")
            response.success = False
            response.message = "Planning component is not initialized"
            return response

        # Create a RobotState object
        robot_state = RobotState(self.robot.get_robot_model())
        # Set the pose from the request
        self.planning_component.set_goal_state(
            pose_stamped_msg=request.pose_goal, pose_link="eoat_camera_link")

        # Log the request
        self.get_logger().info(f"Received request: {request}")

        # Set the robot state to the current state
        robot_state.set_to_default_values()
        # Plan and execute
        multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
            self.robot, ["ompl_rrtc"]
        )

        # self.single_plan_parameters.planner_id = self.planner

        success = self.plan_and_execute()

        # success = self.plan_and_execute(
        #     multi_plan_parameters=multi_pipeline_plan_request_params)
        self.get_logger().info(f"Plan and execute called, success: {success}")

        # Prepare the response
        response.success = success
        response.message = "Motion completed successfully" if success else "Motion failed"

        return response

    # Function for planning and executing a trajectories
    def plan_and_execute(self, single_plan_parameters=None, multi_plan_parameters=None):
        # Check if the planning component is valid
        if not self.planning_component:
            self.get_logger().error("Planning component is not valid")
            return False

        # Create a RobotState object
        robot_state = RobotState(self.robot.get_robot_model())

        # Set the robot state to the current state
        robot_state.set_to_default_values()

        # plan to the specified pose and execute the trajectory
        self.get_logger().info("Planning and executing trajectory")
        if multi_plan_parameters is not None:
            plan_result = self.planning_component.plan(
                multi_plan_parameters=multi_plan_parameters
            )
        elif single_plan_parameters is not None:
            plan_result = self.planning_component.plan(
                single_plan_parameters=single_plan_parameters
            )
        else:
            # plan_result = self.planning_component.plan(
            # single_plan_parameters=self.single_plan_parameters)
            plan_result = self.planning_component.plan()

        print("------------------------------------")
        print(plan_result)
        print("------------------------------------")

        # Execute the Planned Trajectory
        if plan_result:
            self.get_logger().info("Executing plan")
            robot_trajectory = plan_result.trajectory
            # Check if the controller name is correct
            self.robot.execute(plan_result.trajectory, controllers=[])
            return True
        else:
            self.get_logger().error("No trajectory found to execute")
            return False

    def _plan_segment(self, viewpoint, start_joint_positions=None):
        if start_joint_positions is not None:
            start_state = RobotState(self.robot.get_robot_model())
            start_state.set_joint_group_positions(
                self.planning_group, start_joint_positions)
            self.planning_component.set_start_state(robot_state=start_state)
        else:
            self.planning_component.set_start_state_to_current_state()

        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "object_frame"
        pos = viewpoint['position']
        orient = viewpoint['orientation']  # [x, y, z, w]
        pose_goal.pose.position.x = pos[0]
        pose_goal.pose.position.y = pos[1]
        pose_goal.pose.position.z = pos[2]
        pose_goal.pose.orientation.x = orient[0]
        pose_goal.pose.orientation.y = orient[1]
        pose_goal.pose.orientation.z = orient[2]
        pose_goal.pose.orientation.w = orient[3]
        self.planning_component.set_goal_state(
            pose_stamped_msg=pose_goal, pose_link="eoat_camera_link")

        plan_result = self.planning_component.plan()
        if not plan_result:
            return None, 0.0

        jt = plan_result.trajectory.get_robot_trajectory_msg().joint_trajectory
        if not jt.points:
            return None, 0.0

        waypoints = [list(pt.positions) for pt in jt.points]
        last_pt = jt.points[-1]
        duration = (last_pt.time_from_start.sec
                    + last_pt.time_from_start.nanosec * 1e-9)
        return waypoints, duration

    def _fk_waypoints(self, joint_waypoints):
        robot_model = self.robot.get_robot_model()
        robot_state = RobotState(robot_model)
        positions = []
        for jpos in joint_waypoints:
            robot_state.set_joint_group_positions(self.planning_group, jpos)
            robot_state.update()
            T = robot_state.get_frame_transform("eoat_camera_link")
            positions.append([float(T[0, 3]), float(T[1, 3]), float(T[2, 3])])
        return positions

    def _compute_joint_trajectory(self, path, clusters, region_key):
        result = {
            'total_time_s': 0.0,
            'total_joint_distance': 0.0,
            'cartesian_waypoints': [],
            'unreachable': [],
        }
        prev_joint_state = None

        for step in range(len(path) - 1):
            from_idx, to_idx = path[step], path[step + 1]
            to_vp = clusters[to_idx].get('viewpoint') if to_idx < len(clusters) else None

            if not to_vp:
                result['unreachable'].append(to_idx)
                self.get_logger().warning(
                    f'  region {region_key}: viewpoint {to_idx} has no '
                    f'viewpoint data, skipping')
                continue

            waypoints, duration = self._plan_segment(to_vp, prev_joint_state)
            if waypoints is None:
                result['unreachable'].append(to_idx)
                self.get_logger().warning(
                    f'  region {region_key}: viewpoint {to_idx} unreachable '
                    f'from arm pose after viewpoint {from_idx}')
                continue

            joint_dist = float(sum(
                np.linalg.norm(np.array(waypoints[i + 1]) - np.array(waypoints[i]))
                for i in range(len(waypoints) - 1)
            ))
            cartesian = self._fk_waypoints(waypoints)
            result['cartesian_waypoints'].extend(cartesian)
            result['total_time_s'] += duration
            result['total_joint_distance'] += joint_dist
            prev_joint_state = waypoints[-1]

        self.planning_component.set_start_state_to_current_state()
        result['total_time_s'] = round(result['total_time_s'], 3)
        result['total_joint_distance'] = round(result['total_joint_distance'], 4)
        return result

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'workspace.min_x':
                self.workspace['min_x'] = param.value
            elif param.name == 'workspace.min_y':
                self.workspace['min_y'] = param.value
            elif param.name == 'workspace.min_z':
                self.workspace['min_z'] = param.value
            elif param.name == 'workspace.max_x':
                self.workspace['max_x'] = param.value
            elif param.name == 'workspace.max_y':
                self.workspace['max_y'] = param.value
            elif param.name == 'workspace.max_z':
                self.workspace['max_z'] = param.value
            elif param.name == 'tsp_algorithm':
                self.tsp_algorithm = param.value
            elif param.name == 'clear_paths':
                self.clear_paths = param.value

        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    # MoveItPy and its internal moveit_cpp/OMPL loggers are extremely
    # chatty at INFO/WARN level (adapter/planner-stage announcements and
    # benign "planner_id not found"/"planning volume not specified"
    # warnings repeated for every planned segment). Quiet everything
    # down to ERROR by default, then re-enable INFO for this node's own
    # logger so its progress/summary output is unaffected.
    rclpy.logging.set_logger_level(
        '', rclpy.logging.LoggingSeverity.ERROR)
    rclpy.logging.set_logger_level(
        'viewpoint_traversal', rclpy.logging.LoggingSeverity.INFO)
    traversal_node = ViewpointTraversalNode()
    rclpy.spin(traversal_node)


if __name__ == '__main__':
    main()
