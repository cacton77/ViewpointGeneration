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
from viewpoint_generation_interfaces.srv import MoveToPoseStamped, OptimizeViewpointTraversal, FindNearestViewpoint
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from .tsp_solver import TSPSolver
from .vrp_solver import VRPSolver, VRPSolution


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
                # When multiplanning is True, race these named plan-request configs
                # (from motion_planning.yaml) in parallel and keep the shortest result.
                ('multiplanning_pipelines', ['ompl_rrtc', 'chomp_planner', 'ompl_rrt_star']),
                ('workspace.min_x', -1.0),
                ('workspace.max_x', 1.0),
                ('workspace.min_y', -1.0),
                ('workspace.max_y', 1.0),
                ('workspace.min_z', -1.0),
                ('workspace.max_z', 1.0),
                ('clear_paths', False),
                ('tsp_algorithm', 'greedy'),
                ('vrp_algorithm', ''),
                # Objective: 'time' = MoveIt TOTG execution-time surrogate (seconds);
                # 'joint' = weighted joint-space distance (radians).
                ('vrp_cost_mode', 'time'),
                ('vrp_max_velocity', 0.5),      # rad/s, per-joint limit for the time model
                ('vrp_max_acceleration', 1.0),  # rad/s², per-joint limit for the time model
                ('vrp_joint_weights', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # only used in 'joint' mode
                ('vrp_validate_topk', 1),  # >1: plan top-K candidate tours, keep the fastest real time
                ('vrp_aco_n_ants', 20),    # configurable
                ('vrp_aco_n_iter', 100),   # configurable
                ('vrp_aco_alpha', 1.0),    # pheromone weight (configurable)
                ('vrp_aco_beta', 2.0),     # heuristic weight (configurable)
                ('vrp_aco_rho', 0.1),      # evaporation rate (configurable)
                ('vrp_aco_n_jobs', 1),     # >1: run ACO ants across processes
                ('vrp_clustered_k', 6),    # candidate entry/exit ports per region (vrp_clustered)
                ('vrp_n_turntable_samples', 0),  # >0 enables the multi-config turntable sweep
                ('vrp_max_configs_per_vp', 8),   # cap on IK configs kept per viewpoint
                ('vrp_config_dedup_tol', 0.1),   # rad, configs within this L∞ are merged
                ('vrp_chain_max_passes', 1),     # >1 iterates IK-chain <-> re-solve (Spec L); 1 = current behavior
            ]
        )

        self.planning_group = self.get_parameter(
            'planning_group').get_parameter_value().string_value
        self.planner = self.get_parameter(
            'planner').get_parameter_value().string_value
        self.multiplanning = self.get_parameter(
            'multiplanning').get_parameter_value().bool_value
        self.multiplanning_pipelines = list(self.get_parameter(
            'multiplanning_pipelines').get_parameter_value().string_array_value)
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

        self.vrp_algorithm = self.get_parameter('vrp_algorithm').get_parameter_value().string_value
        self.vrp_cost_mode = self.get_parameter('vrp_cost_mode').get_parameter_value().string_value
        self.vrp_max_velocity = self.get_parameter('vrp_max_velocity').get_parameter_value().double_value
        self.vrp_max_acceleration = self.get_parameter('vrp_max_acceleration').get_parameter_value().double_value
        self.vrp_validate_topk = self.get_parameter('vrp_validate_topk').get_parameter_value().integer_value
        self.vrp_aco_n_ants = self.get_parameter('vrp_aco_n_ants').get_parameter_value().integer_value
        self.vrp_aco_n_iter = self.get_parameter('vrp_aco_n_iter').get_parameter_value().integer_value
        self.vrp_aco_alpha = self.get_parameter('vrp_aco_alpha').get_parameter_value().double_value
        self.vrp_aco_beta = self.get_parameter('vrp_aco_beta').get_parameter_value().double_value
        self.vrp_aco_rho = self.get_parameter('vrp_aco_rho').get_parameter_value().double_value
        self.vrp_aco_n_jobs = self.get_parameter('vrp_aco_n_jobs').get_parameter_value().integer_value
        self.vrp_clustered_k = self.get_parameter('vrp_clustered_k').get_parameter_value().integer_value
        self.vrp_n_turntable_samples = self.get_parameter(
            'vrp_n_turntable_samples').get_parameter_value().integer_value
        self.vrp_max_configs_per_vp = self.get_parameter(
            'vrp_max_configs_per_vp').get_parameter_value().integer_value
        self.vrp_config_dedup_tol = self.get_parameter(
            'vrp_config_dedup_tol').get_parameter_value().double_value
        self.vrp_chain_max_passes = self.get_parameter(
            'vrp_chain_max_passes').get_parameter_value().integer_value
        vrp_weights = list(self.get_parameter('vrp_joint_weights').get_parameter_value().double_array_value)
        self.vrp_solver = VRPSolver(
            joint_weights=vrp_weights or None,
            cost_mode=self.vrp_cost_mode,
            max_velocity=self.vrp_max_velocity,
            max_acceleration=self.vrp_max_acceleration,
            n_turntable_samples=self.vrp_n_turntable_samples,
            max_configs_per_vp=self.vrp_max_configs_per_vp,
            config_dedup_tol=self.vrp_config_dedup_tol,
            logger=self.get_logger())
        self._ik_result = None
        self._region_offsets = None

        self.robot = MoveItPy(node_name='moveit_py')
        self.planning_scene_monitor = self.robot.get_planning_scene_monitor()
        self._multi_plan_params = None

        try:
            self.get_logger().info("Initializing MoveItPy")
            self.planning_component = self.robot.get_planning_component(
                self.planning_group)
            self.get_logger().info("Planning component 'disc_to_ur5e' initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to get planning component: {e}")
            self.planning_component = None
            return

        self._multi_plan_params = self._build_multi_plan_params()

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
        self.create_service(
            FindNearestViewpoint,
            f'{node_name}/find_nearest_viewpoint',
            self.find_nearest_viewpoint_callback,
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

        vrp_ran = bool(self.vrp_algorithm and self.vrp_algorithm not in ('', 'Select Algorithm'))
        primary_algorithm = self.vrp_algorithm if vrp_ran else self.tsp_algorithm
        self.get_logger().info(
            f'Optimizing traversal using \'{primary_algorithm}\' for {request.viewpoint_dict_path}')
        with open(request.viewpoint_dict_path, 'r') as f:
            viewpoint_dict = json.load(f)

        viewpoint_dict_optimized = self.tsp(viewpoint_dict)
        self._last_vrp_cost = 0.0
        if vrp_ran:
            viewpoint_dict_optimized = self.vrp(viewpoint_dict_optimized)

        # Auto-reset after a clear so the next optimize call runs the algorithm.
        if self.clear_paths:
            self.clear_paths = False
            self.set_parameters([rclpy.parameter.Parameter('clear_paths', False)])

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_viewpoint_dict_path = re.sub(
            r'(_optimized.*?)?\.json$',
            f'_optimized{timestamp}.json',
            request.viewpoint_dict_path
        )

        with open(new_viewpoint_dict_path, 'w') as f:
            json.dump(viewpoint_dict_optimized, f, indent=4)

        if vrp_ran:
            # VRP already logged its per-region breakdown inside vrp(); just show totals.
            primary_total = self._last_vrp_cost
            units = self.vrp_solver.cost_units
            obj = 'execution-time' if units == 's' else 'joint-space'
            completed_str = ','.join(sorted(self.solver.completed_algorithms)) or '(none)'
            self.get_logger().info(
                f'Completed TSP: {completed_str}  |  '
                f'VRP {primary_algorithm}: {primary_total:.4f} {units} ({obj})')
            total_arm_time = 0.0
            total_arm_distance = 0.0
            total_unreachable = 0
            regions_with_unreachable = []
            for mesh_entry in viewpoint_dict.get('meshes', []):
                for region in mesh_entry.get('regions', []):
                    algo_data = region.get('order', {}).get(primary_algorithm, {})
                    jt = algo_data.get('joint_trajectory', {})
                    total_arm_time += jt.get('total_time_s', 0.0)
                    total_arm_distance += jt.get('total_joint_distance', 0.0)
                    ur = jt.get('unreachable', [])
                    if ur:
                        total_unreachable += len(ur)
                        rname = region.get('name', '?')
                        regions_with_unreachable.append(rname)
                        self.get_logger().warning(f'  {rname}: unreachable viewpoints {ur}')
            self.get_logger().info(
                f'Arm traversal ({primary_algorithm}): {total_arm_time:.1f}s  |  '
                f'{total_arm_distance:.2f} rad joint distance  |  '
                f'{total_unreachable} viewpoint(s) unreachable'
                + (f' across regions {", ".join(regions_with_unreachable)}'
                   if regions_with_unreachable else ''))
            msg = f"Algorithm: {primary_algorithm}  Total: {primary_total:.4f} {units}"
        else:
            # TSP summary: Cartesian distance + per-region gap table
            primary_total = sum(
                v.get('distance', 0.0)
                for v in self.solver.algorithm_results.get(primary_algorithm, {}).values()
            )
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
                f'{primary_algorithm}: {primary_total:.3f} m  |  '
                f'Overall gap: {gap_pct:.2f}%  |  Max region gap: {max_gap:.2f}%')
            total_arm_time = sum(v.get('arm_time_s', 0.0) for v in region_gaps.values())
            total_arm_distance = sum(
                v.get('arm_joint_distance', 0.0) for v in region_gaps.values())
            total_unreachable = sum(
                len(v.get('unreachable', [])) for v in region_gaps.values())
            regions_with_unreachable = sorted(
                rname for rname, info in region_gaps.items() if info.get('unreachable'))
            self.get_logger().info(
                f'Arm traversal ({primary_algorithm}): {total_arm_time:.1f}s  |  '
                f'{total_arm_distance:.2f} rad joint distance  |  '
                f'{total_unreachable} viewpoint(s) unreachable'
                + (f' across regions {", ".join(regions_with_unreachable)}'
                   if regions_with_unreachable else ''))
            for rname in regions_with_unreachable:
                self.get_logger().warning(
                    f'  {rname}: unreachable viewpoints '
                    f'{region_gaps[rname]["unreachable"]}')
            msg = f"Algorithm: {primary_algorithm}  Total: {primary_total:.3f} m"
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
            # Reset every region's order back to an identity list (clears TSP and VRP paths).
            for mesh_entry in viewpoint_dict.get('meshes', []):
                mesh_entry.pop('vrp_orders', None)
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
                    joint_traj, _ = self._compute_joint_trajectory(
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

    def _run_vrp_algorithm(self, ik_result, region_offsets):
        """Dispatch to the selected VRP algorithm on vrp_solver."""
        algo = self.vrp_algorithm
        if algo == 'vrp_greedy':
            return self.vrp_solver.run_greedy(ik_result, region_offsets)
        if algo == 'vrp_2opt':
            return self.vrp_solver.run_2opt(ik_result, region_offsets)
        if algo == 'vrp_3opt':
            return self.vrp_solver.run_3opt(ik_result, region_offsets)
        topk = max(1, self.vrp_validate_topk)
        if algo == 'vrp_ils':
            return self.vrp_solver.run_ils(ik_result, region_offsets, n_candidates=topk)
        if algo == 'vrp_lkh':
            return self.vrp_solver.run_lkh(ik_result, region_offsets, n_candidates=topk)
        if algo == 'vrp_aco':
            return self.vrp_solver.run_aco(
                ik_result, region_offsets,
                n_ants=self.vrp_aco_n_ants, n_iter=self.vrp_aco_n_iter,
                alpha=self.vrp_aco_alpha, beta=self.vrp_aco_beta, rho=self.vrp_aco_rho,
                n_candidates=topk, n_jobs=max(1, self.vrp_aco_n_jobs))
        if algo == 'vrp_hierarchical':
            return self.vrp_solver.run_hierarchical(ik_result, region_offsets)
        if algo == 'vrp_clustered':
            return self.vrp_solver.run_clustered(
                ik_result, region_offsets, K=max(2, self.vrp_clustered_k))
        self.get_logger().warning(f'Unknown VRP algorithm: {algo}')
        return None

    def vrp(self, viewpoint_dict):
        if self.clear_paths:
            for mesh_entry in viewpoint_dict.get('meshes', []):
                mesh_entry.pop('vrp_orders', None)
                for region in mesh_entry.get('regions', []):
                    if isinstance(region.get('order'), dict):
                        for key in list(region['order'].keys()):
                            if key.startswith('vrp_'):
                                del region['order'][key]
            return viewpoint_dict

        algo = self.vrp_algorithm
        robot_model = self.robot.get_robot_model()

        for mesh_entry in viewpoint_dict.get('meshes', []):
            self.get_logger().info(f'VRP {algo}: running IK precomputation...')
            ik_result, region_offsets = self.vrp_solver.precompute_ik(
                mesh_entry, robot_model, self.planning_group)
            self._region_offsets = region_offsets

            # Pass 1: initial region ordering using DEPOT_Q-seeded IK
            solution = self._run_vrp_algorithm(ik_result, region_offsets)
            if solution is None:
                continue

            multi_config = (ik_result.q_configs is not None
                            and any(len(c) > 1 for c in ik_result.q_configs))
            if multi_config:
                # Config-aware solve already selects the per-viewpoint config to form a
                # continuous chain, so the single-config reseed pass would only discard
                # that freedom. Keep the swept configs and the pass-1 solution.
                self._ik_result = ik_result
            else:
                # Pass 2+: re-seed IK along the VRP path so joint configs form a
                # continuous chain from depot through all regions rather than clustering
                # near home, then re-optimize on the chained IK for accurate dm values.
                #
                # With vrp_chain_max_passes > 1 this iterates chain <-> re-solve toward a
                # fixed point (Spec L): each pass re-chains along the *current* best path,
                # tightening order/config consistency. Keep-best guarantees the result is
                # never worse than a single pass; it stops early once the order stabilizes.
                # vrp_chain_max_passes == 1 reproduces the original single-pass behavior.
                base_ik = ik_result
                order, paths = solution.region_order, solution.region_paths
                best_ik = best_sol = None
                best_cost = float('inf')
                for _pass in range(max(1, self.vrp_chain_max_passes)):
                    self.get_logger().info(
                        f'VRP {algo}: reseeding IK along path (pass {_pass + 1})...')
                    chained = self.vrp_solver.recompute_ik_chained(
                        base_ik, region_offsets, order, paths,
                        mesh_entry, robot_model, self.planning_group)
                    cand = self._run_vrp_algorithm(chained, region_offsets)
                    if cand is None:
                        break
                    if cand.cost < best_cost - 1e-9:
                        best_ik, best_sol, best_cost = chained, cand, cand.cost
                    converged = (cand.region_order == order)
                    order, paths = cand.region_order, cand.region_paths
                    if converged:
                        break
                if best_sol is None:
                    continue
                ik_result, solution = best_ik, best_sol
                self._ik_result = ik_result

            dm = self.vrp_solver.build_cost_matrix(ik_result)
            dd = self.vrp_solver.depot_dists(ik_result)
            regions = mesh_entry.get('regions', [])
            units = self.vrp_solver.cost_units

            # Validation: plan the top-K candidate tours with MoveIt and keep the one
            # with the lowest real execution time. This grounds the surrogate objective
            # in true planned motion. With validate_topk == 1 this simply plans the
            # single winning tour (no extra cost).
            jt_map, validated_time = {}, None
            if self.planning_component:
                candidates = (solution.candidates
                              if (self.vrp_validate_topk > 1 and solution.candidates)
                              else [(solution.region_order, solution.region_paths, solution.cost)])
                best_time, best, best_sel = float('inf'), None, None
                for c_order, c_paths, _ in candidates:
                    # When multiple IK configs exist (turntable sweep), pick the swing-free
                    # config assignment for this tour so *every* algorithm — not just
                    # vrp_clustered — plans to joint goals instead of free-IK poses.
                    sel = (self.vrp_solver.select_tour_configs(
                               c_order, c_paths, ik_result, region_offsets)
                           if multi_config else None)
                    cfg = sel[0] if sel else None
                    t, jm = self._plan_vrp_tour(c_order, c_paths, regions, region_configs=cfg)
                    if t < best_time:
                        best_time, jt_map, best, best_sel = t, jm, (c_order, c_paths), sel
                validated_time = best_time
                if len(candidates) > 1:
                    self.get_logger().info(
                        f'VRP {algo}: validated {len(candidates)} candidate tour(s); '
                        f'best real MoveIt time={best_time:.1f}s')
                # Adopt the validated winner and its cost breakdown.
                w_order, w_paths = best
                if best_sel is not None:
                    # config-aware winner: config-space cost + chosen configs
                    cfg_map, total, intra_c, inter_c, depot_c = best_sel
                    solution = VRPSolution(w_order, w_paths, total, intra_c, inter_c, depot_c,
                                           algorithm=algo, region_configs=cfg_map)
                else:
                    solution = VRPSolution(w_order, w_paths, 0.0, 0.0, 0.0, 0.0, algorithm=algo)
                    solution.cost, solution.intra_cost, solution.inter_cost, solution.depot_cost = \
                        self.vrp_solver.full_tour_cost(solution, dm, dd, region_offsets)

            mesh_entry.setdefault('vrp_orders', {})[algo] = solution.region_order
            for r_idx, region in enumerate(regions):
                if r_idx not in solution.region_paths:
                    continue
                path = solution.region_paths[r_idx]
                off = region_offsets[r_idx]
                goal = self._resolve_goal_configs(r_idx, list(path), solution.region_configs)
                if goal is not None and all(g is not None for g in goal):
                    intra_r = float(sum(self.vrp_solver._config_cost(goal[k] - goal[k + 1])
                                        for k in range(len(path) - 1)))
                else:
                    intra_r = float(sum(dm[off + path[k], off + path[k + 1]]
                                        for k in range(len(path) - 1)))
                order_dict = self._ensure_order_dict(region)
                order_dict[algo] = {'order': list(path), 'distance': intra_r}
                if solution.region_configs is not None and r_idx in solution.region_configs:
                    order_dict[algo]['configs'] = list(solution.region_configs[r_idx])
                if r_idx in jt_map:
                    order_dict[algo]['joint_trajectory'] = jt_map[r_idx]

            self._last_vrp_cost = solution.cost
            arm_c, tt_c = self.vrp_solver.tour_cost_breakdown(
                solution, ik_result, region_offsets)
            self.get_logger().info(
                f'VRP {algo}: total={solution.cost:.4f} {units}  '
                f'(intra={solution.intra_cost:.4f}  inter={solution.inter_cost:.4f}  '
                f'depot={solution.depot_cost:.4f})  region_order={solution.region_order}')
            self.get_logger().info(
                f'VRP {algo}: arm={arm_c:.4f} {units}  turntable={tt_c:.4f} {units}  '
                f'(arm {arm_c/(solution.cost+1e-10)*100:.1f}%  '
                f'tt {tt_c/(solution.cost+1e-10)*100:.1f}%)')
            if validated_time is not None:
                self.get_logger().info(
                    f'VRP {algo}: planned MoveIt tour time={validated_time:.1f}s')

        return viewpoint_dict

    def _plan_vrp_tour(self, region_order, region_paths, regions, region_configs=None):
        """Plan the full VRP tour with MoveIt, chaining inter-region transitions.

        When region_configs is provided (vrp_clustered config-aware solve), each
        viewpoint is planned to its chosen IK config as a joint-space goal rather than
        letting MoveIt re-resolve IK from the Cartesian pose.

        Returns (total_time_s, {r_idx: joint_trajectory}).
        """
        total_time, jt_map, prev_exit = 0.0, {}, None
        for r_idx in region_order:
            path = region_paths.get(r_idx)
            if not path:
                continue
            goal_qs = self._resolve_goal_configs(r_idx, list(path), region_configs)
            jt, prev_exit = self._compute_joint_trajectory(
                list(path), regions[r_idx].get('clusters', []), f'vrp_{r_idx}',
                initial_joint_state=prev_exit, goal_qs=goal_qs)
            jt_map[r_idx] = jt
            total_time += jt['total_time_s']
        return total_time, jt_map

    def _resolve_goal_configs(self, r_idx, path, region_configs):
        """Map region_configs[r_idx] (config idx per path position) to joint goal vectors
        via the precomputed q_configs, aligned to `path`. Returns None if unavailable."""
        if (region_configs is None or self._ik_result is None
                or self._ik_result.q_configs is None
                or r_idx not in region_configs or self._region_offsets is None):
            return None
        off = self._region_offsets[r_idx]
        cfg_idx = region_configs[r_idx]
        if len(cfg_idx) != len(path):
            return None
        qc = self._ik_result.q_configs
        goal_qs = []
        for k, vp in enumerate(path):
            configs = qc[off + vp]
            q = configs[cfg_idx[k]] if 0 <= cfg_idx[k] < len(configs) else None
            goal_qs.append(None if q is None or np.any(np.isnan(q)) else np.asarray(q))
        return goal_qs

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

        self.planning_component.set_goal_state(
            pose_stamped_msg=request.pose_goal, pose_link="eoat_camera_link")
        self.get_logger().info(f"Received request: {request}")
        # Plan and execute (uses multiplanning when enabled; see plan_and_execute).
        success = self.plan_and_execute()
        self.get_logger().info(f"Plan and execute called, success: {success}")

        # Prepare the response
        response.success = success
        response.message = "Motion completed successfully" if success else "Motion failed"

        return response

    def _build_multi_plan_params(self):
        """Multi-pipeline plan params when multiplanning is on, else None. MoveItPy
        races the listed named configs and keeps the shortest solution."""
        if not (self.multiplanning and self.multiplanning_pipelines):
            return None
        return MultiPipelinePlanRequestParameters(
            self.robot, list(self.multiplanning_pipelines))

    def _plan(self):
        """Plan from the set start/goal, racing pipelines when multiplanning is on."""
        if self._multi_plan_params is not None:
            return self.planning_component.plan(
                multi_plan_parameters=self._multi_plan_params)
        return self.planning_component.plan()

    # Function for planning and executing a trajectories
    def plan_and_execute(self, single_plan_parameters=None, multi_plan_parameters=None):
        # Check if the planning component is valid
        if not self.planning_component:
            self.get_logger().error("Planning component is not valid")
            return False

        self.planning_component.set_start_state_to_current_state()

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
            plan_result = self._plan()

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

    def _plan_segment(self, viewpoint, start_joint_positions=None, goal_joints=None):
        if start_joint_positions is not None:
            start_state = RobotState(self.robot.get_robot_model())
            start_state.set_joint_group_positions(
                self.planning_group, start_joint_positions)
            self.planning_component.set_start_state(robot_state=start_state)
        else:
            self.planning_component.set_start_state_to_current_state()

        if goal_joints is not None:
            # Plan directly to the chosen IK configuration (config-aware clustered solve).
            goal_state = RobotState(self.robot.get_robot_model())
            goal_state.set_joint_group_positions(self.planning_group, goal_joints)
            goal_state.update()
            self.planning_component.set_goal_state(robot_state=goal_state)
        else:
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

        plan_result = self._plan()
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

    def find_nearest_viewpoint_callback(self, request, response):
        r = request.region_idx
        if self._ik_result is None or self._region_offsets is None:
            response.nearest_viewpoint_idx = 0
            return response

        with self.robot.get_planning_scene_monitor().read_only() as scene:
            rs = scene.current_state
            current_q = np.array(rs.get_joint_group_positions(self.planning_group))

        off = self._region_offsets[r]
        n_total = len(self._ik_result.q_all)
        next_off = self._region_offsets[r + 1] if r + 1 < len(self._region_offsets) else n_total
        region_q = self._ik_result.q_all[off:next_off]
        fallback = self._ik_result.cartesian_fallback[off:next_off]

        best_idx, best_dist = 0, float('inf')
        for local_i, (q, is_fallback) in enumerate(zip(region_q, fallback)):
            if is_fallback:
                continue
            d = float(self.vrp_solver._config_cost(q - current_q))
            if d < best_dist:
                best_dist, best_idx = d, local_i

        response.nearest_viewpoint_idx = best_idx
        return response

    def _compute_joint_trajectory(self, path, clusters, region_key, initial_joint_state=None,
                                  goal_qs=None):
        result = {
            'total_time_s': 0.0,
            'total_joint_distance': 0.0,
            'cartesian_waypoints': [],
            'unreachable': [],
        }
        prev_joint_state = initial_joint_state

        # Plan the approach to the entry viewpoint path[0]. For the first region
        # prev_joint_state is None and _plan_segment starts from the current state;
        # for later regions it chains from the previous region's exit.
        if path:
            entry_idx = path[0]
            entry_vp = clusters[entry_idx].get('viewpoint') if entry_idx < len(clusters) else None
            if entry_vp is None:
                result['unreachable'].append(entry_idx)
            else:
                waypoints, duration = self._plan_segment(
                    entry_vp, prev_joint_state,
                    goal_joints=(goal_qs[0] if goal_qs else None))
                if waypoints is None:
                    result['unreachable'].append(entry_idx)
                    self.get_logger().warning(
                        f'  region {region_key}: entry viewpoint {entry_idx} unreachable')
                else:
                    joint_dist = float(sum(
                        np.linalg.norm(np.array(waypoints[i + 1]) - np.array(waypoints[i]))
                        for i in range(len(waypoints) - 1)
                    ))
                    cartesian = self._fk_waypoints(waypoints)
                    ds = max(1, len(cartesian) // 20)
                    result['cartesian_waypoints'].extend(cartesian[::ds])
                    result['total_time_s'] += duration
                    result['total_joint_distance'] += joint_dist
                    prev_joint_state = waypoints[-1]

        for i in range(len(path) - 1):
            from_idx, to_idx = path[i], path[i + 1]
            to_vp = clusters[to_idx].get('viewpoint') if to_idx < len(clusters) else None

            if not to_vp:
                result['unreachable'].append(to_idx)
                self.get_logger().warning(
                    f'  region {region_key}: viewpoint {to_idx} has no viewpoint data, skipping')
                continue

            waypoints, duration = self._plan_segment(
                to_vp, prev_joint_state,
                goal_joints=(goal_qs[i + 1] if goal_qs else None))
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
            ds = max(1, len(cartesian) // 20)
            result['cartesian_waypoints'].extend(cartesian[::ds])
            result['total_time_s'] += duration
            result['total_joint_distance'] += joint_dist
            prev_joint_state = waypoints[-1]

        self.planning_component.set_start_state_to_current_state()
        result['total_time_s'] = round(result['total_time_s'], 3)
        result['total_joint_distance'] = round(result['total_joint_distance'], 4)
        return result, prev_joint_state

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
            elif param.name == 'vrp_algorithm':
                self.vrp_algorithm = param.value
            elif param.name == 'vrp_joint_weights':
                self.vrp_solver.joint_weights = np.array(list(param.value), dtype=float)
                self.vrp_solver._w = np.sqrt(self.vrp_solver.joint_weights)
            elif param.name == 'vrp_cost_mode':
                self.vrp_cost_mode = param.value
                self.vrp_solver.cost_mode = param.value
            elif param.name == 'vrp_max_velocity':
                self.vrp_max_velocity = param.value
                self.vrp_solver._vmax = np.broadcast_to(float(param.value), (7,)).astype(float)
                self.vrp_solver._d_crit = self.vrp_solver._vmax ** 2 / self.vrp_solver._amax
            elif param.name == 'vrp_max_acceleration':
                self.vrp_max_acceleration = param.value
                self.vrp_solver._amax = np.broadcast_to(float(param.value), (7,)).astype(float)
                self.vrp_solver._d_crit = self.vrp_solver._vmax ** 2 / self.vrp_solver._amax
            elif param.name == 'vrp_validate_topk':
                self.vrp_validate_topk = param.value
            elif param.name == 'vrp_aco_n_jobs':
                self.vrp_aco_n_jobs = param.value
            elif param.name == 'vrp_aco_n_ants':
                self.vrp_aco_n_ants = param.value
            elif param.name == 'vrp_aco_n_iter':
                self.vrp_aco_n_iter = param.value
            elif param.name == 'vrp_aco_alpha':
                self.vrp_aco_alpha = param.value
            elif param.name == 'vrp_aco_beta':
                self.vrp_aco_beta = param.value
            elif param.name == 'vrp_aco_rho':
                self.vrp_aco_rho = param.value
            elif param.name == 'vrp_clustered_k':
                self.vrp_clustered_k = param.value
            elif param.name == 'vrp_n_turntable_samples':
                self.vrp_n_turntable_samples = param.value
                self.vrp_solver.n_turntable_samples = param.value
            elif param.name == 'vrp_max_configs_per_vp':
                self.vrp_max_configs_per_vp = param.value
                self.vrp_solver.max_configs_per_vp = param.value
            elif param.name == 'vrp_config_dedup_tol':
                self.vrp_config_dedup_tol = param.value
                self.vrp_solver.config_dedup_tol = param.value
            elif param.name == 'vrp_chain_max_passes':
                self.vrp_chain_max_passes = param.value
            elif param.name == 'multiplanning':
                self.multiplanning = param.value
                self._multi_plan_params = self._build_multi_plan_params()
            elif param.name == 'multiplanning_pipelines':
                self.multiplanning_pipelines = list(param.value)
                self._multi_plan_params = self._build_multi_plan_params()

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
