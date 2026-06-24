import time
import rclpy
import json
import re
import datetime
import random
from pprint import pprint
from scipy.spatial.distance import euclidean
# moveit python library
from rclpy.node import Node
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    PlanRequestParameters,
    MultiPipelinePlanRequestParameters,
)
from rclpy.logging import get_logger
from rcl_interfaces.msg import SetParametersResult
from viewpoint_generation_interfaces.srv import MoveToPoseStamped, OptimizeViewpointTraversal
from geometry_msgs.msg import PoseStamped, Pose
import pprint
import copy
import numpy as np

from std_srvs.srv import Trigger
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class ViewpointTraversalNode(Node):

    viewpoint_dict = {}
    algorithm_results = {}
    completed_algorithms = set()
    _region_gaps = {}

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

        self.robot = MoveItPy(node_name='moveit_py')

        # setting planner_id (Try)
        # self.single_plan_parameters = PlanRequestParameters(
        #     self.robot, self.get_parameter('planning_group').value)
        # planner_id = ['ompl_rrtc', 'chomp', 'pilz_industrial_motion_planner']
        # self.single_plan_parameters.planning_pipeline = 'ompl'
        # self.single_plan_parameters.planner_id = planner_id[0]
        # self.get_logger().info(f"Using planner: {planner_id[0]}")

        print(type(self.robot))
        print("------------------------------------")
        self.planning_scene_monitor = self.robot.get_planning_scene_monitor()
        # self.add_ground_plane()

        try:
            self.get_logger().info("Initializing MoveItPy")
            print("Initializing MoveItPy")
            self.planning_component = self.robot.get_planning_component(
                self.planning_group)
            self.get_logger().info("Planning component 'disc_to_ur5e' initialized successfully")
        except Exception as e:
            self.get_logger().error(
                f"Failed to get planning component: {e}")
            self.planning_component = None
            return

        print("Planning component initialized successfully")
        # Create a service to move to a specific pose

        services_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_service(
            MoveToPoseStamped,
            'viewpoint_traversal/move_to_pose_stamped',
            self.move_to_pose_stamped_callback,
            callback_group=services_cb_group
        )
        self.get_logger().info("Service 'move_to_pose_stamped' created successfully")

        self.create_service(OptimizeViewpointTraversal,
                            f'{node_name}/optimize_traversal',
                            self.optimize_traversal,
                            callback_group=services_cb_group
                            )

        self.add_on_set_parameters_callback(self.parameter_callback)

        # self.init_workspace()

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

    # Create a Distance Matrix from viewpoint co-ordinates using Euclidean distance.

    def _held_karp(self, dm):
        """Exact open-path TSP via bitmask DP.
        Uses numpy arrays for N > 12 to handle up to N = 18 in reasonable time."""
        n = dm.shape[0]
        size = 1 << n
        INF = 1e18
        dp = np.full((size, n), INF)
        for s in range(n):
            dp[1 << s, s] = 0.0

        nexts_for = [np.array([j for j in range(n) if not (mask >> j & 1)], dtype=np.int32)
                     for mask in range(size)]

        for mask in range(1, size):
            nexts = nexts_for[mask]
            if len(nexts) == 0:
                continue
            for last in range(n):
                if not (mask >> last & 1):
                    continue
                cur = dp[mask, last]
                if cur >= INF:
                    continue
                costs = cur + dm[last, nexts]
                new_masks = mask | (1 << nexts)
                for k in range(len(nexts)):
                    nm = int(new_masks[k])
                    nxt = int(nexts[k])
                    if costs[k] < dp[nm, nxt]:
                        dp[nm, nxt] = costs[k]

        full = size - 1
        return float(np.min(dp[full]))

    def _mst_cost(self, dm):
        """Prim's MST — O(N²), lower bound on any open Hamiltonian path."""
        n = dm.shape[0]
        in_mst = np.zeros(n, dtype=bool)
        key = np.full(n, np.inf)
        key[0] = 0.0
        total = 0.0
        for _ in range(n):
            candidates = np.where(~in_mst)[0]
            u = candidates[np.argmin(key[candidates])]
            in_mst[u] = True
            total += key[u]
            mask = ~in_mst
            better = dm[u, mask] < key[mask]
            key[mask] = np.where(better, dm[u, mask], key[mask])
        return float(total)

    def dist_matrix(self, viewpoints):
        pts = np.array(viewpoints)
        diff = pts[:, None, :] - pts[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def dist_calc(self, dm, path):
        t = path[0] if (isinstance(path, list) and len(path) == 2
                        and isinstance(path[0], list)) else path
        if len(t) > 1 and t[-1] == t[0]:
            t = t[:-1]
        return float(sum(dm[t[k], t[k + 1]] for k in range(len(t) - 1)))

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
            for v in self.algorithm_results.get(self.tsp_algorithm, {}).values()
        )

        # MST lower bound (summed across all regions of all meshes)
        mst_total = 0.0
        for mesh_entry in viewpoint_dict.get('meshes', []):
            for region in mesh_entry.get('regions', []):
                clusters = region.get('clusters', [])
                vps = [c['viewpoint']['position']
                       for c in clusters if 'viewpoint' in c]
                if len(vps) >= 2:
                    mst_total += self._mst_cost(self.dist_matrix(vps))

        gap_pct = ((primary_total - mst_total) / mst_total * 100.0
                   if mst_total > 1e-9 else 0.0)

        # Per-region summary table
        region_gaps = getattr(self, '_region_gaps', {})
        max_gap = max((v['gap_pct'] for v in region_gaps.values()), default=0.0)
        self.get_logger().info('─' * 70)
        self.get_logger().info(
            f'{"Region":<10} {"N":>4}  {"Distance":>10}  {"LowerBound":>11}  '
            f'{"Gap%":>7}  {"BoundType":>12}')
        for rname, info in sorted(region_gaps.items()):
            btype = 'Held-Karp' if info['exact'] else 'MST'
            opt_flag = ' ✓' if info['exact'] and info['gap_pct'] < 0.01 else ''
            self.get_logger().info(
                f'{rname:<10} {info["n"]:>4}  {info["distance"]:>10.4f}'
                f'  {info["lower_bound"]:>11.4f}  {info["gap_pct"]:>6.2f}%'
                f'  {btype:>12}{opt_flag}')
        self.get_logger().info('─' * 70)

        completed_str = ','.join(sorted(self.completed_algorithms))
        self.get_logger().info(
            f'Completed: {completed_str}  |  '
            f'MST total: {mst_total:.3f} m  |  '
            f'{self.tsp_algorithm}: {primary_total:.3f} m  |  '
            f'Overall gap: {gap_pct:.2f}%  |  Max region gap: {max_gap:.2f}%')

        # Per-algorithm path metrics (distance, etc.) are stored per region in
        # the results JSON under each algorithm's key, not in node parameters.
        msg = f"Algorithm: {self.tsp_algorithm}  Total: {primary_total:.3f} m"
        self.get_logger().info(msg)

        response.success = True
        response.message = msg
        response.new_viewpoint_dict_path = new_viewpoint_dict_path
        return response

    def _run_algorithm(self, algo, viewpoints, dist_matrix):
        if algo == 'greedy':
            return self.run_greedy(viewpoints)
        elif algo == '2opt':
            return self.run_2opt(viewpoints, dist_matrix)
        elif algo == '3opt':
            return self.run_3opt(viewpoints, dist_matrix)
        elif algo == 'ILS':
            return self.run_ils(viewpoints, dist_matrix)
        elif algo == 'LKH':
            return self.run_lkh(viewpoints, dist_matrix)
        else:
            self.get_logger().error(f"Unknown TSP algorithm: {algo}")
            return None, None

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
            self.algorithm_results.clear()
            self.completed_algorithms.clear()
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
        self.algorithm_results.setdefault(primary, {})

        self._region_gaps = {}

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
                        identity_distance = self.dist_calc(
                            self.dist_matrix(vps), identity)
                    else:
                        identity_distance = 0.0
                    order_dict[primary] = {
                        'order': identity,
                        'distance': float(identity_distance),
                    }
                    continue

                viewpoints = [c['viewpoint']['position'] for c in clusters]
                n_vp = len(viewpoints)
                dm = self.dist_matrix(viewpoints)

                # Lower bound for this region
                if n_vp <= 18:
                    if n_vp > 12:
                        self.get_logger().info(
                            f'  Running Held-Karp exact solver for N={n_vp} (~30s)...')
                    lower_bound = self._held_karp(dm)
                    bound_label = f'HK(exact)={lower_bound:.4f}'
                else:
                    lower_bound = self._mst_cost(dm)
                    bound_label = f'MST={lower_bound:.4f}'

                self.get_logger().info(
                    f"Running {primary} on region {region_key} (N={n_vp})...")
                path, distance = self._run_algorithm(primary, viewpoints, dm)
                if path is None:
                    continue

                prev_best = self.algorithm_results[primary].get(region_key, {}).get('distance', float('inf'))
                improved = distance < prev_best - 1e-10
                if improved or region_key not in self.algorithm_results[primary]:
                    self.algorithm_results[primary][region_key] = {
                        'path': path.copy(),
                        'distance': distance
                    }
                else:
                    # Keep the previously stored best; restore distance for logging
                    distance = prev_best
                    path = self.algorithm_results[primary][region_key]['path']

                # Store this algorithm's path and metrics under its own key.
                path_to_save = path[:-1] if (len(path) > 0 and path[-1] == path[0]) else path
                order_dict[primary] = {
                    'order': list(path_to_save),
                    'distance': float(distance),
                }

                gap_pct = ((distance - lower_bound) / lower_bound * 100.0
                        if lower_bound > 1e-9 else 0.0)
                tag = ' [EXACT OPTIMAL]' if (n_vp <= 18 and gap_pct < 0.01) else (
                    ' [best kept]' if not improved else '')
                self.get_logger().info(
                    f'  {primary}: {distance:.4f} m | {bound_label} | '
                    f'Gap={gap_pct:.2f}%{tag}  (N={n_vp})')

                self._region_gaps[region_key] = {
                    'n': n_vp,
                    'distance': distance,
                    'lower_bound': lower_bound,
                    'gap_pct': gap_pct,
                    'exact': n_vp <= 18
                }

                self.completed_algorithms.add(primary)

        # The selected algorithm is a parameter on the task_planning node (set
        # via the GUI), which drives both the visualized path and the execution
        # order. Each region keeps its per-algorithm 'order' dict here — each
        # value being {'order': [...], 'distance': ...} — so any optimized
        # algorithm can be chosen.
        return viewpoint_dict

    # ── Core primitives ───────────────────────────────────────────────────────

    def _unpack(self, initial):
        """Accept either a plain list or [path, dist] pair."""
        if isinstance(initial, list) and len(initial) == 2 and isinstance(initial[0], list):
            return list(initial[0])
        return list(initial)

    def _greedy_from(self, dm, start):
        n = dm.shape[0]
        visited = [False] * n
        visited[start] = True
        path = [start]
        cost = 0.0
        cur = start
        for _ in range(n - 1):
            row = dm[cur]
            best_d, best_j = float('inf'), -1
            for j in range(n):
                if not visited[j] and row[j] < best_d:
                    best_d, best_j = row[j], j
            cost += best_d
            visited[best_j] = True
            path.append(best_j)
            cur = best_j
        return path, cost

    def _two_opt(self, path, dm):
        """Best-improvement 2-opt on open path. No tour closure anywhere."""
        t = list(path)
        n = len(t)
        while True:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    # Reversing t[i+1..j] removes edges (t[i],t[i+1]) and (t[j],t[j+1])
                    # and adds (t[i],t[j]) and (t[i+1],t[j+1]).
                    # When j == n-1 there is no edge after t[j], so cost change is only:
                    # remove d(t[i],t[i+1]), add d(t[i],t[j])
                    remove = dm[t[i], t[i + 1]]
                    add = dm[t[i], t[j]]
                    if j + 1 < n:
                        remove += dm[t[j], t[j + 1]]
                        add += dm[t[i + 1], t[j + 1]]
                    if add - remove < -1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
            if not improved:
                break
        return t, self.dist_calc(dm, t)

    def _or_opt(self, path, dm, seg_len=1):
        """Relocate every run of seg_len nodes to its best insertion position."""
        t = list(path)
        n = len(t)
        while True:
            improved = False
            i = 0
            while i < n - seg_len:
                prev = i - 1
                nxt = i + seg_len
                # Segment must be strictly interior (has a node before and after)
                if prev < 0 or nxt >= n:
                    i += 1
                    continue
                seg = t[i:i + seg_len]
                # Gain from removing the segment
                remove_gain = (dm[t[prev], seg[0]] + dm[seg[-1], t[nxt]]
                               - dm[t[prev], t[nxt]])
                rest = t[:i] + t[i + seg_len:]
                m = len(rest)
                best_gain = 1e-10
                best_j = -1
                for j in range(m - 1):
                    ins_cost = (dm[rest[j], seg[0]] + dm[seg[-1], rest[j + 1]]
                                - dm[rest[j], rest[j + 1]])
                    gain = remove_gain - ins_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0:
                    t = rest[:best_j + 1] + seg + rest[best_j + 1:]
                    n = len(t)
                    improved = True
                else:
                    i += 1
            if not improved:
                break
        return t, self.dist_calc(dm, t)

    def _local_search(self, path, dm):
        t = list(path)
        prev_d = float('inf')
        while True:
            t, _ = self._two_opt(t, dm)
            t, _ = self._or_opt(t, dm, seg_len=1)
            t, _ = self._or_opt(t, dm, seg_len=2)
            t, _ = self._or_opt(t, dm, seg_len=3)
            d = self.dist_calc(dm, t)
            if prev_d - d < 1e-10:
                break
            prev_d = d
        return t, d

    def _double_bridge(self, tour):
        n = len(tour)
        if n < 8:
            return tour[:]
        cuts = sorted(random.sample(range(1, n), 4))
        a, b, c, d = cuts
        return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]

    # ── Public algorithm entry points ─────────────────────────────────────────

    def nearest_neighbors_tsp(self, points):
        dm = self.dist_matrix(points)
        best_path, best_cost = self._greedy_from(dm, 0)
        for s in range(1, len(points)):
            p, c = self._greedy_from(dm, s)
            if c < best_cost:
                best_cost, best_path = c, p
        return best_path, best_cost

    def local_search_2_opt(self, dm, initial, recursive_seeding=-1, verbose=False):
        return self._two_opt(self._unpack(initial), dm)

    def local_search_3_opt(self, dm, initial, recursive_seeding=-1):
        t = self._unpack(initial)
        n = len(t)
        while True:
            best_t = t[:]
            best_d = self.dist_calc(dm, t)
            improved = False
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        A, B, C, D = t[:i+1], t[i+1:j+1], t[j+1:k+1], t[k+1:]
                        for cand in [
                            A + B[::-1] + C + D,
                            A + B + C[::-1] + D,
                            A + C + B + D,
                            A + B[::-1] + C[::-1] + D,
                            A + C + B[::-1] + D,
                            A + C[::-1] + B + D,
                            A + C[::-1] + B[::-1] + D,
                        ]:
                            d = self.dist_calc(dm, cand)
                            if d < best_d - 1e-10:
                                best_d, best_t, improved = d, cand, True
            t = best_t
            if not improved:
                break
        return t, self.dist_calc(dm, t)

    def lin_kernighan_helsgaun(self, viewpoints, dm, initial_solution, num_iterations=10, num_candidates=5):
        best_path, best_dist = self._local_search(self._unpack(initial_solution), dm)
        n = len(viewpoints)
        max_restarts = max(num_iterations, n)
        patience = max(15, n // 3)
        no_improve = 0
        for _ in range(max_restarts):
            candidate, c_dist = self._local_search(self._double_bridge(best_path), dm)
            if c_dist < best_dist - 1e-10:
                best_path, best_dist = candidate, c_dist
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        return best_path, best_dist

    def build_candidate_sets(self, viewpoints, dm, k):
        n = len(viewpoints)
        candidates = {}
        for i in range(n):
            row = sorted((dm[i, j], j) for j in range(n) if j != i)
            candidates[i] = [j for _, j in row[:k]]
        return candidates

    def double_bridge_perturbation(self, tour, num_breaks=4):
        return self._double_bridge(tour)

    def clear_paths_callback(self, request, response):
        self.algorithm_results.clear()
        self.completed_algorithms.clear()
        response.success = True
        response.message = "Paths cleared"
        return response

    def run_greedy(self, viewpoints):
        return self.nearest_neighbors_tsp(viewpoints)

    def run_2opt(self, viewpoints, dm):
        path, _ = self.nearest_neighbors_tsp(viewpoints)
        return self._two_opt(path, dm)

    def run_3opt(self, viewpoints, dm):
        path, _ = self.nearest_neighbors_tsp(viewpoints)
        return self.local_search_3_opt(dm, path)

    def run_ils(self, viewpoints, dm):
        """ILS: multi-start greedy + exhaustive (2-opt + Or-opt) + double-bridge restarts."""
        path, dist = self.nearest_neighbors_tsp(viewpoints)
        return self.lin_kernighan_helsgaun(viewpoints, dm, [path, dist])

    def _lk_local_search(self, path, dm, candidates_set):
        """
        True Lin-Kernighan local search with variable-depth sequential moves.

        For each edge (t[i], t[i+1]), tries a sequential chain:

        Depth 1 — 2-opt with sequential gain criterion:
          Remove (t[i],t[i+1]) and (t[j],t[j+1]).
          Add    (t[i],t[j])   and (t[i+1],t[j+1]).
          Sequential criterion: d(t[i],t[i+1]) - d(t[i],t[j]) > 0  (must hold at depth 1)

        Depth 2 — specific 3-opt S1+S2_rev+S3_rev+S4 (when depth-1 alone fails):
          Remove (t[i],t[i+1]), (t[j],t[j+1]), (t[k],t[k+1]).
          Add    (t[i],t[j]),   (t[i+1],t[k]), (t[j+1],t[k+1]).
          Sequential criterion: ALSO requires g1+d(t[j],t[j+1])-d(t[i+1],t[k]) > 0.
          t[k] must be a candidate of t[i+1] (the "dangling end" from depth 1).

        The depth-2 move is what makes LK distinct: it cannot be produced by any single
        2-opt or Or-opt move. It is the direct product of the LK sequential chain.
        """
        t = list(path)
        n = len(t)

        improved = True
        while improved:
            improved = False

            for i in range(n - 2):
                if improved:
                    break

                t_i  = t[i]
                t_i1 = t[i + 1]
                d1   = dm[t_i, t_i1]   # cost of the edge we want to remove

                for j in range(i + 2, n):
                    if improved:
                        break

                    t_j = t[j]
                    if t_j not in candidates_set.get(t_i, set()):
                        continue

                    # LK sequential gain criterion at depth 1
                    g1 = d1 - dm[t_i, t_j]
                    if g1 <= 0:
                        continue

                    t_j1 = t[j + 1] if j + 1 < n else None

                    # ── Depth-1 close: standard 2-opt ─────────────────────
                    d2_rem = dm[t_j,  t_j1] if t_j1 is not None else 0.0
                    d2_add = dm[t_i1, t_j1] if t_j1 is not None else 0.0
                    if g1 + d2_rem - d2_add > 1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
                        break

                    # ── Depth-2 close: S1+S2_rev+S3_rev+S4 ───────────────
                    if t_j1 is None:
                        continue   # need the j+1 edge to go deeper

                    # Accumulated gain after removing first two edges and
                    # adding the first new edge (t_i → t_j).
                    G2 = g1 + dm[t_j, t_j1]

                    for k in range(j + 2, n):
                        t_k = t[k]
                        if t_k not in candidates_set.get(t_i1, set()):
                            continue

                        # LK sequential gain criterion at depth 2
                        g2 = G2 - dm[t_i1, t_k]
                        if g2 <= 0:
                            continue

                        t_k1 = t[k + 1] if k + 1 < n else None
                        d3_rem = dm[t_k,  t_k1] if t_k1 is not None else 0.0
                        d3_add = dm[t_j1, t_k1] if t_k1 is not None else 0.0
                        if g2 + d3_rem - d3_add > 1e-10:
                            # Apply S1 + S2_rev + S3_rev + S4
                            t = (t[:i + 1]
                                 + t[i + 1:j + 1][::-1]
                                 + t[j + 1:k + 1][::-1]
                                 + t[k + 1:])
                            n = len(t)
                            improved = True
                            break

        return t, self.dist_calc(dm, t)

    def run_lkh(self, viewpoints, dm, n_runs=50):
        """
        Lin-Kernighan solver: n_runs independent ILS passes with candidate-restricted local search.

        Each run starts from a different greedy seed node so the double-bridge
        restarts explore different regions of solution space. The global best
        across all runs is returned.  n_runs=5 by default so one button press
        is equivalent to what previously required 5 manual presses.
        """
        n = len(viewpoints)
        k = min(n - 1, 7)
        raw_cands = self.build_candidate_sets(viewpoints, dm, k)
        candidates_set = {i: set(nbrs) for i, nbrs in raw_cands.items()}

        patience = max(15, n // 3)
        global_best_path, global_best_dist = None, float('inf')

        for run in range(n_runs):
            # Spread starting seeds evenly across the node indices
            seed = (run * max(1, n // n_runs)) % n
            path, _ = self._greedy_from(dm, seed)
            path, dist = self._lk_local_search(path, dm, candidates_set)

            no_improve = 0
            for _ in range(max(15, n)):
                candidate, c_dist = self._lk_local_search(
                    self._double_bridge(path), dm, candidates_set)
                if c_dist < dist - 1e-10:
                    path, dist = candidate, c_dist
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

            if dist < global_best_dist - 1e-10:
                global_best_path, global_best_dist = path, dist
            self.get_logger().debug(f'    LKH run {run + 1}/{n_runs}: {dist:.4f} m')

        return global_best_path, global_best_dist

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

    def parameter_callback(self, params):
        """ Callback for parameter changes.
        :param params: List of parameters that have changed.
        :return: SetParametersResult indicating success or failure.
        """

        # Iterate through the parameters and set the corresponding values
        # based on the parameter name
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

    traversal_node = ViewpointTraversalNode()
    # traversal_node.plan1()  # Call the plan1 method to execute the first plan
    rclpy.spin(traversal_node)


if __name__ == '__main__':
    main()
