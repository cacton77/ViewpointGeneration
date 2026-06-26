import rclpy
import json
import re
import datetime
from rclpy.node import Node
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from rcl_interfaces.msg import SetParametersResult
from viewpoint_generation_interfaces.srv import MoveToPoseStamped, OptimizeViewpointTraversal
from geometry_msgs.msg import Pose
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
                ('compare', False),
                ('compare_algorithms', '-- none --'),
                ('result.primary_algorithm', ''),
                ('result.primary_distance', -1.0),
                ('result.compare_algorithm', ''),
                ('result.compare_distance', -1.0),
                ('result.mst_lower_bound', -1.0),
                ('result.optimality_gap_pct', -1.0),
                ('result.max_region_gap_pct', -1.0),
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
        self.compare = self.get_parameter(
            'compare').get_parameter_value().bool_value
        self.tsp_algorithm = self.get_parameter(
            'tsp_algorithm').get_parameter_value().string_value
        self.compare_algorithms = self.get_parameter(
            'compare_algorithms').get_parameter_value().string_value

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

        # Compute total distances across all regions for display
        primary_total = sum(
            v.get('distance', 0.0)
            for v in self.solver.algorithm_results.get(self.tsp_algorithm, {}).values()
        )
        compare_algo = (self.compare_algorithms
                        if self.compare and self.compare_algorithms != '-- none --'
                        else '')
        compare_total = sum(
            v.get('distance', 0.0)
            for v in self.solver.algorithm_results.get(compare_algo, {}).values()
        ) if compare_algo and compare_algo in self.solver.algorithm_results else -1.0

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

        completed_str = ','.join(sorted(self.solver.completed_algorithms))
        self.get_logger().info(
            f'Completed: {completed_str}  |  '
            f'MST total: {mst_total:.3f} m  |  '
            f'{self.tsp_algorithm}: {primary_total:.3f} m  |  '
            f'Overall gap: {gap_pct:.2f}%  |  Max region gap: {max_gap:.2f}%')

        self.set_parameters([
            rclpy.parameter.Parameter('result.primary_algorithm',
                                      rclpy.Parameter.Type.STRING, self.tsp_algorithm),
            rclpy.parameter.Parameter('result.primary_distance',
                                      rclpy.Parameter.Type.DOUBLE, float(primary_total)),
            rclpy.parameter.Parameter('result.compare_algorithm',
                                      rclpy.Parameter.Type.STRING, compare_algo),
            rclpy.parameter.Parameter('result.compare_distance',
                                      rclpy.Parameter.Type.DOUBLE, float(compare_total)),
            rclpy.parameter.Parameter('result.mst_lower_bound',
                                      rclpy.Parameter.Type.DOUBLE, float(mst_total)),
            rclpy.parameter.Parameter('result.optimality_gap_pct',
                                      rclpy.Parameter.Type.DOUBLE, float(gap_pct)),
            rclpy.parameter.Parameter('result.max_region_gap_pct',
                                      rclpy.Parameter.Type.DOUBLE, float(max_gap)),
        ])

        msg = f"Algorithm: {self.tsp_algorithm}  Total: {primary_total:.3f} m"
        if compare_algo and compare_total >= 0:
            improvement = (compare_total - primary_total) / compare_total * 100 if compare_total > 0 else 0.0
            msg += f" | Compare ({compare_algo}): {compare_total:.3f} m  Improvement: {improvement:.1f}%"
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
            self.set_parameters([
                rclpy.parameter.Parameter(
                    'result.primary_algorithm', rclpy.Parameter.Type.STRING, ''),
            ])
            return viewpoint_dict

        if self.tsp_algorithm == 'Select Algorithm':
            self.get_logger().warning('No algorithm selected.')
            return viewpoint_dict

        primary = self.tsp_algorithm
        algorithms = [primary]
        compare_algo = self.compare_algorithms
        if (self.compare and compare_algo and compare_algo != primary
                and compare_algo != '-- none --'):
            algorithms.append(compare_algo)

        for algo in algorithms:
            self.solver.algorithm_results.setdefault(algo, {})

        self.solver._region_gaps = {}

        for mesh_idx, mesh_entry in enumerate(viewpoint_dict.get('meshes', [])):
            for region_idx, region in enumerate(mesh_entry.get('regions', [])):
                clusters = region.get('clusters', [])
                order_dict = self._ensure_order_dict(region)
                n_clusters = len(clusters)
                region_key = f"{mesh_idx}:{region_idx}"

                # Regions too small to optimize (or missing viewpoints) keep an
                # identity order under every algorithm so downstream consumers
                # always find their selected algorithm's key.
                if n_clusters < 3 or any('viewpoint' not in c for c in clusters):
                    identity = list(range(n_clusters))
                    for algo in algorithms:
                        order_dict[algo] = identity
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

                for algo in algorithms:
                    self.get_logger().info(
                        f"Running {algo} on region {region_key} (N={n_vp})...")
                    path, distance = self.solver._run_algorithm(algo, viewpoints, dm)
                    if path is None:
                        continue

                    prev_best = self.solver.algorithm_results[algo].get(region_key, {}).get('distance', float('inf'))
                    improved = distance < prev_best - 1e-10
                    if improved or region_key not in self.solver.algorithm_results[algo]:
                        self.solver.algorithm_results[algo][region_key] = {
                            'path': path.copy(),
                            'distance': distance
                        }
                    else:
                        # Keep the previously stored best; restore distance for logging
                        distance = prev_best
                        path = self.solver.algorithm_results[algo][region_key]['path']

                    # Store this algorithm's path under its own key in the region.
                    path_to_save = path[:-1] if (len(path) > 0 and path[-1] == path[0]) else path
                    order_dict[algo] = list(path_to_save)

                    gap_pct = ((distance - lower_bound) / lower_bound * 100.0
                            if lower_bound > 1e-9 else 0.0)
                    tag = ' [EXACT OPTIMAL]' if (n_vp <= 18 and gap_pct < 0.01) else (
                        ' [best kept]' if not improved else '')
                    self.get_logger().info(
                        f'  {algo}: {distance:.4f} m | {bound_label} | '
                        f'Gap={gap_pct:.2f}%{tag}  (N={n_vp})')

                    if algo == primary:
                        self.solver._region_gaps[region_key] = {
                            'n': n_vp,
                            'distance': distance,
                            'lower_bound': lower_bound,
                            'gap_pct': gap_pct,
                            'exact': n_vp <= 18
                        }

                self.solver.completed_algorithms.add(primary)

        viewpoint_dict['selected_traversal_algorithm'] = primary
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
            elif param.name == 'compare':
                self.compare = param.value
            elif param.name == 'compare_algorithms':
                self.compare_algorithms = param.value

        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    traversal_node = ViewpointTraversalNode()
    rclpy.spin(traversal_node)


if __name__ == '__main__':
    main()
