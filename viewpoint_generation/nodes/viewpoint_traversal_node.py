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
                ('compare_algorithms', '2opt'),
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

    def dist_matrix(self, viewpoints):
        n = len(viewpoints)  # Extract position from .json under viewpoint
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = euclidean(viewpoints[i], viewpoints[j])

        return dist_matrix

    # Calculate the distance of the tour
    def dist_calc(self, dist_matrix, viewpoint):
        dist = 0
        tour = viewpoint[0] if isinstance(
            viewpoint, list) and len(viewpoint) > 1 else viewpoint
        for k in range(len(tour) - 1):
            m = k + 1
            dist += dist_matrix[tour[k], tour[m]]

        return dist

    def optimize_traversal(self, request, response):
        if not request.viewpoint_dict_path:
            response.success = False
            response.message = "No viewpoint dictionary path provided"
            return response

        self.get_logger().info(
            f'Optimizing traversal for file {request.viewpoint_dict_path}')
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

        response.success = True
        response.message = "Traversal optimization completed successfully"
        response.new_viewpoint_dict_path = new_viewpoint_dict_path
        return response

    def tsp(self, viewpoint_dict):
        if self.clear_paths:
            self.algorithm_results.clear()
            self.completed_algorithms.clear()
            return viewpoint_dict

        for region_name, region in viewpoint_dict['regions'].items():
            num_clusters = len(region['clusters'])
            viewpoint_dict['regions'][region_name]['order'] = list(
                range(num_clusters))

        regions_dict = viewpoint_dict['regions']
        current_algorithm = self.tsp_algorithm

        if current_algorithm not in self.algorithm_results:
            self.algorithm_results[current_algorithm] = {}

        for region_name, region in regions_dict.items():
            clusters_dict = region['clusters']
            viewpoints = []

            if len(viewpoint_dict['regions'][region_name]['order']) < 3:
                continue

            for cluster_name, cluster in clusters_dict.items():
                viewpoints.append(cluster['viewpoint']['position'])

            dist_matrix = self.dist_matrix(viewpoints)

            if current_algorithm == 'greedy':
                self.get_logger().info("Calling run_greedy...")
                path, distance = self.run_greedy(viewpoints)
            elif current_algorithm == '2opt':
                self.get_logger().info("Calling run_2opt...")
                path, distance = self.run_2opt(viewpoints, dist_matrix)
            elif current_algorithm == '3opt':
                self.get_logger().info("Calling run_3opt...")
                path, distance = self.run_3opt(viewpoints, dist_matrix)
            elif current_algorithm == 'LKH':
                self.get_logger().info("Calling run_lkh...")
                path, distance = self.run_lkh(viewpoints, dist_matrix)
            else:
                self.get_logger().error(
                    f"Unknown TSP algorithm: {current_algorithm}")
                continue

            self.get_logger().info(
                f"Algorithm completed. Distance: {distance:.4f}")
            self.get_logger().info(f"Path length: {len(path)}")

            self.algorithm_results[current_algorithm][region_name] = {
                'path': path.copy(),
                'distance': distance
            }

            path_to_save = path[:-1] if (len(path)
                                         > 0 and path[-1] == path[0]) else path
            self.get_logger().info(
                f"Saving path with {len(path_to_save)} points")
            viewpoint_dict['regions'][region_name]['order'] = path_to_save

        self.completed_algorithms.add(current_algorithm)

        # if self.compare and self.compare_algorithms in self.completed_algorithms:
        #         self.show_comparison(current_algorithm, self.compare_algorithms)

        return viewpoint_dict

    def nearest_neighbors_tsp(self, points):
        num_points = len(points)
        unvisited = set(range(num_points))
        current_point = 0
        path = [current_point]
        unvisited.remove(current_point)
        total_distance = 0

        while unvisited:
            next_point = min(unvisited, key=lambda point: euclidean(
                points[current_point], points[point]))
            total_distance += euclidean(points[current_point],
                                        points[next_point])
            current_point = next_point
            path.append(current_point)
            unvisited.remove(current_point)

        return path, total_distance

    def local_search_2_opt(self, dist_matrix, viewpoint, recursive_seeding=-1, verbose=False):
        if recursive_seeding < 0:
            count = -2
        else:
            count = 0

        if isinstance(viewpoint, list) and len(viewpoint) == 2:
            vp_list = copy.deepcopy(viewpoint)
        else:
            dist = self.dist_calc(dist_matrix, viewpoint)
            vp_list = [viewpoint.copy(), dist]

        dist = vp_list[1] * 2
        iteration = 0

        while count < recursive_seeding:
            best_route = copy.deepcopy(vp_list)
            seed = copy.deepcopy(vp_list)

            for i in range(len(vp_list[0]) - 2):
                for j in range(i + 2, len(vp_list[0])):
                    new_tour = (vp_list[0][:i+1] + vp_list[0]
                                [i+1:j+1][::-1] + vp_list[0][j+1:])

                    if len(new_tour) > 0 and new_tour[-1] != new_tour[0]:
                        new_tour.append(new_tour[0])

                    new_dist = self.dist_calc(dist_matrix, [new_tour, 0])

                    if new_dist < best_route[1]:
                        best_route[0] = new_tour
                        best_route[1] = new_dist

            if best_route[1] < vp_list[1]:
                vp_list = copy.deepcopy(best_route)

            count += 1
            iteration += 1

            if dist > vp_list[1] and recursive_seeding < 0:
                dist = vp_list[1]
                count = -2
                recursive_seeding = -1
            elif vp_list[1] >= dist and recursive_seeding < 0:
                count = -1
                recursive_seeding = -2

        return vp_list[0], vp_list[1]

    def local_search_3_opt(self, dist_matrix, viewpoint, recursive_seeding=-1):
        if recursive_seeding < 0:
            count = -2
        else:
            count = 0

        if isinstance(viewpoint, list) and len(viewpoint) == 2:
            vp_list = copy.deepcopy(viewpoint)
        else:
            dist = self.dist_calc(dist_matrix, viewpoint)
            vp_list = [viewpoint.copy(), dist]

        dist = vp_list[1] * 2
        iteration = 0
        max_iterations = 10

        while count < recursive_seeding and iteration < max_iterations:
            best_route = copy.deepcopy(vp_list)
            improvement_found = False

            tour = vp_list[0]
            if tour[-1] == tour[0]:
                tour = tour[:-1]

            n = len(tour)

            for i in range(n - 2):
                if improvement_found:
                    break
                for j in range(i + 1, n - 1):
                    if improvement_found:
                        break
                    for k in range(j + 1, n):
                        if improvement_found:
                            break
                        # Generate all possible 3-opt moves
                        segment_A = tour[:i+1]
                        segment_B = tour[i+1:j+1]
                        segment_C = tour[j+1:k+1]
                        segment_D = tour[k+1:]

                        new_tours = [
                            segment_A + segment_B[::-1] +
                            segment_C + segment_D,
                            segment_A + segment_B +
                            segment_C[::-1] + segment_D,
                            segment_A + segment_C + segment_B + segment_D,
                            segment_A + segment_B[::-1] +
                            segment_C[::-1] + segment_D,
                            segment_A + segment_C +
                            segment_B[::-1] + segment_D,
                            segment_A + segment_C[::-1] +
                            segment_B + segment_D,
                            segment_A + segment_C[::-1] +
                            segment_B[::-1] + segment_D,
                        ]
                        # Check each new tour and update to the best route
                        for new_tour in new_tours:
                            if len(new_tour) > 0 and new_tour[-1] != new_tour[0]:
                                new_tour_closed = new_tour + [new_tour[0]]
                            else:
                                new_tour_closed = new_tour
                            new_dist = self.dist_calc(
                                dist_matrix, [new_tour_closed, 0])
                            if new_dist < best_route[1]:
                                best_route[0] = new_tour_closed
                                best_route[1] = new_dist
                                improvement_found = True
                                break
            # update the best route if an improvement was found
            if best_route[1] < vp_list[1]:
                vp_list = copy.deepcopy(best_route)

            count += 1
            iteration += 1
            if dist > vp_list[1] and recursive_seeding < 0:
                dist = vp_list[1]
                count = -2
                recursive_seeding = -1
            elif vp_list[1] >= dist and recursive_seeding < 0:
                count = -1
                recursive_seeding = -2

        return vp_list[0], vp_list[1]

    def lin_kernighan_helsgaun(self, viewpoints, dist_matrix, initial_solution, num_iterations=10, num_candidates=5):
        viewpoints_array = np.array(viewpoints)
        n = len(viewpoints)
        # initialize the tour and distance
        if isinstance(initial_solution, list) and len(initial_solution) == 2:
            current_tour = copy.deepcopy(initial_solution[0])
            current_dist = initial_solution[1]
        else:
            current_tour = copy.deepcopy(initial_solution)
            current_dist = self.dist_calc(dist_matrix, [current_tour, 0])
        if current_tour[-1] == current_tour[0]:
            current_tour = current_tour[:-1]
        # build candidate sets based on nearest neighbors for each viewpoint
        candidate_edges = self.build_candidate_sets(
            viewpoints_array, dist_matrix, num_candidates)
        # check the best tour and distance
        best_tour = current_tour.copy()
        best_dist = current_dist
        for iteration in range(num_iterations):  # perform LKH iterations
            improved_tour, improved_dist = self.lk_with_candidates(
                current_tour, dist_matrix, candidate_edges)
            if improved_dist < best_dist:
                best_tour = improved_tour.copy()
                best_dist = improved_dist
                current_tour = improved_tour.copy()
                current_dist = improved_dist
            if iteration < num_iterations - 1:
                current_tour = self.double_bridge_perturbation(current_tour)
                current_dist = self.dist_calc(
                    dist_matrix, [current_tour + [current_tour[0]], 0])

        best_tour_closed = best_tour + [best_tour[0]]
        return best_tour_closed, best_dist

    # Build candidate sets based on nearest neighbors
    def build_candidate_sets(self, viewpoints, dist_matrix, k):
        n = len(viewpoints)
        candidates = {}
        for i in range(n):
            distances = [(dist_matrix[i, j], j) for j in range(n) if i != j]
            distances.sort()
            candidates[i] = [j for _, j in distances[:k]]
        return candidates

    # Lin-Kernighan heuristic with candidate sets
    def lk_with_candidates(self, tour, dist_matrix, candidates):
        n = len(tour)
        current_tour = tour.copy()
        current_dist = self.dist_calc(
            dist_matrix, [current_tour + [current_tour[0]], 0])

        improved = True
        iterations = 0
        max_iter = 50
        # Perform 2-opt moves using candidate edges
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(n):
                if improved:
                    break
                for j in candidates[current_tour[i]]:
                    try:
                        j_pos = current_tour.index(j)
                    except ValueError:
                        continue
                    if abs(i - j_pos) <= 1:
                        continue
                    if i < j_pos:
                        new_tour = (
                            current_tour[:i+1] + current_tour[i+1:j_pos+1][::-1] + current_tour[j_pos+1:])
                    else:
                        new_tour = (
                            current_tour[:j_pos+1] + current_tour[j_pos+1:i+1][::-1] + current_tour[i+1:])

                    new_tour_closed = new_tour + [new_tour[0]]
                    new_dist = self.dist_calc(
                        dist_matrix, [new_tour_closed, 0])
                    if new_dist < current_dist:
                        current_tour = new_tour
                        current_dist = new_dist
                        improved = True
                        break

        return current_tour, current_dist

    # Perturbation method to escape local minima:
    def double_bridge_perturbation(self, tour, num_breaks=4):
        n = len(tour)
        if n < 8:
            return tour
        cuts = sorted(random.sample(range(1, n), num_breaks))
        segments = []
        prev = 0
        for cut in cuts:
            segments.append(tour[prev:cut])
            prev = cut
        segments.append(tour[prev:])

        if len(segments) >= 5:
            new_tour = segments[0] + segments[2] + \
                segments[1] + segments[3] + segments[4]
        else:
            new_tour = tour

        return new_tour

    # Clearing the Traversal Paths which were generated in the previous run and highlight only the projected viewpoints.
    def clear_paths_callback(self, request, response):
        self.algorithm_results.clear()
        self.completed_algorithms.clear()

        response.success = True
        response.message = "Paths cleared"
        return response

    def run_greedy(self, viewpoints):
        path, distance = self.nearest_neighbors_tsp(viewpoints)
        return path, distance

    def run_2opt(self, viewpoints, dist_matrix):
        initial_path, initial_dist = self.nearest_neighbors_tsp(viewpoints)
        optimized_path, optimized_dist = self.local_search_2_opt(
            dist_matrix, [initial_path, initial_dist], recursive_seeding=-1)
        return optimized_path, optimized_dist

    def run_3opt(self, viewpoints, dist_matrix):
        initial_path, initial_dist = self.nearest_neighbors_tsp(viewpoints)
        optimized_path, optimized_dist = self.local_search_3_opt(
            dist_matrix, [initial_path, initial_dist], recursive_seeding=-1)
        return optimized_path, optimized_dist

    def run_lkh(self, viewpoints, dist_matrix):
        initial_path, initial_dist = self.nearest_neighbors_tsp(viewpoints)
        optimized_path, optimized_dist = self.lin_kernighan_helsgaun(
            viewpoints, dist_matrix, [initial_path, initial_dist])
        return optimized_path, optimized_dist

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
            elif param.name == 'compare':
                self.compare = param.value
            elif param.name == 'compare_algorithms':
                self.compare_algorithms = param.value

        return SetParametersResult(successful=True)


def main():
    rclpy.init()

    traversal_node = ViewpointTraversalNode()
    # traversal_node.plan1()  # Call the plan1 method to execute the first plan
    rclpy.spin(traversal_node)


if __name__ == '__main__':
    main()
