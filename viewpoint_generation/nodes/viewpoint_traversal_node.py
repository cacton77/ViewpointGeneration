import time
import rclpy
import json
import re
import datetime
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
                ('use_2opt', True)
            ]
        )

        self.planning_group = self.get_parameter(
            'planning_group').get_parameter_value().string_value
        self.planner = self.get_parameter(
            'planner').get_parameter_value().string_value
        self.multiplanning = self.get_parameter(
            'multiplanning').get_parameter_value().bool_value
        self.use_2opt = self.get_parameter(
            'use_2opt').get_parameter_value().bool_value

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
        n = len(viewpoints) #Extract position from .json under viewpoint
        dist_matrix = cp.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = euclidean(viewpoints[i], viewpoints[j])

        return dist_matrix                        

    # Calculate the distance of the tour
    def dist_calc(self, dist_matrix, viewpoint):
        dist = 0
        tour = viewpoint[0] if isinstance(viewpoint, list) and len(viewpoint) > 1 else viewpoint
        for k in range(len(tour) - 1):
            m = k + 1
            dist += dist_matrix[tour[k], tour[m]]
        
        return dist

    def optimize_traversal(self, request, response):
        self.get_logger().info(
            f'Optimizing traversal for file {request.viewpoint_dict_path}')
        with open(request.viewpoint_dict_path, 'r') as f:
            viewpoint_dict = json.load(f)


        viewpoint_dict_optimized = self.tsp(viewpoint_dict)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_viewpoint_dict_path = re.sub(
            r'_optimized.*?\.json$',
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

        regions_dict = viewpoint_dict['regions']
        for region_name, region in regions_dict.items():
            clusters_dict = region['clusters']

            viewpoints = []

            if len(viewpoint_dict['regions'][region_name]['order']) < 3:
                self.get_logger().info(f"Not enough viewpoints for region '{region_name}', skipping optimization.")
                continue

            for cluster_name, cluster in clusters_dict.items():
                viewpoints.append(cluster['viewpoint']['position'])
            
            if self.use_2opt:
                self.get_logger().info(f"Using 2-opt optimization for region '{region_name}'")
                dist_matrix = self.dist_matrix(viewpoints)
                initial_path, initial_dist = self.nearest_neighbors_tsp(viewpoints)
                self.get_logger().info(f"Initial nearest neighbor distance: {initial_dist:.4f}")
                optimized_path, optimized_dist = self.local_search_2_opt(dist_matrix, [initial_path, initial_dist], recursive_seeding=-1, verbose=True)           
                self.get_logger().info(f"Optimized path for region '{region_name}' with total distance {optimized_dist}")
                improved = ((initial_dist - optimized_dist) / initial_dist) * 100
                self.get_logger().info(f"Improvement: {improved:.2f}%")
                path = optimized_path[:-1] if optimized_path[-1] == optimized_path[0] else optimized_path
            else:    
                path, total_distance = self.nearest_neighbors_tsp(viewpoints)
                self.get_logger().info(
                    f"Optimized path for region '{region_name}' with total distance {total_distance}")

            viewpoint_dict['regions'][region_name]['order'] = path

        return viewpoint_dict
    
    def local_search_2_opt(self, dist_matrix, viewpoint, recursive_seeding=-1, verbose=True):
        if recursive_seeding < 0:
            count = -2
        else:
            count = 0
            
        # Initialize vp_list with proper format [tour, distance]
        if isinstance(viewpoint, list) and len(viewpoint) == 2:
            vp_list = copy.deepcopy(viewpoint)
        else:
            dist = self.dist_calc(dist_matrix, viewpoint)
            vp_list = [viewpoint.copy(), dist]
            
        dist = vp_list[1] * 2 
        iteration = 0
        
        while count < recursive_seeding:
            if verbose:
                self.get_logger().info(f'2-opt Iteration {iteration}: Distance = {round(vp_list[1], 4)}')
                
            best_route = copy.deepcopy(vp_list)
            seed = copy.deepcopy(vp_list)
            
            # Try all possible 2-opt swaps
            for i in range(len(vp_list[0]) - 2):
                for j in range(i + 1, len(vp_list[0]) - 1):
                    # Reverse the segment between i and j (2-opt swap)
                    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
                    
                    # Ensure the tour is closed (last point = first point)
                    if len(best_route[0]) > 0:
                        best_route[0][-1] = best_route[0][0]
                    
                    # Calculate new distance
                    best_route[1] = self.dist_calc(dist_matrix, best_route)
                    
                    # Keep improvement if found
                    if vp_list[1] > best_route[1]:
                        vp_list = copy.deepcopy(best_route)
                        
                    # Reset to original state for next iteration
                    best_route = copy.deepcopy(seed)
            
            count += 1
            iteration += 1
            
            # Check for improvement and reset counter if found
            if dist > vp_list[1] and recursive_seeding < 0:
                dist = vp_list[1]
                count = -2
                recursive_seeding = -1
            elif vp_list[1] >= dist and recursive_seeding < 0:
                count = -1
                recursive_seeding = -2
                
        return vp_list[0], vp_list[1]

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
            elif param.name == 'use_2opt':
                self.use_2opt = param.value

        self.init_workspace()

        return SetParametersResult(successful=True)


def main():
    rclpy.init()

    traversal_node = ViewpointTraversalNode()
    # traversal_node.plan1()  # Call the plan1 method to execute the first plan
    rclpy.spin(traversal_node)


if __name__ == '__main__':
    main()
