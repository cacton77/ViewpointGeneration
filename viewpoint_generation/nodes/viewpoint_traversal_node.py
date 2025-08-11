import time
import rclpy
# moveit python library
from rclpy.node import Node
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    PlanRequestParameters,
    MultiPipelinePlanRequestParameters,
)
from rclpy.logging import get_logger
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from geometry_msgs.msg import PoseStamped, Pose
import pprint
from std_srvs.srv import Trigger
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


class ViewpointTraversalNode(Node):

    def __init__(self):
        node_name = 'viewpoint_traversal'
        super().__init__(node_name)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('planning_group', 'disc_to_ur5e'),
                ('planner', 'chomp'),
                ('multiplanning', False),
            ]
        )

        self.planning_group = self.get_parameter(
            'planning_group').get_parameter_value().string_value
        self.planner = self.get_parameter(
            'planner').get_parameter_value().string_value
        self.multiplanning = self.get_parameter(
            'multiplanning').get_parameter_value().bool_value

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
        self.srv = self.create_service(
            MoveToPoseStamped,
            'viewpoint_traversal/move_to_pose_stamped',
            self.move_to_pose_stamped_callback
        )
        self.get_logger().info("Service 'move_to_pose_stamped' created successfully")

    def add_ground_plane(self):
        with self.planning_scene_monitor.read_write() as scene:
            collision_object = CollisionObject()
            collision_object.header.frame_id = "table_link"
            collision_object.id = "ground_plane"

            ground = SolidPrimitive()
            ground.type = SolidPrimitive.BOX
            ground.dimensions = [10.0, 10.0, 0.01]

            collision_object.primitives = [ground]
            collision_object.primitive_poses = [Pose()]
            collision_object.operation = CollisionObject.ADD

            scene.apply_collision_object(collision_object)
            scene.current_state.update()

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

        print("DEBUG: Goal state set to the requested pose")

        # Log the request
        self.get_logger().info(f"Received request: {request}")

        # Log the request
        self.get_logger().info(f"Received request: {request}")
        print("DEBUG: Request pose:", request.pose_goal)
        # Set the robot state to the current state
        robot_state.set_to_default_values()
        print("DEBUG: Robot state set to default values")
        # Plan and execute
        multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
            self.robot, ["ompl_rrtc"]
        )

        # self.single_plan_parameters.planner_id = self.planner

        success = self.plan_and_execute()

        # success = self.plan_and_execute(
        #     multi_plan_parameters=multi_pipeline_plan_request_params)
        print("DEBUG: Plan and execute called, success:", success)

        # Prepare the response
        response.success = success
        # response.message = "Motion completed successfully" if success else "Motion failed"

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
        else:
            self.get_logger().error("No trajectory found to execute")
            return False

        # Return success
        return True

    # def plan1(self):
    #     ###################################################################
    #     # Define the goal pose for the disc_to_ur5e group
    #     ###################################################################
    #     self.planning_component.set_start_state_to_current_state()
    #     pose_goal = PoseStamped()
    #     pose_goal.header.frame_id = "eoat_camera_link"
    #     pose_goal.pose.position.x = 0.05
    #     pose_goal.pose.position.y = 0.0
    #     pose_goal.pose.position.z = 0.05
    #     pose_goal.pose.orientation.w = 1.0
    #     pose_goal.pose.orientation.x = 0.0
    #     pose_goal.pose.orientation.y = 0.0
    #     pose_goal.pose.orientation.z = 0.0

    #     # Set the goal pose for the disc_to_ur5e group
    #     self.planning_component.set_goal_state(
    #         pose_stamped_msg=pose_goal, pose_link="eoat_camera_link")
    #     plan_result = self.planning_component.plan()
    #     if plan_result.error_code.val != 1:
    #         self.get_logger().error("Failed to plan trajectory")
    #         return False

    #     # Execute the planned trajectory
    #     self.get_logger().info("Executing plan")
    #     robot_trajectory = plan_result.trajectory
    #     self.robot.execute(robot_trajectory, controllers=[
    #         'inspection_cell_controller'])

    def test_service_callback(self, request, response):
        self.get_logger().info("Test service called")
        self.plan2()
        response.success = True
        response.message = "Test service executed successfully"
        return response

    def plan2(self):
        ###################################################################
        # Define the goal using joint states
        ###################################################################
        # Create RobotState objects
        robot_initial_state = RobotState(self.robot.get_robot_model())
        robot_state = RobotState(self.robot.get_robot_model())

        # # Set robot state to default values (zero angles)
        # robot_state.set_to_default_values()

        # Set robot state to custom values
        joint_positions = {
            "turntable_disc_joint": 0.0,  # radians
            "shoulder_pan_joint": 3.14,     # radians
            "shoulder_lift_joint": -1.8398,   # radians
            "elbow_joint": -1.8224,            # radians
            "wrist_1_joint": -1.0,         # radians
            "wrist_2_joint": 1.57,          # radians
            "wrist_3_joint": 1.57           # radians
        }

        # Get current joint positions and modify specific ones
        goal_positions = robot_state.joint_positions
        for joint_name, position in joint_positions.items():
            if joint_name in goal_positions:
                goal_positions[joint_name] = position
        robot_state.joint_positions = goal_positions

        # Set start state to current state
        self.planning_component.set_start_state_to_current_state()
        self.get_logger().info("Set goal state to the initialized robot state")

        # Set goal state using the robot state
        self.planning_component.set_goal_state(robot_state=robot_state)

        # Plan the trajectory
        plan_result = self.planning_component.plan()
        if plan_result:
            self.get_logger().info("Executing plan")
            self.robot.execute(plan_result.trajectory, controllers=[])
        else:
            self.get_logger().error("Planning failed")
            return False

        return True


def main():
    rclpy.init()

    traversal_node = ViewpointTraversalNode()
    # traversal_node.plan1()  # Call the plan1 method to execute the first plan
    rclpy.spin(traversal_node)

    # rclpy.init()
    # logger = get_logger("moveit_py.pose_goal")

    # # instantiate MoveItPy instance and get planning component
    # robot = MoveItPy(node_name="moveit_py")
    # disc_to_ur5e = robot.get_planning_component("disc_to_ur5e")
    # logger.info("MoveItPy instance created")
    # while True:
    #     try:
    #         # Create a RobotState object
    #         robot_state = RobotState(robot.get_robot_model())
    #         logger.info("RobotState object created successfully")
    #     except Exception as e:
    #         logger.error(f"Failed to create RobotState object: {e}")
    #     time.sleep(1)


if __name__ == '__main__':
    main()
