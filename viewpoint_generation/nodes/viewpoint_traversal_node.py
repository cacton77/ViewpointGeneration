import time
import rclpy
# moveit python library
from rclpy.node import Node
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from rclpy.logging import get_logger
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from geometry_msgs.msg import PoseStamped
import pprint


class ViewpointTraversalNode(Node):

    def __init__(self):
        node_name = 'viewpoint_traversal'
        super().__init__(node_name)
        self.robot = MoveItPy(node_name='moveit_py')

        try:
            self.get_logger().info("Initializing MoveItPy")
            print("Initializing MoveItPy")
            self.planning_component = self.robot.get_planning_component(
                'disc_to_ur5e')
            planning_scene_monitor = self.robot.get_planning_scene_monitor()
            with planning_scene_monitor.read_write() as scene:
                scene.current_state.update()
        except Exception as e:
            self.get_logger().error(
                f"Failed to get planning component: {e}")
            rclpy.shutdown()
            return

        # Debug: see what parameters are being passed
        moveit_params = self.get_parameters_by_prefix('')
        pprint.pprint(dict(moveit_params))

        print("Planning component initialized successfully")
        # Create a service to move to a specific pose
        self.srv = self.create_service(
            MoveToPoseStamped,
            'viewpoint_traversal/move_to_pose_stamped',
            self.move_to_pose_stamped_callback
        )
        print("Service 'move_to_pose_stamped' created successfully")

    def move_to_pose_stamped_callback(self, request, response):
        # Create a RobotState object
        robot_state = RobotState(self.robot.get_robot_model())
        print("DEBUG: RobotState object created successfully")
        # Set the pose from the request
        self.robot.planning_component.set_goal_state(
            pose_stamped_msg=request.pose)

        print("DEBUG: Goal state set to the requested pose")

        # Log the request
        self.get_logger().info(f"Received request: {request}")

        # Log the request
        self.get_logger().info(f"Received request: {request}")
        print("DEBUG: Request pose:", request.pose)
        # Set the robot state to the current state
        robot_state.set_to_default_values()
        print("DEBUG: Robot state set to default values")
        # Plan and execute
        success = self.robot.plan_and_execute()
        print("DEBUG: Plan and execute called, success:", success)

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
        if single_plan_parameters is not None:
            plan_result = self.planning_component.plan(
                robot_state=robot_state,
                single_plan_parameters=single_plan_parameters
            )
            if plan_result.error_code.val != 1:
                self.get_logger().error("Failed to plan trajectory")
                return False
        elif multi_plan_parameters is not None:
            plan_result = self.planning_component.plan(
                robot_state=robot_state,
                multi_plan_parameters=multi_plan_parameters
            )
            if plan_result.error_code.val != 1:
                self.get_logger().error("Failed to plan trajectory")
                return False
        else:
            plan_result = self.planning_component.plan()

        # Execute the Planned Trajectory
        if plan_result:
            self.get_logger().info("Executing plan")
            robot_trajectory = plan_result.trajectory
            # Check if the controller name is correct
            self.execute(robot_trajectory, controllers=[
                'inspection_cell_controller'])
        else:
            self.get_logger().error("No trajectory found to execute")
            return False

        # Return success
        return True

    def plan1(self):
        ###################################################################
        # Define the goal pose for the disc_to_ur5e group
        ###################################################################
        self.planning_component.set_start_state_to_current_state()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "eoat_camera_link"
        goal_pose.pose.position.x = 0.05
        goal_pose.pose.position.y = 0.0
        goal_pose.pose.position.z = 0.05
        goal_pose.pose.orientation.w = 1.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0

        # Set the goal pose for the disc_to_ur5e group
        self.planning_component.set_goal_state(
            pose_stamped_msg=goal_pose, pose_link="eoat_camera_link")
        plan_result = self.planning_component.plan()
        if plan_result.error_code.val != 1:
            self.get_logger().error("Failed to plan trajectory")
            return False

        # Execute the planned trajectory
        self.get_logger().info("Executing plan")
        robot_trajectory = plan_result.trajectory
        self.robot.execute(robot_trajectory, controllers=[
            'inspection_cell_controller'])

    def plan2(self):
        ###################################################################
        # Define the goal using joint states
        ###################################################################
        # Create RobotState objects
        robot_initial_state = RobotState(self.robot.get_robot_model())
        robot_state = RobotState(self.robot.get_robot_model())

        # Set robot state to default values (zero angles)
        robot_state.set_to_default_values()

        # Set start state to current state
        self.planning_component.set_start_state_to_current_state()
        self.get_logger().info("Set goal state to the initialized robot state")

        # Set goal state using the robot state
        self.planning_component.set_goal_state(robot_state=robot_state)

        # Plan the trajectory
        plan_result = self.planning_component.plan()
        if plan_result:
            self.get_logger().info("Executing plan")
            self.robot.execute(plan_result.trajectory, controllers=[
                               'inspection_cell_controller'])
        else:
            self.get_logger().error("Planning failed")
            return False

        # Then go back to the initial state
        self.planning_component.set_start_state_to_current_state()
        self.planning_component.set_goal_state(robot_state=robot_initial_state)
        plan_result = self.planning_component.plan()
        if plan_result:
            self.get_logger().info("Moving arm back to initial joints goal (RobotState goal)...")
            self.robot.execute(plan_result.trajectory, controllers=[
                               'inspection_cell_controller'])
        else:
            self.get_logger().error("Planning back to initial state failed")
            return False

        return True


# def plan2(self):
#     ###################################################################
#     # Define the goal pose for the disc_to_ur5e group
#     ###################################################################

#     robot_initial_state = RobotState(self.robot.get_robot_model())
#     robot_state = RobotState(self.robot_model)
#     robot_state.set_to_default_values()
#     self.robot.set_start_state_to_current_state()
#     self.logger.info("Setting start state to current state")
#     self.robot.set_goal_state(robot_state=robot_state)
#     # Move a specific joint to a desired position
#     joint_name = "joint_1"
#     joint_position = 0.5


def main():
    rclpy.init()

    traversal_node = ViewpointTraversalNode()
    # traversal_node.plan1()  # Call the plan1 method to execute the first plan
    traversal_node.plan2()  # Call the plan2 method to execute the second plan
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
