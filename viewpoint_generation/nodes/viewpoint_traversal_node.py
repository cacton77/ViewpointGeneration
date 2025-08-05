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


def main():
    # rclpy.init()
    # traversal_node = ViewpointTraversalNode()
    # rclpy.spin(traversal_node)

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.pose_goal")

    # instantiate MoveItPy instance and get planning component
    robot = MoveItPy(node_name="moveit_py")
    disc_to_ur5e = robot.get_planning_component("disc_to_ur5e")
    logger.info("MoveItPy instance created")
    while True:
        try:
            # Create a RobotState object
            robot_state = RobotState(robot.get_robot_model())
            logger.info("RobotState object created successfully")
        except Exception as e:
            logger.error(f"Failed to create RobotState object: {e}")
        time.sleep(1)


if __name__ == '__main__':
    main()
