import rclpy
# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from rclpy.logging import get_logger
from viewpoint_generation_interfaces.srv import MoveToPoseStamped
from geometry_msgs.msg import PoseStamped


class ViewpointTraversalNode(MoveItPy):

    def __init__(self):
        node_name = 'viewpoint_traversal'
        super().__init__(node_name)
        try:
            self.planning_component = self.get_planning_component(
                'disc_to_ur5e')
        except Exception as e:
            get_logger(node_name).error(
                f"Failed to get planning component: {e}")
            rclpy.shutdown()
            return

    # Create a service to move to a specific pose
        self.srv = self.create_service(
            MoveToPoseStamped,
            node_name + 'move_to_pose_stamped',
            self.move_to_pose_stamped_callback
        )

    def move_to_pose_stamped_callback(self, request, response):
        # Create a RobotState object
        robot_state = RobotState(self.robot_model)

        # Set the pose from the request
        pose_goal = request

        # Log the request
        self.get_logger().info(f"Received request: {request}")

        # Set the robot state to the current state
        robot_state.set_to_default_values()

        # Plan to the specified pose
        plan_request = MultiPipelinePlanRequestParameters(
            robot_state=robot_state,
            target_pose=pose_goal,
            group_name='disc_to_ur5e'
        )

        self.planning_component.set_goal_state(pose_stamped_msg=pose_goal)

        self.plan_and_execute()

    # Function for planning and executing a trajectories
    def plan_and_execute(self, single_plan_parameters=None, multi_plan_parameters=None):
        # Check if the planning component is valid
        if not self.planning_component:
            self.get_logger().error("Planning component is not valid")
            return False

        # Create a RobotState object
        robot_state = RobotState(self.robot_model)

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
    rclpy.init()
    traversal_node = ViewpointTraversalNode()
    rclpy.spin(traversal_node)


if __name__ == '__main__':
    main()
