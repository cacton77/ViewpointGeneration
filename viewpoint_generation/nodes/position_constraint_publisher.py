#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from moveit_msgs.msg import PositionConstraint, Constraints
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


class PositionConstraintPublisher(Node):
    def __init__(self):
        super().__init__('position_constraint_publisher')

        # Declare parameters with descriptors
        self.declare_custom_parameters()

        # Publishers
        self.constraint_pub = self.create_publisher(
            Constraints,
            'position_constraints',
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            'constraint_visualization',
            10
        )

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Timer to republish constraints periodically
        self.timer = self.create_timer(1.0, self.publish_constraints)

        # Store current constraint
        self.current_constraints = None

        # Initial constraint creation
        self.create_constraint()

        self.get_logger().info('Position Constraint Publisher started')
        self.get_logger().info('Publishing on topics:')
        self.get_logger().info('  - /position_constraints (moveit_msgs/Constraints)')
        self.get_logger().info('  - /constraint_visualization (visualization_msgs/Marker)')

    def declare_custom_parameters(self):
        """Declare all parameters with descriptions and default values"""

        # Frame parameters
        self.declare_parameter(
            'planning_frame',
            'ur_base_link',
            ParameterDescriptor(
                description='Reference frame for the constraint',
                type=ParameterType.PARAMETER_STRING
            )
        )

        self.declare_parameter(
            'end_effector_link',
            'eoat_camera_link',
            ParameterDescriptor(
                description='End effector link to constrain',
                type=ParameterType.PARAMETER_STRING
            )
        )

        # Box center position
        self.declare_parameter(
            'box_center_x',
            0.5,
            ParameterDescriptor(
                description='X coordinate of constraint box center (meters)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'box_center_y',
            0.0,
            ParameterDescriptor(
                description='Y coordinate of constraint box center (meters)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'box_center_z',
            1.0,
            ParameterDescriptor(
                description='Z coordinate of constraint box center (meters)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        # Box dimensions
        self.declare_parameter(
            'box_size_x',
            2.0,
            ParameterDescriptor(
                description='X dimension of constraint box (meters)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'box_size_y',
            2.0,
            ParameterDescriptor(
                description='Y dimension of constraint box (meters)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'box_size_z',
            2.0,
            ParameterDescriptor(
                description='Z dimension of constraint box (meters)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        # Constraint weight
        self.declare_parameter(
            'constraint_weight',
            1.0,
            ParameterDescriptor(
                description='Weight of the position constraint (0.0-1.0)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        # Visualization parameters
        self.declare_parameter(
            'visualization_alpha',
            0.3,
            ParameterDescriptor(
                description='Transparency of visualization marker (0.0-1.0)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'visualization_color_r',
            0.0,
            ParameterDescriptor(
                description='Red component of visualization color (0.0-1.0)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'visualization_color_g',
            1.0,
            ParameterDescriptor(
                description='Green component of visualization color (0.0-1.0)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

        self.declare_parameter(
            'visualization_color_b',
            0.0,
            ParameterDescriptor(
                description='Blue component of visualization color (0.0-1.0)',
                type=ParameterType.PARAMETER_DOUBLE
            )
        )

    def parameter_callback(self, params):
        """Callback when parameters are changed"""
        self.get_logger().info('Parameters updated, recreating constraint...')

        # Recreate constraint with new parameters
        self.create_constraint()

        # Immediately publish the updated constraint
        self.publish_constraints()

        return SetParametersResult(successful=True)

    def create_constraint(self):
        """Create PositionConstraint from current parameters"""

        # Get current parameter values
        planning_frame = self.get_parameter('planning_frame').value
        end_effector_link = self.get_parameter('end_effector_link').value

        box_center_x = self.get_parameter('box_center_x').value
        box_center_y = self.get_parameter('box_center_y').value
        box_center_z = self.get_parameter('box_center_z').value

        box_size_x = self.get_parameter('box_size_x').value
        box_size_y = self.get_parameter('box_size_y').value
        box_size_z = self.get_parameter('box_size_z').value

        constraint_weight = self.get_parameter('constraint_weight').value

        # Create the position constraint
        position_constraint = PositionConstraint()

        # Set header
        position_constraint.header = Header()
        position_constraint.header.frame_id = planning_frame
        position_constraint.header.stamp = self.get_clock().now().to_msg()

        # Set link name
        position_constraint.link_name = end_effector_link

        # Create box primitive
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [box_size_x, box_size_y, box_size_z]

        # Set box pose (center position)
        box_pose = Pose()
        box_pose.position.x = box_center_x
        box_pose.position.y = box_center_y
        box_pose.position.z = box_center_z
        box_pose.orientation.x = 0.0
        box_pose.orientation.y = 0.0
        box_pose.orientation.z = 0.0
        box_pose.orientation.w = 1.0

        # Add to constraint region
        position_constraint.constraint_region.primitives = [box]
        position_constraint.constraint_region.primitive_poses = [box_pose]

        # Set weight
        position_constraint.weight = constraint_weight

        # Create Constraints message
        constraints_msg = Constraints()
        constraints_msg.position_constraints = [position_constraint]

        # Store the constraint
        self.current_constraints = constraints_msg

        self.get_logger().info(f'Created constraint:')
        self.get_logger().info(f'  Frame: {planning_frame}')
        self.get_logger().info(f'  Link: {end_effector_link}')
        self.get_logger().info(
            f'  Center: ({box_center_x:.2f}, {box_center_y:.2f}, {box_center_z:.2f})')
        self.get_logger().info(
            f'  Size: ({box_size_x:.2f}, {box_size_y:.2f}, {box_size_z:.2f})')
        self.get_logger().info(f'  Weight: {constraint_weight:.2f}')

    def create_visualization_marker(self):
        """Create visualization marker for the constraint box"""

        # Get parameters
        planning_frame = self.get_parameter('planning_frame').value
        box_center_x = self.get_parameter('box_center_x').value
        box_center_y = self.get_parameter('box_center_y').value
        box_center_z = self.get_parameter('box_center_z').value
        box_size_x = self.get_parameter('box_size_x').value
        box_size_y = self.get_parameter('box_size_y').value
        box_size_z = self.get_parameter('box_size_z').value

        alpha = self.get_parameter('visualization_alpha').value
        color_r = self.get_parameter('visualization_color_r').value
        color_g = self.get_parameter('visualization_color_g').value
        color_b = self.get_parameter('visualization_color_b').value

        # Create marker
        marker = Marker()
        marker.header.frame_id = planning_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "position_constraints"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = box_center_x
        marker.pose.position.y = box_center_y
        marker.pose.position.z = box_center_z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale
        marker.scale.x = box_size_x
        marker.scale.y = box_size_y
        marker.scale.z = box_size_z

        # Set color
        marker.color.r = color_r
        marker.color.g = color_g
        marker.color.b = color_b
        marker.color.a = alpha

        return marker

    def publish_constraints(self):
        """Publish the current constraints and visualization"""

        if self.current_constraints is not None:
            # Update timestamp
            self.current_constraints.position_constraints[0].header.stamp = \
                self.get_clock().now().to_msg()

            # Publish constraint
            self.constraint_pub.publish(self.current_constraints)

            # Publish visualization
            marker = self.create_visualization_marker()
            self.marker_pub.publish(marker)

    def print_usage_info(self):
        """Print usage information"""
        self.get_logger().info('=== Position Constraint Publisher ===')
        self.get_logger().info('To modify constraints, use ros2 param set:')
        self.get_logger().info('')
        self.get_logger().info('Box position:')
        self.get_logger().info('  ros2 param set /position_constraint_publisher box_center_x 0.6')
        self.get_logger().info('  ros2 param set /position_constraint_publisher box_center_y 0.1')
        self.get_logger().info('  ros2 param set /position_constraint_publisher box_center_z 1.2')
        self.get_logger().info('')
        self.get_logger().info('Box size:')
        self.get_logger().info('  ros2 param set /position_constraint_publisher box_size_x 1.5')
        self.get_logger().info('  ros2 param set /position_constraint_publisher box_size_y 1.5')
        self.get_logger().info('  ros2 param set /position_constraint_publisher box_size_z 1.5')
        self.get_logger().info('')
        self.get_logger().info('Visualization:')
        self.get_logger().info(
            '  ros2 param set /position_constraint_publisher visualization_alpha 0.5')
        self.get_logger().info(
            '  ros2 param set /position_constraint_publisher visualization_color_r 1.0')
        self.get_logger().info('')
        self.get_logger().info('Add Marker display in RViz2 with topic: /constraint_visualization')


def main(args=None):
    rclpy.init(args=args)

    node = PositionConstraintPublisher()

    # Print usage info after a short delay
    node.create_timer(2.0, lambda: node.print_usage_info())

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
