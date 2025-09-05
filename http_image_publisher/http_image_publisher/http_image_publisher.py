#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np


class HttpImagePublisher(Node):
    def __init__(self):
        super().__init__('http_image_publisher')

        # Parameters
        self.declare_parameter('stream_url', 'http://example.com/stream.mjpg')
        # Creates camera/image_raw
        self.declare_parameter('base_topic', 'camera')
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('connection_timeout', 5.0)
        self.declare_parameter('use_opencv', True)

        # Camera calibration parameters
        self.declare_parameter('camera_name', 'http_camera')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter(
            'camera_matrix', [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0])
        self.declare_parameter('distortion_coefficients', [
                               0.0, 0.0, 0.0, 0.0, 0.0])

        self.stream_url = self.get_parameter('stream_url').value
        self.base_topic = self.get_parameter('base_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.connection_timeout = self.get_parameter(
            'connection_timeout').value
        self.use_opencv = self.get_parameter('use_opencv').value

        # Camera info parameters
        self.camera_name = self.get_parameter('camera_name').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.camera_matrix = self.get_parameter('camera_matrix').value
        self.distortion_coeffs = self.get_parameter(
            'distortion_coefficients').value

        # CV bridge
        self.bridge = CvBridge()

        # Standard ROS2 publishers - these will be the base for image_transport
        self.image_pub = self.create_publisher(
            Image, f'{self.base_topic}/image_raw', 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, f'{self.base_topic}/camera_info', 10)

        # Stream handling
        self.cap = None
        self.stream_session = None
        self.connection_failed_count = 0
        self.max_connection_failures = 5
        self.is_mjpeg_stream = False
        self.actual_width = self.image_width
        self.actual_height = self.image_height

        # Initialize camera info message
        self.setup_camera_info()

        # Determine stream type and initialize
        self.initialize_stream()

        # Create timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Publishing images from {self.stream_url} to {self.base_topic}/image_raw at {self.publish_rate} Hz')
        self.get_logger().info("To enable image transport, run:")
        self.get_logger().info(
            f"  ros2 run image_transport republish raw compressed --ros-args -r in:={self.base_topic}/image_raw -r out:={self.base_topic}/image_raw")
        self.get_logger().info(
            f"This will create {self.base_topic}/image_raw/compressed automatically")

    def setup_camera_info(self):
        """Setup camera info message with calibration data"""
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = self.frame_id

        # Image dimensions
        self.camera_info_msg.width = self.image_width
        self.camera_info_msg.height = self.image_height

        # Camera matrix (3x3)
        self.camera_info_msg.k = self.camera_matrix

        # Distortion coefficients
        self.camera_info_msg.d = self.distortion_coeffs
        self.camera_info_msg.distortion_model = "plumb_bob"

        # Rectification matrix (identity for monocular camera)
        self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix (3x4)
        self.camera_info_msg.p = [
            self.camera_matrix[0], 0.0, self.camera_matrix[2], 0.0,
            0.0, self.camera_matrix[4], self.camera_matrix[5], 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

    def initialize_stream(self):
        """Initialize the stream connection"""
        try:
            self.get_logger().info(
                f'Initializing connection to {self.stream_url}')

            # First, check what type of stream this is
            test_response = requests.head(
                self.stream_url, timeout=self.connection_timeout)
            content_type = test_response.headers.get('content-type', '')
            self.get_logger().info(f'Content-Type: {content_type}')

            if 'multipart' in content_type and self.use_opencv:
                # Try OpenCV first for MJPEG streams
                self.get_logger().info('Attempting OpenCV VideoCapture for MJPEG stream')
                self.cap = cv2.VideoCapture(self.stream_url)

                # Configure OpenCV
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, self.publish_rate)

                if self.cap.isOpened():
                    # Test read and get actual dimensions
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.actual_height, self.actual_width = frame.shape[:2]
                        self.update_camera_info_dimensions()
                        self.get_logger().info(
                            f'OpenCV VideoCapture successful - detected {self.actual_width}x{self.actual_height}')
                        self.is_mjpeg_stream = True
                        self.connection_failed_count = 0
                        return
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    self.cap = None

            # Fallback to requests method
            self.get_logger().info('Using requests method for image fetching')
            self.stream_session = requests.Session()
            self.is_mjpeg_stream = False
            self.connection_failed_count = 0

        except Exception as e:
            self.get_logger().error(f'Failed to initialize stream: {e}')
            self.connection_failed_count += 1
            if self.cap:
                self.cap.release()
                self.cap = None
            self.stream_session = None

    def update_camera_info_dimensions(self):
        """Update camera info with actual image dimensions"""
        if self.actual_width != self.image_width or self.actual_height != self.image_height:
            # Scale camera matrix for different resolution
            scale_x = self.actual_width / self.image_width
            scale_y = self.actual_height / self.image_height

            self.camera_info_msg.width = self.actual_width
            self.camera_info_msg.height = self.actual_height

            # Scale camera matrix
            scaled_matrix = self.camera_matrix.copy()
            scaled_matrix[0] *= scale_x  # fx
            scaled_matrix[2] *= scale_x  # cx
            scaled_matrix[4] *= scale_y  # fy
            scaled_matrix[5] *= scale_y  # cy

            self.camera_info_msg.k = scaled_matrix

            # Update projection matrix
            self.camera_info_msg.p = [
                scaled_matrix[0], 0.0, scaled_matrix[2], 0.0,
                0.0, scaled_matrix[4], scaled_matrix[5], 0.0,
                0.0, 0.0, 1.0, 0.0
            ]

    def timer_callback(self):
        """Timer callback to fetch and publish images"""
        if self.cap is None and self.stream_session is None:
            if self.connection_failed_count < self.max_connection_failures:
                self.initialize_stream()
            else:
                return

        try:
            frame = None

            if self.cap is not None:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.handle_connection_error()
                    return

            elif self.stream_session is not None:
                frame = self.get_single_image()
                if frame is not None:
                    # Update dimensions if first frame
                    if self.actual_width == self.image_width and self.actual_height == self.image_height:
                        self.actual_height, self.actual_width = frame.shape[:2]
                        self.update_camera_info_dimensions()

            if frame is not None:
                self.publish_image_and_info(frame)

        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')
            self.handle_connection_error()

    def get_single_image(self):
        """Fetch a single image from the URL using requests"""
        try:
            response = self.stream_session.get(
                self.stream_url,
                timeout=self.connection_timeout
            )
            response.raise_for_status()

            nparr = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame

        except Exception as e:
            self.get_logger().error(f'Error fetching single image: {e}')
            raise

    def publish_image_and_info(self, cv_image):
        """Publish both image and camera info with synchronized timestamps"""
        try:
            # Create timestamp
            timestamp = self.get_clock().now().to_msg()

            # Convert BGR to RGB (ROS standard)
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Create and publish image message
            ros_image = self.bridge.cv2_to_imgmsg(
                cv_image_rgb, encoding='rgb8')
            ros_image.header.stamp = timestamp
            ros_image.header.frame_id = self.frame_id
            self.image_pub.publish(ros_image)

            # Update and publish camera info
            self.camera_info_msg.header.stamp = timestamp
            self.camera_info_pub.publish(self.camera_info_msg)

        except Exception as e:
            self.get_logger().error(f'Image conversion/publishing error: {e}')

    def handle_connection_error(self):
        """Handle connection errors and attempt reconnection"""
        self.connection_failed_count += 1

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.stream_session:
            self.stream_session.close()
            self.stream_session = None

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        if self.cap:
            self.cap.release()
        if self.stream_session:
            self.stream_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HttpImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
