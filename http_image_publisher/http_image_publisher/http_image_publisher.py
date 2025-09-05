#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np
import time
import threading
from enum import Enum


class ConnectionState(Enum):
    DISCONNECTED = 1
    CONNECTING = 2
    CONNECTED = 3
    RECONNECTING = 4


class HttpImagePublisher(Node):
    def __init__(self):
        super().__init__('http_image_publisher')

        # Parameters
        self.declare_parameter('stream_url', 'http://example.com/stream.mjpg')
        self.declare_parameter('base_topic', 'camera')
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('connection_timeout', 5.0)
        self.declare_parameter('use_opencv', True)

        # Reconnection parameters
        self.declare_parameter('max_reconnection_attempts', 0)  # 0 = infinite
        self.declare_parameter('initial_reconnect_delay', 1.0)  # seconds
        self.declare_parameter('max_reconnect_delay', 30.0)  # seconds
        self.declare_parameter('backoff_multiplier', 2.0)
        self.declare_parameter('connection_check_interval', 1.0)  # seconds

        # Camera calibration parameters
        self.declare_parameter('camera_name', 'http_camera')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter(
            'camera_matrix', [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0])
        self.declare_parameter('distortion_coefficients', [
                               0.0, 0.0, 0.0, 0.0, 0.0])

        # Get parameters
        self.stream_url = self.get_parameter('stream_url').value
        self.base_topic = self.get_parameter('base_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.connection_timeout = self.get_parameter(
            'connection_timeout').value
        self.use_opencv = self.get_parameter('use_opencv').value

        # Reconnection parameters
        self.max_reconnection_attempts = self.get_parameter(
            'max_reconnection_attempts').value
        self.initial_reconnect_delay = self.get_parameter(
            'initial_reconnect_delay').value
        self.max_reconnect_delay = self.get_parameter(
            'max_reconnect_delay').value
        self.backoff_multiplier = self.get_parameter(
            'backoff_multiplier').value
        self.connection_check_interval = self.get_parameter(
            'connection_check_interval').value

        # Camera info parameters
        self.camera_name = self.get_parameter('camera_name').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.camera_matrix = self.get_parameter('camera_matrix').value
        self.distortion_coeffs = self.get_parameter(
            'distortion_coefficients').value

        # CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.image_pub = self.create_publisher(
            Image, f'{self.base_topic}/image_raw', 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, f'{self.base_topic}/camera_info', 10)

        # Connection state management
        self.connection_state = ConnectionState.DISCONNECTED
        self.connection_lock = threading.Lock()
        self.reconnection_attempts = 0
        self.current_reconnect_delay = self.initial_reconnect_delay
        self.last_successful_frame_time = time.time()
        self.reconnection_thread = None
        self.shutdown_requested = False

        # Stream handling
        self.cap = None
        self.stream_session = None
        self.is_mjpeg_stream = False
        self.actual_width = self.image_width
        self.actual_height = self.image_height

        # Frame statistics
        self.frames_received = 0
        self.last_stats_time = time.time()
        self.stats_timer = self.create_timer(
            10.0, self.print_stats)  # Print stats every 10 seconds

        # Initialize camera info message
        self.setup_camera_info()

        # Start connection monitoring
        self.connection_monitor_timer = self.create_timer(
            self.connection_check_interval, self.monitor_connection)

        # Create timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Start initial connection
        self.start_connection()

        self.get_logger().info(
            f'HTTP Image Publisher initialized for {self.stream_url}')
        self.get_logger().info(
            f'Publishing to {self.base_topic}/image_raw at {self.publish_rate} Hz')
        self.get_logger().info(f'Reconnection settings: max_attempts={self.max_reconnection_attempts}, '
                               f'initial_delay={self.initial_reconnect_delay}s, max_delay={self.max_reconnect_delay}s')

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

    def start_connection(self):
        """Start the initial connection attempt"""
        with self.connection_lock:
            if self.connection_state in [ConnectionState.CONNECTING, ConnectionState.RECONNECTING]:
                return

            self.connection_state = ConnectionState.CONNECTING
            self.get_logger().info('Starting connection to stream...')

        # Start connection in a separate thread to avoid blocking
        connection_thread = threading.Thread(target=self._attempt_connection)
        connection_thread.daemon = True
        connection_thread.start()

    def _attempt_connection(self):
        """Attempt to establish connection to the stream"""
        try:
            self.get_logger().info(f'Connecting to {self.stream_url}')

            # First, check what type of stream this is
            test_response = requests.head(
                self.stream_url, timeout=self.connection_timeout)
            content_type = test_response.headers.get('content-type', '')
            self.get_logger().info(f'Content-Type: {content_type}')

            success = False

            if 'multipart' in content_type and self.use_opencv:
                # Try OpenCV first for MJPEG streams
                self.get_logger().info('Attempting OpenCV VideoCapture for MJPEG stream')
                cap = cv2.VideoCapture(self.stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, self.publish_rate)

                if cap.isOpened():
                    # Test read and get actual dimensions
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        with self.connection_lock:
                            if self.cap:
                                self.cap.release()
                            self.cap = cap
                            self.actual_height, self.actual_width = frame.shape[:2]
                            self.update_camera_info_dimensions()
                            self.is_mjpeg_stream = True
                            success = True
                        self.get_logger().info(
                            f'OpenCV connection successful - {self.actual_width}x{self.actual_height}')
                    else:
                        cap.release()
                else:
                    cap.release()

            if not success:
                # Fallback to requests method
                self.get_logger().info('Using requests method for image fetching')
                session = requests.Session()

                # Test the connection with a quick request
                test_response = session.get(
                    self.stream_url, timeout=self.connection_timeout, stream=True)
                test_response.raise_for_status()
                test_response.close()

                with self.connection_lock:
                    if self.stream_session:
                        self.stream_session.close()
                    self.stream_session = session
                    self.is_mjpeg_stream = False
                    success = True
                self.get_logger().info('Requests connection successful')

            if success:
                with self.connection_lock:
                    self.connection_state = ConnectionState.CONNECTED
                    self.reconnection_attempts = 0
                    self.current_reconnect_delay = self.initial_reconnect_delay
                    self.last_successful_frame_time = time.time()
                self.get_logger().info('Stream connection established successfully')
            else:
                raise Exception(
                    "Failed to establish connection with any method")

        except Exception as e:
            self.get_logger().error(f'Connection attempt failed: {e}')
            with self.connection_lock:
                self.connection_state = ConnectionState.DISCONNECTED
            self._schedule_reconnection()

    def _schedule_reconnection(self):
        """Schedule a reconnection attempt"""
        if self.shutdown_requested:
            return

        with self.connection_lock:
            if self.max_reconnection_attempts > 0 and self.reconnection_attempts >= self.max_reconnection_attempts:
                self.get_logger().error(
                    f'Maximum reconnection attempts ({self.max_reconnection_attempts}) reached. Giving up.')
                return

            self.reconnection_attempts += 1
            self.connection_state = ConnectionState.RECONNECTING

        delay = min(self.current_reconnect_delay, self.max_reconnect_delay)
        self.get_logger().info(
            f'Scheduling reconnection attempt #{self.reconnection_attempts} in {delay:.1f} seconds')

        # Start reconnection timer
        if self.reconnection_thread and self.reconnection_thread.is_alive():
            return  # Already have a reconnection scheduled

        self.reconnection_thread = threading.Thread(
            target=self._reconnection_worker, args=(delay,))
        self.reconnection_thread.daemon = True
        self.reconnection_thread.start()

        # Increase delay for next attempt
        self.current_reconnect_delay *= self.backoff_multiplier

    def _reconnection_worker(self, delay):
        """Worker thread for handling reconnection delays"""
        time.sleep(delay)
        if not self.shutdown_requested:
            self._attempt_connection()

    def monitor_connection(self):
        """Monitor connection health and trigger reconnection if needed"""
        if self.shutdown_requested:
            return

        with self.connection_lock:
            current_state = self.connection_state

        if current_state == ConnectionState.CONNECTED:
            # Check if we've received frames recently
            time_since_last_frame = time.time() - self.last_successful_frame_time
            if time_since_last_frame > 5.0:  # No frames for 5 seconds
                self.get_logger().warn(
                    f'No frames received for {time_since_last_frame:.1f} seconds, triggering reconnection')
                self._handle_connection_loss()
        elif current_state == ConnectionState.DISCONNECTED:
            # Try to reconnect if we're not already trying
            self.start_connection()

    def _handle_connection_loss(self):
        """Handle loss of connection"""
        with self.connection_lock:
            self.connection_state = ConnectionState.DISCONNECTED

            # Clean up current connections
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.stream_session:
                self.stream_session.close()
                self.stream_session = None

        self.get_logger().warn('Connection lost, attempting to reconnect...')
        self._schedule_reconnection()

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
        with self.connection_lock:
            current_state = self.connection_state
            cap = self.cap
            session = self.stream_session

        if current_state != ConnectionState.CONNECTED or (cap is None and session is None):
            return

        try:
            frame = None

            if cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.get_logger().debug('OpenCV read failed')
                    self._handle_connection_loss()
                    return

                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)

            elif session is not None:
                frame = self.get_single_image(session)
                if frame is not None:
                    # Update dimensions if first frame
                    if self.actual_width == self.image_width and self.actual_height == self.image_height:
                        self.actual_height, self.actual_width = frame.shape[:2]
                        self.update_camera_info_dimensions()

            # Validate and publish frame
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                self.publish_image_and_info(frame)
                self.last_successful_frame_time = time.time()
                self.frames_received += 1
            else:
                self.get_logger().debug('Invalid frame received')
                self._handle_connection_loss()

        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')
            self._handle_connection_loss()

    def get_single_image(self, session):
        """Fetch a single image from the URL using requests"""
        try:
            response = session.get(
                self.stream_url, timeout=self.connection_timeout)
            response.raise_for_status()

            if not response.content:
                return None

            nparr = np.frombuffer(response.content, np.uint8)
            if nparr.size == 0:
                return None

            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame

        except Exception as e:
            self.get_logger().debug(f'Error fetching single image: {e}')
            return None

    def publish_image_and_info(self, cv_image):
        """Publish both image and camera info with synchronized timestamps"""
        try:
            # Validation
            if not isinstance(cv_image, np.ndarray) or cv_image.size == 0:
                return

            if len(cv_image.shape) != 3 or cv_image.shape[2] != 3:
                return

            # Ensure proper format
            if cv_image.dtype != np.uint8:
                cv_image = cv_image.astype(np.uint8)
            if not cv_image.flags.c_contiguous:
                cv_image = np.ascontiguousarray(cv_image)

            # Create timestamp
            timestamp = self.get_clock().now().to_msg()

            # Create and publish image message (keeping BGR format for ROS)
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            ros_image.header.stamp = timestamp
            ros_image.header.frame_id = self.frame_id
            self.image_pub.publish(ros_image)

            # Update and publish camera info
            self.camera_info_msg.header.stamp = timestamp
            self.camera_info_pub.publish(self.camera_info_msg)

        except Exception as e:
            self.get_logger().error(f'Image publishing error: {e}')

    def print_stats(self):
        """Print connection statistics"""
        current_time = time.time()
        time_elapsed = current_time - self.last_stats_time
        fps = self.frames_received / time_elapsed if time_elapsed > 0 else 0

        with self.connection_lock:
            state = self.connection_state

        self.get_logger().info(f'Stats: State={state.name}, FPS={fps:.1f}, '
                               f'Total_frames={self.frames_received}, '
                               f'Reconnect_attempts={self.reconnection_attempts}')

        self.frames_received = 0
        self.last_stats_time = current_time

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        self.shutdown_requested = True

        with self.connection_lock:
            if self.cap:
                self.cap.release()
            if self.stream_session:
                self.stream_session.close()

        # Wait for reconnection thread to finish
        if self.reconnection_thread and self.reconnection_thread.is_alive():
            self.reconnection_thread.join(timeout=1.0)

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
