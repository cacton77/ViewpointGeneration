std_msgs/Header header

#visual data
#sensor_msgs/Image        depth_image         # latest depth frame (original encoding)
#sensor_msgs/Image        depth_filtered_image # filtered depth (32FC1 meters)
#float32 dmap_filter_min
#float32 dmap_filter_max
#sensor_msgs/PointCloud2  cloud_bbox          # points after bbox, in main_camera_frame
#sensor_msgs/PointCloud2  cloud_crop          # points after cylindrical crop, in main_camera_frame
sensor_msgs/PointCloud2   point_cloud        # final point cloud after all filtering

# Geometry and pose data
string                     normal_method     # method used to compute normal
geometry_msgs/Point        surface_centroid  # smoothed centroid (EMA)
geometry_msgs/Vector3      surface_normal    # smoothed normal (+Z goal)
geometry_msgs/PoseStamped  current_pose      # Current EOAT pose
geometry_msgs/TwistStamped current_twist     # Current EOAT twist
geometry_msgs/TwistStamped jacobian_twist
geometry_msgs/AccelStamped current_accel    # Current EOAT accel (linear m/s^2, angular rad/s^2)
geometry_msgs/PoseStamped  goal_pose         # EOAT desired pose (its own header)
geometry_msgs/AccelStamped commanded_accel

# Simulated / real IMU measurement
sensor_msgs/Imu imu

# Control data - separate pitch (X-axis) and yaw (Y-axis) control
# Pitch: rotation about camera X-axis
float32                pitch_error_raw        # raw pitch angle error before filtering (radians)
float32                dpitch_error_raw       # raw derivative of pitch error before filtering (radians/s)
float32                pitch_error            # pitch angle error (radians)
float32                dpitch_error           # derivative of pitch error (radians/s)
float32                ipitch_error           # integral of pitch error over time

# Yaw: rotation about camera Y-axis
float32                yaw_error_raw          # raw yaw angle error before filtering (radians)
float32                dyaw_error_raw         # raw derivative of yaw error before filtering (radians/s)
float32                yaw_error              # yaw angle error (radians)
float32                dyaw_error             # derivative of yaw error (radians/s)
float32                iyaw_error             # integral of yaw error over time

# Roll: rotation about camera Z-axis
float32                roll_error             # roll error around camera Z (radians)
float32                droll_error            # derivative of roll error (radians/s)
float32                iroll_error            # integral of roll error over time

# PID gains (same for pitch and yaw)
float32                k_p
float32                k_d
float32                k_i

# Torque commands
float32                pitch_torque_command   # torque command about X-axis (Nm)
float32                yaw_torque_command     # torque command about Y-axis (Nm)
float32                roll_torque_command    # torque command about Z-axis (Nm)

string kalman_model   # "PLANT" or "CA"
float32 kalman_r
float32 kalman_q_angle
float32 kalman_q_dangle
float32 kalman_q_ddangle    # CA-only; 0 in PLANT mode

# CA-only state (angular acceleration)
float32 ddpitch_error       # rad/s^2
float32 ddyaw_error         # rad/s^2


# Frames
geometry_msgs/TransformStamped camera_transform  # Transform from object_frame to camera_frame