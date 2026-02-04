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
geometry_msgs/PoseStamped  goal_pose         # EOAT desired pose (its own header)

# Control data
#geometry_msgs/Vector3  rotvec_error          # Z-axis alignment error 
#geometry_msgs/Vector3  torque_cmd            # final torque command [Tx,Ty,Tz]
#geometry_msgs/Vector3  cam_force_cmd         # Force sent to admittance control
#geometry_msgs/Vector3  cam_torque_cmd        # Torque sent to admittance control
float32                theta_error            # angle error between camera Z and target normal (radians)
float32                theta_error_filtered   # Kalman filtered angle error (radians)
float32                dtheta_error           # derivative of error (radians/s)
float32                dtheta_error_filtered  # Kalman filtered derivative of error (radians/s)
float32                itheta_error           # integral of error over time
float32                roll_error             # roll error around camera Z (radians)
float32                droll_error            # derivative of roll error (radians/s)
float32                iroll_error            # integral of roll error over time

float32                k_p_theta
float32                k_d_theta
float32                k_i_theta
float32                k_p_roll
float32                k_d_roll
float32                k_i_roll

geometry_msgs/Vector3  theta_axis    # final torque command axis [ax,ay,az]
float32                theta_torque_command  # torque command around theta axis
float32                roll_torque_command    # torque command around roll axis

# Frames
geometry_msgs/TransformStamped camera_transform  # Transform from object_frame to camera_frame