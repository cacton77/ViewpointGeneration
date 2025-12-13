std_msgs/Header header

#visual data
sensor_msgs/Image        depth_image         # latest depth frame (original encoding)
sensor_msgs/Image        depth_filtered_image # filtered depth (32FC1 meters)
float32 dmap_filter_min
float32 dmap_filter_max
sensor_msgs/PointCloud2  cloud_bbox          # points after bbox, in main_camera_frame
sensor_msgs/PointCloud2  cloud_crop          # points after cylindrical crop, in main_camera_frame

# Geometry and pose data
geometry_msgs/Point        centroid          # smoothed centroid (EMA)
geometry_msgs/Vector3      normal            # smoothed normal (+Z goal)
geometry_msgs/PoseStamped  goal_pose         # EOAT desired pose (its own header)

# Control data
geometry_msgs/Vector3  rotvec_error          # Z-axis alignment error 
geometry_msgs/Vector3  torque_cmd            # final torque command [Tx,Ty,Tz]
geometry_msgs/Vector3  force_cmd              # final force command [Fx,Fy,Fz]      # <<< NEW
geometry_msgs/Vector3  pos_error              # translational error (p_des_cf)      # <<< NEW
float32                k_p
float32                k_d
float32                k_i

# Frames
string  main_camera_frame
string  target_frame

# EMA & safety config
bool    ema_enable
float32 ema_tau_s
float32 no_target_timeout_s
bool    publish_zero_when_lost