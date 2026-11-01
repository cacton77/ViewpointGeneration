import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (EnvironmentVariable, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("cell", default_value="false"),
        DeclareLaunchArgument("sim", default_value="false"),
        DeclareLaunchArgument("dual_arm", default_value="false",
                              description="Add the second UR5e carrying the Zivid 2+ MR60 "
                              "(cell beta only)."),
        DeclareLaunchArgument("object", default_value="new.yaml",
                              description="Configuration file for viewpoint generation."),
        DeclareLaunchArgument("data_path", default_value="/data/ViewpointGenerationData",
                              description="Path to the data directory."),
        DeclareLaunchArgument("controller", default_value="xbox_controller.yaml",
                              description="Controller configuration file for teleoperation."),
        DeclareLaunchArgument("admittance_config_file", default_value="admittance_control_coupled_pendulum.yaml",
                              description="Configuration file for admittance control (Tier 3 coupled-pendulum plant)."),
        DeclareLaunchArgument("rviz_config",
                              default_value="/config/rviz/inspection_cell.rviz",
                              description="Absolute path to an RViz config file for the whole "
                              "inspection cell. Defaults to the file mounted into the container "
                              "at /config/rviz/inspection_cell.rviz."),
        DeclareLaunchArgument("launch_control", default_value="true",
                              description="Launch the ros2_control layer locally. Set false to "
                              "run control on a dedicated real-time host (bring it up there with "
                              "inspection_cell_control.launch.py); this machine then runs only "
                              "MoveIt + perception."),
        DeclareLaunchArgument("compute_threads", default_value="6",
                              description="Cap on CPU threads/worker processes for the "
                              "viewpoint_generation segmentation/clustering compute, so it "
                              "doesn't starve the real-time controller when they share a box."),
        DeclareLaunchArgument("headless_mode", default_value="false",
                              description="Run without the Open3D GUI (gui_node). Use on hosts "
                              "with no X display; an rqt GUI is started instead."),
        DeclareLaunchArgument("catalog", default_value="true",
                              description="Launch the PLM parts catalog node (plm_bridge_ros). It serves "
                              "whatever is already cached even without DX_* credentials, so it is "
                              "safe to leave on."),
        DeclareLaunchArgument("picker", default_value="true",
                              description="Launch the browser part picker UI alongside the catalog "
                              "node (requires catalog:=true)."),
        DeclareLaunchArgument("picker_port", default_value="5050",
                              description="Port the part picker UI listens on."),
        DeclareLaunchArgument("catalog_sync_interval", default_value="300",
                              description="Seconds between background incremental catalog syncs "
                              "(0 disables the timer)."),
        DeclareLaunchArgument("catalog_sync_on_startup", default_value="false",
                              description="Run a full catalog sync when the system starts. Off by "
                              "default because a full pass over a broad DX_BOOKMARK_SCOPE takes "
                              "minutes; the periodic timer and the picker's Sync button cover it."),
        DeclareLaunchArgument("plm_backends",
                              default_value=EnvironmentVariable("PLM_BACKENDS", default_value="dx3"),
                              description="Comma-separated PLM backends the catalog hosts."),
        DeclareLaunchArgument("catalog_mesh_units", default_value="mm",
                              description="Units used to load STEP files selected from the catalog."),
        DeclareLaunchArgument("foxglove_viz", default_value="true",
                              description="Publish each viewpoint-generation stage's results "
                              "(regions, clusters, viewpoints, paths) on /viewpoint_generation/viz/* "
                              "for Foxglove's 3D panel: the remote alternative to the Open3D GUI."),
        DeclareLaunchArgument("pi_camera_video", default_value="true",
                              description="Re-encode magpi's /pi_camera JPEG stream as H.264 on "
                              "/pi_camera/video/foxglove for remote Foxglove viewers. The JPEG "
                              "stream needs ~90 Mbit/s; this needs a few. Only encodes while "
                              "something is subscribed."),
        DeclareLaunchArgument("pi_camera_video_bit_rate", default_value="4000000",
                              description="Target bit rate (bit/s) of /pi_camera/video/foxglove."),
    ]

    cell_enabled = PythonExpression(
        ["'", LaunchConfiguration("cell"), "' != 'false'"])

    macro_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("http_image_publisher"),
                "launch",
                "http_image_publisher.launch.py"
            ])
        ]),
        launch_arguments={
            "stream_url": "http://192.168.0.92:5000/video_feed",
            "base_topic": "camera",
            "frame_id": "eoat_camera_link",
            "publish_rate": "30.0",
            "connection_timeout": "5.0"
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    control_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("inspection_cell_moveit_config"),
                "launch",
                "inspection_cell.launch.py"
            ])
        ]),
        launch_arguments={
            "cell": LaunchConfiguration("cell"),
            "use_fake_hardware": LaunchConfiguration("sim"),
            "rviz_config": LaunchConfiguration("rviz_config"),
            "launch_control": LaunchConfiguration("launch_control"),
            "dual_arm": LaunchConfiguration("dual_arm"),
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    task_planning_node = Node(
        package="viewpoint_generation",
        executable="task_planning_node",
        name="task_planning",
        output="screen",
        parameters=[
            ParameterFile(
                PathJoinSubstitution([
                    LaunchConfiguration("data_path"),
                    LaunchConfiguration("object")
                ]),
                allow_substs=True
            )
        ],
    )

    viewpoint_traversal_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("viewpoint_generation"),
                "launch",
                "viewpoint_traversal.launch.py"
            ])
        ]),
        launch_arguments={
            "cell": LaunchConfiguration("cell"),
            "use_fake_hardware": LaunchConfiguration("sim"),
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    admittance_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("inspection_control"),
                "launch",
                "admittance_control.launch.py"
            ])
        ]),
        launch_arguments={
            "controller_config_file": LaunchConfiguration("controller"),
            "admittance_config_file": LaunchConfiguration("admittance_config_file"),
            "dual_arm": LaunchConfiguration("dual_arm"),
        }.items(),
        condition=IfCondition(cell_enabled)
    )

    # Foxglove bridge: serves all topics over a WebSocket (ws://<host>:8765)
    # for Foxglove Studio. Always-on (not gated on cell) so visualization is
    # available regardless of cell hardware. Host networking makes 8765
    # directly reachable on the host.
    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        # output="screen",
        parameters=[{
            "port": 8765,
            "address": "0.0.0.0",
            # The bridge has no authentication and is reachable from the whole
            # tailnet, and its defaults let any client publish any topic, call
            # any service and set any parameter -- including the UR arms'
            # controllers. So: no parameter writes, publishing only on the
            # topics listed here, and only the services listed here. Add an
            # entry when a Foxglove panel genuinely needs it.
            "capabilities": ["clientPublish", "services", "connectionGraph", "assets"],
            # The default is '.*', i.e. every service on the graph, the
            # controller_manager's and the UR drivers' included -- so this
            # list is what makes the services capability safe to enable.
            "service_whitelist": [
                # PLM catalog (plm_bridge): browse, select, sync, plan check.
                # Not record_plan/record_run/upload_*, which write to the PLM.
                "^/catalog/(list_backends|list_parts|select_part|sync_now|ensure_plan)$",
                # Viewpoint-generation pipeline stages (std_srvs/Trigger).
                "^/viewpoint_generation/(segment_regions|fov_clustering|viewpoint_projection|optimize_traversal)$",
            ],
            "client_topic_whitelist": [
                # LED ring requests, handled by lens_tuner on magpi (it stays
                # the only publisher of /camera_head/led_ring/command).
                "^/camera_head/led_ring/request$",
                # Focus-lens requests, same arrangement. A velocity hold sent
                # here lapses in lens_tuner 0.6 s after the panel stops
                # refreshing it, so a dropped connection stops the lens.
                "^/camera_head/lens/request$",
                # Region/cluster selection for the viewpoint-generation viz.
                "^/viewpoint_generation/viz/select$",
            ],
        }],
    )

    # Remote-viewing copy of magpi's camera. /pi_camera/image_raw/compressed
    # (2028x1520 JPEG @ 30 fps, ~90 Mbit/s) is more than a home connection
    # over Tailscale can carry, so Foxglove stutters and lags. This decodes it
    # here and re-encodes H.264 (foxglove_msgs/CompressedVideo), which Foxglove
    # plays natively at a few Mbit/s. zerolatency = no B-frames (Foxglove
    # can't decode them); gop_size 30 = a keyframe every second, so a new
    # viewer gets a picture within 1 s. Lazy: idle when nothing subscribes.
    try:
        get_package_share_directory("foxglove_compressed_video_transport")
        have_video_transport = True
    except PackageNotFoundError:
        have_video_transport = False
    if have_video_transport:
        pi_camera_video_node = Node(
            package="image_transport",
            executable="republish",
            name="pi_camera_video_republisher",
            parameters=[{
                "in_transport": "compressed",
                "out_transport": "foxglove",
                "out.foxglove.encoder": "libx264",
                "out.foxglove.preset": "ultrafast",
                "out.foxglove.tune": "zerolatency",
                "out.foxglove.gop_size": 30,
                "out.foxglove.bit_rate": ParameterValue(
                    LaunchConfiguration("pi_camera_video_bit_rate"), value_type=int),
                "out.foxglove.qmax": 51,
                "out.foxglove.pixel_format": "yuv420p",
                # Subscribe BEST_EFFORT. A RELIABLE reader on this host makes
                # magpi's camera writer wait on retransmits of ~370 KB JPEG
                # frames fragmented over UDP; one bad loss stalls the camera
                # for everyone (seen 2026-10-31: /pi_camera stopped publishing
                # even on magpi). Same reason the D405/pi_camera publishers are
                # best-effort in inspection_eoat's bringup. Depth 1: a late
                # frame is worthless to a live view.
                "qos_overrides./pi_camera/image_raw/compressed.subscription.reliability":
                    "best_effort",
                "qos_overrides./pi_camera/image_raw/compressed.subscription.history":
                    "keep_last",
                "qos_overrides./pi_camera/image_raw/compressed.subscription.depth": 1,
            }],
            remappings=[
                ("in/compressed", "/pi_camera/image_raw/compressed"),
                ("out", "/pi_camera/video"),
                ("out/foxglove", "/pi_camera/video/foxglove"),
            ],
            condition=IfCondition(LaunchConfiguration("pi_camera_video")),
        )
    else:
        pi_camera_video_node = LogInfo(
            msg="pi_camera_video:=true, but foxglove_compressed_video_transport is not "
                "installed, so /pi_camera/video/foxglove is not published. Rebuild the "
                "image (see docker/Dockerfile.jazzy), or pass pi_camera_video:=false.",
            condition=IfCondition(LaunchConfiguration("pi_camera_video")))

    viewpoint_generation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("viewpoint_generation"),
                "launch",
                "viewpoint_generation.launch.py"
            ])
        ]),
        launch_arguments={
            "data_path": LaunchConfiguration("data_path"),
            "object": LaunchConfiguration("object"),
            "compute_threads": LaunchConfiguration("compute_threads"),
            "headless_mode": LaunchConfiguration("headless_mode"),
        }.items()
    )

    # PLM parts catalog + picker UI (the plm_bridge repository). Always-on
    # (not gated on cell), like the Foxglove bridge: browsing the catalog and
    # caching STEP files is useful with or without cell hardware attached.
    try:
        plm_bridge_share = get_package_share_directory("plm_bridge_ros")
    except PackageNotFoundError:
        plm_bridge_share = None
    if plm_bridge_share is not None:
        catalog_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(plm_bridge_share, "launch", "plm_bridge.launch.py")),
            launch_arguments={
                "sync_interval": LaunchConfiguration("catalog_sync_interval"),
                "sync_on_startup": LaunchConfiguration("catalog_sync_on_startup"),
                "mesh_units": LaunchConfiguration("catalog_mesh_units"),
                "picker": LaunchConfiguration("picker"),
                "picker_port": LaunchConfiguration("picker_port"),
                "plm_backends": LaunchConfiguration("plm_backends"),
                # This package's node describes plan files the catalog finds.
                "describe_plans_service": "viewpoint_generation/describe_plans",
            }.items(),
            condition=IfCondition(LaunchConfiguration("catalog"))
        )
    else:
        catalog_launch = LogInfo(
            msg="catalog:=true, but the plm_bridge_ros package is not installed, so the "
                "parts catalog and picker are not running. Add the plm_bridge repository "
                "to the workspace (see the README), or pass catalog:=false.",
            condition=IfCondition(LaunchConfiguration("catalog")))

    # Foxglove view of the viewpoint-generation results (see
    # nodes/foxglove_viz_node.py). Follows the pipeline node's parameters the
    # way the Open3D GUI does, so it never sits in the pipeline's path.
    foxglove_viz_node = Node(
        package="viewpoint_generation",
        executable="foxglove_viz_node",
        name="viewpoint_generation_viz",
        condition=IfCondition(LaunchConfiguration("foxglove_viz")),
    )

    return LaunchDescription(declared_arguments + [
        control_moveit_launch,
        viewpoint_generation_launch,
        viewpoint_traversal_launch,
        task_planning_node,
        admittance_control_launch,
        foxglove_bridge_node,
        pi_camera_video_node,
        foxglove_viz_node,
        catalog_launch,
    ])
