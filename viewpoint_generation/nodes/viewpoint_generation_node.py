import os
import time
import rclpy
import rclpy.node
import numpy as np
import open3d as o3d
from rclpy.duration import Duration
from rclpy.action import ActionServer
from viewpoint_generation.viewpoint_generation import ViewpointGeneration
from rcl_interfaces.msg import (
    SetParametersResult, ParameterDescriptor,
    FloatingPointRange, IntegerRange,
)
from ament_index_python.packages import get_package_prefix
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_srvs.srv import Trigger
# from viewpoint_generation_interfaces.action import ViewpointGeneration

from std_msgs.msg import Bool, ColorRGBA
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, Point
from visualization_msgs.msg import Marker
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject, ObjectColor
from viewpoint_generation_interfaces.srv import ImportCadModel, OptimizeViewpointTraversal


class ViewpointGenerationNode(rclpy.node.Node):

    block_next_param_callback = False
    initialized = False
    mesh = None
    bbox_marker = None
    pvolume_marker = None
    viewpoints_bbox_marker = None

    def __init__(self):
        node_name = 'viewpoint_generation'
        super().__init__(node_name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model.mesh.file', ''),
                ('model.mesh.units', 'm'),
                ('model.point_cloud.file', ''),
                ('model.point_cloud.units', 'm'),
                ('model.point_cloud.sampling.number_of_points', 100000),
                ('results.file', ''),
                ('model.point_cloud.curvature.file', ''),
                ('viewpoints.traversal', ''),
                ('settings.data_path', '/tmp'),
                ('settings.planning_volume_opacity', 0.0),
            ]
        )

        # Bounding Box Marker Publisher
        self.bbox_marker_publisher = self.create_publisher(
            Marker, f'{node_name}/bounding_box_marker', 10)
        # Planning Volume Marker Publisher
        self.create_planning_volume_mesh()

        # Create planning scene publisher
        self.planning_scene_diff_publisher = self.create_publisher(
            PlanningScene, '/planning_scene', 10)
        self.get_logger().info('Planning scene publisher created.')
        # Update planning scene timer
        self.create_timer(1.0, self.update_planning_scene)

        # --- ROS Services and Actions ---

        services_cb_group = MutuallyExclusiveCallbackGroup()
        # Import CAD Model Service
        self.create_service(ImportCadModel, node_name + '/import_cad_model',
                            self.import_cad_model_callback, callback_group=services_cb_group)
        # Sample PCD Service
        self.create_service(Trigger, node_name + '/sample_point_cloud',
                            self.sample_point_cloud_callback, callback_group=services_cb_group)
        # Estimate Curvature Service
        self.create_service(Trigger, node_name + '/estimate_curvature',
                            self.estimate_curvature_callback, callback_group=services_cb_group)
        # Region Growth Service
        self.create_service(Trigger, node_name + '/region_growth',
                            self.region_growth_callback, callback_group=services_cb_group)
        # FOV Clustering Service
        self.create_service(Trigger, node_name + '/fov_clustering',
                            self.fov_clustering_callback, callback_group=services_cb_group)
        # Viewpoint Projection Service
        self.create_service(Trigger, node_name + '/viewpoint_projection',
                            self.viewpoint_projection_callback, callback_group=services_cb_group)
        # Optimize Traversal Service
        self.create_service(Trigger, node_name +
                            '/optimize_traversal', self.optimize_traversal, callback_group=services_cb_group)

        # Connect to viewpoint traversal service
        viewpoint_traversal_node_name = 'viewpoint_traversal'

        self.optimize_traversal_client = self.create_client(
            OptimizeViewpointTraversal, f'{viewpoint_traversal_node_name}/optimize_traversal', callback_group=services_cb_group)

        # Viewpoint Generation Helpers
        self.viewpoint_generation = ViewpointGeneration()

        # Declare RegionGrowingConfig and FOVClusteringConfig fields as ROS
        # parameters with full ParameterDescriptors (range, description,
        # additional_constraints).  Adding a field to either config dataclass
        # and its to_dict() is sufficient — no changes needed here or in the
        # parameter_callback.  Both configs are routed via set_algorithm_param.

        def _make_descriptor(field_info):
            desc = ParameterDescriptor()
            desc.description = field_info.get('description', '')
            desc.additional_constraints = field_info.get('control', '')
            range_val = field_info.get('range')
            if range_val is not None:
                if field_info['type'] == 'float':
                    fp = FloatingPointRange()
                    fp.from_value = float(range_val[0])
                    fp.to_value = float(range_val[1])
                    fp.step = 0.0
                    desc.floating_point_range = [fp]
                elif field_info['type'] == 'integer':
                    ir = IntegerRange()
                    ir.from_value = int(range_val[0])
                    ir.to_value = int(range_val[1])
                    ir.step = 0
                    desc.integer_range = [ir]
            return desc

        def _auto_declare_parameters(prefix, config_dict, excluded):
            """Declare params and apply launch-file overrides for one config."""
            params = [
                (f'{prefix}{field_name}', field_info['value'],
                 _make_descriptor(field_info))
                for field_name, field_info in config_dict.items()
                if field_name not in excluded
            ]
            if params:
                self.declare_parameters(namespace='', parameters=params)
                for ros_param_name, _, _ in params:
                    field_name = ros_param_name[len(prefix):]
                    value = self.get_parameter(ros_param_name).value
                    success, message = self.viewpoint_generation.set_algorithm_param(
                        field_name, value)
                    if not success:
                        self.get_logger().error(message)

        _auto_declare_parameters(
            prefix='regions.region_growth.',
            config_dict=self.viewpoint_generation.region_growing_config.to_dict(),
            excluded=set(),
        )
        _auto_declare_parameters(
            prefix='regions.fov_clustering.',
            config_dict=self.viewpoint_generation.fc_config.to_dict(),
            excluded=set(),
        )
        _auto_declare_parameters(
            prefix='viewpoints.projection.',
            config_dict=self.viewpoint_generation.vp_config.to_dict(),
            excluded=set(),
        )

        self.set_data_path(self.get_parameter(
            'settings.data_path').get_parameter_value().string_value)
        # Load results FIRST so that set_mesh_file / set_point_cloud_file can
        # detect that those files are already recorded in self.results and skip
        # resetting the regions/clusters.
        self.set_results_file(self.get_parameter(
            'results.file').get_parameter_value().string_value)
        self.set_mesh_file(self.get_parameter('model.mesh.file').get_parameter_value().string_value,
                           self.get_parameter('model.mesh.units').get_parameter_value().string_value)
        self.set_point_cloud_file(self.get_parameter('model.point_cloud.file').get_parameter_value().string_value,
                                  self.get_parameter('model.point_cloud.units').get_parameter_value().string_value)
        self.set_sampling_number_of_points(
            self.get_parameter(
                'model.point_cloud.sampling.number_of_points').get_parameter_value().integer_value)
        self.set_curvature_file(self.get_parameter(
            'model.point_cloud.curvature.file').get_parameter_value().string_value)
        self.planning_volume_opacity = self.get_parameter(
            'settings.planning_volume_opacity').get_parameter_value().double_value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.block_next_param_callback = False
        self.initialized = True

    def set_parameters_blocked(self, params):
        for param in params:
            print(param.name)
            self.block_next_param_callback = True
            self.set_parameters([param])

    def set_data_path(self, data_path):
        """
        Helper function to set the data path for the partitioner.
        :param data_path: The path to the data directory.
        """
        data_path = os.path.expandvars(data_path)
        if data_path.startswith('package://'):
            package_name, relative_path = data_path.split(
                'package://')[1].split('/', 1)
            package_path = get_package_prefix(package_name)
            data_path = os.path.join(
                package_path, 'share', package_name, relative_path)

        # If path doesn't exist, create it
        if not os.path.exists(data_path):
            data_path = '/tmp'

        self.data_path = data_path

    def import_cad_model_callback(self, request, response):
        pass

    def set_mesh_file(self, mesh_file, mesh_units):
        """
        Helper function to set the triangle mesh file for the partitioner.
        :param mesh_file: The path to the triangle mesh file.
        :return: None
        """
        # If mesh_file exists, copy it to the data path
        if not mesh_file == '' and not os.path.exists(mesh_file):
            mesh_file = os.path.join(self.data_path, mesh_file)

        # If file doesn't exist, look in data path
        if not os.path.exists(mesh_file):
            mesh_file = os.path.join(self.data_path, mesh_file)

        success, message = self.viewpoint_generation.set_mesh_file(
            mesh_file, mesh_units)


        if not success:
            mesh_file_param = rclpy.parameter.Parameter(
                'model.mesh.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            # self.block_next_param_callback = True
            # self.set_parameters([mesh_file_param])
            self.get_logger().warn(message)
            return False
        else:
            self.get_logger().info('Import mesh completed successfully.')
            mesh_file_param = rclpy.parameter.Parameter(
                'model.mesh.file',
                rclpy.Parameter.Type.STRING,
                mesh_file
            )
            mesh_units_param = rclpy.parameter.Parameter(
                'model.mesh.units',
                rclpy.Parameter.Type.STRING,
                mesh_units
            )
            params = [mesh_file_param, mesh_units_param]
            # results_file is None only when set_mesh_file reset self.results
            # (i.e. a genuinely different mesh was loaded). In that case clear
            # the results.file parameter so stale results are not shown.
            if self.viewpoint_generation.results_file is None:
                params.append(rclpy.parameter.Parameter(
                    'results.file', rclpy.Parameter.Type.STRING, ''))
            self.set_parameters_blocked(params)

            if self.initialized:
                self.set_point_cloud_file(
                    point_cloud_file='', point_cloud_units='mm')

            # Create Bounding Box 

            min_x, min_y, min_z, max_x, max_y, max_z = self.viewpoint_generation.get_mesh_bounds()

            if min_z < 0.0:
                self.get_logger().warn(
                    f'Minimum Z value of the mesh is below 0. This may cause issues with registration and planning.'
                )
                min_z = 0.0

            # Add padding to the bounding box. Keep min_z at 0.
            padding = 0.02
            min_x = min_x - padding
            min_y = min_y - padding
            min_z = max(0.01, min_z - padding)
            max_x = max_x + padding
            max_y = max_y + padding
            max_z = max_z + padding

            cx = 0.5 * (max_x + min_x)
            cy = 0.5 * (max_y + min_y)
            cz = 0.5 * (max_z + min_z)

            sx = max(1e-6, max_x - min_x)
            sy = max(1e-6, max_y - min_y)
            sz = max(1e-6, max_z - min_z)

            # Create a Marker message for the bounding box
            self.bbox_marker = Marker()
            self.bbox_marker.header.frame_id = 'object_frame'
            self.bbox_marker.header.stamp = self.get_clock().now().to_msg()
            self.bbox_marker.ns = 'bbox'
            self.bbox_marker.id = 0
            self.bbox_marker.type = Marker.CUBE
            self.bbox_marker.action = Marker.ADD
            self.bbox_marker.pose.position.x = cx
            self.bbox_marker.pose.position.y = cy
            self.bbox_marker.pose.position.z = cz
            self.bbox_marker.pose.orientation.w = 1.0
            self.bbox_marker.scale.x = sx
            self.bbox_marker.scale.y = sy
            self.bbox_marker.scale.z = sz
            # semi-transparent cyan
            self.bbox_marker.color.r = 0.0
            self.bbox_marker.color.g = 1.0
            self.bbox_marker.color.b = 1.0
            self.bbox_marker.color.a = 0.25
            # short lifetime; republish each frame
            self.bbox_marker.lifetime = Duration(seconds=1.0).to_msg()
            # self.bbox_marker.header.frame_id = 'object_frame'

            # Create a mesh from the file
            self.mesh = Mesh()
            self.mesh.triangles = []
            self.mesh.vertices = []
            vertices, triangles = self.viewpoint_generation.get_mesh_vertices_and_triangles()

            for vertex in vertices:
                point = Point()
                point.x = vertex[0]
                point.y = vertex[1]
                point.z = vertex[2]
                self.mesh.vertices.append(point)

            for triangle in triangles:
                mesh_triangle = MeshTriangle()
                mesh_triangle.vertex_indices = triangle.astype(
                    np.uint32).tolist()
                self.mesh.triangles.append(mesh_triangle)

            return True

    def create_planning_volume_mesh(self, radius=0.5, height=0.75):
        planning_volume_mesh_path = get_package_prefix(
            'viewpoint_generation') + '/share/viewpoint_generation/meshes/planning_volume.stl'
        planning_volume_mesh_o3d = o3d.io.read_triangle_mesh(
            planning_volume_mesh_path)
        vertices = np.asarray(planning_volume_mesh_o3d.vertices)
        triangles = np.asarray(planning_volume_mesh_o3d.triangles)

        planning_volume_mesh = Mesh()

        for vertex in vertices:
            point = Point()
            point.x = radius*vertex[0]
            point.y = radius*vertex[1]
            point.z = height*vertex[2]
            planning_volume_mesh.vertices.append(point)

        for triangle in triangles:
            mesh_triangle = MeshTriangle()
            mesh_triangle.vertex_indices = triangle.astype(
                np.uint32).tolist()
            planning_volume_mesh.triangles.append(mesh_triangle)

        self.planning_volume_mesh = planning_volume_mesh

    def update_planning_scene(self):
        if not self.mesh:
            self.get_logger().warning(
                'No mesh loaded. Cannot update planning scene.')
            return False

        # Publish bounding box as marker
        self.bbox_marker_publisher.publish(self.bbox_marker)

        # Pose of object relative to 'object_frame'
        # Will be changed by Yusen's point cloud registration
        pose = Pose()

        # Update planning scene with the new mesh
        attached_object = AttachedCollisionObject()
        attached_object.link_name = 'object_frame'
        attached_object.object.header.frame_id = 'object_frame'
        attached_object.object.pose = pose
        attached_object.object.id = 'object'
        attached_object.object.meshes = [self.mesh]
        attached_object.object.mesh_poses = [Pose()]
        attached_object.object.operation = CollisionObject.ADD
        attached_object.touch_links = [
            'turntable_disc_link', 'turntable_base_link']

        # Planning Volume
        planning_volume = AttachedCollisionObject()
        planning_volume.link_name = 'object_frame'
        planning_volume.object.header.frame_id = 'object_frame'
        planning_volume.object.pose = pose
        planning_volume.object.id = 'planning_volume'
        planning_volume.object.meshes = [self.planning_volume_mesh]
        planning_volume.object.mesh_poses = [Pose()]
        planning_volume.object.operation = CollisionObject.ADD
        planning_volume.touch_links = [
            'table_link', 'ur5e_mount_link', 'planning_volume',
            'turntable_base_link_inertia', 'turntable_disc_link',
            'base_link', 'base_link_inertia', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'
        ]

        planning_scene = PlanningScene()
        planning_scene.world.collision_objects.clear()
        planning_scene.robot_state.attached_collision_objects.append(
            attached_object)
        planning_scene.robot_state.attached_collision_objects.append(
            planning_volume)
        planning_scene.robot_state.is_diff = True
        planning_scene.is_diff = True
        planning_scene.object_colors.append(ObjectColor(
            id='object', color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)))
        planning_scene.object_colors.append(ObjectColor(
            id='planning_volume', color=ColorRGBA(r=0.0, g=1.0, b=1.0, a=self.planning_volume_opacity)))

        self.planning_scene_diff_publisher.publish(planning_scene)

    def set_point_cloud_file(self, point_cloud_file, point_cloud_units):
        """
        Helper function to set the point cloud file for the partitioner.
        :param point_cloud_file: The path to the point cloud file.
        :return: None
        """

        # If file doesn't exist, look in data path
        if not point_cloud_file == '' and not os.path.exists(point_cloud_file):
            point_cloud_file = os.path.join(self.data_path, point_cloud_file)

        success, message = self.viewpoint_generation.set_point_cloud_file(
            point_cloud_file, point_cloud_units)

        if not success:
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters_blocked([point_cloud_file_param])
            self.get_logger().warn(message)
            return False
        else:
            self.get_logger().info('Import point cloud completed successfully.')
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                point_cloud_file
            )
            point_cloud_units_param = rclpy.parameter.Parameter(
                'model.point_cloud.units',
                rclpy.Parameter.Type.STRING,
                point_cloud_units
            )
            params = [point_cloud_file_param, point_cloud_units_param]
            # Only clear results.file when the PCD change caused a results reset
            if self.viewpoint_generation.results_file is None:
                params.append(rclpy.parameter.Parameter(
                    'results.file', rclpy.Parameter.Type.STRING, ''))
            self.set_parameters_blocked(params)

            if self.initialized:
                # Clear the curvature file parameter
                self.set_curvature_file(curvature_file='')

            return True

    def set_curvature_file(self, curvature_file):
        """
        Helper function to set the curvature file for the partitioner.
        :param curvature_file: The path to the curvature file.
        :return: None
        """

        # If file doesn't exist, look in data path
        if not curvature_file == '' and not os.path.exists(curvature_file):
            curvature_file = os.path.join(self.data_path, curvature_file)

        success, message = self.viewpoint_generation.set_curvature_file(
            curvature_file)

        if not success:
            self.get_logger().error(message)

            curvature_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.curvature.file',
                rclpy.Parameter.Type.STRING,
                ''
            )

            self.set_parameters_blocked([curvature_file_param])

            return False
        else:
            self.get_logger().info(message)

            curvature_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.curvature.file',
                rclpy.Parameter.Type.STRING,
                curvature_file
            )
            self.set_parameters_blocked([curvature_file_param])

        return True

    def set_results_file(self, results_file):
        """
        Helper function to set the results.file for the partitioner.
        :param results_file: The path to the results.file.
        :return: None
        """

        # If file doesn't exist, look in data path
        if not results_file == '' and not os.path.exists(results_file):
            results_file = os.path.join(self.data_path, results_file)

        success, message = self.viewpoint_generation.set_results_file(
            results_file)

        # Get Viewpoint Bounds
        max_radius, max_z = self.viewpoint_generation.get_viewpoint_bounds()
        print(f"Max Radius: {max_radius}, Max Z: {max_z}")
        if max_radius < 0 or max_z < 0:
            self.create_planning_volume_mesh()
        else:
            self.create_planning_volume_mesh(
                radius=max_radius + 0.2, height=2 * max_z + 0.1)

        if not success:
            self.get_logger().error(message)

            results_file_param = rclpy.parameter.Parameter(
                'results.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.block_next_param_callback = True
            self.set_parameters([results_file_param])

            return False
        else:
            self.get_logger().info(message)

            results_file_param = rclpy.parameter.Parameter(
                'results.file',
                rclpy.Parameter.Type.STRING,
                results_file
            )
            self.block_next_param_callback = True
            self.set_parameters([results_file_param])

            return True

    def enable_cuda_callback(self, enable):
        """
        Helper function to enable or disable CUDA for the partitioner.
        :param enable: Boolean indicating whether to enable CUDA.
        :return: True if successful, False otherwise.
        """

        success = self.viewpoint_generation.enable_cuda(enable)

        if not success:
            self.get_logger().error(
                'Failed to enable CUDA. Please check your setup.'
            )
            return False
        else:
            if enable:
                self.get_logger().info('CUDA enabled successfully.')
            else:
                self.get_logger().info('CUDA disabled.')
            return True

    def set_sampling_number_of_points(self, number_of_points):
        """
        Helper function to set the number of points to sample.
        :param number_of_points: The number of points to sample.
        :return: True if successful, False otherwise.
        """

        if number_of_points <= 0:
            self.get_logger().error(
                'Number of points must be greater than 0.'
            )
            return False

        success, number_of_points = self.viewpoint_generation.set_sampling_number_of_points(
            number_of_points)

        if not success:
            self.get_logger().error(
                f'Failed to set number of points to {number_of_points}.'
            )
            return False
        else:
            self.get_logger().info(
                f'Number of points set to {number_of_points}.')
            number_of_points_param = rclpy.parameter.Parameter(
                'model.point_cloud.sampling.number_of_points',
                rclpy.Parameter.Type.INTEGER,
                number_of_points
            )
            self.block_next_param_callback = True
            self.set_parameters([number_of_points_param])

        return True

    def sample_point_cloud_callback(self, request, response):
        """ Callback for the sample point cloud service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
            success (bool): True if point cloud sampling was successful, False otherwise.
            message (str): Returns the file path of the sampled point cloud if successful, or an error message if not.
        """

        self.get_logger().info('Sampling point cloud...')

        success, message = self.viewpoint_generation.sample_point_cloud()

        if success:
            self.get_logger().info(
                f"Point cloud sampled successfully. File: {message}")

            # Set the point cloud of the partitioner
            pcd_file = message

            # Update the point cloud units to meters
            point_cloud_units_param = rclpy.parameter.Parameter(
                'model.point_cloud.units',
                rclpy.Parameter.Type.STRING,
                'm'
            )
            self.block_next_param_callback = True
            self.set_parameters([point_cloud_units_param])

            # Update the point cloud file parameter with the sampled file
            point_cloud_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.file',
                rclpy.Parameter.Type.STRING,
                pcd_file
            )

            self.set_parameters(
                [point_cloud_file_param])

        else:
            self.get_logger().error(f"Failed to sample point cloud: {message}")

        response.success = success
        response.message = message

        return response

    def set_knn_neighbors(self, number_of_neighbors):
        """
        Helper function to set the number of neighbors for curvature estimation.
        :param number_of_neighbors: The number of neighbors to use for curvature estimation.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_knn_neighbors(
            number_of_neighbors)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            curvature_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.curvature.file',
                rclpy.Parameter.Type.STRING,
                ''
            )
            self.set_parameters([curvature_file_param])

        return True

    def set_seed_threshold(self, seed_threshold):
        """
        Helper function to set the seed threshold for region growth.
        :param seed_threshold: The seed threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_seed_threshold(
            seed_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_min_cluster_size(self, min_cluster_size):
        """
        Helper function to set the minimum cluster size for region growth.
        :param min_cluster_size: The minimum cluster size to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_min_cluster_size(
            min_cluster_size)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_max_cluster_size(self, max_cluster_size):
        """
        Helper function to set the maximum cluster size for region growth.
        :param max_cluster_size: The maximum cluster size to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_max_cluster_size(
            max_cluster_size)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_normal_angle_threshold(self, normal_angle_threshold):
        """
        Helper function to set the angle threshold for region growth.
        :param normal_angle_threshold: The angle threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_normal_angle_threshold(
            normal_angle_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def set_curvature_threshold(self, curvature_threshold):
        """
        Helper function to set the curvature threshold for region growth.
        :param curvature_threshold: The curvature threshold to use for region growth.
        :return: True if successful, False otherwise.
        """

        success, message = self.viewpoint_generation.set_curvature_threshold(
            curvature_threshold)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def estimate_curvature_callback(self, request, response):
        """
        Callback for the estimate curvature service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
            success (bool): True if curvature estimation was successful, False otherwise.
            message (str): Returns the file path of the estimated curvature file if successful, or an error message if not.
        """
        self.get_logger().info('Estimating curvature...')

        success, message = self.viewpoint_generation.estimate_curvature()

        if success:
            self.get_logger().info(
                f"Curvature estimation completed successfully. Curvature file: {message}")
            # Set the curvature file parameter with the estimated curvature file
            curvature_file_param = rclpy.parameter.Parameter(
                'model.point_cloud.curvature.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([curvature_file_param])
        else:
            self.get_logger().error("Curvature estimation failed.")

        response.success = success
        response.message = message

        return response

    def region_growth_callback(self, request, response):
        """
        Callback for the region growth service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
        """
        self.get_logger().info('Performing region growth...')

        success, message = self.viewpoint_generation.region_growth()

        if success:
            self.get_logger().info(
                f"Region growth completed successfully. results.file: {message}")
            results_file_param = rclpy.parameter.Parameter(
                'results.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([results_file_param])
        else:
            self.get_logger().error(f"Region growth failed: {message}")

        response.success = success
        response.message = message

        return response

    def set_camera_parameters(self, fov_width, fov_height, dof, focal_distance):
        success, message = self.viewpoint_generation.set_camera_parameters(
            fov_width, fov_height, dof, focal_distance)

        if not success:
            self.get_logger().error(message)
            return False
        else:
            self.get_logger().info(message)
            return True

    def fov_clustering_callback(self, request, response):
        """
        Callback for the FOV clustering service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
        """
        self.get_logger().info('Performing FOV clustering...')

        success, message = self.viewpoint_generation.fov_clustering()

        if success:
            self.get_logger().info(
                f"FOV clustering completed successfully. results.file updated: {message}")
            # Set the results.file parameter with the FOV clustering result
            results_file_param = rclpy.parameter.Parameter(
                'results.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([results_file_param])
        else:
            self.get_logger().error(f"FOV clustering failed: {message}")

        response.success = success
        response.message = message

        return response

    def viewpoint_projection_callback(self, request, response):
        """
        Callback for the viewpoint projection service.
        :param request: The request object.
        :param response: The response object.
        :return: The response object.
            success (bool): True if viewpoint projection was successful, False otherwise.
            message (str): Returns the file path of the projected viewpoints if successful, or an error message if not.
        """
        self.get_logger().info('Projecting viewpoints...')

        success, message = self.viewpoint_generation.project_viewpoints()

        if success:
            self.get_logger().info(
                f"Viewpoint projection completed successfully. Viewpoints file: {message}")
            # Set the viewpoints file parameter with the projected viewpoints file
            # Set the results.file parameter with the FOV clustering result
            results_file_param = rclpy.parameter.Parameter(
                'results.file',
                rclpy.Parameter.Type.STRING,
                message
            )
            self.set_parameters([results_file_param])
        else:
            self.get_logger().error(f"Viewpoint projection failed: {message}")

        response.success = success
        response.message = message

        return response

    def optimize_traversal(self, request, response):
        request = OptimizeViewpointTraversal.Request()
        request.viewpoint_dict_path = self.get_parameter(
            'results.file').get_parameter_value().string_value

        future = self.optimize_traversal_client.call_async(request)
        future.add_done_callback(self.optimize_traversal_callback)

        response.success = True
        response.message = 'Optimization started...'
        return response

    def optimize_traversal_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info('Optimization successful.')
                new_viewpoint_dict_path = result.new_viewpoint_dict_path
                # Update the results.file parameter with the new viewpoint dictionary path
                results_file_param = rclpy.parameter.Parameter(
                    'results.file',
                    rclpy.Parameter.Type.STRING,
                    new_viewpoint_dict_path
                )
                self.block_next_param_callback = True
                self.set_parameters([results_file_param])
                self.get_logger().info(
                    f'New viewpoint dictionary saved at: {new_viewpoint_dict_path}')
            else:
                self.get_logger().error(
                    f'Optimization failed: {result.message}')
                return
        except Exception as e:
            self.get_logger().error(f"Optimization failed: {e}")

    def parameter_callback(self, params):
        """ Callback for parameter changes.
        :param params: List of parameters that have changed.
        :return: SetParametersResult indicating success or failure.
        """

        # If we are blocking the next parameter callback, return success
        if self.block_next_param_callback:
            self.block_next_param_callback = False
            return SetParametersResult(successful=True)

        success = True

        # Iterate through the parameters and set the corresponding values
        # based on the parameter name
        for param in params:
            if param.name == 'model.mesh.file':
                success = self.set_mesh_file(param.value,
                                             self.get_parameter(
                                                 'model.mesh.units').get_parameter_value().string_value)
            elif param.name == 'model.mesh.units':
                success = self.set_mesh_file(
                    self.get_parameter(
                        'model.mesh.file').get_parameter_value().string_value,
                    param.value)
            elif param.name == 'model.point_cloud.file':
                success = self.set_point_cloud_file(param.value, self.get_parameter(
                    'model.point_cloud.units').get_parameter_value().string_value)
            elif param.name == 'model.point_cloud.units':
                success = self.set_point_cloud_file(
                    self.get_parameter(
                        'model.point_cloud.file').get_parameter_value().string_value,
                    param.value)
            elif param.name == 'model.point_cloud.sampling.number_of_points':
                success = self.set_sampling_number_of_points(param.value)
            elif param.name == 'results.file':
                success = self.set_results_file(param.value)
            elif param.name == 'model.point_cloud.curvature.file':
                success = self.set_curvature_file(param.value)
            elif (param.name.startswith('regions.region_growth.') or
                  param.name.startswith('regions.fov_clustering.') or
                  param.name.startswith('viewpoints.projection.')):
                field_name = param.name.split('.')[-1]
                success, message = self.viewpoint_generation.set_algorithm_param(
                    field_name, param.value)
                if success:
                    self.get_logger().info(message)
                    if field_name == 'knn_neighbors':
                        self.set_curvature_file(curvature_file='')
                else:
                    self.get_logger().error(message)
            elif param.name == 'settings.data_path':
                self.set_data_path(param.value)
            elif param.name == 'settings.planning_volume_opacity':
                self.planning_volume_opacity = param.value

            print(f'{param.name}: {success}')

        result = SetParametersResult()
        result.successful = success

        return result


def main():
    rclpy.init()
    node = ViewpointGenerationNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
