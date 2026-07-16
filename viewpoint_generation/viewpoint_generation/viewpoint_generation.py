import os
import json
import time
import math
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from matplotlib import colormaps
from open3d.geometry import PointCloud, TriangleMesh

from viewpoint_generation.region_growth import *
from viewpoint_generation.partfield_segmentation import *
from viewpoint_generation.fov_clustering import *
from viewpoint_generation.viewpoint_projection import *
from viewpoint_generation.mesh_utils import submesh_from_faces


class ViewpointGeneration():

    mesh_file = None
    mesh_units = 'm'
    point_cloud_file = None
    point_cloud_units = 'm'
    results_file = None

    mesh = None
    point_cloud = None

    # Region Growth Parameters

    results = {}

    mesh_color = (0.5, 0.5, 0.5)
    bb_color = (1., 1., 1.)
    text_color = (1., 1., 1.)

    # Half-width (m) of the flat ground-plane occluder added to the shared
    # raycasting scene, centered at the origin above the table/turntable
    # surface the part sits on. Comfortably exceeds any plausible part or
    # planning-volume footprint so it always spans the workspace.
    ground_plane_half_size = 5.0
    # Z offset (m) of the ground plane below the nominal table surface
    # (z=0). Placed strictly below zero, not at it, so a mesh point resting
    # exactly at z=0 doesn't coincide with the plane: the occlusion ray's
    # self-intersection-avoidance shrink (occlusion_epsilon) stops each ray
    # just short of its own target, so a ground plane exactly AT a target's
    # z-level never gets crossed and silently fails to register as blocking
    # -- confirmed empirically. -0.02 clears occlusion_epsilon's entire
    # declared range ([0, 0.01], see FOVClusteringConfig) with 2x margin,
    # while still being negligible next to any genuinely underground camera.
    ground_plane_z = -0.02

    # Segmentation algorithm used to partition the surface into regions.
    # One of: 'region_growth' (mesh face-adjacency) or 'partfield' (PartField parts).
    segmentation_algorithm = 'partfield'

    region_growing_config = RegionGrowingConfig(
        seed_threshold=0.1,
        min_cluster_size=10,
        normal_angle_threshold=np.pi / 3,
        curvature_threshold=0.1,
    )
    partfield_config = PartFieldSegmentationConfig(
        num_parts=12,
    )
    fc_config = FOVClusteringConfig(
        fov_diameter=0.03,
        dof=0.02,
        point_density=0.5,
        lambda_weight=1.0,
        beta_weight=1.0,
        max_point_out_percentage=0.001,
        point_weight=1.0,
        normal_weight=1.0,
        number_of_runs=10,
        maximum_iterations=100
    )
    vp_config = ViewpointProjectionConfig(
        focal_distance=0.3,
    )

    rg = RegionGrowing(region_growing_config)
    pf = PartFieldSegmentation(partfield_config)
    fc = FOVClustering(fc_config)
    vp = ViewpointProjection(vp_config)

    raycasting_scene = o3d.t.geometry.RaycastingScene()

    def __init__(self):
        pass

    def get_mesh_bounds(self):
        """
        Get the mininum and maximum bounds of the mesh.
        Returns:
            tuple: (min_x, min_y, min_z, max_x, max_y, max_z) of the mesh in meters.
        """
        if self.mesh is None:
            return None, None, None, None, None, None

        bbox = self.mesh.get_axis_aligned_bounding_box()

        min_x = bbox.min_bound[0]
        min_y = bbox.min_bound[1]
        min_z = bbox.min_bound[2]
        max_x = bbox.max_bound[0]
        max_y = bbox.max_bound[1]
        max_z = bbox.max_bound[2]

        return min_x, min_y, min_z, max_x, max_y, max_z

    def _build_ground_plane_mesh(self):
        """A large flat quad just below z=0, added to the shared raycasting
        scene as an occluder alongside the part mesh. Any ray between a
        candidate camera position and a target point that would have to
        pass through the table/turntable surface to connect them now
        correctly registers as occluded, so occlusion-checked
        coverage/rescue/refinement (see occlusion_search.py) can never
        place or credit a viewpoint whose camera position is underground."""
        s = self.ground_plane_half_size
        z = self.ground_plane_z
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = o3d.utility.Vector3dVector(np.array([
            [-s, -s, z],
            [ s, -s, z],
            [ s,  s, z],
            [-s,  s, z],
        ]))
        plane.triangles = o3d.utility.Vector3iVector(np.array([
            [0, 1, 2],
            [0, 2, 3],
        ]))
        return plane

    def get_mesh_vertices_and_triangles(self):
        """
        Get the vertices and triangles of the mesh.
        Returns:
            tuple: (vertices, triangles) where vertices is a numpy array of shape (N, 3)
            and triangles is a numpy array of shape (M, 3).
        """
        if self.mesh is None:
            return None, None

        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        return vertices, triangles

    def set_mesh_file(self, mesh_file, units):
        """Set the mesh file and its units."""

        # Check if the new mesh file and units are the same as the current ones
        if mesh_file == self.mesh_file and units == self.mesh_units:
            return False, 'Mesh file and units already set.'

        # Clear existing mesh
        if mesh_file == '':
            self.mesh = None
            self.mesh_file = None
            self.mesh_units = None
            return True, 'No triangle mesh file provided. Mesh cleared.'

        try:
            mesh = o3d.io.read_triangle_mesh(mesh_file)
        except Exception as e:
            return False, f'Could not load requested triangle mesh file: {e}'

        # Check if the mesh is empty
        if mesh.is_empty():
            return False, 'The loaded triangle mesh is empty.'

        mesh.compute_vertex_normals()

        # Scale the mesh to meters
        if units == 'cm':
            mesh.scale(0.01, center=(0, 0, 0))
        elif units == 'mm':
            mesh.scale(0.001, center=(0, 0, 0))
        elif units == 'in':
            mesh.scale(0.0254, center=(0, 0, 0))
        elif units == 'm':
            # No scaling needed for meters
            pass
        else:
            return False, 'Unknown units. Mesh not scaled.'

        # Check if the mesh has colors
        mesh.paint_uniform_color(self.mesh_color)

        # Update the triangle mesh file
        self.mesh_file = mesh_file
        self.mesh_units = units
        self.mesh = mesh

        # Generate dimensions string
        min_x, min_y, min_z, max_x, max_y, max_z = self.get_mesh_bounds()
        dimensions_str = f"(LxWxH): {max_x - min_x:.2f} x {max_y - min_y:.2f} x {max_z - min_z:.2f} m"
        # Generate Surface Area string
        surface_area_str = f"Surface Area: {self.mesh.get_surface_area():.2f} m^2"

        # Only reset results when a genuinely different mesh is loaded.
        # When launching from a saved config the results file is loaded first;
        # set_mesh_file is then called with the same mesh path AND units, so we
        # preserve all regions/clusters already in self.results. If either the
        # file or the units differ, the derived data is invalid and we reset.
        existing_mesh = self.results.get('meshes', [{}])[0]
        existing_mesh_file = existing_mesh.get('file', '')
        existing_mesh_units = existing_mesh.get('units', '')
        if mesh_file != existing_mesh_file or units != existing_mesh_units:
            self.results = {'meshes': [
                {
                    'file': mesh_file,
                    'units': units,
                    'material': 'unknown',
                    'dimensions': dimensions_str,
                    'surface_area': surface_area_str,
                    'regions': [],
                    'order': [],
                    'noise_faces': []
                }
            ]
            }
            self.results_file = None
        # else: same mesh file and units as the already-loaded results —
        # preserve self.results and self.results_file so a saved config can be
        # resumed without discarding its regions/clusters.

        # Raycasting Scene, built once and shared by FOV clustering's
        # occlusion check and viewpoint projection (both need line-of-sight
        # against the full part mesh, not just their own region/cluster).
        # Includes a ground-plane occluder so no computed viewpoint's camera
        # position or line-of-sight can end up underground.
        self.raycasting_scene = o3d.t.geometry.RaycastingScene()
        self.raycasting_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        self.raycasting_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self._build_ground_plane_mesh()))
        self.fc.set_scene(self.raycasting_scene)
        self.vp.set_scene(self.raycasting_scene)

        return True, ''

    def set_point_cloud_file(self, point_cloud_file, point_cloud_units):
        """Set the point cloud file and its units."""

        # Check if the new point cloud file and units are the same as the current ones
        if point_cloud_file == self.point_cloud_file and point_cloud_units == self.point_cloud_units:
            return False, 'Point cloud file already set.'

        # Clear existing point cloud
        if point_cloud_file == '':
            self.point_cloud_file = None
            self.point_cloud = None
            return False, 'Point cloud file cleared.'

        # Load the new point cloud
        try:
            point_cloud = o3d.io.read_point_cloud(point_cloud_file)
        except Exception as e:
            return False, f'Could not load requested point cloud file: {e}'

        # Check if the point cloud is empty
        if point_cloud.is_empty():
            return False, 'The loaded point cloud is empty.'
        # Check if the point cloud has normals
        if not point_cloud.has_normals():
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))

        # Check if the point cloud has colors
        if not point_cloud.has_colors():
            point_cloud.paint_uniform_color((1, 1, 1))

        # Scale the point cloud to meters
        if point_cloud_units == 'cm':
            point_cloud.scale(0.01, center=(0, 0, 0))
        elif point_cloud_units == 'mm':
            point_cloud.scale(0.001, center=(0, 0, 0))
        elif point_cloud_units == 'in':
            point_cloud.scale(0.0254, center=(0, 0, 0))
        elif point_cloud_units == 'm':
            # No scaling needed for meters
            pass
        else:
            return False, f'Unknown units \'{point_cloud_units}\'. Point cloud not scaled.'

        self.point_cloud_file = point_cloud_file
        self.point_cloud = point_cloud
        self.point_cloud_units = point_cloud_units

        return True, ''

    def set_seed_threshold(self, seed_threshold):
        if seed_threshold <= 0:
            return False, 'Seed threshold must be greater than 0.'
        self.region_growing_config.seed_threshold = seed_threshold
        self.rg.config = self.region_growing_config
        return True, f'Seed threshold set to {seed_threshold}.'

    def set_min_cluster_size(self, min_cluster_size):
        if min_cluster_size <= 0:
            return False, 'Minimum cluster size must be greater than 0.'
        self.region_growing_config.min_cluster_size = min_cluster_size
        self.rg.config = self.region_growing_config
        return True, f'Minimum cluster size set to {min_cluster_size}.'

    def set_max_cluster_size(self, max_cluster_size):
        if max_cluster_size <= 0:
            return False, 'Maximum cluster size must be greater than 0.'
        elif max_cluster_size < self.region_growing_config.min_cluster_size:
            return False, 'Maximum cluster size must be greater than or equal to minimum cluster size.'
        self.region_growing_config.max_cluster_size = max_cluster_size
        self.rg.config = self.region_growing_config
        return True, f'Maximum cluster size set to {max_cluster_size}.'

    def set_normal_angle_threshold(self, normal_angle_threshold):
        """
        Set the angle threshold for region growing.
        Args:
            normal_angle_threshold (float): Angle threshold in radians.
        Returns:
            bool: True if the angle threshold was set successfully, False otherwise.
        """
        if normal_angle_threshold <= 0 or normal_angle_threshold > np.pi:
            return False, 'Angle threshold must be between 0 and π radians.'

        self.region_growing_config.normal_angle_threshold = normal_angle_threshold
        self.rg.config = self.region_growing_config

        return True, f'Region growth angle threshold set to {normal_angle_threshold:.4f} radians.'

    def set_curvature_threshold(self, curvature_threshold):
        """
        Set the curvature threshold for region growing.
        Args:
            curvature_threshold (float): Curvature threshold in percent.
        Returns:
            bool: True if the curvature threshold was set successfully, False otherwise.
        """
        if curvature_threshold <= 0 or curvature_threshold > 1:
            return False, 'Curvature threshold must be between 0 and 1.'

        self.region_growing_config.curvature_threshold = curvature_threshold
        self.rg.config = self.region_growing_config

        return True, f'Region growth curvature threshold set to {100*curvature_threshold} percent.'

    def _set_config_param(self, config, module, field_name, value):
        """
        Shared generic setter for any config dataclass that implements to_dict().
        Validates the field name, updates the config, and syncs it to the algorithm object.
        Args:
            config: The config dataclass instance (e.g. region_growing_config, fc_config).
            module: The algorithm module whose .config attribute should be kept in sync.
            field_name (str): Field name as it appears in config.to_dict().
            value: New value for the field.
        Returns:
            tuple: (bool, str) success flag and message.
        """
        if field_name not in config.to_dict():
            return False, f'Unknown config field: \'{field_name}\'.'
        setattr(config, field_name, value)
        module.config = config
        return True, f'\'{field_name}\' set to {value}.'

    def set_algorithm_param(self, field_name, value):
        """
        Generic setter that routes field_name to whichever algorithm config owns it.
        Checks RegionGrowingConfig, FOVClusteringConfig, then ViewpointProjectionConfig.
        Returns (False, message) if the field is not found in any config.
        """
        if field_name in self.region_growing_config.to_dict():
            return self._set_config_param(
                self.region_growing_config, self.rg, field_name, value)
        if field_name in self.partfield_config.to_dict():
            return self._set_config_param(
                self.partfield_config, self.pf, field_name, value)
        if field_name in self.fc_config.to_dict():
            return self._set_config_param(
                self.fc_config, self.fc, field_name, value)
        if field_name in self.vp_config.to_dict():
            return self._set_config_param(
                self.vp_config, self.vp, field_name, value)
        return False, f'Unknown algorithm config field: \'{field_name}\'.'

    def set_segmentation_algorithm(self, algorithm):
        """
        Select the algorithm used by segment_regions() to partition the surface.
        Args:
            algorithm (str): 'region_growth' or 'partfield'.
        Returns:
            tuple: (bool, str) success flag and message.
        """
        valid = ('region_growth', 'partfield')
        if algorithm not in valid:
            return False, f'Unknown segmentation algorithm: \'{algorithm}\'. Valid: {valid}.'
        self.segmentation_algorithm = algorithm
        return True, f'Segmentation algorithm set to \'{algorithm}\'.'

    def set_camera_parameters(self, fov_height, fov_width, dof, focal_distance):
        """
        Set the camera parameters for FOV clustering.
        Args:
            fov_height (float): Height of the field of view in meters.
            fov_width (float): Width of the field of view in meters.
            dof (float): Depth of field in meters.
            focal_distance (float): Focal distance in meters.
        Returns:
            bool: True if the camera parameters were set successfully, False otherwise.
        """
        if fov_height <= 0 or fov_width <= 0 or dof <= 0 or focal_distance <= 0:
            return False, 'FOV height, width, DOF and focal distance must be greater than 0.'

        self.fc_config.fov_height = fov_height
        self.fc_config.fov_width = fov_width
        self.fc_config.dof = dof
        self.vp_config.focal_distance = focal_distance

        self.fc.config = self.fc_config
        self.vp.config = self.vp_config

        return True, f'Camera parameters set to FOV height: {fov_height} m, FOV width: {fov_width} m, DOF: {dof} m, focal distance: {focal_distance} m.'

    def set_results_file(self, results_file):
        """
        Set the regions file to be used for region growing.
        Args:
            results_file (str): Path to the regions file.
        Returns:
            bool: True if the regions file was set successfully, False otherwise.
        """
        if results_file == '':
            self.results_file = None
            return True, 'Regions file cleared.'

        if not os.path.exists(results_file):
            return False, f'Regions file does not exist: \'{results_file}\'.'
        elif not results_file.endswith('.json'):
            return False, f'Regions file is not a .json file: \'{results_file}\'.'

        with open(results_file, 'r') as f:
            self.results = json.load(f)

        self.results_file = results_file

        return True, f'Regions file set to \'{results_file}\'.'

    def get_viewpoint_bounds(self):
        if not self.results:
            return None

        max_x = 0
        max_y = 0
        max_z = 0

        for region in self.results['meshes'][0]['regions']:
            # If no cluster is found, skip this region
            if 'clusters' not in region:
                continue
            for cluster in region['clusters']:
                if 'viewpoint' not in cluster:
                    continue
                viewpoint_position = cluster["viewpoint"]["position"]
                max_x = max(max_x, abs(viewpoint_position[0]))
                max_y = max(max_y, abs(viewpoint_position[1]))
                max_z = max(max_z, abs(viewpoint_position[2]))

        return max_x, max_y, max_z

    def _segment_surface(self):
        """
        Partition the mesh into regions using the selected algorithm.
        Returns:
            tuple: (regions, noise_faces). Each region is a list of triangle
            indices into self.mesh.triangles; noise_faces is a list of
            unassigned triangle indices.
        """
        if self.mesh is None:
            raise ValueError('No triangle mesh loaded. Cannot segment surface.')
        if self.segmentation_algorithm == 'partfield':
            return self.pf.segment(self.mesh)
        return self.rg.segment(self.mesh)

    def _results_dir(self):
        """Directory where results/region artifacts for the current mesh are
        stored, derived from the point cloud file if set, otherwise the mesh
        file. Returns None if neither is set."""
        base_file = self.point_cloud_file if self.point_cloud_file is not None else self.mesh_file
        if base_file is None:
            return None

        base_dir = base_file.rsplit('/', 1)[0]
        base_name = base_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        results_dir = base_dir + '/' + base_name + self.mesh_units + '_results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir

    def _sample_region_point_cloud(self, region_id, faces):
        """Poisson-disk-sample a region's submesh, sized from
        fc_config.point_density (points per square millimeter), and cache it
        to disk next to the results file.
        Returns:
            tuple: (point_cloud, point_cloud_file)
        """
        mesh_triangles = np.asarray(self.mesh.triangles)
        submesh, _ = submesh_from_faces(self.mesh, mesh_triangles[np.asarray(faces)])
        area_mm2 = submesh.get_surface_area() * 1e6
        n_points = max(4, int(round(self.fc_config.point_density * area_mm2)))

        results_dir = self._results_dir()
        point_cloud_file = f'{results_dir}/region_{region_id}_{n_points}points.ply'

        if not os.path.exists(point_cloud_file):
            point_cloud = submesh.sample_points_poisson_disk(
                number_of_points=n_points, init_factor=5, use_triangle_normal=True)
            o3d.io.write_point_cloud(point_cloud_file, point_cloud)
        else:
            point_cloud = o3d.io.read_point_cloud(point_cloud_file)

        return point_cloud, point_cloud_file

    def segment_regions(self):
        """
        Segment the mesh into regions and sample a point cloud for each
        region (used later by FOV clustering/viewpoint projection).
        Returns:
            tuple: (bool, str) success flag and results file path (or message).
        """
        for i in range(len(self.results['meshes'])):

            regions, noise_faces = self._segment_surface()

            self.results['meshes'][i]['regions'] = []
            for region_id, region in enumerate(regions):
                point_cloud, point_cloud_file = self._sample_region_point_cloud(
                    region_id, region)
                self.results['meshes'][i]['regions'].append({
                    'faces': region,
                    'point_cloud': {
                        'file': point_cloud_file,
                        'units': 'm',
                        'points': len(point_cloud.points),
                    },
                    'clusters': [],
                    'order': []
                })
            self.results['meshes'][i]['noise_faces'] = noise_faces

            self.results['meshes'][i]['order'] = list(
                range(len(self.results['meshes'][i]['regions'])))

        self.results_file = self.save_results(self.results)

        return True, self.results_file

    def save_results(self, results):
        # Add Camera Config to results
        results['meshes'][0]['camera_config'] = {
            'fov_diameter': self.fc_config.fov_diameter,
            'dof': self.fc_config.dof,
            'focal_distance': self.vp_config.focal_distance,
        }
        # Count regions and clusters
        N_regions = 0
        N_clusters = 0
        regions = results.get('meshes', [{}])[0].get('regions', {})
        for region in regions:
            N_regions += 1
            if 'clusters' in region:
                N_clusters += len(region['clusters'])

        # With no regions yet, write to /tmp to avoid cluttering the data directory
        if N_regions == 0:
            results_dir = '/tmp'
        else:
            results_dir = self._results_dir()
            if results_dir is None:
                return None

        # Build filename
        results_file = results_dir + '/'
        if N_regions > 0:
            results_file += str(N_regions) + '_regions_'
        if N_clusters > 0:
            results_file += str(N_clusters) + '_clusters_'
        results_file += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'Results saved to {results_file}.')
        return results_file

    def _load_region_point_cloud(self, region):
        """Load a region's cached, per-region sampled point cloud (produced
        by segment_regions()/_sample_region_point_cloud())."""
        pcd_file = region.get('point_cloud', {}).get('file')
        if not pcd_file or not os.path.exists(pcd_file):
            raise ValueError(
                'Region has no cached point cloud file. Run segment_regions() first.')
        return o3d.io.read_point_cloud(pcd_file)

    def fov_clustering(self):
        """
        Perform FOV clustering on the regions obtained from segmentation.
        Returns:
            bool: True if FOV clustering was successful, False otherwise.
        """
        if self.results_file is None:
            success, msg = self.segment_regions()
            if not success:
                return False, msg

        # Shared reference point for structured_candidates' cylindrical
        # elevation/azimuth grid, so different regions' bins line up. The
        # whole part's (not each region's own) center, so azimuth=0 means
        # the same absolute direction everywhere.
        min_x, min_y, _, max_x, max_y, _ = self.get_mesh_bounds()
        part_center_xy = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

        # Iterate through regions in the region dictionary
        for region_id, region in enumerate(self.results['meshes'][0]['regions']):
            print(
                f"Performing FOV clustering on region {region_id} with {len(region['faces'])} faces.")
            region_point_cloud = self._load_region_point_cloud(region)
            # Perform FOV clustering
            fov_clusters, blind_spots = self.fc.fov_clustering(
                region_point_cloud, focal_distance=self.vp_config.focal_distance,
                part_center_xy=part_center_xy)
            print(
                f"Region {region_id} clustered into {len(fov_clusters)} FOV clusters.")
            if blind_spots.get('points'):
                print(
                    f"WARNING: region {region_id}: {len(blind_spots['points'])} points "
                    f"are unreachable ({blind_spots['reason']}) — genuine inspection "
                    f"blind spot, not a clustering bug.")
            self.results['meshes'][0]['regions'][region_id]['clusters'] = []
            for fov_cluster in fov_clusters:
                points = fov_cluster['points']
                if len(points) <= 3:
                    continue
                cluster_entry = {'points': points}
                if fov_cluster.get('axis') is not None:
                    cluster_entry['candidate_axis'] = fov_cluster['axis']
                self.results['meshes'][0]['regions'][region_id]['clusters'].append(
                    cluster_entry)
            self.results['meshes'][0]['regions'][region_id]['blind_spots'] = blind_spots

            # Assign default order
            self.results['meshes'][0]['regions'][region_id]['order'] = list(
                range(len(self.results['meshes'][0]['regions'][region_id]['clusters'])))

        self.results_file = self.save_results(self.results)

        return True, self.results_file

    def project_viewpoints(self):
        """
        Project viewpoints for each FOV cluster in the regions.
        Returns:
            bool: True if viewpoint projection was successful, False otherwise.
        """
        if self.results_file is None:
            return False, 'No region file loaded. Please run region growth first.'

        # Iterate through regions in the region dictionary
        for region_id, region in enumerate(self.results['meshes'][0]['regions']):
            if 'clusters' not in region:
                continue

            region_point_cloud = self._load_region_point_cloud(region)

            for fov_cluster_id, fov_cluster in enumerate(region['clusters']):
                fov_pcd = region_point_cloud.select_by_index(
                    fov_cluster['points'])
                fov_points = np.asarray(fov_pcd.points)
                fov_normals = np.asarray(fov_pcd.normals)
                # Project viewpoint for the FOV cluster. candidate_axis, when
                # present, is the occlusion-validated axis clustering already
                # proved works for this cluster's points — passed through so
                # the direction-refinement search below can never do worse
                # than what clustering already established, regardless of
                # its own independent RNG draw.
                candidate_axis = fov_cluster.get('candidate_axis')
                origin, position, direction, orientation, imaging_mode = self.vp.generate_viewpoint(
                    fov_points, fov_normals,
                    fov_normal_threshold=self.fc_config.fov_normal_threshold,
                    occlusion_epsilon=self.fc_config.occlusion_epsilon,
                    rng_seed=self.fc_config.rng_seed,
                    candidate_axis=candidate_axis)
                # Store the viewpoint in the region dictionary
                self.results['meshes'][0]['regions'][region_id]['clusters'][fov_cluster_id]['viewpoint'] = {
                    'origin': origin.tolist(),
                    'position': position.tolist(),
                    'direction': direction.tolist(),
                    'orientation': orientation.tolist(),
                    'imaging_mode': imaging_mode,
                }
                # Visualize the projected viewpoint
                # viewpoint_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                # viewpoint_mesh.translate(viewpoint)
                # viewpoint_mesh.paint_uniform_color((1, 0, 0))  # Red color
                # o3d.visualization.draw_geometries(
                #     [viewpoint_mesh, fov_points],
                #     window_name=f'Region {region_id} FOV Cluster {fov_cluster_id} Viewpoint Projection',
                #     mesh_show_back_face=True
                # )

        # Save the updated regions dictionary with viewpoints
        self.results_file = self.save_results(self.results)

        return True, self.results_file

    def get_viewpoint_bounds(self):
        """
        Get the mininum and maximum viewpoint positions.
        Returns:
            tuple: (min_x, min_y, min_z, max_x, max_y, max_z) of the mesh in meters.
        """
        max_radius = max_z = float('-inf')

        if self.results and 'meshes' in self.results:
            for region in self.results['meshes'][0]['regions']:
                if 'clusters' not in region:
                    continue

                for cluster in region['clusters']:
                    if 'viewpoint' not in cluster:
                        continue

                    viewpoint = cluster['viewpoint']
                    max_radius = max(
                        max_radius, np.linalg.norm(viewpoint['position'][:2]))
                    max_z = max(max_z, viewpoint['position'][2])

        return max_radius, max_z

    def get_viewpoint(self, region_index, cluster_index):
        """
        Get the viewpoint for a specific region and cluster.
        Args:
            region_index (int): Index of the region.
            cluster_index (int): Index of the cluster.
        Returns:
            dict: Viewpoint information including origin, viewpoint, and direction.
        """
        if self.results_file is None:
            return None, 'No results file loaded. Please run region growth first.'

        if self.results is None or 'meshes' not in self.results:
            return None, 'No regions found in the results dictionary.'

        if region_index < 0 or region_index >= len(self.results['meshes'][0]['regions']):
            return None, f'Invalid region index: {region_index}.'

        if 'clusters' not in self.results['meshes'][0]['regions'][region_index]:
            return None, f'No clusters found for region index: {region_index}.'

        if cluster_index < 0 or cluster_index >= len(self.results['meshes'][0]['regions'][region_index]['clusters']):
            return None, f'Invalid cluster index: {cluster_index}.'

        viewpoint = self.results['meshes'][0]['regions'][region_index]['clusters'][cluster_index].get(
            'viewpoint', None)

        if viewpoint is None:
            return None, 'No viewpoint found for the specified region and cluster.'

        return viewpoint, 'Viewpoint retrieved successfully.'

# Utility functions


def create_sample_mesh(k: int = 3, point_density: float = 1000) -> o3d.geometry.PointCloud:
    """Create a sample point cloud for testing."""

    # Randomly create k primitive shapes
    combined_mesh = o3d.geometry.TriangleMesh()
    for i in range(k):
        rand_n = np.random.randint(0, 3)
        if rand_n == 0:
            # Create a sphere
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        elif rand_n == 1:
            # Create a box
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=0.1, height=0.1, depth=0.1)
        elif rand_n == 2:
            # Create a cylinder
            mesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.05, height=0.2)

        mesh.translate(np.random.rand(3))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(
            np.random.rand(3) * np.pi))
        combined_mesh += mesh

    # Set a uniform color for the mesh
    combined_mesh.paint_uniform_color((0.5, 0.5, 0.5))

    return combined_mesh


# Example usage
if __name__ == "__main__":
    # Create sample point cloud
    print("Creating sample point cloud...")
    mesh = o3d.read_triangle_mesh("mesh.stl")
    o3d.visualization.draw_geometries([mesh])
    exit()

    pc = mesh.sample_points_uniformly(
        int(mesh.get_surface_area() * point_density * 1e6), use_triangle_normal=True)

    # Configure region growing
    config = FOVClusteringConfig()
    config.point_density = point_density  # Points per square millimeter
    config.fov_width = 50*2*0.001*np.sqrt(1/np.pi)
    config.fov_height = 50*2*0.001*np.sqrt(1/np.pi)
    config.dof = 1

    # Perform segmentation
    fc = FOVClustering(config)

    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)

    regions = fc.partition(points, normals, k)
    # Random colors for each region
    region_colors = np.random.rand(len(regions), 3)
    fov_cluster_meshes = [mesh]

    for i, region in enumerate(regions):
        region_pc = pc.select_by_index(regions[i])
        region_points = np.asarray(region_pc.points)
        region_normals = np.asarray(region_pc.normals)

        # FOV clustering
        region_fov_clusters = fc.fov_clustering(region_pc)
        for fov_cluster in region_fov_clusters:
            fov_cluster_pc = region_pc.select_by_index(fov_cluster)
            # Translate slightly by average normal to avoid overlapping
            avg_normal = np.mean(np.asarray(fov_cluster_pc.normals), axis=0)
            fov_cluster_pc.translate(avg_normal * 0.001)  # Translate by
            fov_cluster_mesh = fov_cluster_pc.compute_convex_hull(joggle_inputs=True)[
                0]
            # Slightly adjust color
            color = region_colors[i] + 0.1*(np.random.rand(3) - 0.5)
            # Ensure color values are between 0 and 1
            color = np.clip(color, 0, 1)
            fov_cluster_mesh.paint_uniform_color(
                color)  # Set color for the FOV cluster
            fov_cluster_meshes.append(fov_cluster_mesh)

    # Visualize the clusters
    print("Visualizing clusters...")
    o3d.visualization.draw_geometries(
        fov_cluster_meshes, window_name="FOV Clusters")
