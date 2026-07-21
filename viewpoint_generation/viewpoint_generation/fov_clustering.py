import time
import threading
import math
import copy
import numpy as np
import open3d as o3d
from multiprocessing import Queue, Process
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
from itertools import permutations

from sklearn.cluster import KMeans  # . . . . . . . . K-means
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from open3d.geometry import PointCloud, TriangleMesh
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from dataclasses import dataclass

from viewpoint_generation.occlusion_search import search_hemisphere_direction


@dataclass
class FOVClusteringConfig:
    """Configuration parameters for region growing algorithms."""

    fov_diameter: float = 0.02  # Field of view height in m
    dof: float = 0.02  # Depth of field in m

    # Cost function parameters.
    # Also sizes each region's sampled point cloud (points/mm^2) — only needed
    # to resolve FOV coverage/packing decisions, not final inspection imagery,
    # so this can stay low (e.g. ~1mm point spacing is already far finer than
    # a typical fov_diameter).
    point_density: float = 0.5  # Points per square millimeter
    lambda_weight: float = 1.0  # Weight for point out percentage in cost function
    beta_weight: float = 1.0  # Weight for packing efficiency in cost function
    # Maximum allowed point out percentage for a valid cluster
    max_point_out_percentage: float = 0.001

    # Clustering parameters
    point_weight: float = 2.0  # Weight for point locations in clustering
    normal_weight: float = 1.0  # Weight for normals in clustering
    number_of_runs: int = 10  # Number of runs for KMeans
    maximum_iterations: int = 100  # Maximum iterations for KMeans

    # Algorithm selector
    algorithm: str = 'greedy_cover'  # 'kmeans' or 'greedy_cover'

    # Greedy set-cover parameters
    # Max incidence angle (rad) for photometric-stereo coverage
    fov_normal_threshold: float = math.pi / 4
    # Max incidence angle (rad) for standard imaging. Between
    # fov_normal_threshold and this angle a direction is valid for standard
    # (non-photometric) imaging; beyond it the view is too glancing to capture
    # surface information and is treated as inaccessible (neither sampled nor
    # considered valid).
    standard_normal_threshold: float = math.pi / 3
    # Anchor spacing (m); 0.0 = auto = fov_diameter/2
    candidate_spacing: float = 0.0
    prune_redundant: bool = True  # Drop redundant viewpoints after greedy cover
    rng_seed: int = 0  # Seed for reproducible candidate sampling

    # Occlusion (criterion 4 of the coverage predicate)
    occlusion_check: bool = True  # Require unobstructed line-of-sight to the full part mesh
    # Shrink margin (m) subtracted from the occlusion ray's tfar so a point's
    # own triangle is never mistaken for its own occluder.
    occlusion_epsilon: float = 1e-4

    # Blind-spot rescue: Monte Carlo hemisphere search for an alternative
    # viewing angle for points the straight-normal predicate leaves occluded.
    rescue_search: bool = True  # Attempt to rescue occluded blind-spot points
    rescue_samples: int = 64  # Hemisphere samples per rescue attempt

    # Candidate anchor generation: 'farthest_point' (default, greedy maximin
    # spacing) or structured. Structured lays candidates on a per-region
    # cylindrical grid (elevation = world Z, azimuth = angle around a
    # vertical axis through the part's center) instead of blue-noise
    # farthest-point sampling, so viewpoints tend to line up at shared
    # elevations/azimuths across regions -- at some cost to the greedy
    # algorithm's freedom to pick the most locally-efficient anchor.
    structured_candidates: bool = False

    def to_dict(self):
        return {
            "fov_diameter": {
                "value": self.fov_diameter,
                "type": "float",
                "description": "Field of view diameter in meters",
                "control": "slider",
                "range": [0.01, 0.05],
            },
            "dof": {
                "value": self.dof,
                "type": "float",
                "description": "Depth of field in meters",
                "control": "slider",
                "range": [0.001, 0.1],
            },
            "point_density": {
                "value": self.point_density,
                "type": "float",
                "description": "Points per square millimeter — sizes each region's sampled point cloud and the cluster-evaluation cost function",
                "control": "slider",
                "range": [0.01, 100.0],
            },
            "lambda_weight": {
                "value": self.lambda_weight,
                "type": "float",
                "description": "Weight for point-out percentage in the cost function",
                "control": "slider",
                "range": [0.0, 10.0],
            },
            "beta_weight": {
                "value": self.beta_weight,
                "type": "float",
                "description": "Weight for packing efficiency in the cost function",
                "control": "slider",
                "range": [0.0, 10.0],
            },
            "max_point_out_percentage": {
                "value": self.max_point_out_percentage,
                "type": "float",
                "description": "Maximum allowed point-out percentage for a valid cluster",
                "control": "slider",
                "range": [0.0, 1.0],
            },
            "point_weight": {
                "value": self.point_weight,
                "type": "float",
                "description": "Weight for point locations in k-means clustering",
                "control": "slider",
                "range": [0.0, 10.0],
            },
            "normal_weight": {
                "value": self.normal_weight,
                "type": "float",
                "description": "Weight for normals in k-means clustering",
                "control": "slider",
                "range": [0.0, 10.0],
            },
            "number_of_runs": {
                "value": self.number_of_runs,
                "type": "integer",
                "description": "Number of k-means runs (best result is kept)",
                "control": "slider",
                "range": [1, 100],
            },
            "maximum_iterations": {
                "value": self.maximum_iterations,
                "type": "integer",
                "description": "Maximum iterations per k-means run",
                "control": "slider",
                "range": [1, 1000],
            },
            "algorithm": {
                "value": self.algorithm,
                "type": "string",
                "description": "FOV clustering algorithm: 'kmeans' or 'greedy_cover'",
                "control": "dropdown",
            },
            "fov_normal_threshold": {
                "value": self.fov_normal_threshold,
                "type": "float",
                "description": "Max surface-normal incidence angle (radians) for photometric-stereo imaging (greedy_cover coverage predicate)",
                "control": "slider",
                "range": [0.0, math.pi],
            },
            "standard_normal_threshold": {
                "value": self.standard_normal_threshold,
                "type": "float",
                "description": "Max surface-normal incidence angle (radians) for standard imaging; beyond this the view is too glancing and is treated as inaccessible",
                "control": "slider",
                "range": [0.0, math.pi / 2],
            },
            "candidate_spacing": {
                "value": self.candidate_spacing,
                "type": "float",
                "description": "Greedy cover anchor spacing in meters (0.0 = auto = fov_diameter/2)",
                "control": "slider",
                "range": [0.0, 0.5],
            },
            "prune_redundant": {
                "value": self.prune_redundant,
                "type": "bool",
                "description": "Remove redundant viewpoints after greedy cover (preserves full coverage)",
                "control": "checkbox",
            },
            "rng_seed": {
                "value": self.rng_seed,
                "type": "integer",
                "description": "Random seed for greedy_cover candidate sampling (reproducibility)",
                "control": "slider",
                "range": [0, 2147483647],
            },
            "occlusion_check": {
                "value": self.occlusion_check,
                "type": "bool",
                "description": "Require unobstructed line-of-sight to the full part mesh (criterion 4 of the greedy_cover coverage predicate)",
                "control": "checkbox",
            },
            "occlusion_epsilon": {
                "value": self.occlusion_epsilon,
                "type": "float",
                "description": "Shrink margin (m) subtracted from the occlusion ray's tfar so a point's own triangle is never mistaken for its own occluder",
                "control": "slider",
                "range": [0.0, 0.01],
            },
            "rescue_search": {
                "value": self.rescue_search,
                "type": "bool",
                "description": "Attempt a Monte Carlo hemisphere search for an alternative viewing angle before giving up on an occluded blind-spot point",
                "control": "checkbox",
            },
            "rescue_samples": {
                "value": self.rescue_samples,
                "type": "integer",
                "description": "Hemisphere samples per blind-spot rescue attempt",
                "control": "slider",
                "range": [4, 1000],
            },
            "structured_candidates": {
                "value": self.structured_candidates,
                "type": "bool",
                "description": "Lay candidate anchors on a per-region cylindrical grid (elevation/azimuth around the part center) instead of farthest-point sampling",
                "control": "checkbox",
            },
        }


def _farthest_point_sample(pts: np.ndarray, spacing: float, rng: np.random.Generator) -> list:
    """Greedy maximin farthest-point sampling.

    Returns a list of indices from pts such that every selected point is at
    least `spacing` from all previously selected points (approximately — the
    first point is chosen randomly and subsequent points maximise the minimum
    distance to the selected set).  Always returns at least one index.
    """
    m = len(pts)
    if m == 0:
        return []
    start = int(rng.integers(0, m))
    selected = [start]
    if m == 1:
        return selected
    min_dists = np.linalg.norm(pts - pts[start], axis=1)
    min_dists[start] = 0.0
    while True:
        farthest_dist = float(np.max(min_dists))
        if farthest_dist < spacing:
            break
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        dists_to_new = np.linalg.norm(pts - pts[farthest], axis=1)
        np.minimum(min_dists, dists_to_new, out=min_dists)
        min_dists[farthest] = 0.0
    return selected


def _structured_grid_sample(pts: np.ndarray, spacing: float, part_center_xy: tuple) -> list:
    """Structured cylindrical-grid candidate anchors, scoped to one region.

    Bins this region's points by (elevation, azimuth) in a frame shared
    across the whole part -- elevation = world Z, azimuth = angle around a
    vertical axis through `part_center_xy` -- so bins line up consistently
    across different regions rather than each region getting its own
    arbitrary local frame. Each occupied bin keeps one representative point
    as a candidate anchor. `spacing` sets both the elevation ring spacing
    (meters) and, via this region's mean radius from the vertical axis, the
    azimuthal spacing (converted from an arc-length target to an angular
    step). Always returns at least one index for a non-empty `pts`.

    Regions whose mean radius from the axis is smaller than `spacing` are
    too close to it for azimuthal binning to be meaningful (the "pole"
    case) -- those fall back to elevation-only bands.

    Each bin keeps whichever of its points is closest to the bin's exact
    center (in elevation / arc-length units), not just the first point
    encountered -- otherwise the chosen anchor could land anywhere within a
    `spacing`-wide bin, undermining the point of sharing bin edges across
    regions: two regions occupying the same bin should end up with anchors
    that actually agree on elevation/azimuth, not just on which coarse
    bucket they fell into.
    """
    m = len(pts)
    if m == 0:
        return []

    cx, cy = part_center_xy
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    mean_radius = float(np.hypot(dx, dy).mean())

    z_bin = np.round(pts[:, 2] / spacing).astype(np.int64)
    z_offset = pts[:, 2] - z_bin * spacing
    if mean_radius < spacing:
        theta_bin = np.zeros(m, dtype=np.int64)
        arc_offset = np.zeros(m)
    else:
        theta = np.arctan2(dy, dx)
        angular_step = spacing / mean_radius
        theta_bin = np.round(theta / angular_step).astype(np.int64)
        theta_offset = theta - theta_bin * angular_step
        theta_offset = (theta_offset + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
        arc_offset = theta_offset * mean_radius
    dist_to_center = np.hypot(z_offset, arc_offset)

    keys = np.stack([z_bin, theta_bin], axis=1)
    _, bin_of = np.unique(keys, axis=0, return_inverse=True)
    # Sort by (bin, distance-to-center) so the first row of each bin group
    # is that bin's closest-to-center point, then take one per bin.
    order = np.lexsort((dist_to_center, bin_of))
    _, first_pos = np.unique(bin_of[order], return_index=True)
    return order[first_pos].tolist()


class FOVClustering:
    def __init__(self, config: FOVClusteringConfig = None):
        self.config = config or FOVClusteringConfig()
        self.raycasting_scene = None

    def set_scene(self, raycasting_scene: 'o3d.t.geometry.RaycastingScene'):
        """Share a RaycastingScene built once from the full part mesh (by
        ViewpointGeneration.set_mesh_file, alongside ViewpointProjection) so
        occlusion checks in greedy_cover_clustering see the whole part, not
        just the region being clustered."""
        self.raycasting_scene = raycasting_scene

    def evaluate_fov_cluster(self, points, normals):
        """ Function to evaluate a cluster based on field of view and depth of field. """
        # Check for minimum number of points
        if points.shape[0] < 4:
            print("Not enough points in the cluster to evaluate.")
            return False, 0, 0, False

        # Convert points and normals to Open3D PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.normals = o3d.utility.Vector3dVector(normals)

        z = np.array([0, 0, 1])
        z_hat = np.average(normals, axis=0)
        x_hat = np.cross(z, z_hat) if np.linalg.norm(
            np.cross(z, z_hat)) != 0 else np.array([1, 0, 0])
        y_hat = np.cross(z_hat, x_hat)
        x_hat = x_hat / np.linalg.norm(x_hat)
        y_hat = y_hat / np.linalg.norm(y_hat)
        z_hat = z_hat / np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))
        t = point_cloud.get_center()

        camera_radius = self.config.fov_diameter / 2

        # Visualize
        obb = point_cloud.get_minimal_oriented_bounding_box(robust=True)
        camera = o3d.geometry.TriangleMesh.create_cylinder(
            radius=camera_radius, height=2*self.config.dof)
        # Remove top and bottom faces of the cylinder
        camera = camera.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-camera_radius, -camera_radius, -self.config.dof/2],
            max_bound=[camera_radius, camera_radius, self.config.dof/2]))
        camera.paint_uniform_color([1, 0, 0])  # Red color for camera
        obb.color = (0, 1, 0)
        point_cloud.rotate(np.linalg.inv(R), t)
        point_cloud.translate(-t)
        obb.rotate(np.linalg.inv(R), t)
        obb.translate(-t)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([point_cloud, obb, camera, axis], mesh_show_back_face=True)

        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        # Convert to square millimeters
        max_points_in = (np.pi * camera_radius**2) * \
            (self.config.point_density * 1e6)

        # Get height and radial coordinates
        height_coords = points[:, 2]

        # Get indices for the two radial dimensions
        radial_indices = [i for i in range(3) if i != 2]
        radial_coords = points[:, radial_indices]

        # Check constraints
        height_mask = (height_coords >= 0) & (height_coords <= self.config.dof)
        radial_distances_sq = np.sum(radial_coords**2, axis=1)
        extreme_radial_distance = np.max(
            np.sqrt(radial_distances_sq)) if radial_distances_sq.size > 0 else 0
        radius_mask = radial_distances_sq <= camera_radius**2

        # TODO: figure out whether to use height mask or not points_in = np.sum(height_mask & radius_mask)
        points_in = np.sum(radius_mask)
        points_in_xy = np.sum(radius_mask)
        points_out = points.shape[0] - points_in

        point_out_percentage = points_out / \
            points.shape[0] if points.shape[0] > 0 else 0
        packing_efficiency = points_in_xy / max_points_in if max_points_in > 0 else 0

        if point_out_percentage > self.config.max_point_out_percentage:
            valid = False
        else:
            valid = True

        borderline = False
        epsilon = 0.05
        if (extreme_radial_distance < (camera_radius*(1+epsilon)) and
                extreme_radial_distance > (camera_radius*(1-epsilon))) or (
                extreme_radial_distance < (camera_radius*(1+epsilon)) and
                extreme_radial_distance > (camera_radius*(1-epsilon))) and valid == True:

            borderline = True

        print(
            f'Points in: {points_in}, Points out: {points_out}, Point out percentage: {point_out_percentage}')
        print(f'Packing efficiency: {packing_efficiency}')

        return valid, point_out_percentage, packing_efficiency, borderline

    def partition(self, points, normals, k) -> list:
        """ K-Means clustering function. """

        if (self.config.normal_weight != 0):
            # Scale point locations to lie between [-1, 1]
            points = 2 * (minmax_scale(points) - 0.5)

        # Combine weighted vertex and normal data
        data = np.concatenate((self.config.point_weight * points,
                               self.config.normal_weight * normals), axis=1)

        # Scikit Learn KMeans
        KM = KMeans(init='k-means++',
                    n_clusters=k,
                    n_init=self.config.number_of_runs,
                    max_iter=self.config.maximum_iterations)
        KM.fit(data)

        labels = KM.labels_
        clusters = [[] for i in range(k)]

        for j in range(len(labels)):
            clusters[labels[j]].append(j)

        return clusters

    def evaluate_k(self, points, normals, k, eval_fun, tries=1):
        """ Run multiple k-means partitions to determine if current k is valid. """
        for i in range(tries):
            clusters = self.partition(points, normals, k)
            k_valid = True
            for j, cluster in enumerate(clusters):
                cluster_valid, cost, _ = eval_fun(cluster)
                k_valid = k_valid and cluster_valid
                if not cluster_valid:
                    break
            if k_valid:
                return True, clusters, cost
        return False, clusters, cost

    def evaluate_k_cost(self, points, normals, k, eval_fun, tries=1):
        """ Evaluate the cost of a given k value for clustering. """
        # this is to make sure that k is always an integer and minimum value of k is 1
        k = max(1, min(int(k), len(points)))

        for i in range(tries):
            clusters = self.partition(points, normals, k)
            k_valid = True
            total_cost = 0
            total_count = len(clusters)
            total_point_out_percentage = 0
            total_packing_efficiency = 0
            anyborderline = False

            for j, cluster in enumerate(clusters):
                cluster_points = points[cluster, :]
                cluster_normals = normals[cluster, :]

                cluster_valid, point_out_percentage, packing_efficiency, borderline = eval_fun(
                    copy.deepcopy(cluster_points), copy.deepcopy(cluster_normals))

                total_point_out_percentage += point_out_percentage
                total_packing_efficiency += packing_efficiency
                anyborderline = anyborderline or borderline
                k_valid = k_valid and cluster_valid

            total_point_out_percentage = total_point_out_percentage/total_count
            total_packing_efficiency = total_packing_efficiency/total_count

            if (total_point_out_percentage > 0.001):
                # Remove packing efficiency from cost if point out percentage is greater than 0.001
                s = 0
            else:
                print("total_point_out_percentage is zero")
                s = 1
                if (anyborderline == True):
                    s = 0
                # print(total_packing_eff)

            total_cost = (self.config.lambda_weight)*total_point_out_percentage + \
                s*((1/total_packing_efficiency)**self.config.beta_weight)

        return -total_cost

    def optimize_k_b_opt(self, points, normals, eval_fun) -> int:

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)

        temp_mesh = temp_pcd.compute_convex_hull(joggle_inputs=True)[0]
        # calculate the bounding box of the mesh
        bounding_box = temp_mesh.get_axis_aligned_bounding_box()
        bounding_box.color = (0, 1, 0)

        # Show the temp_pcd, temp_mesh, and bounding_box
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([temp_pcd, temp_mesh, bounding_box, axis])

        # calculate the length and width of the bounding box
        length = bounding_box.get_max_bound(
        )[0]-bounding_box.get_min_bound()[0]
        width = bounding_box.get_max_bound(
        )[1]-bounding_box.get_min_bound()[1]

        # check if the length is greater than the width
        greater_dimension = max(length, width)
        ratio = max(length, width)/min(length, width)

        # camera_area = camera_width*camera_height
        camera_r = self.config.fov_diameter/2
        camera_area = np.pi*(camera_r**2)
        surface_area = temp_mesh.get_surface_area()/2

        n_points = len(points)

        # Check for long and narrow regions.
        # If the ratio is greater than 100, we assume that the region is long and narrow.
        # It may be worth trying different cases for different bounds.
        if ratio > 100:
            k_min = greater_dimension/(2*camera_r)
            k_lo = max(1.0, k_min / 2)
            k_hi = min(3.0 * (k_min / 2), float(n_points))
        else:
            k_min = surface_area/camera_area
            k_lo = max(1.0, k_min)
            k_hi = min(3.0 * k_min, float(n_points))

        # Ensure valid bounds: both within [1, n_points] and lo < hi.
        # k_hi can fall below 1.0 for very small regions relative to the FOV,
        # causing inverted bounds that the Bayesian optimizer cannot handle.
        k_lo = max(1.0, min(k_lo, float(n_points - 1)))
        k_hi = max(k_lo + 1.0, min(k_hi, float(n_points)))

        pbounds = {"k": (k_lo, k_hi)}

        def f(k): return self.evaluate_k_cost(
            points, normals, k=k, eval_fun=eval_fun)

        optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            verbose=2,  # verbose=1 prints only at max, verbose=0 is silent
            random_state=1,
        )
        acq_function = UtilityFunction(kind="ei", kappa=5)
        optimizer.maximize(
            init_points=1,
            n_iter=3,
        )
        y = optimizer.max
        k_opt = max(1, int(y["params"]["k"]))
        valid_clusters = self.partition(points, normals, k_opt)
        non_valid_pcd = 0
        total_count = 0
        total_point_out_percentage = 0
        total_packing_eff = 0

        for j, cluster in enumerate(valid_clusters):
            cluster_valid, point_out_percentage, packing_eff, borderline = eval_fun(
                points, normals)
            total_point_out_percentage += point_out_percentage
            total_packing_eff += packing_eff
            total_count += 1
            # print(f'k-{k} pcd {j}: {cluster_valid}')
            if not cluster_valid:
                non_valid_pcd += 1

        avg_point_out_percentage = total_point_out_percentage/total_count
        avg_packing_eff = total_packing_eff/total_count

        return k_opt, valid_clusters

    def fov_clustering(self, point_cloud, focal_distance=None, part_center_xy=None):
        """ Partition a region of a point cloud into regions within camera fov and dof.
        input: point cloud of a region (with normals) Open3D PointCloud
        focal_distance: camera focal distance (m), from ViewpointProjectionConfig.
            Only consumed by greedy_cover (needed for the occlusion ray's
            camera position); ignored by the kmeans path.
        part_center_xy: (x, y) of the whole part's center, from
            ViewpointGeneration. Only consumed by greedy_cover when
            cfg.structured_candidates is True (the shared reference point
            for the cylindrical elevation/azimuth candidate grid, so
            different regions' bins line up); ignored otherwise.
        Returns: (clusters, blind_spots).
            clusters is a list of {'points': [...], 'axis': [...]|None}
            dicts. 'axis' is the owning candidate's occlusion-validated
            view axis for greedy_cover clusters, used by project_viewpoints
            as a guaranteed-good starting point; always None for kmeans,
            which never validates occlusion during clustering.
            blind_spots is {'points': [...]} and, when non-empty, also
            {'reason': 'occluded'|'geometric'|'mixed'} — points the greedy
            safety break (even after blind-spot rescue) could not assign to
            any cluster (kmeans always returns {'points': []}, since it has
            no coverage predicate to fail). """

        if self.config.algorithm == 'greedy_cover':
            return self.greedy_cover_clustering(point_cloud, focal_distance, part_center_xy)

        # Default: K-means + Bayesian optimization path
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        k_opt, fov_clusters = self.optimize_k_b_opt(
            points, normals, self.evaluate_fov_cluster)

        return [{'points': c, 'axis': None} for c in fov_clusters], {'points': []}

    def greedy_cover_clustering(self, point_cloud, focal_distance=None, part_center_xy=None):
        """Greedy set-cover FOV clustering.

        Replaces the K-means+BO path when algorithm=='greedy_cover'.
        Returns (clusters, blind_spots): clusters is the same container the
        K-means path returns (a list of clusters, each a list of LOCAL point
        indices into the sub-cloud); blind_spots is {'points': [...]} (plus
        'reason' when non-empty) — points the 5.4 safety break left uncovered,
        never assigned to any cluster.
        """
        pts = np.asarray(point_cloud.points)     # (m, 3)
        normals = np.asarray(point_cloud.normals)  # (m, 3)
        m = len(pts)

        if m == 0:
            return [], {'points': []}

        cfg = self.config
        fov_radius = cfg.fov_diameter / 2.0
        half_dof = cfg.dof / 2.0
        spacing = cfg.candidate_spacing if cfg.candidate_spacing > 0.0 else fov_radius
        cos_thr = math.cos(cfg.fov_normal_threshold)
        rng = np.random.default_rng(cfg.rng_seed)

        # Normalize normals defensively
        magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes == 0.0, 1.0, magnitudes)
        normals = normals / magnitudes

        # Build KD-tree on the region sub-cloud
        kdtree = cKDTree(pts)
        broad_radius = math.sqrt(fov_radius**2 + half_dof**2)

        occlusion_active = (cfg.occlusion_check
                            and self.raycasting_scene is not None
                            and focal_distance is not None)

        def _occluded_mask(cam, target_idx):
            """Batched any-hit line-of-sight test from cam to each target
            point. target_idx: array of local ids that already passed
            criteria 1-3. Returns a bool array, True where the ray is
            blocked before reaching the target (i.e. NOT visible).

            RaycastingScene.test_occlusions only accepts a single scalar
            tfar for the whole batch, not one per ray, so a per-ray
            "stop just short of this ray's own target" bound (the point of
            occlusion_epsilon) can't be expressed via tfar directly. Instead
            each ray's direction vector is left UN-normalized and scaled to
            length (dist - epsilon) — Open3D's ray convention measures hit
            distance in units of the direction vector, so this makes a
            single scalar tfar=1.0 bound every ray to just short of its own
            target in one batched call.
            """
            if len(target_idx) == 0:
                return np.zeros(0, dtype=bool)
            target_pts = pts[target_idx]                        # (k, 3)
            d = target_pts - cam                                 # (k, 3)
            dist = np.linalg.norm(d, axis=1)
            shrink = np.clip(
                1.0 - cfg.occlusion_epsilon / np.maximum(dist, 1e-12), 0.0, 1.0)
            dirs = d * shrink[:, None]
            rays_np = np.concatenate(
                [np.tile(cam, (len(target_idx), 1)), dirs], axis=1).astype(np.float32)
            rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
            return self.raycasting_scene.test_occlusions(rays, tfar=1.0).numpy()

        # Union of local ids that pass criteria 1-3 for ANY candidate,
        # regardless of occlusion outcome. Used only to classify blind spots
        # (Section 8: occluded vs. geometrically unreachable) — cheap to
        # accumulate as a byproduct of coverage computation.
        covered_union_geo = set()

        def _coverage_set(center, axis, force_include=None):
            """Return set of local indices covered by a candidate at
            (center, axis): criteria 1-3 (DoF/FOV/incidence) first to shrink
            the neighborhood, then criterion 4 (occlusion) as one batched
            any-hit query over just those survivors.

            force_include: a local point index whose incidence check
            (criterion 3 only — DoF/FOV/occlusion still apply) is skipped.
            Used by the blind-spot rescue pass: that point's axis was
            deliberately chosen via hemisphere search specifically to see
            it, possibly outside the photometric incidence cone (a
            'standard imaging' rather than 'photometric' viewpoint)."""
            candidates = kdtree.query_ball_point(center, r=broad_radius)
            if not candidates:
                return set()
            idx = np.array(candidates, dtype=np.intp)
            d = pts[idx] - center                              # (k, 3)
            axial = d @ axis                                   # (k,)
            lateral = np.linalg.norm(d - np.outer(axial, axis), axis=1)  # (k,)
            incidence = normals[idx] @ axis                    # (k,)
            incidence_ok = incidence >= cos_thr
            if force_include is not None:
                incidence_ok = incidence_ok | (idx == force_include)
            mask = (np.abs(axial) <= half_dof) & (
                lateral <= fov_radius) & incidence_ok
            survivors = idx[mask]
            covered_union_geo.update(survivors.tolist())
            if len(survivors) == 0 or not occlusion_active:
                return set(survivors.tolist())
            cam = center + focal_distance * axis
            blocked = _occluded_mask(cam, survivors)
            return set(survivors[~blocked].tolist())

        # --- 5.1  Candidate anchors: structured cylindrical grid (if
        # enabled and a shared part-center reference was given) or the
        # default blue-noise farthest-point sampling ---
        if cfg.structured_candidates and part_center_xy is not None:
            anchor_ids = _structured_grid_sample(pts, spacing, part_center_xy)
        else:
            anchor_ids = _farthest_point_sample(pts, spacing, rng)
        cand_centers = [pts[i] for i in anchor_ids]
        cand_axes = [normals[i] for i in anchor_ids]
        coverages = [_coverage_set(c, a)
                     for c, a in zip(cand_centers, cand_axes)]

        # --- 5.3  Coverability guarantee: self-anchor any uncovered point ---
        covered_union = set().union(*coverages) if coverages else set()
        for p in range(m):
            if p not in covered_union:
                c, a = pts[p], normals[p]
                cand_centers.append(c)
                cand_axes.append(a)
                coverages.append(_coverage_set(c, a))

        # --- 5.4  Greedy forward set cover ---
        uncovered = set(range(m))
        chosen = []
        n_cands = len(coverages)
        while uncovered:
            best = max(range(n_cands), key=lambda c: len(
                coverages[c] & uncovered))
            gain = coverages[best] & uncovered
            if not gain:
                break   # remaining points intrinsically uncoverable — can now
                        # trigger for real on occluded points (Section 5.4)
            chosen.append(best)
            uncovered -= gain

        # --- 5.4.5  Blind-spot rescue: Monte Carlo hemisphere search for an
        # alternative viewing angle, for points the straight-normal
        # predicate above left occluded. Tried once per point (skipping any
        # already covered incidentally by an earlier rescue this pass); a
        # successful rescue is added as a new candidate via force_include so
        # criteria 1/2/4 still apply normally, only criterion 3 (incidence)
        # is skipped for this specific point. Does not track/report tier —
        # that's authoritatively (re)determined later at projection time,
        # where the emitted direction and full cluster are both known. ---
        if occlusion_active and cfg.rescue_search and uncovered:
            for p in sorted(uncovered):
                if p not in uncovered:
                    continue  # already incidentally rescued this pass
                axis, tier, vf = search_hemisphere_direction(
                    self.raycasting_scene, pts[p], normals[p], pts[p:p + 1],
                    cfg.fov_normal_threshold, cfg.standard_normal_threshold,
                    focal_distance, cfg.occlusion_epsilon,
                    cfg.rescue_samples, rng)
                if axis is None or vf <= 0.0:
                    continue  # still genuinely blind
                cov = _coverage_set(pts[p], axis, force_include=p)
                if p not in cov:
                    continue
                new_idx = len(coverages)
                cand_centers.append(pts[p])
                cand_axes.append(axis)
                coverages.append(cov)
                chosen.append(new_idx)
                uncovered -= cov

        blind_points = sorted(uncovered)
        if blind_points:
            bs_set = set(blind_points)
            if bs_set <= covered_union_geo:
                reason = 'occluded'
            elif bs_set.isdisjoint(covered_union_geo):
                reason = 'geometric'
            else:
                reason = 'mixed'
            blind_spots = {'points': blind_points, 'reason': reason}
        else:
            blind_spots = {'points': []}

        # --- 5.5  Redundancy prune (AFTER coverage is guaranteed) ---
        if cfg.prune_redundant and len(chosen) > 1:
            for c in list(reversed(chosen)):
                if len(chosen) == 1:
                    break
                others_union = set().union(
                    *(coverages[o] for o in chosen if o != c))
                if coverages[c] <= others_union:
                    chosen.remove(c)

        # --- 5.6  Disjoint assignment (blind-spot points are never assigned:
        # their camera can't actually see them, so crediting them to a
        # cluster would violate the occlusion-respected guarantee) ---
        assignment = {c: [] for c in chosen}
        assignable = set(range(m)) - uncovered
        for p in assignable:
            owners = [c for c in chosen if p in coverages[c]]
            def _sort_key(c):
                dot = float(np.dot(normals[p], cand_axes[c]))
                dot = max(-1.0, min(1.0, dot))
                angle = math.acos(dot)
                axial = abs(
                    float(np.dot(pts[p] - cand_centers[c], cand_axes[c])))
                return (angle, axial)
            owner = min(owners, key=_sort_key)
            assignment[owner].append(p)

        # --- 5.7  Emit (drop empty clusters), each tagged with its owning
        # candidate's occlusion-validated axis so project_viewpoints can
        # reuse it as a guaranteed-good starting point instead of
        # re-deriving an unvalidated one from the mean of assigned normals.
        clusters = [
            {'points': members, 'axis': cand_axes[owner].tolist()}
            for owner, members in assignment.items() if members
        ]
        return clusters, blind_spots


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
    k = 5
    point_density = 0.5
    mesh = create_sample_mesh(k)

    pc = mesh.sample_points_uniformly(
        int(mesh.get_surface_area() * point_density * 1e6), use_triangle_normal=True)

    # Configure region growing
    config = FOVClusteringConfig()
    config.point_density = point_density  # Points per square millimeter
    config.fov_diameter = 50*2*0.001*np.sqrt(1/np.pi)
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
        region_fov_clusters, _blind_spots = fc.fov_clustering(region_pc)
        for fov_cluster in region_fov_clusters:
            fov_cluster_pc = region_pc.select_by_index(fov_cluster['points'])
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
