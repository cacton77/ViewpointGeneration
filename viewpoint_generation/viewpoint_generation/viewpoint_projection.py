import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pytransform3d.rotations as pr
from bayes_opt import BayesianOptimization


@dataclass
class ViewpointProjectionConfig:

    focal_distance: float = 0.3
    hemisphere_points: int = 10000


class ViewpointProjection:
    def __init__(self, config: ViewpointProjectionConfig):
        self.config = config
        self.raycasting_scene = o3d.t.geometry.RaycastingScene()

    def set_mesh(self, mesh: o3d.geometry.TriangleMesh):
        """
        Sets the mesh for the raycasting scene.
        """

        self.raycasting_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def generate_viewpoint(self, surface_points: list, surface_normals: list) -> list:
        """
        Projects the point cloud to a viewpoint based on the focal distance.
        """
        origin = np.mean(surface_points, axis=0)
        origin_normal = np.mean(surface_normals, axis=0)
        # Normalize the average normal vector
        surface_normal = origin_normal / np.linalg.norm(origin_normal)

        # Project point along the surface normal
        viewpoint, orientation = self.project_point_along_direction(
            origin, surface_normal)

        # Check occlusion
        distances, occluded = self.check_occlusion(viewpoint, surface_points)

        # return origin, viewpoint, direction, orientation

    def project_point_along_direction(self, origin: list, direction: list) -> list:
        """
        Projects a point along its normal direction by a certain distance.
        """
        # Apply a simple translation to simulate projection
        translation = np.array(direction) * self.config.focal_distance
        viewpoint = np.array(origin) + translation
        direction = viewpoint - origin
        direction = direction / np.linalg.norm(direction)
        principal_axis = -direction

        # Calculate orientation as a quaternion
        z = np.array([0, 0, 1])
        z_hat = principal_axis
        x_hat = np.cross(z_hat, z)
        if np.linalg.norm(x_hat) < 1e-6:
            x_hat = np.array([1, 0, 0])
        x_hat = x_hat / np.linalg.norm(x_hat)
        y_hat = np.cross(z_hat, x_hat)

        R = np.array([x_hat, y_hat, z_hat]).T

        quat_wxyz = pr.quaternion_from_matrix(R)
        # Convert to ROS quaternion format
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        orientation = quat_xyzw

        return viewpoint, orientation

    def check_occlusion(self, viewpoint: list, surface_points: list):
        """
        Checks if a point is occluded in the raycasting scene.
        """
        # If viewpoint z value < 0 return occluded
        # if viewpoint[2] < 0:
        # return True

        rays = np.zeros((len(surface_points), 6), dtype=np.float32)
        distances = [0]*len(surface_points)
        for i, point in enumerate(surface_points):
            rays[i, :3] = viewpoint
            distances[i] = np.linalg.norm(point - viewpoint)
            rays[i, 3:] = (point - viewpoint) / \
                distances[i] if distances[i] > 0 else np.zeros(3)

        intersections_dict = self.raycasting_scene.list_intersections(rays)

        ray_idx = -1
        occluded = [False]*len(surface_points)
        hit_distances = [0]*len(surface_points)
        for i in range(len(intersections_dict['ray_ids'])):
            if intersections_dict['ray_ids'][i] == ray_idx:
                continue
            else:
                ray_idx = intersections_dict['ray_ids'][i].numpy()
                hit_distance = intersections_dict['t_hit'][i].numpy()
                hit_distances[ray_idx] = hit_distance
                if abs(distances[ray_idx] - hit_distance) > 1e-6:
                    occluded[ray_idx] = True

        # Print percentage of occluded points
        occluded_percentage = np.sum(occluded) / len(occluded) * 100
        print(f"Occluded points: {occluded_percentage:.2f}%")

        return distances, occluded


if __name__ == "__main__":
    mesh_name = "mesh"
    mesh_name = "c"
    mesh = o3d.io.read_triangle_mesh(f"{mesh_name}.stl")
    interior_mesh = o3d.io.read_triangle_mesh(f"{mesh_name}_surface.stl")
    point_cloud = interior_mesh.sample_points_poisson_disk(
        number_of_points=2000, init_factor=5, use_triangle_normal=True)
    point_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Paint point cloud gray

    config = ViewpointProjectionConfig()
    config.focal_distance = 3.0
    config.hemisphere_points = 10000
    vp = ViewpointProjection(config)
    vp.set_mesh(mesh)

    surface_points = point_cloud.points
    surface_normals = point_cloud.normals

    origin = np.mean(surface_points, axis=0)
    origin_normal = np.mean(surface_normals, axis=0)
    # Normalize the average normal vector
    surface_normal = origin_normal / np.linalg.norm(origin_normal)

    # Project point along the surface normal
    viewpoint, orientation = vp.project_point_along_direction(
        origin, surface_normal)
    viewpoint_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    viewpoint_mesh.translate(viewpoint)
    viewpoint_mesh.paint_uniform_color([1, 0, 0])  # Paint red

    # Check occlusion
    distances, occluded = vp.check_occlusion(viewpoint, surface_points)

    # Paint occluded points red and non-occluded points green
    for i, point in enumerate(surface_points):
        if occluded[i]:
            point_cloud.colors[i] = [1, 0, 0]  # Red
        else:
            point_cloud.colors[i] = [0, 1, 0]  # Green

    z_hat = surface_normal
    x_hat = np.cross(np.array([0, 0, 1]), z_hat) if np.linalg.norm(
        np.cross(np.array([0, 0, 1]), z_hat)) > 0 else np.array([1.0, 0.0, 0.0])
    print(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat /= np.linalg.norm(y_hat)
    z_hat /= np.linalg.norm(z_hat)
    tf = np.eye(4)
    tf[:3, 0] = x_hat
    tf[:3, 1] = y_hat
    tf[:3, 2] = z_hat
    tf[:3, 3] = origin
    origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    origin_mesh.transform(tf)

    # Create points in positive z hemisphere
    radius = config.focal_distance
    num_points = config.hemisphere_points
    # Sample points in phi and theta randomly
    phi = np.random.rand(num_points) * (np.pi/2)
    theta = np.random.rand(num_points) * (2*np.pi)
    points = []
    for p, t in zip(phi, theta):
        x = radius * np.sin(p) * np.cos(t)
        y = radius * np.sin(p) * np.sin(t)
        z = radius * np.cos(p)
        point = [x, y, z]
        transformed_point = point @ tf[:3, :3].T + tf[:3, 3]
        # if transformed_point[2] < 0:
        #     continue
        points.append(transformed_point)
    hemisphere_points = np.array(points)
    # Transform points to origin coordinate frame

    # Create point cloud for visualization
    hemisphere_point_cloud = o3d.geometry.PointCloud()
    hemisphere_point_cloud.points = o3d.utility.Vector3dVector(
        hemisphere_points)
    hemisphere_point_cloud.paint_uniform_color([0, 1, 0])  # Paint blue
    # Create rays
    rays = np.zeros((len(hemisphere_points), 6), dtype=np.float32)
    for i, point in enumerate(hemisphere_points):
        rays[i, :3] = origin
        rays[i, 3:] = (point - origin) / \
            np.linalg.norm(
                point - origin) if np.linalg.norm(point - origin) > 0 else np.zeros(3)

    ray_line_set = o3d.geometry.LineSet()
    ray_line_set.points.append(origin)
    for i, point in enumerate(hemisphere_points):
        ray_line_set.points.append(point)
        ray_line_set.lines.append([0, i + 1])

    # Check for intersections with the scene
    intersections_dict = vp.raycasting_scene.list_intersections(rays)
    ray_idx = -1
    occluded = [False]*len(hemisphere_points)
    hit_distances = [0]*len(hemisphere_points)

    for i in range(len(intersections_dict['ray_ids'])):
        if intersections_dict['ray_ids'][i] == ray_idx:
            continue
        else:
            hit_distance = intersections_dict['t_hit'][i].numpy()
            if hit_distance == 0.0:
                continue
            ray_idx = intersections_dict['ray_ids'][i].numpy()
            # Check if the hit distance is less than the focal distance
            if hit_distance - config.focal_distance < -1e-2:
                hit_distances[ray_idx] = hit_distance
                occluded[ray_idx] = True
                # Paint occluded points red
                hemisphere_point_cloud.colors[ray_idx] = [1, 0, 0]

    # Copy non-occluded points to new point cloud
    non_occluded_phi = phi[~np.array(occluded)]
    non_occluded_theta = theta[~np.array(occluded)]
    non_occluded_points = hemisphere_points[~np.array(occluded)]
    non_occluded_colors = np.asarray(hemisphere_point_cloud.colors)[
        ~np.array(occluded)]
    non_occluded_point_cloud = o3d.geometry.PointCloud()
    non_occluded_point_cloud.points = o3d.utility.Vector3dVector(
        non_occluded_points)
    non_occluded_point_cloud.colors = o3d.utility.Vector3dVector(
        non_occluded_colors)
    # Run DBSCAN on non-occluded points
    eps = 8 * 0.89 * radius / np.sqrt(num_points)
    labels = np.array(non_occluded_point_cloud.cluster_dbscan(
        eps=eps, min_points=10, print_progress=True))
    if labels is None:
        labels = np.array([0])
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    non_occluded_point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries(
        [mesh, point_cloud, viewpoint_mesh, origin_mesh, hemisphere_point_cloud, ray_line_set, non_occluded_point_cloud])

    def evaluate_cost(phi, theta):
        viewpoint = generate_viewpoint(theta, phi)
        # Cast rays from the viewpoint to all surface points
        distances, occluded = vp.check_occlusion(viewpoint, surface_points)
        # Cost function for the viewpoint is a combination of standard deviation of distance to surface points
        # cost = np.std(np.array(distances)[~np.array(occluded)]) if np.all(
        # ~np.array(occluded)) else 0
        occluded_cost = np.mean(occluded)
        print(f"Occluded cost for viewpoint {viewpoint}: {occluded_cost}")
        deviation_cost = np.std(np.array(distances)[~np.array(
            occluded)]) if np.any(~np.array(occluded)) else 0
        print(f"Deviation cost for viewpoint {viewpoint}: {deviation_cost}")
        cost = 100*occluded_cost + 10*deviation_cost + phi
        return cost

    def generate_viewpoint(theta, phi):
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        viewpoint = np.array([x, y, z])
        viewpoint = viewpoint @ tf[:3, :3] + tf[:3, 3]
        return viewpoint

    def f(theta, phi):
        cost = evaluate_cost(phi, theta)
        return cost

    theta_res_list = []
    phi_res_list = []

    for label in range(max_label + 1):
        theta_init = np.mean(non_occluded_theta[labels == label])
        phi_init = np.mean(non_occluded_phi[labels == label])
        theta_max = np.max(non_occluded_theta[labels == label])
        theta_min = np.min(non_occluded_theta[labels == label])
        phi_max = np.max(non_occluded_phi[labels == label])
        phi_min = np.min(non_occluded_phi[labels == label])
        pbounds = {"theta": (theta_min, theta_max), "phi": (phi_min, phi_max)}

        optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            verbose=2,
            allow_duplicate_points=True,
        )
        optimizer.probe({"theta": theta_init, "phi": phi_init})
        optimizer.maximize(init_points=5, n_iter=10)

        # Store results
        theta_res = []
        phi_res = []
        for i, res in enumerate(optimizer.res):
            theta_res.append(res['params']['theta'])
            phi_res.append(res['params']['phi'])
        opt = optimizer.max
        theta_res.append(opt['params']['theta'])
        phi_res.append(opt['params']['phi'])
        theta_res_list.append(theta_res)
        phi_res_list.append(phi_res)

    # If colors.txt exists, load it
    if os.path.exists(f"{mesh_name}_colors.txt"):
        colors = np.loadtxt(f"{mesh_name}_colors.txt")
        non_occluded_phi = np.loadtxt(f"{mesh_name}_non_occluded_phi.txt")
        non_occluded_theta = np.loadtxt(f"{mesh_name}_non_occluded_theta.txt")
        labels = np.loadtxt(f"{mesh_name}_labels.txt")
    else:
        # Evaluate cost landscape
        costs = np.zeros(len(non_occluded_points))
        for i, viewpoint in enumerate(non_occluded_points):
            phi = non_occluded_phi[i]
            theta = non_occluded_theta[i]
            cost = evaluate_cost(phi, theta)
            costs[i] = cost
        # Normalize costs for coloring
        colors = plt.get_cmap('RdYlGn')(np.max(costs) - costs)
        # save colors to file
        np.savetxt(f"{mesh_name}_colors.txt", colors[:, :3])
        np.savetxt(f"{mesh_name}_non_occluded_phi.txt", non_occluded_phi)
        np.savetxt(f"{mesh_name}_non_occluded_theta.txt", non_occluded_theta)
        np.savetxt(f"{mesh_name}_labels.txt", labels)
    # non_occluded_point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Plot non-occluded phi and theta points colored based on labels
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(max_label + 1):
        if i == -1:
            continue
        # Get points in this cluster
        cluster_phi = non_occluded_phi[labels == i]
        cluster_theta = non_occluded_theta[labels == i]
        theta_res = theta_res_list[i]
        phi_res = phi_res_list[i]
        theta_opt = theta_res[-1]
        phi_opt = phi_res[-1]
        ax.scatter(cluster_theta, cluster_phi,
                   color=colors[labels == i], label=f"Cluster {i}", zorder=1)
        ax.plot(theta_res, phi_res, marker='o',
                color='black', linestyle='--', label="Optimization Path", zorder=2)
        ax.scatter(theta_opt, phi_opt, marker='*',
                   color='magenta', label="Optimal Viewpoint", s=200, zorder=3)
    ax.set_xlabel("Theta")
    ax.set_ylabel("Phi")
    ax.set_title("Non-occluded Points in Spherical Coordinates")
    ax.legend()
    # Add colorbar for cost
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Cost")
    plt.show()

    # Show optimal viewpoint in o3d
    optimal_viewpoint = generate_viewpoint(theta_opt, phi_opt)
    optimal_viewpoint_mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.05)
    optimal_viewpoint_mesh.translate(optimal_viewpoint)
    o3d.visualization.draw_geometries(
        [mesh, non_occluded_point_cloud, optimal_viewpoint_mesh])
