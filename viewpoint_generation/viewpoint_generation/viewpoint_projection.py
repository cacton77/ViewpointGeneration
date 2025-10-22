import numpy as np
import open3d as o3d
from dataclasses import dataclass
import pytransform3d.rotations as pr


@dataclass
class ViewpointProjectionConfig:

    focal_distance: float = 0.3


class ViewpointProjection:
    def __init__(self, config: ViewpointProjectionConfig):
        self.config = config

    def project(self, surface_points: list, surface_normals: list) -> list:
        """
        Projects the point cloud to a viewpoint based on the focal distance.
        """
        origin = np.mean(surface_points, axis=0)
        origin_normal = np.mean(surface_normals, axis=0)
        # Normalize the average normal vector
        surface_normal = origin_normal / np.linalg.norm(origin_normal)
        # Apply a simple translation to simulate projection
        translation = np.array(surface_normal) * self.config.focal_distance
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

        return origin, viewpoint, direction, orientation

    def check_occlusion(self, viewpoint: list, surface_points: list, raycasting_scene: o3d.t.geometry.RaycastingScene):
        """
        Checks if a point is occluded in the raycasting scene.
        """
        if not isinstance(raycasting_scene, o3d.t.geometry.RaycastingScene):
            raise TypeError(
                "Raycasting scene must be an instance of open3d.t.geometry.RaycastingScene")

        # If viewpoint z value < 0 return occluded
        if viewpoint[2] < 0:
            return True

        rays = np.zeros((len(surface_points), 6), dtype=np.float32)
        distances = [0]*len(surface_points)
        for i, point in enumerate(surface_points):
            rays[i, :3] = viewpoint
            distances[i] = np.linalg.norm(point - viewpoint)
            rays[i, 3:] = (point - viewpoint) / \
                distances[i] if distances[i] > 0 else np.zeros(3)

        intersections_dict = raycasting_scene.list_intersections(rays)

        ray_idx = -1
        occluded = [False]*len(surface_points)
        for i in range(len(intersections_dict['ray_ids'])):
            if intersections_dict['ray_ids'][i] == ray_idx:
                continue
            else:
                ray_idx = intersections_dict['ray_ids'][i].numpy()
                hit_distance = intersections_dict['t_hit'][i].numpy()
                if abs(distances[ray_idx] - hit_distance) < 1e-6:
                    occluded[ray_idx] = True

        # Print percentage of occluded points
        occluded_percentage = np.sum(occluded) / len(surface_points) * 100
        print(f"Occluded points: {occluded_percentage:.2f}%")

        return occluded
