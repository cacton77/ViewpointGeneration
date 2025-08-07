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
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # Convert to ROS quaternion format
        orientation = quat_xyzw

        return origin, viewpoint, direction, orientation

    def check_occlusion(self, viewpoint: list, surface_points: list, raycasting_scene: o3d.t.geometry.RaycastingScene):
        """
        Checks if a point is occluded in the raycasting scene.
        """
        if not isinstance(raycasting_scene, o3d.t.geometry.RaycastingScene):
            raise TypeError(
                "Raycasting scene must be an instance of open3d.t.geometry.RaycastingScene")

        occluded = False

        for point in surface_points:
            ray = o3d.t.geometry.RaycastingScene.create_ray(
                viewpoint, point, max_distance=self.config.focal_distance)
            result = raycasting_scene.cast_rays(ray)

            if result.has_hit:
                occluded = True
                break

        return occluded
