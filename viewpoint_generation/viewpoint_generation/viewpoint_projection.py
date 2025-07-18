import numpy as np
import open3d as o3d
from dataclasses import dataclass


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

        return origin, viewpoint, direction

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
