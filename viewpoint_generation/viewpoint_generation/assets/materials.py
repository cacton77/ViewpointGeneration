import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import MaterialRecord


class Materials:

    background_color = [36/255, 37/255, 39/255, 1.0]

    panel_color = gui.Color(25/255, 25/255, 25/255, 0.8)

    mesh_material = rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    mesh_material.base_color = [0.1, 0.1, 0.1, 1.0]

    point_cloud_material = rendering.MaterialRecord()
    point_cloud_material.shader = "defaultUnlit"
    point_cloud_material.point_size = 10.0
    point_cloud_material.base_color = [1.0, 1.0, 1.0, 1.0]

    curvature_colormap = 'RdYlGn'
    regions_colormap = 'plasma'

    fov_cluster_material = rendering.MaterialRecord()
    fov_cluster_material.shader = "defaultUnlit"
    fov_cluster_material.base_color = [1.0, 1.0, 1.0, 1.0]
    
