import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import MaterialRecord


class Materials:

    background_color = [36/255, 37/255, 39/255, 1.0]

    panel_color = gui.Color(25/255, 25/255, 25/255, 0.8)

    mesh_material = rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    mesh_material.base_color = [0.1, 0.1, 0.1, 1.0]

    bounding_box_material = rendering.MaterialRecord()
    bounding_box_material.shader = "defaultUnlit"
    bounding_box_material.base_color = [0.8, 0.8, 0.8, 1.0]
    bounding_box_material.line_width = 2.0

    text_material = rendering.MaterialRecord()
    text_material.shader = "defaultUnlit"
    text_material.base_color = [1.0, 1.0, 1.0, 1.0]

    point_cloud_material = rendering.MaterialRecord()
    point_cloud_material.shader = "defaultUnlit"
    point_cloud_material.point_size = 7.0
    point_cloud_material.base_color = [0.7, 0.7, 0.7, 1.0]

    curvature_colormap = 'RdYlGn'
    regions_colormap = 'plasma'

    fov_cluster_material = rendering.MaterialRecord()
    fov_cluster_material.shader = "defaultUnlit"
    fov_cluster_material.base_color = [1.0, 1.0, 1.0, 1.0]

    region_view_material = rendering.MaterialRecord()
    region_view_material.shader = "defaultLitTransparency"
    region_view_material.base_color = [1.0, 1.0, 1.0, 0.25]

    viewpoint_material = rendering.MaterialRecord()
    viewpoint_material.shader = "defaultUnlit"
    viewpoint_material.base_color = [1.0, 1.0, 1.0, 1.0]

    fov_material = rendering.MaterialRecord()
    fov_material.shader = "defaultUnlit"
    fov_material.base_color = [1.0, 1.0, 1.0, 1.0]

    reticle_material = rendering.MaterialRecord()
    reticle_material.shader = "defaultUnlit"
    reticle_material.base_color = [204/255, 108/255, 231/255, 1.0]

    tabletop_material = rendering.MaterialRecord()
    tabletop_material.shader = "defaultLitTransparency"
    tabletop_material.base_color = [1.0, 1.0, 1.0, 0.75]
    tabletop_diameter = 100.0  # Diameter in mm
    tabletop_thickness = 2.0  # Thickness in mm
