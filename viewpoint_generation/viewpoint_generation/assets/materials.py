import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import MaterialRecord


class Materials:

    font_size = 13

    scene_background_color = [36/255, 37/255, 39/255, 1.0]

    panel_color = gui.Color(25/255, 25/255, 25/255, 0.8)
    panel_color = gui.Color(175/255, 175/255, 175/255, 0.8)
    panel_color = gui.Color(0.5, 0.5, 0.5, 0.8)
    panel_color = gui.Color(36/255, 36/255, 36/255, 0.9)
    panel_color = gui.Color(0.14, 0.14, 0.14, 0.5)
    collapsable_panel_color = gui.Color(0.4, 0.4, 0.4, 1.0)
    tab_control_background_color = gui.Color(0.1, 0.1, 0.1, 0.6)
    tab_control_background_color = gui.Color(36/255, 37/255, 39/255, 0.6)
    content_color = gui.Color(0.0, 1.0, 0.0, 0.8)
    footer_text_color = gui.Color(0.75, 0.75, 0.75, 1.0)
    footer_panel_color = gui.Color(0.0, 0.0, 0.0, 0.8)

    button_background_color = gui.Color(0.4, 0.4, 0.4, 1.0)
    go_button_background_color = gui.Color(0.2, 0.8, 0.2, 1.0)
    text_color = gui.Color(1., 1., 1., 1.0)
    text_edit_background_color = gui.Color(0.5, 0.5, 0.5, 1.0)

    mesh_material = rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    mesh_material.base_color = [0.8, 0.8, 0.8, 1.0]

    mesh_material_transparent = rendering.MaterialRecord()
    mesh_material_transparent.shader = "defaultLitTransparency"
    mesh_material_transparent.base_color = [0.8, 0.8, 0.8, 0.8]

    mesh_material_vertex_colors = rendering.MaterialRecord()
    mesh_material_vertex_colors.shader = "defaultLit"
    mesh_material_vertex_colors.base_color = [1.0, 1.0, 1.0, 1.0]

    mesh_material_vertex_colors_transparent = rendering.MaterialRecord()
    mesh_material_vertex_colors_transparent.shader = "defaultLitTransparency"
    mesh_material_vertex_colors_transparent.base_color = [1.0, 1.0, 1.0, 0.8]

    bounding_box_material = rendering.MaterialRecord()
    bounding_box_material.shader = "defaultLit"
    bounding_box_material.base_color = [0.8, 0.8, 0.8, 1.0]
    bounding_box_material.line_width = 2.0

    text_material = rendering.MaterialRecord()
    text_material.shader = "defaultUnlit"
    text_material.base_color = [1.0, 1.0, 1.0, 1.0]

    point_cloud_material = rendering.MaterialRecord()
    point_cloud_material.shader = "defaultUnlit"
    point_cloud_material.point_size = 7.0
    point_cloud_material.base_color = [1.0, 1.0, 1.0, 1.0]

    selected_point_cloud_material = rendering.MaterialRecord()
    selected_point_cloud_material.shader = "defaultUnlit"
    selected_point_cloud_material.point_size = 7.0
    selected_point_cloud_material.base_color = [0.0, 1.0, 0.0, 1.0]

    curvature_colormap = 'plasma'
    regions_colormap = 'BuPu'
    regions_colormap = 'cubehelix'
    regions_colormap = 'rainbow'

    cluster_material = rendering.MaterialRecord()
    cluster_material.shader = "defaultUnlit"
    cluster_material.base_color = [1.0, 1.0, 1.0, 1.0]

    selected_cluster_material = rendering.MaterialRecord()
    selected_cluster_material.shader = "defaultUnlit"
    selected_cluster_material.base_color = [0.0, 0.0, 1.0, 1.0]

    region_view_material = rendering.MaterialRecord()
    region_view_material.shader = "defaultLitTransparency"
    region_view_material.base_color = [1.0, 1.0, 1.0, 0.25]

    selected_region_view_material = rendering.MaterialRecord()
    selected_region_view_material.shader = "defaultLitTransparency"
    selected_region_view_material.base_color = [0.0, 1.0, 1.0, 0.25]

    viewpoint_material = rendering.MaterialRecord()
    viewpoint_material.shader = "defaultUnlit"
    viewpoint_material.base_color = [1.0, 1.0, 1.0, 1.0]
    viewpoint_type = "sphere"
    viewpoint_sphere_size = 3  # Size in mm
    viewpoint_axis_size = 20
    viewpoint_arrow_scale = 2  # Scale factor for the arrow

    selected_viewpoint_material = rendering.MaterialRecord()
    selected_viewpoint_material.shader = "defaultUnlit"
    selected_viewpoint_material.base_color = [0.0, 0.0, 1.0, 1.0]
    # selected_viewpoint_material.base_color = [193/255, 76/255, 61/255, 1.0]

    fov_material = rendering.MaterialRecord()
    fov_material.shader = "defaultUnlit"
    fov_material.base_color = [1.0, 1.0, 1.0, 1.0]

    tabletop_material = rendering.MaterialRecord()
    tabletop_material.shader = "defaultLitTransparency"
    tabletop_material.base_color = [1.0, 1.0, 1.0, 0.75]
    tabletop_diameter = 100.0  # Diameter in mm
    tabletop_thickness = 2.0  # Thickness in mm

    path_material = MaterialRecord()
    path_material.shader = 'unlitLine'
    path_material.line_width = 2.0
    path_material.base_color = [92/255, 140/255, 207/255, 1.0]
    path_material.base_color = [0.5, 0.5, 0.5, 1.0]

    selected_path_material = MaterialRecord()
    selected_path_material.shader = 'unlitLine'
    selected_path_material.line_width = 3.0
    selected_path_material.base_color = [0.0, 1.0, 1.0, 1.0]

    joint_path_material = MaterialRecord()
    joint_path_material.shader = 'unlitLine'
    joint_path_material.line_width = 2.5
    joint_path_material.base_color = [1.0, 0.2, 0.2, 1.0]

    joint_marker_material = rendering.MaterialRecord()
    joint_marker_material.shader = 'defaultUnlit'
    joint_marker_material.point_size = 7.0
    joint_marker_material.base_color = [1.0, 0.2, 0.2, 1.0]

    unreachable_marker_material = rendering.MaterialRecord()
    unreachable_marker_material.shader = 'defaultUnlit'
    unreachable_marker_material.point_size = 14.0
    unreachable_marker_material.base_color = [1.0, 0.85, 0.0, 1.0]

    ground_plane_material = MaterialRecord()
    ground_plane_material.shader = 'defaultLitTransparency'
    ground_plane_material.base_color = [0.0, 0.0, 0.0, 0.0]

    grid_line_material = MaterialRecord()
    grid_line_material.shader = 'unlitLine'
    grid_line_material.base_color = [0.5, 0.5, 0.5, 1.0]
    grid_line_material.line_width = 2.0

    axes_line_material = MaterialRecord()
    axes_line_material.shader = 'unlitLine'
    axes_line_material.base_color = [1.0, 1.0, 1.0, 1.0]
    axes_line_material.line_width = 3.0

    fov_cylinder_material = MaterialRecord()
    fov_cylinder_material.shader = 'unlitLine'
    fov_cylinder_material.base_color = [1.0, 1.0, 1.0, 1.0]
    fov_cylinder_material.line_width = 1.5

    selected_fov_cylinder_material = MaterialRecord()
    selected_fov_cylinder_material.shader = 'unlitLine'
    selected_fov_cylinder_material.base_color = [0.0, 1.0, 1.0, 1.0]
    selected_fov_cylinder_material.line_width = 2.5
