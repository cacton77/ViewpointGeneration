#!/usr/bin/env python3
from curses.panel import panel
import os
import rclpy
import sys
import time
import yaml
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from pprint import pprint

from viewpoint_generation.gui_node import ROSThread
from viewpoint_generation.assets.materials import Materials
from viewpoint_generation.visualizer import Visualizer, ClusterViewpointMode

sys.stdout.reconfigure(line_buffering=True)
isMacOS = sys.platform == 'darwin'
# o3d.visualization.webrtc_server.enable_webrtc()


class GUIClient():

    MENU_LOGO = 0
    MENU_OPEN = 1
    MENU_SAVE = 2
    MENU_SAVE_AS = 3
    MENU_IMPORT_MODEL = 4
    MENU_IMPORT_PCD = 5
    MENU_QUIT = 6
    MENU_NEW = 7
    MENU_UNDO = 8
    MENU_REDO = 9
    MENU_PREFERENCES = 10
    MENU_SHOW_AXES = 11
    MENU_SHOW_GRID = 12
    MENU_SHOW_MODEL_BB = 13
    MENU_SHOW_MESH = 15
    MENU_SHOW_POINT_CLOUD = 16
    MENU_SHOW_CURVATURES = 17
    MENU_SHOW_REGIONS = 18
    MENU_SHOW_NOISE_POINTS = 19
    MENU_SHOW_CLUSTERS = 20
    MENU_SHOW_VIEWPOINTS = 21
    MENU_SHOW_REGION_VIEW_MANIFOLDS = 22
    MENU_SHOW_SETTINGS = 23
    MENU_SHOW_ERRORS = 24
    MENU_SHOW_PATH = 25
    MENU_ABOUT = 26
    MENU_RENDER_CONVEX_HULL    = 27
    MENU_RENDER_CLUSTER_CLOUD  = 28
    MENU_RENDER_FRUSTUM        = 29
    MENU_RENDER_LINES          = 30
    MENU_RENDER_VIEWPOINT_ONLY = 31
    MENU_RENDER_ORIGIN_SPHERE  = 32

    camera_updated = False
    camera_fov_width = 0.03
    camera_fov_height = 0.02
    last_intersection_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    last_slider_value = 1

    def __init__(self):
        self.app = gui.Application.instance

        # Fonts
        deja_vu_sans = gui.FontDescription(
            typeface="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", style=gui.FontStyle.NORMAL, point_size=Materials.font_size)

        # Must be run before application.create_window()
        self.font_id_sans_serif = gui.Application.instance.add_font(
            deja_vu_sans)

        self.app.set_font(0, deja_vu_sans)

        self.window = self.app.create_window(
            "Viewpoint Generation", width=1620, height=1080, x=30, y=30)

        self.fps = 30

        em = self.window.theme.font_size
        r = self.window.content_rect
        self.menu_height = 2.5 * em
        self.header_height = 3 * em

        self.ros_thread = ROSThread()
        self.ros_thread.start()

        # Wait until self.ros_thread.parameters_dict is populated
        while any(value == {} for value in self.ros_thread.parameters_dict.values()):
            time.sleep(0.1)
        self.parameters_dict = self.ros_thread.expand_params_dict()

        w = self.window
        self.window.set_on_close(self.on_main_window_closing)
        self.update_delay = -1  # Set to -1 to use tick event
        self.window.set_on_tick_event(self.on_main_window_tick_event)

        # Register the layout callback BEFORE adding any widgets so that
        # Open3D's initial layout pass (fired when the first child is added)
        # already has a callback to invoke.  _on_layout guards against the
        # early calls that arrive before all attributes are initialised.
        self.window.set_on_layout(self._on_layout)

        self.init_main_layout()
        self.init_menu_bar()

        # 3D SCENE ################################################################
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.scene_widget.enable_scene_caching(False)

        self.scene_widget.scene.set_background(
            Materials.scene_background_color)

        def on_mouse(event):
            # Consume events that land on the side panels so they don't
            # reach the camera controller.
            for layout in [self.main_layout, self.action_layout]:
                frame = layout.frame
                if (frame.get_left() <= event.x <= frame.get_right() and
                        frame.get_top()  <= event.y <= frame.get_bottom()):
                    return gui.Widget.EventCallbackResult.CONSUMED
            return self.viz.on_mouse(event)

        self.scene_widget.set_on_mouse(on_mouse)

        self.window.add_child(self.scene_widget)

        # Create Visualizer — owns all scene geometry and rendering state
        self.viz = Visualizer(self.scene_widget)
        self.viz.setup_multi_directional_lighting()

        self.last_draw_time = time.time()


    def init_menu_bar(self):
        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        w = self.window
        isMacOS = sys.platform == 'darwin'

        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item(
                    "About", self.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item(
                    "Quit", self.MENU_QUIT)
            logo_menu = gui.Menu()
            file_menu = gui.Menu()
            file_menu.add_item(
                "New", self.MENU_NEW)
            file_menu.add_item(
                "Open...", self.MENU_OPEN)
            file_menu.add_separator()
            file_menu.add_item(
                "Save", self.MENU_SAVE)
            file_menu.add_item(
                "Save As...", self.MENU_SAVE_AS)
            file_menu.add_separator()
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item(
                    "Quit", self.MENU_QUIT)
            edit_menu = gui.Menu()
            edit_menu.add_item("Undo", self.MENU_UNDO)
            edit_menu.add_item("Redo", self.MENU_REDO)
            edit_menu.add_separator()
            edit_menu.add_item("Preferences...", self.MENU_PREFERENCES)

            view_menu = gui.Menu()
            # Scene display options
            view_menu.add_item("Show Axes", self.MENU_SHOW_AXES)
            view_menu.set_checked(self.MENU_SHOW_AXES, True)
            view_menu.add_item("Show Grid", self.MENU_SHOW_GRID)
            view_menu.set_checked(self.MENU_SHOW_GRID, True)
            view_menu.add_item("Show Model Bounding Box",
                               self.MENU_SHOW_MODEL_BB)
            view_menu.set_checked(self.MENU_SHOW_MODEL_BB,
                                  self.ros_thread.show_model_bounding_box)
            ground_plane_menu = gui.Menu()
            ground_plane_menu.add_item("XY", 100)
            ground_plane_menu.add_item("XZ", 101)
            ground_plane_menu.add_item("YZ", 102)
            view_menu.add_menu("Ground Plane", ground_plane_menu)
            view_menu.add_separator()
            # Object display options
            view_menu.add_item("Show Model", self.MENU_SHOW_MESH)
            view_menu.set_checked(self.MENU_SHOW_MESH,
                                  self.ros_thread.show_mesh)
            view_menu.add_item("Show Point Clouds",
                               self.MENU_SHOW_POINT_CLOUD)
            view_menu.set_checked(self.MENU_SHOW_POINT_CLOUD,
                                  self.ros_thread.show_point_cloud)
            view_menu.add_item("Show Curvatures",
                               self.MENU_SHOW_CURVATURES)
            view_menu.set_checked(self.MENU_SHOW_CURVATURES,
                                  self.ros_thread.show_curvatures)
            view_menu.add_item("Show Regions", self.MENU_SHOW_REGIONS)
            view_menu.set_checked(self.MENU_SHOW_REGIONS,
                                  self.ros_thread.show_regions)
            view_menu.add_item("Show Noise Points",
                               self.MENU_SHOW_NOISE_POINTS)
            view_menu.set_checked(
                self.MENU_SHOW_NOISE_POINTS, self.ros_thread.show_noise_points)
            view_menu.add_item("Show FOV Clusters",
                               self.MENU_SHOW_CLUSTERS)
            view_menu.set_checked(
                self.MENU_SHOW_CLUSTERS, self.ros_thread.show_fov_clusters)
            view_menu.add_item("Show Viewpoints", self.MENU_SHOW_VIEWPOINTS)
            view_menu.set_checked(self.MENU_SHOW_VIEWPOINTS,
                                  self.ros_thread.show_viewpoints)
            view_menu.add_item("Show Region View Manifolds",
                               self.MENU_SHOW_REGION_VIEW_MANIFOLDS)
            view_menu.set_checked(
                self.MENU_SHOW_REGION_VIEW_MANIFOLDS, self.ros_thread.show_region_view_manifolds)

            view_menu.add_item("Show Path", self.MENU_SHOW_PATH)
            view_menu.set_checked(self.MENU_SHOW_PATH,
                                  self.ros_thread.show_path)
            view_menu.add_separator()
            # Cluster-viewpoint rendering mode
            render_mode_menu = gui.Menu()
            render_mode_menu.add_item("Convex Hull",    self.MENU_RENDER_CONVEX_HULL)
            render_mode_menu.add_item("Cluster Cloud",  self.MENU_RENDER_CLUSTER_CLOUD)
            render_mode_menu.add_item("Frustum",        self.MENU_RENDER_FRUSTUM)
            render_mode_menu.add_item("Lines",          self.MENU_RENDER_LINES)
            render_mode_menu.add_item("Viewpoint Only", self.MENU_RENDER_VIEWPOINT_ONLY)
            render_mode_menu.add_item("Origin Sphere",  self.MENU_RENDER_ORIGIN_SPHERE)
            render_mode_menu.set_checked(self.MENU_RENDER_CONVEX_HULL, True)
            view_menu.add_menu("Rendering Mode", render_mode_menu)
            view_menu.add_separator()
            # Panel display options
            view_menu.add_item("Lighting & Materials",
                               self.MENU_SHOW_SETTINGS)
            view_menu.set_checked(
                self.MENU_SHOW_SETTINGS, True)
            view_menu.add_item("Error Logging",
                               self.MENU_SHOW_ERRORS)
            view_menu.set_checked(
                self.MENU_SHOW_ERRORS, False)
            help_menu = gui.Menu()
            help_menu.add_item(
                "About", self.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("View", view_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("View", view_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        # menu item is activated.
        # w.set_on_menu_item_activated(
        #     self.MENU_NEW, self._on_menu_new)
        # w.set_on_menu_item_activated(
        #     self.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(
            self.MENU_SAVE, self._on_menu_save)
        w.set_on_menu_item_activated(
            self.MENU_SAVE_AS, self._on_menu_save_as)
        w.set_on_menu_item_activated(
            self.MENU_OPEN, self._on_menu_open
        )
        # w.set_on_menu_item_activated(self.MENU_IMPORT_MODEL,
        #                              self._on_menu_import_model)
        # w.set_on_menu_item_activated(self.MENU_IMPORT_PCD,
        #                              self._on_menu_import_pcd)
        # w.set_on_menu_item_activated(
        #     self.MENU_QUIT, self._on_menu_quit)
        # w.set_on_menu_item_activated(
        #     self.MENU_PREFERENCES, self._on_menu_preferences)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_AXES, self._on_menu_show_axes)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_GRID, self._on_menu_show_grid)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_MODEL_BB, self._on_menu_show_model_bounding_box)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_MESH, self._on_menu_show_mesh)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_POINT_CLOUD, self._on_menu_show_point_cloud)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_CURVATURES, self._on_menu_show_curvatures)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_REGIONS, self._on_menu_show_regions)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_CLUSTERS, self._on_menu_show_fov_clusters)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_NOISE_POINTS, self._on_menu_show_noise_points)
        w.set_on_menu_item_activated(self.MENU_SHOW_VIEWPOINTS,
                                     self._on_menu_show_viewpoints)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_PATH, self._on_menu_show_path)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_REGION_VIEW_MANIFOLDS,
            self._on_menu_show_region_view_manifolds)
        w.set_on_menu_item_activated(
            self.MENU_RENDER_CONVEX_HULL,
            lambda: self._on_menu_set_render_mode(ClusterViewpointMode.CONVEX_HULL))
        w.set_on_menu_item_activated(
            self.MENU_RENDER_CLUSTER_CLOUD,
            lambda: self._on_menu_set_render_mode(ClusterViewpointMode.CLUSTER_CLOUD))
        w.set_on_menu_item_activated(
            self.MENU_RENDER_FRUSTUM,
            lambda: self._on_menu_set_render_mode(ClusterViewpointMode.FRUSTUM))
        w.set_on_menu_item_activated(
            self.MENU_RENDER_LINES,
            lambda: self._on_menu_set_render_mode(ClusterViewpointMode.LINES))
        w.set_on_menu_item_activated(
            self.MENU_RENDER_VIEWPOINT_ONLY,
            lambda: self._on_menu_set_render_mode(ClusterViewpointMode.VIEWPOINT_ONLY))
        w.set_on_menu_item_activated(
            self.MENU_RENDER_ORIGIN_SPHERE,
            lambda: self._on_menu_set_render_mode(ClusterViewpointMode.ORIGIN_SPHERE))

        # w.set_on_menu_item_activated(self.MENU_SHOW_SETTINGS,
        #                              self._on_menu_toggle_settings_panel)
        # w.set_on_menu_item_activated(
        #     self.MENU_ABOUT, self._on_menu_about)
        # ----

    def _refresh_file_list(self):
        """Refresh the main list of YAML files in the data path"""
        yaml_files = [f for f in os.listdir(self.ros_thread.data_path) if f.endswith('.yaml')]
        yaml_files = [f[:-5] for f in yaml_files] # Subtract .yaml extension for display
        self.file_list.set_items(yaml_files)
        self.file_list.selected_index = yaml_files.index(self.ros_thread.file_name) if self.ros_thread.file_name in yaml_files else 0
    
    def _refresh_file_tree(self):
        """Refresh the file tree to show a summary of the loaded results.

        Parses self.viz.results_dict into the hierarchy:

            Meshes:
              0:
                File: ...
                Units: ...
                Material: ...
                Dimensions: ...
                Surface Area: ...
                Point Cloud:
                  File: ...
                  Units: ...
                  Points: ...
                Regions: N
                Clusters: N

        Open3D's TreeView has no clear/remove API, so on each call the old
        widget is hidden and a fresh one is added directly to the window.
        """
        if not self.viz.results_dict:
            return

        # Hide the stale tree; Open3D treats visible=False as zero-height in layout
        self.file_tree.visible = False

        new_tree = gui.TreeView()
        new_tree.background_color = Materials.panel_color
        new_tree.can_select_items_with_children = True
        new_tree.visible = True

        root = new_tree.get_root_item()
        meshes_id = new_tree.add_text_item(root, "Meshes")

        for i, mesh_entry in enumerate(self.viz.results_dict.get('meshes', [])):
            mesh_id = new_tree.add_text_item(meshes_id, str(i))

            new_tree.add_text_item(mesh_id, f"File: {mesh_entry.get('file', '')}")
            new_tree.add_text_item(mesh_id, f"Units: {mesh_entry.get('units', '')}")
            new_tree.add_text_item(mesh_id, f"Material: {mesh_entry.get('material', '')}")
            new_tree.add_text_item(mesh_id, f"Dimensions: {mesh_entry.get('dimensions', '')}")
            new_tree.add_text_item(mesh_id, f"Surface Area: {mesh_entry.get('surface_area', '')}")

            pcd = mesh_entry.get('point_cloud', {})
            pcd_id = new_tree.add_text_item(mesh_id, "Point Cloud")
            new_tree.add_text_item(pcd_id, f"File: {pcd.get('file', '')}")
            new_tree.add_text_item(pcd_id, f"Units: {pcd.get('units', '')}")
            new_tree.add_text_item(pcd_id, f"Points: {pcd.get('points', 0)}")
            regions = mesh_entry.get('regions', {})
            n_regions = len(regions)
            n_clusters = sum(
                len(r.get('clusters', {})) for r in regions
            )
            new_tree.add_text_item(mesh_id, f"Regions: {n_regions}")
            new_tree.add_text_item(mesh_id, f"Clusters: {n_clusters}")

        self.window.add_child(new_tree)
        self.file_tree = new_tree

    def _update_save_menu(self):
        is_new = self.ros_thread.file_name == 'new'
        gui.Application.instance.menubar.set_enabled(self.MENU_SAVE, not is_new)

    def _on_menu_save(self):
        self.ros_thread.save_parameters_to_file(os.path.join(
            self.ros_thread.data_path, self.ros_thread.file_name) + ".yaml")

    def _on_menu_save_as(self):
        """Handle the Save As menu item"""
        # Open a file dialog to select the save location
        file_dialog = gui.FileDialog(
            gui.FileDialog.SAVE, "Save Config As...", self.window.theme)
        file_dialog.add_filter(".yaml", "YAML Files (*.yaml)")
        file_dialog.add_filter(".json", "JSON Files (*.json)")
        file_dialog.set_path(self.ros_thread.data_path)
        file_dialog.set_on_done(self._on_save_as_done)
        file_dialog.set_on_cancel(self.window.close_dialog)
        self.window.show_dialog(file_dialog)

    def _on_save_as_done(self, file_path):
        """Callback for when the Save As dialog is done"""
        if file_path:
            # Save the parameters to the specified file
            self.ros_thread.save_parameters_to_file(file_path)
            self._update_save_menu()
        else:
            print("Save As cancelled")

        self.window.close_dialog()
        self._refresh_file_list()

    def _on_menu_open(self):
        file_dialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Open Config...", self.window.theme)
        file_dialog.add_filter(".yaml", "YAML Files (*.yaml)")
        file_dialog.set_path(self.ros_thread.data_path)
        file_dialog.set_on_done(self._on_open_done)
        file_dialog.set_on_cancel(self.window.close_dialog)
        self.window.show_dialog(file_dialog)

    def _on_open_done(self, file_path):
        if file_path:
            self.ros_thread.load_config(file_path)
            self._update_save_menu()
            print(f"Config loaded from {file_path}")
        else:
            print("Open cancelled")

        self.window.close_dialog()

    def _refresh_view_menu(self):
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_MESH, self.viz.show_mesh_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_POINT_CLOUD, self.viz.show_point_cloud_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_CURVATURES, self.viz.show_curvatures_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_NOISE_POINTS, self.viz.show_noise_points_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_CLUSTERS, self.viz.show_clusters_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_VIEWPOINTS, self.viz.show_viewpoints_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_REGION_VIEW_MANIFOLDS, self.viz.show_region_view_manifolds_flag)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_PATH, self.viz.show_path_flag)

    def _on_menu_show_axes(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_AXES)

        self.show_axes(show)

    def _on_menu_show_grid(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_GRID)

        self.show_grid(show)

    def _on_menu_show_model_bounding_box(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_MODEL_BB)

        self.show_model_bounding_box(show)

    def _on_menu_show_mesh(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_MESH)
        self.viz.show_mesh(show)
        self._refresh_view_menu()

    def _on_menu_show_point_cloud(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_POINT_CLOUD)
        self.viz.show_point_cloud(show)
        self._refresh_view_menu()

    def _on_menu_show_curvatures(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_CURVATURES)
        self.viz.show_curvatures(show)
        self._refresh_view_menu()

    def _on_menu_show_regions(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_REGIONS)
        self.viz.show_regions(show)
        self._refresh_view_menu()

    def _on_menu_show_fov_clusters(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_CLUSTERS)
        self.viz.show_fov_clusters(show)
        self._refresh_view_menu()
    def _on_menu_show_viewpoints(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_VIEWPOINTS)
        self.viz.show_viewpoints(show)
        self._refresh_view_menu()

    def _on_menu_show_noise_points(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_NOISE_POINTS)
        self.viz.show_noise_points(show)
        self._refresh_view_menu()

    def _on_menu_show_region_view_manifolds(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_REGION_VIEW_MANIFOLDS)
        self.viz.show_region_view_manifolds(show)
        self._refresh_view_menu()

    def _on_menu_show_path(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_PATH)
        self.viz.show_path(show)
        self._refresh_view_menu()

    def _on_menu_set_render_mode(self, mode: ClusterViewpointMode):
        self.viz.set_mode(mode)
        self._refresh_render_mode_menu()

    def show_axes(self, show=True):
        self.scene_widget.scene.show_axes(show)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_AXES, show)
        self.ros_thread.show_axes = show
        self.ros_thread.set_parameter('show_axes', show)

    def show_grid(self, show=True):
        self.scene_widget.scene.show_ground_plane(
            show, o3d.visualization.rendering.Scene.GroundPlane.XZ)
        gui.Application.instance.menubar.set_checked(self.MENU_SHOW_GRID, show)
        self.ros_thread.show_grid = show
        self.ros_thread.set_parameter('show_grid', show)

    def show_model_bounding_box(self, show=True):
        self.scene_widget.scene.show_geometry('model_bounding_box', show)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_MODEL_BB, show)
        self.ros_thread.show_model_bounding_box = show
        self.ros_thread.set_parameter('show_model_bounding_box', show)

    def show_skybox(self, show=True):
        self.scene_widget.scene.show_skybox(show)
        self.ros_thread.show_skybox = show
        self.ros_thread.set_parameter('show_skybox', show)

    def _refresh_render_mode_menu(self):
        """Update render-mode submenu checkmarks to reflect the active mode."""
        current = self.viz._mode
        menubar = gui.Application.instance.menubar
        menubar.set_checked(self.MENU_RENDER_CONVEX_HULL,    current == ClusterViewpointMode.CONVEX_HULL)
        menubar.set_checked(self.MENU_RENDER_CLUSTER_CLOUD,  current == ClusterViewpointMode.CLUSTER_CLOUD)
        menubar.set_checked(self.MENU_RENDER_FRUSTUM,        current == ClusterViewpointMode.FRUSTUM)
        menubar.set_checked(self.MENU_RENDER_LINES,          current == ClusterViewpointMode.LINES)
        menubar.set_checked(self.MENU_RENDER_VIEWPOINT_ONLY, current == ClusterViewpointMode.VIEWPOINT_ONLY)
        menubar.set_checked(self.MENU_RENDER_ORIGIN_SPHERE,  current == ClusterViewpointMode.ORIGIN_SPHERE)

    # ============================================================================
    # INIT MAIN LAYOUT
    # ============================================================================

    def init_main_layout(self):
        """Initialize the Open3D GUI with fixed tabs"""
        self.parameter_widgets = {}
        # Get theme for consistent styling
        theme = self.window.theme
        em = theme.font_size
        # YAML file list — direct window child, sized by _on_layout
        self.file_list = gui.ListView()
        self.file_list.background_color = Materials.panel_color
        self._refresh_file_list()
        def _on_list_item_activated(yaml_file, _):
            self.ros_thread.load_config(os.path.join(self.ros_thread.data_path, yaml_file + ".yaml"))
            self._update_save_menu()
        self.file_list.set_on_selection_changed(_on_list_item_activated)
        # Hidden until _on_layout sets its frame — same reason as file_tree.
        self.file_list.visible = False

        # Regions tree — direct window child, sized by _on_layout.
        # Kept separate from file_list so _refresh_file_tree can swap it out
        # via visible=False without affecting any Vert layout.
        self.file_tree = gui.TreeView()
        self.file_tree.background_color = Materials.panel_color
        self.file_tree.can_select_items_with_children = True
        # Hidden until regions are loaded — prevents the empty widget from
        # occupying its full default frame and covering other panels before
        # _on_layout has had a chance to run.
        self.file_tree.visible = False

        self.window.add_child(self.file_list)
        self.window.add_child(self.file_tree)

        # main_layout IS the TabControl — no Vert wrapper, so _on_layout frames
        # the TabControl directly and the ScrollableVert pages are constrained
        # to the area below the tab strip (fixing the "tab scrolls, content stays"
        # bug caused by the outer Vert adding a spurious scrollbar).
        self.main_layout = gui.TabControl()
        self.main_layout.background_color = Materials.tab_control_background_color
        self.main_layout = self.main_layout  # alias used by _on_layout and on_mouse

        self.action_layout = gui.TabControl()
        self.action_layout.background_color = Materials.tab_control_background_color

        # Create tabs for each top-level key
        for node_name, tab_data in self.parameters_dict.items():
            self.parameter_widgets[node_name] = {}
            if node_name in ['viewpoint_traversal', 'task_planning']:
                action_tab_panel = self.create_action_tab_panel(node_name, tab_data, em)
                self.action_layout.add_tab(
                    node_name.title().replace('_', ' '), action_tab_panel)
            elif node_name == 'gui':
                continue
            else:
                main_tab_panel = self.create_main_tab_panel(node_name, tab_data, em)
                self.main_layout.add_tab(
                    node_name.title().replace('_', ' '), main_tab_panel)

        self.window.add_child(self.main_layout)
        self.window.add_child(self.action_layout)
    
    def create_action_tab_panel(self, node_name, tab_data, em):
        """Create a panel for action tabs - NO scrolling, just buttons"""
        panel = gui.Vert(0.5 * em, gui.Margins(0.25 * em))
        panel.background_color = Materials.panel_color

        for section_name, section_data in tab_data.items():
            if section_name in ['servo_controllers', 'trajectory_controllers', 'servo_node_name', 'controller_manager_name', 'viewpoints_file', 'data_path']:
                continue
            if isinstance(section_data, dict):
                if section_name in ['selected_region', 'selected_cluster']:
                    widget_grid = self.create_parameter_widget(node_name, section_data, em)
                    panel.add_child(widget_grid)
                    

        return panel

    def create_main_tab_panel(self, node_name, tab_data, em):
        """Create a scrollable panel for a tab - ONLY scrolling here"""
        # Create scrollable area directly - no intermediate containers
        scroll_area = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 2 * em, 0.25 * em))
        scroll_area.background_color = Materials.panel_color

        # Create the content recursively
        content = self.create_nested_content(
            node_name, node_name, tab_data, em)
        content.background_color = Materials.content_color
        scroll_area.add_child(content)

        return scroll_area

    def create_collapsable_section(self, node_name, section_name, section_data, em, level):
        """Create a collapsable section for nested parameters"""
        # Create a collapsable widget
        collapsable = gui.CollapsableVert(section_name.replace('_', ' ').title(
        ), 0.25 * em, gui.Margins(0.25 * em, 0.00 * em, 0.00 * em, 0.00 * em))
        collapsable.background_color = Materials.collapsable_panel_color

        # Add content to the collapsable section
        content = self.create_nested_content(
            node_name, section_name, section_data, em, level + 1)

        # If section_name is 'Sampling', add a button
        if section_name == 'sampling':
            button = gui.Button("Sample Point Cloud")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.sample_point_cloud())
            content.add_child(button)
        elif section_name == 'curvature':
            button = gui.Button("Compute Curvature")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.estimate_curvature())
            content.add_child(button)
        elif section_name == 'region_growth':
            button = gui.Button("Run Region Growth")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.region_growth())
            content.add_child(button)
        elif section_name == 'fov_clustering':
            button = gui.Button("Run FOV Clustering")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.fov_clustering())
            content.add_child(button)
        elif section_name == 'projection':
            button = gui.Button("Project Viewpoints")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.project_viewpoints())
            content.add_child(button)

        collapsable.add_child(content)

        # Expand by default for first level
        if level == 0:
            collapsable.set_is_open(True)
        # collapsable.set_is_open(False)

        return collapsable

    def create_nested_content(self, node_name, parent_name, data, em, level=0):
        """Recursively create nested content for parameters"""
        # Increase spacing based on nesting level for better hierarchy
        spacing = 0.5 * em if level == 0 else 0.33 * em
        margins = gui.Margins(
            left=0.25 * em + (level * 0.125 * em),  # Indent nested levels
            top=0.25 * em,
            right=0.0 * em,
            bottom=0.25 * em
        )
        container = gui.Vert(spacing, margins)
        container.background_color = Materials.content_color

        # If this is a leaf parameter (has 'name', 'type', 'value')
        if isinstance(data, dict) and 'name' in data and 'type' in data and 'value' in data:
            widget_grid = self.create_parameter_widget(node_name, data, em)
            container.add_child(widget_grid)
        else:
            # This is a nested structure, process each sub-item
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'name' in value and 'type' in value and 'value' in value:
                        # This is a parameter
                        widget_grid = self.create_parameter_widget(
                            node_name, value, em)
                        container.add_child(widget_grid)
                    else:
                        # This is a nested group, create a collapsable section
                        section = self.create_collapsable_section(
                            node_name, key, value, em, level)
                        container.add_child(section)

        return container

    def create_parameter_widget(self, node_name, param_data, em):
        row = gui.Horiz(0.5 * em, gui.Margins(0.25 * em,
                        0.25 * em, 0.0 * em, 0.25 * em))

        param_name = param_data['name']
        param_type = param_data['type']
        param_value = param_data['value']
        param_control = param_data['control']
        param_range = param_data['range']

        # Create label
        label_container = gui.Vert()

        # If "percentage" in param_name, replace with "%"
        if "percentage" in param_name:
            param_name = param_name.replace("percentage", "%")
        label = gui.Label(param_name.split(
            '.')[-1].replace('_', ' ').title() + ":")
        label.text_color = Materials.text_color
        label_container.add_child(label)

        row.add_child(label_container)
        row.add_stretch()

        # Set preferred_width based on length of text with 6 * em as max:
        if len(label.text) > 12:
            label_container.preferred_width = 6 * em

        # Create appropriate widget based on type
        widget = None

        if param_type == 'bool':
            widget = gui.Checkbox("")
            widget.checked = bool(param_value)
            widget.set_on_checked(
                lambda checked, name=param_name: self.on_parameter_changed(node_name, name, checked))
            row.add_child(widget)  # ADD HERE

        elif param_type == 'integer':
            if param_control == 'slider':
                print(f"Creating slider for {param_name} with value {param_value}")
                # Special case for selected_region and selected_cluster - create a slider to select index
                widget = gui.Slider(gui.Slider.INT)  
                widget.set_limits(param_range[0], param_range[1])
                widget.int_value = int(param_value)
                widget.set_on_value_changed(
                    lambda value, name=param_name: self.on_parameter_changed(node_name, name, value))
                row.add_child(widget)  # ADD HERE
            else:
                widget = gui.NumberEdit(gui.NumberEdit.INT)
                widget.int_value = int(param_value)
                widget.set_on_value_changed(
                    lambda value, name=param_name: self.on_parameter_changed(node_name, name, value))
                row.add_child(widget)  # ADD HERE

        elif param_type == 'double':
            if param_control == 'slider':
                widget = gui.Slider(gui.Slider.DOUBLE)
                widget.set_limits(param_range[0], param_range[1])
            else:
                widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                widget.double_value = float(param_value)

            widget.set_on_value_changed(
                lambda value, name=param_name: self.on_parameter_changed(node_name, name, value))
            row.add_child(widget)  # ADD HERE

        elif param_type == 'string':
            if 'file' in param_name.lower() or 'path' in param_name.lower():
                widget = gui.TextEdit()
                widget.background_color = Materials.text_edit_background_color
                widget.text_value = str(param_value)
                widget.set_on_text_changed(
                    lambda text, name=param_name: self.on_parameter_changed(node_name, name, text))

                row.add_child(widget)
                row.add_fixed(0.25 * em)

                browse_button = gui.Button("...")
                browse_button.background_color = Materials.button_background_color
                browse_button.horizontal_padding_em = 0.5
                browse_button.vertical_padding_em = 0
                if 'mesh' in param_name.lower():
                    browse_button.set_on_clicked(
                        lambda node_name=node_name, name=param_name: self.on_browse_mesh(node_name, name))
                elif 'point_cloud' in param_name.lower():
                    browse_button.set_on_clicked(
                        lambda node_name=node_name, name=param_name: self.on_browse_pointcloud(node_name, name))
                elif 'regions' in param_name.lower():
                    browse_button.set_on_clicked(
                        lambda node_name=node_name, name=param_name: self.on_browse_regions(node_name, name))
                else:
                    browse_button.set_on_clicked(
                        lambda node_name=node_name, name=param_name: self.on_browse_file(node_name, name))

                row.add_child(browse_button)

            elif 'unit' in param_name.lower():
                widget = gui.Combobox()
                units = ['m', 'cm', 'mm', 'in', 'ft']
                for unit in units:
                    widget.add_item(unit)
                widget.selected_index = units.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE

            elif 'focus_metric' in param_name.lower():
                widget = gui.Combobox()
                metrics = ['sobel', 'squared_gradient', 'fswm']
                for metric in metrics:
                    widget.add_item(metric)
                widget.selected_index = metrics.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE

            elif 'focus_algorithm' in param_name.lower():
                widget = gui.Combobox()
                algorithms = ['default', 'adaptive', 'ehc']
                for algorithm in algorithms:
                    widget.add_item(algorithm)
                widget.selected_index = algorithms.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE

            elif 'tsp_algorithm' in param_name.lower():
                widget = gui.Combobox()
                algorithms = ['greedy', '2opt', '3opt', 'LKH'] # TSP algorithms
                for algorithm in algorithms:
                    widget.add_item(algorithm)
                widget.selected_index = algorithms.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE

            elif 'compare_algorithms' in param_name.lower():
                widget = gui.Combobox()
                algorithms = ['2opt', '3opt', 'LKH'] # compare TSP algorithms
                for algorithm in algorithms:
                    widget.add_item(algorithm)
                widget.selected_index = algorithms.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE  
            
            elif 'controller_type' in param_name.lower():
                widget = gui.Combobox()
                controller_types = ['PD', 'PID']
                for ctype in controller_types:
                    widget.add_item(ctype)
                widget.selected_index = controller_types.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE
            
            elif 'normal_estimation_algorithm' in param_name.lower():
                widget = gui.Combobox()
                algorithms = ['PCA', 'RANSAC']
                for algorithm in algorithms:
                    widget.add_item(algorithm)
                widget.selected_index = algorithms.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE

            else:
                widget = gui.TextEdit()
                widget.background_color = Materials.text_edit_background_color
                widget.text_value = str(param_value)
                widget.set_on_text_changed(
                    lambda text, name=param_name: self.on_parameter_changed(node_name, name, text))
                row.add_child(widget)  # ADD HERE



        else:
            # Default to text edit for unknown types
            widget = gui.TextEdit()
            widget.background_color = Materials.text_edit_background_color
            widget.text_value = str(param_value)
            widget.set_on_text_changed(
                lambda text, name=param_name: self.on_parameter_changed(node_name, name, text))
            row.add_child(widget)  # ADD HERE

        # Store widget reference
        if widget is not None:
            self.parameter_widgets[node_name][param_name] = widget

        return row

    def on_parameter_changed(self, node_name, param_name, new_value):
        """Handle parameter value changes"""
        # Update the internal dictionary
        self.update_parameter_value(node_name, param_name, new_value)
        # Set Parameter via ROS thread
        self.ros_thread.set_target_node_parameter(
            node_name, param_name, new_value)

    def update_parameter_value(self, node_name, param_name, new_value):
        """Update parameter value in the nested dictionary"""
        keys = param_name.split('.')
        current = self.parameters_dict[node_name]

        # Navigate to the parent of the target parameter
        for key in keys[:-1]:
            if key in current:
                current = current[key]
            else:
                return  # Path not found

        # Update the value if the parameter exists
        final_key = keys[-1]
        if final_key in current and isinstance(current[final_key], dict):
            current[final_key]['value'] = new_value

    def on_browse_mesh(self, node_name, param_name):
        """Handle mesh file browse button clicks"""
        file_dialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose mesh file", self.window.theme)
        file_dialog.set_path(self.ros_thread.data_path)
        file_dialog.add_filter(".stl", "Stereolithography Mesh(*.stl)")

        file_dialog.set_on_cancel(self.window.close_dialog)
        file_dialog.set_on_done(
            lambda path: self.on_file_selected(node_name, param_name, path))
        self.window.show_dialog(file_dialog)

    def on_browse_pointcloud(self, node_name, param_name):
        """Handle point cloud file browse button clicks"""
        file_dialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose point cloud file", self.window.theme)
        file_path = self.ros_thread.data_path
        if self.viz.mesh_name:
            pointcloud_path = os.path.join(
                self.ros_thread.data_path,
                f"{self.viz.mesh_name}_{self.viz.mesh_units}")
            if os.path.exists(pointcloud_path):
                file_path = pointcloud_path

        file_dialog.set_path(file_path)
        file_dialog.add_filter(".ply", "Polygon File Format(*.ply)")

        file_dialog.set_on_cancel(self.window.close_dialog)
        file_dialog.set_on_done(
            lambda path: self.on_file_selected(node_name, param_name, path))
        self.window.show_dialog(file_dialog)

    def on_browse_regions(self, node_name, param_name):
        """Handle regions file browse button clicks"""
        file_dialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose regions file", self.window.theme)
        file_path = self.ros_thread.data_path
        if self.viz.point_cloud_name:
            pointcloud_path = os.path.join(
                self.ros_thread.data_path,
                f"{self.viz.mesh_name}_{self.viz.mesh_units}")
            regions_path = os.path.join(
                pointcloud_path, f"{self.viz.point_cloud_name}_regions")
            # If regions_path exists, use it
            if os.path.exists(regions_path):
                file_path = regions_path
        file_dialog.set_path(file_path)
        file_dialog.add_filter(".json", "JSON files (*.json)")

        file_dialog.set_on_cancel(self.window.close_dialog)
        file_dialog.set_on_done(
            lambda path: self.on_file_selected(node_name, param_name, path))
        self.window.show_dialog(file_dialog)

    def on_browse_file(self, node_name, param_name):
        """Handle file browse button clicks"""
        file_dialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose file", self.window.theme)
        file_dialog.set_path(self.ros_thread.data_path)
        file_dialog.set_on_cancel(self.window.close_dialog)
        file_dialog.set_on_done(
            lambda path: self.on_file_selected(node_name, param_name, path))
        self.window.show_dialog(file_dialog)

    def on_file_selected(self, node_name, param_name, file_path):
        """Handle file selection"""
        print(f"File selected for {param_name}: {file_path}")
        self.on_parameter_changed(node_name, param_name, file_path)

        # Update the widget display
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            if hasattr(widget, 'text_value'):
                widget.text_value = file_path

        self.window.close_dialog()  # Close the dialog after selection

        # Immediately trigger the appropriate import — don't wait for the
        # update_flag polling loop, which won't fire because update_parameter_value
        # already wrote the new value locally, making the poll see no change.
        if 'results.file' in param_name:
            self.visualize_results(file_path)

    def on_sample_point_cloud(self):
        """Handle sample point cloud button click"""
        self.ros_thread.sample_point_cloud()

    def init_viewpoint_traversal_layout(self):
        """Initialize the region tabs"""
        # Disable the previous layout
        self.viewpoint_traversal_layout.enabled = False
        self.viewpoint_traversal_layout.visible = False

        em = self.window.theme.font_size
        # Viewpoint Traversal Layout
        self.viewpoint_traversal_layout = gui.Vert(
            0.5 * em, gui.Margins(0.5 * em))
        self.viewpoint_traversal_layout.background_color = Materials.panel_color

        horiz = gui.Horiz(0.25 * em, gui.Margins(0.25 * em,
                          0.25 * em, 0.25 * em, 0.25 * em))

        # Add region tabs
        self.region_tabs = gui.TabControl()
        self.region_tabs.background_color = Materials.panel_color

        for region_name in self.viz.region_names:
            formatted_name = region_name.replace("_", " ").capitalize()
            formatted_name = formatted_name.replace("Region", "").strip()
            self.region_tabs.add_tab(
                formatted_name, gui.Vert(0.5*em, gui.Margins(0.25*em)))

        def _on_region_tab_changed(value):
            """Handle region tab change"""
            max_value = (len(self.viz.traversal_order[value]) - 1
                         if self.viz.traversal_order else 0)
            self.viewpoint_slider.set_limits(0, max_value)
            self.viewpoint_slider.int_value = 0
            self.ros_thread.select_cluster(0)
            self.ros_thread.select_region(value)

        self.region_tabs.set_on_selected_tab_changed(_on_region_tab_changed)

        self.viewpoint_traversal_layout.add_child(self.region_tabs)

        # Create a UI slider for selection of Viewpoints with 3 buttons to its right
        self.viewpoint_slider = gui.Slider(gui.Slider.INT)
        max_value = (len(self.viz.traversal_order[
            self.region_tabs.selected_tab_index]) - 1
            if self.viz.traversal_order else 0)
        self.viewpoint_slider.set_limits(0, max_value)  # Example range

        def _on_viewpoint_slider_changed(value):
            """Handle viewpoint slider change"""
            # cluster_index = self.traversal_order[
            # int(self.region_tabs.selected_tab_index)][int(value)]
            cluster_index = int(value)
            self.ros_thread.select_cluster(cluster_index)

            return gui.Widget.EventCallbackResult.HANDLED

        self.viewpoint_slider.set_on_value_changed(
            _on_viewpoint_slider_changed)
        horiz.add_child(self.viewpoint_slider)

        # Add "Move to Viewpoint" button
        move_button = gui.Button("Move to Viewpoint")
        move_button.background_color = Materials.button_background_color
        move_button.set_on_clicked(self.ros_thread.move_to_viewpoint)

        go_button = gui.Button("Go")
        go_button.background_color = Materials.go_button_background_color

        def _on_go_button_clicked():
            self.ros_thread.inspect_region()

        go_button.set_on_clicked(_on_go_button_clicked)

        optimize_button = gui.Button("Optimize Traversal")
        optimize_button.background_color = Materials.button_background_color
        optimize_button.set_on_clicked(self.ros_thread.optimize_traversal)

        horiz.add_child(optimize_button)
        horiz.add_child(move_button)
        horiz.add_child(go_button)

        self.viewpoint_traversal_layout.add_child(horiz)

        self.window.add_child(self.viewpoint_traversal_layout)

    # -----------------------------------------------------------------------------------#

    def load_config(self, file_path):
        self.ros_thread.load_config(file_path)

    def visualize_results(self, file_path):
        print(f"Visualizing results from: {file_path}")
        result = self.viz.visualize_results(file_path)

        # ROS interaction: notify task_planning when viewpoints are present
        if result['show_viewpoints']:
            self.ros_thread.set_target_node_parameter(
                'task_planning', 'viewpoints_file', file_path)

        # Sync menu checkmarks with the visibility flags set by the Visualizer
        self._refresh_view_menu()

        # Build region tab control (GUI-only, not part of Visualizer)
        em = self.window.theme.font_size
        self.region_tabs = gui.TabControl()
        for region_name in self.viz.region_names:
            self.region_tabs.add_tab(
                region_name, gui.Vert(0.5 * em, gui.Margins(0.25 * em)))

        # self._refresh_file_tree()

    def select_mesh(self, mesh_idx):
        self.viz.select_mesh(mesh_idx)

    def select_region(self, region_number):
        if self.viz.select_region(region_number):
            self.viewpoint_slider.int_value = 0

    def select_cluster(self, cluster_number):
        if self.viz.select_cluster(cluster_number):
            self.viewpoint_slider.int_value = cluster_number

    def select_viewpoint(self, cluster_number):
        if self.viz.select_cluster(cluster_number):
            self.viewpoint_slider.int_value = cluster_number

    def update_scene(self):


        self.update_all_widgets_from_dict()

        self._refresh_file_tree()

        # Update visible geometry
        self.show_axes(self.ros_thread.show_axes)
        self.show_grid(self.ros_thread.show_grid)
        self.show_model_bounding_box(self.ros_thread.show_model_bounding_box)
        self.show_skybox(self.ros_thread.show_skybox)

        # Sleep to maintain FPS
        this_draw_time = time.time()
        if this_draw_time - self.last_draw_time < 1/self.fps:
            time.sleep(1/self.fps - (this_draw_time - self.last_draw_time))
            this_draw_time = time.time()

    def add_geometry(self, name, geometry, material):
        self.viz.add_geometry(name, geometry, material)

    def on_main_window_closing(self):
        gui.Application.instance.quit()

        rclpy.shutdown()

        return True  # False would cancel the close

    def on_main_window_tick_event(self):
        self.update_scene()
        return True  # request redraw every tick; panels stay visible and widget updates are drawn

    def _on_layout(self, layout_context):
        # Guard against early calls that arrive before __init__ has finished
        # adding all widgets (set_on_layout is registered before init_main_layout).
        if not hasattr(self, 'main_layout'):
            return

        em = self.window.theme.font_size
        margin = 0.5 * em

        r = self.window.content_rect

        self.scene_widget.frame = r

        main_width = 30 * em
        main_height = (r.height - 2 * margin)/2

        panel_x = r.width - main_width - margin

        # YAML file list — size to content, capped at 35 % of the panel
        file_list_h = min(
            self.file_list.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height,
            main_height * 0.35)
        self.file_list.frame = gui.Rect(
            panel_x, r.y + margin, main_width, file_list_h)
        self.file_list.visible = True

        # Regions tree fills the rest of the top half
        tree_h = main_height - file_list_h - margin * 0.5
        self.file_tree.frame = gui.Rect(
            panel_x,
            r.y + margin + file_list_h + margin * 0.5,
            main_width,
            max(0, tree_h))

        # Place main layout below file layout
        self.main_layout.frame = gui.Rect(
            r.width - main_width - margin,
            r.y + margin + main_height + margin,
            main_width,
            main_height)

        vpt_height = self.action_layout.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        vpt_width = r.width - main_width - 3 * margin
        self.action_layout.frame = gui.Rect(
            margin,
            r.y + r.height - vpt_height - margin,
            vpt_width,
            vpt_height)

        # Ensure a redraw follows every layout pass so frames set above are
        # actually rendered.  Without this the draw scheduled by the tick event
        # can fire *before* _on_layout processes, leaving panels invisible until
        # the user triggers another draw (e.g. a mouse click).
        self.window.post_redraw()

    def set_widget_value(self, widget, value, limits=None):
        """Set the value of a widget based on its type.

        If *limits* is a (min, max) tuple and the widget is a slider,
        the slider range is updated before setting the value so the new
        value is always within the valid range.
        """
        try:
            if limits is not None and hasattr(widget, 'set_limits'):
                widget.set_limits(limits[0], limits[1])

            if hasattr(widget, 'checked'):
                # Checkbox widget
                widget.checked = bool(value)
            elif hasattr(widget, 'double_value'):
                # Double NumberEdit / Slider widget
                widget.double_value = float(value)
            elif hasattr(widget, 'int_value'):
                # Integer NumberEdit widget
                widget.int_value = int(value)
            elif hasattr(widget, 'text_value'):
                # TextEdit widget
                widget.text_value = str(value)
                if value == '':
                    widget.text_value = 'None'
            else:
                print(f"Warning: Unknown widget type for value: {value}")
        except Exception as e:
            print(f"Error setting widget value to {value}: {e}")

    def update_all_widgets_from_dict(self):
        """Update all widget values from the current parameter dictionary"""

        parameters_dict = self.ros_thread.parameters_dict
        parameters_updated = False

        for node_name, node_parameter_widgets in self.parameter_widgets.items():
            for param_name, widget in node_parameter_widgets.items():
                if param_name in parameters_dict[node_name]:
                    update_flag = parameters_dict[node_name][param_name]['update_flag']
                    if update_flag:
                        parameters_updated = True
                        # Update the widget value from the parameters dictionary
                        if 'value' in parameters_dict[node_name][param_name]:
                            param_value = parameters_dict[node_name][param_name]['value']
                            param_range = parameters_dict[node_name][param_name].get('range')
                            self.set_widget_value(widget, param_value, limits=param_range)
                            
                            if 'results.file' in param_name:
                                self.visualize_results(param_value)
                            elif 'selected_mesh' in param_name:
                                self.select_mesh(param_value)
                            elif 'selected_region' in param_name:
                                self.select_region(param_value)
                            elif 'selected_cluster' in param_name:
                                self.select_cluster(param_value)
                            elif 'model.camera.fov.height' in param_name:
                                self.camera_fov_height = param_value
                                self.camera_updated = True
                            elif 'model.camera.fov.width' in param_name:
                                self.camera_fov_width = param_value
                                self.camera_updated = True

                            parameters_dict[node_name][param_name]['update_flag'] = False

        if parameters_updated:
            print("------------------------------------")
            self.window.post_redraw()

        self.ros_thread.parameters_dict = parameters_dict


def main(args=None):
    rclpy.init(args=args)
    print(args)

    gui.Application.instance.initialize()

    gui_client = GUIClient()

    def initial_draw():
        gui_client.window.set_needs_layout()
        gui_client.window.post_redraw()

    gui.Application.instance.post_to_main_thread(
        gui_client.window, initial_draw)

    gui.Application.instance.run()


if __name__ == '__main__':
    print("Open3D version:", o3d.__version__)
    main()
