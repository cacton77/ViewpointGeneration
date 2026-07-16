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
from viewpoint_generation.visualizer import Visualizer, RegionSurfaceMode, OverlayKind

sys.stdout.reconfigure(line_buffering=True)
isMacOS = sys.platform == 'darwin'
# o3d.visualization.webrtc_server.enable_webrtc()

# Params handled entirely by the custom traversal section; excluded from auto-generation.
_TRAVERSAL_CUSTOM_PARAMS = frozenset({
    'tsp_algorithm', 'vrp_algorithm', 'vrp_joint_weights',
    'vrp_aco_n_ants', 'vrp_aco_n_iter', 'vrp_aco_alpha', 'vrp_aco_beta', 'vrp_aco_rho',
})


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
    MENU_SHOW_SETTINGS = 23
    MENU_SHOW_ERRORS = 24
    MENU_ABOUT = 26
    MENU_SHOW_JOINT_PATH = 39
    MENU_SHOW_UNREACHABLE = 40
    MENU_SHOW_BLIND_SPOTS = 46
    # Region surface mode (exclusive)
    MENU_SURFACE_SOLID = 27
    MENU_SURFACE_CLUSTER = 28
    # Viewpoint overlays (inclusive)
    MENU_OVERLAY_MARKER = 34
    MENU_OVERLAY_FOV_CYLINDER = 35
    MENU_OVERLAY_ORIGIN_LINE = 36
    MENU_OVERLAY_FRUSTUM = 37
    MENU_OVERLAY_ORIGIN_MARKER = 38

    # (menu id, overlay kind) pairs for the Viewpoint Overlays submenu.
    _OVERLAY_MENU_ITEMS = [
        (MENU_OVERLAY_MARKER,        OverlayKind.MARKER),
        (MENU_OVERLAY_FOV_CYLINDER,  OverlayKind.FOV_CYLINDER),
        (MENU_OVERLAY_ORIGIN_LINE,   OverlayKind.ORIGIN_LINE),
        (MENU_OVERLAY_FRUSTUM,       OverlayKind.FRUSTUM),
        (MENU_OVERLAY_ORIGIN_MARKER, OverlayKind.ORIGIN_MARKER),
    ]

    MENU_SEL_OVERLAY_MARKER = 41
    MENU_SEL_OVERLAY_FOV_CYLINDER = 42
    MENU_SEL_OVERLAY_ORIGIN_LINE = 43
    MENU_SEL_OVERLAY_FRUSTUM = 44
    MENU_SEL_OVERLAY_ORIGIN_MARKER = 45

    # (menu id, overlay kind) pairs for the Selected Viewpoint Overlays submenu.
    _SELECTED_OVERLAY_MENU_ITEMS = [
        (MENU_SEL_OVERLAY_MARKER,        OverlayKind.MARKER),
        (MENU_SEL_OVERLAY_FOV_CYLINDER,  OverlayKind.FOV_CYLINDER),
        (MENU_SEL_OVERLAY_ORIGIN_LINE,   OverlayKind.ORIGIN_LINE),
        (MENU_SEL_OVERLAY_FRUSTUM,       OverlayKind.FRUSTUM),
        (MENU_SEL_OVERLAY_ORIGIN_MARKER, OverlayKind.ORIGIN_MARKER),
    ]

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

        self.init_layouts()
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
            panels = [self.main_layout]
            if getattr(self, 'task_planning_panel', None) is not None:
                panels.append(self.task_planning_panel)
            for layout in panels:
                if not layout.visible:
                    continue
                frame = layout.frame
                if (frame.get_left() <= event.x <= frame.get_right() and
                        frame.get_top() <= event.y <= frame.get_bottom()):
                    return gui.Widget.EventCallbackResult.CONSUMED
            return self.viz.on_mouse(event)

        self.scene_widget.set_on_mouse(on_mouse)

        self.window.add_child(self.scene_widget)

        # Create Visualizer — owns all scene geometry and rendering state
        self.viz = Visualizer(self.scene_widget)
        self.viz.setup_multi_directional_lighting()

        # Draw a default ground grid and frame the camera to it so the scene
        # has a reference plane before any mesh/results are loaded.
        self.viz.setup_default_scene()

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
            view_menu.add_item("Show Cartesian Path",
                               self.MENU_SHOW_JOINT_PATH)
            view_menu.set_checked(self.MENU_SHOW_JOINT_PATH,
                                  self.ros_thread.show_joint_path)
            view_menu.add_item("Show Unreachable Viewpoints",
                               self.MENU_SHOW_UNREACHABLE)
            view_menu.set_checked(self.MENU_SHOW_UNREACHABLE,
                                  self.ros_thread.show_unreachable)
            view_menu.add_item("Show Blind Spots",
                               self.MENU_SHOW_BLIND_SPOTS)
            view_menu.set_checked(self.MENU_SHOW_BLIND_SPOTS,
                                  self.ros_thread.show_blind_spots)
            view_menu.add_separator()
            # Region surface coloring — exclusive (one at a time).
            surface_menu = gui.Menu()
            surface_menu.add_item("Solid",   self.MENU_SURFACE_SOLID)
            surface_menu.add_item("Cluster", self.MENU_SURFACE_CLUSTER)
            surface_menu.set_checked(self.MENU_SURFACE_SOLID, True)
            view_menu.add_menu("Region Surface", surface_menu)
            # Viewpoint overlays — inclusive (any combination on).
            overlay_menu = gui.Menu()
            overlay_menu.add_item("Viewpoint Marker", self.MENU_OVERLAY_MARKER)
            overlay_menu.add_item(
                "FOV Cylinder",     self.MENU_OVERLAY_FOV_CYLINDER)
            overlay_menu.add_item(
                "Origin Line",      self.MENU_OVERLAY_ORIGIN_LINE)
            overlay_menu.add_item(
                "Frustum",          self.MENU_OVERLAY_FRUSTUM)
            overlay_menu.add_item(
                "Origin Marker",    self.MENU_OVERLAY_ORIGIN_MARKER)
            overlay_menu.set_checked(self.MENU_OVERLAY_MARKER, True)
            overlay_menu.set_checked(self.MENU_OVERLAY_ORIGIN_MARKER, True)
            view_menu.add_menu("Viewpoint Overlays", overlay_menu)
            # Overlays for the selected viewpoint only (independent set).
            sel_overlay_menu = gui.Menu()
            sel_overlay_menu.add_item(
                "Viewpoint Marker", self.MENU_SEL_OVERLAY_MARKER)
            sel_overlay_menu.add_item(
                "FOV Cylinder",     self.MENU_SEL_OVERLAY_FOV_CYLINDER)
            sel_overlay_menu.add_item(
                "Origin Line",      self.MENU_SEL_OVERLAY_ORIGIN_LINE)
            sel_overlay_menu.add_item(
                "Frustum",          self.MENU_SEL_OVERLAY_FRUSTUM)
            sel_overlay_menu.add_item(
                "Origin Marker",    self.MENU_SEL_OVERLAY_ORIGIN_MARKER)
            sel_overlay_menu.set_checked(self.MENU_SEL_OVERLAY_MARKER, True)
            sel_overlay_menu.set_checked(self.MENU_SEL_OVERLAY_FOV_CYLINDER, True)
            sel_overlay_menu.set_checked(self.MENU_SEL_OVERLAY_ORIGIN_LINE, True)
            sel_overlay_menu.set_checked(self.MENU_SEL_OVERLAY_ORIGIN_MARKER, True)
            view_menu.add_menu("Selected Viewpoint Overlays", sel_overlay_menu)
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
            self.MENU_SHOW_JOINT_PATH, self._on_menu_show_joint_path)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_UNREACHABLE, self._on_menu_show_unreachable)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_BLIND_SPOTS, self._on_menu_show_blind_spots)
        w.set_on_menu_item_activated(
            self.MENU_SURFACE_SOLID,
            lambda: self._on_menu_set_surface_mode(RegionSurfaceMode.SOLID))
        w.set_on_menu_item_activated(
            self.MENU_SURFACE_CLUSTER,
            lambda: self._on_menu_set_surface_mode(RegionSurfaceMode.CLUSTER))
        for menu_id, kind in self._OVERLAY_MENU_ITEMS:
            w.set_on_menu_item_activated(
                menu_id, lambda k=kind, m=menu_id: self._on_menu_toggle_overlay(k, m))
        for menu_id, kind in self._SELECTED_OVERLAY_MENU_ITEMS:
            w.set_on_menu_item_activated(
                menu_id,
                lambda k=kind, m=menu_id: self._on_menu_toggle_selected_overlay(k, m))

        # w.set_on_menu_item_activated(self.MENU_SHOW_SETTINGS,
        #                              self._on_menu_toggle_settings_panel)
        # w.set_on_menu_item_activated(
        #     self.MENU_ABOUT, self._on_menu_about)
        # ----

    def _refresh_file_list(self):
        """Refresh the main list of YAML files in the data path"""
        yaml_files = [f for f in os.listdir(
            self.ros_thread.data_path) if f.endswith('.yaml')]
        # Subtract .yaml extension for display
        yaml_files = [f[:-5] for f in yaml_files]
        self.file_list.set_items(yaml_files)
        self.file_list.selected_index = yaml_files.index(
            self.ros_thread.file_name) if self.ros_thread.file_name in yaml_files else 0

    def _refresh_file_tree(self):
        """Rebuild the file-tree widget from viz's render-agnostic contents.

        The Visualizer owns the translation of results_dict (and selection
        state) into a nested ``{'label', 'children'}`` structure via
        ``get_file_tree_contents()``; this method only renders that structure
        into an Open3D TreeView.

        Open3D's TreeView has no clear/remove API (and renders opaquely
        regardless of background color), so on each call the old widget is
        hidden and a fresh one is added directly to the window.

        This runs every tick, but the widget is only rebuilt when the contents
        actually change. Rebuilding every frame would replace the TreeView
        continuously and discard the user's expand/collapse and selection state,
        making the tree impossible to interact with.
        """
        contents = self.viz.get_file_tree_contents()
        if not contents:
            return

        # Skip the rebuild when nothing changed so the live widget (and its
        # interaction state) is preserved between frames.
        if contents == self._file_tree_contents:
            return
        self._file_tree_contents = contents

        # Hide the stale tree; Open3D treats visible=False as zero-height in layout
        self.file_tree.visible = False

        new_tree = gui.TreeView()
        new_tree.background_color = Materials.tab_control_background_color
        new_tree.can_select_items_with_children = True
        new_tree.visible = True

        # Item id -> selection action, and item id -> deferred children for
        # collapsed nodes. Both are rebuilt with the tree since item ids are
        # only valid for this widget instance.
        self._tree_item_actions = {}
        self._tree_pending_children = {}
        self._populate_tree(new_tree, new_tree.get_root_item(), contents)
        new_tree.set_on_selection_changed(self._on_tree_item_selected)

        self.window.add_child(new_tree)
        self.file_tree = new_tree

    def _populate_tree(self, tree, parent_id, nodes):
        """Add nodes under parent_id, recording selection actions.

        Open3D's TreeView auto-expands everything and offers no collapse API,
        so a node flagged ``collapsed`` is added empty and its children stashed
        in _tree_pending_children to be added on first click (lazy population).
        """
        for node in nodes:
            node_id = tree.add_text_item(parent_id, node['label'])
            if node.get('select'):
                self._tree_item_actions[node_id] = node['select']
            children = node.get('children', [])
            if node.get('collapsed') and children:
                self._tree_pending_children[node_id] = children
            else:
                self._populate_tree(tree, node_id, children)

    def _on_tree_item_selected(self, item_id):
        """Drive region/cluster selection from a file-tree click.

        First reveals any deferred children (collapsed nodes start empty and are
        populated on click). Then looks up the action attached to the selected
        item and updates the visualizer. Selecting a cluster selects its region
        first so the cluster index is scoped correctly (mirrors select_cluster).
        """
        pending = self._tree_pending_children.pop(item_id, None)
        if pending is not None:
            self._populate_tree(self.file_tree, item_id, pending)

        action = self._tree_item_actions.get(item_id)
        if not action:
            return
        # Update the visualizer immediately for responsiveness, and set the
        # matching task_planning parameter so selection state (and the robot's
        # execution target) stays in sync — the param round-trip would redraw
        # the same selection, so the direct call just avoids the latency.
        if action['type'] == 'region':
            self.viz.select_region(action['region'])
            self.ros_thread.select_region(action['region'])
        elif action['type'] == 'cluster':
            self.viz.select_region(action['region'])
            self.viz.select_cluster(action['cluster'])
            self.ros_thread.select_region(action['region'])
            self.ros_thread.select_cluster(action['cluster'])
        elif action['type'] == 'algorithm':
            # Update the drawn path immediately and set the parameter so the
            # execution order follows the same algorithm.
            self.viz.set_traversal_algorithm(action['algorithm'])
            self.ros_thread.select_traversal_algorithm(action['algorithm'])

    def _update_save_menu(self):
        is_new = self.ros_thread.file_name == 'new'
        gui.Application.instance.menubar.set_enabled(
            self.MENU_SAVE, not is_new)

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
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_MESH, self.viz.show_mesh_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_POINT_CLOUD, self.viz.show_point_cloud_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_CURVATURES, self.viz.show_curvatures_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_NOISE_POINTS, self.viz.show_noise_points_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_CLUSTERS, self.viz.show_clusters_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_VIEWPOINTS, self.viz.show_viewpoints_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_JOINT_PATH, self.viz.show_joint_path_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_UNREACHABLE, self.viz.show_unreachable_flag)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_BLIND_SPOTS, self.viz.show_blind_spots_flag)

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

    def _on_menu_show_joint_path(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_JOINT_PATH)
        self.viz.show_joint_path(show)
        self.ros_thread.show_joint_path = show
        self._refresh_view_menu()

    def _on_menu_show_unreachable(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_UNREACHABLE)
        self.viz.show_unreachable(show)
        self.ros_thread.show_unreachable = show
        self._refresh_view_menu()

    def _on_menu_show_blind_spots(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_BLIND_SPOTS)
        self.viz.show_blind_spots(show)
        self.ros_thread.show_blind_spots = show
        self._refresh_view_menu()

    def _on_menu_set_surface_mode(self, mode: RegionSurfaceMode):
        self.viz.set_region_surface_mode(mode)
        self._refresh_render_mode_menu()

    def _on_menu_toggle_overlay(self, kind: OverlayKind, menu_id: int):
        on = not gui.Application.instance.menubar.is_checked(menu_id)
        self.viz.set_overlay_enabled(kind, on)
        gui.Application.instance.menubar.set_checked(menu_id, on)

    def _on_menu_toggle_selected_overlay(self, kind: OverlayKind, menu_id: int):
        on = not gui.Application.instance.menubar.is_checked(menu_id)
        self.viz.set_selected_overlay_enabled(kind, on)
        gui.Application.instance.menubar.set_checked(menu_id, on)

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
        """Update Region Surface / Viewpoint Overlay menu checkmarks."""
        current = self.viz._surface_mode
        menubar = gui.Application.instance.menubar
        menubar.set_checked(self.MENU_SURFACE_SOLID,
                            current == RegionSurfaceMode.SOLID)
        menubar.set_checked(self.MENU_SURFACE_CLUSTER,
                            current == RegionSurfaceMode.CLUSTER)
        for menu_id, kind in self._OVERLAY_MENU_ITEMS:
            menubar.set_checked(menu_id, kind in self.viz._enabled_overlays)
        for menu_id, kind in self._SELECTED_OVERLAY_MENU_ITEMS:
            menubar.set_checked(menu_id, kind in self.viz._selected_overlays)

    # ============================================================================
    # INIT MAIN LAYOUT
    # ============================================================================

    def init_layouts(self):
        """Build every persistent panel: the YAML file list, the regions tree,
        the tabbed per-node parameter panel, and the always-visible
        task_planning panel. Parses self.parameters_dict and creates the
        styled panels/sections/content for each node."""
        self.parameter_widgets = {}
        self._companion_widgets = {}
        # Always-visible task_planning panel — created below, framed by
        # _on_layout. Declared up front so _on_layout / on_mouse can reference
        # it safely even if the task_planning node is not connected.
        self.task_planning_panel = None
        # Get theme for consistent styling
        theme = self.window.theme
        em = theme.font_size
        # YAML file list — direct window child, sized by _on_layout
        self.file_list = gui.ListView()
        self.file_list.background_color = Materials.panel_color
        self._refresh_file_list()

        def _on_list_item_activated(yaml_file, _):
            self.ros_thread.load_config(os.path.join(
                self.ros_thread.data_path, yaml_file + ".yaml"))
            self._update_save_menu()
        self.file_list.set_on_selection_changed(_on_list_item_activated)
        # Hidden until _on_layout sets its frame — same reason as file_tree.
        self.file_list.visible = False

        # Regions tree — direct window child, sized by _on_layout. The TreeView
        # renders opaquely regardless of background color (it can't be made
        # transparent like the main layout). Kept separate from file_list so
        # _refresh_file_tree can swap it out via visible=False.
        self.file_tree = gui.TreeView()
        self.file_tree.background_color = Materials.tab_control_background_color
        self.file_tree.can_select_items_with_children = True
        # Last-rendered tree contents; used to rebuild the widget only when the
        # contents change (see _refresh_file_tree).
        self._file_tree_contents = None
        # Map of TreeView item id -> selection action for the current tree.
        self._tree_item_actions = {}
        # Map of TreeView item id -> deferred child nodes for collapsed nodes.
        self._tree_pending_children = {}
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

        # Create tabs for each top-level key. The gui node has no tab, and
        # task_planning is rendered in its own always-visible horizontal panel
        # (below) rather than as a tab.
        tp_name = self.ros_thread.task_planning_node_name
        for node_name, tab_data in self.parameters_dict.items():
            self.parameter_widgets[node_name] = {}
            if node_name == 'gui' or node_name == tp_name:
                continue
            main_tab_panel = self.create_main_tab_panel(
                node_name, tab_data, em)
            self.main_layout.add_tab(
                node_name.title().replace('_', ' '), main_tab_panel)

        self.window.add_child(self.main_layout)

        # Always-visible task_planning panel spanning the gap between the
        # regions tree and the main panel. Hidden until _on_layout frames it.
        if tp_name in self.parameters_dict:
            self.task_planning_panel = self.create_task_planning_panel(
                tp_name, self.parameters_dict[tp_name], em)
            self.task_planning_panel.visible = False
            self.window.add_child(self.task_planning_panel)

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

        # The traversal optimization controls (tsp_algorithm, compare, etc.) are
        # top-level params with no nested section to host a button, so attach the
        # action button at the tab level — mirroring the FOV clustering and
        # viewpoint projection buttons, which call their node's service.
        if node_name == 'viewpoint_traversal':
            content.add_child(self._build_traversal_mode_section(
                node_name, tab_data, em))
            button = gui.Button("Optimize Traversal")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.optimize_traversal())
            content.add_child(button)
        elif node_name == self.ros_thread.tsdf_node_name:
            button = gui.Button("Reset Voxel Block Grid")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.reset_tsdf())
            content.add_child(button)

        scroll_area.add_child(content)

        return scroll_area

    def _build_traversal_mode_section(self, node_name, tab_data, em):
        """TSP/VRP mode selector with a shared Algorithm dropdown and VRP parameter panel."""
        def _val(key, default):
            info = tab_data.get(key, {})
            v = info.get('value', default) if isinstance(
                info, dict) else default
            return v if v is not None else default

        tsp_algos = ['greedy', '2opt', '3opt', 'ILS', 'LKH']
        vrp_algos = ['vrp_greedy', 'vrp_2opt', 'vrp_3opt',
                     'vrp_ils', 'vrp_lkh', 'vrp_aco', 'vrp_hierarchical']
        init_tsp = _val('tsp_algorithm', 'greedy')
        init_vrp = _val('vrp_algorithm', '')
        init_weights = list(
            _val('vrp_joint_weights', [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        if len(init_weights) < 7:
            init_weights = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        vrp_mode = bool(init_vrp and init_vrp in vrp_algos)

        section = gui.Vert(0.5 * em, gui.Margins(0.25 * em,
                           0.25 * em, 0.25 * em, 0.25 * em))
        section.background_color = Materials.content_color

        # ── Mode + Algorithm row ──────────────────────────────────────────────
        top_row = gui.Horiz(0.5 * em, gui.Margins(0.25 * em,
                            0.25 * em, 0.25 * em, 0.25 * em))
        tsp_check = gui.Checkbox("TSP")
        tsp_check.checked = not vrp_mode
        vrp_check = gui.Checkbox("VRP")
        vrp_check.checked = vrp_mode
        top_row.add_child(tsp_check)
        top_row.add_child(vrp_check)
        top_row.add_stretch()
        algo_lbl = gui.Label("Algorithm:")
        algo_lbl.text_color = Materials.text_color
        top_row.add_child(algo_lbl)

        tsp_combo = gui.Combobox()
        for a in tsp_algos:
            tsp_combo.add_item(a)
        tsp_combo.selected_index = tsp_algos.index(
            init_tsp) if init_tsp in tsp_algos else 0
        tsp_combo.visible = not vrp_mode
        top_row.add_child(tsp_combo)

        vrp_combo = gui.Combobox()
        for a in vrp_algos:
            vrp_combo.add_item(a)
        vrp_combo.selected_index = vrp_algos.index(
            init_vrp) if init_vrp in vrp_algos else 0
        vrp_combo.visible = vrp_mode
        top_row.add_child(vrp_combo)

        section.add_child(top_row)
        self.parameter_widgets[node_name]['tsp_algorithm'] = tsp_combo
        self.parameter_widgets[node_name]['vrp_algorithm'] = vrp_combo

        # ── VRP parameters (collapsed unless VRP mode) ────────────────────────
        vrp_params = gui.CollapsableVert(
            "VRP Parameters", 0.25 * em,
            gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        vrp_params.background_color = Materials.collapsable_panel_color
        vrp_params.set_is_open(vrp_mode)

        jw_lbl = gui.Label("Joint Weights:")
        jw_lbl.text_color = Materials.text_color
        vrp_params.add_child(jw_lbl)

        joint_labels = ['Turntable', 'Pan', 'Lift',
                        'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
        weight_edits = []
        for jlbl, w0 in zip(joint_labels, init_weights):
            jw_row = gui.Horiz(
                0.5 * em, gui.Margins(0.25 * em, 0.0 * em, 0.0 * em, 0.0 * em))
            lbl = gui.Label(jlbl + ':')
            lbl.text_color = Materials.text_color
            jw_row.add_child(lbl)
            jw_row.add_stretch()
            ne = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            ne.double_value = float(w0)
            ne.set_preferred_width(4 * em)
            jw_row.add_child(ne)
            vrp_params.add_child(jw_row)
            weight_edits.append(ne)

        def _on_weight(idx, v):
            vals = [float(we.double_value) for we in weight_edits]
            vals[idx] = float(v)
            self.on_parameter_changed(node_name, 'vrp_joint_weights', vals)
        for i, we in enumerate(weight_edits):
            we.set_on_value_changed(lambda v, i=i: _on_weight(i, v))

        aco_lbl = gui.Label("ACO Parameters:")
        aco_lbl.text_color = Materials.text_color
        vrp_params.add_child(aco_lbl)

        aco_defs = [
            ('vrp_aco_n_ants',  'Ants',       _val(
                'vrp_aco_n_ants',  20),  gui.NumberEdit.INT),
            ('vrp_aco_n_iter',  'Iterations', _val(
                'vrp_aco_n_iter',  100), gui.NumberEdit.INT),
            ('vrp_aco_alpha',   'Alpha',      _val(
                'vrp_aco_alpha',   1.0), gui.NumberEdit.DOUBLE),
            ('vrp_aco_beta',    'Beta',       _val(
                'vrp_aco_beta',    2.0), gui.NumberEdit.DOUBLE),
            ('vrp_aco_rho',     'Rho',        _val(
                'vrp_aco_rho',     0.1), gui.NumberEdit.DOUBLE),
        ]
        for pkey, plbl, pval, ptype in aco_defs:
            aco_row = gui.Horiz(
                0.5 * em, gui.Margins(0.25 * em, 0.0 * em, 0.0 * em, 0.0 * em))
            lbl = gui.Label(plbl + ':')
            lbl.text_color = Materials.text_color
            aco_row.add_child(lbl)
            aco_row.add_stretch()
            ne = gui.NumberEdit(ptype)
            if ptype == gui.NumberEdit.INT:
                ne.int_value = int(pval)
            else:
                ne.double_value = float(pval)
            ne.set_preferred_width(4 * em)
            ne.set_on_value_changed(
                lambda v, k=pkey: self.on_parameter_changed(node_name, k, v))
            aco_row.add_child(ne)
            vrp_params.add_child(aco_row)
            self.parameter_widgets[node_name][pkey] = ne

        section.add_child(vrp_params)

        # ── Checkbox callbacks (mutually exclusive) ───────────────────────────
        # guard against re-entrant callbacks from programmatic checked changes
        _busy = [False]

        def _to_tsp():
            if _busy[0]:
                return
            _busy[0] = True
            tsp_check.checked = True
            vrp_check.checked = False
            tsp_combo.visible = True
            vrp_combo.visible = False
            vrp_params.set_is_open(False)
            self.on_parameter_changed(node_name, 'vrp_algorithm', '')
            self.on_parameter_changed(node_name, 'tsp_algorithm',
                                      tsp_algos[tsp_combo.selected_index])
            _busy[0] = False

        def _to_vrp():
            if _busy[0]:
                return
            _busy[0] = True
            tsp_check.checked = False
            vrp_check.checked = True
            tsp_combo.visible = False
            vrp_combo.visible = True
            vrp_params.set_is_open(True)
            self.on_parameter_changed(
                node_name, 'tsp_algorithm', 'Select Algorithm')
            self.on_parameter_changed(node_name, 'vrp_algorithm',
                                      vrp_algos[vrp_combo.selected_index])
            _busy[0] = False

        tsp_check.set_on_checked(lambda c: _to_tsp() if c else _to_vrp())
        vrp_check.set_on_checked(lambda c: _to_vrp() if c else _to_tsp())
        tsp_combo.set_on_selection_changed(
            lambda t, _: self.on_parameter_changed(node_name, 'tsp_algorithm', t))
        vrp_combo.set_on_selection_changed(
            lambda t, _: self.on_parameter_changed(node_name, 'vrp_algorithm', t))

        return section

    def create_collapsable_section(self, node_name, section_name, section_data, em, level):
        """Create a collapsable section for nested parameters"""
        # Create a collapsable widget
        collapsable = gui.CollapsableVert(section_name.replace('_', ' ').title(
        ), 0.25 * em, gui.Margins(0.25 * em, 0.00 * em, 0.00 * em, 0.00 * em))
        collapsable.background_color = Materials.collapsable_panel_color

        # Add content to the collapsable section
        content = self.create_nested_content(
            node_name, section_name, section_data, em, level + 1)

        if section_name == 'regions':
            button = gui.Button("Segment Regions")
            button.background_color = Materials.button_background_color
            button.set_on_clicked(lambda: self.ros_thread.segment_regions())
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
                        if node_name == 'viewpoint_traversal' and key in _TRAVERSAL_CUSTOM_PARAMS:
                            continue
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
                widget = gui.Slider(gui.Slider.INT)
                widget.set_limits(param_range[0], param_range[1])
                widget.int_value = int(param_value)

                number_edit = gui.NumberEdit(gui.NumberEdit.INT)
                number_edit.int_value = int(param_value)
                number_edit.set_limits(param_range[0], param_range[1])
                number_edit.set_preferred_width(4 * em)

                syncing = [False]

                def _slider_changed_int(value, name=param_name, ne=number_edit, s=syncing):
                    if s[0]:
                        return
                    s[0] = True
                    ne.int_value = int(value)
                    self.on_parameter_changed(node_name, name, value)
                    s[0] = False

                def _numedit_changed_int(value, name=param_name, sl=widget, s=syncing):
                    if s[0]:
                        return
                    s[0] = True
                    sl.int_value = int(value)
                    self.on_parameter_changed(node_name, name, value)
                    s[0] = False

                widget.set_on_value_changed(_slider_changed_int)
                number_edit.set_on_value_changed(_numedit_changed_int)

                slider_container = gui.Vert()
                slider_container.preferred_width = 10 * em
                slider_container.add_child(widget)
                row.add_child(slider_container)
                row.add_fixed(0.25 * em)
                row.add_child(number_edit)

                self._companion_widgets[f"{node_name}/{param_name}"] = number_edit
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
                widget.double_value = float(param_value)

                number_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                number_edit.double_value = float(param_value)
                number_edit.set_limits(param_range[0], param_range[1])
                number_edit.set_preferred_width(4 * em)

                syncing = [False]

                def _slider_changed_dbl(value, name=param_name, ne=number_edit, s=syncing):
                    if s[0]:
                        return
                    s[0] = True
                    ne.double_value = float(value)
                    self.on_parameter_changed(node_name, name, value)
                    s[0] = False

                def _numedit_changed_dbl(value, name=param_name, sl=widget, s=syncing):
                    if s[0]:
                        return
                    s[0] = True
                    sl.double_value = float(value)
                    self.on_parameter_changed(node_name, name, value)
                    s[0] = False

                widget.set_on_value_changed(_slider_changed_dbl)
                number_edit.set_on_value_changed(_numedit_changed_dbl)

                slider_container = gui.Vert()
                slider_container.preferred_width = 10 * em
                slider_container.add_child(widget)
                row.add_child(slider_container)
                row.add_fixed(0.25 * em)
                row.add_child(number_edit)

                self._companion_widgets[f"{node_name}/{param_name}"] = number_edit
            else:
                widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                widget.double_value = float(param_value)
                widget.set_on_value_changed(
                    lambda value, name=param_name: self.on_parameter_changed(node_name, name, value))
                row.add_child(widget)

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
                algorithms = ['greedy', '2opt', '3opt', 'LKH', 'ILS']
                for algorithm in algorithms:
                    widget.add_item(algorithm)
                widget.selected_index = algorithms.index(
                    param_value) if param_value in algorithms else 0
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)

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

            elif 'regions.algorithm' in param_name.lower():
                widget = gui.Combobox()
                algorithms = ['region_growth', 'partfield']
                for algorithm in algorithms:
                    widget.add_item(algorithm)
                widget.selected_index = algorithms.index(param_value)
                widget.set_on_selection_changed(
                    lambda selected_text, selected_index: self.on_parameter_changed(
                        node_name, param_name, selected_text))
                row.add_child(widget)  # ADD HERE

            elif 'fov_clustering.algorithm' in param_name.lower():
                widget = gui.Combobox()
                algorithms = ['kmeans', 'greedy_cover']
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

    # ============================================================================
    # TASK PLANNING PANEL (always-visible, horizontal)
    # ============================================================================

    def _iter_leaf_params(self, data):
        """Yield each leaf parameter dict (those with name/type/value) found by
        walking the nested parameter structure in declaration order."""
        if isinstance(data, dict) and 'name' in data and 'type' in data and 'value' in data:
            yield data
        elif isinstance(data, dict):
            for value in data.values():
                yield from self._iter_leaf_params(value)

    # Preferred left-to-right order for the task_planning namespace tabs.
    # Namespaces not listed here are appended afterwards in their natural order.
    TASK_PLANNING_TAB_ORDER = ['navigation', 'controllers', 'settings']

    def create_task_planning_panel(self, node_name, data, em):
        """Build the always-visible task_planning panel.

        Unlike the tabbed node panels (one tab per node), this panel has one
        tab per top-level parameter namespace (e.g. 'navigation', 'controllers',
        'settings'). Tabs are ordered by TASK_PLANNING_TAB_ORDER. Each tab spans
        the horizontal gap between the regions tree and the main panel, so its
        parameters are laid out left-to-right as compact label-over-control
        cells with no collapsing or nesting (the panel is short and wide).
        """
        tabs = gui.TabControl()
        tabs.background_color = Materials.tab_control_background_color

        # Per-tab lists of cell resizers, applied at layout time so the cells
        # fill the panel width (gui.Horiz has no per-child grow factor).
        self._tp_row_resizers = []

        ordered = sorted(
            data.keys(),
            key=lambda ns: (self.TASK_PLANNING_TAB_ORDER.index(ns)
                            if ns in self.TASK_PLANNING_TAB_ORDER
                            else len(self.TASK_PLANNING_TAB_ORDER)))
        for namespace in ordered:
            row, resizers = self.create_horizontal_param_row(
                node_name, data[namespace], em)
            tabs.add_tab(namespace.title().replace('_', ' '), row)
            self._tp_row_resizers.append(resizers)

        return tabs

    def create_horizontal_param_row(self, node_name, data, em):
        """Lay out every leaf parameter under *data* across the full width as
        compact label-over-control cells. Used for each tab of the task_planning
        panel. Cells are given equal, space-filling widths at layout time (see
        _apply_task_planning_cell_widths); returns ``(row, resizers)`` where
        each resizer assigns its cell's width."""
        row = gui.Horiz(
            0.5 * em, gui.Margins(0.5 * em, 0.25 * em, 0.5 * em, 0.25 * em))
        row.background_color = Materials.panel_color

        resizers = []
        for param_data in self._iter_leaf_params(data):
            cell, resize = self.create_horizontal_parameter_cell(
                node_name, param_data, em)
            row.add_child(cell)
            resizers.append(resize)

        return row, resizers

    def create_horizontal_parameter_cell(self, node_name, param_data, em):
        """Build one compact label-over-control cell for the task_planning panel.

        Registers the control in self.parameter_widgets (and any slider
        companion in self._companion_widgets) exactly like
        create_parameter_widget, so update_all_widgets_from_dict keeps the cell
        live. The cell stacks the label above the control vertically so the
        cells stay narrow when packed horizontally.

        Returns ``(cell, resize)`` where ``resize(cell_w)`` sets the cell's width
        and widens its stretchy control to fill that width. gui.Horiz has no
        per-child grow factor, so the row driver (_apply_task_planning_cell_widths)
        calls these at layout time to make the columns span the panel instead of
        left-justifying.
        """
        param_name = param_data['name']
        param_type = param_data['type']
        param_value = param_data['value']
        param_control = param_data['control']
        param_range = param_data['range']

        cell = gui.Vert(0.15 * em)
        # Widgets/containers to widen when the cell is resized; each entry is a
        # callable taking the target cell width (in px).
        fillers = []

        label = gui.Label(param_name.split(
            '.')[-1].replace('_', ' ').title() + ":")
        label.text_color = Materials.text_color
        cell.add_child(label)

        widget = None

        if param_type == 'bool':
            widget = gui.Checkbox("")
            widget.checked = bool(param_value)
            widget.set_on_checked(
                lambda checked, name=param_name: self.on_parameter_changed(node_name, name, checked))
            cell.add_child(widget)
            cell.preferred_width = 8 * em

        elif param_type == 'integer' and param_control == 'slider' and param_range:
            widget = gui.Slider(gui.Slider.INT)
            widget.set_limits(param_range[0], param_range[1])
            widget.int_value = int(param_value)

            number_edit = gui.NumberEdit(gui.NumberEdit.INT)
            number_edit.int_value = int(param_value)
            number_edit.set_limits(param_range[0], param_range[1])
            number_edit.set_preferred_width(3.5 * em)

            syncing = [False]

            def _slider_changed(value, name=param_name, ne=number_edit, s=syncing):
                if s[0]:
                    return
                s[0] = True
                ne.int_value = int(value)
                self.on_parameter_changed(node_name, name, value)
                s[0] = False

            def _numedit_changed(value, name=param_name, sl=widget, s=syncing):
                if s[0]:
                    return
                s[0] = True
                sl.int_value = int(value)
                self.on_parameter_changed(node_name, name, value)
                s[0] = False

            widget.set_on_value_changed(_slider_changed)
            number_edit.set_on_value_changed(_numedit_changed)

            control = gui.Horiz(0.25 * em)
            slider_container = gui.Vert()
            slider_container.preferred_width = 8 * em
            slider_container.add_child(widget)
            control.add_child(slider_container)
            control.add_child(number_edit)
            cell.add_child(control)
            cell.preferred_width = 13 * em

            # Grow the slider; reserve room for the number edit (3.5em) + the
            # control's inter-widget spacing (0.25em).
            fillers.append(lambda cw, c=slider_container, em=em:
                           setattr(c, 'preferred_width', int(max(2 * em, cw - 3.75 * em))))

            self._companion_widgets[f"{node_name}/{param_name}"] = number_edit

        elif param_type in ('integer', 'double'):
            widget = gui.NumberEdit(
                gui.NumberEdit.INT if param_type == 'integer' else gui.NumberEdit.DOUBLE)
            if param_type == 'integer':
                widget.int_value = int(param_value)
            else:
                widget.double_value = float(param_value)
            widget.set_on_value_changed(
                lambda value, name=param_name: self.on_parameter_changed(node_name, name, value))
            cell.add_child(widget)
            cell.preferred_width = 8 * em
            fillers.append(
                lambda cw, w=widget: w.set_preferred_width(int(max(1, cw))))

        else:
            # string / string_array / fallback — a text field, with a browse
            # button for file/path params (mirrors create_parameter_widget).
            widget = gui.TextEdit()
            widget.background_color = Materials.text_edit_background_color
            widget.text_value = str(param_value)
            widget.set_on_text_changed(
                lambda text, name=param_name: self.on_parameter_changed(node_name, name, text))

            is_file = 'file' in param_name.lower() or 'path' in param_name.lower()
            if is_file:
                control = gui.Horiz(0.25 * em)
                text_container = gui.Vert()
                text_container.preferred_width = 10 * em
                text_container.add_child(widget)
                control.add_child(text_container)
                browse_button = gui.Button("...")
                browse_button.background_color = Materials.button_background_color
                browse_button.horizontal_padding_em = 0.5
                browse_button.vertical_padding_em = 0
                browse_button.set_on_clicked(
                    lambda node_name=node_name, name=param_name: self.on_browse_file(node_name, name))
                control.add_child(browse_button)
                cell.add_child(control)
                cell.preferred_width = 13 * em
                # Grow the text field; reserve room for the browse button + spacing.
                fillers.append(lambda cw, c=text_container, em=em:
                               setattr(c, 'preferred_width', int(max(2 * em, cw - 2.5 * em))))
            else:
                cell.add_child(widget)
                cell.preferred_width = 11 * em

        if widget is not None:
            self.parameter_widgets[node_name][param_name] = widget

        def _resize(cell_w, cell=cell, fillers=fillers):
            cell_w = int(cell_w)
            cell.preferred_width = cell_w
            for filler in fillers:
                filler(cell_w)

        return cell, _resize

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

    # -----------------------------------------------------------------------------------#

    def load_config(self, file_path):
        self.ros_thread.load_config(file_path)

    def visualize_results(self, file_path):
        print(f"Visualizing results from: {file_path}")
        self.viz.visualize_results(file_path)

        # Point task_planning at the same results file so its planning/execution
        # uses these viewpoints. It loads from its own settings.results_file
        # param, which is otherwise never set by the current results-loading flow.
        self.ros_thread.set_results_file(file_path)

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
        self.viz.select_region(region_number)

    def select_cluster(self, cluster_number):
        self.viz.select_cluster(cluster_number)

    def select_viewpoint(self, cluster_number):
        self.viz.select_cluster(cluster_number)

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
        # adding all widgets (set_on_layout is registered before init_layouts).
        if not hasattr(self, 'main_layout'):
            return

        em = self.window.theme.font_size
        margin = 0.5 * em

        r = self.window.content_rect

        self.scene_widget.frame = r

        main_width = 30 * em
        main_height = (r.height - 2 * margin)/2

        panel_x = r.width - main_width - margin

        # YAML file list — top-right, size to content, capped at 35 % of panel
        file_list_h = min(
            self.file_list.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height,
            main_height * 0.35)
        self.file_list.frame = gui.Rect(
            panel_x, r.y + margin, main_width, file_list_h)
        self.file_list.visible = True

        # Main layout fills the right column below the file list, expanding up
        # into the space the regions tree used to occupy.
        main_top = r.y + margin + file_list_h + margin * 0.5
        self.main_layout.frame = gui.Rect(
            panel_x,
            main_top,
            main_width,
            max(0, r.y + r.height - margin - main_top))

        # Regions tree — full-height panel on the left side of the window,
        # slightly narrower than the right-hand panels.
        tree_width = 24 * em
        self.file_tree.frame = gui.Rect(
            r.x + margin,
            r.y + margin,
            tree_width,
            max(0, r.height - 2 * margin))

        # Task planning panel — always-visible horizontal bar pinned to the
        # bottom, spanning the gap between the regions tree (left) and the main
        # panel (right). Short height sized to its content.
        if self.task_planning_panel is not None:
            tp_left = r.x + margin + tree_width + margin
            tp_right = panel_x - margin
            tp_width = max(0, tp_right - tp_left)
            tp_height = self.task_planning_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height + em
            tp_top = r.y + r.height - margin - tp_height
            self.task_planning_panel.frame = gui.Rect(
                tp_left, tp_top, tp_width, tp_height)
            self.task_planning_panel.visible = True
            self._apply_task_planning_cell_widths(tp_width, em)

        # Ensure a redraw follows every layout pass so frames set above are
        # actually rendered.  Without this the draw scheduled by the tick event
        # can fire *before* _on_layout processes, leaving panels invisible until
        # the user triggers another draw (e.g. a mouse click).
        self.window.post_redraw()

    def _apply_task_planning_cell_widths(self, tp_width, em):
        """Give each task_planning cell an equal, space-filling share of its
        tab's width. gui.Horiz sizes children to their preferred width with no
        per-child grow factor, so to make the columns span the panel (rather
        than left-justify with empty space on the right) the widths must be
        assigned explicitly at layout time, once the panel width is known.

        ``CHROME`` reserves the row margins plus the TabControl's own padding;
        it is approximate, so a small right-hand gap is preferred over clipping.
        """
        spacing = 0.5 * \
            em       # gui.Horiz inter-cell spacing (see the row's ctor)
        CHROME = 3.0 * em        # row margins + TabControl content inset
        for resizers in getattr(self, '_tp_row_resizers', []):
            n = len(resizers)
            if n == 0:
                continue
            cell_w = max(4 * em, (tp_width - CHROME - (n - 1) * spacing) / n)
            for resize in resizers:
                resize(cell_w)

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
            elif hasattr(widget, 'selected_index'):
                pass  # Combobox: selection driven only by user callbacks
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
                            param_range = parameters_dict[node_name][param_name].get(
                                'range')
                            self.set_widget_value(
                                widget, param_value, limits=param_range)
                            companion_key = f"{node_name}/{param_name}"
                            if companion_key in self._companion_widgets:
                                self.set_widget_value(
                                    self._companion_widgets[companion_key],
                                    param_value, limits=param_range)

                            # Selection state lives on the task_planning node;
                            # results.file (mesh/pcd/regions geometry) lives on
                            # viewpoint_generation. Gate by node so the two
                            # nodes' identically named params don't cross-drive.
                            if (node_name == self.ros_thread.viewpoint_generation_node_name
                                    and 'results.file' in param_name):
                                self.visualize_results(param_value)
                            elif node_name == self.ros_thread.task_planning_node_name and param_name == 'navigation.selected_mesh':
                                self.select_mesh(param_value)
                            elif node_name == self.ros_thread.task_planning_node_name and param_name == 'navigation.selected_region':
                                self.select_region(param_value)
                            elif node_name == self.ros_thread.task_planning_node_name and param_name == 'navigation.selected_viewpoint':
                                self.select_cluster(param_value)
                            elif node_name == self.ros_thread.task_planning_node_name and param_name == 'navigation.selected_traversal_algorithm':
                                self.viz.set_traversal_algorithm(param_value)
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
