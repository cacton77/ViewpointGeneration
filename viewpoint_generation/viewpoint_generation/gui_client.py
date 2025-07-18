#!/usr/bin/env python3
import rclpy
import sys
import copy
import time
import json
import random
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from pprint import pprint
from matplotlib import colormaps

from viewpoint_generation.threads.ros_client import ROSThread
from viewpoint_generation.assets.materials import Materials

sys.stdout.reconfigure(line_buffering=True)
isMacOS = sys.platform == 'darwin'


class GUIClient():

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
    MENU_SHOW_RETICLE = 14
    MENU_SHOW_MESH = 15
    MENU_SHOW_POINT_CLOUD = 16
    MENU_SHOW_CURVATURES = 17
    MENU_SHOW_REGIONS = 18
    MENU_SHOW_NOISE_POINTS = 19
    MENU_SHOW_FOV_CLUSTERS = 20
    MENU_SHOW_VIEWPOINTS = 21
    MENU_SHOW_SETTINGS = 22
    MENU_SHOW_ERRORS = 23
    MENU_SHOW_PATH = 24
    MENU_ABOUT = 25

    camera_updated = False
    camera_fov_width = 0.03
    camera_fov_height = 0.02
    last_intersection_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(self):
        self.app = gui.Application.instance
        self.window = self.app.create_window(
            "Viewpoint Generation", width=1280, height=720, x=0, y=30)

        self.fps = 30

        em = self.window.theme.font_size
        r = self.window.content_rect
        self.menu_height = 2.5 * em
        self.header_height = 3 * em
        self.footer_height = 10 * em

        self.ros_thread = ROSThread()
        self.ros_thread.start()

        # Wait until self.ros_thread.parameters_dict is populated
        while not self.ros_thread.parameters_dict:
            print("Waiting for parameters from ROS...")
            time.sleep(0.1)
        self.parameters_dict = self.ros_thread.expand_dict_keys()

        w = self.window
        self.window.set_on_close(self.on_main_window_closing)
        self.update_delay = -1  # Set to -1 to use tick event
        self.window.set_on_tick_event(self.on_main_window_tick_event)

        # 3D SCENE ################################################################
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.scene_widget.enable_scene_caching(False)
        self.scene_widget.scene.show_axes(True)
        self.scene_widget.scene.show_ground_plane(
            True, o3d.visualization.rendering.Scene.GroundPlane.XY)
        self.scene_widget.scene.set_background(
            [0.1, 0.1, 0.1, 1.0])
        # Set view
        self.scene_widget.look_at(
            np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([0, 0, 1]))

        self.ray_casting_scene = o3d.t.geometry.RaycastingScene()

        self.window.add_child(self.scene_widget)

        self.parameter_widgets = {}

        self.init_gui()
        self.init_menu_bar()
        self.setup_multi_directional_lighting()
        # self.window.add_child(self.parameter_panel)
        self.window.set_on_layout(self._on_layout)

        camera = self.scene_widget.scene.camera
        view_matrix = camera.get_view_matrix()
        self.current_view_matrix = view_matrix

        self.last_draw_time = time.time()

    def setup_multi_directional_lighting(self):
        """Configure multiple light sources for even illumination"""

        # Get the scene from the widget
        scene = self.scene_widget.scene

        # Set lighting profile with required sun direction
        # Normalized [1,-1,-1]
        sun_direction = np.array([0.577, -0.577, -0.577], dtype=np.float32)
        scene.set_lighting(scene.LightingProfile.NO_SHADOWS, sun_direction)

        # Method 1: Add multiple directional lights
        # self.add_directional_lights(scene)

        # Method 2: Add ambient + directional combination
        # self.add_ambient_plus_directional(scene)

        # Method 3: Add point lights at multiple positions
        # self.add_multiple_point_lights(scene)

    def add_directional_lights(self, scene):
        """Add directional lights from multiple angles"""

        # Light intensity (adjust as needed)
        intensity = 50000

        # Light directions (normalized vectors pointing TO the object)
        light_directions = [
            [1, -1, -1],    # From top-right-front
            [-1, -1, -1],   # From top-left-front
            [1, -1, 1],     # From top-right-back
            [-1, -1, 1],    # From top-left-back
            [0, 1, 0],      # From below
            [0, 0, -1],     # From front (towards viewer)
        ]

        # Add each directional light
        for i, direction in enumerate(light_directions):
            # Normalize direction
            direction = np.array(direction)
            direction = direction / np.linalg.norm(direction)

            # Add directional light
            scene.add_directional_light(
                name=f"directional_light_{i}",
                color=[1.0, 1.0, 1.0],  # White light
                direction=direction.tolist(),
                intensity=intensity
            )

        print(f"Added {len(light_directions)} directional lights")

    def add_ambient_plus_directional(self, scene):
        """Combine ambient lighting with a few directional lights"""

        # Add ambient light for overall illumination
        scene.add_ambient_light(
            name="ambient",
            color=[0.3, 0.3, 0.3],  # Soft ambient light
            intensity=30000
        )

        # Add a few key directional lights
        key_lights = [
            {"direction": [1, -1, -1], "intensity": 80000,
                "color": [1.0, 1.0, 1.0]},
            {"direction": [-1, -1, 1], "intensity": 60000,
                "color": [0.9, 0.9, 1.0]},
            {"direction": [0, 1, 0], "intensity": 40000,
                "color": [1.0, 0.9, 0.9]},
        ]

        for i, light in enumerate(key_lights):
            direction = np.array(light["direction"])
            direction = direction / np.linalg.norm(direction)

            scene.add_directional_light(
                name=f"key_light_{i}",
                color=light["color"],
                direction=direction.tolist(),
                intensity=light["intensity"]
            )

        print("Added ambient + 3 directional lights")

    def add_multiple_point_lights(self, scene):
        """Add point lights at strategic positions around the object"""

        # Estimate object bounds (you might want to pass this in)
        # For demo, assume object is centered at origin with size ~2 units
        object_center = [0, 0, 0]
        light_distance = 5.0  # Distance from object
        intensity = 100000

        # Point light positions (around the object)
        light_positions = [
            [light_distance, light_distance, light_distance],      # Top-front-right
            [-light_distance, light_distance, light_distance],     # Top-front-left
            [light_distance, light_distance, -light_distance],     # Top-back-right
            [-light_distance, light_distance, -light_distance],    # Top-back-left
            # Bottom-front-right
            [light_distance, -light_distance, light_distance],
            [-light_distance, -light_distance,
                light_distance],    # Bottom-front-left
            [0, 0, light_distance * 1.5],                          # Front center
            [0, 0, -light_distance * 1.5],                         # Back center
        ]

        for i, position in enumerate(light_positions):
            scene.add_point_light(
                name=f"point_light_{i}",
                color=[1.0, 1.0, 1.0],
                position=position,
                intensity=intensity,
                falloff=2.0,  # How quickly light falls off with distance
                light_falloff_radius=light_distance * 2
            )

        # Add some ambient light to fill in shadows
        scene.add_ambient_light(
            name="ambient_fill",
            color=[0.2, 0.2, 0.2],
            intensity=20000
        )

        print(f"Added {len(light_positions)} point lights + ambient")

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
            file_menu.add_item("Import Model", self.MENU_IMPORT_MODEL)
            file_menu.add_item("Import Point Cloud", self.MENU_IMPORT_PCD)
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
            view_menu.set_checked(self.MENU_SHOW_MODEL_BB, True)
            view_menu.add_item("Show Reticle", self.MENU_SHOW_RETICLE)
            view_menu.set_checked(self.MENU_SHOW_RETICLE, True)
            ground_plane_menu = gui.Menu()
            ground_plane_menu.add_item("XY", 100)
            ground_plane_menu.add_item("XZ", 101)
            ground_plane_menu.add_item("YZ", 102)
            view_menu.add_menu("Ground Plane", ground_plane_menu)
            view_menu.add_separator()
            # Object display options
            view_menu.add_item("Show Model", self.MENU_SHOW_MESH)
            view_menu.set_checked(self.MENU_SHOW_MESH, True)
            view_menu.add_item("Show Point Clouds",
                               self.MENU_SHOW_POINT_CLOUD)
            view_menu.set_checked(self.MENU_SHOW_POINT_CLOUD, False)
            view_menu.add_item("Show Curvatures",
                               self.MENU_SHOW_CURVATURES)
            view_menu.set_checked(self.MENU_SHOW_CURVATURES, False)
            view_menu.add_item("Show Regions", self.MENU_SHOW_REGIONS)
            view_menu.set_checked(self.MENU_SHOW_REGIONS, False)
            view_menu.add_item("Show Noise Points",
                               self.MENU_SHOW_NOISE_POINTS)
            view_menu.set_checked(self.MENU_SHOW_NOISE_POINTS, False)
            view_menu.add_item("Show FOV Clusters",
                               self.MENU_SHOW_FOV_CLUSTERS)
            view_menu.set_checked(self.MENU_SHOW_FOV_CLUSTERS, False)
            view_menu.add_item("Show Viewpoints", self.MENU_SHOW_VIEWPOINTS)
            view_menu.set_checked(self.MENU_SHOW_VIEWPOINTS, False)

            view_menu.add_item("Show Path", self.MENU_SHOW_PATH)
            view_menu.set_checked(self.MENU_SHOW_PATH, False)
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
        # w.set_on_menu_item_activated(
        #     self.MENU_SAVE, self._on_menu_save)
        # w.set_on_menu_item_activated(
        #     self.MENU_SAVE_AS, self._on_menu_save_as)
        # w.set_on_menu_item_activated(self.MENU_IMPORT_MODEL,
        #                              self._on_menu_import_model)
        # w.set_on_menu_item_activated(self.MENU_IMPORT_PCD,
        #                              self._on_menu_import_pcd)
        # w.set_on_menu_item_activated(
        #     self.MENU_QUIT, self._on_menu_quit)
        # w.set_on_menu_item_activated(
        #     self.MENU_PREFERENCES, self._on_menu_preferences)
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_AXES, self._on_menu_show_axes)
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_GRID, self._on_menu_show_grid)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_MODEL_BB, self._on_menu_show_model_bounding_box)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_RETICLE, self._on_menu_show_reticle)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_MESH, self._on_menu_show_mesh)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_POINT_CLOUD, self._on_menu_show_point_cloud)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_CURVATURES, self._on_menu_show_curvatures)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_REGIONS, self._on_menu_show_regions)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_FOV_CLUSTERS, self._on_menu_show_fov_clusters)
        w.set_on_menu_item_activated(
            self.MENU_SHOW_NOISE_POINTS, self._on_menu_show_noise_points)
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_PATH, self._on_menu_show_path)
        w.set_on_menu_item_activated(self.MENU_SHOW_VIEWPOINTS,
                                     self._on_menu_show_viewpoints)
        # w.set_on_menu_item_activated(self.MENU_SHOW_SETTINGS,
        #                              self._on_menu_toggle_settings_panel)
        # w.set_on_menu_item_activated(
        #     self.MENU_ABOUT, self._on_menu_about)
        # ----

    def _on_menu_show_model_bounding_box(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_MODEL_BB)

        self.show_model_bounding_box(show)

    def _on_menu_show_reticle(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_RETICLE)

        self.show_reticle(show)

    def _on_menu_show_mesh(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_MESH)
        self.show_mesh(show)

    def _on_menu_show_point_cloud(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_POINT_CLOUD)
        self.show_point_cloud(show)

    def _on_menu_show_curvatures(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_CURVATURES)
        self.show_curvatures(show)

    def _on_menu_show_regions(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_REGIONS)
        self.show_regions(show)

    def _on_menu_show_fov_clusters(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_FOV_CLUSTERS)
        self.show_fov_clusters(show)

    def _on_menu_show_viewpoints(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_VIEWPOINTS)
        self.show_viewpoints(show)

    def _on_menu_show_noise_points(self):
        show = not gui.Application.instance.menubar.is_checked(
            self.MENU_SHOW_NOISE_POINTS)
        self.show_noise_points(show)

    def show_model_bounding_box(self, show=True):
        # Show/hide model bounding box.
        if show:
            self.scene_widget.scene.show_geometry('model_bounding_box', True)
        else:
            self.scene_widget.scene.show_geometry('model_bounding_box', False)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_MODEL_BB, show)

    def show_reticle(self, show=True):
        self.scene_widget.scene.show_geometry('reticle', show)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_RETICLE, show)

    def show_mesh(self, show=True):
        # Show/hide mesh.
        self.scene_widget.scene.show_geometry('mesh', show)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_MESH, show)

    def show_point_cloud(self, show=True):
        # Show/hide original point cloud. Hide curvatures, regions, and noise points if true.
        self.scene_widget.scene.show_geometry('point_cloud', show)
        if show:
            self.show_curvatures(False)
            self.show_regions(False)
            self.show_noise_points(False)
            self.show_fov_clusters(False)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_POINT_CLOUD, show)

    def show_curvatures(self, show=True):
        # Show/hide curvatures. If showing, hide regions and noise points.
        self.scene_widget.scene.show_geometry('curvatures', show)
        if show:
            self.show_point_cloud(False)
            self.show_regions(False)
            self.show_noise_points(False)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_CURVATURES, show)

    def show_regions(self, show=True):
        self.scene_widget.scene.show_geometry('regions', show)
        if show:
            self.show_point_cloud(False)
            self.show_curvatures(False)
            self.show_noise_points(True)
            self.show_fov_clusters(False)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_REGIONS, show)

    def show_fov_clusters(self, show=True):
        self.scene_widget.scene.show_geometry('fov_clusters', show)
        if show:
            self.show_point_cloud(False)
            self.show_curvatures(False)
            self.show_regions(False)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_FOV_CLUSTERS, show)

    def show_viewpoints(self, show=True):
        """Show or hide the viewpoints in the scene."""
        self.scene_widget.scene.show_geometry('viewpoints', show)
        self.scene_widget.scene.show_geometry('region_view_meshes', show)

        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_VIEWPOINTS, show)

    def show_noise_points(self, show=True):
        self.scene_widget.scene.show_geometry('noise_points', show)
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_NOISE_POINTS, show)

    def init_gui(self):
        """Initialize the Open3D GUI"""
        # Get theme for consistent styling
        theme = self.window.theme
        em = theme.font_size

        # Create main layout
        self.main_layout = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        self.main_layout.background_color = Materials.panel_color

        # Create tab widget
        self.tab_widget = gui.TabControl()
        self.tab_widget.background_color = Materials.panel_color

        # Create tabs for each top-level key
        for tab_name, tab_data in self.parameters_dict.items():
            tab_panel = self.create_tab_panel(tab_name, tab_data, em)
            self.tab_widget.add_tab(tab_name.title(), tab_panel)

        # Add tab widget to main layout
        self.main_layout.add_child(self.tab_widget)

        # Set the main layout
        self.window.add_child(self.main_layout)

    def create_tab_panel(self, tab_name, tab_data, em):
        """Create a panel for a tab with nested structure"""
        # Create scrollable area for the tab content
        scroll_area = gui.Vert(
            0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 1.25 * em, 0.25 * em))
        scroll_area.background_color = Materials.panel_color

        # Create the content recursively
        content = self.create_nested_content(tab_name, tab_data, em)
        content.background_color = Materials.panel_color
        scroll_area.add_child(content)

        return scroll_area

    def create_nested_content(self, parent_name, data, em, level=0):
        """Recursively create nested content for parameters"""
        container = gui.Vert(
            0.25 * em, gui.Margins(0.25 * em, 0.00 * em, 0.00 * em, 0.00 * em))

        # If this is a leaf parameter (has 'name', 'type', 'value')
        if isinstance(data, dict) and 'name' in data and 'type' in data and 'value' in data:
            widget_grid = self.create_parameter_widget(data, em)
            container.add_child(widget_grid)
        else:
            # This is a nested structure, process each sub-item
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'name' in value and 'type' in value and 'value' in value:
                        # This is a parameter
                        widget_grid = self.create_parameter_widget(value, em)
                        container.add_child(widget_grid)
                    else:
                        # This is a nested group, create a collapsible section
                        section = self.create_collapsible_section(
                            key, value, em, level)
                        container.add_child(section)

        return container

    def create_collapsible_section(self, section_name, section_data, em, level):
        """Create a collapsible section for nested parameters"""
        # Create a collapsible widget
        collapsible = gui.CollapsableVert(section_name.replace('_', ' ').title(
        ), 0.25 * em, gui.Margins(0.25 * em, 0.00 * em, 0.00 * em, 0.00 * em))

        # Add content to the collapsible section
        content = self.create_nested_content(
            section_name, section_data, em, level + 1)

        button_horiz = gui.Horiz(
            0.25 * em, gui.Margins(0.75 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # If section_name is 'Sampling', add a button
        if section_name == 'sampling':
            button = gui.Button("Sample Point Cloud")
            button.set_on_clicked(lambda: self.ros_thread.sample_point_cloud())
            content.add_child(button)
        elif section_name == 'curvature':
            button = gui.Button("Compute Curvature")
            button.set_on_clicked(lambda: self.ros_thread.estimate_curvature())
            content.add_child(button)
        elif section_name == 'region_growth':
            button = gui.Button("Run Region Growth")
            button.set_on_clicked(lambda: self.ros_thread.region_growth())
            content.add_child(button)
        elif section_name == 'fov_clustering':
            button = gui.Button("Run FOV Clustering")
            button.set_on_clicked(lambda: self.ros_thread.fov_clustering())
            content.add_child(button)
        elif section_name == 'projection':
            button = gui.Button("Project Viewpoints")
            button.set_on_clicked(lambda: self.ros_thread.project_viewpoints())
            content.add_child(button)

        collapsible.add_child(content)

        # Expand by default for first level
        if level == 0:
            collapsible.set_is_open(True)

        return collapsible

    def create_parameter_widget(self, param_data, em):
        """Create a widget for a single parameter"""
        # Create a grid layout for label and widget
        grid = gui.VGrid(2, 0.25 * em, gui.Margins(0.75 * em,
                         0.25 * em, 0.25 * em, 0.25 * em))

        param_name = param_data['name']
        param_type = param_data['type']
        param_value = param_data['value']

        # Create label
        label = gui.Label(param_name.split('.')[-1].replace('_', ' ').title())
        grid.add_child(label)

        # Create appropriate widget based on type
        widget = None

        if param_type == 'bool':
            widget = gui.Checkbox("")
            widget.checked = bool(param_value)
            widget.set_on_checked(
                lambda checked, name=param_name: self.on_parameter_changed(name, checked))

        elif param_type == 'integer':
            widget = gui.NumberEdit(gui.NumberEdit.INT)
            widget.int_value = int(param_value)
            widget.set_on_value_changed(
                lambda value, name=param_name: self.on_parameter_changed(name, value))

        elif param_type == 'double':
            widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            widget.double_value = float(param_value)
            widget.set_on_value_changed(
                lambda value, name=param_name: self.on_parameter_changed(name, value))

        elif param_type == 'string':
            if 'file' in param_name.lower() or 'path' in param_name.lower():
                # Create a horizontal layout for file path + browse button
                file_layout = gui.Horiz(0.25 * em)

                widget = gui.TextEdit()
                widget.text_value = str(param_value)
                widget.set_on_text_changed(
                    lambda text, name=param_name: self.on_parameter_changed(name, text))

                browse_button = gui.Button("...")
                browse_button.horizontal_padding_em = 0.5
                browse_button.vertical_padding_em = 0
                browse_button.set_on_clicked(
                    lambda name=param_name: self.on_browse_file(name))

                file_layout.add_child(widget)
                # Add some space between text and button
                file_layout.add_fixed(0.25 * em)
                file_layout.add_child(browse_button)

                grid.add_child(file_layout)
            elif 'unit' in param_name.lower():
                # Create a dropdown for unit selection
                layout = gui.Horiz(0.25 * em)
                widget = gui.Combobox()
                units = ['m', 'cm', 'mm', 'in', 'ft']
                for unit in units:
                    widget.add_item(unit)
                widget.selected_index = units.index(param_value)

                def on_unit_changed(selected_text, selected_index):
                    """Handle unit selection change"""
                    self.on_parameter_changed(param_name, selected_text)

                widget.set_on_selection_changed(on_unit_changed)

                layout.add_child(widget)
                grid.add_child(layout)
            else:
                widget = gui.TextEdit()
                widget.text_value = str(param_value)
                widget.set_on_text_changed(
                    lambda text, name=param_name: self.on_parameter_changed(name, text))
                grid.add_child(widget)

        else:
            # Default to text edit for unknown types
            widget = gui.TextEdit()
            widget.text_value = str(param_value)
            widget.set_on_text_changed(
                lambda text, name=param_name: self.on_parameter_changed(name, text))
            grid.add_child(widget)

        # Store widget reference
        if widget is not None and param_type != 'string' or 'file' not in param_name.lower():
            self.parameter_widgets[param_name] = widget
            grid.add_child(widget)
        else:
            self.parameter_widgets[param_name] = widget

        return grid

    def init_gui(self):
        """Initialize the Open3D GUI with fixed tabs"""
        # Get theme for consistent styling
        theme = self.window.theme
        em = theme.font_size

        # Create tab widget directly - NO intermediate scrollable layout
        self.main_layout = gui.TabControl()
        self.main_layout.background_color = Materials.panel_color

        # Create tabs for each top-level key
        for tab_name, tab_data in self.parameters_dict.items():
            tab_panel = self.create_tab_panel(tab_name, tab_data, em)
            self.main_layout.add_tab(tab_name.title(), tab_panel)

        # Add tab widget DIRECTLY to window - no intermediate layout
        self.window.add_child(self.main_layout)

    def create_tab_panel(self, tab_name, tab_data, em):
        """Create a scrollable panel for a tab - ONLY scrolling here"""
        # Create scrollable area directly - no intermediate containers
        scroll_area = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 1.25 * em, 0.25 * em))
        scroll_area.background_color = Materials.panel_color

        # Create the content recursively
        content = self.create_nested_content(tab_name, tab_data, em)
        content.background_color = Materials.panel_color
        scroll_area.add_child(content)

        return scroll_area

    # Alternative approach: Use ScrollableVert directly as tab content
    def create_tab_panel_simple(self, tab_name, tab_data, em):
        """Simplified version using ScrollableVert directly"""
        # Create scrollable area that will be the tab content
        scroll_area = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 1.25 * em, 0.25 * em))
        scroll_area.background_color = Materials.panel_color

        # Create the content recursively
        content = self.create_nested_content(tab_name, tab_data, em)
        content.background_color = Materials.panel_color
        scroll_area.add_child(content)

        return scroll_area

    # If you want more control, you can also structure it this way:
    def init_gui_advanced(self):
        """Advanced layout with explicit control over fixed and scrollable areas"""
        theme = self.window.theme
        em = theme.font_size

        # Create main vertical layout
        self.main_layout = gui.Vert(0, gui.Margins(0.5 * em))
        self.main_layout.background_color = Materials.panel_color

        # Create fixed header area for tabs
        header_area = gui.Vert(0, gui.Margins(0))
        header_area.background_color = Materials.panel_color

        # Create tab widget in header (fixed)
        self.tab_widget = gui.TabControl()
        self.tab_widget.background_color = Materials.panel_color

        # Create tabs
        for tab_name, tab_data in self.parameters_dict.items():
            # Each tab panel will handle its own scrolling
            tab_panel = self.create_scrollable_tab_content(
                tab_name, tab_data, em)
            self.tab_widget.add_tab(tab_name.title(), tab_panel)

        # Add tabs to header
        header_area.add_child(self.tab_widget)

        # Add header to main layout (this makes tabs fixed)
        self.main_layout.add_child(header_area)

        # Set the layout
        self.window.add_child(self.main_layout)

    def create_scrollable_tab_content(self, tab_name, tab_data, em):
        """Create tab content with internal scrolling"""
        # Create a container for the entire tab
        tab_container = gui.Vert(0, gui.Margins(0))

        # Create scrollable content area
        scrollable_content = gui.ScrollableVert(
            0.5 * em,
            gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )
        scrollable_content.background_color = Materials.panel_color

        # Create the actual content
        content = self.create_nested_content(tab_name, tab_data, em)
        scrollable_content.add_child(content)

        # Add scrollable content to tab container
        tab_container.add_child(scrollable_content)

        return tab_container

    # Updated create_nested_content with better spacing
    def create_nested_content(self, parent_name, data, em, level=0):
        """Recursively create nested content for parameters"""
        container = gui.Vert(
            0.25 * em, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        container.background_color = Materials.panel_color

        # If this is a leaf parameter (has 'name', 'type', 'value')
        if isinstance(data, dict) and 'name' in data and 'type' in data and 'value' in data:
            widget_grid = self.create_parameter_widget(data, em)
            container.add_child(widget_grid)
        else:
            # This is a nested structure, process each sub-item
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'name' in value and 'type' in value and 'value' in value:
                        # This is a parameter
                        widget_grid = self.create_parameter_widget(value, em)
                        container.add_child(widget_grid)
                    else:
                        # This is a nested group, create a collapsible section
                        section = self.create_collapsible_section(
                            key, value, em, level)
                        container.add_child(section)

        return container

    # Optional: Add a fixed footer or status bar
    def init_gui_with_footer(self):
        """GUI with fixed header (tabs) and optional fixed footer"""
        theme = self.window.theme
        em = theme.font_size

        # Main layout
        self.main_layout = gui.Vert(0, gui.Margins(0.5 * em))
        self.main_layout.background_color = Materials.panel_color

        # Fixed header with tabs
        self.tab_widget = gui.TabControl()
        self.tab_widget.background_color = Materials.panel_color

        for tab_name, tab_data in self.parameters_dict.items():
            tab_panel = self.create_scrollable_tab_content(
                tab_name, tab_data, em)
            self.tab_widget.add_tab(tab_name.title(), tab_panel)

        # Add tabs to main layout (fixed at top)
        self.main_layout.add_child(self.tab_widget)

        # Optional: Add fixed footer/status bar
        if hasattr(self, 'show_status_bar') and self.show_status_bar:
            footer = gui.Horiz(0.25 * em, gui.Margins(0.5 * em))
            footer.background_color = Materials.panel_color

            status_label = gui.Label("Ready")
            footer.add_child(status_label)

            # Add some stretch space
            footer.add_stretch()

            # Add status info
            info_label = gui.Label("Parameters loaded")
            footer.add_child(info_label)

            # Add footer to main layout (fixed at bottom)
            self.main_layout.add_child(footer)

        self.window.add_child(self.main_layout)

    # Quick fix version - just replace your create_tab_panel method with this:
    def create_tab_panel_fixed(self, tab_name, tab_data, em):
        """Fixed version - use ScrollableVert instead of regular Vert"""
        # Use ScrollableVert instead of regular Vert
        scroll_area = gui.ScrollableVert(
            0.5 * em, gui.Margins(0.25 * em, 0.25 * em, 1.25 * em, 0.25 * em))
        scroll_area.background_color = Materials.panel_color

        # Create the content recursively (same as before)
        content = self.create_nested_content(tab_name, tab_data, em)
        content.background_color = Materials.panel_color
        scroll_area.add_child(content)

        return scroll_area

    def on_parameter_changed(self, param_name, new_value):
        """Handle parameter value changes"""
        print(f"Parameter {param_name} changed to: {new_value}")
        # Update the internal dictionary
        self.update_parameter_value(param_name, new_value)
        # Set Parameter via ROS thread
        self.set_parameter(param_name, new_value)

    def update_parameter_value(self, param_name, new_value):
        """Update parameter value in the nested dictionary"""
        keys = param_name.split('.')
        current = self.parameters_dict

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

    def on_browse_file(self, param_name):
        """Handle file browse button clicks"""
        file_dialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose file", self.window.theme)
        file_dialog.set_on_cancel(self.window.close_dialog)
        file_dialog.set_on_done(
            lambda path: self.on_file_selected(param_name, path))
        self.window.show_dialog(file_dialog)

    def on_file_selected(self, param_name, file_path):
        """Handle file selection"""
        print(f"File selected for {param_name}: {file_path}")
        self.on_parameter_changed(param_name, file_path)

        # Update the widget display
        if param_name in self.parameter_widgets:
            widget = self.parameter_widgets[param_name]
            if hasattr(widget, 'text_value'):
                widget.text_value = file_path

        self.window.close_dialog()  # Close the dialog after selection

    def on_sample_point_cloud(self):
        """Handle sample point cloud button click"""
        self.ros_thread.sample_point_cloud()

    def set_parameter(self, parameter_name, new_value):
        self.ros_thread.set_parameter(parameter_name, new_value)

    def update_all_widgets_from_dict(self):
        """Update all widget values from the current parameter dictionary"""

        parameters_dict = self.ros_thread.parameters_dict
        parameters_updated = False

        for param_name, widget in self.parameter_widgets.items():
            if param_name in parameters_dict:
                update_flag = parameters_dict[param_name]['update_flag']
                if update_flag:
                    parameters_updated = True
                    widget = self.parameter_widgets[param_name]
                    # Update the widget value from the parameters dictionary
                    if 'value' in parameters_dict[param_name]:
                        param_value = parameters_dict[param_name]['value']
                        self.set_widget_value(widget, param_value)

                        # if param_name is 'model.mesh.file' load the mesh
                        if 'model.mesh.file' in param_name:
                            self.import_mesh(param_value)
                        elif 'model.mesh.units' in param_name:
                            # If units change, we need to rescale the mesh
                            mesh_file = self.ros_thread.parameters_dict['model.mesh.file']['value']
                            if mesh_file:
                                self.import_mesh(mesh_file)
                        elif 'model.point_cloud.file' in param_name:
                            self.import_point_cloud(param_value)
                        elif 'model.point_cloud.units' in param_name:
                            # If units change, we need to rescale the point cloud
                            pcd_file = self.ros_thread.parameters_dict['model.point_cloud.file']['value']
                            if pcd_file:
                                self.import_point_cloud(pcd_file)
                        elif 'regions.region_growth.curvature.file' in param_name:
                            self.import_curvature(param_value)
                        elif 'regions.file' in param_name:
                            self.import_regions(param_value)
                        elif 'model.camera.fov.height' in param_name:
                            self.camera_fov_height = param_value
                            self.camera_updated = True
                        elif 'model.camera.fov.width' in param_name:
                            self.camera_fov_width = param_value
                            self.camera_updated = True

                        parameters_dict[param_name]['update_flag'] = False
                        print(f"Updated \'{param_name}\' to \'{param_value}\'")

        if parameters_updated:
            print("------------------------------------")

        self.ros_thread.parameters_dict = parameters_dict

        # def update_recursive(data_dict, prefix=""):
        #     """Recursively find and update parameters"""
        #     parameters_updated = False

        #     for key, value in data_dict.items():
        #         if isinstance(value, dict):
        #             if 'name' in value and 'type' in value and 'value' in value:
        #                 # This is a parameter - update its widget
        #                 param_name = value['name']
        #                 param_value = value['value']
        #                 # Check if the parameter has an update flag
        #                 # If it does, only update if the flag is True
        #                 param_update_flag = value['update_flag']
        #                 parameters_updated = parameters_updated or param_update_flag

        #                     # Turn update flag off after updating
        #                     value['update_flag'] = False
        #             else:
        #                 # This is a nested structure - recurse
        #                 new_prefix = f"{prefix}.{key}" if prefix else key
        #                 parameters_updated = parameters_updated or update_recursive(
        #                     value, new_prefix)

        #     return parameters_updated

        # # Start the recursive update
        # parameters_updated = update_recursive(self.parameters_dict)

        # self.ros_thread.collapse_dict_keys(self.parameters_dict)

    def import_mesh(self, file_path):
        print(f"Importing mesh from {file_path}")
        # Remove point cloud if it exists
        self.point_cloud = None
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            if mesh.is_empty():
                print(f"Warning: Mesh file {file_path} is empty or invalid.")
            else:

                mesh_units = self.ros_thread.parameters_dict['model.mesh.units']['value']
                if mesh_units == 'mm':
                    mesh.scale(1.0, center=(0, 0, 0))
                elif mesh_units == 'cm':
                    mesh.scale(10, center=(0, 0, 0))
                elif mesh_units == 'm':
                    mesh.scale(1000, center=(0, 0, 0))
                elif mesh_units == 'in':
                    mesh.scale(25.4, center=(0, 0, 0))

                # Create model bounding box
                bbox = mesh.get_axis_aligned_bounding_box()

                self.scene_widget.scene.remove_geometry("model_bounding_box")
                self.scene_widget.scene.add_geometry(
                    "model_bounding_box", bbox, Materials.bounding_box_material)

                self.scene_widget.scene.remove_geometry(
                    "mesh")  # Remove previous mesh if exists
                self.scene_widget.scene.remove_geometry("point_cloud")
                self.scene_widget.scene.remove_geometry("fov_clusters")
                self.scene_widget.scene.remove_geometry("noise_points")
                self.scene_widget.scene.add_geometry(
                    "mesh", mesh, Materials.mesh_material)

                self.ray_casting_scene = o3d.t.geometry.RaycastingScene()
                self.ray_casting_scene.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(mesh))

                # Set camera view to fit the mesh
                bb = mesh.get_axis_aligned_bounding_box()
                self.scene_widget.look_at(
                    bb.get_center(), bb.get_max_bound() * 1.5, np.array([0, 0, 1]))

                self.show_mesh(True)
                self.show_point_cloud(False)
                self.show_curvatures(False)
                self.show_regions(False)
                self.show_fov_clusters(False)
                self.show_noise_points(False)
                self.show_viewpoints(False)

        except Exception as e:
            print(f"Error loading mesh from {file_path}: {e}")


    def import_point_cloud(self, file_path):
        if file_path is None or file_path == "":
            print("Point cloud empty or not specified.")
            return

        print(f"Importing point cloud from {file_path}")

        try:
            point_cloud = o3d.io.read_point_cloud(file_path)
            if point_cloud.is_empty():
                print(
                    f"Warning: Point cloud file {file_path} is empty or invalid.")
            else:

                pcd_units = self.ros_thread.parameters_dict['model.point_cloud.units']['value']
                if pcd_units == 'mm':
                    point_cloud.scale(1.0, center=(0, 0, 0))
                elif pcd_units == 'cm':
                    point_cloud.scale(10, center=(0, 0, 0))
                elif pcd_units == 'm':
                    point_cloud.scale(1000, center=(0, 0, 0))
                elif pcd_units == 'in':
                    point_cloud.scale(25.4, center=(0, 0, 0))

            # Remove previous point cloud if exists
            self.scene_widget.scene.remove_geometry("point_cloud")
            self.scene_widget.scene.remove_geometry("curvatures")
            self.scene_widget.scene.remove_geometry("regions")
            self.scene_widget.scene.remove_geometry("fov_clusters")
            self.scene_widget.scene.remove_geometry("noise_points")
            self.scene_widget.scene.add_geometry(
                "point_cloud", point_cloud, Materials.point_cloud_material)

            self.point_cloud = point_cloud  # Store the point cloud for later use

        except Exception as e:
            print(f"Error loading point cloud from {file_path}: {e}")

        self.show_point_cloud(True)

    def import_curvature(self, file_path):
        print(f"Importing curvature data from {file_path}")
        """ Load curvature data from file and color point cloud based on curvature data"""
        curvatures_cloud = copy.deepcopy(self.point_cloud)
        try:
            curvature = np.load(file_path)

            max_curvature = np.max(curvature)
            min_curvature = np.min(curvature)

            normalized_curvature = (
                curvature - min_curvature) / (max_curvature - min_curvature)

            cmap = colormaps[Materials.curvature_colormap]

            for i in range(len(normalized_curvature)):
                val = 1 - normalized_curvature[i]
                # color = np.array(list(cmap(val)))[0, 0:3]  # Get RGB values
                color = np.array(list(cmap(val)))[0:3]
                np.asarray(curvatures_cloud.colors)[i] = color

            # Remove previous point cloud if exists
            self.scene_widget.scene.remove_geometry("curvatures")
            self.scene_widget.scene.remove_geometry("regions")
            self.scene_widget.scene.remove_geometry("fov_clusters")
            self.scene_widget.scene.remove_geometry("noise_points")
            self.scene_widget.scene.add_geometry(
                "curvatures", curvatures_cloud, Materials.point_cloud_material)

            self.show_curvatures(True)

        except Exception as e:
            print(f"Error loading curvature data from {file_path}: {e}")

    def import_regions(self, file_path):
        if file_path is None or file_path == "":
            print("Regions file empty or not specified.")
            return
        
        print(f"Importing regions from {file_path}")
        """ Load regions from file and paint point cloud based on regions """
        regions_cloud = copy.deepcopy(self.point_cloud)
        try:

            self.point_cloud.paint_uniform_color((1, 1, 1))

            regions_dict = json.load(open(file_path, 'r'))

            colors = np.zeros((len(self.point_cloud.points), 3))

            np.random.seed(42)  # For reproducibility
            fov_meshes = o3d.geometry.TriangleMesh()
            viewpoint_meshes = o3d.geometry.TriangleMesh()
            region_view_clouds = o3d.geometry.PointCloud()
            region_view_meshes = o3d.geometry.TriangleMesh()
            show_clusters = False
            show_viewpoints = False
            for region, region_dict in regions_dict['regions'].items():
                region_indices = region_dict['points']
                region_point_cloud = self.point_cloud.select_by_index(
                    region_indices)
                region_color = np.random.rand(3)

                region_view_cloud = o3d.geometry.PointCloud()
                region_points = np.asarray(region_point_cloud.points)
                region_view_points = []
                region_view_normals = []

                # If dict has 'fov_clusters' key, process and display them
                if 'fov_clusters' in region_dict:
                    show_clusters = True
                    # Iterate over each cluster in fov_clusters
                    for fov_cluster_id, fov_cluster_dict in region_dict['fov_clusters'].items():
                        fov_cluster_points = fov_cluster_dict['points']
                        fov_cluster_color = region_color + \
                            0.1*(np.random.rand(3) - 0.5)
                        fov_cluster_color = np.clip(fov_cluster_color, 0, 1)

                        fov_point_cloud = region_point_cloud.select_by_index(
                            fov_cluster_points)
                        # Remove outliers from fov_point_cloud
                        fov_point_cloud, _ = fov_point_cloud.remove_statistical_outlier(
                            nb_neighbors=20, std_ratio=2.0)
                        fov_mesh = fov_point_cloud.compute_convex_hull(joggle_inputs=True)[
                            0]
                        fov_mesh.paint_uniform_color(fov_cluster_color)
                        fov_mesh.compute_vertex_normals()
                        avg_normal = np.mean(np.asarray(
                            fov_point_cloud.normals), axis=0)
                        fov_mesh.translate(avg_normal * 0.005)

                        if 'viewpoint' in fov_cluster_dict:
                            show_viewpoints = True
                            viewpoint_mesh = o3d.geometry.TriangleMesh.create_sphere(
                                radius=5)
                            viewpoint_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                size=6, origin=[0, 0, 0])
                            origin = 1000*np.array(fov_cluster_dict['viewpoint']['origin'])
                            viewpoint = 1000*np.array(fov_cluster_dict['viewpoint']['viewpoint'])
                            direction = np.array(fov_cluster_dict['viewpoint']['direction'])
                            # Rotate viewpoint_mesh to match the direction
                            viewpoint_mesh.rotate(
                                o3d.geometry.get_rotation_matrix_from_xyz(np.array([
                                    np.arctan2(direction[1], direction[0]),
                                    np.arctan2(direction[2], np.linalg.norm(direction[:2])),
                                    0])
                                ),
                                center=(0, 0, 0)
                            )

                            # Add to region view points
                            region_view_points.append(viewpoint.tolist())
                            region_view_normals.append(direction.tolist())

                            viewpoint_mesh.translate(viewpoint)

                            viewpoint_mesh.paint_uniform_color([1.0, 1.0, 1.0])
                            viewpoint_meshes += viewpoint_mesh

                        fov_meshes += fov_mesh

                region_point_cloud.paint_uniform_color(region_color)
                regions_cloud += region_point_cloud

                # Region View Surface
                if show_viewpoints:
                    region_view_cloud.points = o3d.utility.Vector3dVector(np.array(region_view_points))
                    region_view_cloud.normals = o3d.utility.Vector3dVector(np.array(region_view_normals))
                    if len(region_view_cloud.points) > 4:
                        region_view_mesh = region_view_cloud.compute_convex_hull(
                            joggle_inputs=True)[0]
                        region_view_mesh.paint_uniform_color(region_color)
                        region_view_meshes += region_view_mesh
                        region_view_clouds += region_view_cloud

            # Create noise point cloud
            noise_points = regions_dict['noise_points']
            noise_point_cloud = self.point_cloud.select_by_index(
                noise_points)
            noise_point_cloud.paint_uniform_color(
                [1.0, 0.0, 0.0])  # Red for noise points

            self.scene_widget.scene.remove_geometry("regions")
            self.scene_widget.scene.remove_geometry("fov_clusters")
            self.scene_widget.scene.remove_geometry("noise_points")
            self.scene_widget.scene.remove_geometry("fov_clusters")
            self.scene_widget.scene.remove_geometry("viewpoints")
            self.scene_widget.scene.remove_geometry("region_view_meshes")

            self.scene_widget.scene.add_geometry(
                "regions", regions_cloud, Materials.point_cloud_material)
            self.scene_widget.scene.add_geometry(
                "noise_points", noise_point_cloud, Materials.point_cloud_material)
            self.scene_widget.scene.add_geometry(
                "fov_clusters", fov_meshes, Materials.fov_cluster_material)
            self.scene_widget.scene.add_geometry(
                "viewpoints", viewpoint_meshes, Materials.viewpoint_material)
            self.scene_widget.scene.add_geometry(
                "region_view_meshes", region_view_meshes, Materials.region_view_material)

            if show_clusters:
                self.show_regions(False)
                self.show_noise_points(False)
                self.show_fov_clusters(True)
                if show_viewpoints:
                    self.show_viewpoints(True)
                    # Set camera view to fit the mesh
                    bb = viewpoint_meshes.get_axis_aligned_bounding_box()
                    self.scene_widget.look_at(
                        bb.get_center(), bb.get_max_bound(), np.array([0, 0, 1]))
                else:
                    self.show_viewpoints(False)
            else:
                self.show_regions(True)
                self.show_noise_points(True)
                self.show_fov_clusters(False)
                self.show_viewpoints(False)

            print(
                f"Loaded regions from {file_path} and updated point cloud colors")

        except Exception as e:
            print(f"Error loading regions from {file_path}: {e}")
            return

    def set_widget_value(self, widget, value):
        """Set the value of a widget based on its type"""
        try:
            if hasattr(widget, 'checked'):
                # Checkbox widget
                widget.checked = bool(value)
            elif hasattr(widget, 'int_value'):
                # Integer NumberEdit widget
                widget.int_value = int(value)
            elif hasattr(widget, 'double_value'):
                # Double NumberEdit widget
                # Round to 3 decimal places for consistency
                value = round(float(value), 3)
                widget.double_value = float(value)
            elif hasattr(widget, 'text_value'):
                # TextEdit widget
                widget.text_value = str(value)
                if value == '':
                    widget.text_value = 'None'
            else:
                print(f"Warning: Unknown widget type for value: {value}")
        except Exception as e:
            print(f"Error setting widget value to {value}: {e}")

    def cast_ray_from_center(self):
        """Cast a ray from the center of the current view"""
        scene = self.scene_widget.scene
        camera = scene.camera

        # Get camera position and forward direction
        view_matrix = camera.get_view_matrix()
        inv_view_matrix = np.linalg.inv(view_matrix)

        camera_position = inv_view_matrix[:3, 3]
        camera_forward = -inv_view_matrix[:3, 2]  # Negative Z is forward
        camera_forward = camera_forward / np.linalg.norm(camera_forward)

        # Prepare ray for casting
        rays = o3d.core.Tensor([
            [camera_position[0], camera_position[1], camera_position[2],
             camera_forward[0], camera_forward[1], camera_forward[2]]
        ], dtype=o3d.core.Dtype.Float32)

        # Cast the ray
        try:
            result = scene.cast_rays(rays)

            if len(result['t_hit']) > 0 and result['t_hit'][0] < 1000.0:
                t = float(result['t_hit'][0])
                intersection_point = camera_position + t * camera_forward

                print(f"Intersection found at: {intersection_point}")
                print(f"Distance: {t:.3f}")

                return {
                    'point': intersection_point,
                    'distance': t,
                    'hit': True
                }
            else:
                print("No intersection found")
                return {'hit': False}

        except Exception as e:
            print(f"Ray casting error: {e}")
            return {'hit': False}

    def cast_ray_from_center(self):
        # Get camera info
        camera = self.scene_widget.scene.camera
        view_matrix = camera.get_view_matrix()
        # Check if view_matrix has changed
        if np.array_equal(view_matrix, self.current_view_matrix):
            return {'hit': False}

        self.current_view_matrix = view_matrix
        inv_view_matrix = np.linalg.inv(view_matrix)

        camera_position = inv_view_matrix[:3, 3]
        camera_forward = -inv_view_matrix[:3, 2]
        camera_forward = camera_forward / np.linalg.norm(camera_forward)

        # Create ray: [origin_x, origin_y, origin_z, direction_x, direction_y, direction_z]
        ray = np.array([[
            camera_position[0], camera_position[1], camera_position[2],
            camera_forward[0], camera_forward[1], camera_forward[2]
        ]], dtype=np.float32)

        # Cast ray
        rays_tensor = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
        result = self.ray_casting_scene.cast_rays(rays_tensor)

        # Check for intersection
        if len(result['t_hit']) > 0:
            t = result['t_hit'][0].item()
            if t < np.inf:
                intersection_point = camera_position + t * camera_forward
                self.last_intersection_point = intersection_point
                return {'point': intersection_point, 'distance': t, 'hit': True}

        return {'hit': False}

    def add_cylinder_pointing_at_camera_simple(self, intersection_result, cylinder_name="ray_cylinder"):
        """Simple version using Open3D's align_vector_to_vector"""

        self.camera_updated = False

        scene = self.scene_widget.scene
        camera = scene.camera

        view_matrix = camera.get_view_matrix()
        inv_view_matrix = np.linalg.inv(view_matrix)
        camera_position = inv_view_matrix[:3, 3]
        camera_forward = -inv_view_matrix[:3, 2]
        camera_forward = camera_forward / np.linalg.norm(camera_forward)

        # Get intersection point and camera position
        if not intersection_result['hit']:
            # If no intersection, use the last intersection point
            if hasattr(self, 'last_intersection_point'):
                # Set intersection to a point at a distance of 350mm away from the camera
                intersection_point = camera_position + 100 * camera_forward
            else:
                return False
        else:
            intersection_point = intersection_result['point']

        # Create cylinder
        height = 5.0
        camera_radius = min(self.camera_fov_width,
                            self.camera_fov_height) / 2.0
        # Create a cylinder with the specified radius and height
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=1000*camera_radius, height=2*height)
        # Crop top and bottom of cylinder
        cylinder = cylinder.crop(o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-20, -20, -height/2),
            max_bound=(20, 20, height/2)
        ))
        # Scale down along z-axis and translate up by height/2
        # cylinder.scale(0.1, center=(0, 0, 0))
        # cylinder.translate((0, 0, height/2))
        reticle = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        cylinder += reticle

        # Use Open3D's built-in method to align vectors
        # Default cylinder points along Z-axis [0, 0, 1]
        default_direction = np.array([0, 0, 1])

        # Calculate rotation matrix using Open3D's utility
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
            (0, 0, 0))  # Identity first

        # Manual rotation calculation (simple version)
        # If vectors are not parallel, calculate rotation
        if not np.allclose(default_direction, camera_forward):
            # Use cross product for rotation axis
            rotation_axis = np.cross(default_direction, camera_forward)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # Calculate angle
                dot_product = np.dot(default_direction, camera_forward)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

                # Create rotation matrix manually
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                ux, uy, uz = rotation_axis

                rotation_matrix = np.array([
                    [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle) -
                     uz*sin_angle, ux*uz*(1-cos_angle) + uy*sin_angle],
                    [uy*ux*(1-cos_angle) + uz*sin_angle, cos_angle + uy *
                     uy*(1-cos_angle), uy*uz*(1-cos_angle) - ux*sin_angle],
                    [uz*ux*(1-cos_angle) - uy*sin_angle, uz*uy*(1-cos_angle) +
                     ux*sin_angle, cos_angle + uz*uz*(1-cos_angle)]
                ])

        # Apply rotation and translation
        cylinder.rotate(rotation_matrix, center=[0, 0, 0])
        cylinder.translate(intersection_point)

        # Color and add to scene
        cylinder.compute_vertex_normals()

        # Remove previous cylinder if exists
        if np.equal(intersection_point, self.last_intersection_point).all():
            scene.remove_geometry('reticle')
            scene.add_geometry('reticle', cylinder, Materials.fov_material)
        return True

    def set_mouse_orbit_center_to_intersection(self, intersection_result):
        """Set the mouse orbit center to intersection point for left-click-drag rotation
        Not working yet, but could be used if we need a lock-on feature 
        """

        if not intersection_result['hit']:
            return False

        intersection_point = intersection_result['point']
        camera = self.scene_widget.scene.camera

        # Get current camera position to maintain view
        view_matrix = camera.get_view_matrix()
        current_position = np.linalg.inv(view_matrix)[:3, 3]

        # Create small bounding box around intersection point
        # setup_camera uses the center of this box as the orbit center
        padding = 0.1  # Small padding
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=intersection_point - padding,
            max_bound=intersection_point + padding
        )

        # This sets the orbit center to the intersection point
        self.scene_widget.setup_camera(60, bounds, intersection_point)

        # Restore similar camera view
        camera.look_at(intersection_point, current_position, [0, 0, 1])

        print(f"Mouse orbit center set to: {intersection_point}")
        return True

    def update_scene(self):

        self.update_all_widgets_from_dict()

        intersection_result = self.cast_ray_from_center()

        self.add_cylinder_pointing_at_camera_simple(intersection_result)

        lockon = False
        if lockon:
            self.set_mouse_orbit_center_to_intersection(intersection_result)

        this_draw_time = time.time()
        if this_draw_time - self.last_draw_time < 1/self.fps:
            time.sleep(1/self.fps - (this_draw_time - self.last_draw_time))
            this_draw_time = time.time()

    def on_main_window_closing(self):
        gui.Application.instance.quit()

        rclpy.shutdown()

        return True  # False would cancel the close

    def on_main_window_tick_event(self):
        self.update_scene()

    def _on_layout(self, layout_context):

        em = self.window.theme.font_size

        r = self.window.content_rect

        self.scene_widget.frame = r

        width = 22 * em
        # Set height to preferred size with a maximum of 80% of the window height
        height = self.main_layout.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        if height > r.height - 5 * em:
            height = r.height - 5 * em

        right_margin = 0.25 * em
        # Place main layout on the right side in the middle of the window
        self.main_layout.frame = gui.Rect(
            r.width - width - right_margin, 0.5 * r.height - height/2 + em, width, height)


def main(args=None):
    rclpy.init(args=args)
    print(args)

    gui.Application.instance.initialize()

    use_tick = -1

    gui_client = GUIClient()

    gui.Application.instance.run()


if __name__ == '__main__':
    print("Open3D version:", o3d.__version__)
    main()
