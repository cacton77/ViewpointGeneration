#!/usr/bin/env python3
import rclpy
import sys
import time
import json
import random
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from pprint import pprint
from matplotlib import colormaps

from viewpoint_generation.threads.ros import ROSThread
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
    MENU_SHOW_MODEL = 13
    MENU_SHOW_POINT_CLOUDS = 14
    MENU_SHOW_REGIONS = 15
    MENU_SHOW_VIEWPOINT = 16
    MENU_SHOW_SETTINGS = 17
    MENU_SHOW_ERRORS = 18
    MENU_SHOW_PATH = 19
    MENU_ABOUT = 21

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

        self.window.add_child(self.scene_widget)

        self.parameter_widgets = {}

        self.init_gui()
        self.init_menu_bar()
        # self.window.add_child(self.parameter_panel)
        self.window.set_on_layout(self._on_layout)

        self.last_draw_time = time.time()

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
            ground_plane_menu = gui.Menu()
            ground_plane_menu.add_item("XY", 100)
            ground_plane_menu.add_item("XZ", 101)
            ground_plane_menu.add_item("YZ", 102)
            view_menu.add_menu("Ground Plane", ground_plane_menu)
            view_menu.add_separator()
            # Object display options
            view_menu.add_item("Show Model", self.MENU_SHOW_MODEL)
            view_menu.set_checked(self.MENU_SHOW_MODEL, True)
            view_menu.add_item("Show Point Clouds",
                               self.MENU_SHOW_POINT_CLOUDS)
            view_menu.set_checked(self.MENU_SHOW_POINT_CLOUDS, True)
            view_menu.add_item("Show Regions", self.MENU_SHOW_REGIONS)
            view_menu.set_checked(self.MENU_SHOW_REGIONS, True)
            view_menu.add_item("Show Path", self.MENU_SHOW_PATH)
            view_menu.set_checked(self.MENU_SHOW_PATH, False)
            view_menu.add_separator()
            # Panel display options
            view_menu.add_item("Viewpoint Generation",
                               self.MENU_SHOW_VIEWPOINT)
            view_menu.set_checked(
                self.MENU_SHOW_VIEWPOINT, True)
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
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_MODEL, self._on_menu_show_model)
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_POINT_CLOUDS, self._on_menu_show_point_clouds)
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_REGIONS, self._on_menu_show_regions)
        # w.set_on_menu_item_activated(
        #     self.MENU_SHOW_PATH, self._on_menu_show_path)
        # w.set_on_menu_item_activated(self.MENU_SHOW_VIEWPOINT,
        #                              self._on_menu_toggle_viewpoint_generation_panel)
        # w.set_on_menu_item_activated(self.MENU_SHOW_SETTINGS,
        #                              self._on_menu_toggle_settings_panel)
        # w.set_on_menu_item_activated(
        #     self.MENU_ABOUT, self._on_menu_about)
        # ----

    def create_tab_panel(self, tab_name, tab_data, em):
        """Create a panel for a tab with nested structure"""
        # Create scrollable area for the tab content
        scroll_area = gui.ScrollableVert(
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
            button_horiz.add_child(button)
        elif section_name == 'curvature':
            button = gui.Button("Compute Curvature")
            button.set_on_clicked(lambda: self.ros_thread.estimate_curvature())
            button_horiz.add_child(button)
        elif section_name == 'region_growth':
            button = gui.Button("Run Region Growth")
            button.set_on_clicked(lambda: self.ros_thread.region_growth())
            button_horiz.add_child(button)

        content.add_child(button_horiz)
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

        self.parameters_dict = self.ros_thread.expand_dict_keys()

        def update_recursive(data_dict, prefix=""):
            """Recursively find and update parameters"""
            parameters_updated = False

            for key, value in data_dict.items():
                if isinstance(value, dict):
                    if 'name' in value and 'type' in value and 'value' in value:
                        # This is a parameter - update its widget
                        param_name = value['name']
                        param_value = value['value']
                        # Check if the parameter has an update flag
                        # If it does, only update if the flag is True
                        param_update_flag = value['update_flag']
                        if param_update_flag:
                            parameters_updated = True

                        if param_name in self.parameter_widgets and param_update_flag:
                            widget = self.parameter_widgets[param_name]
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

                            # Turn update flag off after updating
                            value['update_flag'] = False
                    else:
                        # This is a nested structure - recurse
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        parameters_updated = parameters_updated or update_recursive(
                            value, new_prefix)

            return parameters_updated

        # Start the recursive update
        parameters_updated = update_recursive(self.parameters_dict)

        self.ros_thread.collapse_dict_keys(self.parameters_dict)

        if parameters_updated:
            print("-----------------------------------")

    def import_mesh(self, file_path):
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

                self.scene_widget.scene.remove_geometry(
                    "mesh")  # Remove previous mesh if exists
                self.scene_widget.scene.add_geometry(
                    "mesh", mesh, Materials.mesh_material)

                # Set camera view to fit the mesh
                bb = mesh.get_axis_aligned_bounding_box()
                self.scene_widget.look_at(
                    bb.get_center(), bb.get_max_bound() * 1.5, np.array([0, 0, 1]))
        except Exception as e:
            print(f"Error loading mesh from {file_path}: {e}")

    def import_point_cloud(self, file_path):
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
                self.scene_widget.scene.add_geometry(
                    "point_cloud", point_cloud, Materials.point_cloud_material)

            self.point_cloud = point_cloud  # Store the point cloud for later use

        except Exception as e:
            print(f"Error loading point cloud from {file_path}: {e}")

    def import_curvature(self, file_path):
        """ Load curvature data from file and color point cloud based on curvature data"""
        try:
            curvature = np.load(file_path)

            max_curvature = np.max(curvature)
            min_curvature = np.min(curvature)

            normalized_curvature = (
                curvature - min_curvature) / (max_curvature - min_curvature)

            cmap = colormaps[Materials.curvature_colormap]

            for i in range(len(normalized_curvature)):
                val = 1 - normalized_curvature[i]
                color = np.array(list(cmap(val)))[0, 0:3]  # Get RGB values
                np.asarray(self.point_cloud.colors)[i] = color

            # Remove previous point cloud if exists
            self.scene_widget.scene.remove_geometry("point_cloud")
            self.scene_widget.scene.add_geometry(
                "point_cloud", self.point_cloud, Materials.point_cloud_material)

        except Exception as e:
            print(f"Error loading curvature data from {file_path}: {e}")

    def import_regions(self, file_path):
        """ Load regions from file and paint point cloud based on regions """
        try:
            self.point_cloud.paint_uniform_color((1, 1, 1))

            regions_dict = json.load(open(file_path, 'r'))

            colors = np.zeros((len(self.point_cloud.points), 3))

            np.random.seed(42)  # For reproducibility
            for region, dict in regions_dict['regions'].items():
                cluster = dict['points']
                color = np.random.rand(3)

                for point_idx in cluster:
                    colors[point_idx] = color

            for point_idx in regions_dict['noise_points']:
                colors[point_idx] = [0.5, 0.5, 0.5]

            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Remove previous point cloud if exists
            self.scene_widget.scene.remove_geometry("point_cloud")
            self.scene_widget.scene.add_geometry(
                "point_cloud", self.point_cloud, Materials.point_cloud_material)
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
            else:
                print(f"Warning: Unknown widget type for value: {value}")
        except Exception as e:
            print(f"Error setting widget value to {value}: {e}")

    def update_scene(self):
        # # Remove axes from scene if they exit
        # if self.scene_widget.scene.has_geometry("axes"):
        #     self.scene_widget.scene.remove_geometry("axes")
        # # Add axes to scene
        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.5, origin=[0, 0, 0])
        # self.scene_widget.scene.add_geometry("axes", axes, o3d.visualization.rendering.MaterialRecord())
        # Update the scene

        self.update_all_widgets_from_dict()

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
            r.width - width - right_margin, 0.5 * (r.height - 2.5) - height/2 + 2.5 * em, width, height)


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
