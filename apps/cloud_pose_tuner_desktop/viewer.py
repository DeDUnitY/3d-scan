from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .filters import FILTER_COLOR_PRESETS
from .state import DesktopTunerState


class DesktopCloudPoseTunerApp:
    def __init__(self, state: DesktopTunerState) -> None:
        self.state = state
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Cloud Pose Tuner", 1680, 960)
        self.window.set_on_layout(self._on_layout)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.07, 0.07, 0.07, 1.0])
        self.scene_widget.scene.show_axes(True)

        em = self.window.theme.font_size
        self.panel_width = int(30 * em)
        self.margin = int(0.5 * em)
        margins = gui.Margins(self.margin, self.margin, self.margin * 2, self.margin)
        self.panel = gui.ScrollableVert(self.margin, margins)

        self.window.add_child(self.panel)
        self.window.add_child(self.scene_widget)

        self.stats_label = gui.Label("")
        self.status_label = gui.Label("")
        self.frame_checks: dict[int, gui.Checkbox] = {}
        self.color_include_checks: dict[str, gui.Checkbox] = {}
        self.color_exclude_checks: dict[str, gui.Checkbox] = {}
        self._section_expanded: dict[str, bool] = {}
        self._section_buttons: dict[str, gui.Button] = {}
        self._section_contents: dict[str, gui.Vert] = {}
        self.camera_pivot_mode = "visible_bbox_center"
        self.position_param_step = 0.1
        self._param_labels = {
            "camera_start_angle_deg": "Start camera angle (deg)",
            "table_rotation_step": "Turntable step per frame (deg)",
            "extra_frame_rot_z_deg": "Extra frame Z rotation (deg)",
            "platform_rotation_sign": "Rotation direction (-1 or 1)",
            "use_turntable": "Use turntable model",
            "table_center_x": "Turntable center X",
            "table_center_y": "Turntable center Y",
            "table_center_z": "Turntable center Z",
            "orbit_radius": "Camera orbit radius",
            "camera_height": "Camera height",
            "camera_tilt_deg": "Camera tilt (deg)",
            "frame_tilt_x_deg": "Frame tilt X (deg, camera space)",
            "frame_tilt_y_deg": "Frame tilt Y (deg, camera space)",
            "frame_roll_z_deg": "Frame roll Z (deg, camera view axis)",
            "camera_offset_y": "Camera Y offset",
            "camera_min_distance": "Min camera distance",
            "camera_max_distance": "Max camera distance",
            "invert_x": "Mirror cloud by X axis",
            "use_crop": "Enable radial crop",
            "crop_radius": "Crop radius",
            "z_min": "Min Z (height floor)",
            "z_max": "Max Z (height ceiling)",
            "use_color_filter": "Enable color filter",
            "color_tolerance": "Color tolerance",
            "dark_threshold": "Dark pixel threshold",
            "use_isolated_filter": "Remove isolated points",
            "isolated_radius": "Isolated-point radius",
            "isolated_min_neighbors": "Min neighbors",
        }

        self._build_controls()
        self._refresh_scene(reset_camera=True, status="Project loaded")

    def _on_layout(self, layout_context) -> None:
        content = self.window.content_rect
        self.panel.frame = gui.Rect(content.x, content.y, self.panel_width, content.height)
        self.scene_widget.frame = gui.Rect(
            content.x + self.panel_width,
            content.y,
            max(content.width - self.panel_width, 0),
            content.height,
        )

    def _section_label(self, text: str) -> gui.Label:
        label = gui.Label(text)
        label.text_color = gui.Color(0.9, 0.9, 0.9)
        return label

    def _add_checkbox(self, label: str, checked: bool, callback) -> gui.Checkbox:
        return self._add_checkbox_to(self.panel, label, checked, callback)

    def _add_checkbox_to(self, parent, label: str, checked: bool, callback) -> gui.Checkbox:
        widget = gui.Checkbox(label)
        widget.checked = checked
        widget.set_on_checked(callback)
        parent.add_child(widget)
        return widget

    def _add_number(
        self,
        label: str,
        value: float,
        callback,
        minimum: float = -1e9,
        maximum: float = 1e9,
        is_int: bool = False,
        step: float | None = None,
    ):
        return self._add_number_to(
            self.panel,
            label,
            value,
            callback,
            minimum=minimum,
            maximum=maximum,
            is_int=is_int,
            step=step,
        )

    def _add_number_to(
        self,
        parent,
        label: str,
        value: float,
        callback,
        minimum: float = -1e9,
        maximum: float = 1e9,
        is_int: bool = False,
        step: float | None = None,
    ):
        parent.add_child(gui.Label(label))
        row = gui.Horiz(self.margin)
        widget = gui.NumberEdit(gui.NumberEdit.INT if is_int else gui.NumberEdit.DOUBLE)
        widget.set_limits(minimum, maximum)
        def current_step() -> float:
            if callable(step):
                return float(step())
            if step is not None:
                return float(step)
            return float(1 if is_int else 0.1)

        def clamp(v: float) -> float:
            return max(min(v, maximum), minimum)

        if is_int:
            widget.int_value = int(value)
            widget.set_on_value_changed(lambda v: callback(int(v)))
        else:
            dec_button = gui.Button("-")
            inc_button = gui.Button("+")
            widget.double_value = float(value)
            widget.set_on_value_changed(lambda v: callback(float(v)))

            def shift(delta: float = 0.0) -> None:
                new_value = clamp(widget.double_value + delta)
                widget.double_value = new_value
                callback(new_value)

            dec_button.set_on_clicked(lambda: shift(-current_step()))
            inc_button.set_on_clicked(lambda: shift(current_step()))
            row.add_child(dec_button)
        row.add_child(widget)
        if not is_int:
            row.add_child(inc_button)
        parent.add_child(row)
        return widget

    def _add_combobox_row(self, parent, label: str, combobox: gui.Combobox) -> None:
        row = gui.Horiz(self.margin)
        row.add_child(gui.Label(label))
        row.add_child(combobox)
        parent.add_child(row)

    def _add_collapsible_section(self, section_key: str, title: str, expanded: bool = True) -> gui.Vert:
        self._section_expanded[section_key] = bool(expanded)
        header = gui.Button("")
        header.horizontal_padding_em = 0.4
        header.vertical_padding_em = 0.2
        header.set_on_clicked(lambda key=section_key: self._toggle_section(key))
        self._section_buttons[section_key] = header
        self.panel.add_child(header)

        content = gui.Vert(self.margin, gui.Margins(self.margin, 0, self.margin * 2, self.margin))
        content.visible = bool(expanded)
        self._section_contents[section_key] = content
        self.panel.add_child(content)
        self._update_section_header(section_key, title)
        return content

    def _update_section_header(self, section_key: str, title: str) -> None:
        marker = "[-]" if self._section_expanded.get(section_key, True) else "[+]"
        button = self._section_buttons.get(section_key)
        if button is not None:
            button.text = f"{marker} {title}"

    def _toggle_section(self, section_key: str) -> None:
        self._section_expanded[section_key] = not self._section_expanded.get(section_key, True)
        content = self._section_contents.get(section_key)
        if content is not None:
            content.visible = self._section_expanded[section_key]
        title_map = {
            "overview": "Overview",
            "display": "Display",
            "pose": "Pose alignment",
            "filters": "Cloud filters",
            "frames": "Frames",
            "actions": "Actions",
        }
        self._update_section_header(section_key, title_map.get(section_key, section_key))
        self.window.set_needs_layout()
        self.window.post_redraw()

    def _build_controls(self) -> None:
        overview = self._add_collapsible_section("overview", "Overview", expanded=True)
        overview.add_child(self.stats_label)
        overview.add_child(self.status_label)

        self.panel.add_fixed(self.margin)
        display = self._add_collapsible_section("display", "Display", expanded=True)
        self._add_checkbox_to(display, "Show merged point cloud", self.state.show_merged, self._on_toggle_show_merged)
        self._add_checkbox_to(display, "Show reference mesh (CAD)", self.state.show_reference, self._on_toggle_show_reference)
        self._add_checkbox_to(display, "White background", self.state.white_background, self._on_toggle_white_background)

        self.color_mode = gui.Combobox()
        self.color_mode.add_item("Original RGB")
        self.color_mode.add_item("Per-frame colors")
        self.color_mode.selected_index = 0 if self.state.color_mode == "rgb" else 1
        self.color_mode.set_on_selection_changed(self._on_color_mode)
        self._add_combobox_row(display, "Point colors", self.color_mode)

        self._add_number_to(display, "Preview point limit", self.state.preview_max_points, self._on_preview_max_points, minimum=1000, maximum=5_000_000, is_int=True)
        self._add_number_to(display, "Point size (px)", self.state.point_size, self._on_point_size, minimum=1.0, maximum=10.0)
        self.position_step_combo = gui.Combobox()
        for step in ("0.01", "0.05", "0.1", "0.5", "1.0", "5.0"):
            self.position_step_combo.add_item(step)
        self.position_step_combo.selected_index = 2
        self.position_step_combo.set_on_selection_changed(self._on_position_step_changed)
        self._add_combobox_row(display, "Position params step", self.position_step_combo)
        self.camera_pivot_mode_combo = gui.Combobox()
        self.camera_pivot_mode_combo.add_item("Visible cloud bounds center")
        self.camera_pivot_mode_combo.add_item("Center of mass (visible points)")
        self.camera_pivot_mode_combo.add_item("World origin (0,0,0)")
        self.camera_pivot_mode_combo.selected_index = 0
        self.camera_pivot_mode_combo.set_on_selection_changed(self._on_camera_pivot_mode_changed)
        self._add_combobox_row(display, "Camera orbit center", self.camera_pivot_mode_combo)

        self.panel.add_fixed(self.margin)
        pose = self._add_collapsible_section("pose", "Pose alignment", expanded=False)
        self._add_number_to(pose, "Start camera angle (deg)", self.state.params.camera_start_angle_deg, self._bind_param("camera_start_angle_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Turntable step per frame (deg)", self.state.params.table_rotation_step, self._bind_param("table_rotation_step"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Extra frame Z rotation (deg)", self.state.params.extra_frame_rot_z_deg, self._bind_param("extra_frame_rot_z_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Rotation direction (-1 or 1)", self.state.params.platform_rotation_sign, self._bind_param("platform_rotation_sign"), minimum=-1, maximum=1, is_int=True)
        self._add_checkbox_to(pose, "Use turntable model", self.state.params.use_turntable, self._bind_bool("use_turntable"))
        self._add_number_to(pose, "Turntable center X", self.state.params.table_center_x, self._bind_param("table_center_x"), step=lambda: self.position_param_step)
        self._add_number_to(pose, "Turntable center Y", self.state.params.table_center_y, self._bind_param("table_center_y"), step=lambda: self.position_param_step)
        self._add_number_to(pose, "Turntable center Z", self.state.params.table_center_z, self._bind_param("table_center_z"), step=lambda: self.position_param_step)
        self._add_number_to(pose, "Camera orbit radius", self.state.params.orbit_radius, self._bind_param("orbit_radius"), minimum=-2000.0, maximum=2000.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Camera height", self.state.params.camera_height, self._bind_param("camera_height"), minimum=-2000.0, maximum=2000.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Camera tilt (deg)", self.state.params.camera_tilt_deg, self._bind_param("camera_tilt_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Frame tilt X (deg)", self.state.params.frame_tilt_x_deg, self._bind_param("frame_tilt_x_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Frame tilt Y (deg)", self.state.params.frame_tilt_y_deg, self._bind_param("frame_tilt_y_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Frame roll Z (deg)", self.state.params.frame_roll_z_deg, self._bind_param("frame_roll_z_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Camera Y offset", self.state.params.camera_offset_y, self._bind_param("camera_offset_y"), minimum=-2000.0, maximum=2000.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Min camera distance", self.state.params.camera_min_distance, self._bind_param("camera_min_distance"), minimum=0.0, maximum=5000.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Max camera distance", self.state.params.camera_max_distance, self._bind_param("camera_max_distance"), minimum=0.0, maximum=5000.0, step=lambda: self.position_param_step)

        self.panel.add_fixed(self.margin)
        filters = self._add_collapsible_section("filters", "Cloud filters", expanded=False)
        self._add_checkbox_to(filters, "Mirror cloud by X axis", self.state.params.invert_x, self._bind_bool("invert_x"))
        self._add_checkbox_to(filters, "Enable radial crop", self.state.params.use_crop, self._bind_bool("use_crop"))
        self._add_number_to(filters, "Crop radius", self.state.params.crop_radius, self._bind_param("crop_radius"), minimum=0.0, maximum=5000.0, step=lambda: self.position_param_step)
        self._add_number_to(filters, "Min Z (height floor)", self.state.params.z_min, self._bind_param("z_min"), minimum=-5000.0, maximum=5000.0, step=lambda: self.position_param_step)
        self._add_number_to(filters, "Max Z (height ceiling)", self.state.params.z_max, self._bind_param("z_max"), minimum=-5000.0, maximum=5000.0, step=lambda: self.position_param_step)
        self._add_checkbox_to(filters, "Enable color filter", self.state.params.use_color_filter, self._bind_bool("use_color_filter"))
        filters.add_child(gui.Label("Keep only selected colors"))
        for color_name in FILTER_COLOR_PRESETS.keys():
            checkbox = gui.Checkbox(color_name.title())
            checkbox.checked = bool(self.state.params.color_include_filters.get(color_name, False))
            checkbox.set_on_checked(self._on_color_include(color_name))
            self.color_include_checks[color_name] = checkbox
            filters.add_child(checkbox)
        filters.add_child(gui.Label("Exclude selected colors"))
        for color_name in FILTER_COLOR_PRESETS.keys():
            checkbox = gui.Checkbox(color_name.title())
            checkbox.checked = bool(self.state.params.color_exclude_filters.get(color_name, False))
            checkbox.set_on_checked(self._on_color_exclude(color_name))
            self.color_exclude_checks[color_name] = checkbox
            filters.add_child(checkbox)
        self._add_number_to(filters, "Color tolerance", self.state.params.color_tolerance, self._bind_param("color_tolerance"), minimum=0, maximum=255, is_int=True)
        self._add_number_to(filters, "Dark pixel threshold", self.state.params.dark_threshold, self._bind_param("dark_threshold"), minimum=0, maximum=255, is_int=True)
        self._add_checkbox_to(filters, "Remove isolated points", self.state.params.use_isolated_filter, self._bind_bool("use_isolated_filter"))
        self._add_number_to(filters, "Isolated-point radius", self.state.params.isolated_radius, self._bind_param("isolated_radius"), minimum=0.0, maximum=1000.0, step=lambda: self.position_param_step)
        self._add_number_to(filters, "Min neighbors", self.state.params.isolated_min_neighbors, self._bind_param("isolated_min_neighbors"), minimum=1, maximum=256, is_int=True)

        self.panel.add_fixed(self.margin)
        frames = self._add_collapsible_section("frames", "Frames", expanded=False)
        for frame_id in self.state.frame_ids():
            checkbox = gui.Checkbox(f"Enable frame {frame_id}")
            checkbox.checked = self.state.params.frame_enabled.get(frame_id, True)
            checkbox.set_on_checked(self._on_frame_enabled(frame_id))
            self.frame_checks[frame_id] = checkbox
            frames.add_child(checkbox)

        self.panel.add_fixed(self.margin)
        actions = self._add_collapsible_section("actions", "Actions", expanded=True)
        save_button = gui.Button("Save tuning to JSON")
        save_button.set_on_clicked(self._save_pose_json)
        actions.add_child(save_button)

        export_button = gui.Button("Export filtered cloud (PLY)")
        export_button.set_on_clicked(self._export_cloud)
        actions.add_child(export_button)

        export_npz_button = gui.Button("Export filtered cloud (NPZ)")
        export_npz_button.set_on_clicked(self._export_cloud_npz)
        actions.add_child(export_npz_button)

        reset_button = gui.Button("Reset camera view")
        reset_button.set_on_clicked(lambda: self._refresh_scene(reset_camera=True, status="Camera view reset"))
        actions.add_child(reset_button)

        refresh_button = gui.Button("Rebuild preview now")
        refresh_button.set_on_clicked(lambda: self._refresh_scene(reset_camera=False, status="Preview rebuilt"))
        actions.add_child(refresh_button)

        apply_pivot_button = gui.Button("Apply selected camera center")
        apply_pivot_button.set_on_clicked(lambda: self._refresh_scene(reset_camera=True, status="Camera center updated"))
        actions.add_child(apply_pivot_button)

    def _bind_param(self, attr_name: str):
        def callback(value):
            setattr(self.state.params, attr_name, value)
            label = self._param_labels.get(attr_name, attr_name)
            self._refresh_scene(reset_camera=False, status=f"Updated: {label}")

        return callback

    def _bind_bool(self, attr_name: str):
        def callback(value):
            setattr(self.state.params, attr_name, bool(value))
            label = self._param_labels.get(attr_name, attr_name)
            self._refresh_scene(reset_camera=False, status=f"Updated: {label}")

        return callback

    def _on_color_mode(self, text, index) -> None:
        self.state.color_mode = "rgb" if index == 0 else "frame"
        self._refresh_scene(reset_camera=False, status="Updated: point colors")

    def _on_color_include(self, color_name: str):
        def callback(value: bool) -> None:
            self.state.params.color_include_filters[color_name] = bool(value)
            self._refresh_scene(reset_camera=False, status=f"Updated include color: {color_name}")
        return callback

    def _on_color_exclude(self, color_name: str):
        def callback(value: bool) -> None:
            self.state.params.color_exclude_filters[color_name] = bool(value)
            self._refresh_scene(reset_camera=False, status=f"Updated exclude color: {color_name}")
        return callback

    def _on_preview_max_points(self, value: int) -> None:
        self.state.preview_max_points = int(value)
        self._refresh_scene(reset_camera=False, status=f"Preview limit: {value}")

    def _on_point_size(self, value: float) -> None:
        self.state.point_size = float(value)
        self._refresh_scene(reset_camera=False, status="Updated: point size")

    def _on_camera_pivot_mode_changed(self, text, index) -> None:
        mode_by_index = {
            0: "visible_bbox_center",
            1: "visible_centroid",
            2: "world_origin",
        }
        self.camera_pivot_mode = mode_by_index.get(index, "visible_bbox_center")
        self._refresh_scene(reset_camera=True, status="Updated: camera orbit center")

    def _on_position_step_changed(self, text, index) -> None:
        try:
            self.position_param_step = max(float(text), 1e-6)
        except Exception:
            self.position_param_step = 0.1
        self.status_label.text = f"Position step: {self.position_param_step:g}"

    def _on_toggle_show_merged(self, value: bool) -> None:
        self.state.show_merged = bool(value)
        self._apply_visibility()

    def _on_toggle_show_reference(self, value: bool) -> None:
        self.state.show_reference = bool(value)
        self._apply_visibility()

    def _on_toggle_white_background(self, value: bool) -> None:
        self.state.white_background = bool(value)
        if self.state.white_background:
            self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        else:
            self.scene_widget.scene.set_background([0.07, 0.07, 0.07, 1.0])
        self.window.post_redraw()

    def _on_frame_enabled(self, frame_id: int):
        def callback(checked: bool) -> None:
            self.state.params.frame_enabled[frame_id] = bool(checked)
            self._refresh_scene(reset_camera=False, status=f"Frame {frame_id}: {'enabled' if checked else 'disabled'}")

        return callback

    def _material_for_points(self, point_size: float) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = float(point_size)
        return material

    def _material_for_reference(self) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [0.72, 0.72, 0.72, 0.35]
        return material

    def _replace_geometry(self, name: str, geometry, material) -> None:
        scene = self.scene_widget.scene
        if scene.has_geometry(name):
            scene.remove_geometry(name)
        if geometry is None:
            return
        if isinstance(geometry, o3d.geometry.PointCloud) and geometry.is_empty():
            return
        if isinstance(geometry, o3d.geometry.TriangleMesh) and geometry.is_empty():
            return
        scene.add_geometry(name, geometry, material)

    def _bounds_from_geometries(self, geometries) -> o3d.geometry.AxisAlignedBoundingBox:
        boxes = []
        for geometry in geometries:
            if geometry is None:
                continue
            if isinstance(geometry, o3d.geometry.PointCloud) and geometry.is_empty():
                continue
            if isinstance(geometry, o3d.geometry.TriangleMesh) and geometry.is_empty():
                continue
            boxes.append(geometry.get_axis_aligned_bounding_box())
        if not boxes:
            return o3d.geometry.AxisAlignedBoundingBox(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        bbox = boxes[0]
        for other in boxes[1:]:
            bbox += other
        if np.any((bbox.get_max_bound() - bbox.get_min_bound()) < 1e-6):
            bbox = o3d.geometry.AxisAlignedBoundingBox(bbox.get_min_bound() - 1.0, bbox.get_max_bound() + 1.0)
        return bbox

    def _visible_geometries(self, geometries):
        visible = []
        if self.state.show_merged:
            visible.append(geometries.merged_cloud)
        if self.state.show_reference:
            visible.append(geometries.reference_mesh)
        return visible

    def _centroid_or_none(self, geometry) -> np.ndarray | None:
        if geometry is None:
            return None
        if isinstance(geometry, o3d.geometry.PointCloud):
            if geometry.is_empty():
                return None
            points = np.asarray(geometry.points)
            if points.size == 0:
                return None
            return points.mean(axis=0)
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            if geometry.is_empty():
                return None
            vertices = np.asarray(geometry.vertices)
            if vertices.size == 0:
                return None
            return vertices.mean(axis=0)
        return None

    def _camera_center_from_mode(self, geometries, fallback_bbox: o3d.geometry.AxisAlignedBoundingBox) -> np.ndarray:
        if self.camera_pivot_mode == "world_origin":
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if self.camera_pivot_mode == "visible_centroid":
            centers = []
            for geometry in self._visible_geometries(geometries):
                center = self._centroid_or_none(geometry)
                if center is not None:
                    centers.append(center)
            if centers:
                return np.mean(np.vstack(centers), axis=0)
            return fallback_bbox.get_center()
        return fallback_bbox.get_center()

    def _apply_visibility(self) -> None:
        scene = self.scene_widget.scene
        if scene.has_geometry("merged_cloud"):
            scene.show_geometry("merged_cloud", self.state.show_merged)
        if scene.has_geometry("reference_mesh"):
            scene.show_geometry("reference_mesh", self.state.show_reference)
        self.window.post_redraw()

    def _refresh_scene(self, reset_camera: bool, status: str) -> None:
        preview = self.state.rebuild_preview()
        geometries = self.state.build_scene_geometries()
        self._replace_geometry("merged_cloud", geometries.merged_cloud, self._material_for_points(self.state.point_size))
        self._replace_geometry("reference_mesh", geometries.reference_mesh, self._material_for_reference())
        self._apply_visibility()

        if reset_camera:
            visible_geometries = self._visible_geometries(geometries)
            bbox = self._bounds_from_geometries(visible_geometries)
            camera_center = self._camera_center_from_mode(geometries, bbox)
            self.scene_widget.setup_camera(60.0, bbox, camera_center)

        self.stats_label.text = self.state.current_stats_text()
        self.status_label.text = status
        self.scene_widget.force_redraw()

    def _save_pose_json(self) -> None:
        path = self.state.save_pose_json()
        self._refresh_scene(reset_camera=False, status=f"Saved tuning: {path.name}")

    def _export_cloud(self) -> None:
        path = self.state.export_current_cloud()
        self._refresh_scene(reset_camera=False, status=f"Export complete: {path.name}")

    def _export_cloud_npz(self) -> None:
        path = self.state.export_current_cloud_npz()
        self._refresh_scene(reset_camera=False, status=f"NPZ export complete: {path.name}")

    def run(self) -> None:
        self.app.run()

