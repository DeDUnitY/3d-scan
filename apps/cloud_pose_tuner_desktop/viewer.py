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
        self.scene_widget.scene.show_axes(self.state.show_axes)

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
        self._suppress_frame_callback = False
        self.param_number_edits: dict[str, gui.NumberEdit] = {}
        self.frame_interval = 2
        self.auto_center_search_radius = 5.0
        self.auto_center_max_points = 6000
        self.auto_center_pair_start = 0
        self.auto_center_pair_gap = 1
        self.camera_pivot_mode = "visible_bbox_center"
        self.position_param_step = 0.1
        self.show_crop_radius_guide = True
        self._param_labels = {
            "camera_start_angle_deg": "Start camera angle (deg)",
            "table_rotation_step": "Turntable step per frame (deg)",
            "extra_frame_rot_z_deg": "Extra turntable-axis rotation per frame (deg)",
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
        self._add_checkbox_to(display, "Show axes", self.state.show_axes, self._on_toggle_show_axes)
        self._add_checkbox_to(display, "Show crop radius guide", self.show_crop_radius_guide, self._on_toggle_crop_radius_guide)
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
        self._add_number_to(pose, "Extra turntable-axis rotation per frame (deg)", self.state.params.extra_frame_rot_z_deg, self._bind_param("extra_frame_rot_z_deg"), minimum=-180.0, maximum=180.0, step=lambda: self.position_param_step)
        self._add_number_to(pose, "Rotation direction (-1 or 1)", self.state.params.platform_rotation_sign, self._bind_param("platform_rotation_sign"), minimum=-1, maximum=1, is_int=True)
        self._add_checkbox_to(pose, "Use turntable model", self.state.params.use_turntable, self._bind_bool("use_turntable"))
        self.param_number_edits["table_center_x"] = self._add_number_to(pose, "Turntable center X", self.state.params.table_center_x, self._bind_param("table_center_x"), step=lambda: self.position_param_step)
        self.param_number_edits["table_center_y"] = self._add_number_to(pose, "Turntable center Y", self.state.params.table_center_y, self._bind_param("table_center_y"), step=lambda: self.position_param_step)
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
        self._add_number_to(
            frames,
            "Frame interval N",
            self.frame_interval,
            self._on_frame_interval_changed,
            minimum=1,
            maximum=1000,
            is_int=True,
        )
        frame_buttons_row = gui.Horiz(self.margin)
        enable_all_button = gui.Button("Enable all")
        disable_all_button = gui.Button("Disable all")
        keep_every_button = gui.Button("Keep every N")
        disable_every_button = gui.Button("Disable every N")
        enable_all_button.set_on_clicked(self._enable_all_frames)
        disable_all_button.set_on_clicked(self._disable_all_frames)
        keep_every_button.set_on_clicked(self._keep_every_n_frames)
        disable_every_button.set_on_clicked(self._disable_every_n_frames)
        frame_buttons_row.add_child(enable_all_button)
        frame_buttons_row.add_child(disable_all_button)
        frames.add_child(frame_buttons_row)
        frame_pattern_row = gui.Horiz(self.margin)
        frame_pattern_row.add_child(keep_every_button)
        frame_pattern_row.add_child(disable_every_button)
        frames.add_child(frame_pattern_row)
        self._add_number_to(
            frames,
            "Auto center search radius",
            self.auto_center_search_radius,
            self._on_auto_center_search_radius_changed,
            minimum=0.01,
            maximum=1000.0,
            step=lambda: self.position_param_step,
        )
        self._add_number_to(
            frames,
            "Auto center max points/frame",
            self.auto_center_max_points,
            self._on_auto_center_max_points_changed,
            minimum=100,
            maximum=100000,
            is_int=True,
        )
        self._add_number_to(
            frames,
            "Auto center pair start index",
            self.auto_center_pair_start,
            self._on_auto_center_pair_start_changed,
            minimum=0,
            maximum=100000,
            is_int=True,
        )
        self._add_number_to(
            frames,
            "Auto center pair gap",
            self.auto_center_pair_gap,
            self._on_auto_center_pair_gap_changed,
            minimum=1,
            maximum=100000,
            is_int=True,
        )
        auto_center_button = gui.Button("Auto fit center XY from first enabled pair")
        auto_center_button.set_on_clicked(self._auto_tune_center_xy)
        frames.add_child(auto_center_button)
        frames.add_child(gui.Label("Frame label color matches Per-frame colors mode."))
        for frame_id in self.state.frame_ids():
            row = gui.Horiz(self.margin)
            checkbox = gui.Checkbox("Enable")
            frame_color = self.state.frame_color(frame_id)
            frame_label = gui.Label(f"frame {frame_id}")
            frame_label.text_color = gui.Color(
                float(frame_color[0]),
                float(frame_color[1]),
                float(frame_color[2]),
            )
            checkbox.checked = self.state.params.frame_enabled.get(frame_id, True)
            checkbox.set_on_checked(self._on_frame_enabled(frame_id))
            self.frame_checks[frame_id] = checkbox
            row.add_child(checkbox)
            row.add_child(frame_label)
            frames.add_child(row)

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

    def _on_frame_interval_changed(self, value: int) -> None:
        self.frame_interval = max(int(value), 1)
        self.status_label.text = f"Frame interval N: {self.frame_interval}"

    def _on_auto_center_search_radius_changed(self, value: float) -> None:
        self.auto_center_search_radius = max(float(value), 0.01)
        self.status_label.text = f"Auto center search radius: {self.auto_center_search_radius:g}"

    def _on_auto_center_max_points_changed(self, value: int) -> None:
        self.auto_center_max_points = max(int(value), 100)
        self.status_label.text = f"Auto center max points/frame: {self.auto_center_max_points}"

    def _on_auto_center_pair_start_changed(self, value: int) -> None:
        self.auto_center_pair_start = max(int(value), 0)
        self.status_label.text = f"Auto center pair start index: {self.auto_center_pair_start}"

    def _on_auto_center_pair_gap_changed(self, value: int) -> None:
        self.auto_center_pair_gap = max(int(value), 1)
        self.status_label.text = f"Auto center pair gap: {self.auto_center_pair_gap}"

    def _on_toggle_show_merged(self, value: bool) -> None:
        self.state.show_merged = bool(value)
        self._apply_visibility()

    def _on_toggle_show_reference(self, value: bool) -> None:
        self.state.show_reference = bool(value)
        self._apply_visibility()

    def _on_toggle_show_axes(self, value: bool) -> None:
        self.state.show_axes = bool(value)
        self.scene_widget.scene.show_axes(self.state.show_axes)
        self._replace_table_center_marker()
        self.window.post_redraw()

    def _replace_table_center_marker(self) -> None:
        marker = self._build_table_center_marker() if self.state.show_axes else None
        self._replace_geometry("table_center_marker", marker, self._material_for_reference())

    def _on_toggle_crop_radius_guide(self, value: bool) -> None:
        self.show_crop_radius_guide = bool(value)
        self._replace_geometry(
            "crop_radius_guide",
            self._build_crop_radius_guide(),
            self._material_for_lines([1.0, 0.72, 0.1, 1.0]),
        )
        self.window.post_redraw()

    def _on_toggle_white_background(self, value: bool) -> None:
        self.state.white_background = bool(value)
        if self.state.white_background:
            self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        else:
            self.scene_widget.scene.set_background([0.07, 0.07, 0.07, 1.0])
        self.window.post_redraw()

    def _on_frame_enabled(self, frame_id: int):
        def callback(checked: bool) -> None:
            if self._suppress_frame_callback:
                return
            self.state.params.frame_enabled[frame_id] = bool(checked)
            self._refresh_scene(reset_camera=False, status=f"Frame {frame_id}: {'enabled' if checked else 'disabled'}")

        return callback

    def _set_frames_enabled(self, enabled_by_frame: dict[int, bool], status: str) -> None:
        self._suppress_frame_callback = True
        try:
            for frame_id in self.state.frame_ids():
                enabled = bool(enabled_by_frame.get(frame_id, False))
                self.state.params.frame_enabled[frame_id] = enabled
                checkbox = self.frame_checks.get(frame_id)
                if checkbox is not None:
                    checkbox.checked = enabled
        finally:
            self._suppress_frame_callback = False
        self._refresh_scene(reset_camera=False, status=status)

    def _enable_all_frames(self) -> None:
        self._set_frames_enabled(
            {frame_id: True for frame_id in self.state.frame_ids()},
            status="Enabled all frames",
        )

    def _disable_all_frames(self) -> None:
        self._set_frames_enabled(
            {frame_id: False for frame_id in self.state.frame_ids()},
            status="Disabled all frames",
        )

    def _keep_every_n_frames(self) -> None:
        interval = max(int(self.frame_interval), 1)
        enabled_by_frame = {
            frame_id: index % interval == 0
            for index, frame_id in enumerate(self.state.frame_ids())
        }
        self._set_frames_enabled(enabled_by_frame, status=f"Kept every {interval} frame")

    def _disable_every_n_frames(self) -> None:
        interval = max(int(self.frame_interval), 1)
        enabled_by_frame = {
            frame_id: (index + 1) % interval != 0
            for index, frame_id in enumerate(self.state.frame_ids())
        }
        self._set_frames_enabled(enabled_by_frame, status=f"Disabled every {interval} frame")

    def _sync_param_number(self, attr_name: str) -> None:
        widget = self.param_number_edits.get(attr_name)
        if widget is None:
            return
        widget.double_value = float(getattr(self.state.params, attr_name))

    def _auto_tune_center_xy(self) -> None:
        self.status_label.text = "Auto fitting center XY..."
        self.window.post_redraw()
        try:
            result = self.state.auto_tune_center_xy(
                search_radius=self.auto_center_search_radius,
                max_points_per_frame=self.auto_center_max_points,
                pair_start_index=self.auto_center_pair_start,
                pair_gap=self.auto_center_pair_gap,
            )
        except Exception as exc:
            self.status_label.text = f"Auto center failed: {exc}"
            self.window.post_redraw()
            return

        self._sync_param_number("table_center_x")
        self._sync_param_number("table_center_y")
        self._refresh_scene(
            reset_camera=False,
            status=(
                f"Auto center {result.frame_a}->{result.frame_b}: "
                f"X={result.center_x:.3f}, Y={result.center_y:.3f}, "
                f"score {result.score_before:.3f}->{result.score_after:.3f}"
            ),
        )

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

    def _material_for_lines(self, color: list[float]) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.base_color = color
        return material

    def _visible_table_center(self) -> np.ndarray:
        center = self.state.params.table_center.astype(np.float64)
        if self.state.params.invert_x:
            center = center.copy()
            center[0] *= -1.0
        return center

    def _build_crop_radius_guide(self) -> o3d.geometry.LineSet | None:
        if not self.show_crop_radius_guide or not self.state.params.use_crop or self.state.params.crop_radius <= 0:
            return None
        center = self._visible_table_center()
        radius = float(self.state.params.crop_radius)
        segment_count = 128
        angles = np.linspace(0.0, 2.0 * np.pi, segment_count, endpoint=False)
        points = np.column_stack(
            (
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                np.full(segment_count, center[2], dtype=np.float64),
            )
        )
        lines = [[index, (index + 1) % segment_count] for index in range(segment_count)]
        guide = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        guide.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.72, 0.1]], dtype=np.float64), (len(lines), 1))
        )
        return guide

    def _build_table_center_marker(self) -> o3d.geometry.TriangleMesh:
        marker = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(float(self.state.params.crop_radius) * 0.08, 1.0),
            origin=self._visible_table_center(),
        )
        marker.compute_vertex_normals()
        return marker

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
        self._replace_table_center_marker()
        self._replace_geometry("crop_radius_guide", self._build_crop_radius_guide(), self._material_for_lines([1.0, 0.72, 0.1, 1.0]))
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

