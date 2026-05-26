from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bundle_io import BundleDefaults, LoadedBundle


@dataclass(slots=True)
class CameraPose:
    rotation_cw: np.ndarray
    camera_center: np.ndarray


@dataclass(slots=True)
class PoseParams:
    camera_start_angle_deg: float
    table_rotation_step: float
    extra_frame_rot_z_deg: float
    platform_rotation_sign: int
    use_turntable: bool
    table_center_x: float
    table_center_y: float
    table_center_z: float
    orbit_radius: float
    camera_height: float
    camera_tilt_deg: float
    frame_tilt_x_deg: float
    frame_tilt_y_deg: float
    frame_roll_z_deg: float
    camera_offset_y: float
    camera_min_distance: float
    camera_max_distance: float
    crop_radius: float
    z_min: float
    z_max: float
    invert_x: bool
    use_crop: bool
    use_color_filter: bool
    color_include_filters: dict[str, bool]
    color_exclude_filters: dict[str, bool]
    color_tolerance: int
    dark_threshold: int
    use_cut_background_filter: bool
    cut_background_color_rgb: tuple[int, int, int]
    cut_background_color_tolerance: int
    use_isolated_filter: bool
    isolated_radius: float
    isolated_min_neighbors: int
    frame_enabled: dict[int, bool] = field(default_factory=dict)

    @classmethod
    def from_defaults(cls, defaults: BundleDefaults) -> "PoseParams":
        return cls(
            camera_start_angle_deg=defaults.camera_start_angle_deg,
            table_rotation_step=defaults.rotation_step_deg,
            extra_frame_rot_z_deg=defaults.extra_frame_rot_z_deg,
            platform_rotation_sign=defaults.platform_rotation_sign,
            use_turntable=defaults.use_turntable,
            table_center_x=defaults.table_center_x,
            table_center_y=defaults.table_center_y,
            table_center_z=defaults.table_center_z,
            orbit_radius=defaults.orbit_radius,
            camera_height=defaults.camera_height,
            camera_tilt_deg=defaults.camera_tilt_deg,
            frame_tilt_x_deg=defaults.frame_tilt_x_deg,
            frame_tilt_y_deg=defaults.frame_tilt_y_deg,
            frame_roll_z_deg=defaults.frame_roll_z_deg,
            camera_offset_y=defaults.camera_offset_y,
            camera_min_distance=defaults.camera_min_distance,
            camera_max_distance=defaults.camera_max_distance,
            crop_radius=defaults.crop_radius,
            z_min=defaults.z_min,
            z_max=defaults.z_max,
            invert_x=defaults.invert_x,
            use_crop=defaults.use_crop,
            use_color_filter=defaults.use_color_filter,
            color_include_filters=dict(defaults.color_include_filters),
            color_exclude_filters=dict(defaults.color_exclude_filters),
            color_tolerance=defaults.color_tolerance,
            dark_threshold=defaults.dark_threshold,
            use_cut_background_filter=defaults.use_cut_background_filter,
            cut_background_color_rgb=tuple(defaults.cut_background_color_rgb),
            cut_background_color_tolerance=defaults.cut_background_color_tolerance,
            use_isolated_filter=defaults.use_isolated_filter,
            isolated_radius=defaults.isolated_radius,
            isolated_min_neighbors=defaults.isolated_min_neighbors,
            frame_enabled=dict(defaults.frame_enabled),
        )

    @classmethod
    def from_bundle(cls, bundle: LoadedBundle) -> "PoseParams":
        params = cls.from_defaults(bundle.defaults)
        for frame in bundle.frames:
            params.frame_enabled.setdefault(frame.frame_id, True)
        return params

    def copy(self) -> "PoseParams":
        return PoseParams(
            camera_start_angle_deg=self.camera_start_angle_deg,
            table_rotation_step=self.table_rotation_step,
            extra_frame_rot_z_deg=self.extra_frame_rot_z_deg,
            platform_rotation_sign=self.platform_rotation_sign,
            use_turntable=self.use_turntable,
            table_center_x=self.table_center_x,
            table_center_y=self.table_center_y,
            table_center_z=self.table_center_z,
            orbit_radius=self.orbit_radius,
            camera_height=self.camera_height,
            camera_tilt_deg=self.camera_tilt_deg,
            frame_tilt_x_deg=self.frame_tilt_x_deg,
            frame_tilt_y_deg=self.frame_tilt_y_deg,
            frame_roll_z_deg=self.frame_roll_z_deg,
            camera_offset_y=self.camera_offset_y,
            camera_min_distance=self.camera_min_distance,
            camera_max_distance=self.camera_max_distance,
            crop_radius=self.crop_radius,
            z_min=self.z_min,
            z_max=self.z_max,
            invert_x=self.invert_x,
            use_crop=self.use_crop,
            use_color_filter=self.use_color_filter,
            color_include_filters=dict(self.color_include_filters),
            color_exclude_filters=dict(self.color_exclude_filters),
            color_tolerance=self.color_tolerance,
            dark_threshold=self.dark_threshold,
            use_cut_background_filter=self.use_cut_background_filter,
            cut_background_color_rgb=tuple(self.cut_background_color_rgb),
            cut_background_color_tolerance=self.cut_background_color_tolerance,
            use_isolated_filter=self.use_isolated_filter,
            isolated_radius=self.isolated_radius,
            isolated_min_neighbors=self.isolated_min_neighbors,
            frame_enabled=dict(self.frame_enabled),
        )

    def to_pose_json(self) -> dict:
        return {
            "FRAME_ENABLED": {str(k): bool(v) for k, v in sorted(self.frame_enabled.items())},
            "USE_TURNTABLE": bool(self.use_turntable),
            "TABLE_CENTER_X": float(self.table_center_x),
            "TABLE_CENTER_Y": float(self.table_center_y),
            "TABLE_CENTER_Z": float(self.table_center_z),
            "CAMERA_START_ANGLE_DEG": float(self.camera_start_angle_deg),
            "TABLE_ROTATION_STEP": float(self.table_rotation_step),
            "EXTRA_FRAME_ROT_Z_DEG": float(self.extra_frame_rot_z_deg),
            "PLATFORM_ROTATION_SIGN": int(self.platform_rotation_sign),
            "ORBIT_RADIUS": float(self.orbit_radius),
            "CAMERA_HEIGHT": float(self.camera_height),
            "CAMERA_TILT": float(self.camera_tilt_deg),
            "FRAME_TILT_X_DEG": float(self.frame_tilt_x_deg),
            "FRAME_TILT_Y_DEG": float(self.frame_tilt_y_deg),
            "FRAME_ROLL_Z_DEG": float(self.frame_roll_z_deg),
            "CAMERA_OFFSET_Y": float(self.camera_offset_y),
            "CAMERA_MIN_DISTANCE": float(self.camera_min_distance),
            "CAMERA_MAX_DISTANCE": float(self.camera_max_distance),
            "USE_GEOMETRIC_CROP": bool(self.use_crop),
            "CROP_RADIUS": float(self.crop_radius),
            "Z_MIN": float(self.z_min),
            "Z_MAX": float(self.z_max),
            "INVERT_X_FINAL": bool(self.invert_x),
            "USE_COLOR_FILTER": bool(self.use_color_filter),
            "COLOR_INCLUDE_FILTERS": {
                str(k): bool(v) for k, v in sorted(self.color_include_filters.items())
            },
            "COLOR_EXCLUDE_FILTERS": {
                str(k): bool(v) for k, v in sorted(self.color_exclude_filters.items())
            },
            "COLOR_TOLERANCE": int(self.color_tolerance),
            "DARK_THRESHOLD": int(self.dark_threshold),
            "USE_CUT_BACKGROUND_FILTER": bool(self.use_cut_background_filter),
            "CUT_BACKGROUND_COLOR_RGB": [int(x) for x in self.cut_background_color_rgb[:3]],
            "CUT_BACKGROUND_COLOR_TOLERANCE": int(self.cut_background_color_tolerance),
            "USE_ISOLATED_FILTER": bool(self.use_isolated_filter),
            "ISOLATED_RADIUS": float(self.isolated_radius),
            "ISOLATED_MIN_NEIGHBORS": int(self.isolated_min_neighbors),
        }

    @property
    def table_center(self) -> np.ndarray:
        return np.array([self.table_center_x, self.table_center_y, self.table_center_z], dtype=np.float64)


def normalize3(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 1e-12 else None


def frame_angle_rad(frame_id: int, params: PoseParams) -> float:
    start_rad = np.deg2rad(params.camera_start_angle_deg)
    step_rad = np.deg2rad(params.table_rotation_step)
    extra_rad = np.deg2rad(params.extra_frame_rot_z_deg)
    return float(start_rad + params.platform_rotation_sign * step_rad * frame_id + extra_rad * frame_id)


def frame_turntable_delta_rad(frame_id: int, params: PoseParams) -> float:
    step_rad = np.deg2rad(params.table_rotation_step)
    extra_rad = np.deg2rad(params.extra_frame_rot_z_deg)
    return float((params.platform_rotation_sign * step_rad + extra_rad) * frame_id)


def build_camera_pose(angle_rad: float, params: PoseParams) -> CameraPose:
    camera_center = np.array(
        [
            params.orbit_radius * np.cos(angle_rad),
            params.orbit_radius * np.sin(angle_rad) + params.camera_offset_y,
            params.camera_height,
        ],
        dtype=np.float64,
    )
    to_target = normalize3(-camera_center)
    if to_target is None:
        to_target = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = normalize3(np.cross(to_target, up))
    if x_axis is None:
        x_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    y_up = normalize3(np.cross(x_axis, to_target))
    if y_up is None:
        y_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_down = -y_up
    rotation_cw = np.column_stack((x_axis, y_down, to_target))

    tilt = np.deg2rad(params.camera_tilt_deg)
    if abs(tilt) > 1e-12:
        c = np.cos(tilt)
        s = np.sin(tilt)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
        rotation_cw = rotation_cw @ rx
    return CameraPose(rotation_cw=rotation_cw, camera_center=camera_center)


def cam_to_world(points_cam: np.ndarray, pose: CameraPose) -> np.ndarray:
    return points_cam @ pose.rotation_cw.T + pose.camera_center.reshape(1, 3)


def rotate_around_z(points: np.ndarray, angle_rad: float, center: np.ndarray) -> np.ndarray:
    if len(points) == 0 or abs(angle_rad) <= 1e-12:
        return points
    shifted = points.copy()
    shifted[:, 0] -= center[0]
    shifted[:, 1] -= center[1]
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    x_new = c * shifted[:, 0] - s * shifted[:, 1]
    y_new = s * shifted[:, 0] + c * shifted[:, 1]
    shifted[:, 0] = x_new + center[0]
    shifted[:, 1] = y_new + center[1]
    return shifted


def rotate_camera_frame_tilt(
    points_cam: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    roll_z_deg: float = 0.0,
) -> np.ndarray:
    if len(points_cam) == 0:
        return points_cam
    tx = np.deg2rad(float(tilt_x_deg))
    ty = np.deg2rad(float(tilt_y_deg))
    tz = np.deg2rad(float(roll_z_deg))
    if abs(tx) <= 1e-12 and abs(ty) <= 1e-12 and abs(tz) <= 1e-12:
        return points_cam

    cx = np.cos(tx)
    sx = np.sin(tx)
    cy = np.cos(ty)
    sy = np.sin(ty)
    cz = np.cos(tz)
    sz = np.sin(tz)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot = rz @ ry @ rx
    return (points_cam.astype(np.float64, copy=False) @ rot.T).astype(points_cam.dtype, copy=False)

