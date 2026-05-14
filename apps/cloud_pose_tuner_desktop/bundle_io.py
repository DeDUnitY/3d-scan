from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from object_config import get_pose_params_file, get_reconstruction_dir


BUNDLE_NAME = "stereo_alignment_bundle.npz"


@dataclass(slots=True)
class FrameBundle:
    frame_id: int
    points_cam: np.ndarray
    colors_rgb: np.ndarray | None = None


@dataclass(slots=True)
class BundleDefaults:
    rotation_step_deg: float
    camera_start_angle_deg: float
    extra_frame_rot_z_deg: float
    platform_rotation_sign: int
    orbit_radius: float
    camera_height: float
    camera_tilt_deg: float
    frame_tilt_x_deg: float
    frame_tilt_y_deg: float
    frame_roll_z_deg: float
    camera_offset_y: float
    camera_min_distance: float
    camera_max_distance: float
    use_turntable: bool
    table_center_x: float
    table_center_y: float
    table_center_z: float
    frame_enabled: dict[int, bool]
    use_crop: bool
    crop_radius: float
    z_min: float
    z_max: float
    invert_x: bool
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


@dataclass(slots=True)
class LoadedBundle:
    bundle_path: Path
    frames: list[FrameBundle]
    defaults: BundleDefaults
    has_colors: bool


def resolve_bundle_path(bundle_path: Path | None = None) -> Path:
    if bundle_path is not None:
        return Path(bundle_path)
    candidates = [
        get_reconstruction_dir() / BUNDLE_NAME,
        Path(__file__).resolve().parents[2] / "outputs" / "reconstruction" / BUNDLE_NAME,
        Path(__file__).resolve().parents[2] / "outputs" / "tri" / "reconstruction" / BUNDLE_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _coerce_frame_enabled(raw: object) -> dict[int, bool]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, bool] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = bool(value)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_named_flags(raw: object) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        out[key.strip().lower()] = bool(value)
    return out


def _legacy_target_to_named_filter(mode: str, target_color_rgb: tuple[int, int, int]) -> tuple[dict[str, bool], dict[str, bool]]:
    named_targets = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
        "magenta": (255, 0, 255),
    }
    picked = "red"
    for color_name, rgb in named_targets.items():
        if tuple(int(x) for x in target_color_rgb[:3]) == rgb:
            picked = color_name
            break
    if mode == "exclude_target":
        return {}, {picked: True}
    if mode == "target":
        return {picked: True}, {}
    return {}, {}


def _read_pose_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_defaults(data: np.lib.npyio.NpzFile, pose: dict) -> BundleDefaults:
    def get_float(name: str, fallback: float) -> float:
        return float(data[name]) if name in data.files else float(fallback)

    def pose_or_default(pose_key: str, default_value):
        return pose.get(pose_key, default_value)

    legacy_mode = str(pose_or_default("COLOR_FILTER_MODE", "exclude_dark"))
    legacy_target = tuple(int(x) for x in pose_or_default("TARGET_COLOR_RGB", [255, 50, 50])[:3])
    include_filters = _coerce_named_flags(pose_or_default("COLOR_INCLUDE_FILTERS", {}))
    exclude_filters = _coerce_named_flags(pose_or_default("COLOR_EXCLUDE_FILTERS", {}))
    if not include_filters and not exclude_filters:
        include_filters, exclude_filters = _legacy_target_to_named_filter(legacy_mode, legacy_target)

    return BundleDefaults(
        rotation_step_deg=float(pose_or_default("TABLE_ROTATION_STEP", get_float("rotation_step_deg", 0.0))),
        camera_start_angle_deg=float(pose_or_default("CAMERA_START_ANGLE_DEG", get_float("camera_start_angle_deg", 0.0))),
        extra_frame_rot_z_deg=float(pose_or_default("EXTRA_FRAME_ROT_Z_DEG", get_float("extra_frame_rot_z_deg", 0.0))),
        platform_rotation_sign=int(pose_or_default("PLATFORM_ROTATION_SIGN", int(data["platform_rotation_sign"]))),
        orbit_radius=float(pose_or_default("ORBIT_RADIUS", get_float("orbit_radius", 0.0))),
        camera_height=float(pose_or_default("CAMERA_HEIGHT", get_float("camera_height", 0.0))),
        camera_tilt_deg=float(pose_or_default("CAMERA_TILT", get_float("camera_tilt_deg", 0.0))),
        frame_tilt_x_deg=float(pose_or_default("FRAME_TILT_X_DEG", get_float("frame_tilt_x_deg", 0.0))),
        frame_tilt_y_deg=float(pose_or_default("FRAME_TILT_Y_DEG", get_float("frame_tilt_y_deg", 0.0))),
        frame_roll_z_deg=float(pose_or_default("FRAME_ROLL_Z_DEG", get_float("frame_roll_z_deg", 0.0))),
        camera_offset_y=float(pose_or_default("CAMERA_OFFSET_Y", get_float("camera_offset_y", 0.0))),
        camera_min_distance=float(pose_or_default("CAMERA_MIN_DISTANCE", get_float("camera_min_distance", 0.0))),
        camera_max_distance=float(pose_or_default("CAMERA_MAX_DISTANCE", get_float("camera_max_distance", 0.0))),
        use_turntable=bool(pose_or_default("USE_TURNTABLE", True)),
        table_center_x=float(pose_or_default("TABLE_CENTER_X", 0.0)),
        table_center_y=float(pose_or_default("TABLE_CENTER_Y", 0.0)),
        table_center_z=float(pose_or_default("TABLE_CENTER_Z", 0.0)),
        frame_enabled=_coerce_frame_enabled(pose_or_default("FRAME_ENABLED", {})),
        use_crop=bool(pose_or_default("USE_GEOMETRIC_CROP", True)),
        crop_radius=float(pose_or_default("CROP_RADIUS", 30.0)),
        z_min=float(pose_or_default("Z_MIN", -2.0)),
        z_max=float(pose_or_default("Z_MAX", 15.0)),
        invert_x=bool(pose_or_default("INVERT_X_FINAL", True)),
        use_color_filter=bool(pose_or_default("USE_COLOR_FILTER", False)),
        color_include_filters=include_filters,
        color_exclude_filters=exclude_filters,
        color_tolerance=int(pose_or_default("COLOR_TOLERANCE", 150)),
        dark_threshold=int(pose_or_default("DARK_THRESHOLD", 40)),
        use_cut_background_filter=bool(pose_or_default("USE_CUT_BACKGROUND_FILTER", True)),
        cut_background_color_rgb=tuple(int(x) for x in pose_or_default("CUT_BACKGROUND_COLOR_RGB", [255, 0, 255])[:3]),
        cut_background_color_tolerance=int(pose_or_default("CUT_BACKGROUND_COLOR_TOLERANCE", 5)),
        use_isolated_filter=bool(pose_or_default("USE_ISOLATED_FILTER", False)),
        isolated_radius=float(pose_or_default("ISOLATED_RADIUS", 1.5)),
        isolated_min_neighbors=int(pose_or_default("ISOLATED_MIN_NEIGHBORS", 6)),
    )


def _build_defaults_from_pose_only(pose: dict) -> BundleDefaults:
    def pose_or_default(pose_key: str, default_value):
        return pose.get(pose_key, default_value)

    legacy_mode = str(pose_or_default("COLOR_FILTER_MODE", "exclude_dark"))
    legacy_target = tuple(int(x) for x in pose_or_default("TARGET_COLOR_RGB", [255, 50, 50])[:3])
    include_filters = _coerce_named_flags(pose_or_default("COLOR_INCLUDE_FILTERS", {}))
    exclude_filters = _coerce_named_flags(pose_or_default("COLOR_EXCLUDE_FILTERS", {}))
    if not include_filters and not exclude_filters:
        include_filters, exclude_filters = _legacy_target_to_named_filter(legacy_mode, legacy_target)

    return BundleDefaults(
        rotation_step_deg=float(pose_or_default("TABLE_ROTATION_STEP", 0.0)),
        camera_start_angle_deg=float(pose_or_default("CAMERA_START_ANGLE_DEG", 0.0)),
        extra_frame_rot_z_deg=float(pose_or_default("EXTRA_FRAME_ROT_Z_DEG", 0.0)),
        platform_rotation_sign=int(pose_or_default("PLATFORM_ROTATION_SIGN", 1)),
        orbit_radius=float(pose_or_default("ORBIT_RADIUS", 0.0)),
        camera_height=float(pose_or_default("CAMERA_HEIGHT", 0.0)),
        camera_tilt_deg=float(pose_or_default("CAMERA_TILT", 0.0)),
        frame_tilt_x_deg=float(pose_or_default("FRAME_TILT_X_DEG", 0.0)),
        frame_tilt_y_deg=float(pose_or_default("FRAME_TILT_Y_DEG", 0.0)),
        frame_roll_z_deg=float(pose_or_default("FRAME_ROLL_Z_DEG", 0.0)),
        camera_offset_y=float(pose_or_default("CAMERA_OFFSET_Y", 0.0)),
        # NPY clouds are typically already merged/exported, so strict distance gating
        # and turntable math from capture-time bundle should be off by default.
        camera_min_distance=0.0,
        camera_max_distance=0.0,
        use_turntable=False,
        table_center_x=float(pose_or_default("TABLE_CENTER_X", 0.0)),
        table_center_y=float(pose_or_default("TABLE_CENTER_Y", 0.0)),
        table_center_z=float(pose_or_default("TABLE_CENTER_Z", 0.0)),
        frame_enabled=_coerce_frame_enabled(pose_or_default("FRAME_ENABLED", {"0": True})),
        use_crop=False,
        crop_radius=float(pose_or_default("CROP_RADIUS", 30.0)),
        z_min=float(pose_or_default("Z_MIN", -200.0)),
        z_max=float(pose_or_default("Z_MAX", 200.0)),
        invert_x=False,
        use_color_filter=False,
        color_include_filters=include_filters,
        color_exclude_filters=exclude_filters,
        color_tolerance=int(pose_or_default("COLOR_TOLERANCE", 150)),
        dark_threshold=int(pose_or_default("DARK_THRESHOLD", 40)),
        use_cut_background_filter=bool(pose_or_default("USE_CUT_BACKGROUND_FILTER", True)),
        cut_background_color_rgb=tuple(int(x) for x in pose_or_default("CUT_BACKGROUND_COLOR_RGB", [255, 0, 255])[:3]),
        cut_background_color_tolerance=int(pose_or_default("CUT_BACKGROUND_COLOR_TOLERANCE", 5)),
        use_isolated_filter=bool(pose_or_default("USE_ISOLATED_FILTER", False)),
        isolated_radius=float(pose_or_default("ISOLATED_RADIUS", 1.5)),
        isolated_min_neighbors=int(pose_or_default("ISOLATED_MIN_NEIGHBORS", 6)),
    )


def load_bundle(
    bundle_path: Path | None = None,
    pose_path: Path | None = None,
) -> LoadedBundle:
    bundle_path = resolve_bundle_path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Run apps/main.py first.")

    pose_path = Path(pose_path) if pose_path is not None else get_pose_params_file()
    pose = _read_pose_json(pose_path)

    if bundle_path.suffix.lower() == ".npy":
        raw = np.load(bundle_path, allow_pickle=False)
        points = np.asarray(raw)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"NPY cloud must have shape (N,3+) but got {points.shape} from {bundle_path}")

        xyz = points[:, :3].astype(np.float32, copy=False)
        colors_rgb = None
        has_colors = points.shape[1] >= 6
        if has_colors:
            colors_rgb = np.clip(points[:, 3:6], 0, 255).astype(np.uint8, copy=False)
        frames = [FrameBundle(frame_id=0, points_cam=xyz, colors_rgb=colors_rgb)]
        defaults = _build_defaults_from_pose_only(pose)
        return LoadedBundle(bundle_path=bundle_path, frames=frames, defaults=defaults, has_colors=has_colors)

    data = np.load(bundle_path, allow_pickle=True)
    frame_ids = data["frame_indices"].astype(np.int32).tolist()
    clouds_cam = [np.asarray(cloud, dtype=np.float32) for cloud in data["clouds_cam"].tolist()]
    colors_list = None
    has_colors = "colors_rgb" in data.files
    if has_colors:
        colors_list = [np.asarray(colors, dtype=np.uint8) if colors is not None else None for colors in data["colors_rgb"].tolist()]

    frames: list[FrameBundle] = []
    for index, (frame_id, points_cam) in enumerate(zip(frame_ids, clouds_cam)):
        colors_rgb = colors_list[index] if colors_list is not None else None
        frames.append(FrameBundle(frame_id=int(frame_id), points_cam=points_cam, colors_rgb=colors_rgb))

    defaults = _build_defaults(data, pose)
    return LoadedBundle(bundle_path=bundle_path, frames=frames, defaults=defaults, has_colors=has_colors)

