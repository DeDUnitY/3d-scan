from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .bundle_io import FrameBundle, LoadedBundle
from .pose_math import (
    PoseParams,
    build_camera_pose,
    cam_to_world,
    frame_angle_rad,
    frame_turntable_delta_rad,
    rotate_around_z,
    rotate_camera_frame_tilt,
)

FILTER_COLOR_PRESETS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
}


@dataclass(slots=True)
class FrameRenderData:
    frame_id: int
    points_world: np.ndarray
    colors_rgb: np.ndarray | None
    raw_points: int
    filtered_points: int


@dataclass(slots=True)
class CloudRenderData:
    merged_points: np.ndarray
    merged_colors_rgb: np.ndarray | None
    merged_frame_ids: np.ndarray
    frames: list[FrameRenderData]
    raw_points: int
    filtered_points: int
    rendered_points: int


@dataclass(slots=True)
class AutoCenterResult:
    frame_a: int
    frame_b: int
    center_x: float
    center_y: float
    score_before: float
    score_after: float
    points_a: int
    points_b: int


def _downsample_indices(length: int, max_points: int) -> np.ndarray | None:
    if max_points <= 0 or length <= max_points:
        return None
    return np.linspace(0, length - 1, max_points, dtype=np.int32)


def _apply_color_filter(params: PoseParams, colors_rgb: np.ndarray | None) -> np.ndarray | None:
    if colors_rgb is None:
        return None
    mask = np.ones(len(colors_rgb), dtype=bool)
    colors_i16 = colors_rgb.astype(np.int16, copy=False)

    if params.use_cut_background_filter:
        target = np.asarray(params.cut_background_color_rgb[:3], dtype=np.int16).reshape(1, 3)
        tolerance = int(params.cut_background_color_tolerance)
        delta = np.abs(colors_i16 - target)
        mask &= ~np.all(delta <= tolerance, axis=1)

    if not params.use_color_filter:
        return mask

    tolerance = int(params.color_tolerance)

    include_names = [
        name
        for name, is_enabled in params.color_include_filters.items()
        if is_enabled and name in FILTER_COLOR_PRESETS
    ]
    if include_names:
        include_mask = np.zeros(len(colors_rgb), dtype=bool)
        for name in include_names:
            target = np.asarray(FILTER_COLOR_PRESETS[name], dtype=np.int16).reshape(1, 3)
            delta = np.abs(colors_i16 - target)
            include_mask |= np.all(delta <= tolerance, axis=1)
        mask &= include_mask

    exclude_names = [
        name
        for name, is_enabled in params.color_exclude_filters.items()
        if is_enabled and name in FILTER_COLOR_PRESETS
    ]
    if exclude_names:
        exclude_mask = np.zeros(len(colors_rgb), dtype=bool)
        for name in exclude_names:
            target = np.asarray(FILTER_COLOR_PRESETS[name], dtype=np.int16).reshape(1, 3)
            delta = np.abs(colors_i16 - target)
            exclude_mask |= np.all(delta <= tolerance, axis=1)
        mask &= ~exclude_mask

    # Preserve existing dark-threshold behavior as an optional extra cleanup stage.
    threshold = int(params.dark_threshold)
    if threshold > 0:
        mask &= np.any(colors_rgb > threshold, axis=1)

    return mask


def _frame_pose(frame_id: int, params: PoseParams):
    if params.use_turntable:
        return build_camera_pose(np.deg2rad(params.camera_start_angle_deg), params)
    return build_camera_pose(frame_angle_rad(frame_id, params), params)


def _remove_isolated_points(
    points_world: np.ndarray,
    colors_rgb: np.ndarray | None,
    radius: float,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points_world) == 0:
        return points_world, colors_rgb
    radius = float(radius)
    min_neighbors = int(min_neighbors)
    if radius <= 0 or min_neighbors <= 0:
        return points_world, colors_rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world.astype(np.float64, copy=False))
    if colors_rgb is not None and len(colors_rgb) == len(points_world):
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(colors_rgb.astype(np.float64) / 255.0, 0.0, 1.0)
        )
    filtered, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    out_points = np.asarray(filtered.points, dtype=np.float32)
    out_colors = None
    if colors_rgb is not None:
        filtered_colors = np.asarray(filtered.colors)
        if filtered_colors.size > 0:
            out_colors = np.clip(filtered_colors * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            out_colors = np.empty((0, 3), dtype=np.uint8)
    return out_points, out_colors


def _transform_frame(frame: FrameBundle, params: PoseParams) -> FrameRenderData:
    points_cam = np.asarray(frame.points_cam, dtype=np.float32)
    colors_rgb = None if frame.colors_rgb is None else np.asarray(frame.colors_rgb, dtype=np.uint8)
    raw_points = int(len(points_cam))
    if raw_points == 0:
        return FrameRenderData(frame_id=frame.frame_id, points_world=np.empty((0, 3), dtype=np.float32), colors_rgb=colors_rgb, raw_points=0, filtered_points=0)

    pose = _frame_pose(frame.frame_id, params)
    mask = np.ones(raw_points, dtype=bool)

    if params.camera_min_distance > 0 or params.camera_max_distance > 0:
        d_cam = np.linalg.norm(points_cam, axis=1)
        if params.camera_min_distance > 0:
            mask &= d_cam >= float(params.camera_min_distance)
        if params.camera_max_distance > 0:
            mask &= d_cam <= float(params.camera_max_distance)

    color_mask = _apply_color_filter(params, colors_rgb)
    if color_mask is not None:
        mask &= color_mask

    if not np.any(mask):
        return FrameRenderData(frame_id=frame.frame_id, points_world=np.empty((0, 3), dtype=np.float32), colors_rgb=None if colors_rgb is None else np.empty((0, 3), dtype=np.uint8), raw_points=raw_points, filtered_points=0)

    points_cam = points_cam[mask]
    if colors_rgb is not None:
        colors_rgb = colors_rgb[mask]

    points_cam = rotate_camera_frame_tilt(
        points_cam,
        params.frame_tilt_x_deg,
        params.frame_tilt_y_deg,
        params.frame_roll_z_deg,
    )
    world = cam_to_world(points_cam.astype(np.float64), pose).astype(np.float32)
    if params.use_turntable:
        angle_undo = -frame_turntable_delta_rad(frame.frame_id, params)
        world = rotate_around_z(world, angle_undo, params.table_center.astype(np.float32)).astype(np.float32)

    if params.use_crop:
        center = params.table_center.astype(np.float32)
        rx = world[:, 0] - center[0]
        ry = world[:, 1] - center[1]
        rxy = np.hypot(rx, ry)
        crop_mask = (rxy < float(params.crop_radius)) & (world[:, 2] > float(params.z_min)) & (world[:, 2] < float(params.z_max))
        world = world[crop_mask]
        if colors_rgb is not None:
            colors_rgb = colors_rgb[crop_mask]

    if params.invert_x:
        world = world.copy()
        world[:, 0] *= -1.0

    if params.use_isolated_filter:
        world, colors_rgb = _remove_isolated_points(
            world,
            colors_rgb,
            radius=params.isolated_radius,
            min_neighbors=params.isolated_min_neighbors,
        )

    return FrameRenderData(
        frame_id=frame.frame_id,
        points_world=world.astype(np.float32, copy=False),
        colors_rgb=None if colors_rgb is None else colors_rgb.astype(np.uint8, copy=False),
        raw_points=raw_points,
        filtered_points=int(len(world)),
    )


def _frame_points_before_turntable(frame: FrameBundle, params: PoseParams) -> np.ndarray:
    points_cam = np.asarray(frame.points_cam, dtype=np.float32)
    colors_rgb = None if frame.colors_rgb is None else np.asarray(frame.colors_rgb, dtype=np.uint8)
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=np.float32)

    mask = np.ones(len(points_cam), dtype=bool)
    if params.camera_min_distance > 0 or params.camera_max_distance > 0:
        d_cam = np.linalg.norm(points_cam, axis=1)
        if params.camera_min_distance > 0:
            mask &= d_cam >= float(params.camera_min_distance)
        if params.camera_max_distance > 0:
            mask &= d_cam <= float(params.camera_max_distance)

    color_mask = _apply_color_filter(params, colors_rgb)
    if color_mask is not None:
        mask &= color_mask
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    points_cam = points_cam[mask]
    points_cam = rotate_camera_frame_tilt(
        points_cam,
        params.frame_tilt_x_deg,
        params.frame_tilt_y_deg,
        params.frame_roll_z_deg,
    )
    pose = _frame_pose(frame.frame_id, params)
    return cam_to_world(points_cam.astype(np.float64), pose).astype(np.float32)


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    indices = _downsample_indices(len(points), max_points)
    if indices is None:
        return points
    return points[indices]


def _nearest_neighbor_score(source: np.ndarray, target: np.ndarray) -> float:
    if len(source) == 0 or len(target) == 0:
        return float("inf")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64, copy=False))
    tree = o3d.geometry.KDTreeFlann(pcd)
    distances = np.empty(len(source), dtype=np.float64)
    for index, point in enumerate(source.astype(np.float64, copy=False)):
        count, _, dist2 = tree.search_knn_vector_3d(point, 1)
        distances[index] = dist2[0] if count else np.inf
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return float("inf")
    return float(np.sqrt(np.percentile(finite, 70.0)))


def _pair_alignment_score(points_a: np.ndarray, points_b: np.ndarray) -> float:
    score_ab = _nearest_neighbor_score(points_a, points_b)
    score_ba = _nearest_neighbor_score(points_b, points_a)
    centroid_delta = np.linalg.norm(np.median(points_a[:, :2], axis=0) - np.median(points_b[:, :2], axis=0))
    return float((score_ab + score_ba) * 0.5 + 0.25 * centroid_delta)


def _transform_for_center(points: np.ndarray, frame_id: int, params: PoseParams, center_xy: tuple[float, float]) -> np.ndarray:
    center = np.array([center_xy[0], center_xy[1], params.table_center_z], dtype=np.float32)
    angle_undo = -frame_turntable_delta_rad(frame_id, params)
    transformed = rotate_around_z(points, angle_undo, center).astype(np.float32)
    if params.invert_x:
        transformed = transformed.copy()
        transformed[:, 0] *= -1.0
    return transformed


def _estimate_center_xy_from_centroids(
    points_a: np.ndarray,
    frame_a: int,
    points_b: np.ndarray,
    frame_b: int,
    params: PoseParams,
) -> tuple[float, float] | None:
    theta_a = frame_turntable_delta_rad(frame_a, params)
    theta_b = frame_turntable_delta_rad(frame_b, params)
    delta = theta_b - theta_a
    if abs(np.sin(delta)) <= 1e-6 and abs(1.0 - np.cos(delta)) <= 1e-6:
        return None

    ma = np.median(points_a[:, :2], axis=0).astype(np.float64)
    mb = np.median(points_b[:, :2], axis=0).astype(np.float64)
    c = np.cos(delta)
    s = np.sin(delta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    system = np.eye(2, dtype=np.float64) - rot
    rhs = mb - rot @ ma
    try:
        center = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(center).all():
        return None
    return float(center[0]), float(center[1])


def auto_tune_turntable_center_xy(
    bundle: LoadedBundle,
    params: PoseParams,
    *,
    search_radius: float = 5.0,
    max_points_per_frame: int = 6000,
    grid_size: int = 9,
    levels: int = 4,
    pair_start_index: int = 0,
    pair_gap: int = 1,
) -> AutoCenterResult:
    if not params.use_turntable:
        raise ValueError("Auto center tuning requires Use turntable model to be enabled.")

    enabled_frames = [
        frame
        for frame in bundle.frames
        if params.frame_enabled.get(frame.frame_id, True)
    ]
    if len(enabled_frames) < 2:
        raise ValueError("Enable at least two frames for auto center tuning.")

    pair_start_index = min(max(int(pair_start_index), 0), len(enabled_frames) - 2)
    pair_gap = max(int(pair_gap), 1)
    pair_b_index = min(pair_start_index + pair_gap, len(enabled_frames) - 1)
    if pair_b_index == pair_start_index:
        pair_b_index = min(pair_start_index + 1, len(enabled_frames) - 1)
    frame_a = enabled_frames[pair_start_index]
    frame_b = enabled_frames[pair_b_index]
    points_a_base = _downsample_points(
        _frame_points_before_turntable(frame_a, params),
        max(int(max_points_per_frame), 100),
    )
    points_b_base = _downsample_points(
        _frame_points_before_turntable(frame_b, params),
        max(int(max_points_per_frame), 100),
    )
    if len(points_a_base) == 0 or len(points_b_base) == 0:
        raise ValueError("Selected frame pair has no points after basic filters.")

    def score(center_xy: tuple[float, float]) -> float:
        points_a = _transform_for_center(points_a_base, frame_a.frame_id, params, center_xy)
        points_b = _transform_for_center(points_b_base, frame_b.frame_id, params, center_xy)
        return _pair_alignment_score(points_a, points_b)

    current_xy = (float(params.table_center_x), float(params.table_center_y))
    estimated_xy = _estimate_center_xy_from_centroids(
        points_a_base,
        frame_a.frame_id,
        points_b_base,
        frame_b.frame_id,
        params,
    )
    seeds = [current_xy]
    if estimated_xy is not None:
        seeds.append(estimated_xy)

    best_xy = current_xy
    best_score = score(best_xy)
    before_score = best_score
    step_radius = max(float(search_radius), 1e-6)
    grid_size = max(int(grid_size), 3)
    levels = max(int(levels), 1)

    for seed_xy in seeds:
        local_xy = seed_xy
        local_score = score(local_xy)
        local_radius = step_radius
        for _ in range(levels):
            offsets = np.linspace(-local_radius, local_radius, grid_size)
            for dx in offsets:
                for dy in offsets:
                    candidate = (local_xy[0] + float(dx), local_xy[1] + float(dy))
                    candidate_score = score(candidate)
                    if candidate_score < local_score:
                        local_xy = candidate
                        local_score = candidate_score
            local_radius /= 3.0
        if local_score < best_score:
            best_xy = local_xy
            best_score = local_score

    return AutoCenterResult(
        frame_a=frame_a.frame_id,
        frame_b=frame_b.frame_id,
        center_x=best_xy[0],
        center_y=best_xy[1],
        score_before=before_score,
        score_after=best_score,
        points_a=int(len(points_a_base)),
        points_b=int(len(points_b_base)),
    )


def build_render_data(bundle: LoadedBundle, params: PoseParams, max_points: int = 0) -> CloudRenderData:
    rendered_frames: list[FrameRenderData] = []
    merged_points: list[np.ndarray] = []
    merged_colors: list[np.ndarray] = []
    merged_frame_ids: list[np.ndarray] = []
    raw_points = 0
    filtered_points = 0

    for frame in bundle.frames:
        if params.frame_enabled.get(frame.frame_id, True) is False:
            continue
        frame_render = _transform_frame(frame, params)
        rendered_frames.append(frame_render)
        raw_points += frame_render.raw_points
        filtered_points += frame_render.filtered_points
        if frame_render.filtered_points == 0:
            continue
        merged_points.append(frame_render.points_world)
        merged_frame_ids.append(np.full(frame_render.filtered_points, frame_render.frame_id, dtype=np.int32))
        if frame_render.colors_rgb is not None:
            merged_colors.append(frame_render.colors_rgb)

    if merged_points:
        points = np.concatenate(merged_points, axis=0)
        frame_ids = np.concatenate(merged_frame_ids, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        frame_ids = np.empty((0,), dtype=np.int32)

    colors_rgb: np.ndarray | None
    if merged_colors and len(merged_colors) == len(merged_points):
        colors_rgb = np.concatenate(merged_colors, axis=0)
    else:
        colors_rgb = None

    indices = _downsample_indices(len(points), max_points)
    if indices is not None:
        points = points[indices]
        frame_ids = frame_ids[indices]
        if colors_rgb is not None:
            colors_rgb = colors_rgb[indices]

        resized_frames: list[FrameRenderData] = []
        for frame_render in rendered_frames:
            frame_indices = _downsample_indices(frame_render.filtered_points, max_points)
            if frame_indices is None:
                resized_frames.append(frame_render)
                continue
            resized_frames.append(
                FrameRenderData(
                    frame_id=frame_render.frame_id,
                    points_world=frame_render.points_world[frame_indices],
                    colors_rgb=None if frame_render.colors_rgb is None else frame_render.colors_rgb[frame_indices],
                    raw_points=frame_render.raw_points,
                    filtered_points=len(frame_indices),
                )
            )
        rendered_frames = resized_frames

    return CloudRenderData(
        merged_points=points,
        merged_colors_rgb=colors_rgb,
        merged_frame_ids=frame_ids,
        frames=rendered_frames,
        raw_points=raw_points,
        filtered_points=filtered_points,
        rendered_points=int(len(points)),
    )


def build_export_cloud(bundle: LoadedBundle, params: PoseParams) -> tuple[np.ndarray, np.ndarray | None]:
    full = build_render_data(bundle, params, max_points=0)
    return full.merged_points, full.merged_colors_rgb

