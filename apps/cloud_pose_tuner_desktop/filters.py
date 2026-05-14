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
        angle_undo = -np.deg2rad(params.platform_rotation_sign * params.table_rotation_step * frame.frame_id)
        world = rotate_around_z(world, angle_undo, params.table_center.astype(np.float32)).astype(np.float32)

    if params.invert_x:
        world = world.copy()
        world[:, 0] *= -1.0

    if params.use_crop:
        center = params.table_center.astype(np.float32)
        rx = world[:, 0] - center[0]
        ry = world[:, 1] - center[1]
        rxy = np.hypot(rx, ry)
        crop_mask = (rxy < float(params.crop_radius)) & (world[:, 2] > float(params.z_min)) & (world[:, 2] < float(params.z_max))
        world = world[crop_mask]
        if colors_rgb is not None:
            colors_rgb = colors_rgb[crop_mask]

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

