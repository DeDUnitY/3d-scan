from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from object_config import get_default_cad_path, get_pose_params_file, get_reconstruction_dir

from .bundle_io import LoadedBundle, load_bundle
from .filters import CloudRenderData, build_export_cloud, build_render_data
from .pose_math import PoseParams


def _ensure_color_array(colors_rgb: np.ndarray | None, count: int, fallback: tuple[float, float, float]) -> np.ndarray:
    if colors_rgb is None or len(colors_rgb) != count:
        return np.tile(np.asarray(fallback, dtype=np.float64).reshape(1, 3), (count, 1))
    return np.clip(colors_rgb.astype(np.float64) / 255.0, 0.0, 1.0)


def _frame_palette(frame_ids: np.ndarray) -> np.ndarray:
    if len(frame_ids) == 0:
        return np.empty((0, 3), dtype=np.float64)
    unique_ids = np.unique(frame_ids)
    palette = {}
    for index, frame_id in enumerate(unique_ids):
        hue = (index * 0.61803398875) % 1.0
        palette[int(frame_id)] = np.asarray(_hsv_to_rgb(hue, 0.65, 1.0), dtype=np.float64)
    return np.vstack([palette[int(frame_id)] for frame_id in frame_ids])


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


@dataclass(slots=True)
class SceneGeometries:
    merged_cloud: o3d.geometry.PointCloud
    reference_mesh: o3d.geometry.TriangleMesh | None


class DesktopTunerState:
    def __init__(
        self,
        bundle_path: Path | None = None,
        pose_path: Path | None = None,
        preview_max_points: int = 120_000,
        reference_path: Path | None = None,
    ) -> None:
        self.pose_path = Path(pose_path) if pose_path is not None else get_pose_params_file()
        self.bundle = load_bundle(bundle_path=bundle_path, pose_path=self.pose_path)
        self.params = PoseParams.from_bundle(self.bundle)
        self.preview_max_points = int(preview_max_points)
        self.color_mode = "rgb"
        self.show_merged = True
        self.show_reference = True
        self.white_background = False
        self.point_size = 2.0
        self.reference_path = Path(reference_path) if reference_path is not None else get_default_cad_path()
        self.reference_mesh = self._load_reference_mesh(self.reference_path)
        self.last_preview = build_render_data(self.bundle, self.params, self.preview_max_points)

    def _load_reference_mesh(self, path: Path) -> o3d.geometry.TriangleMesh | None:
        if not path.exists():
            return None
        try:
            mesh = o3d.io.read_triangle_mesh(str(path))
        except Exception:
            return None
        if mesh.is_empty():
            return None
        mesh.compute_vertex_normals()
        return mesh

    def rebuild_preview(self) -> CloudRenderData:
        self.last_preview = build_render_data(self.bundle, self.params, self.preview_max_points)
        return self.last_preview

    def export_current_cloud(self, export_path: Path | None = None) -> Path:
        export_path = Path(export_path) if export_path is not None else get_reconstruction_dir() / "tuned_cloud_preview.ply"
        points, colors_rgb = build_export_cloud(self.bundle, self.params)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors_rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb.astype(np.float64) / 255.0, 0.0, 1.0))
        o3d.io.write_point_cloud(str(export_path), pcd)
        return export_path

    def export_current_cloud_npz(self, export_path: Path | None = None) -> Path:
        export_path = Path(export_path) if export_path is not None else get_reconstruction_dir() / "tuned_cloud_filtered.npz"
        points, colors_rgb = build_export_cloud(self.bundle, self.params)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "points_xyz": points.astype(np.float32, copy=False),
        }
        if colors_rgb is not None:
            payload["colors_rgb"] = np.clip(colors_rgb, 0, 255).astype(np.uint8, copy=False)
        np.savez_compressed(str(export_path), **payload)
        return export_path

    def save_pose_json(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else self.pose_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.params.to_pose_json(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def frame_ids(self) -> list[int]:
        return [frame.frame_id for frame in self.bundle.frames]

    def current_stats_text(self) -> str:
        preview = self.last_preview
        return (
            f"Points: raw {preview.raw_points:,} | "
            f"after filters {preview.filtered_points:,} | "
            f"in preview {preview.rendered_points:,}"
        )

    def build_scene_geometries(self) -> SceneGeometries:
        preview = self.last_preview
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(preview.merged_points.astype(np.float64))
        merged.colors = o3d.utility.Vector3dVector(self._build_merged_colors(preview))
        return SceneGeometries(merged_cloud=merged, reference_mesh=self.reference_mesh)

    def _build_merged_colors(self, preview: CloudRenderData) -> np.ndarray:
        if self.color_mode == "frame":
            return _frame_palette(preview.merged_frame_ids)
        return _ensure_color_array(preview.merged_colors_rgb, len(preview.merged_points), fallback=(0.306, 0.631, 1.0))

