from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from object_config import get_default_cad_path, get_pose_params_file, get_reconstruction_dir

from cloud_pose_tuner_desktop.state import DesktopTunerState
from cloud_pose_tuner_desktop.viewer import DesktopCloudPoseTunerApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive desktop app for tuning point-cloud pose "
            "without recomputing disparity."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=(
            "Path to input cloud data: alignment bundle (.npz) or raw cloud (.npy, Nx3/Nx6). "
            f"Default: {get_reconstruction_dir() / 'stereo_alignment_bundle.npz'}"
        ),
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=None,
        help=f"Path to pose tuning JSON for load/save (default: {get_pose_params_file()})",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=f"Optional CAD/reference mesh path (default: {get_default_cad_path()})",
    )
    parser.add_argument(
        "--preview-max-points",
        type=int,
        default=4_000_000,
        help=(
            "Maximum number of points shown in preview for smoother UI. "
            "Export still saves the full filtered cloud."
        ),
    )
    return parser


def _open_file_dialogs(
    bundle: Path | None,
    pose: Path | None,
    reference: Path | None,
) -> Tuple[Path | None, Path, Path | None]:
    """Open native file dialogs so app can be used without CLI arguments."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return bundle, pose or get_pose_params_file(), reference or get_default_cad_path()

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected_bundle = bundle
    selected_pose = pose or get_pose_params_file()
    selected_reference = reference or get_default_cad_path()

    if selected_bundle is None:
        bundle_raw = filedialog.askopenfilename(
            title="Select input cloud file (.npz bundle or .npy cloud)",
            initialdir=str(get_reconstruction_dir()),
            filetypes=[
                ("Cloud files", "*.npz *.npy"),
                ("NPZ bundle", "*.npz"),
                ("NPY cloud", "*.npy"),
                ("All files", "*.*"),
            ],
        )
        selected_bundle = Path(bundle_raw) if bundle_raw else None

    pose_raw = filedialog.askopenfilename(
        title="Select pose JSON (Cancel to use default)",
        initialdir=str(get_pose_params_file().parent),
        initialfile=get_pose_params_file().name,
        filetypes=[("JSON", "*.json"), ("All files", "*.*")],
    )
    if pose_raw:
        selected_pose = Path(pose_raw)

    reference_raw = filedialog.askopenfilename(
        title="Select reference mesh (Cancel to skip)",
        initialdir=str(get_default_cad_path().parent),
        initialfile=get_default_cad_path().name,
        filetypes=[
            ("Mesh files", "*.stl *.ply *.obj *.glb *.gltf"),
            ("All files", "*.*"),
        ],
    )
    if reference_raw:
        selected_reference = Path(reference_raw)
    elif reference is None and not get_default_cad_path().exists():
        selected_reference = None

    root.destroy()
    return selected_bundle, selected_pose, selected_reference


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = args.bundle
    pose_path = args.pose or get_pose_params_file()
    reference_path = args.reference or get_default_cad_path()

    if args.bundle is None and args.pose is None and args.reference is None:
        bundle_path, pose_path, reference_path = _open_file_dialogs(bundle_path, pose_path, reference_path)

    state = DesktopTunerState(
        bundle_path=bundle_path,
        pose_path=pose_path,
        preview_max_points=args.preview_max_points,
        reference_path=reference_path,
    )
    app = DesktopCloudPoseTunerApp(state)
    app.run()


if __name__ == "__main__":
    main()

