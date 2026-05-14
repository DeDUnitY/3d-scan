"""
Mask recorded stereo frames by selecting the object once per stereo side.

The script uses OpenCV GrabCut with the selected rectangle on every recorded
left/right frame, fills the background with a marker color, and can overwrite
the original PNG files after creating a backup.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "outputs" / "recorded" / "frames"
METADATA_FILE = "object_mask_metadata.json"
DEFAULT_FILL_RGB = (255, 0, 255)
cv2 = None
np = None


def parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        parts = [int(x.strip()) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must be R,G,B, for example 255,0,255") from exc
    if len(parts) != 3 or any(x < 0 or x > 255 for x in parts):
        raise argparse.ArgumentTypeError("Color must contain three values in range 0..255")
    return parts[0], parts[1], parts[2]


def collect_frame_images(input_dir: Path) -> list[Path]:
    patterns = (
        "capture_*_left.png",
        "capture_*_right.png",
        "frame_*_left.png",
        "frame_*_right.png",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.glob(pattern))
    return sorted(set(files), key=lambda p: p.name)


def expand_rect(rect: tuple[int, int, int, int], padding: int, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def select_roi(image_path: Path, scale: float, padding: int, window_title: str) -> tuple[int, int, int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for ROI selection: {image_path}")

    h, w = image.shape[:2]
    if scale <= 0:
        raise ValueError("--select-scale must be > 0")
    if scale != 1.0:
        shown = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        shown = image

    print(f"Select the object rectangle for {image_path.name}, then press Enter or Space. Press C to cancel.")
    raw = cv2.selectROI(window_title, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_title)
    sx, sy, sw, sh = (int(v) for v in raw)
    if sw <= 0 or sh <= 0:
        raise RuntimeError("ROI selection was cancelled or empty.")

    rect = (
        int(round(sx / scale)),
        int(round(sy / scale)),
        int(round(sw / scale)),
        int(round(sh / scale)),
    )
    return expand_rect(rect, padding=padding, width=w, height=h)


def image_side(path: Path) -> str:
    name = path.name.lower()
    if "_right." in name:
        return "right"
    return "left"


def first_file_for_side(files: list[Path], side: str) -> Path | None:
    for path in files:
        if image_side(path) == side:
            return path
    return None


def load_rois_from_metadata(path: Path, image_shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    h, w = image_shape
    rois_raw = data.get("roi_by_side")
    if isinstance(rois_raw, dict):
        rois: dict[str, tuple[int, int, int, int]] = {}
        for side in ("left", "right"):
            raw = rois_raw.get(side)
            if isinstance(raw, list) and len(raw) == 4:
                rect = tuple(int(v) for v in raw)
                rois[side] = expand_rect(rect, padding=0, width=w, height=h)
        if rois:
            if "left" not in rois and "right" in rois:
                rois["left"] = rois["right"]
            if "right" not in rois and "left" in rois:
                rois["right"] = rois["left"]
            return rois

    raw = data.get("roi_xywh")
    if not isinstance(raw, list) or len(raw) != 4:
        raise RuntimeError(f"Metadata does not contain roi_by_side or roi_xywh: {path}")
    rect = expand_rect(tuple(int(v) for v in raw), padding=0, width=w, height=h)
    return {"left": rect, "right": rect}


def make_backup_dir(input_dir: Path) -> Path:
    base = input_dir.parent / f"{input_dir.name}_backup_before_mask"
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_dir.parent / f"{input_dir.name}_backup_before_mask_{stamp}"


def backup_files(files: list[Path], backup_dir: Path, input_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        rel = src.relative_to(input_dir)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def fill_enclosed_mask_holes(keep: np.ndarray) -> np.ndarray:
    """Treat background fully enclosed by object pixels as object."""
    h, w = keep.shape[:2]
    background = (~keep).astype(np.uint8)
    padded = np.pad(background, pad_width=1, mode="constant", constant_values=1)
    flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 2)
    outside_background = padded[1:-1, 1:-1] == 2
    enclosed_background = background.astype(bool) & ~outside_background
    return keep | enclosed_background


def apply_grabcut(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    fill_bgr: tuple[int, int, int],
    iterations: int,
    work_scale: float,
    fill_holes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x, y, rw, rh = expand_rect(rect, padding=0, width=w, height=h)
    if rw < 2 or rh < 2:
        raise RuntimeError(f"Invalid ROI for image size {w}x{h}: {(x, y, rw, rh)}")

    scale = min(max(float(work_scale), 0.05), 1.0)
    if scale < 1.0:
        work_w = max(2, int(round(w * scale)))
        work_h = max(2, int(round(h * scale)))
        work_image = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        work_rect = (
            max(0, int(round(x * scale))),
            max(0, int(round(y * scale))),
            max(2, int(round(rw * scale))),
            max(2, int(round(rh * scale))),
        )
        work_rect = expand_rect(work_rect, padding=0, width=work_w, height=work_h)
    else:
        work_image = image
        work_rect = (x, y, rw, rh)

    mask = np.zeros(work_image.shape[:2], dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(work_image, mask, work_rect, bg_model, fg_model, max(1, iterations), cv2.GC_INIT_WITH_RECT)

    keep = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if scale < 1.0:
        keep = cv2.resize(keep.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if fill_holes:
        keep = fill_enclosed_mask_holes(keep)

    output = np.empty_like(image)
    output[:, :] = np.array(fill_bgr, dtype=np.uint8)
    output[keep] = image[keep]
    return output, keep.astype(np.uint8) * 255


def process_image(
    src: Path,
    input_dir: Path,
    output_dir: Path,
    masks_dir: Path,
    rois: dict[str, tuple[int, int, int, int]],
    fill_bgr: tuple[int, int, int],
    iterations: int,
    work_scale: float,
    fill_holes: bool,
    save_masks: bool,
) -> str | None:
    image = cv2.imread(str(src))
    if image is None:
        print(f"Skip unreadable image: {src}")
        return None
    side = image_side(src)
    masked, mask = apply_grabcut(image, rois[side], fill_bgr, iterations, work_scale, fill_holes)

    rel = src.relative_to(input_dir)
    dst = output_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), masked):
        raise RuntimeError(f"Could not write masked image: {dst}")
    if save_masks:
        mask_path = masks_dir / f"{src.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
    return str(rel)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select an object once and fill background on all recorded stereo frames."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help=f"Input frames directory (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=None, help="Output directory when not using --in-place.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite source PNG files after creating a backup.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup when using --in-place.")
    parser.add_argument("--reuse-metadata", action="store_true", help=f"Reuse ROI from {METADATA_FILE} without opening ROI selector.")
    parser.add_argument("--same-roi-both-sides", action="store_true", help="Select one ROI and use it for both left and right images.")
    parser.add_argument("--select-scale", type=float, default=0.5, help="Scale for ROI selection window (default: 0.5).")
    parser.add_argument("--roi-padding", type=int, default=20, help="Pixels added around selected ROI (default: 20).")
    parser.add_argument("--grabcut-iterations", type=int, default=5, help="GrabCut iterations per image (default: 5).")
    parser.add_argument("--work-scale", type=float, default=1.0, help="Run GrabCut on resized image, 0.05..1.0 (default: 1.0).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker threads for image processing (default: 1).")
    parser.add_argument("--no-fill-holes", action="store_true", help="Do not fill background holes fully enclosed by object mask.")
    parser.add_argument("--fill-rgb", type=parse_rgb, default=DEFAULT_FILL_RGB, help="Background marker color as R,G,B (default: 255,0,255).")
    parser.add_argument("--save-masks", action="store_true", help="Also save binary masks to a masks subfolder.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images, useful for testing.")
    return parser


def main() -> None:
    global cv2, np
    args = build_parser().parse_args()

    import cv2 as cv2_module
    import numpy as np_module

    cv2 = cv2_module
    np = np_module
    if args.workers > 1:
        cv2.setNumThreads(1)

    input_dir = args.input
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = collect_frame_images(input_dir)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No recorded frame PNG files found in {input_dir}")

    first_left = first_file_for_side(files, "left")
    first_right = first_file_for_side(files, "right")
    first_sample = first_left or first_right or files[0]
    first_image = cv2.imread(str(first_sample))
    if first_image is None:
        raise RuntimeError(f"Could not read first image: {first_sample}")

    metadata_path = input_dir / METADATA_FILE
    if args.reuse_metadata:
        rois = load_rois_from_metadata(metadata_path, image_shape=first_image.shape[:2])
    else:
        if first_left is None:
            raise RuntimeError("No left images found for ROI selection.")
        left_roi = select_roi(first_left, scale=args.select_scale, padding=args.roi_padding, window_title="Select LEFT object ROI")
        if args.same_roi_both_sides or first_right is None:
            right_roi = left_roi
        else:
            right_roi = select_roi(first_right, scale=args.select_scale, padding=args.roi_padding, window_title="Select RIGHT object ROI")
        rois = {"left": left_roi, "right": right_roi}

    output_dir = input_dir if args.in_place else (args.output or input_dir.parent / f"{input_dir.name}_masked")
    output_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = None
    if args.in_place and not args.no_backup:
        backup_dir = make_backup_dir(input_dir)
        backup_files(files, backup_dir, input_dir)
        print(f"Backup saved: {backup_dir}")

    fill_rgb = tuple(int(x) for x in args.fill_rgb)
    fill_bgr = (fill_rgb[2], fill_rgb[1], fill_rgb[0])
    processed: list[str] = []
    masks_dir = output_dir / "masks"
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    work_scale = min(max(float(args.work_scale), 0.05), 1.0)
    workers = max(1, int(args.workers))
    print(
        f"Processing {len(files)} images: iterations={args.grabcut_iterations}, "
        f"work_scale={work_scale:g}, workers={workers}, fill_holes={not args.no_fill_holes}"
    )

    if workers == 1:
        for index, src in enumerate(files, start=1):
            rel = process_image(
                src=src,
                input_dir=input_dir,
                output_dir=output_dir,
                masks_dir=masks_dir,
                rois=rois,
                fill_bgr=fill_bgr,
                iterations=args.grabcut_iterations,
                work_scale=work_scale,
                fill_holes=not args.no_fill_holes,
                save_masks=args.save_masks,
            )
            if rel is not None:
                processed.append(rel)
            print(f"[{index}/{len(files)}] {src.name}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_src = {
                executor.submit(
                    process_image,
                    src,
                    input_dir,
                    output_dir,
                    masks_dir,
                    rois,
                    fill_bgr,
                    args.grabcut_iterations,
                    work_scale,
                    not args.no_fill_holes,
                    args.save_masks,
                ): src
                for src in files
            }
            for index, future in enumerate(as_completed(future_to_src), start=1):
                src = future_to_src[future]
                rel = future.result()
                if rel is not None:
                    processed.append(rel)
                print(f"[{index}/{len(files)}] {src.name}")
        processed.sort()

    metadata = {
        "roi_xywh": [int(v) for v in rois["left"]],
        "roi_by_side": {
            "left": [int(v) for v in rois["left"]],
            "right": [int(v) for v in rois["right"]],
        },
        "fill_color_rgb": list(fill_rgb),
        "grabcut_iterations": int(args.grabcut_iterations),
        "work_scale": float(work_scale),
        "workers": int(workers),
        "fill_holes": bool(not args.no_fill_holes),
        "roi_padding": int(args.roi_padding),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "in_place": bool(args.in_place),
        "backup_dir": None if backup_dir is None else str(backup_dir),
        "processed_files": processed,
    }
    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Processed {len(processed)} images. Metadata: {output_dir / METADATA_FILE}")


if __name__ == "__main__":
    main()
