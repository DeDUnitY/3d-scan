"""
Interactive stereo disparity tuner with OpenCV trackbars.

Features:
- Uses stereo calibration from configs/stereo_calib.npz
- Lets you switch frame pair index
- Tune key SGBM parameters with sliders
- Shows semi-transparent depth overlay over the rectified left image
- Save tuned parameters to configs/sgbm_tuned_params.json (shared across objects)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


from object_config import get_calibration_file, get_sgbm_params_file

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "outputs" / "recorded" / "frames"
CALIBRATION_FILE = get_calibration_file()
OUTPUT_JSON = get_sgbm_params_file()

WIN_OVERLAY = "Stereo Tune - Overlay"
WIN_DISP = "Stereo Tune - Disparity"
WIN_CTRL = "Stereo Tune - Controls"

TB = {
    "pair_idx": "pair_idx",
    "min_disp": "min_disp",
    "num_disp_x16": "num_x16",
    "block_size": "block",
    "uniqueness": "uniq",
    "speckle_window": "sp_win",
    "speckle_range": "sp_rng",
    "disp12_diff": "d12",
    "pre_filter_cap": "pre_cap",
    "median_k": "median",
    "alpha": "alpha",
    "color_map": "cmap",
    "preview_pct": "scale",
    "use_3way": "3way",
    "backend": "backend",
}

PARAM_HELP = [
    ("pair_idx", "номер стереопары/кадра"),
    ("min_disp", "минимальная диспаратность; увеличивай, если глубина начинается далеко"),
    ("num_disp_x16", "диапазон поиска (x16); больше = дальше глубина, но медленнее"),
    ("block_size", "размер окна сопоставления; больше = глаже, но меньше деталей"),
    ("uniqueness", "строгость совпадения; больше = меньше ложных точек, но больше дыр"),
    ("speckle_window", "фильтр мелких шумовых областей; больше = сильнее чистка"),
    ("speckle_range", "допустимый разброс диспаратности внутри шумовой области"),
    ("disp12_diff", "порог проверки левый-правый; меньше = строже"),
    ("pre_filter_cap", "ограничение яркости перед матчингом; обычно близко к максимуму"),
    ("median_k", "медианный постфильтр: 0 выкл, больше = глаже, но размывает границы"),
    ("alpha", "прозрачность оверлея глубины поверх изображения"),
    ("color_map", "индекс цветовой палитры глубины"),
    ("preview_pct", "масштаб превью в процентах; меньше = быстрее интерфейс"),
    ("use_3way", "0 = SGBM (быстрее), 1 = SGBM_3WAY (часто чище, но медленнее)"),
    ("backend", "0 = CPU SGBM, 1 = CUDA BM (быстрый превью-режим)"),
]


def detect_cuda_support() -> Tuple[bool, str]:
    """Returns (is_available, details)."""
    if not hasattr(cv2, "cuda"):
        return False, "cv2.cuda module not found"
    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on local OpenCV build
        return False, f"cv2.cuda query failed: {exc}"
    if count <= 0:
        return False, "CUDA devices not found by OpenCV"
    if not hasattr(cv2.cuda, "createStereoBM"):
        return False, "cv2.cuda.createStereoBM not found"
    return True, f"CUDA devices: {count}"


def load_stereo_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    if not calib_path.exists():
        raise FileNotFoundError(f"Stereo calibration not found: {calib_path}")
    data = np.load(calib_path, allow_pickle=True)
    return {
        "K1": data["camera_matrix_left"],
        "D1": data["dist_coeffs_left"],
        "K2": data["camera_matrix_right"],
        "D2": data["dist_coeffs_right"],
        "R1": data["R1"],
        "R2": data["R2"],
        "P1": data["P1"],
        "P2": data["P2"],
        "Q": data["Q"],
        "image_size": tuple(int(x) for x in data["image_size"]),
    }


def collect_pairs(images_dir: Path) -> List[Tuple[Path, Path]]:
    left_files = sorted(images_dir.glob("capture_*_left.png")) + sorted(images_dir.glob("frame_*_left.png"))
    pairs: List[Tuple[Path, Path]] = []
    for left in left_files:
        right = images_dir / left.name.replace("_left.", "_right.")
        if right.exists():
            pairs.append((left, right))
    return pairs


def build_rectify_maps(calib: Dict[str, np.ndarray]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    size = calib["image_size"]
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_32FC1
    )
    return (map1x, map1y), (map2x, map2y)


def get_slider_params() -> Dict[str, int]:
    return {
        "pair_idx": cv2.getTrackbarPos(TB["pair_idx"], WIN_CTRL),
        "min_disp": cv2.getTrackbarPos(TB["min_disp"], WIN_CTRL),
        "num_disp_x16": cv2.getTrackbarPos(TB["num_disp_x16"], WIN_CTRL),
        "block_size": cv2.getTrackbarPos(TB["block_size"], WIN_CTRL),
        "uniqueness": cv2.getTrackbarPos(TB["uniqueness"], WIN_CTRL),
        "speckle_window": cv2.getTrackbarPos(TB["speckle_window"], WIN_CTRL),
        "speckle_range": cv2.getTrackbarPos(TB["speckle_range"], WIN_CTRL),
        "disp12_diff": cv2.getTrackbarPos(TB["disp12_diff"], WIN_CTRL),
        "pre_filter_cap": cv2.getTrackbarPos(TB["pre_filter_cap"], WIN_CTRL),
        "median_k": cv2.getTrackbarPos(TB["median_k"], WIN_CTRL),
        "alpha": cv2.getTrackbarPos(TB["alpha"], WIN_CTRL),
        "color_map": cv2.getTrackbarPos(TB["color_map"], WIN_CTRL),
        "preview_pct": cv2.getTrackbarPos(TB["preview_pct"], WIN_CTRL),
        "use_3way": cv2.getTrackbarPos(TB["use_3way"], WIN_CTRL),
        "backend": cv2.getTrackbarPos(TB["backend"], WIN_CTRL),
    }


def to_sgbm_params(raw: Dict[str, int]) -> Dict[str, int]:
    block_size = raw["block_size"]
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    num_disp = max(1, raw["num_disp_x16"]) * 16
    pre_filter_cap = max(1, raw["pre_filter_cap"])
    median_k = raw["median_k"]
    if median_k > 0 and median_k % 2 == 0:
        median_k += 1

    params = {
        "MIN_DISPARITY": raw["min_disp"],
        "NUM_DISPARITIES": num_disp,
        "BLOCK_SIZE": block_size,
        "P1": 8 * 3 * block_size * block_size,
        "P2": 32 * 3 * block_size * block_size,
        "DISP12_MAX_DIFF": raw["disp12_diff"],
        "UNIQUENESS_RATIO": raw["uniqueness"],
        "SPECKLE_WINDOW_SIZE": raw["speckle_window"],
        "SPECKLE_RANGE": raw["speckle_range"],
        "PRE_FILTER_CAP": pre_filter_cap,
        "DISPARITY_MEDIAN_SIZE": median_k,
        "ALPHA": raw["alpha"] / 100.0,
        "PREVIEW_PCT": max(20, raw["preview_pct"]),
        "USE_3WAY": 1 if raw["use_3way"] > 0 else 0,
    }
    return params


def compute_disparity(rect_l_gray: np.ndarray, rect_r_gray: np.ndarray, p: Dict[str, int]) -> np.ndarray:
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY if p["USE_3WAY"] else cv2.STEREO_SGBM_MODE_SGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=p["MIN_DISPARITY"],
        numDisparities=p["NUM_DISPARITIES"],
        blockSize=p["BLOCK_SIZE"],
        P1=p["P1"],
        P2=p["P2"],
        disp12MaxDiff=p["DISP12_MAX_DIFF"],
        uniquenessRatio=p["UNIQUENESS_RATIO"],
        speckleWindowSize=p["SPECKLE_WINDOW_SIZE"],
        speckleRange=p["SPECKLE_RANGE"],
        preFilterCap=p["PRE_FILTER_CAP"],
        mode=mode,
    )
    disparity = stereo.compute(rect_l_gray, rect_r_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.0
    k = p["DISPARITY_MEDIAN_SIZE"]
    if k >= 3:
        # In some OpenCV builds medianBlur supports only CV_8U.
        # Convert disparity to uint8 domain for filtering, then map back.
        valid = disparity > 0
        max_disp = max(1.0, float(p["NUM_DISPARITIES"] - 1))
        disp_u8 = np.clip((disparity / max_disp) * 255.0, 0.0, 255.0).astype(np.uint8)
        disp_u8 = cv2.medianBlur(disp_u8, k)
        disparity = disp_u8.astype(np.float32) * (max_disp / 255.0)
        disparity[~valid] = 0.0
    return disparity


def compute_disparity_cuda_bm(rect_l_gray: np.ndarray, rect_r_gray: np.ndarray, p: Dict[str, int]) -> np.ndarray:
    """
    CUDA backend using StereoBM.
    Note: this is for responsive tuning preview; output differs from CPU SGBM.
    """
    num_disp = p["NUM_DISPARITIES"]
    block_size = p["BLOCK_SIZE"]
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, block_size)
    # In some OpenCV CUDA builds StereoBM is stable only for limited ndisp values.
    # Keep CUDA preview robust by constraining search range.
    num_disp = max(16, (num_disp // 16) * 16)
    num_disp = min(num_disp, 96)

    gpu_l = cv2.cuda_GpuMat()
    gpu_r = cv2.cuda_GpuMat()
    gpu_l.upload(rect_l_gray)
    gpu_r.upload(rect_r_gray)

    stereo = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block_size)
    # Keep min disparity fixed for CUDA preview stability.
    if hasattr(stereo, "setMinDisparity"):
        stereo.setMinDisparity(0)
    if hasattr(stereo, "setPreFilterCap"):
        stereo.setPreFilterCap(int(p["PRE_FILTER_CAP"]))
    if hasattr(stereo, "setSpeckleWindowSize"):
        stereo.setSpeckleWindowSize(int(p["SPECKLE_WINDOW_SIZE"]))
    if hasattr(stereo, "setSpeckleRange"):
        stereo.setSpeckleRange(int(p["SPECKLE_RANGE"]))

    disp_gpu = stereo.compute(gpu_l, gpu_r, cv2.cuda.Stream_Null())
    disparity = disp_gpu.download().astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.0
    k = p["DISPARITY_MEDIAN_SIZE"]
    if k >= 3:
        valid = disparity > 0
        max_disp = max(1.0, float(p["NUM_DISPARITIES"] - 1))
        disp_u8 = np.clip((disparity / max_disp) * 255.0, 0.0, 255.0).astype(np.uint8)
        disp_u8 = cv2.medianBlur(disp_u8, k)
        disparity = disp_u8.astype(np.float32) * (max_disp / 255.0)
        disparity[~valid] = 0.0
    return disparity


def disparity_to_color(disparity: np.ndarray, cmap_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    valid = disparity > 0
    color = np.zeros((disparity.shape[0], disparity.shape[1], 3), dtype=np.uint8)
    if not np.any(valid):
        return color, valid

    vals = disparity[valid]
    lo = np.percentile(vals, 5)
    hi = np.percentile(vals, 95)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.clip((disparity - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    colormaps = [
        cv2.COLORMAP_JET,
        cv2.COLORMAP_TURBO,
        cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_MAGMA,
        cv2.COLORMAP_PLASMA,
        cv2.COLORMAP_VIRIDIS,
    ]
    cmap = colormaps[cmap_idx % len(colormaps)]
    color = cv2.applyColorMap(u8, cmap)
    return color, valid


def overlay_depth_on_image(base_bgr: np.ndarray, depth_color: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    out = base_bgr.copy()
    if not np.any(valid):
        return out
    valid3 = valid[:, :, None]
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, depth_color, alpha, 0.0)
    out = np.where(valid3, blended, base_bgr)
    return out


def save_params_json(path: Path, params: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def print_snippet(params: Dict[str, int]) -> None:
    print("\nUse these values in Config (main.py):")
    for key in [
        "MIN_DISPARITY",
        "NUM_DISPARITIES",
        "BLOCK_SIZE",
        "P1",
        "P2",
        "DISP12_MAX_DIFF",
        "UNIQUENESS_RATIO",
        "SPECKLE_WINDOW_SIZE",
        "SPECKLE_RANGE",
        "PRE_FILTER_CAP",
        "DISPARITY_MEDIAN_SIZE",
    ]:
        print(f"    {key} = {params[key]}")


def print_parameter_help() -> None:
    print("\nСправка по параметрам:")
    for name, desc in PARAM_HELP:
        print(f"  - {name}: {desc}")


def create_trackbars(num_pairs: int) -> None:
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 520, 440)

    def noop(_: int) -> None:
        return

    cv2.createTrackbar(TB["pair_idx"], WIN_CTRL, 0, max(0, num_pairs - 1), noop)
    cv2.createTrackbar(TB["min_disp"], WIN_CTRL, 0, 64, noop)
    cv2.createTrackbar(TB["num_disp_x16"], WIN_CTRL, 40, 128, noop)
    cv2.createTrackbar(TB["block_size"], WIN_CTRL, 9, 21, noop)
    cv2.createTrackbar(TB["uniqueness"], WIN_CTRL, 12, 50, noop)
    cv2.createTrackbar(TB["speckle_window"], WIN_CTRL, 120, 300, noop)
    cv2.createTrackbar(TB["speckle_range"], WIN_CTRL, 8, 64, noop)
    cv2.createTrackbar(TB["disp12_diff"], WIN_CTRL, 2, 25, noop)
    cv2.createTrackbar(TB["pre_filter_cap"], WIN_CTRL, 63, 63, noop)
    cv2.createTrackbar(TB["median_k"], WIN_CTRL, 3, 9, noop)
    cv2.createTrackbar(TB["alpha"], WIN_CTRL, 55, 100, noop)
    cv2.createTrackbar(TB["color_map"], WIN_CTRL, 1, 5, noop)
    cv2.createTrackbar(TB["preview_pct"], WIN_CTRL, 55, 100, noop)
    cv2.createTrackbar(TB["use_3way"], WIN_CTRL, 0, 1, noop)
    cv2.createTrackbar(TB["backend"], WIN_CTRL, 0, 1, noop)  # 0=CPU SGBM, 1=CUDA BM


def load_rectified_pair(
    pair: Tuple[Path, Path],
    maps: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = cv2.imread(str(pair[0]))
    right = cv2.imread(str(pair[1]))
    if left is None or right is None:
        raise RuntimeError(f"Cannot read pair: {pair[0].name}, {pair[1].name}")
    if left.shape[:2][::-1] != size:
        left = cv2.resize(left, size)
    if right.shape[:2][::-1] != size:
        right = cv2.resize(right, size)
    rect_l_bgr = cv2.remap(left, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
    rect_r_bgr = cv2.remap(right, maps[1][0], maps[1][1], cv2.INTER_LINEAR)
    rect_l_gray = cv2.cvtColor(rect_l_bgr, cv2.COLOR_BGR2GRAY)
    rect_r_gray = cv2.cvtColor(rect_r_bgr, cv2.COLOR_BGR2GRAY)
    return rect_l_bgr, rect_l_gray, rect_r_gray


def main() -> None:
    calib = load_stereo_calib(CALIBRATION_FILE)
    pairs = collect_pairs(INPUT_DIR)
    if not pairs:
        raise RuntimeError(f"No stereo pairs found in {INPUT_DIR}")

    maps = build_rectify_maps(calib)
    size = calib["image_size"]
    create_trackbars(len(pairs))
    cv2.namedWindow(WIN_OVERLAY, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DISP, cv2.WINDOW_NORMAL)

    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    last_idx = -1
    last_params: Dict[str, int] | None = None
    last_raw: Dict[str, int] | None = None
    last_overlay_show: np.ndarray | None = None
    last_disp_show: np.ndarray | None = None
    cuda_ok, cuda_info = detect_cuda_support()

    print("Управление:")
    print("  - двигай ползунки в окне 'Stereo Tune - Controls'")
    print("  - [S] сохранить параметры в JSON")
    print("  - [P] вывести сниппет для Config")
    print("  - [H] вывести справку по параметрам")
    print("  - для плавности: scale=55..70, 3way=0")
    print("  - backend: 0=CPU SGBM, 1=CUDA BM (preview)")
    print(f"Статус CUDA: {cuda_info}")
    print("  - [Q] или [Esc] выход")
    print_parameter_help()

    while True:
        raw = get_slider_params()
        idx = min(max(0, raw["pair_idx"]), len(pairs) - 1)
        if idx != last_idx or idx not in cache:
            cache[idx] = load_rectified_pair(pairs[idx], maps, size)
            last_idx = idx
        rect_l_bgr, rect_l_gray, rect_r_gray = cache[idx]

        recompute = (
            (last_raw != raw)
            or (last_overlay_show is None)
            or (last_disp_show is None)
        )
        if recompute:
            params = to_sgbm_params(raw)

            preview_scale = params["PREVIEW_PCT"] / 100.0
            if preview_scale < 0.999:
                interp = cv2.INTER_AREA if preview_scale < 1.0 else cv2.INTER_LINEAR
                rect_l_gray_work = cv2.resize(rect_l_gray, None, fx=preview_scale, fy=preview_scale, interpolation=interp)
                rect_r_gray_work = cv2.resize(rect_r_gray, None, fx=preview_scale, fy=preview_scale, interpolation=interp)
                rect_l_bgr_work = cv2.resize(rect_l_bgr, None, fx=preview_scale, fy=preview_scale, interpolation=interp)
            else:
                rect_l_gray_work = rect_l_gray
                rect_r_gray_work = rect_r_gray
                rect_l_bgr_work = rect_l_bgr

            backend_name = "CPU-SGBM"
            use_cuda_backend = raw["backend"] == 1
            if use_cuda_backend and cuda_ok:
                try:
                    disparity = compute_disparity_cuda_bm(rect_l_gray_work, rect_r_gray_work, params)
                    backend_name = "CUDA-BM"
                except Exception:
                    disparity = compute_disparity(rect_l_gray_work, rect_r_gray_work, params)
                    backend_name = "CPU-SGBM(fallback)"
            else:
                disparity = compute_disparity(rect_l_gray_work, rect_r_gray_work, params)
                if use_cuda_backend and not cuda_ok:
                    backend_name = "CPU-SGBM(no CUDA)"
            depth_color, valid = disparity_to_color(disparity, raw["color_map"])
            overlay = overlay_depth_on_image(rect_l_bgr_work, depth_color, valid, params["ALPHA"])

            stats_text = (
                f"pair={idx} valid={int(np.count_nonzero(valid))} "
                f"minDisp={params['MIN_DISPARITY']} numDisp={params['NUM_DISPARITIES']} "
                f"block={params['BLOCK_SIZE']} uniq={params['UNIQUENESS_RATIO']} "
                f"scale={params['PREVIEW_PCT']}% mode={'3WAY' if params['USE_3WAY'] else 'SGBM'} "
                f"backend={backend_name}"
            )
            overlay_show = overlay.copy()
            cv2.putText(
                overlay_show, stats_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA
            )

            disp_show = depth_color.copy()
            cv2.putText(disp_show, stats_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

            last_overlay_show = overlay_show
            last_disp_show = disp_show
            last_raw = raw.copy()
            last_params = params

        cv2.imshow(WIN_OVERLAY, last_overlay_show)
        cv2.imshow(WIN_DISP, last_disp_show)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            if last_params is None:
                continue
            payload = {k: v for k, v in last_params.items() if k != "ALPHA"}
            save_params_json(OUTPUT_JSON, payload)
            print(f"Saved: {OUTPUT_JSON}")
            print_snippet(last_params)
        if key in (ord("p"), ord("P")):
            if last_params is not None:
                print_snippet(last_params)
        if key in (ord("h"), ord("H")):
            print_parameter_help()

    cv2.destroyAllWindows()
    if last_params is not None:
        print("\nLast tuned parameters:")
        print_snippet(last_params)


if __name__ == "__main__":
    main()

