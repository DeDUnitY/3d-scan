"""
Объединённый тунер: калибровка камеры (одни параметры для обеих) + карта глубины (SGBM).

- Параметры камеры (exposure, gain, brightness, contrast, gamma) задаются один раз и
  применяются к левой и правой камере одинаково.
- Режим камеры: живой предпросмотр L/R, слайдеры параметров, сохранение в camera_params.json.
- Режим глубины: ректифицированные кадры с оверлеем диспаратности, слайдеры SGBM,
  сохранение в sgbm_tuned_params.json (требуется stereo_calib.npz).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from object_config import (
    get_calibration_file,
    get_camera_params_file,
    get_sgbm_params_file,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CAMERA_PARAMS_FILE = get_camera_params_file()
CALIBRATION_FILE = get_calibration_file()
SGBM_JSON = get_sgbm_params_file()

# --- Камеры ---
CAM_LEFT_INDEX = 1
CAM_RIGHT_INDEX = 0
DEFAULT_WIDTH = 2560
DEFAULT_HEIGHT = 1440
EXPOSURE_MIN, EXPOSURE_MAX = -13.0, 0.0
TRACKBAR_STEPS = 130
GAIN_MIN, GAIN_MAX = 0, 100
GAIN_TRACKBAR_MAX = 100
PROP_RANGE = 100
PREVIEW_W, PREVIEW_H = 1920, 1080
TARGET_H = 360

# --- Окна ---
WIN_CAM = "Camera (same params L=R)"
WIN_DEPTH_OVERLAY = "Depth - Overlay"
WIN_DEPTH_DISP = "Depth - Disparity"
WIN_CTRL = "Controls"

CAM_LOCK_TB = {
    "exp": "Lock Exp",
    "gain": "Lock Gain",
    "bri": "Lock Bri",
    "con": "Lock Con",
    "gam": "Lock Gam",
}

SGBM_LOCK_TB = {
    "min_disp": "Lk min",
    "num_disp_x16": "Lk num",
    "block_size": "Lk blk",
    "uniqueness": "Lk uniq",
    "speckle_window": "Lk spw",
    "speckle_range": "Lk spr",
    "disp12_diff": "Lk d12",
    "pre_filter_cap": "Lk pre",
    "median_k": "Lk med",
    "use_3way": "Lk 3w",
}

# --- SGBM trackbar names ---
TB = {
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

DEFAULT_RAW: Dict[str, int] = {
    "min_disp": 0,
    "num_disp_x16": 40,
    "block_size": 9,
    "uniqueness": 12,
    "speckle_window": 120,
    "speckle_range": 8,
    "disp12_diff": 2,
    "pre_filter_cap": 63,
    "median_k": 3,
    "alpha": 55,
    "color_map": 1,
    "preview_pct": 55,
    "use_3way": 0,
    "backend": 0,
}

# Описание ползунков для вывода в консоль
CAMERA_SLIDER_HELP = [
    ("Exp", "Экспозиция (примерно -13..0). Больше — светлее кадр. Одинаково для L и R."),
    ("Gain", "Усиление камеры (0..100). Больше — светлее, но больше шума."),
    ("Bri", "Яркость (0..100). Зависит от поддержки драйвером."),
    ("Con", "Контраст (0..100). Зависит от поддержки драйвером."),
    ("Gam", "Гамма (0..100). Зависит от поддержки драйвером."),
    ("Save Cam", "Кнопка: сохранить camera_params.json и (если есть калибровка) sgbm_tuned_params.json."),
]

SGBM_SLIDER_HELP = [
    ("min_disp", "Минимальная диспаратность. Увеличивай, если ближние объекты не в кадре."),
    ("num_x16", "Диапазон поиска диспаратности (×16). Больше — дальше глубина, но медленнее."),
    ("block", "Размер окна сопоставления (нечётный). Больше — глаже карта, меньше деталей."),
    ("uniq", "Строгость совпадения (uniqueness). Больше — меньше шума, но больше «дыр»."),
    ("sp_win", "Окно фильтра пятен (speckle). Больше — сильнее убираются мелкие артефакты."),
    ("sp_rng", "Допустимый разброс диспаратности внутри пятна (speckle range)."),
    ("d12", "Порог проверки левый–правый (disp12MaxDiff). Меньше — строже."),
    ("pre_cap", "Ограничение яркости перед матчингом (preFilterCap). Обычно близко к макс."),
    ("median", "Размер медианного постфильтра (0=выкл). Больше — глаже, размывает границы."),
    ("alpha", "Прозрачность оверлея глубины поверх изображения (0..100%)."),
    ("cmap", "Цветовая палитра карты глубины (Jet, Turbo, Inferno, …)."),
    ("scale", "Масштаб превью в %. Меньше — быстрее отклик интерфейса."),
    ("3way", "0 = SGBM (быстрее), 1 = SGBM_3WAY (часто чище, медленнее)."),
    ("backend", "0 = CPU SGBM, 1 = CUDA StereoBM (быстрый превью, другой алгоритм)."),
]


def print_slider_help() -> None:
    """Вывести в консоль описание всех ползунков."""
    print("\n" + "=" * 60)
    print("Окно «Camera» — ползунки камеры (одни для L и R):")
    print("=" * 60)
    for name, desc in CAMERA_SLIDER_HELP:
        print(f"  {name:12} — {desc}")
    print("\n" + "=" * 60)
    print("Окно «Controls» — ползунки карты глубины (SGBM):")
    print("=" * 60)
    for name, desc in SGBM_SLIDER_HELP:
        print(f"  {name:12} — {desc}")
    print("\nКлавиши:")
    print("  F            — freeze/unfreeze текущую rectified-пару для режима глубины")
    print("  R            — сбросить ROI")
    print("  T            — запустить автоподбор по ROI")
    print("  H            — вывести эту справку")
    print("  S            — сохранить camera_params.json и sgbm_tuned_params.json")
    print("\nLock-ползунки:")
    print("  0 = параметр участвует в автоподборе")
    print("  1 = параметр зафиксирован и не меняется")
    print("=" * 60 + "\n")


# ---------- Camera params (single set for both) ----------

def load_camera_params_json() -> Dict | None:
    """Загрузить параметры камеры. Поддержка: один блок 'camera' или left/right (берём left для обоих)."""
    if not CAMERA_PARAMS_FILE.exists():
        return None
    try:
        with open(CAMERA_PARAMS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    if "camera" in data:
        return data["camera"]
    if "left" in data:
        return data["left"]
    if "exposure" in data:
        return data
    return None


def save_camera_params_json(exposure: float, gain: int, brightness: int, contrast: int, gamma: int) -> None:
    """Сохранить одни и те же параметры для левой и правой (capture_frames читает left/right)."""
    data = {
        "camera": {
            "exposure": exposure,
            "gain": gain,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
        },
        "left": {"exposure": exposure, "gain": gain, "brightness": brightness, "contrast": contrast, "gamma": gamma},
        "right": {"exposure": exposure, "gain": gain, "brightness": brightness, "contrast": contrast, "gamma": gamma},
    }
    CAMERA_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMERA_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Параметры камеры сохранены в {CAMERA_PARAMS_FILE} (одинаковые для L и R)")


# ---------- Stereo calibration & rectification ----------

def load_stereo_calib(calib_path: Path) -> Dict[str, np.ndarray] | None:
    if not calib_path.exists():
        return None
    try:
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
    except Exception:
        return None


def build_rectify_maps(calib: Dict[str, np.ndarray]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    size = calib["image_size"]
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_32FC1
    )
    return (map1x, map1y), (map2x, map2y)


# ---------- SGBM ----------

def get_slider_params_sgbm() -> Dict[str, int]:
    out = {}
    for key, tb_name in TB.items():
        try:
            out[key] = cv2.getTrackbarPos(tb_name, WIN_CTRL)
        except cv2.error:
            out[key] = DEFAULT_RAW.get(key, 0)
    return out


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
    return {
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
        valid = disparity > 0
        max_disp = max(1.0, float(p["NUM_DISPARITIES"] - 1))
        disp_u8 = np.clip((disparity / max_disp) * 255.0, 0.0, 255.0).astype(np.uint8)
        disp_u8 = cv2.medianBlur(disp_u8, k)
        disparity = disp_u8.astype(np.float32) * (max_disp / 255.0)
        disparity[~valid] = 0.0
    return disparity


def detect_cuda_support() -> Tuple[bool, str]:
    if not hasattr(cv2, "cuda"):
        return False, "cv2.cuda not found"
    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as e:
        return False, str(e)
    if count <= 0:
        return False, "no CUDA devices"
    if not hasattr(cv2.cuda, "createStereoBM"):
        return False, "createStereoBM not found"
    return True, f"CUDA devices: {count}"


def compute_disparity_cuda_bm(rect_l_gray: np.ndarray, rect_r_gray: np.ndarray, p: Dict[str, int]) -> np.ndarray:
    num_disp = max(16, (p["NUM_DISPARITIES"] // 16) * 16)
    num_disp = min(num_disp, 96)
    block_size = p["BLOCK_SIZE"]
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, block_size)
    gpu_l = cv2.cuda_GpuMat()
    gpu_r = cv2.cuda_GpuMat()
    gpu_l.upload(rect_l_gray)
    gpu_r.upload(rect_r_gray)
    stereo = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block_size)
    if hasattr(stereo, "setMinDisparity"):
        stereo.setMinDisparity(0)
    if hasattr(stereo, "setPreFilterCap"):
        stereo.setPreFilterCap(int(p["PRE_FILTER_CAP"]))
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


def disparity_to_color(
    disparity: np.ndarray,
    cmap_idx: int,
    disp_min: float,
    disp_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Преобразовать диспаратность в цвет с фиксированной шкалой,
    чтобы цвета не «прыгали» от кадра к кадру.

    Диапазон берётся из параметров SGBM: [MIN_DISPARITY, MIN_DISPARITY + NUM_DISPARITIES].
    """
    valid = disparity > 0
    color = np.zeros((*disparity.shape[:2], 3), dtype=np.uint8)
    if not np.any(valid):
        return color, valid

    # Фиксированный диапазон для нормализации
    lo = float(disp_min)
    hi = float(disp_max)
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
    color = cv2.applyColorMap(u8, colormaps[cmap_idx % len(colormaps)])
    return color, valid


def overlay_depth(base_bgr: np.ndarray, depth_color: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    out = base_bgr.copy()
    if not np.any(valid):
        return out
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, depth_color, alpha, 0.0)
    out = np.where(valid[:, :, None], blended, base_bgr)
    return out


def load_sgbm_params_from_json() -> Dict[str, int] | None:
    if not SGBM_JSON.exists():
        return None
    try:
        with SGBM_JSON.open(encoding="utf-8") as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        return None


def saved_sgbm_to_raw(saved: Dict[str, int]) -> Dict[str, int]:
    num_disp = saved.get("NUM_DISPARITIES", 640)
    block = saved.get("BLOCK_SIZE", 9)
    if block % 2 == 0:
        block = max(3, block - 1)
    median_k = saved.get("DISPARITY_MEDIAN_SIZE", 3)
    if median_k > 0 and median_k % 2 == 0:
        median_k = max(0, median_k - 1)
    return {
        "min_disp": max(0, min(64, saved.get("MIN_DISPARITY", 0))),
        "num_disp_x16": max(1, min(128, num_disp // 16)),
        "block_size": max(3, min(21, block)),
        "uniqueness": max(0, min(50, saved.get("UNIQUENESS_RATIO", 12))),
        "speckle_window": max(0, min(300, saved.get("SPECKLE_WINDOW_SIZE", 120))),
        "speckle_range": max(0, min(64, saved.get("SPECKLE_RANGE", 8))),
        "disp12_diff": max(0, min(25, saved.get("DISP12_MAX_DIFF", 2))),
        "pre_filter_cap": max(1, min(63, saved.get("PRE_FILTER_CAP", 63))),
        "median_k": max(0, min(9, median_k)),
        "alpha": 55,
        "color_map": 1,
        "preview_pct": max(20, min(100, saved.get("PREVIEW_PCT", 55))),
        "use_3way": 1 if saved.get("USE_3WAY", 0) else 0,
        "backend": 0,
    }


def save_sgbm_json(params: Dict[str, int]) -> None:
    payload = {k: v for k, v in params.items() if k != "ALPHA"}
    SGBM_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SGBM_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"SGBM параметры сохранены в {SGBM_JSON}")


def rectify_frames(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    maps: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = frame_left.shape[:2]
    calib_w, calib_h = size[0], size[1]
    if (w, h) != (calib_w, calib_h):
        frame_left = cv2.resize(frame_left, (calib_w, calib_h))
        frame_right = cv2.resize(frame_right, (calib_w, calib_h))
    rect_l_bgr = cv2.remap(frame_left, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
    rect_r_bgr = cv2.remap(frame_right, maps[1][0], maps[1][1], cv2.INTER_LINEAR)
    rect_l_gray = cv2.cvtColor(rect_l_bgr, cv2.COLOR_BGR2GRAY)
    rect_r_gray = cv2.cvtColor(rect_r_bgr, cv2.COLOR_BGR2GRAY)
    return rect_l_bgr, rect_r_bgr, rect_l_gray, rect_r_gray


def build_depth_preview(
    rect_l_bgr: np.ndarray,
    rect_l_gray: np.ndarray,
    rect_r_gray: np.ndarray,
    raw: Dict[str, int],
    cuda_ok: bool,
    force_cpu: bool = False,
) -> Dict[str, np.ndarray | Dict[str, int] | str]:
    params = to_sgbm_params(raw)
    preview_scale = params["PREVIEW_PCT"] / 100.0
    if preview_scale < 0.999:
        rect_l_gray_w = cv2.resize(rect_l_gray, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
        rect_r_gray_w = cv2.resize(rect_r_gray, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
        rect_l_bgr_w = cv2.resize(rect_l_bgr, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    else:
        rect_l_gray_w = rect_l_gray
        rect_r_gray_w = rect_r_gray
        rect_l_bgr_w = rect_l_bgr

    use_cuda = raw["backend"] == 1 and cuda_ok and not force_cpu
    backend_name = "CPU-SGBM"
    try:
        if use_cuda:
            disparity = compute_disparity_cuda_bm(rect_l_gray_w, rect_r_gray_w, params)
            backend_name = "CUDA-BM"
        else:
            disparity = compute_disparity(rect_l_gray_w, rect_r_gray_w, params)
    except Exception:
        disparity = compute_disparity(rect_l_gray_w, rect_r_gray_w, params)
        backend_name = "CPU-SGBM(fallback)"

    disp_min = params["MIN_DISPARITY"]
    disp_max = params["MIN_DISPARITY"] + params["NUM_DISPARITIES"]
    depth_color, valid = disparity_to_color(disparity, raw["color_map"], disp_min, disp_max)
    overlay = overlay_depth(rect_l_bgr_w, depth_color, valid, params["ALPHA"])
    return {
        "params": params,
        "disparity": disparity,
        "valid": valid,
        "depth_color": depth_color,
        "overlay": overlay,
        "backend_name": backend_name,
        "preview_left_bgr": rect_l_bgr_w,
        "preview_left_gray": rect_l_gray_w,
    }


def make_square_rect(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int] | None:
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return None

    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    max_side_x = (w - 1 - x0) if sx > 0 else x0
    max_side_y = (h - 1 - y0) if sy > 0 else y0
    side = int(min(max(abs(dx), abs(dy)), max_side_x, max_side_y))
    if side < 2:
        return None

    x2 = x0 + sx * side
    y2 = y0 + sy * side
    x = min(x0, x2)
    y = min(y0, y2)
    return x, y, side, side


def rect_to_norm(rect: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
    x, y, rw, rh = rect
    return x / w, y / h, (x + rw) / w, (y + rh) / h


def norm_to_rect(roi_norm: Tuple[float, float, float, float] | None, w: int, h: int) -> Tuple[int, int, int, int] | None:
    if roi_norm is None or w <= 0 or h <= 0:
        return None
    x0 = int(np.clip(round(roi_norm[0] * w), 0, max(0, w - 1)))
    y0 = int(np.clip(round(roi_norm[1] * h), 0, max(0, h - 1)))
    x1 = int(np.clip(round(roi_norm[2] * w), x0 + 1, w))
    y1 = int(np.clip(round(roi_norm[3] * h), y0 + 1, h))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def score_roi(
    disparity: np.ndarray,
    left_gray: np.ndarray,
    roi_rect: Tuple[int, int, int, int] | None,
) -> Dict[str, float | int]:
    """
    Метрика качества ROI.

    Недостаточно просто считать valid-пиксели: это часто даёт «кашу» из точек.
    Поэтому итоговый score учитывает:
    - сколько valid-точек попало в ROI;
    - насколько они образуют одну плотную связанную область;
    - насколько глубина покрывает текстурные участки объекта;
    - сколько дыр остаётся после простого морфологического закрытия.
    """
    if roi_rect is None:
        roi_disp = disparity
        roi_gray = left_gray
    else:
        x, y, w, h = roi_rect
        roi_disp = disparity[y:y + h, x:x + w]
        roi_gray = left_gray[y:y + h, x:x + w]

    valid = roi_disp > 0
    total = int(valid.size)
    count = int(np.count_nonzero(valid))
    valid_ratio = count / total if total > 0 else 0.0

    if total == 0:
        return {
            "count": 0,
            "total": 0,
            "ratio": 0.0,
            "largest_component_ratio": 0.0,
            "texture_valid_ratio": 0.0,
            "hole_ratio": 1.0,
            "score": 0.0,
        }

    valid_u8 = valid.astype(np.uint8)
    largest_component_ratio = 0.0
    if count > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_u8, 8)
        if num_labels > 1:
            largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
            largest_component_ratio = largest_area / total

    grad_x = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    if grad_mag.size > 0:
        grad_threshold = float(np.percentile(grad_mag, 65))
        textured = grad_mag >= grad_threshold
    else:
        textured = np.zeros_like(valid, dtype=bool)
    textured_total = int(np.count_nonzero(textured))
    if textured_total > 0:
        texture_valid_ratio = float(np.count_nonzero(valid & textured)) / textured_total
    else:
        texture_valid_ratio = valid_ratio

    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel)
    hole_pixels = int(np.count_nonzero((closed > 0) & (~valid)))
    hole_ratio = hole_pixels / total

    mean_intensity = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0
    p90_intensity = float(np.percentile(roi_gray, 90)) if roi_gray.size > 0 else 0.0
    dark_penalty = 0.0
    if mean_intensity < 18.0:
        dark_penalty += (18.0 - mean_intensity) * 25.0
    if p90_intensity < 45.0:
        dark_penalty += (45.0 - p90_intensity) * 8.0

    score = (
        valid_ratio * 1000.0
        + largest_component_ratio * 700.0
        + texture_valid_ratio * 500.0
        - hole_ratio * 350.0
        - dark_penalty
    )

    return {
        "count": count,
        "total": total,
        "ratio": valid_ratio,
        "largest_component_ratio": largest_component_ratio,
        "texture_valid_ratio": texture_valid_ratio,
        "hole_ratio": hole_ratio,
        "mean_intensity": mean_intensity,
        "p90_intensity": p90_intensity,
        "dark_penalty": dark_penalty,
        "score": score,
    }


def draw_roi(
    image: np.ndarray,
    roi_rect: Tuple[int, int, int, int] | None,
    score_text: str,
    status_text: str,
) -> np.ndarray:
    out = image.copy()
    if roi_rect is not None:
        x, y, w, h = roi_rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, "ROI", (x + 4, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, score_text, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, status_text, (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1, cv2.LINE_AA)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Калибровка камеры (одни параметры L=R) + карта глубины")
    parser.add_argument("--left-cam", type=int, default=CAM_LEFT_INDEX, help="Индекс левой камеры")
    parser.add_argument("--right-cam", type=int, default=CAM_RIGHT_INDEX, help="Индекс правой камеры")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Высота кадра")
    args = parser.parse_args()

    cap_left = cv2.VideoCapture(args.left_cam)
    cap_right = cv2.VideoCapture(args.right_cam)
    if not cap_left.isOpened():
        raise RuntimeError(f"Не удалось открыть левую камеру (индекс {args.left_cam})")
    if not cap_right.isOpened():
        raise RuntimeError(f"Не удалось открыть правую камеру (индекс {args.right_cam})")

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Единые параметры камеры (одни для обеих)
    saved_cam = load_camera_params_json()
    if saved_cam:
        exp = np.clip(float(saved_cam.get("exposure", -6.0)), EXPOSURE_MIN, EXPOSURE_MAX)
        gain = int(np.clip(saved_cam.get("gain", 0), GAIN_MIN, GAIN_MAX))
        bri = int(np.clip(saved_cam.get("brightness", 50), 0, PROP_RANGE))
        con = int(np.clip(saved_cam.get("contrast", 50), 0, PROP_RANGE))
        gam = int(np.clip(saved_cam.get("gamma", 50), 0, PROP_RANGE))
        print(f"Параметры камеры загружены из {CAMERA_PARAMS_FILE} (одинаковые L=R)")
    else:
        try:
            exp = np.clip(float(cap_left.get(cv2.CAP_PROP_EXPOSURE)), EXPOSURE_MIN, EXPOSURE_MAX)
        except Exception:
            exp = -6.0
        try:
            gain = int(np.clip(cap_left.get(cv2.CAP_PROP_GAIN), GAIN_MIN, GAIN_MAX))
        except Exception:
            gain = 0
        bri = 50
        con = 50
        gam = 50

    def apply_camera_params(e: float, g: int, br: int, co: int, ga: int) -> None:
        for cap in (cap_left, cap_right):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, e)
            cap.set(cv2.CAP_PROP_GAIN, g)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, br)
            cap.set(cv2.CAP_PROP_CONTRAST, co)
            cap.set(cv2.CAP_PROP_GAMMA, ga)

    apply_camera_params(exp, gain, bri, con, gam)

    def exposure_from_slider(v: int) -> float:
        return EXPOSURE_MIN + (v / TRACKBAR_STEPS) * (EXPOSURE_MAX - EXPOSURE_MIN)

    def slider_from_exposure(e: float) -> int:
        return int(np.clip((e - EXPOSURE_MIN) / (EXPOSURE_MAX - EXPOSURE_MIN) * TRACKBAR_STEPS, 0, TRACKBAR_STEPS))

    # Окно камеры
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM, 620, 700)

    def set_exp(v):
        nonlocal exp
        exp = exposure_from_slider(v)
        apply_camera_params(exp, gain, bri, con, gam)

    def set_gain(v):
        nonlocal gain
        gain = int(np.clip(v, GAIN_MIN, GAIN_MAX))
        apply_camera_params(exp, gain, bri, con, gam)

    def set_bri(v):
        nonlocal bri
        bri = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params(exp, gain, bri, con, gam)

    def set_con(v):
        nonlocal con
        con = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params(exp, gain, bri, con, gam)

    def set_gam(v):
        nonlocal gam
        gam = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params(exp, gain, bri, con, gam)

    cv2.createTrackbar("Exp", WIN_CAM, slider_from_exposure(exp), TRACKBAR_STEPS, set_exp)
    cv2.createTrackbar("Gain", WIN_CAM, gain, GAIN_TRACKBAR_MAX, set_gain)
    cv2.createTrackbar("Bri", WIN_CAM, bri, PROP_RANGE, set_bri)
    cv2.createTrackbar("Con", WIN_CAM, con, PROP_RANGE, set_con)
    cv2.createTrackbar("Gam", WIN_CAM, gam, PROP_RANGE, set_gam)
    cv2.createTrackbar("Save Cam", WIN_CAM, 0, 1, lambda _: None)
    for tb_name in CAM_LOCK_TB.values():
        cv2.createTrackbar(tb_name, WIN_CAM, 0, 1, lambda _: None)

    # Калибровка и карта глубины
    calib = load_stereo_calib(CALIBRATION_FILE)
    has_depth = calib is not None
    if calib:
        maps = build_rectify_maps(calib)
        size = calib["image_size"]
    else:
        maps = (None, None)
        size = (args.width, args.height)
        print("Файл калибровки не найден — режим глубины недоступен.")

    cuda_ok, cuda_info = detect_cuda_support()
    saved_sgbm = load_sgbm_params_from_json() if has_depth else None
    initial_raw = saved_sgbm_to_raw(saved_sgbm) if saved_sgbm else DEFAULT_RAW.copy()

    # Окно управления (SGBM) — создаём всегда, чтобы при появлении калибровки не перезапускать
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 620, 860)

    def noop(_):
        pass

    _max_vals = {
        "num_disp_x16": 128, "min_disp": 64, "block_size": 21, "uniqueness": 50,
        "speckle_window": 300, "speckle_range": 64, "disp12_diff": 25, "pre_filter_cap": 63,
        "median_k": 9, "alpha": 100, "color_map": 5, "preview_pct": 100, "use_3way": 1, "backend": 1,
    }
    for key, default in DEFAULT_RAW.items():
        val = initial_raw.get(key, default)
        mx = _max_vals.get(key, 100)
        cv2.createTrackbar(TB[key], WIN_CTRL, val, mx, noop)
    for tb_name in SGBM_LOCK_TB.values():
        cv2.createTrackbar(tb_name, WIN_CTRL, 0, 1, noop)

    if has_depth:
        cv2.namedWindow(WIN_DEPTH_OVERLAY, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_DEPTH_DISP, cv2.WINDOW_NORMAL)

    last_params: Dict[str, int] | None = None
    state: Dict[str, object] = {
        "roi_norm": None,
        "drag_start": None,
        "dragging": False,
        "last_depth_shape": None,
        "frozen_mode": False,
        "frozen_rectified": None,
        "last_live_rectified": None,
        "last_auto_score": None,
        "last_auto_label": "idle",
        "best_camera": None,
        "best_sgbm_raw": None,
    }

    def get_camera_values() -> Dict[str, float | int]:
        return {"exp": float(exp), "gain": int(gain), "bri": int(bri), "con": int(con), "gam": int(gam)}

    def set_camera_values(values: Dict[str, float | int], sync_trackbars: bool = True) -> None:
        nonlocal exp, gain, bri, con, gam
        exp = float(values["exp"])
        gain = int(values["gain"])
        bri = int(values["bri"])
        con = int(values["con"])
        gam = int(values["gam"])
        apply_camera_params(exp, gain, bri, con, gam)
        if sync_trackbars:
            cv2.setTrackbarPos("Exp", WIN_CAM, slider_from_exposure(exp))
            cv2.setTrackbarPos("Gain", WIN_CAM, gain)
            cv2.setTrackbarPos("Bri", WIN_CAM, bri)
            cv2.setTrackbarPos("Con", WIN_CAM, con)
            cv2.setTrackbarPos("Gam", WIN_CAM, gam)

    def get_camera_locks() -> Dict[str, bool]:
        return {key: cv2.getTrackbarPos(tb_name, WIN_CAM) == 1 for key, tb_name in CAM_LOCK_TB.items()}

    def get_sgbm_locks() -> Dict[str, bool]:
        return {key: cv2.getTrackbarPos(tb_name, WIN_CTRL) == 1 for key, tb_name in SGBM_LOCK_TB.items()}

    def set_sgbm_raw_values(raw_values: Dict[str, int]) -> None:
        for key, tb_name in TB.items():
            if key in raw_values:
                cv2.setTrackbarPos(tb_name, WIN_CTRL, int(raw_values[key]))

    def capture_live_rectified() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ok_l, live_left = cap_left.read()
        ok_r, live_right = cap_right.read()
        if not ok_l or not ok_r:
            raise RuntimeError("Не удалось считать кадры с камер для оценки.")
        return rectify_frames(live_left, live_right, maps, size)

    def current_eval_roi(shape: Tuple[int, int]) -> Tuple[int, int, int, int] | None:
        roi_norm = state["roi_norm"]
        return norm_to_rect(roi_norm, shape[1], shape[0]) if roi_norm is not None else None

    def evaluate_score_on_rectified(
        rectified: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        raw_values: Dict[str, int],
        force_cpu: bool = True,
    ) -> Tuple[float, Dict[str, np.ndarray | Dict[str, int] | str], Tuple[int, int, int, int] | None, Dict[str, float | int]]:
        rect_l_bgr, _rect_r_bgr, rect_l_gray, rect_r_gray = rectified
        preview = build_depth_preview(rect_l_bgr, rect_l_gray, rect_r_gray, raw_values, cuda_ok, force_cpu=force_cpu)
        disparity = preview["disparity"]
        roi_rect = current_eval_roi(disparity.shape[:2])
        metrics = score_roi(disparity, preview["preview_left_gray"], roi_rect)
        return float(metrics["score"]), preview, roi_rect, metrics

    def evaluate_camera_candidate(
        camera_values: Dict[str, float | int],
        sgbm_raw_values: Dict[str, int],
    ) -> Tuple[float, Dict[str, np.ndarray | Dict[str, int] | str], Tuple[int, int, int, int] | None, Dict[str, float | int]]:
        set_camera_values(camera_values, sync_trackbars=False)
        time.sleep(0.15)
        scores: List[float] = []
        last_preview = None
        last_roi_rect = None
        last_stats: Dict[str, float | int] = {"count": 0, "total": 0, "ratio": 0.0, "score": 0.0}
        for _ in range(3):
            rectified = capture_live_rectified()
            score, preview, roi_rect, stats = evaluate_score_on_rectified(rectified, sgbm_raw_values, force_cpu=True)
            scores.append(score)
            last_preview = preview
            last_roi_rect = roi_rect
            last_stats = stats
            time.sleep(0.04)
        assert last_preview is not None
        return float(np.mean(scores)), last_preview, last_roi_rect, last_stats

    def evaluate_sgbm_candidate(
        raw_values: Dict[str, int],
    ) -> Tuple[float, Dict[str, np.ndarray | Dict[str, int] | str], Tuple[int, int, int, int] | None, Dict[str, float | int]]:
        frozen_rectified = state["frozen_rectified"]
        if state["frozen_mode"] and frozen_rectified is not None:
            return evaluate_score_on_rectified(frozen_rectified, raw_values, force_cpu=True)

        scores: List[float] = []
        last_preview = None
        last_roi_rect = None
        last_stats: Dict[str, float | int] = {"count": 0, "total": 0, "ratio": 0.0, "score": 0.0}
        for _ in range(2):
            rectified = capture_live_rectified()
            score, preview, roi_rect, stats = evaluate_score_on_rectified(rectified, raw_values, force_cpu=True)
            scores.append(score)
            last_preview = preview
            last_roi_rect = roi_rect
            last_stats = stats
        assert last_preview is not None
        return float(np.mean(scores)), last_preview, last_roi_rect, last_stats

    def build_camera_candidates(name: str, current_value: float | int, stage: str = "coarse") -> List[float | int]:
        if name == "exp":
            center = slider_from_exposure(float(current_value))
            deltas = (-30, -20, -10, -5, 0, 5, 10, 20, 30) if stage == "coarse" else (-4, -2, -1, 0, 1, 2, 4)
            positions = [center + d for d in deltas]
            positions = sorted(set(int(np.clip(v, 0, TRACKBAR_STEPS)) for v in positions))
            return [float(exposure_from_slider(v)) for v in positions]

        max_val = GAIN_TRACKBAR_MAX if name == "gain" else PROP_RANGE
        center = int(current_value)
        deltas = (-30, -20, -10, -5, 0, 5, 10, 20, 30) if stage == "coarse" else (-4, -2, -1, 0, 1, 2, 4)
        positions = [center + d for d in deltas]
        return sorted(set(int(np.clip(v, 0, max_val)) for v in positions))

    def build_sgbm_candidates(name: str, current_value: int, stage: str = "coarse") -> List[int]:
        if name == "min_disp":
            deltas = (-16, -8, -4, 0, 4, 8, 16) if stage == "coarse" else (-3, -2, -1, 0, 1, 2, 3)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 0, 64)) for v in vals))
        if name == "num_disp_x16":
            deltas = (-16, -8, -4, 0, 4, 8, 16) if stage == "coarse" else (-3, -2, -1, 0, 1, 2, 3)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 1, 128)) for v in vals))
        if name == "block_size":
            deltas = (-6, -4, -2, 0, 2, 4, 6) if stage == "coarse" else (-2, -1, 0, 1, 2)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 3, 21)) for v in vals))
        if name == "uniqueness":
            deltas = (-20, -10, -5, 0, 5, 10, 20) if stage == "coarse" else (-3, -2, -1, 0, 1, 2, 3)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 0, 50)) for v in vals))
        if name == "speckle_window":
            deltas = (-120, -60, -30, 0, 30, 60, 120) if stage == "coarse" else (-20, -10, -5, 0, 5, 10, 20)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 0, 300)) for v in vals))
        if name == "speckle_range":
            deltas = (-16, -8, -4, 0, 4, 8, 16) if stage == "coarse" else (-3, -2, -1, 0, 1, 2, 3)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 0, 64)) for v in vals))
        if name == "disp12_diff":
            deltas = (-8, -4, -2, 0, 2, 4, 8) if stage == "coarse" else (-2, -1, 0, 1, 2)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 0, 25)) for v in vals))
        if name == "pre_filter_cap":
            deltas = (-20, -10, -5, 0, 5, 10, 20) if stage == "coarse" else (-3, -2, -1, 0, 1, 2, 3)
            vals = [current_value + d for d in deltas]
            return sorted(set(int(np.clip(v, 1, 63)) for v in vals))
        if name == "median_k":
            return [0, 3, 5, 7, 9] if stage == "coarse" else [max(0, current_value - 2), current_value, min(9, current_value + 2)]
        if name == "use_3way":
            return [0, 1]
        return [current_value]

    def freeze_current_rectified() -> None:
        live_rectified = state["last_live_rectified"]
        if live_rectified is None:
            print("Freeze пока недоступен: ещё нет рассчитанной rectified-пары.")
            return
        rect_l_bgr, rect_r_bgr, rect_l_gray, rect_r_gray = live_rectified
        state["frozen_rectified"] = (
            rect_l_bgr.copy(),
            rect_r_bgr.copy(),
            rect_l_gray.copy(),
            rect_r_gray.copy(),
        )
        state["frozen_mode"] = True
        print("Freeze включён: режим глубины использует текущую rectified-пару.")

    def toggle_freeze() -> None:
        if state["frozen_mode"]:
            state["frozen_mode"] = False
            state["frozen_rectified"] = None
            print("Freeze выключен: снова live-режим.")
        else:
            freeze_current_rectified()

    def auto_tune() -> None:
        if not has_depth:
            print("Автоподбор недоступен: нет stereo_calib.npz.")
            return
        if state["roi_norm"] is None:
            print("Сначала выдели квадрат ROI мышкой в окне глубины.")
            return

        state["last_auto_label"] = "camera-stage"
        cam_values = get_camera_values()
        sgbm_raw_values = get_slider_params_sgbm()
        cam_locks = get_camera_locks()
        sgbm_locks = get_sgbm_locks()

        best_score, best_preview, best_roi_rect, best_stats = evaluate_camera_candidate(cam_values.copy(), sgbm_raw_values.copy())
        state["last_auto_score"] = best_score
        state["best_camera"] = cam_values.copy()
        print(
            f"[auto] стартовый score={best_score:.1f} "
            f"valid={best_stats['count']}/{best_stats['total']} "
            f"lcc={best_stats['largest_component_ratio']:.2f} "
            f"tex={best_stats['texture_valid_ratio']:.2f} "
            f"holes={best_stats['hole_ratio']:.2f}"
        )

        if state["frozen_mode"]:
            print("[auto] camera-stage всегда оценивается по live-кадрам, даже если включён freeze.")

        for name in ("exp", "gain", "bri", "con", "gam"):
            if cam_locks[name]:
                continue
            current_best_val = cam_values[name]
            local_best_score = best_score
            print(f"[auto][camera] подбор {name} ...")
            for candidate in build_camera_candidates(name, current_best_val):
                trial = cam_values.copy()
                trial[name] = candidate
                score, preview, roi_rect, stats = evaluate_camera_candidate(trial, sgbm_raw_values.copy())
                print(
                    f"  {name}={candidate} -> score={score:.1f} "
                    f"valid={stats['count']}/{stats['total']} "
                    f"lcc={stats['largest_component_ratio']:.2f} "
                    f"tex={stats['texture_valid_ratio']:.2f} "
                    f"holes={stats['hole_ratio']:.2f}"
                )
                if score > local_best_score:
                    local_best_score = score
                    current_best_val = candidate
                    best_preview = preview
                    best_roi_rect = roi_rect
                    best_stats = stats
            cam_values[name] = current_best_val
            set_camera_values(cam_values, sync_trackbars=True)
            best_score = local_best_score
            state["last_auto_score"] = best_score
            state["best_camera"] = cam_values.copy()
            print(f"[auto][camera] лучший {name}={current_best_val}, score={best_score:.1f}")

        if state["frozen_mode"]:
            freeze_current_rectified()

        state["last_auto_label"] = "sgbm-stage"
        best_raw = get_slider_params_sgbm()
        best_sgbm_score, best_preview, best_roi_rect, best_stats = evaluate_sgbm_candidate(best_raw.copy())
        best_score = max(best_score, best_sgbm_score)
        print(
            f"[auto][sgbm] стартовый score={best_sgbm_score:.1f} "
            f"valid={best_stats['count']}/{best_stats['total']} "
            f"lcc={best_stats['largest_component_ratio']:.2f} "
            f"tex={best_stats['texture_valid_ratio']:.2f} "
            f"holes={best_stats['hole_ratio']:.2f}"
        )

        for name in ("min_disp", "num_disp_x16", "block_size", "uniqueness", "speckle_window", "speckle_range", "disp12_diff", "pre_filter_cap", "median_k", "use_3way"):
            if sgbm_locks[name]:
                continue
            current_best_val = int(best_raw[name])
            local_best_score = best_sgbm_score
            print(f"[auto][sgbm] подбор {name} ...")
            for candidate in build_sgbm_candidates(name, current_best_val):
                trial = best_raw.copy()
                trial[name] = int(candidate)
                score, preview, roi_rect, stats = evaluate_sgbm_candidate(trial)
                print(
                    f"  {name}={candidate} -> score={score:.1f} "
                    f"valid={stats['count']}/{stats['total']} "
                    f"lcc={stats['largest_component_ratio']:.2f} "
                    f"tex={stats['texture_valid_ratio']:.2f} "
                    f"holes={stats['hole_ratio']:.2f}"
                )
                if score > local_best_score:
                    local_best_score = score
                    current_best_val = int(candidate)
                    best_preview = preview
                    best_roi_rect = roi_rect
                    best_stats = stats
            best_raw[name] = current_best_val
            set_sgbm_raw_values(best_raw)
            best_sgbm_score = local_best_score
            state["last_auto_score"] = best_sgbm_score
            state["best_sgbm_raw"] = best_raw.copy()
            print(f"[auto][sgbm] лучший {name}={current_best_val}, score={best_sgbm_score:.1f}")

        state["last_auto_label"] = "done"
        state["last_auto_score"] = best_sgbm_score
        state["best_camera"] = cam_values.copy()
        state["best_sgbm_raw"] = best_raw.copy()
        set_camera_values(cam_values, sync_trackbars=True)
        set_sgbm_raw_values(best_raw)

        if best_preview is not None:
            score_text = (
                f"ROI valid={best_stats['count']}/{best_stats['total']} "
                f"({best_stats['ratio'] * 100:.1f}%) "
                f"score={best_sgbm_score:.1f}"
            )
            status_text = f"mode={'FROZEN' if state['frozen_mode'] else 'LIVE'} auto=done best={best_sgbm_score:.1f}"
            overlay_show = draw_roi(best_preview["overlay"], best_roi_rect, score_text, status_text)
            disp_show = draw_roi(best_preview["depth_color"], best_roi_rect, score_text, status_text)
            cv2.imshow(WIN_DEPTH_OVERLAY, overlay_show)
            cv2.imshow(WIN_DEPTH_DISP, disp_show)
        print(f"[auto] готово. Лучший score={best_sgbm_score:.1f}")

    def on_depth_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        last_shape = state["last_depth_shape"]
        if last_shape is None:
            return
        h, w = last_shape
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = (x, y)
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            start = state["drag_start"]
            if start is None:
                return
            rect = make_square_rect(start[0], start[1], x, y, w, h)
            state["roi_norm"] = rect_to_norm(rect, w, h) if rect is not None else None
        elif event == cv2.EVENT_LBUTTONUP:
            start = state["drag_start"]
            rect = make_square_rect(start[0], start[1], x, y, w, h) if start is not None else None
            state["roi_norm"] = rect_to_norm(rect, w, h) if rect is not None else None
            state["drag_start"] = None
            state["dragging"] = False

    if has_depth:
        cv2.setMouseCallback(WIN_DEPTH_OVERLAY, on_depth_mouse)
        cv2.setMouseCallback(WIN_DEPTH_DISP, on_depth_mouse)

    print("Управление:")
    print("  Камера: слайдеры Exp/Gain/Bri/Con/Gam — одни для L и R.")
    print(f"  CUDA: {cuda_info}")
    if has_depth:
        print("  [S] — сохранить оба: camera_params.json и sgbm_tuned_params.json.")
    else:
        print("  [S] — сохранить camera_params.json.")
    print("  [H] — вывести в консоль описание всех ползунков.")
    print("  [Q] / [Esc] выход.")
    print_slider_help()

    try:
        while True:
            ok1, frame_left = cap_left.read()
            ok2, frame_right = cap_right.read()
            if not ok1 or not ok2:
                continue

            # Превью камеры (одни параметры для обеих)
            scale_l = TARGET_H / frame_left.shape[0]
            scale_r = TARGET_H / frame_right.shape[0]
            small_l = cv2.resize(frame_left, (int(frame_left.shape[1] * scale_l), TARGET_H), interpolation=cv2.INTER_AREA)
            small_r = cv2.resize(frame_right, (int(frame_right.shape[1] * scale_r), TARGET_H), interpolation=cv2.INTER_AREA)
            mean_l = np.mean(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY))
            mean_r = np.mean(cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))
            cv2.putText(small_l, f"L mean={mean_l:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(small_r, f"R mean={mean_r:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            combined = np.hstack([small_l, small_r])
            cam_lock_count = sum(1 for v in get_camera_locks().values() if v)
            cv2.putText(combined, f"Same params L=R. [T]=auto [F]=freeze [S]=save camLocks={cam_lock_count}", (10, TARGET_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imshow(WIN_CAM, combined)

            # Кнопка Save Cam — сохраняем и камеру, и SGBM (если есть)
            if cv2.getTrackbarPos("Save Cam", WIN_CAM) == 1:
                save_camera_params_json(exp, gain, bri, con, gam)
                if has_depth and last_params is not None:
                    save_sgbm_json(last_params)
                cv2.setTrackbarPos("Save Cam", WIN_CAM, 0)

            # Режим глубины
            if has_depth:
                rectified_live = rectify_frames(frame_left, frame_right, maps, size)
                state["last_live_rectified"] = rectified_live
                raw = get_slider_params_sgbm()
                eval_rectified = state["frozen_rectified"] if state["frozen_mode"] and state["frozen_rectified"] is not None else rectified_live
                preview = build_depth_preview(eval_rectified[0], eval_rectified[2], eval_rectified[3], raw, cuda_ok, force_cpu=False)
                params = preview["params"]
                disparity = preview["disparity"]
                roi_rect = current_eval_roi(disparity.shape[:2])
                metrics = score_roi(disparity, preview["preview_left_gray"], roi_rect)
                state["last_depth_shape"] = disparity.shape[:2]
                state["best_sgbm_raw"] = raw.copy()

                score_text = (
                    f"ROI valid={metrics['count']}/{metrics['total']} "
                    f"({metrics['ratio'] * 100:.1f}%) score={metrics['score']:.1f}"
                )
                mode_text = "FROZEN" if state["frozen_mode"] else "LIVE"
                auto_score = state["last_auto_score"]
                auto_label = state["last_auto_label"]
                status_text = f"mode={mode_text} backend={preview['backend_name']} auto={auto_label}"
                if auto_score is not None:
                    status_text += f" best={float(auto_score):.1f}"

                overlay = draw_roi(preview["overlay"], roi_rect, score_text, status_text)
                depth_color = draw_roi(preview["depth_color"], roi_rect, score_text, status_text)
                stats = f"minDisp={params['MIN_DISPARITY']} numDisp={params['NUM_DISPARITIES']} block={params['BLOCK_SIZE']} scale={params['PREVIEW_PCT']}%"
                cv2.putText(overlay, stats, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(depth_color, stats, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                metric_text = (
                    f"lcc={metrics['largest_component_ratio']:.2f} "
                    f"tex={metrics['texture_valid_ratio']:.2f} "
                    f"holes={metrics['hole_ratio']:.2f}"
                )
                cv2.putText(overlay, metric_text, (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
                cv2.putText(depth_color, metric_text, (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
                cv2.putText(overlay, "[Mouse] ROI  [F] freeze  [R] reset ROI  [T] auto", (8, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                cv2.putText(depth_color, "[Mouse] ROI  [F] freeze  [R] reset ROI  [T] auto", (8, depth_color.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                cv2.imshow(WIN_DEPTH_OVERLAY, overlay)
                cv2.imshow(WIN_DEPTH_DISP, depth_color)
                last_params = params

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                save_camera_params_json(exp, gain, bri, con, gam)
                if has_depth and last_params is not None:
                    save_sgbm_json(last_params)
                if has_depth and last_params is not None:
                    print("Сохранено: camera_params.json и sgbm_tuned_params.json")
                else:
                    print("Сохранено: camera_params.json")
            if key in (ord("f"), ord("F")) and has_depth:
                toggle_freeze()
            if key in (ord("h"), ord("H")):
                print_slider_help()
            if key in (ord("r"), ord("R")):
                state["roi_norm"] = None
                print("ROI сброшен.")
            if key in (ord("t"), ord("T")):
                auto_tune()

            if cv2.getWindowProperty(WIN_CAM, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    print("Готово.")


if __name__ == "__main__":
    main()
