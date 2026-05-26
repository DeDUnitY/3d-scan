"""
Stereo reconstruction for turntable: camera on orbit (36 cm XY, 3.5 cm Z), looks at origin.
Model: camera rotates around static object; frame angle from frame_id in filename.
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "reconstruction"
CALIBRATION_FILE = ROOT_DIR / "outputs" / "calibration" / "stereo_calib.npz"
SGBM_PARAMS_FILE = ROOT_DIR / "outputs" / "calibration" / "sgbm_tuned_params.json"


class Config:
    INPUT_DIR = str(ROOT_DIR / "outputs" / "recorded" / "frames")
    # True — поменять местами левый и правый кадр (если физические камеры наоборот относительно калибровки).
    SWAP_LEFT_RIGHT = False

    # Градусы на одну позицию стола (см. camera_pose_for_frame). Переопределяется из
    # capture_metadata.json рядом с кадрами, если файл есть (см. apply_capture_metadata_to_config).
    TABLE_ROTATION_STEP = 18.0
    # +1 => математически положительный (CCW), -1 => CW.
    PLATFORM_ROTATION_SIGN = 1
    CAMERA_START_ANGLE_DEG = 0.0
    ORBIT_RADIUS_XY = 32.0
    CAMERA_HEIGHT_Z = 3.5
    CAMERA_OFFSET_Y = 0.0
    CAMERA_TILT_DEG = 0.0

    MIN_DISPARITY = 0
    NUM_DISPARITIES = 16 * 25
    BLOCK_SIZE = 0
    P1 = 8 * 3 * BLOCK_SIZE**2
    P2 = 32 * 3 * BLOCK_SIZE**2
    DISP12_MAX_DIFF = 1
    UNIQUENESS_RATIO = 20
    SPECKLE_WINDOW_SIZE = 200
    SPECKLE_RANGE = 12
    PRE_FILTER_CAP = 63
    DISPARITY_MEDIAN_SIZE = 5
    DISPARITY_SCALE = 1
    # Режим сравнения стереопары для disparity:
    # "gray" - старый черно-белый режим, "color" - сравнение по BGR-каналам.
    DISPARITY_COMPARE_MODE = "gray"

    ROI_MARGIN = 70
    ROI_MARGIN_TOP = 40
    ROI_MARGIN_BOTTOM = 20

    CAMERA_MIN_DISTANCE = 0.0
    CAMERA_MAX_DISTANCE = 400.0

    CROP_RADIUS = 500.0
    Z_MIN = -200.0
    Z_MAX = 150.0

    USE_NOISE_FILTER = False
    SOR_K = 20
    SOR_STD_MULTIPLIER = 2.0

    # Фильтр по цвету: оставлять только точки определенного цвета
    USE_COLOR_FILTER = False  # Включить/выключить фильтр цвета
    TARGET_COLOR_RGB = [255, 50, 50]  # Целевой цвет в RGB (красный по умолчанию)
    COLOR_TOLERANCE = 200  # Допустимое отклонение по каждому каналу RGB (0-255)
    # Служебный цвет фона после apps/mask_recorded_frames.py; такие точки нужно удалить.
    USE_CUT_BACKGROUND_FILTER = True
    CUT_BACKGROUND_COLOR_RGB = [255, 0, 255]
    CUT_BACKGROUND_COLOR_TOLERANCE = 5

    SAVE_POINTS = True
    POINTS_FILE = str(OUTPUT_DIR / "stereo_points.npy")
    SAVE_DEBUG_DISPARITY = True
    SAVE_ALIGNMENT_BUNDLE = True
    ALIGNMENT_BUNDLE_FILE = str(OUTPUT_DIR / "stereo_alignment_bundle.npz")
    # 0 = keep all points per frame in bundle (no downsampling)
    ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME = 0

    PLOTLY_MODE = "html"
    PLOTLY_FILE = str(OUTPUT_DIR / "points_cloud.html")
    PICK_MAX_POINTS = 20000
    SHOTS_PER_POSITION = 3
    # Номера снимков (1-based) внутри позиции, которые участвуют в расчёте.
    # Пример: [3] -> использовать только 3-й снимок каждой позиции.
    # None или [] -> использовать все доступные снимки.
    SHOT_NUMBERS_TO_USE = None
    # Номер снимка (1-based), из которого брать цвет текстуры.
    # None -> брать цвет из первого обработанного depth-снимка.
    TEXTURE_SHOT_NUMBER = None
    # Номера снимков (1-based), которые участвуют в расчете глубины.
    # None или [] -> использовать все доступные снимки.
    DEPTH_SHOT_NUMBERS = None
    # Ограничение количества выбранных снимков на позицию (0 = без ограничения).
    # Пример: 1 -> взять только первый подходящий снимок из SHOT_NUMBERS_TO_USE.
    MAX_SELECTED_SHOTS_PER_POSITION = 0
    DISPARITY_OUTLIER_REL_THRESHOLD = 0.25
    PIXEL_DISTANCE_STATS_FILE = str(OUTPUT_DIR / "pixel_distance_per_position.json")


def load_sgbm_params(path: Path) -> dict | None:
    """Load SGBM parameters from JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"Warning: Could not load SGBM params from {path}: {e}")
        return None


def apply_capture_metadata_to_config(input_dir: str) -> None:
    """Подставить угол шага и число снимков на позицию из capture_metadata.json (после capture_frames_with_rotation)."""
    meta_path = Path(input_dir) / "capture_metadata.json"
    if not meta_path.exists():
        return
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not read {meta_path}: {e}")
        return
    if "degrees_per_position" in data:
        step = float(data["degrees_per_position"])
        Config.TABLE_ROTATION_STEP = step
        print(f"Capture metadata: TABLE_ROTATION_STEP={step:g}° ({meta_path.name})")
    elif "positions_count" in data:
        pc = int(data["positions_count"])
        if pc > 0:
            step = 360.0 / float(pc)
            Config.TABLE_ROTATION_STEP = step
            print(f"Capture metadata: TABLE_ROTATION_STEP={step:g}° (=360°/{pc}) ({meta_path.name})")
    sp = data.get("shots_per_position")
    if sp is not None:
        sp = int(sp)
        if sp >= 1:
            Config.SHOTS_PER_POSITION = sp
            print(f"  SHOTS_PER_POSITION={sp} (from metadata)")
    texture_shot_number = data.get("texture_shot_number")
    if texture_shot_number is not None:
        try:
            tsn = int(texture_shot_number)
            if tsn >= 1:
                Config.TEXTURE_SHOT_NUMBER = tsn
                print(f"  TEXTURE_SHOT_NUMBER={tsn} (from metadata)")
        except (TypeError, ValueError):
            pass
    depth_shot_numbers = data.get("depth_shot_numbers")
    if isinstance(depth_shot_numbers, list):
        clean = sorted({int(x) for x in depth_shot_numbers if int(x) >= 1})
        Config.DEPTH_SHOT_NUMBERS = clean or None
        if Config.DEPTH_SHOT_NUMBERS:
            print(f"  DEPTH_SHOT_NUMBERS={Config.DEPTH_SHOT_NUMBERS} (from metadata)")
    # rotation_dir from capture script: 1=CW, 0=CCW.
    # In math coordinates positive angle is CCW, so CW should be -1.
    rotation_dir = data.get("rotation_dir")
    if rotation_dir is not None:
        try:
            dir_int = int(rotation_dir)
            if dir_int in (0, 1):
                Config.PLATFORM_ROTATION_SIGN = -1 if dir_int == 1 else 1
                dir_name = "CW" if dir_int == 1 else "CCW"
                print(
                    f"  PLATFORM_ROTATION_SIGN={Config.PLATFORM_ROTATION_SIGN} "
                    f"(from rotation_dir={dir_int} {dir_name})"
                )
        except (TypeError, ValueError):
            pass


def apply_object_mask_metadata_to_config(input_dir: str) -> None:
    meta_path = Path(input_dir) / "object_mask_metadata.json"
    if not meta_path.exists():
        return
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not read {meta_path}: {e}")
        return

    fill_color = data.get("fill_color_rgb")
    if isinstance(fill_color, list) and len(fill_color) == 3:
        try:
            Config.CUT_BACKGROUND_COLOR_RGB = [int(x) for x in fill_color[:3]]
            Config.USE_CUT_BACKGROUND_FILTER = True
            print(
                "Object mask metadata: "
                f"CUT_BACKGROUND_COLOR_RGB={Config.CUT_BACKGROUND_COLOR_RGB} ({meta_path.name})"
            )
        except (TypeError, ValueError):
            pass


def load_stereo_calib(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stereo calibration not found: {p}")
    d = np.load(p, allow_pickle=True)
    out = {
        "K1": d["camera_matrix_left"],
        "D1": d["dist_coeffs_left"],
        "K2": d["camera_matrix_right"],
        "D2": d["dist_coeffs_right"],
        "R1": d["R1"],
        "R2": d["R2"],
        "P1": d["P1"],
        "P2": d["P2"],
        "Q": d["Q"],
        "image_size": tuple(int(x) for x in d["image_size"]),
    }
    return out


def extract_frame_info(name: str):
    """
    Возвращает (global_frame_id, position_id, shot_id) из имени файла.
    Поддерживает:
      - capture_p0003_s01_left.png (новый формат)
      - capture_0012_left.png / frame_0012_left.png (legacy)
    """
    m_new = re.search(r"capture_p(\d+)_s(\d+)_left\.", name)
    if m_new:
        pos_id = int(m_new.group(1))
        shot_id = int(m_new.group(2))
        global_id = pos_id * max(Config.SHOTS_PER_POSITION, 1) + shot_id
        return global_id, pos_id, shot_id
    m_old = re.search(r"(?:capture|frame)_(\d+)_left\.", name)
    if m_old:
        fid = int(m_old.group(1))
        return fid, None, None
    return None, None, None


def collect_pairs(images_dir):
    root = Path(images_dir)
    left_files = sorted(root.glob("capture_*_left.png")) + sorted(root.glob("frame_*_left.png"))
    out = []
    for left in left_files:
        right = root / left.name.replace("_left.", "_right.")
        if not right.exists():
            continue
        fid, pos_id, shot_id = extract_frame_info(left.name)
        out.append(
            {
                "left": str(left),
                "right": str(right),
                "fid": int(fid if fid is not None else len(out)),
                "pos_id": pos_id,
                "shot_id": shot_id,
            }
        )
    return out


def get_selected_shot_numbers() -> set[int] | None:
    raw = Config.SHOT_NUMBERS_TO_USE
    if not raw:
        return None
    out = {int(x) for x in raw if int(x) >= 1}
    return out or None


def get_depth_shot_numbers() -> set[int] | None:
    raw = Config.DEPTH_SHOT_NUMBERS
    if not raw:
        return None
    out = {int(x) for x in raw if int(x) >= 1}
    return out or None


def build_rectify_maps(calib):
    size = calib["image_size"]
    map1x, map1y = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_32FC1)
    return (map1x, map1y), (map2x, map2y)


def prepare_disparity_images(img_l, img_r, size, maps):
    mode = str(Config.DISPARITY_COMPARE_MODE).lower()
    if mode not in ("gray", "color"):
        raise ValueError('DISPARITY_COMPARE_MODE must be "gray" or "color".')

    if mode == "color":
        left = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR) if img_l.ndim == 2 else img_l
        right = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR) if img_r.ndim == 2 else img_r
    else:
        left = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if img_l.ndim == 3 else img_l
        right = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if img_r.ndim == 3 else img_r

    if left.shape[:2][::-1] != size:
        left = cv2.resize(left, size)
    if right.shape[:2][::-1] != size:
        right = cv2.resize(right, size)

    rl = cv2.remap(left, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
    rr = cv2.remap(right, maps[1][0], maps[1][1], cv2.INTER_LINEAR)
    return rl, rr


def compute_points_3d(img_l, img_r, calib, maps):
    size = calib["image_size"]
    rl, rr = prepare_disparity_images(img_l, img_r, size, maps)

    # Ректифицированное цветное изображение для сохранения цветов в облаке
    if img_l.shape[:2][::-1] != size:
        img_l_color = cv2.resize(img_l, size)
    else:
        img_l_color = img_l
    rl_color = cv2.remap(img_l_color, maps[0][0], maps[0][1], cv2.INTER_LINEAR)

    scale = Config.DISPARITY_SCALE if 0 < Config.DISPARITY_SCALE <= 1.0 else 1.0
    if scale < 1.0:
        h, w = rl.shape[:2]
        rl = cv2.resize(rl, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rr = cv2.resize(rr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=Config.MIN_DISPARITY,
        numDisparities=Config.NUM_DISPARITIES,
        blockSize=Config.BLOCK_SIZE,
        P1=Config.P1,
        P2=Config.P2,
        disp12MaxDiff=Config.DISP12_MAX_DIFF,
        uniquenessRatio=Config.UNIQUENESS_RATIO,
        speckleWindowSize=Config.SPECKLE_WINDOW_SIZE,
        speckleRange=Config.SPECKLE_RANGE,
        preFilterCap=Config.PRE_FILTER_CAP,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = np.clip(sgbm.compute(rl, rr).astype(np.float32), 0, None)
    disp = (cv2.resize(disp / 16.0, size, interpolation=cv2.INTER_LINEAR) * (1.0 / scale)) if scale < 1.0 else disp / 16.0
    if Config.DISPARITY_MEDIAN_SIZE >= 3:
        k = Config.DISPARITY_MEDIAN_SIZE if Config.DISPARITY_MEDIAN_SIZE % 2 else Config.DISPARITY_MEDIAN_SIZE + 1
        mask0 = disp <= 0
        disp = cv2.medianBlur(disp, k)
        disp[mask0] = 0.0

    pts = cv2.reprojectImageTo3D(disp, calib["Q"])
    bad = (pts[:, :, 2] <= 0) | (pts[:, :, 2] > 100.0)
    pts[bad] = [np.nan, np.nan, np.nan]
    return pts, disp, rl_color


def robust_mean(values, outlier_rel_threshold):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return np.nan, np.array([], dtype=bool)
    if vals.size < 3:
        return float(np.mean(vals)), np.ones(vals.size, dtype=bool)
    med = float(np.median(vals))
    scale = max(abs(med), 1e-6)
    rel_dev = np.abs(vals - med) / scale
    keep = rel_dev <= outlier_rel_threshold
    if keep.sum() < 2:
        keep[np.argmin(rel_dev)] = True
        sorted_idx = np.argsort(rel_dev)
        keep[sorted_idx[1]] = True
    return float(np.mean(vals[keep])), keep


def points_map_to_cloud_cm(points_3d, colors_bgr=None):
    flat = points_3d.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1) & (flat[:, 2] > 1e-6)
    xyz = flat[valid] * 100.0
    if colors_bgr is None:
        return xyz
    # Сопоставляем тем же валидным точкам их цвет (BGR -> RGB)
    flat_colors = colors_bgr.reshape(-1, 3)[valid]
    rgb = flat_colors[:, ::-1].astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def _normalize(v):
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-12 else None


def camera_pose_for_frame(frame_id):
    theta = np.radians(
        Config.CAMERA_START_ANGLE_DEG
        + Config.PLATFORM_ROTATION_SIGN * Config.TABLE_ROTATION_STEP * frame_id
    )
    c_block = np.array(
        [
            Config.ORBIT_RADIUS_XY * np.cos(theta),
            Config.ORBIT_RADIUS_XY * np.sin(theta) + Config.CAMERA_OFFSET_Y,
            Config.CAMERA_HEIGHT_Z,
        ],
        dtype=np.float64,
    )
    target = np.zeros(3, dtype=np.float64)
    z_axis = _normalize(target - c_block)
    if z_axis is None:
        z_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _normalize(np.cross(z_axis, up))
    if x_axis is None:
        x_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    y_up = _normalize(np.cross(x_axis, z_axis))
    y_down = -(y_up if y_up is not None else np.array([0.0, 1.0, 0.0], dtype=np.float64))
    r_cw = np.column_stack((x_axis, y_down, z_axis))

    tilt = np.radians(Config.CAMERA_TILT_DEG)
    if abs(tilt) > 1e-12:
        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]],
            dtype=np.float64,
        )
        r_cw = r_cw @ rx

    return r_cw, c_block


def cam_to_world(points_cam_cm, frame_id):
    r_cw, c_block = camera_pose_for_frame(frame_id)
    return points_cam_cm @ r_cw.T + c_block.reshape(1, 3)


def downsample(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[idx]


def filter_by_color(points_with_colors, target_rgb, tolerance):
    """
    Фильтрует точки по цвету.
    
    Args:
        points_with_colors: массив формы (N, 6) где первые 3 колонки - XYZ, последние 3 - RGB
        target_rgb: целевой цвет [R, G, B] в диапазоне 0-255
        tolerance: допустимое отклонение по каждому каналу
    
    Returns:
        Отфильтрованный массив точек
    """
    if points_with_colors.shape[1] < 6:
        # Нет цветов, возвращаем как есть
        return points_with_colors
    
    xyz = points_with_colors[:, :3]
    rgb = points_with_colors[:, 3:6]
    
    # Проверяем, попадает ли цвет в диапазон для каждого канала
    target = np.array(target_rgb, dtype=np.float32)
    lower = np.maximum(0, target - tolerance)
    upper = np.minimum(255, target + tolerance)
    
    # Маска: точка проходит фильтр если все три канала RGB попадают в диапазон
    mask = (
        (rgb[:, 0] >= lower[0]) & (rgb[:, 0] <= upper[0]) &
        (rgb[:, 1] >= lower[1]) & (rgb[:, 1] <= upper[1]) &
        (rgb[:, 2] >= lower[2]) & (rgb[:, 2] <= upper[2])
    )
    
    filtered = points_with_colors[mask]
    return filtered


def filter_out_color(points_with_colors, background_rgb, tolerance):
    if points_with_colors.shape[1] < 6:
        return points_with_colors

    rgb = points_with_colors[:, 3:6]
    target = np.array(background_rgb, dtype=np.float32)
    diff = np.abs(rgb.astype(np.float32) - target.reshape(1, 3))
    background_mask = np.all(diff <= float(tolerance), axis=1)
    return points_with_colors[~background_mask]


def save_bundle(all_clouds_cam):
    if not Config.SAVE_ALIGNMENT_BUNDLE:
        return
    frames = []
    clouds = []
    colors_list = []
    for fid, cloud in all_clouds_cam:
        frames.append(int(fid))
        cloud_arr = np.asarray(cloud, dtype=np.float32)
        if cloud_arr.shape[1] >= 6:
            # Есть цвета: отделяем XYZ от RGB
            xyz = cloud_arr[:, :3]
            rgb = cloud_arr[:, 3:6]
            cloud_ds = downsample(xyz, Config.ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME)
            # Даунсэмплинг цветов по тем же индексам
            if Config.ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME > 0 and len(cloud_arr) > Config.ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME:
                idx = np.linspace(0, len(cloud_arr) - 1, Config.ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME, dtype=int)
                rgb_ds = rgb[idx]
            else:
                rgb_ds = rgb
            clouds.append(cloud_ds)
            colors_list.append(np.clip(rgb_ds, 0, 255).astype(np.uint8))
        else:
            # Только координаты
            clouds.append(downsample(cloud_arr, Config.ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME))
            colors_list.append(None)
    payload = {
        "frame_indices": np.asarray(frames, dtype=np.int32),
        "clouds_cam": np.array(clouds, dtype=object),
        "rotation_step_deg": np.float32(Config.TABLE_ROTATION_STEP),
        "extra_frame_rot_z_deg": np.float32(0.0),
        "platform_rotation_sign": np.int32(Config.PLATFORM_ROTATION_SIGN),
        "orbit_radius": np.float32(Config.ORBIT_RADIUS_XY),
        "camera_height": np.float32(Config.CAMERA_HEIGHT_Z),
        "camera_tilt_deg": np.float32(Config.CAMERA_TILT_DEG),
        "camera_offset_y": np.float32(Config.CAMERA_OFFSET_Y),
        "camera_min_distance": np.float32(Config.CAMERA_MIN_DISTANCE),
        "camera_max_distance": np.float32(Config.CAMERA_MAX_DISTANCE),
    }
    # Сохраняем цвета отдельно, если они есть
    has_colors = any(c is not None for c in colors_list)
    if has_colors:
        payload["colors_rgb"] = np.array(colors_list, dtype=object)
    out = Path(Config.ALIGNMENT_BUNDLE_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **payload)
    print(f"Saved bundle: {out}" + (" (with colors)" if has_colors else ""))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tuned SGBM parameters from JSON if available
    tuned_params = load_sgbm_params(SGBM_PARAMS_FILE)
    if tuned_params:
        print(f"Loading SGBM parameters from {SGBM_PARAMS_FILE}")
        # Override Config values with tuned parameters
        for key, value in tuned_params.items():
            if hasattr(Config, key):
                setattr(Config, key, value)
                print(f"  {key} = {value}")
        # Recalculate P1 and P2 if BLOCK_SIZE was changed
        if "BLOCK_SIZE" in tuned_params:
            Config.P1 = 8 * 3 * Config.BLOCK_SIZE**2
            Config.P2 = 32 * 3 * Config.BLOCK_SIZE**2
            print(f"  P1 = {Config.P1} (recalculated)")
            print(f"  P2 = {Config.P2} (recalculated)")
    else:
        print(f"Using default SGBM parameters (tuned params not found at {SGBM_PARAMS_FILE})")
    print(f"Disparity compare mode: {Config.DISPARITY_COMPARE_MODE}")

    apply_capture_metadata_to_config(Config.INPUT_DIR)
    apply_object_mask_metadata_to_config(Config.INPUT_DIR)

    calib = load_stereo_calib(CALIBRATION_FILE)
    maps = build_rectify_maps(calib)
    pairs = collect_pairs(Config.INPUT_DIR)
    if not pairs:
        raise RuntimeError(f"No stereo pairs in {Config.INPUT_DIR}")

    n = len(pairs)
    step = Config.TABLE_ROTATION_STEP
    pos_ids = {int(item["pos_id"]) for item in pairs if item["pos_id"] is not None}
    n_positions = len(pos_ids)
    print(f"Pairs: {n} | orbit_xy={Config.ORBIT_RADIUS_XY} cm, z={Config.CAMERA_HEIGHT_Z} cm")
    if n_positions > 0:
        print(
            f"  TABLE_ROTATION_STEP={step}° per turntable position (capture_p####). "
            f"Distinct positions: {n_positions}; ~{360 / step:.1f} steps for 360°. "
            f"{n} stereo pairs > {n_positions} positions because of multiple shots per position."
        )
    else:
        print(f"  TABLE_ROTATION_STEP={step}° per turntable index (~{360 / step:.1f} steps for 360°)")
    if Config.SWAP_LEFT_RIGHT:
        print("  SWAP_LEFT_RIGHT=True (L/R images swapped before stereo)")

    selected_shot_numbers = get_selected_shot_numbers()
    if selected_shot_numbers:
        print(f"  Using shot numbers per position: {sorted(selected_shot_numbers)}")
    if Config.MAX_SELECTED_SHOTS_PER_POSITION > 0:
        print(f"  Max selected shots per position: {Config.MAX_SELECTED_SHOTS_PER_POSITION}")
    depth_shot_numbers = get_depth_shot_numbers()
    if depth_shot_numbers:
        print(f"  Depth shot numbers per position: {sorted(depth_shot_numbers)}")
    if Config.TEXTURE_SHOT_NUMBER is not None:
        print(f"  Texture shot number: {Config.TEXTURE_SHOT_NUMBER}")

    position_groups = {}
    for item in pairs:
        fid = int(item["fid"])
        if item["pos_id"] is not None:
            pos_id = int(item["pos_id"])
            shot_id = int(item["shot_id"]) if item["shot_id"] is not None else 0
        else:
            # Legacy names (capture_XXXX_left.png / frame_XXXX_left.png):
            # each frame index is a distinct turntable position.
            # Do not derive position by SHOTS_PER_POSITION, it may be unrelated.
            pos_id = fid
            shot_id = 0
        shot_num = int(shot_id) + 1
        if selected_shot_numbers is not None and shot_num not in selected_shot_numbers:
            continue
        position_groups.setdefault(int(pos_id), []).append(
            {
                "left": item["left"],
                "right": item["right"],
                "fid": fid,
                "shot_id": int(shot_id),
            }
        )

    all_clouds_cam = []
    pixel_distance_stats = []
    cut_background_removed = 0
    sorted_positions = sorted(position_groups.items(), key=lambda x: x[0])
    for pos_i, (pos_id, group_pairs) in enumerate(sorted_positions):
        group_pairs = sorted(group_pairs, key=lambda x: (x["shot_id"], x["fid"]))
        if Config.MAX_SELECTED_SHOTS_PER_POSITION > 0:
            group_pairs = group_pairs[: Config.MAX_SELECTED_SHOTS_PER_POSITION]
        disp_maps = []
        shot_mean_disparities = []
        left_color_rect_ref = None
        processed_shots = 0
        depth_pairs = []
        texture_pair = None
        for pair in group_pairs:
            shot_num = int(pair["shot_id"]) + 1
            if Config.TEXTURE_SHOT_NUMBER is not None and shot_num == Config.TEXTURE_SHOT_NUMBER and texture_pair is None:
                texture_pair = pair
            if depth_shot_numbers is not None and shot_num not in depth_shot_numbers:
                continue
            depth_pairs.append(pair)
        if not depth_pairs:
            depth_pairs = list(group_pairs)
        if texture_pair is None and depth_pairs:
            texture_pair = depth_pairs[0]

        for pair in depth_pairs:
            pl, pr = pair["left"], pair["right"]
            left, right = cv2.imread(pl), cv2.imread(pr)
            if left is None or right is None:
                continue
            if Config.SWAP_LEFT_RIGHT:
                left, right = right, left
            points_3d_shot, disp, left_color_rect = compute_points_3d(left, right, calib, maps)
            valid = disp > 0
            if np.any(valid):
                shot_mean_disparities.append(float(np.mean(disp[valid])))
            else:
                shot_mean_disparities.append(0.0)
            disp_maps.append(disp)
            processed_shots += 1

        if texture_pair is not None:
            tex_left = cv2.imread(texture_pair["left"])
            tex_right = cv2.imread(texture_pair["right"])
            if tex_left is not None and tex_right is not None:
                if Config.SWAP_LEFT_RIGHT:
                    tex_left, tex_right = tex_right, tex_left
                _, _, left_color_rect_ref = compute_points_3d(tex_left, tex_right, calib, maps)

        if not disp_maps or left_color_rect_ref is None:
            continue

        mean_pixel_distance, keep_mask = robust_mean(
            shot_mean_disparities,
            outlier_rel_threshold=Config.DISPARITY_OUTLIER_REL_THRESHOLD,
        )
        kept_indices = np.where(keep_mask)[0]
        if kept_indices.size == 0:
            kept_indices = np.arange(len(disp_maps))
        disp_selected = np.mean([disp_maps[idx] for idx in kept_indices], axis=0).astype(np.float32)

        points_3d = cv2.reprojectImageTo3D(disp_selected, calib["Q"])
        bad = (points_3d[:, :, 2] <= 0) | (points_3d[:, :, 2] > 100.0)
        points_3d[bad] = [np.nan, np.nan, np.nan]

        pixel_distance_stats.append(
            {
                "position_id": int(pos_id),
                "shots_used": int(len(kept_indices)),
                "shots_total": int(processed_shots),
                "used_shot_numbers": [int(group_pairs[idx]["shot_id"]) + 1 for idx in kept_indices if idx < len(group_pairs)],
                "mean_pixel_distance": float(mean_pixel_distance),
                "all_shot_mean_distances": [float(x) for x in shot_mean_disparities],
            }
        )

        h, w = points_3d.shape[:2]
        top = Config.ROI_MARGIN_TOP if Config.ROI_MARGIN_TOP is not None else Config.ROI_MARGIN
        bottom = Config.ROI_MARGIN_BOTTOM if Config.ROI_MARGIN_BOTTOM is not None else Config.ROI_MARGIN
        margin = Config.ROI_MARGIN
        nan3 = np.array([np.nan, np.nan, np.nan], dtype=points_3d.dtype)
        if top > 0:
            points_3d[:top, :, :] = nan3
        if bottom > 0:
            points_3d[max(0, h - bottom) :, :, :] = nan3
        if margin > 0:
            points_3d[:, :margin, :] = nan3
            points_3d[:, w - margin :, :] = nan3

        cloud_cam = points_map_to_cloud_cm(points_3d, colors_bgr=left_color_rect_ref)
        if Config.USE_CUT_BACKGROUND_FILTER and cloud_cam.shape[1] >= 6:
            before_count = len(cloud_cam)
            cloud_cam = filter_out_color(
                cloud_cam,
                Config.CUT_BACKGROUND_COLOR_RGB,
                Config.CUT_BACKGROUND_COLOR_TOLERANCE,
            )
            cut_background_removed += before_count - len(cloud_cam)
        if len(cloud_cam) == 0:
            continue
        all_clouds_cam.append((pos_id, cloud_cam))

        if Config.SAVE_DEBUG_DISPARITY and pos_i == 0:
            vis = cv2.normalize(disp_selected, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(OUTPUT_DIR / "debug_disparity.png"), vis)

    if not all_clouds_cam:
        raise RuntimeError("No points after ROI.")

    stats_out = Path(Config.PIXEL_DISTANCE_STATS_FILE)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(pixel_distance_stats, f, ensure_ascii=False, indent=2)
    print(f"Saved pixel distance stats: {stats_out}")
    if Config.USE_CUT_BACKGROUND_FILTER:
        print(
            "Cut-background filter: "
            f"removed {cut_background_removed} points "
            f"(RGB={Config.CUT_BACKGROUND_COLOR_RGB}, tolerance={Config.CUT_BACKGROUND_COLOR_TOLERANCE})"
        )

    save_bundle(all_clouds_cam)

    # Как в single: камера на орбите, точки в мировую СК. Фильтр по расстоянию до камеры — оставляем.
    all_world = []
    for fid, cloud_cam in all_clouds_cam:
        xyz_cam = cloud_cam[:, :3]
        colors = cloud_cam[:, 3:] if cloud_cam.shape[1] > 3 else None

        d = np.linalg.norm(xyz_cam, axis=1)
        mask = np.ones(len(cloud_cam), dtype=bool)
        if Config.CAMERA_MIN_DISTANCE > 0:
            mask &= d >= Config.CAMERA_MIN_DISTANCE
        if Config.CAMERA_MAX_DISTANCE > 0:
            mask &= d <= Config.CAMERA_MAX_DISTANCE
        xyz_cam = xyz_cam[mask]
        if colors is not None:
            colors = colors[mask]
        if len(xyz_cam) == 0:
            continue
        xyz_world = cam_to_world(xyz_cam, int(fid))
        if colors is not None:
            cloud_world = np.concatenate([xyz_world, colors], axis=1)
        else:
            cloud_world = xyz_world
        all_world.append(cloud_world)

    if not all_world:
        raise RuntimeError("No points after stereo reconstruction.")
    combined = np.vstack(all_world)

    # Инверсия оси X для исправления зеркальности (как в single).
    combined[:, 0] = -combined[:, 0]

    # Геометрическая обрезка: цилиндр по радиусу и диапазону Z (как в single).
    rxy = np.linalg.norm(combined[:, :2], axis=1)
    geo_mask = (rxy < Config.CROP_RADIUS) & (combined[:, 2] > Config.Z_MIN) & (combined[:, 2] < Config.Z_MAX)
    combined = combined[geo_mask]

    # Фильтр по цвету: оставляем только точки определенного цвета
    if Config.USE_COLOR_FILTER and combined.shape[1] >= 6:
        before_count = len(combined)
        combined = filter_by_color(combined, Config.TARGET_COLOR_RGB, Config.COLOR_TOLERANCE)
        after_count = len(combined)
        print(f"Color filter: {before_count} -> {after_count} points (target RGB={Config.TARGET_COLOR_RGB}, tolerance={Config.COLOR_TOLERANCE})")

    if Config.USE_NOISE_FILTER and len(combined) > Config.SOR_K:
        try:
            from scipy.spatial import cKDTree
            xyz_only = combined[:, :3]
            tree = cKDTree(xyz_only)
            dists, _ = tree.query(xyz_only, k=Config.SOR_K + 1)
            m = dists[:, 1:].mean(axis=1)
            combined = combined[m < (m.mean() + Config.SOR_STD_MULTIPLIER * m.std())]
        except Exception:
            pass
    if Config.SAVE_POINTS:
        np.save(Config.POINTS_FILE, combined)
        print(f"Saved points: {Config.POINTS_FILE} ({len(combined)})")

    if Config.PLOTLY_MODE == "off" or len(combined) == 0:
        return

    xyz = combined[:, :3]
    colors = combined[:, 3:] if combined.shape[1] >= 6 else None
    pick_idx = None
    if Config.PICK_MAX_POINTS > 0 and len(xyz) > Config.PICK_MAX_POINTS:
        pick_idx = np.linspace(0, len(xyz) - 1, Config.PICK_MAX_POINTS, dtype=int)
        xyz_pick = xyz[pick_idx]
        colors_pick = colors[pick_idx] if colors is not None else None
    else:
        xyz_pick = xyz
        colors_pick = colors

    if colors_pick is not None:
        # Преобразуем RGB в формат 'rgb(r,g,b)' для Plotly
        colors_pick_uint8 = np.clip(colors_pick, 0, 255).astype(np.uint8)
        plot_colors = [f"rgb({r},{g},{b})" for r, g, b in colors_pick_uint8]
        marker_kwargs = dict(size=2, color=plot_colors)
    else:
        marker_kwargs = dict(size=2, color="blue")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz_pick[:, 0], y=xyz_pick[:, 1], z=xyz_pick[:, 2],
            mode="markers", marker=marker_kwargs, hoverinfo="skip",
        )
    )
    mins, maxs = combined.min(axis=0), combined.max(axis=0)
    span = max(np.max(maxs - mins), 1.0)
    c = (mins + maxs) / 2.0
    h = span / 2.0
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[c[0] - h, c[0] + h]),
            yaxis=dict(range=[c[1] - h, c[1] + h]),
            zaxis=dict(range=[c[2] - h, c[2] + h]),
        ),
        title="Stereo cloud (camera on orbit)",
    )
    fig.write_html(Config.PLOTLY_FILE, auto_open=(Config.PLOTLY_MODE == "browser"))
    print(f"Plotly: {Config.PLOTLY_FILE}")
    sys.exit(0)


if __name__ == "__main__":
    main()
