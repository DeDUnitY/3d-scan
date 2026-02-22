"""
Stereo reconstruction for turntable: camera on orbit (36 cm XY, 3.5 cm Z), looks at origin.
Model: camera rotates around static object; frame angle from frame_id in filename.
"""
# #region agent log
def _dbg_log(loc, msg, data=None):
    import time, json as _json
    p = Path(__file__).resolve().parents[2] / "debug-ca5cb8.log"
    entry = {"id": f"log_{int(time.time()*1000)}", "timestamp": int(time.time() * 1000), "location": loc, "message": msg}
    if data is not None:
        def _conv(v):
            if hasattr(v, "tolist"): return v.tolist()
            if isinstance(v, (list, tuple)) and len(v) > 10: return str(v)[:150]
            return v
        entry["data"] = {str(k): _conv(v) for k, v in data.items()}
    with open(p, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
# #endregion
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go

from object_config import (
    get_calibration_file,
    get_frames_dir,
    get_reconstruction_dir,
    get_capture_metadata_file,
    get_pose_params_file,
    get_sgbm_params_file,
    OBJECT_NAME,
)

OUTPUT_DIR = get_reconstruction_dir()
CALIBRATION_FILE = get_calibration_file()
SGBM_PARAMS_FILE = get_sgbm_params_file()


class Config:
    INPUT_DIR = str(get_frames_dir())

    # Degrees per frame (e.g. 360/10=36). Must match actual turntable.
    TABLE_ROTATION_STEP = 36.0
    CAMERA_START_ANGLE_DEG = 0.0
    ORBIT_RADIUS_XY = 32.0  # см, радиус орбиты камеры в плоскости XY
    CAMERA_HEIGHT_Z = 3.5   # см, высота камеры над столом (ось Z)
    CAMERA_OFFSET_Y = 0.0
    CAMERA_TILT_DEG = 0.0

    MIN_DISPARITY = 0
    NUM_DISPARITIES = 16 * 40
    BLOCK_SIZE = 13  # больше = глаже на плоскостях, точнее при малой текстуре
    P1 = 8 * 3 * BLOCK_SIZE**2
    P2 = 32 * 3 * BLOCK_SIZE**2
    DISP12_MAX_DIFF = 1  # строже left-right проверка → меньше ложных, но больше дыр
    UNIQUENESS_RATIO = 35  # выше = отбрасывать неоднозначные совпадения
    SPECKLE_WINDOW_SIZE = 150
    SPECKLE_RANGE = 8  # меньше = агрессивнее убирать шумовые кластеры
    PRE_FILTER_CAP = 63
    DISPARITY_MEDIAN_SIZE = 5
    DISPARITY_SCALE = 1.0  # 1.0 = полное разрешение (точность важнее скорости)

    # WLS постфильтр: left-right consistency, улучшает плоскостность
    USE_WLS_FILTER = False
    WLS_LAMBDA = 8000.0
    WLS_SIGMA = 1.5

    # CLAHE перед матчингом — усиливает локальный контраст на однородных поверхностях
    # (красный тетраэдр) для лучшего stereo matching
    USE_CLAHE = False

    ROI_MARGIN =  800
    ROI_MARGIN_TOP =  400
    ROI_MARGIN_BOTTOM =  500

    # Swap left/right images for stereo matching (use if Z comes out negative; fix physical wiring instead)
    SWAP_LEFT_RIGHT = False

    CAMERA_MIN_DISTANCE = 10.0
    CAMERA_MAX_DISTANCE = 50.0

    CROP_RADIUS = 60.0
    Z_MIN = -10.0
    Z_MAX = 20.0

    USE_NOISE_FILTER = True
    SOR_K = 20
    SOR_STD_MULTIPLIER = 2.0

    # Фильтр по цвету: оставлять только точки определенного цвета
    USE_COLOR_FILTER = False  # Включить/выключить фильтр цвета
    TARGET_COLOR_RGB = [200, 100, 50]  # Целевой цвет в RGB (красный по умолчанию)
    COLOR_TOLERANCE = 100  # Допустимое отклонение по каждому каналу RGB (0-255)

    SAVE_POINTS = True
    POINTS_FILE = str(OUTPUT_DIR / "stereo_points.npy")
    SAVE_DEBUG_DISPARITY = True
    SAVE_ALIGNMENT_BUNDLE = True
    ALIGNMENT_BUNDLE_FILE = str(OUTPUT_DIR / "stereo_alignment_bundle.npz")
    ALIGNMENT_BUNDLE_MAX_POINTS_PER_FRAME = 0  # 0 = без лимита (все точки в bundle)
    PLOTLY_MODE = "html"
    PLOTLY_FILE = str(OUTPUT_DIR / "points_cloud.html")
    # Limit points in HTML to avoid "Array buffer allocation failed" in browser WebGL.
    # Plotly embeds data into the file; too many points exceed JS memory limits (~2GB).
    PICK_MAX_POINTS = 500_000  # 0 = no limit (risks browser crash on large clouds)


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


def load_capture_metadata() -> dict | None:
    """Загрузить метаданные съёмки (углы поворота и т.д.) из capture_metadata.json."""
    path = get_capture_metadata_file()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load capture metadata from {path}: {e}")
        return None


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
    if "T" in d:
        T = np.asarray(d["T"]).ravel()
        out["baseline_cm"] = float(np.linalg.norm(T) * 100.0)
    elif "P2" in d:
        P2 = d["P2"]
        fx = P2[0, 0]
        if abs(fx) > 1e-6:
            out["baseline_cm"] = float(abs(P2[0, 3] / fx) * 100.0)
        else:
            out["baseline_cm"] = 0.0
    else:
        out["baseline_cm"] = 0.0
    return out


def extract_frame_id(name):
    m = re.search(r"(?:capture|frame)_(\d+)_left\.", name)
    return int(m.group(1)) if m else None


def collect_pairs(images_dir):
    root = Path(images_dir)
    left_files = sorted(root.glob("capture_*_left.png")) + sorted(root.glob("frame_*_left.png"))
    out = []
    for left in left_files:
        right = root / left.name.replace("_left.", "_right.")
        if not right.exists():
            continue
        fid = extract_frame_id(left.name)
        out.append((str(left), str(right), int(fid if fid is not None else len(out))))
    return out


def build_rectify_maps(calib):
    size = calib["image_size"]
    map1x, map1y = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_32FC1)
    return (map1x, map1y), (map2x, map2y)


def _create_wls_filter(matcher):
    try:
        import cv2.ximgproc as ximgproc
        wls = ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(getattr(Config, "WLS_LAMBDA", 8000.0))
        wls.setSigmaColor(getattr(Config, "WLS_SIGMA", 1.5))
        return wls, ximgproc.createRightMatcher(matcher)
    except Exception:
        return None, None


def compute_points_3d(img_l, img_r, calib, maps):
    size = calib["image_size"]
    # Подготовка градаций серого для disparity
    gl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if img_l.ndim == 3 else img_l
    gr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if img_r.ndim == 3 else img_r
    if gl.shape[:2][::-1] != size:
        gl = cv2.resize(gl, size)
    if gr.shape[:2][::-1] != size:
        gr = cv2.resize(gr, size)
    rl = cv2.remap(gl, maps[0][0], maps[0][1], cv2.INTER_LINEAR)
    rr = cv2.remap(gr, maps[1][0], maps[1][1], cv2.INTER_LINEAR)

    if getattr(Config, "USE_CLAHE", False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        rl = clahe.apply(rl)
        rr = clahe.apply(rr)

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
    use_wls = getattr(Config, "USE_WLS_FILTER", False)
    wls_filter, right_matcher = _create_wls_filter(sgbm) if use_wls else (None, None)
    if use_wls and wls_filter is not None and right_matcher is not None:
        disp_left = sgbm.compute(rl, rr)
        disp_right = right_matcher.compute(rr, rl)
        disp_filtered = wls_filter.filter(disp_left, rl, None, disp_right)
        disp = np.clip(disp_filtered.astype(np.float32), 0, None)  # ещё *16, как у SGBM
    else:
        disp = np.clip(sgbm.compute(rl, rr).astype(np.float32), 0, None)
    disp = (cv2.resize(disp / 16.0, size, interpolation=cv2.INTER_LINEAR) * (1.0 / scale)) if scale < 1.0 else disp / 16.0
    if Config.DISPARITY_MEDIAN_SIZE >= 3:
        k = Config.DISPARITY_MEDIAN_SIZE if Config.DISPARITY_MEDIAN_SIZE % 2 else Config.DISPARITY_MEDIAN_SIZE + 1
        mask0 = disp <= 0
        disp = cv2.medianBlur(disp, k)
        disp[mask0] = 0.0

    pts = cv2.reprojectImageTo3D(disp, calib["Q"])
    z_ch = pts[:, :, 2]
    valid_d = disp > 0
    if valid_d.any():
        z_where_disp = z_ch[valid_d]
        _n_neg = int(np.sum(z_where_disp <= 0))
        _n_inf = int(np.sum(~np.isfinite(z_where_disp)))
        _n_gt100 = int(np.sum(z_where_disp > 100))
        _n_ok = int(np.sum((z_where_disp > 0) & (z_where_disp <= 100) & np.isfinite(z_where_disp)))
        if _n_ok == 0 and not hasattr(compute_points_3d, "_diag_printed"):
            d_pos = disp[valid_d]
            print(f"  [diag] disp (px): min={float(np.min(d_pos)):.3f} max={float(np.max(d_pos)):.3f} mean={float(np.mean(d_pos)):.3f}")
            print(f"  [diag] Z (m) where disp>0: neg={_n_neg} inf={_n_inf} >100={_n_gt100} ok(0,100]={_n_ok}")
            compute_points_3d._diag_printed = True
    bad = (z_ch <= 0) | (z_ch > 100.0)
    pts[bad] = [np.nan, np.nan, np.nan]

    # #region agent log — depth diagnostic (hyp A: spatial bias, B: Q/calib)
    h, w = disp.shape[:2]
    valid_d = disp > 0
    if valid_d.any():
        mid_h, mid_w = h // 2, w // 2
        ql = disp[:mid_h, :mid_w][disp[:mid_h, :mid_w] > 0]
        qr = disp[:mid_h, mid_w:][disp[:mid_h, mid_w:] > 0]
        qt = disp[mid_h:, :mid_w][disp[mid_h:, :mid_w] > 0]
        qb = disp[mid_h:, mid_w:][disp[mid_h:, mid_w:] > 0]
        q_means = {"left": float(np.mean(ql)) if len(ql) else 0, "right": float(np.mean(qr)) if len(qr) else 0, "top": float(np.mean(qt)) if len(qt) else 0, "bottom": float(np.mean(qb)) if len(qb) else 0}
        center = disp[mid_h - 200 : mid_h + 200, mid_w - 200 : mid_w + 200]
        center_val = center[center > 0]
        q_means["center_mean"] = float(np.mean(center_val)) if len(center_val) else 0
        q_means["center_std"] = float(np.std(center_val)) if len(center_val) else 0
        lr_d = (q_means.get("right") or 0) - (q_means.get("left") or 0)
        tb_d = (q_means.get("bottom") or 0) - (q_means.get("top") or 0)
        _dbg_log("main.py:compute_points_3d", "disparity_quadrants", {"hypothesisId": "A", **q_means, "lr_diff": lr_d, "tb_diff": tb_d})
        valid_pts = np.isfinite(pts).all(axis=2) & (pts[:, :, 2] > 0.01)
        if valid_pts.any():
            z_vals = pts[:, :, 2][valid_pts] * 100
            z_left = pts[:mid_h, :mid_w, 2][valid_pts[:mid_h, :mid_w]]
            z_right = pts[:mid_h, mid_w:, 2][valid_pts[:mid_h, mid_w:]]
            z_left = z_left[np.isfinite(z_left) & (z_left > 0.01)] * 100
            z_right = z_right[np.isfinite(z_right) & (z_right > 0.01)] * 100
            _dbg_log("main.py:compute_points_3d", "depth_quadrants_cm", {"hypothesisId": "A", "z_left_mean": float(np.mean(z_left)) if len(z_left) else 0, "z_right_mean": float(np.mean(z_right)) if len(z_right) else 0, "z_diff_cm": (float(np.mean(z_right)) if len(z_right) else 0) - (float(np.mean(z_left)) if len(z_left) else 0)})
    # #endregion
    return pts, disp, rl_color


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
        Config.CAMERA_START_ANGLE_DEG + Config.TABLE_ROTATION_STEP * frame_id
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

    # #region agent log
    if frame_id <= 1:
        _dbg_log("main.py:camera_pose_for_frame", "pose", {"frame_id": frame_id, "theta_deg": float(np.degrees(theta)), "c_block": c_block.tolist()})
    # #endregion
    return r_cw, c_block


def cam_to_world(points_cam_cm, frame_id):
    r_cw, c_block = camera_pose_for_frame(frame_id)
    return points_cam_cm @ r_cw.T + c_block.reshape(1, 3)


def _rotate_around_z(points_xy, angle_rad, center_x=0.0, center_y=0.0):
    """Поворот точек вокруг оси Z в плоскости XY."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x = points_xy[:, 0] - center_x
    y = points_xy[:, 1] - center_y
    x_new = c * x - s * y + center_x
    y_new = s * x + c * y + center_y
    return np.column_stack([x_new, y_new])


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
            # Даунсэмплинг цветов по тем же индексам (при лимите 0 не режем)
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
        "camera_start_angle_deg": np.float32(Config.CAMERA_START_ANGLE_DEG),
        "extra_frame_rot_z_deg": np.float32(0.0),
        "platform_rotation_sign": np.int32(1),
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

    # Углы поворота из метаданных съёмки (capture_frames_with_rotation.py)
    capture_meta = load_capture_metadata()
    if capture_meta and "rotation_deg_per_frame" in capture_meta:
        Config.TABLE_ROTATION_STEP = float(capture_meta["rotation_deg_per_frame"])
        print(f"TABLE_ROTATION_STEP из capture_metadata: {Config.TABLE_ROTATION_STEP}°")
    # #region agent log
    _dbg_log("main.py:capture_meta", "capture metadata", {"rotation_deg": Config.TABLE_ROTATION_STEP, "dir_cw": capture_meta.get("dir_cw") if capture_meta else None})
    # #endregion

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

    # Параметры позы из cloud_pose_tuner (configs/pose_tuned_params.json), если есть
    pose_params_file = get_pose_params_file()
    pose_params = {}
    if pose_params_file.exists():
        try:
            with pose_params_file.open("r", encoding="utf-8") as f:
                pose_params = json.load(f)
            mapping = {
                "CAMERA_START_ANGLE_DEG": "CAMERA_START_ANGLE_DEG",
                "TABLE_ROTATION_STEP": "TABLE_ROTATION_STEP",
                "ORBIT_RADIUS": "ORBIT_RADIUS_XY",
                "CAMERA_HEIGHT": "CAMERA_HEIGHT_Z",
                "CAMERA_TILT": "CAMERA_TILT_DEG",
                "CAMERA_OFFSET_Y": "CAMERA_OFFSET_Y",
                "CAMERA_MIN_DISTANCE": "CAMERA_MIN_DISTANCE",
                "CAMERA_MAX_DISTANCE": "CAMERA_MAX_DISTANCE",
                "SWAP_LEFT_RIGHT": "SWAP_LEFT_RIGHT",
            }
            for json_key, config_attr in mapping.items():
                if json_key in pose_params and hasattr(Config, config_attr):
                    setattr(Config, config_attr, pose_params[json_key])
                    print(f"  {config_attr} = {pose_params[json_key]} (from pose_tuned_params)")
        except Exception as e:
            print(f"Warning: Could not load pose params from {pose_params_file}: {e}")

    print(f"Calibration: {CALIBRATION_FILE}")
    calib = load_stereo_calib(CALIBRATION_FILE)
    # #region agent log — calibration (hyp B: Q error)
    Q = calib["Q"]
    _dbg_log("main.py:main", "calib_Q", {"hypothesisId": "B", "Q_row3": Q[3, :].tolist(), "baseline_cm": calib.get("baseline_cm"), "image_size": list(calib["image_size"])})
    # #endregion
    maps = build_rectify_maps(calib)
    pairs = collect_pairs(Config.INPUT_DIR)
    if not pairs:
        raise RuntimeError(f"No stereo pairs in {Config.INPUT_DIR}")

    n = len(pairs)
    step = Config.TABLE_ROTATION_STEP
    print(f"Pairs: {n} | orbit_xy={Config.ORBIT_RADIUS_XY} cm, z={Config.CAMERA_HEIGHT_Z} cm")
    print(f"  TABLE_ROTATION_STEP={step}° (full turn 360° = {360/step:.1f} frames; for {n} frames use {360/n:.1f}° per frame)")

    all_clouds_cam = []
    load_failures = 0
    empty_cloud_frames = []
    for i, (pl, pr, fid) in enumerate(pairs):
        left, right = cv2.imread(pl), cv2.imread(pr)
        if left is None or right is None:
            load_failures += 1
            if load_failures <= 3:
                print(f"  [skip] Frame {fid}: failed to load {pl} or {pr}")
            continue
        # #region agent log — image size vs calib (hyp C: resolution mismatch)
        if i == 0:
            _dbg_log("main.py:main", "img_size", {"hypothesisId": "C", "left_shape": list(left.shape), "calib_size": list(calib["image_size"]), "match": left.shape[:2][::-1] == tuple(calib["image_size"])})
        # #endregion
        if getattr(Config, "SWAP_LEFT_RIGHT", False):
            left, right = right, left
        points_3d, disp, left_color_rect = compute_points_3d(left, right, calib, maps)
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

        cloud_cam = points_map_to_cloud_cm(points_3d, colors_bgr=left_color_rect)
        if len(cloud_cam) == 0:
            disp_valid = int(np.sum(disp > 0))
            pts_valid = np.isfinite(points_3d).all(axis=2) & (points_3d[:, :, 2] > 1e-6)
            n_pts = int(np.sum(pts_valid))
            empty_cloud_frames.append((fid, disp_valid, n_pts, disp.size))
            if len(empty_cloud_frames) <= 2:
                print(f"  [skip] Frame {fid}: 0 points (disparity: {disp_valid}/{disp.size} nonzero, valid Z: {n_pts})")
            continue
        if Config.USE_COLOR_FILTER and cloud_cam.shape[1] >= 6:
            cloud_cam = filter_by_color(cloud_cam, Config.TARGET_COLOR_RGB, Config.COLOR_TOLERANCE)
            if len(cloud_cam) == 0:
                empty_cloud_frames.append((fid, 0, 0, disp.size))
                if len(empty_cloud_frames) <= 2:
                    print(f"  [skip] Frame {fid}: 0 points after color filter")
                continue
        d = np.linalg.norm(cloud_cam[:, :3], axis=1)
        mask = np.ones(len(cloud_cam), dtype=bool)
        if Config.CAMERA_MIN_DISTANCE > 0:
            mask &= d >= Config.CAMERA_MIN_DISTANCE
        if Config.CAMERA_MAX_DISTANCE > 0:
            mask &= d <= Config.CAMERA_MAX_DISTANCE
        cloud_cam = cloud_cam[mask]
        if len(cloud_cam) == 0:
            empty_cloud_frames.append((fid, 0, 0, disp.size))
            if len(empty_cloud_frames) <= 2:
                print(f"  [skip] Frame {fid}: 0 points after distance filter")
            continue
        all_clouds_cam.append((fid, cloud_cam))

        if Config.SAVE_DEBUG_DISPARITY and i == 0:
            vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(OUTPUT_DIR / "debug_disparity.png"), vis)

    if not all_clouds_cam:
        msg = "No points after ROI."
        if load_failures == n:
            msg += f" All {n} frames: images failed to load. Check INPUT_DIR paths."
        elif empty_cloud_frames:
            fid0, d_valid, n_pts, d_size = empty_cloud_frames[0]
            if d_valid == 0:
                msg += f" Disparity is all zeros (SGBM produced no matches). Check: left/right swap, calibration, BLOCK_SIZE (try 5 or 9)."
            elif n_pts == 0:
                msg += f" Disparity has {d_valid} nonzero pixels but reprojection yields no valid Z. Check calibration Q matrix."
            else:
                msg += f" First frame: {d_valid}/{d_size} disp nonzero, {n_pts} valid pts before cloud — check points_map_to_cloud_cm."
        raise RuntimeError(msg)
    save_bundle(all_clouds_cam)

    # Физическая модель: камера закреплена, объект вращается (turntable).
    # Точки всех кадров в одной кадровой СК. Трансформация: cam→world (поза кадра 0), затем поворот undo.
    use_turntable = capture_meta is not None and "frame_count" in capture_meta
    platform_sign = int(capture_meta.get("dir_cw", 1)) if capture_meta else 1
    if "PLATFORM_ROTATION_SIGN" in pose_params:
        platform_sign = int(pose_params["PLATFORM_ROTATION_SIGN"])
    pivot_x = float(pose_params.get("TABLE_CENTER_X", 0.0))
    pivot_y = float(pose_params.get("TABLE_CENTER_Y", 0.0))
    if use_turntable:
        print("Model: turntable (camera fixed, object rotates)")
    r_cw_fixed, c_block_fixed = camera_pose_for_frame(0)

    all_world = []
    for frame_index, (fid, cloud_cam) in enumerate(all_clouds_cam):
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

        if use_turntable:
            xyz_world = xyz_cam @ r_cw_fixed.T + c_block_fixed.reshape(1, 3)
            angle_undo_rad = np.radians(-platform_sign * Config.TABLE_ROTATION_STEP * fid)
            xy_rot = _rotate_around_z(xyz_world[:, :2], angle_undo_rad, pivot_x, pivot_y)
            xyz_world = np.column_stack([xy_rot, xyz_world[:, 2]])
        else:
            xyz_world = cam_to_world(xyz_cam, fid)

        # #region agent log
        centroid_cam = np.mean(xyz_cam, axis=0)
        centroid_world = np.mean(xyz_world, axis=0)
        _dbg_log("main.py:transform", "frame transform", {"fid": fid, "turntable": use_turntable, "centroid_world": centroid_world.tolist()})
        # #endregion
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
        print(f"Downsampled {len(xyz)} -> {Config.PICK_MAX_POINTS} points for HTML (browser memory limit)")
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
