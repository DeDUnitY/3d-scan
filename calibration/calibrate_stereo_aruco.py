"""
Стереокалибровка по парам изображений с доской ChArUco.
Читает пары capture_*_left.png / capture_*_right.png из calibration/images_stereo,
находит углы ChArUco на обеих камерах (только общие для левого и правого кадра),
вызывает cv2.stereoCalibrate и cv2.stereoRectify, сохраняет результат в NPZ и JSON.
Параметры доски совпадают с capture_stereo_calib_images.py.
"""
import json
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT_DIR / "calibration" / "images_stereo"
OUTPUT_DIR = ROOT_DIR / "configs"

# Параметры ChArUco доски (как в capture_stereo_calib_images.py)
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_SIZE = 0.015
MARKER_LENGTH = 0.011
DICT_NAME = "DICT_4X4_1000"

# Минимум общих углов ChArUco в паре, чтобы кадр использовался
MIN_COMMON_CORNERS = 12
# Минимум пар для калибровки
MIN_PAIRS = 5


def _get_aruco_dictionary():
    aruco = cv2.aruco
    if hasattr(aruco, DICT_NAME):
        return aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
    raise ValueError(f"Unknown ArUco dictionary: {DICT_NAME}")


def _create_detector_params():
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters"):
        p = aruco.DetectorParameters()
        p.minMarkerPerimeterRate = 0.02
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.05
        return p
    return aruco.DetectorParameters_create()


def _create_charuco_board(squares_x, squares_y, square_length, marker_length, dictionary):
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            dictionary,
        )
    return aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, dictionary
    )


def _detect_markers_auto(gray, dictionary, params):
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is not None and len(ids) > 0:
        return corners, ids, "normal"
    inv = 255 - gray
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(inv)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(inv, dictionary, parameters=params)
    if ids is not None and len(ids) > 0:
        return corners, ids, "inverted"
    return corners, ids, "normal"


def _get_charuco_corners(gray, board, dictionary, params):
    """Детектирует маркеры, интерполирует углы ChArUco. Возвращает (corners, ids) или (None, None)."""
    corners, ids, _ = _detect_markers_auto(gray, dictionary, params)
    if ids is None or len(ids) == 0:
        return None, None
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if not ret or charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None, None
    return charuco_corners, charuco_ids.flatten()


def _collect_stereo_pairs(images_dir):
    """Находит пары (left, right): capture_* или frame_* (_left.png / _right.png)."""
    images_dir = Path(images_dir)
    left_files = (
        sorted(images_dir.glob("capture_*_left.png")) +
        sorted(images_dir.glob("frame_*_left.png"))
    )
    seen = set()
    pairs = []
    for left_path in left_files:
        stem = left_path.name.replace("_left.", "_right.")
        right_path = images_dir / stem
        if right_path.exists() and left_path.name not in seen:
            seen.add(left_path.name)
            pairs.append((str(left_path), str(right_path)))
    return pairs


def _build_board_id_to_obj(board):
    """Словарь id угла ChArUco -> 3D точка в СК доски."""
    obj_pts = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    ids = board.getIds() if hasattr(board, "getIds") else board.ids
    ids_flat = np.asarray(ids).flatten()
    obj_arr = np.asarray(obj_pts)
    return {int(ids_flat[i]): np.array(obj_arr[i], dtype=np.float32).ravel() for i in range(len(ids_flat))}


def calibrate_stereo(
    images_dir=None,
    output_npz=None,
    min_common_corners=MIN_COMMON_CORNERS,
    min_pairs=MIN_PAIRS,
    swap_left_right=False,
):
    images_dir = Path(images_dir or IMAGES_DIR)
    output_npz = output_npz or str(OUTPUT_DIR / "stereo_calib.npz")

    pairs = _collect_stereo_pairs(images_dir)
    if len(pairs) < min_pairs:
        raise RuntimeError(
            f"Найдено пар: {len(pairs)}. Нужно минимум {min_pairs}. "
            f"Папка: {images_dir}. Имена: capture_* или frame_* (_left.png, _right.png)."
        )

    dictionary = _get_aruco_dictionary()
    params = _create_detector_params()
    board = _create_charuco_board(SQUARES_X, SQUARES_Y, SQUARE_SIZE, MARKER_LENGTH, dictionary)
    id_to_obj = _build_board_id_to_obj(board)

    object_points = []   # список массивов (N, 3)
    image_points_left = []
    image_points_right = []
    image_size = None

    used = 0
    for left_path, right_path in pairs:
        # Если имена файлов «задом наперёд»: _left.png = правый кадр, _right.png = левый
        if swap_left_right:
            img_l = cv2.imread(right_path)
            img_r = cv2.imread(left_path)
        else:
            img_l = cv2.imread(left_path)
            img_r = cv2.imread(right_path)
        if img_l is None or img_r is None:
            continue
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        corners_l, ids_l = _get_charuco_corners(gray_l, board, dictionary, params)
        corners_r, ids_r = _get_charuco_corners(gray_r, board, dictionary, params)
        if corners_l is None or corners_r is None:
            continue

        # Только углы, видимые на обеих камерах и принадлежащие доске (одинаковый порядок)
        common_ids = sorted(set(ids_l) & set(ids_r) & set(id_to_obj.keys()))
        if len(common_ids) < min_common_corners:
            continue

        # Индексы в массивах corners/ids для левого и правого
        idx_l = {int(i): np.where(ids_l == i)[0][0] for i in common_ids}
        idx_r = {int(i): np.where(ids_r == i)[0][0] for i in common_ids}

        obj_pts = np.array([id_to_obj[i] for i in common_ids], dtype=np.float32)
        pts_l = np.array([corners_l[idx_l[int(i)]].ravel() for i in common_ids], dtype=np.float32)
        pts_r = np.array([corners_r[idx_r[int(i)]].ravel() for i in common_ids], dtype=np.float32)

        object_points.append(obj_pts)
        image_points_left.append(pts_l)
        image_points_right.append(pts_r)
        if image_size is None:
            image_size = gray_l.shape[::-1]  # (width, height)
        used += 1

    if used < min_pairs:
        raise RuntimeError(
            f"Использовано пар с достаточным числом общих углов: {used}. Нужно минимум {min_pairs}."
        )

    # Начальное приближение внутренних параметров (stereoCalibrate требует K1,D1,K2,D2)
    w, h = image_size
    fx = max(w, h)
    K0 = np.array([
        [fx, 0, w * 0.5],
        [0, fx, h * 0.5],
        [0, 0, 1],
    ], dtype=np.float64)
    D0 = np.zeros(5, dtype=np.float64)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        K0.copy(),
        D0.copy(),
        K0.copy(),
        D0.copy(),
        image_size,
        None, None, None, None,
        flags=0,
        criteria=criteria,
    )

    # Ректификация для дальнейшего stereo matching
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        None, None, None, None, None,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        rms=float(rms),
        camera_matrix_left=K1,
        dist_coeffs_left=D1,
        camera_matrix_right=K2,
        dist_coeffs_right=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        image_size=np.array(image_size),
        roi1=np.array(roi1),
        roi2=np.array(roi2),
        squares_x=SQUARES_X,
        squares_y=SQUARES_Y,
        square_size=SQUARE_SIZE,
        marker_length=MARKER_LENGTH,
        dict_name=DICT_NAME,
    )

    json_path = Path(output_npz).with_suffix(".json")
    payload = {
        "rms": float(rms),
        "camera_matrix_left": K1.tolist(),
        "dist_coeffs_left": D1.tolist(),
        "camera_matrix_right": K2.tolist(),
        "dist_coeffs_right": D2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "image_size": list(image_size),
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_size": SQUARE_SIZE,
        "marker_length": MARKER_LENGTH,
        "dict_name": DICT_NAME,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "rms": rms,
        "K1": K1, "D1": D1, "K2": K2, "D2": D2,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "image_size": image_size,
        "npz": output_npz,
        "json": str(json_path),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Стереокалибровка по ChArUco (пары left/right).")
    parser.add_argument("--images", type=str, default=None, help=f"Папка с парами (по умолчанию {IMAGES_DIR})")
    parser.add_argument("--output", type=str, default=None, help="Путь к выходному .npz")
    parser.add_argument("--min-corners", type=int, default=MIN_COMMON_CORNERS, help="Мин. общих углов в паре")
    parser.add_argument("--min-pairs", type=int, default=MIN_PAIRS, help="Мин. пар для калибровки")
    parser.add_argument("--swap-left-right", action="store_true", help="Файлы сняты с перепутанными именами: _left.png = правый кадр, _right.png = левый")
    args = parser.parse_args()

    images_dir = (Path(args.images) if args.images else IMAGES_DIR).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Папка с кадрами не найдена: {images_dir}")
    output_npz = args.output
    if args.swap_left_right:
        print("Режим: --swap-left-right (файл _left.png = правый кадр, _right.png = левый)")
    print(f"Папка с парами: {images_dir}")

    result = calibrate_stereo(
        images_dir=images_dir,
        output_npz=output_npz,
        min_common_corners=args.min_corners,
        min_pairs=args.min_pairs,
        swap_left_right=args.swap_left_right,
    )

    print("Стереокалибровка выполнена.")
    print(f"  RMS: {result['rms']:.4f}")
    print(f"  Размер изображения: {result['image_size'][0]}x{result['image_size'][1]}")
    print(f"  Left  fx,fy,cx,cy: {result['K1'][0,0]:.2f}, {result['K1'][1,1]:.2f}, {result['K1'][0,2]:.2f}, {result['K1'][1,2]:.2f}")
    print(f"  Right fx,fy,cx,cy: {result['K2'][0,0]:.2f}, {result['K2'][1,1]:.2f}, {result['K2'][0,2]:.2f}, {result['K2'][1,2]:.2f}")
    print(f"  T (baseline): [{result['T'][0,0]:.4f}, {result['T'][1,0]:.4f}, {result['T'][2,0]:.4f}]")
    print(f"  Сохранено: {result['npz']}")
    print(f"  Сохранено: {result['json']}")


if __name__ == "__main__":
    main()
