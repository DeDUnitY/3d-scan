"""
Захват пар кадров с двух камер для стереокалибровки.
Если на ОБЕИХ камерах найдено достаточно маркеров ChArUco — сохраняет пару (left, right).
Камеры в стандартных настройках (автоэкспозиция, без подстройки под запись объекта).
"""
import cv2
import numpy as np
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT_DIR / "calibration" / "images_stereo"

# Индексы камер (левая = 0, правая = 1; поменяйте, если у вас наоборот)
CAM_LEFT_INDEX = 1
CAM_RIGHT_INDEX = 0

# Параметры ChArUco доски (совпадают с calibrate_camera_aruco.py):
# 11 колонок, 8 рядов (по рядам чередование 6/5 маркеров)
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_SIZE = 0.015
MARKER_LENGTH = 0.011
TOTAL_MARKERS = (SQUARES_X * SQUARES_Y) // 2
DICT_NAME = "DICT_4X4_1000"

# Минимальная доля найденных маркеров, чтобы кадр считался «хорошим»
MIN_MARKERS_RATIO = 0.6
# Минимальное абсолютное число маркеров в кадре
MIN_MARKERS_ABS = 15
# Минимальный разрыв по кадрам между автосохранениями пар
MIN_FRAME_GAP = 30


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
        return corners, ids, rejected, "normal"

    inv = 255 - gray
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners_inv, ids_inv, rejected_inv = detector.detectMarkers(inv)
    else:
        corners_inv, ids_inv, rejected_inv = cv2.aruco.detectMarkers(inv, dictionary, parameters=params)

    if ids_inv is not None and len(ids_inv) > 0:
        return corners_inv, ids_inv, rejected_inv, "inverted"
    return corners, ids, rejected, "normal"


def _process_frame(frame, dictionary, params, board_id_set, total_markers):
    """Детектирует маркеры на кадре, возвращает (display, n_unique, ratio, ok)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _, polarity = _detect_markers_auto(gray, dictionary, params)

    detected_ratio = 0.0
    unique_ids = np.array([])

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        valid = np.array([int(x) in board_id_set for x in ids_flat], dtype=bool)
        if np.any(valid):
            ids_flat = ids_flat[valid]
            unique_ids = np.unique(ids_flat)
            detected_ratio = len(unique_ids) / max(1, total_markers)
            corners = [c for c, v in zip(corners, valid) if v]
            ids = ids[valid].reshape(-1, 1)
        else:
            corners = []
            ids = None

    display = frame.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

    ok = detected_ratio >= MIN_MARKERS_RATIO and len(unique_ids) >= MIN_MARKERS_ABS
    return display, len(unique_ids), detected_ratio, polarity, ok


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    cap_left = cv2.VideoCapture(CAM_LEFT_INDEX)
    cap_right = cv2.VideoCapture(CAM_RIGHT_INDEX)
    if not cap_left.isOpened():
        raise RuntimeError(f"Cannot open left camera (index {CAM_LEFT_INDEX}).")
    if not cap_right.isOpened():
        raise RuntimeError(f"Cannot open right camera (index {CAM_RIGHT_INDEX}).")

    for cap in (cap_left, cap_right):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    w1 = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dictionary = _get_aruco_dictionary()
    params = _create_detector_params()
    board = _create_charuco_board(SQUARES_X, SQUARES_Y, SQUARE_SIZE, MARKER_LENGTH, dictionary)
    board_ids = board.getIds() if hasattr(board, "getIds") else board.ids
    board_id_set = set(int(x) for x in np.asarray(board_ids).flatten().tolist())
    total_markers = len(board_id_set)

    print("Стерео захват: две камеры запущены (стандартные настройки, автоэкспозиция).")
    print(f"Left:  {w1}x{h1}, Right: {w2}x{h2}")
    print(f"Пара сохраняется только когда на ОБЕИХ камерах найдено >= {MIN_MARKERS_ABS} маркеров (>= {MIN_MARKERS_RATIO*100:.0f}%).")
    print("Нажмите 'q' чтобы выйти.")

    saved_count = 0
    frame_idx = 0
    last_saved_frame = -MIN_FRAME_GAP

    window_name = "Stereo calib capture (Left | Right)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 600)

    while True:
        ok1, frame_left = cap_left.read()
        ok2, frame_right = cap_right.read()
        if not ok1 or not ok2:
            print("Ошибка чтения с камер.")
            break

        disp_left, n_left, ratio_left, pol_left, ok_left = _process_frame(
            frame_left, dictionary, params, board_id_set, total_markers
        )
        disp_right, n_right, ratio_right, pol_right, ok_right = _process_frame(
            frame_right, dictionary, params, board_id_set, total_markers
        )

        # Подписи на каждом кадре
        for disp, n, ratio, pol, ok, label in [
            (disp_left, n_left, ratio_left, pol_left, ok_left, "Left"),
            (disp_right, n_right, ratio_right, pol_right, ok_right, "Right"),
        ]:
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(
                disp, f"{label}: {n}/{total_markers} ({ratio*100:.0f}%) {pol}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            status = "OK" if ok else "Too few"
            cv2.putText(disp, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Сводим оба кадра в один (лево | право) с одинаковой высотой
        h_max = max(disp_left.shape[0], disp_right.shape[0])
        target_h = min(h_max, 540)
        scale_l = target_h / disp_left.shape[0]
        scale_r = target_h / disp_right.shape[0]
        new_w_l = int(disp_left.shape[1] * scale_l)
        new_w_r = int(disp_right.shape[1] * scale_r)
        left_resized = cv2.resize(disp_left, (new_w_l, target_h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(disp_right, (new_w_r, target_h), interpolation=cv2.INTER_AREA)
        combined = np.hstack([left_resized, right_resized])

        both_ok = ok_left and ok_right
        if both_ok:
            status_global = "OK: auto-save when gap reached"
            color_global = (0, 255, 0)
        else:
            status_global = "Need markers on BOTH cameras"
            color_global = (0, 0, 255)
        cv2.putText(
            combined, status_global,
            (10, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_global, 2
        )

        # Автосохранение пары только если на обеих камерах достаточно маркеров
        if (
            both_ok
            and (frame_idx - last_saved_frame) >= MIN_FRAME_GAP
        ):
            # Сохраняем с именами left/right: кадр с cap_left -> _right.png, cap_right -> _left.png (исправлено «задом наперёд»)
            path_left = IMAGES_DIR / f"capture_{saved_count:04d}_left.png"
            path_right = IMAGES_DIR / f"capture_{saved_count:04d}_right.png"
            cv2.imwrite(str(path_left), frame_right)
            cv2.imwrite(str(path_right), frame_left)
            print(f"Saved pair #{saved_count}: left {n_left} markers, right {n_right} markers")
            saved_count += 1
            last_saved_frame = frame_idx

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print(f"Сохранено пар: {saved_count} в {IMAGES_DIR}")


if __name__ == "__main__":
    main()
