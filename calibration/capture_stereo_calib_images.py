"""
Захват пар кадров с двух камер для стереокалибровки.
Пара сохраняется вручную по клавише, только если на ОБЕИХ камерах найдено
достаточно маркеров ChArUco.
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
# Минимальная пауза по кадрам между ручными сохранениями пар
MIN_FRAME_GAP = 30
# How many frames to show the "Saved" overlay
FLASH_FRAMES = 10


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


def _draw_center_message(image, text, font_scale=1.2, thickness=3):
    """Рисует текст по центру кадра с полупрозрачным фоном."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = image.shape[:2]
    x = (w - text_w) // 2
    y = (h + text_h) // 2
    pad = 16
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - text_h - pad),
        (x + text_w + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


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
    print("Зафиксируйте доску, затем нажмите 's' или Space для снимка. 'q' — выход.")

    saved_count = 0
    frame_idx = 0
    last_saved_frame = -MIN_FRAME_GAP
    flash_until_frame = -1
    flash_message = ""

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
        save_ready = (frame_idx - last_saved_frame) >= MIN_FRAME_GAP
        if both_ok and save_ready:
            status_global = "READY: press S or Space to save"
            color_global = (0, 255, 0)
        elif both_ok:
            frames_left = MIN_FRAME_GAP - (frame_idx - last_saved_frame)
            status_global = f"WAIT: pause before next save ({frames_left} frames)"
            color_global = (0, 255, 255)
        else:
            status_global = "WAIT: need markers on BOTH cameras"
            color_global = (0, 0, 255)
        cv2.putText(
            combined, status_global,
            (10, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_global, 2
        )

        if frame_idx <= flash_until_frame and flash_message:
            _draw_center_message(combined, flash_message)

        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key in (ord('s'), ord(' ')):
            if not save_ready:
                frames_left = MIN_FRAME_GAP - (frame_idx - last_saved_frame)
                print(f"Снимок не сохранен: подождите еще {frames_left} кадров.")
                flash_message = "Wait before next save"
                flash_until_frame = frame_idx + FLASH_FRAMES
                frame_idx += 1
                continue

            if not both_ok:
                print(
                    "Снимок не сохранен: недостаточно маркеров "
                    f"(left {n_left}/{total_markers}, right {n_right}/{total_markers})."
                )
                flash_message = "Not enough markers"
                flash_until_frame = frame_idx + FLASH_FRAMES
                frame_idx += 1
                continue

            # Сохраняем с именами left/right: кадр с cap_left -> _right.png, cap_right -> _left.png (исправлено «задом наперёд»)
            path_left = IMAGES_DIR / f"capture_{saved_count:04d}_left.png"
            path_right = IMAGES_DIR / f"capture_{saved_count:04d}_right.png"
            cv2.imwrite(str(path_left), frame_right)
            cv2.imwrite(str(path_right), frame_left)
            print(f"Saved pair #{saved_count}: left {n_left} markers, right {n_right} markers")
            flash_message = f"Saved #{saved_count + 1}"
            flash_until_frame = frame_idx + FLASH_FRAMES
            saved_count += 1
            last_saved_frame = frame_idx

        frame_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print(f"Сохранено пар: {saved_count} в {IMAGES_DIR}")


if __name__ == "__main__":
    main()
