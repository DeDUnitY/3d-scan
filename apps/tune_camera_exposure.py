"""
Подбор экспозиции и параметров камеры для стереопары.
Параметры OpenCV: Exposure, Gain, Brightness, Contrast, Saturation, Gamma (поддержка зависит от драйвера).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from object_config import get_camera_params_file

ROOT_DIR = Path(__file__).resolve().parents[1]
CAMERA_PARAMS_FILE = get_camera_params_file()

CAM_LEFT_INDEX = 1
CAM_RIGHT_INDEX = 0
DEFAULT_WIDTH = 2560
DEFAULT_HEIGHT = 1440

EXPOSURE_MIN, EXPOSURE_MAX = -13.0, 0.0
TRACKBAR_STEPS = 130
GAIN_MIN, GAIN_MAX = 0, 100
GAIN_TRACKBAR_MAX = 100
# Универсальный диапазон 0–100 для Brightness, Contrast, Saturation, Gamma (драйвер может использовать другой)
PROP_RANGE = 100
TEMP_MIN, TEMP_MAX = 2000, 8000
TEMP_TRACKBAR_MAX = TEMP_MAX - TEMP_MIN
# Окно предпросмотра камер — Full HD для оценки качества
PREVIEW_W, PREVIEW_H = 1920, 1080
PAIR_PREVIEW_H = 720

# Дефолты синхронизированы с apps/capture_frames_with_rotation.py
CAM_DEFAULT_EXPOSURE = -5.60
CAM_DEFAULT_GAIN = 36
CAM_DEFAULT_BRIGHTNESS = 0
CAM_DEFAULT_CONTRAST = 43
CAM_DEFAULT_SATURATION = 35
CAM_DEFAULT_GAMMA = 88
CAM_DEFAULT_TEMPERATURE = 2000
STARTUP_WARMUP_FRAMES = 12

# Параметры камеры в OpenCV (номер, имя)
CAM_PROPS = [
    (10, "Brightness"),
    (11, "Contrast"),
    (12, "Saturation"),
    (14, "Gain"),
    (15, "Exposure"),
    (21, "AutoExp"),
    (22, "Gamma"),
    (23, "Temperature"),
    (32, "Backlight"),
]


BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def load_camera_params_blob() -> dict | None:
    """Загрузить camera_params.json (как в tune_camera_and_disparity)."""
    if not CAMERA_PARAMS_FILE.exists():
        return None
    try:
        with open(CAMERA_PARAMS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def pick_side_blob(data: dict, side: str) -> dict | None:
    if side in data and isinstance(data[side], dict):
        return data[side]
    if "camera" in data and isinstance(data["camera"], dict):
        return data["camera"]
    if "exposure" in data:
        return data
    return None


def open_camera(index: int, backend_name: str) -> cv2.VideoCapture:
    backend = BACKENDS[backend_name]
    if backend == cv2.CAP_ANY:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def backend_label(cap: cv2.VideoCapture) -> str:
    try:
        return cap.getBackendName()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Подбор экспозиции для стереокамер")
    parser.add_argument("--left-cam", type=int, default=CAM_LEFT_INDEX, help="Индекс левой камеры")
    parser.add_argument("--right-cam", type=int, default=CAM_RIGHT_INDEX, help="Индекс правой камеры")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Высота кадра")
    parser.add_argument(
        "--backend",
        choices=tuple(BACKENDS.keys()),
        default="msmf",
        help="Backend OpenCV: any=как раньше, dshow=DirectShow с окном настроек драйвера, msmf=Media Foundation.",
    )
    parser.add_argument(
        "--start-auto",
        action="store_true",
        help="Стартовать в автоэкспозиции и брать начальные значения из cap.get() (старое поведение).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=STARTUP_WARMUP_FRAMES,
        help="Сколько пар кадров выбросить после первого применения manual-параметров.",
    )
    parser.add_argument(
        "--no-color-controls",
        action="store_true",
        help="Скрыть цветовой слайдер Temp.",
    )
    args = parser.parse_args()
    color_controls_enabled = not args.no_color_controls

    cap_left = open_camera(args.left_cam, args.backend)
    cap_right = open_camera(args.right_cam, args.backend)
    if not cap_left.isOpened():
        raise RuntimeError(f"Не удалось открыть левую камеру (индекс {args.left_cam})")
    if not cap_right.isOpened():
        raise RuntimeError(f"Не удалось открыть правую камеру (индекс {args.right_cam})")

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap_left.set(cv2.CAP_PROP_HUE, 0)
    cap_right.set(cv2.CAP_PROP_HUE, 0)
    print(f"Backend: left={backend_label(cap_left)}, right={backend_label(cap_right)}")

    # Вывод всех параметров левой камеры (какие драйвер поддерживает)
    print("--- Параметры левой камеры (что возвращает get) ---")
    for prop_id, name in CAM_PROPS:
        try:
            v = cap_left.get(prop_id)
            print(f"  {name}: {v}")
        except Exception as e:
            print(f"  {name}: err {e}")
    print("  (Brightness/Contrast/Saturation/Gamma/Color — слайдеры ниже; поддержка зависит от драйвера)")

    def exposure_from_slider(v):
        return EXPOSURE_MIN + (v / TRACKBAR_STEPS) * (EXPOSURE_MAX - EXPOSURE_MIN)

    def slider_from_exposure(e):
        return int(np.clip((e - EXPOSURE_MIN) / (EXPOSURE_MAX - EXPOSURE_MIN) * TRACKBAR_STEPS, 0, TRACKBAR_STEPS))

    def temp_from_slider(v):
        return int(np.clip(TEMP_MIN + int(v), TEMP_MIN, TEMP_MAX))

    def slider_from_temp(v):
        return int(np.clip(int(v) - TEMP_MIN, 0, TEMP_TRACKBAR_MAX))

    def apply_camera_params_left() -> None:
        # Как в tune_camera_and_disparity: задаём параметры пакетом, чтобы драйвер
        # не "терял" gain при раздельных set() вызовах.
        cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap_left.set(cv2.CAP_PROP_EXPOSURE, eL)
        cap_left.set(cv2.CAP_PROP_GAIN, gL)
        cap_left.set(cv2.CAP_PROP_BRIGHTNESS, briL)
        cap_left.set(cv2.CAP_PROP_CONTRAST, conL)
        cap_left.set(cv2.CAP_PROP_SATURATION, satL)
        cap_left.set(cv2.CAP_PROP_GAMMA, gamL)

    def apply_camera_params_right() -> None:
        # Как в tune_camera_and_disparity: задаём параметры пакетом, чтобы драйвер
        # не "терял" gain при раздельных set() вызовах.
        cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap_right.set(cv2.CAP_PROP_EXPOSURE, eR)
        cap_right.set(cv2.CAP_PROP_GAIN, gR)
        cap_right.set(cv2.CAP_PROP_BRIGHTNESS, briR)
        cap_right.set(cv2.CAP_PROP_CONTRAST, conR)
        cap_right.set(cv2.CAP_PROP_SATURATION, satR)
        cap_right.set(cv2.CAP_PROP_GAMMA, gamR)

    win = "Exposure"
    win_preview = "Preview (Full HD)"
    win_pair = "Stereo Pair (large)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 500, 720)
    cv2.namedWindow(win_preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_preview, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(win_pair, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_pair, 1920, PAIR_PREVIEW_H)

    saved_blob = load_camera_params_blob()
    left_saved = pick_side_blob(saved_blob, "left") if isinstance(saved_blob, dict) else None
    right_saved = pick_side_blob(saved_blob, "right") if isinstance(saved_blob, dict) else None

    if args.start_auto:
        try:
            eL = float(cap_left.get(cv2.CAP_PROP_EXPOSURE))
            eR = float(cap_right.get(cv2.CAP_PROP_EXPOSURE))
            eL = float(np.clip(eL, EXPOSURE_MIN, EXPOSURE_MAX))
            eR = float(np.clip(eR, EXPOSURE_MIN, EXPOSURE_MAX))
        except Exception:
            eL, eR = -6.0, -6.0

        try:
            gL = float(cap_left.get(cv2.CAP_PROP_GAIN))
            gR = float(cap_right.get(cv2.CAP_PROP_GAIN))
            gL = int(np.clip(gL, GAIN_MIN, GAIN_MAX))
            gR = int(np.clip(gR, GAIN_MIN, GAIN_MAX))
        except Exception:
            gL, gR = 0, 0
    else:
        def _from_saved(saved: dict | None, key: str, default):
            if not saved:
                return default
            if key not in saved:
                return default
            return saved.get(key, default)

        eL = float(
            np.clip(
                float(_from_saved(left_saved, "exposure", CAM_DEFAULT_EXPOSURE)),
                EXPOSURE_MIN,
                EXPOSURE_MAX,
            )
        )
        eR = float(
            np.clip(
                float(_from_saved(right_saved, "exposure", CAM_DEFAULT_EXPOSURE)),
                EXPOSURE_MIN,
                EXPOSURE_MAX,
            )
        )
        gL = int(np.clip(int(_from_saved(left_saved, "gain", CAM_DEFAULT_GAIN)), GAIN_MIN, GAIN_MAX))
        gR = int(np.clip(int(_from_saved(right_saved, "gain", CAM_DEFAULT_GAIN)), GAIN_MIN, GAIN_MAX))

    def _prop(cap, prop_id, default=50):
        try:
            v = float(cap.get(prop_id))
            return int(np.clip(v, 0, PROP_RANGE))
        except Exception:
            return default

    def _temp_prop(cap):
        try:
            v = float(cap.get(cv2.CAP_PROP_TEMPERATURE))
            if not np.isfinite(v) or v <= 0:
                return CAM_DEFAULT_TEMPERATURE
            return int(np.clip(v, TEMP_MIN, TEMP_MAX))
        except Exception:
            return CAM_DEFAULT_TEMPERATURE

    if args.start_auto:
        briL = _prop(cap_left, cv2.CAP_PROP_BRIGHTNESS)
        briR = _prop(cap_right, cv2.CAP_PROP_BRIGHTNESS)
        conL = _prop(cap_left, cv2.CAP_PROP_CONTRAST)
        conR = _prop(cap_right, cv2.CAP_PROP_CONTRAST)
        satL = _prop(cap_left, cv2.CAP_PROP_SATURATION)
        satR = _prop(cap_right, cv2.CAP_PROP_SATURATION)
        gamL = _prop(cap_left, cv2.CAP_PROP_GAMMA)
        gamR = _prop(cap_right, cv2.CAP_PROP_GAMMA)
        tempL = _temp_prop(cap_left)
        tempR = _temp_prop(cap_right)
    else:
        def _bri_from_saved(saved: dict | None) -> int:
            if not saved:
                return int(np.clip(CAM_DEFAULT_BRIGHTNESS, 0, PROP_RANGE))
            return int(np.clip(int(saved.get("brightness", CAM_DEFAULT_BRIGHTNESS)), 0, PROP_RANGE))

        def _con_from_saved(saved: dict | None) -> int:
            if not saved:
                return int(np.clip(CAM_DEFAULT_CONTRAST, 0, PROP_RANGE))
            return int(np.clip(int(saved.get("contrast", CAM_DEFAULT_CONTRAST)), 0, PROP_RANGE))

        def _sat_from_saved(saved: dict | None) -> int:
            if not saved:
                return int(np.clip(CAM_DEFAULT_SATURATION, 0, PROP_RANGE))
            return int(np.clip(int(saved.get("saturation", CAM_DEFAULT_SATURATION)), 0, PROP_RANGE))

        def _gam_from_saved(saved: dict | None) -> int:
            if not saved:
                return int(np.clip(CAM_DEFAULT_GAMMA, 0, PROP_RANGE))
            return int(np.clip(int(saved.get("gamma", CAM_DEFAULT_GAMMA)), 0, PROP_RANGE))

        def _temp_from_saved(saved: dict | None) -> int:
            if not saved:
                return CAM_DEFAULT_TEMPERATURE
            return int(np.clip(int(saved.get("temperature", CAM_DEFAULT_TEMPERATURE)), TEMP_MIN, TEMP_MAX))

        briL = _bri_from_saved(left_saved)
        briR = _bri_from_saved(right_saved)
        conL = _con_from_saved(left_saved)
        conR = _con_from_saved(right_saved)
        satL = _sat_from_saved(left_saved)
        satR = _sat_from_saved(right_saved)
        gamL = _gam_from_saved(left_saved)
        gamR = _gam_from_saved(right_saved)
        tempL = _temp_from_saved(left_saved)
        tempR = _temp_from_saved(right_saved)

    if args.start_auto:
        print("Старт: автоэкспозиция, начальные значения сняты с cap.get().")
    else:
        if saved_blob:
            print(f"Старт: manual из {CAMERA_PARAMS_FILE} (где ключи отсутствуют — дефолты как в capture).")
        else:
            print(
                f"Старт: manual дефолты (как в capture), т.к. {CAMERA_PARAMS_FILE} не найден/не читается."
            )

    def set_exposure_left(v):
        nonlocal eL
        eL = exposure_from_slider(v)
        apply_camera_params_left()

    def set_exposure_right(v):
        nonlocal eR
        eR = exposure_from_slider(v)
        apply_camera_params_right()

    def set_gain_left(v):
        nonlocal gL
        gL = int(np.clip(v, GAIN_MIN, GAIN_MAX))
        apply_camera_params_left()

    def set_gain_right(v):
        nonlocal gR
        gR = int(np.clip(v, GAIN_MIN, GAIN_MAX))
        apply_camera_params_right()

    def set_bri_left(v):
        nonlocal briL
        briL = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_left()

    def set_bri_right(v):
        nonlocal briR
        briR = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_right()

    def set_con_left(v):
        nonlocal conL
        conL = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_left()

    def set_con_right(v):
        nonlocal conR
        conR = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_right()

    def set_sat_left(v):
        nonlocal satL
        satL = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_left()

    def set_sat_right(v):
        nonlocal satR
        satR = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_right()

    def set_gam_left(v):
        nonlocal gamL
        gamL = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_left()

    def set_gam_right(v):
        nonlocal gamR
        gamR = int(np.clip(v, 0, PROP_RANGE))
        apply_camera_params_right()

    def set_temp_left(v):
        nonlocal tempL
        tempL = temp_from_slider(v)
        cap_left.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap_left.set(cv2.CAP_PROP_TEMPERATURE, tempL)

    def set_temp_right(v):
        nonlocal tempR
        tempR = temp_from_slider(v)
        cap_right.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap_right.set(cv2.CAP_PROP_TEMPERATURE, tempR)

    # Короткие подписи, чтобы не обрезались
    cv2.createTrackbar("L.Exp", win, slider_from_exposure(eL), TRACKBAR_STEPS, set_exposure_left)
    cv2.createTrackbar("R.Exp", win, slider_from_exposure(eR), TRACKBAR_STEPS, set_exposure_right)
    cv2.createTrackbar("L.Gain", win, gL, GAIN_TRACKBAR_MAX, set_gain_left)
    cv2.createTrackbar("R.Gain", win, gR, GAIN_TRACKBAR_MAX, set_gain_right)
    cv2.createTrackbar("Reset Auto", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Print", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Preview 0=L 1=R", win, 0, 1, lambda _: None)
    cv2.createTrackbar("L.Bri", win, briL, PROP_RANGE, set_bri_left)
    cv2.createTrackbar("R.Bri", win, briR, PROP_RANGE, set_bri_right)
    cv2.createTrackbar("L.Con", win, conL, PROP_RANGE, set_con_left)
    cv2.createTrackbar("R.Con", win, conR, PROP_RANGE, set_con_right)
    cv2.createTrackbar("L.Sat", win, satL, PROP_RANGE, set_sat_left)
    cv2.createTrackbar("R.Sat", win, satR, PROP_RANGE, set_sat_right)
    cv2.createTrackbar("L.Gam", win, gamL, PROP_RANGE, set_gam_left)
    cv2.createTrackbar("R.Gam", win, gamR, PROP_RANGE, set_gam_right)
    if color_controls_enabled:
        cv2.createTrackbar("L.Temp", win, slider_from_temp(tempL), TEMP_TRACKBAR_MAX, set_temp_left)
        cv2.createTrackbar("R.Temp", win, slider_from_temp(tempR), TEMP_TRACKBAR_MAX, set_temp_right)

    if not args.start_auto:
        # Применяем manual сразу (иначе первые кадры остаются в авто и «гуляют» между запусками).
        set_exposure_left(slider_from_exposure(eL))
        set_exposure_right(slider_from_exposure(eR))
        set_gain_left(gL)
        set_gain_right(gR)
        set_bri_left(briL)
        set_bri_right(briR)
        set_con_left(conL)
        set_con_right(conR)
        set_sat_left(satL)
        set_sat_right(satR)
        set_gam_left(gamL)
        set_gam_right(gamR)
        for _ in range(max(0, args.warmup_frames)):
            cap_left.read()
            cap_right.read()
        set_exposure_left(slider_from_exposure(eL))
        set_exposure_right(slider_from_exposure(eR))
        set_gain_left(gL)
        set_gain_right(gR)

    def reset_both_to_auto():
        cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        print("Обе камеры переведены в автоэкспозицию. Двигай слайдер — снова ручная.")

    def print_params():
        # Не cap.get(): на Windows драйвер часто не отражает set() — печатаем позиции слайдеров (то, что реально задаём).
        el = exposure_from_slider(cv2.getTrackbarPos("L.Exp", win))
        er = exposure_from_slider(cv2.getTrackbarPos("R.Exp", win))
        gl = cv2.getTrackbarPos("L.Gain", win)
        gr = cv2.getTrackbarPos("R.Gain", win)
        bl = cv2.getTrackbarPos("L.Bri", win)
        br = cv2.getTrackbarPos("R.Bri", win)
        cl = cv2.getTrackbarPos("L.Con", win)
        cr = cv2.getTrackbarPos("R.Con", win)
        sl = cv2.getTrackbarPos("L.Sat", win)
        sr = cv2.getTrackbarPos("R.Sat", win)
        gm_l = cv2.getTrackbarPos("L.Gam", win)
        gm_r = cv2.getTrackbarPos("R.Gam", win)
        print()
        print("--- Текущие параметры (по слайдерам) ---")
        print(f"  Left:  exp={el:.2f} gain={gl} bri={bl} con={cl} sat={sl} gamma={gm_l}")
        print(f"  Right: exp={er:.2f} gain={gr} bri={br} con={cr} sat={sr} gamma={gm_r}")
        if color_controls_enabled:
            t_l = temp_from_slider(cv2.getTrackbarPos("L.Temp", win))
            t_r = temp_from_slider(cv2.getTrackbarPos("R.Temp", win))
            print(f"  Left color:  temp={t_l}")
            print(f"  Right color: temp={t_r}")
        print("  (для capture: --left-exposure %s --right-exposure %s)" % (el, er))
        print()

    print(
        "Слайдеры: Exposure/Gain/Bri/Con/Sat/Gamma + Temp. "
        "Hue принудительно выставляется в 0. Клавиши: A=reset auto, P=print, Q=выход."
    )

    try:
        while True:
            ok1, frame_left = cap_left.read()
            ok2, frame_right = cap_right.read()
            if not ok1 or not ok2:
                continue
            target_h = 360
            scale_l = target_h / frame_left.shape[0]
            scale_r = target_h / frame_right.shape[0]
            small_l = cv2.resize(frame_left, (int(frame_left.shape[1] * scale_l), target_h), interpolation=cv2.INTER_AREA)
            small_r = cv2.resize(frame_right, (int(frame_right.shape[1] * scale_r), target_h), interpolation=cv2.INTER_AREA)
            mean_l = np.mean(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY))
            mean_r = np.mean(cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))
            try:
                read_gL = cap_left.get(cv2.CAP_PROP_GAIN)
                read_gR = cap_right.get(cv2.CAP_PROP_GAIN)
            except Exception:
                read_gL, read_gR = 0, 0
            cv2.putText(small_l, f"L mean={mean_l:.0f}  gain={read_gL:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(small_r, f"R mean={mean_r:.0f}  gain={read_gR:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            combined = np.hstack([small_l, small_r])
            cv2.putText(combined, "Bri/Con/Sat/Gam + Temp: поддержка зависит от backend/драйвера. Hue=0.",
                        (10, target_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.imshow(win, combined)

            pair_scale_l = PAIR_PREVIEW_H / frame_left.shape[0]
            pair_scale_r = PAIR_PREVIEW_H / frame_right.shape[0]
            pair_l = cv2.resize(
                frame_left,
                (int(frame_left.shape[1] * pair_scale_l), PAIR_PREVIEW_H),
                interpolation=cv2.INTER_AREA,
            )
            pair_r = cv2.resize(
                frame_right,
                (int(frame_right.shape[1] * pair_scale_r), PAIR_PREVIEW_H),
                interpolation=cv2.INTER_AREA,
            )
            cv2.putText(pair_l, f"LEFT mean={mean_l:.0f} gain={read_gL:.0f}", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(pair_r, f"RIGHT mean={mean_r:.0f} gain={read_gR:.0f}", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            pair_combined = np.hstack([pair_l, pair_r])
            cv2.imshow(win_pair, pair_combined)

            # Окно Full HD: одна камера, центральный кроп 1920x1080 (полный масштаб, без уменьшения)
            which = cv2.getTrackbarPos("Preview 0=L 1=R", win)
            frame = frame_left if which == 0 else frame_right
            h, w = frame.shape[:2]
            if w >= PREVIEW_W and h >= PREVIEW_H:
                x0 = (w - PREVIEW_W) // 2
                y0 = (h - PREVIEW_H) // 2
                preview_fhd = frame[y0:y0 + PREVIEW_H, x0:x0 + PREVIEW_W].copy()
            else:
                # Кадр меньше Full HD — вписываем в центр, чёрные поля по краям
                preview_fhd = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                preview_fhd[:] = 0
                scale = min(PREVIEW_W / w, PREVIEW_H / h)
                nw, nh = int(w * scale), int(h * scale)
                small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                x0 = (PREVIEW_W - nw) // 2
                y0 = (PREVIEW_H - nh) // 2
                preview_fhd[y0:y0 + nh, x0:x0 + nw] = small
            label = "L" if which == 0 else "R"
            mean_val = mean_l if which == 0 else mean_r
            gain_val = read_gL if which == 0 else read_gR
            cv2.putText(preview_fhd, f"{label}  mean={mean_val:.0f} gain={gain_val:.0f}  [crop 1920x1080 1:1]", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(win_preview, preview_fhd)

            # Слайдеры-кнопки: при значении 1 выполняем действие и сбрасываем в 0
            if cv2.getTrackbarPos("Reset Auto", win) == 1:
                reset_both_to_auto()
                cv2.setTrackbarPos("Reset Auto", win, 0)
            if cv2.getTrackbarPos("Print", win) == 1:
                print_params()
                cv2.setTrackbarPos("Print", win, 0)

            key = cv2.waitKey(50)
            if key >= 0:
                key = key & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    print_params()
                if key == ord("a"):
                    reset_both_to_auto()

            if (
                cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
                or cv2.getWindowProperty(win_preview, cv2.WND_PROP_VISIBLE) < 1
                or cv2.getWindowProperty(win_pair, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    print_params()
    print("Готово.")


if __name__ == "__main__":
    main()
