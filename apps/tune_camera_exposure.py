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
# Окно предпросмотра камер — Full HD для оценки качества
PREVIEW_W, PREVIEW_H = 1920, 1080

# Параметры камеры в OpenCV (номер, имя)
CAM_PROPS = [
    (10, "Brightness"),
    (11, "Contrast"),
    (12, "Saturation"),
    (13, "Hue"),
    (14, "Gain"),
    (15, "Exposure"),
    (17, "WB_Blue"),
    (20, "Sharpness"),
    (21, "AutoExp"),
    (22, "Gamma"),
    (26, "WB_Red"),
]


def main():
    parser = argparse.ArgumentParser(description="Подбор экспозиции для стереокамер")
    parser.add_argument("--left-cam", type=int, default=CAM_LEFT_INDEX, help="Индекс левой камеры")
    parser.add_argument("--right-cam", type=int, default=CAM_RIGHT_INDEX, help="Индекс правой камеры")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Высота кадра")
    args = parser.parse_args()

    # Бэкенд по умолчанию — не трогаем экспозицию при старте, картинка как раньше (авто).
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
    # Экспозицию не трогаем — камеры остаются в авто, картинка нормальная.

    # Вывод всех параметров левой камеры (какие драйвер поддерживает)
    print("--- Параметры левой камеры (что возвращает get) ---")
    for prop_id, name in CAM_PROPS:
        try:
            v = cap_left.get(prop_id)
            print(f"  {name}: {v}")
        except Exception as e:
            print(f"  {name}: err {e}")
    print("  (Brightness/Contrast/Saturation/Gamma — слайдеры ниже; поддержка зависит от драйвера)")

    def exposure_from_slider(v):
        return EXPOSURE_MIN + (v / TRACKBAR_STEPS) * (EXPOSURE_MAX - EXPOSURE_MIN)

    def slider_from_exposure(e):
        return int(np.clip((e - EXPOSURE_MIN) / (EXPOSURE_MAX - EXPOSURE_MIN) * TRACKBAR_STEPS, 0, TRACKBAR_STEPS))

    win = "Exposure"
    win_preview = "Preview (Full HD)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 500, 720)
    cv2.namedWindow(win_preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_preview, PREVIEW_W, PREVIEW_H)

    try:
        eL = float(cap_left.get(cv2.CAP_PROP_EXPOSURE))
        eR = float(cap_right.get(cv2.CAP_PROP_EXPOSURE))
        eL = np.clip(eL, EXPOSURE_MIN, EXPOSURE_MAX)
        eR = np.clip(eR, EXPOSURE_MIN, EXPOSURE_MAX)
    except Exception:
        eL, eR = -6.0, -6.0

    try:
        gL = float(cap_left.get(cv2.CAP_PROP_GAIN))
        gR = float(cap_right.get(cv2.CAP_PROP_GAIN))
        gL = int(np.clip(gL, GAIN_MIN, GAIN_MAX))
        gR = int(np.clip(gR, GAIN_MIN, GAIN_MAX))
    except Exception:
        gL, gR = 0, 0

    def _prop(cap, prop_id, default=50):
        try:
            v = float(cap.get(prop_id))
            return int(np.clip(v, 0, PROP_RANGE))
        except Exception:
            return default

    briL = _prop(cap_left, cv2.CAP_PROP_BRIGHTNESS)
    briR = _prop(cap_right, cv2.CAP_PROP_BRIGHTNESS)
    conL = _prop(cap_left, cv2.CAP_PROP_CONTRAST)
    conR = _prop(cap_right, cv2.CAP_PROP_CONTRAST)
    satL = _prop(cap_left, cv2.CAP_PROP_SATURATION)
    satR = _prop(cap_right, cv2.CAP_PROP_SATURATION)
    gamL = _prop(cap_left, cv2.CAP_PROP_GAMMA)
    gamR = _prop(cap_right, cv2.CAP_PROP_GAMMA)

    def set_exposure_left(v):
        nonlocal eL
        eL = exposure_from_slider(v)
        cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap_left.set(cv2.CAP_PROP_EXPOSURE, eL)

    def set_exposure_right(v):
        nonlocal eR
        eR = exposure_from_slider(v)
        cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap_right.set(cv2.CAP_PROP_EXPOSURE, eR)

    def set_gain_left(v):
        nonlocal gL
        gL = int(np.clip(v, GAIN_MIN, GAIN_MAX))
        cap_left.set(cv2.CAP_PROP_GAIN, gL)

    def set_gain_right(v):
        nonlocal gR
        gR = int(np.clip(v, GAIN_MIN, GAIN_MAX))
        cap_right.set(cv2.CAP_PROP_GAIN, gR)

    def set_bri_left(v):
        nonlocal briL
        briL = int(np.clip(v, 0, PROP_RANGE))
        cap_left.set(cv2.CAP_PROP_BRIGHTNESS, briL)

    def set_bri_right(v):
        nonlocal briR
        briR = int(np.clip(v, 0, PROP_RANGE))
        cap_right.set(cv2.CAP_PROP_BRIGHTNESS, briR)

    def set_con_left(v):
        nonlocal conL
        conL = int(np.clip(v, 0, PROP_RANGE))
        cap_left.set(cv2.CAP_PROP_CONTRAST, conL)

    def set_con_right(v):
        nonlocal conR
        conR = int(np.clip(v, 0, PROP_RANGE))
        cap_right.set(cv2.CAP_PROP_CONTRAST, conR)

    def set_sat_left(v):
        nonlocal satL
        satL = int(np.clip(v, 0, PROP_RANGE))
        cap_left.set(cv2.CAP_PROP_SATURATION, satL)

    def set_sat_right(v):
        nonlocal satR
        satR = int(np.clip(v, 0, PROP_RANGE))
        cap_right.set(cv2.CAP_PROP_SATURATION, satR)

    def set_gam_left(v):
        nonlocal gamL
        gamL = int(np.clip(v, 0, PROP_RANGE))
        cap_left.set(cv2.CAP_PROP_GAMMA, gamL)

    def set_gam_right(v):
        nonlocal gamR
        gamR = int(np.clip(v, 0, PROP_RANGE))
        cap_right.set(cv2.CAP_PROP_GAMMA, gamR)

    # Короткие подписи, чтобы не обрезались
    cv2.createTrackbar("L.Exp", win, slider_from_exposure(eL), TRACKBAR_STEPS, set_exposure_left)
    cv2.createTrackbar("R.Exp", win, slider_from_exposure(eR), TRACKBAR_STEPS, set_exposure_right)
    cv2.createTrackbar("L.Gain", win, gL, GAIN_TRACKBAR_MAX, set_gain_left)
    cv2.createTrackbar("R.Gain", win, gR, GAIN_TRACKBAR_MAX, set_gain_right)
    cv2.createTrackbar("Bal R<-L", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Bal L<-R", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Gain R=L", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Gain L=R", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Reset Auto", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Print", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Save JSON", win, 0, 1, lambda _: None)
    cv2.createTrackbar("Preview 0=L 1=R", win, 0, 1, lambda _: None)
    cv2.createTrackbar("L.Bri", win, briL, PROP_RANGE, set_bri_left)
    cv2.createTrackbar("R.Bri", win, briR, PROP_RANGE, set_bri_right)
    cv2.createTrackbar("L.Con", win, conL, PROP_RANGE, set_con_left)
    cv2.createTrackbar("R.Con", win, conR, PROP_RANGE, set_con_right)
    cv2.createTrackbar("L.Sat", win, satL, PROP_RANGE, set_sat_left)
    cv2.createTrackbar("R.Sat", win, satR, PROP_RANGE, set_sat_right)
    cv2.createTrackbar("L.Gam", win, gamL, PROP_RANGE, set_gam_left)
    cv2.createTrackbar("R.Gam", win, gamR, PROP_RANGE, set_gam_right)

    BALANCE_STEP = 1  # шаг в шкале -13..0

    def reset_both_to_auto():
        cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        print("Обе камеры переведены в автоэкспозицию. Двигай слайдер — снова ручная.")

    def print_params():
        el = cap_left.get(cv2.CAP_PROP_EXPOSURE)
        er = cap_right.get(cv2.CAP_PROP_EXPOSURE)
        gl, gr = cap_left.get(cv2.CAP_PROP_GAIN), cap_right.get(cv2.CAP_PROP_GAIN)
        bl, br = cap_left.get(cv2.CAP_PROP_BRIGHTNESS), cap_right.get(cv2.CAP_PROP_BRIGHTNESS)
        cl, cr = cap_left.get(cv2.CAP_PROP_CONTRAST), cap_right.get(cv2.CAP_PROP_CONTRAST)
        sl, sr = cap_left.get(cv2.CAP_PROP_SATURATION), cap_right.get(cv2.CAP_PROP_SATURATION)
        gm_l, gm_r = cap_left.get(cv2.CAP_PROP_GAMMA), cap_right.get(cv2.CAP_PROP_GAMMA)
        print()
        print("--- Текущие параметры ---")
        print(f"  Left:  exp={el:.2f} gain={gl} bri={bl} con={cl} sat={sl} gamma={gm_l}")
        print(f"  Right: exp={er:.2f} gain={gr} bri={br} con={cr} sat={sr} gamma={gm_r}")
        print("  (для capture: --left-exposure %s --right-exposure %s)" % (el, er))
        print()

    def save_params_to_json():
        """Сохранить exp, gain, bri, con, gam в JSON для capture_frames_with_rotation.py."""
        data = {
            "left": {
                "exposure": float(eL),
                "gain": int(gL),
                "brightness": int(briL),
                "contrast": int(conL),
                "gamma": int(gamL),
            },
            "right": {
                "exposure": float(eR),
                "gain": int(gR),
                "brightness": int(briR),
                "contrast": int(conR),
                "gamma": int(gamR),
            },
        }
        CAMERA_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CAMERA_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Параметры сохранены в {CAMERA_PARAMS_FILE}")

    def balance_right_to_left(mean_l, mean_r):
        """Подогнать правую под яркость левой."""
        nonlocal eR
        diff = mean_r - mean_l
        if abs(diff) < 5:
            return
        delta = -BALANCE_STEP if diff > 0 else BALANCE_STEP
        new_eR = np.clip(eR + delta, EXPOSURE_MIN, EXPOSURE_MAX)
        eR = new_eR
        cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap_right.set(cv2.CAP_PROP_EXPOSURE, eR)
        cv2.setTrackbarPos("R.Exp", win, slider_from_exposure(eR))

    def balance_left_to_right(mean_l, mean_r):
        """Подогнать левую под яркость правой."""
        nonlocal eL
        diff = mean_l - mean_r
        if abs(diff) < 5:
            return
        delta = -BALANCE_STEP if diff > 0 else BALANCE_STEP
        new_eL = np.clip(eL + delta, EXPOSURE_MIN, EXPOSURE_MAX)
        eL = new_eL
        cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap_left.set(cv2.CAP_PROP_EXPOSURE, eL)
        cv2.setTrackbarPos("L.Exp", win, slider_from_exposure(eL))

    print("Слайдеры: Exposure; Balance R←L / L←R; Reset auto; Print; Save JSON. Клавиши: B=balance R←L, A=reset auto, S=save JSON, Q=выход.")

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
            cv2.putText(combined, "Bri/Con/Sat/Gam = яркость, контраст, насыщенность, гамма. Работают только если драйвер поддерживает.",
                        (10, target_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.imshow(win, combined)

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
            if cv2.getTrackbarPos("Bal R<-L", win) == 1:
                balance_right_to_left(mean_l, mean_r)
                cv2.setTrackbarPos("Bal R<-L", win, 0)
            if cv2.getTrackbarPos("Bal L<-R", win) == 1:
                balance_left_to_right(mean_l, mean_r)
                cv2.setTrackbarPos("Bal L<-R", win, 0)
            if cv2.getTrackbarPos("Gain R=L", win) == 1:
                gL_now = cv2.getTrackbarPos("L.Gain", win)
                gR = int(np.clip(gL_now, GAIN_MIN, GAIN_MAX))
                cap_right.set(cv2.CAP_PROP_GAIN, gR)
                cv2.setTrackbarPos("R.Gain", win, gR)
                cv2.setTrackbarPos("Gain R=L", win, 0)
            if cv2.getTrackbarPos("Gain L=R", win) == 1:
                gR_now = cv2.getTrackbarPos("R.Gain", win)
                gL = int(np.clip(gR_now, GAIN_MIN, GAIN_MAX))
                cap_left.set(cv2.CAP_PROP_GAIN, gL)
                cv2.setTrackbarPos("L.Gain", win, gL)
                cv2.setTrackbarPos("Gain L=R", win, 0)
            if cv2.getTrackbarPos("Reset Auto", win) == 1:
                reset_both_to_auto()
                cv2.setTrackbarPos("Reset Auto", win, 0)
            if cv2.getTrackbarPos("Print", win) == 1:
                print_params()
                cv2.setTrackbarPos("Print", win, 0)
            if cv2.getTrackbarPos("Save JSON", win) == 1:
                save_params_to_json()
                cv2.setTrackbarPos("Save JSON", win, 0)

            key = cv2.waitKey(50)
            if key >= 0:
                key = key & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    print_params()
                if key == ord("b"):
                    balance_right_to_left(mean_l, mean_r)
                if key == ord("a"):
                    reset_both_to_auto()
                if key == ord("s"):
                    save_params_to_json()

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(win_preview, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Возвращаем автоэкспозицию, иначе камера остаётся в ручном режиме и в Windows показывается плохо
        try:
            cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    print_params()
    print("Готово.")


if __name__ == "__main__":
    main()
