"""
Захват кадров по команде поворота стола по Serial (COM-порт).
MOVE → OK → задержка → снимок. По умолчанию камеры в автоэкспозиции.
Снимки и метаданные (углы поворота и т.д.) сохраняются в папку объекта из object_config.
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    raise ImportError("Установите pyserial: pip install pyserial")

from object_config import get_frames_dir, get_capture_metadata_file, get_camera_params_file, OBJECT_NAME

OUTPUT_DIR = get_frames_dir()
CAMERA_PARAMS_FILE = get_camera_params_file()

CAM_LEFT_INDEX = 1
CAM_RIGHT_INDEX = 0
CAPTURE_WIDTH = 2560
CAPTURE_HEIGHT = 1440
DELAY_AFTER_ROTATION = 0.5
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_READ = 120.0

# Параметры по умолчанию (если нет camera_params.json из tune_camera_exposure)
CAM_EXPOSURE = -5.0
CAM_BRIGHTNESS = 41
CAM_CONTRAST = 38
CAM_GAMMA = 53
CAM_GAIN = 0


def load_camera_params():
    """Загрузить exp, gain, bri, con, gam из JSON (сохранённого в tune_camera_exposure)."""
    if not CAMERA_PARAMS_FILE.is_file():
        return None
    try:
        with open(CAMERA_PARAMS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def send_rotate_serial(
    ser: serial.Serial,
    rev_time: float = 1.2,
    dir_cw: int = 1,
    angle_deg: float | None = None,
    timeout_sec: float = SERIAL_TIMEOUT_READ,
) -> bool:
    """Отправить команду поворота по Serial, дождаться OK.
    Если angle_deg задан — MOVE_DEG <angle> [time] [dir] (поворот на угол в градусах).
    Иначе MOVE <time> <dir> (один фиксированный шаг платы). dir: 1=CW, 0=CCW."""
    ser.reset_input_buffer()
    if angle_deg is not None and angle_deg > 0:
        cmd = f"MOVE_DEG {angle_deg:.2f} {rev_time:.2f} {dir_cw}\n"
    else:
        cmd = f"MOVE {rev_time:.2f} {dir_cw}\n"
    ser.write(cmd.encode("ascii"))
    ser.flush()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        line = ser.readline()
        if not line:
            continue
        try:
            text = line.decode("ascii", errors="ignore").strip()
        except Exception:
            continue
        if text == "OK":
            return True
        # Игнорируем отладочные выводы и прочие строки, ждём именно OK
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Захват кадров: Serial MOVE → OK → 0.5 сек → снимок. Сохраняет в outputs/recorded/frames."
    )
    parser.add_argument("--port", type=str, default="COM3",
                        help="COM-порт платы (Windows: COM3, Linux: /dev/ttyUSB0)")
    parser.add_argument("--list-ports", action="store_true",
                        help="Показать доступные COM-порты и выйти")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="Скорость Serial (по умолчанию 115200)")
    parser.add_argument("--count", type=int, default=10,
                        help="Количество кадров на 360° (например 3 = три снимка по 120°)")
    parser.add_argument("--degrees-per-step", type=float, default=None,
                        help="Градусы поворота стола за одну команду MOVE. Если задано, между кадрами отправляется столько MOVE, чтобы суммарно вышло 360° (калибровка по железу)")
    parser.add_argument("--rev-time", type=float, default=1.2,
                        help="Время одного шага поворота, сек (меньше = быстрее)")
    parser.add_argument("--dir", type=int, default=1, choices=(0, 1),
                        help="Направление: 1=по часовой (CW), 0=против (CCW)")
    parser.add_argument("--delay", type=float, default=DELAY_AFTER_ROTATION,
                        help="Задержка после OK перед снимком, сек")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Папка для кадров (по умолчанию: папка объекта '{OBJECT_NAME}')")
    parser.add_argument("--left-cam", type=int, default=CAM_LEFT_INDEX, help="Индекс левой камеры")
    parser.add_argument("--right-cam", type=int, default=CAM_RIGHT_INDEX, help="Индекс правой камеры")
    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT, help="Высота кадра")
    parser.add_argument("--left-exposure", type=float, default=None, help="Экспозиция левой (-13..0). Не задано = CAM_EXPOSURE.")
    parser.add_argument("--right-exposure", type=float, default=None, help="Экспозиция правой (-13..0). Не задано = CAM_EXPOSURE.")
    parser.add_argument("--no-camera-params", action="store_true", help="Не задавать exp/bri/con/gam (оставить авто)")
    args = parser.parse_args()

    if args.list_ports:
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device} — {p.description}")
        return

    out_dir = Path(args.output) if args.output else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Открытие Serial {args.port} @ {args.baud}...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5, write_timeout=5.0)
    except serial.SerialException as e:
        raise RuntimeError(f"Не удалось открыть порт {args.port}: {e}") from e

    cap_left = cv2.VideoCapture(args.left_cam)
    cap_right = cv2.VideoCapture(args.right_cam)
    if not cap_left.isOpened():
        ser.close()
        raise RuntimeError(f"Не удалось открыть левую камеру (индекс {args.left_cam}).")
    if not cap_right.isOpened():
        ser.close()
        raise RuntimeError(f"Не удалось открыть правую камеру (индекс {args.right_cam}).")

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    saved_params = load_camera_params()
    if not args.no_camera_params:
        if saved_params and "left" in saved_params and "right" in saved_params:
            exp_left = args.left_exposure if args.left_exposure is not None else saved_params["left"]["exposure"]
            exp_right = args.right_exposure if args.right_exposure is not None else saved_params["right"]["exposure"]
            cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap_left.set(cv2.CAP_PROP_EXPOSURE, exp_left)
            cap_left.set(cv2.CAP_PROP_GAIN, saved_params["left"].get("gain", CAM_GAIN))
            cap_left.set(cv2.CAP_PROP_BRIGHTNESS, saved_params["left"]["brightness"])
            cap_left.set(cv2.CAP_PROP_CONTRAST, saved_params["left"]["contrast"])
            cap_left.set(cv2.CAP_PROP_GAMMA, saved_params["left"]["gamma"])
            cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap_right.set(cv2.CAP_PROP_EXPOSURE, exp_right)
            cap_right.set(cv2.CAP_PROP_GAIN, saved_params["right"].get("gain", CAM_GAIN))
            cap_right.set(cv2.CAP_PROP_BRIGHTNESS, saved_params["right"]["brightness"])
            cap_right.set(cv2.CAP_PROP_CONTRAST, saved_params["right"]["contrast"])
            cap_right.set(cv2.CAP_PROP_GAMMA, saved_params["right"]["gamma"])
        else:
            exp_left = args.left_exposure if args.left_exposure is not None else CAM_EXPOSURE
            exp_right = args.right_exposure if args.right_exposure is not None else CAM_EXPOSURE
            for cap, exp in [(cap_left, exp_left), (cap_right, exp_right)]:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, exp)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, CAM_BRIGHTNESS)
                cap.set(cv2.CAP_PROP_CONTRAST, CAM_CONTRAST)
                cap.set(cv2.CAP_PROP_GAMMA, CAM_GAMMA)
    else:
        exp_left = args.left_exposure if args.left_exposure is not None else None
        exp_right = args.right_exposure if args.right_exposure is not None else None
        if exp_left is not None or exp_right is not None:
            if exp_left is not None:
                cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap_left.set(cv2.CAP_PROP_EXPOSURE, exp_left)
            if exp_right is not None:
                cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap_right.set(cv2.CAP_PROP_EXPOSURE, exp_right)

    w_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Разрешение: левая {w_left}x{h_left}, правая {w_right}x{h_right}")
    if not args.no_camera_params:
        if saved_params and "left" in saved_params and "right" in saved_params:
            print(f"Камеры (из {CAMERA_PARAMS_FILE}): exp L={exp_left} R={exp_right}, bri/con/gam — по камерам")
        else:
            print(f"Камеры: exp={exp_left}/{exp_right} bri={CAM_BRIGHTNESS} con={CAM_CONTRAST} gam={CAM_GAMMA}")
    elif exp_left is not None or exp_right is not None:
        print(f"Экспозиция: левая {exp_left}, правая {exp_right}")
    else:
        print("Экспозиция: авто (не задано)")

    # Режим по углу (MOVE_DEG): одна команда = поворот на угол. Иначе — старый режим (несколько MOVE по --degrees-per-step).
    use_angle_mode = args.degrees_per_step is None
    deg_per_frame = 360.0 / args.count if args.count else 0.0
    if use_angle_mode:
        moves_between_frames = 1
        print("Режим: Serial MOVE_DEG <угол> → OK → задержка → снимок.")
        print(f"  Объект: {OBJECT_NAME}")
        print(f"  Порт: {args.port}, кадров: {args.count}, угол между кадрами: {deg_per_frame:.1f}° (ровно 360°)")
        print(f"  Время поворота: {args.rev_time} сек, направление: {'CW' if args.dir == 1 else 'CCW'}")
        print(f"  Задержка после OK: {args.delay} сек")
        print(f"  Сохранение: {out_dir}")
    else:
        moves_between_frames = max(1, math.ceil(deg_per_frame / args.degrees_per_step))
        print("Режим: Serial MOVE (по шагам) → OK → задержка → снимок.")
        print(f"  Объект: {OBJECT_NAME}")
        print(f"  Порт: {args.port}, кадров: {args.count}, 360° / {args.count} = {deg_per_frame:.0f}° между кадрами")
        print(f"  Градусов за 1 MOVE: {args.degrees_per_step}° → между кадрами {moves_between_frames} команд MOVE")
        print(f"  Время шага: {args.rev_time} сек, направление: {'CW' if args.dir == 1 else 'CCW'}")
        print(f"  Задержка после OK: {args.delay} сек")
        print(f"  Сохранение: {out_dir}")
    print()

    try:
        for i in range(args.count):
            if i > 0:
                if use_angle_mode:
                    print(f"  поворот на {deg_per_frame:.1f}°...", end=" ", flush=True)
                    if not send_rotate_serial(ser, rev_time=args.rev_time, dir_cw=args.dir, angle_deg=deg_per_frame):
                        print("таймаут (нет OK).")
                        break
                    print("OK.")
                else:
                    for m in range(moves_between_frames):
                        print(f"  поворот {m + 1}/{moves_between_frames}...", end=" ", flush=True)
                        if not send_rotate_serial(ser, rev_time=args.rev_time, dir_cw=args.dir):
                            print("таймаут (нет OK).")
                            break
                        print("OK.", end=" ", flush=True)
                    if moves_between_frames > 0:
                        print()
            print(f"Кадр {i + 1}/{args.count}: снимок...", end=" ", flush=True)
            time.sleep(args.delay)
            ok1, frame_left = cap_left.read()
            ok2, frame_right = cap_right.read()
            if not ok1 or not ok2:
                print("Ошибка чтения с камер.")
                break
            path_left = out_dir / f"capture_{i:04d}_left.png"
            path_right = out_dir / f"capture_{i:04d}_right.png"
            cv2.imwrite(str(path_left), frame_left)
            cv2.imwrite(str(path_right), frame_right)
            print(f"{path_left.name}, {path_right.name}")
    finally:
        ser.close()

    # Сохраняем метаданные съёмки (углы поворота и т.д.) для использования в main.py и других скриптах
    rotation_deg_per_frame = 360.0 / args.count if args.count else 0.0
    metadata = {
        "object_name": OBJECT_NAME,
        "frame_count": args.count,
        "rev_time_sec": args.rev_time,
        "dir_cw": args.dir,
        "delay_after_rotation_sec": args.delay,
        "rotation_deg_per_frame": rotation_deg_per_frame,
        "moves_between_frames": moves_between_frames,
        "left_cam": args.left_cam,
        "right_cam": args.right_cam,
        "width": w_left,
        "height": h_left,
    }
    metadata["use_angle_mode"] = use_angle_mode
    if args.degrees_per_step is not None:
        metadata["degrees_per_step"] = args.degrees_per_step
    meta_file = get_capture_metadata_file()
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Метаданные съёмки: {meta_file}")

    cap_left.release()
    cap_right.release()
    print()
    print(f"Готово. Кадры сохранены в {out_dir}")


if __name__ == "__main__":
    main()
