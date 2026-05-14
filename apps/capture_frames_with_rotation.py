"""
Захват кадров по команде поворота стола по Serial (COM-порт).
MOVE_DEG (угол = 360°/число_позиций) → OK → задержка → снимок.
Опционально можно задать мощность лазера командой LASER_PWM <0..1023>.
По умолчанию задаются exp/gain/bri/con/sat/gamma (см. константы CAM_*); для авто — флаг --no-camera-params.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    raise ImportError("Установите pyserial: pip install pyserial")

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "recorded" / "frames"
CAPTURE_METADATA_FILE = "capture_metadata.json"

CAM_LEFT_INDEX = 0
CAM_RIGHT_INDEX = 1
CAPTURE_WIDTH = 2560
CAPTURE_HEIGHT = 1440
DELAY_AFTER_ROTATION = 0.5
DELAY_BETWEEN_SHOTS = 0.15
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_READ = 120.0

# Параметры камер (как в tune_camera_exposure, слайдеры)
CAM_EXPOSURE = -4.5
CAM_GAIN = 32
CAM_BRIGHTNESS = 0
CAM_CONTRAST = 60
CAM_SATURATION = 60
CAM_GAMMA = 100
STARTUP_WARMUP_FRAMES = 12


def open_camera(index: int) -> cv2.VideoCapture:
    """Открыть камеру через DirectShow, где на Windows стабильнее работают ручные параметры."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def apply_camera_params(cap: cv2.VideoCapture, exposure: float | None) -> None:
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, CAM_BRIGHTNESS)
    cap.set(cv2.CAP_PROP_CONTRAST, CAM_CONTRAST)
    cap.set(cv2.CAP_PROP_SATURATION, CAM_SATURATION)
    cap.set(cv2.CAP_PROP_GAMMA, CAM_GAMMA)


def warmup_cameras(
    cap_left: cv2.VideoCapture,
    cap_right: cv2.VideoCapture,
    frames_count: int,
) -> None:
    for _ in range(max(0, frames_count)):
        cap_left.read()
        cap_right.read()


def send_rotate_serial(
    ser: serial.Serial,
    angle_deg: float,
    rev_time: float = 1.2,
    dir_cw: int = 1,
    timeout_sec: float = SERIAL_TIMEOUT_READ,
) -> bool:
    """Отправить MOVE_DEG <угол> <rev_time> <dir>, дождаться OK. dir: 1=CW, 0=CCW."""
    ser.reset_input_buffer()
    cmd = f"MOVE_DEG {angle_deg:.6f} {rev_time:.2f} {dir_cw}\n"
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


def send_serial_ok(
    ser: serial.Serial,
    cmd: str,
    timeout_sec: float = 3.0,
) -> bool:
    """Отправить serial-команду и дождаться строки OK."""
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode("ascii"))
    ser.flush()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        line = ser.readline()
        if not line:
            continue
        text = line.decode("ascii", errors="ignore").strip()
        if text == "OK":
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Захват кадров: Serial MOVE_DEG (360°/N) → OK → задержка → снимок. Сохраняет в outputs/recorded/frames."
    )
    parser.add_argument("--port", type=str, default="COM9",
                        help="COM-порт платы (Windows: COM3, Linux: /dev/ttyUSB0)")
    parser.add_argument("--list-ports", action="store_true",
                        help="Показать доступные COM-порты и выйти")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="Скорость Serial (по умолчанию 115200)")
    parser.add_argument("--count", type=int, default=40,
                        help="Количество позиций за полный оборот: угол шага = 360°/count")
    parser.add_argument("--shots-per-position", type=int, default=5,
                        help="Сколько снимков делать на каждой позиции (по умолчанию 3)")
    parser.add_argument("--rev-time", type=float, default=1.4,
                        help="Время одного шага поворота, сек (меньше = быстрее)")
    parser.add_argument("--dir", type=int, default=1, choices=(0, 1),
                        help="Направление: 1=по часовой (CW), 0=против (CCW)")
    parser.add_argument("--delay", type=float, default=DELAY_AFTER_ROTATION,
                        help="Задержка после OK перед снимком, сек")
    parser.add_argument("--shot-delay", type=float, default=DELAY_BETWEEN_SHOTS,
                        help="Задержка между снимками в одной позиции, сек")
    parser.add_argument("--laser-pwm", type=int, default=440,
                        help="Мощность лазера 0..1023. Если задано, перед съемкой отправляется LASER_PWM, а в конце лазер выключается.")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Папка для кадров (по умолчанию {OUTPUT_DIR})")
    parser.add_argument("--left-cam", type=int, default=CAM_LEFT_INDEX, help="Индекс левой камеры")
    parser.add_argument("--right-cam", type=int, default=CAM_RIGHT_INDEX, help="Индекс правой камеры")
    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT, help="Высота кадра")
    parser.add_argument("--left-exposure", type=float, default=None, help="Экспозиция левой (-13..0). Не задано = CAM_EXPOSURE.")
    parser.add_argument("--right-exposure", type=float, default=None, help="Экспозиция правой (-13..0). Не задано = CAM_EXPOSURE.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=STARTUP_WARMUP_FRAMES,
        help="Сколько пар кадров выбросить после установки manual-параметров.",
    )
    parser.add_argument("--no-camera-params", action="store_true", help="Не задавать exp/gain/bri/con/sat/gam (оставить авто)")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("Аргумент --count должен быть >= 1.")
    if args.laser_pwm is not None and not (0 <= args.laser_pwm <= 1023):
        raise SystemExit("Аргумент --laser-pwm должен быть в диапазоне 0..1023.")

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

    cap_left = open_camera(args.left_cam)
    cap_right = open_camera(args.right_cam)
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

    exp_left = args.left_exposure if args.left_exposure is not None else (None if args.no_camera_params else CAM_EXPOSURE)
    exp_right = args.right_exposure if args.right_exposure is not None else (None if args.no_camera_params else CAM_EXPOSURE)

    if not args.no_camera_params:
        apply_camera_params(cap_left, exp_left)
        apply_camera_params(cap_right, exp_right)
        warmup_cameras(cap_left, cap_right, args.warmup_frames)
        apply_camera_params(cap_left, exp_left)
        apply_camera_params(cap_right, exp_right)
    elif exp_left is not None or exp_right is not None:
        apply_camera_params(cap_left, exp_left)
        apply_camera_params(cap_right, exp_right)
        warmup_cameras(cap_left, cap_right, args.warmup_frames)
        apply_camera_params(cap_left, exp_left)
        apply_camera_params(cap_right, exp_right)

    w_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Разрешение: левая {w_left}x{h_left}, правая {w_right}x{h_right}")
    if not args.no_camera_params:
        print(
            f"Камеры: exp={exp_left}/{exp_right} gain={CAM_GAIN} "
            f"bri={CAM_BRIGHTNESS} con={CAM_CONTRAST} sat={CAM_SATURATION} gam={CAM_GAMMA}"
        )
    elif exp_left is not None or exp_right is not None:
        print(f"Экспозиция: левая {exp_left}, правая {exp_right}")
    else:
        print("Экспозиция: авто (не задано)")

    step_deg = 360.0 / float(args.count)
    base_shots_per_position = max(1, args.shots_per_position)
    has_texture_shot_without_laser = args.laser_pwm is not None
    total_shots_per_position = base_shots_per_position + (1 if has_texture_shot_without_laser else 0)
    total_frames = args.count * total_shots_per_position
    print("Режим: Serial MOVE_DEG → OK → задержка → серия снимков.")
    print(
        f"  Порт: {args.port}, позиций: {args.count}, снимков/позицию: {total_shots_per_position}, "
        f"всего кадров: {total_frames}"
    )
    print(f"  Угол на позицию: {step_deg:.4f}° (360°/{args.count}), время поворота: {args.rev_time} сек, направление: {'CW' if args.dir == 1 else 'CCW'}")
    print(f"  Задержка после OK: {args.delay} сек, между снимками: {args.shot_delay} сек")
    if args.laser_pwm is not None:
        print(f"  Лазер PWM: {args.laser_pwm}/1023")
        print(
            f"  Режим лазера: 1-й снимок без лазера + {base_shots_per_position} снимков с лазером на позицию"
        )
    print(f"  Сохранение: {out_dir}")
    print()

    try:
        if args.laser_pwm is not None:
            print("Отключение лазера перед стартом...", end=" ", flush=True)
            if not send_serial_ok(ser, "LASER_PWM 0"):
                print("не подтверждено (нет OK), продолжаем.")
            else:
                print("OK.")

        print("Включение удержания мотора...", end=" ", flush=True)
        if not send_serial_ok(ser, "HOLD_ON"):
            print("не подтверждено (нет OK), продолжаем.")
        else:
            print("OK.")

        frame_idx = 0
        capture_records = []
        for pos in range(args.count):
            print(f"Позиция {pos + 1}/{args.count}: поворот...", end=" ", flush=True)
            if not send_rotate_serial(ser, angle_deg=step_deg, rev_time=args.rev_time, dir_cw=args.dir):
                print("таймаут (нет OK).")
                continue
            print("OK.", end=" ", flush=True)
            time.sleep(args.delay)
            position_saved = 0
            if has_texture_shot_without_laser:
                print("texture(no laser)...", end=" ", flush=True)
            for shot in range(total_shots_per_position):
                shot_laser_enabled = (not has_texture_shot_without_laser) or (shot > 0)
                ok1, frame_left = cap_left.read()
                ok2, frame_right = cap_right.read()
                if not ok1 or not ok2:
                    print(f"Ошибка чтения с камер на снимке {shot + 1}/{total_shots_per_position}.")
                    break
                # Явная структура имени: позиция + номер снимка внутри позиции.
                path_left = out_dir / f"capture_p{pos:04d}_s{shot:02d}_left.png"
                path_right = out_dir / f"capture_p{pos:04d}_s{shot:02d}_right.png"
                cv2.imwrite(str(path_left), frame_left)
                cv2.imwrite(str(path_right), frame_right)
                capture_records.append(
                    {
                        "frame_idx": int(frame_idx),
                        "position_id": int(pos),
                        "position_num": int(pos + 1),
                        "shot_id": int(shot),
                        "shot_num": int(shot + 1),
                        "laser_enabled": bool(shot_laser_enabled),
                        "left_file": path_left.name,
                        "right_file": path_right.name,
                    }
                )
                frame_idx += 1
                position_saved += 1
                if shot == 0 and has_texture_shot_without_laser:
                    if not send_serial_ok(ser, f"LASER_PWM {args.laser_pwm}"):
                        print("LASER_PWM не подтвержден (нет OK), продолжаем.", end=" ", flush=True)
                    # Даём лазеру стабилизироваться до первого depth-кадра.
                    time.sleep(args.shot_delay)
                    continue
                if shot + 1 < total_shots_per_position:
                    time.sleep(args.shot_delay)
            if has_texture_shot_without_laser:
                send_serial_ok(ser, "LASER_PWM 0")
            print(f"снимков сохранено: {position_saved}/{total_shots_per_position}")
    finally:
        if args.laser_pwm is not None:
            print("Выключение лазера...", end=" ", flush=True)
            if not send_serial_ok(ser, "LASER_PWM 0"):
                print("не подтверждено (нет OK).")
            else:
                print("OK.")
        print("Освобождение мотора...", end=" ", flush=True)
        if not send_serial_ok(ser, "HOLD_OFF"):
            print("не подтверждено (нет OK).")
        else:
            print("OK.")
        ser.close()

    cap_left.release()
    cap_right.release()

    metadata_path = out_dir / CAPTURE_METADATA_FILE
    metadata = {
        "positions_count": int(args.count),
        "shots_per_position": int(total_shots_per_position),
        "laser_shots_per_position": int(base_shots_per_position),
        "texture_shot_without_laser": bool(has_texture_shot_without_laser),
        "texture_shot_number": (1 if has_texture_shot_without_laser else None),
        "depth_shot_numbers": (
            list(range(2, total_shots_per_position + 1))
            if has_texture_shot_without_laser
            else list(range(1, total_shots_per_position + 1))
        ),
        "serial_port": args.port,
        "rotation_rev_time": float(args.rev_time),
        "rotation_dir": int(args.dir),
        "platform_rotation_sign": int(-1 if int(args.dir) == 1 else 1),
        "laser_pwm": None if args.laser_pwm is None else int(args.laser_pwm),
        "degrees_per_position": float(step_deg),
        "frame_width": int(args.width),
        "frame_height": int(args.height),
        "frames": capture_records,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Готово. Кадры сохранены в {out_dir}")
    print(f"Метаданные сохранены: {metadata_path}")


if __name__ == "__main__":
    main()
