"""
Управление лазером через Serial-команды платы:
  LASER_ON  -> OK
  LASER_OFF -> OK
"""

import argparse
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    raise ImportError("Установите pyserial: pip install pyserial")


SERIAL_BAUD = 115200
SERIAL_TIMEOUT_READ = 5.0


def send_laser_command(
    ser: serial.Serial,
    turn_on: bool,
    timeout_sec: float = SERIAL_TIMEOUT_READ,
) -> bool:
    """Отправить команду LASER_ON/LASER_OFF и дождаться ответа OK."""
    ser.reset_input_buffer()
    command = "LASER_ON\n" if turn_on else "LASER_OFF\n"
    ser.write(command.encode("ascii"))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Включение/выключение лазера на плате через Serial."
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="COM-порт платы (Windows: COM3, Linux: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=SERIAL_BAUD,
        help="Скорость Serial (по умолчанию 115200)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=SERIAL_TIMEOUT_READ,
        help="Таймаут ожидания ответа OK, сек",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Показать доступные COM-порты и выйти",
    )
    parser.add_argument(
        "--laser-on",
        action="store_true",
        help="Включить лазер",
    )
    parser.add_argument(
        "--laser-off",
        action="store_true",
        help="Выключить лазер",
    )
    args = parser.parse_args()

    if args.list_ports:
        for port in serial.tools.list_ports.comports():
            print(f"  {port.device} - {port.description}")
        return

    if args.laser_on == args.laser_off:
        raise SystemExit("Укажите ровно один флаг: --laser-on или --laser-off.")

    turn_on = args.laser_on
    action_name = "включение" if turn_on else "выключение"
    print(f"Открытие Serial {args.port} @ {args.baud}...")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5, write_timeout=5.0)
    except serial.SerialException as exc:
        raise RuntimeError(f"Не удалось открыть порт {args.port}: {exc}") from exc

    try:
        print(f"Команда: {action_name} лазера...")
        ok = send_laser_command(ser, turn_on=turn_on, timeout_sec=args.timeout)
    finally:
        ser.close()

    if not ok:
        raise SystemExit("Нет ответа OK от платы (таймаут).")

    print(f"Готово: лазер {'включен' if turn_on else 'выключен'}.")


if __name__ == "__main__":
    main()
