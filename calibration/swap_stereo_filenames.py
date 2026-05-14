"""
Меняет местами имена файлов в стереопарах: *_left.* <-> *_right.*
Содержимое файлов не трогает — только переименование (три шага, без коллизий).

Имеет смысл, если при съёмке перепутаны подписи: в *_left.png лежит кадр правой камеры
и наоборот. После скрипта можно калибровать и гонять main.py без --swap-left-right / SWAP_LEFT_RIGHT.

Имена как в calibrate_stereo_aruco.py: capture_*_left.png / capture_*_right.png,
frame_*_left.png / frame_*_right.png (любое расширение: .png, .jpg, ...).
"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT_DIR / "calibration" / "images_stereo"


def collect_pairs(images_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = Path(images_dir)
    left_files = sorted(images_dir.glob("capture_*_left.*")) + sorted(
        images_dir.glob("frame_*_left.*")
    )
    seen: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    for left_path in left_files:
        right_path = images_dir / left_path.name.replace("_left.", "_right.")
        if not right_path.exists() or left_path == right_path:
            continue
        if left_path.name not in seen:
            seen.add(left_path.name)
            pairs.append((left_path, right_path))
    return pairs


def swap_one_pair(left: Path, right: Path, dry_run: bool) -> None:
    tmp = left.with_name(left.stem + ".__swap_tmp__" + left.suffix)
    if tmp.exists():
        raise RuntimeError(f"В папке уже есть {tmp.name} — удалите вручную и повторите.")

    def doit(old: Path, new: Path) -> None:
        if dry_run:
            print(f"  {old.name} -> {new.name}")
        else:
            old.rename(new)

    if dry_run:
        print(f"Пара: {left.name} <-> {right.name}")
    doit(left, tmp)
    doit(right, left)
    doit(tmp, right)
    if not dry_run:
        print(f"  OK: {left.name} <-> {right.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Поменять местами имена *_left.* и *_right.* в каждой стереопаре."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=DEFAULT_DIR,
        help=f"Папка с парами (по умолчанию: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать переименования, не выполнять",
    )
    args = parser.parse_args()
    d = args.directory.resolve()
    if not d.is_dir():
        raise SystemExit(f"Не папка: {d}")

    pairs = collect_pairs(d)
    if not pairs:
        raise SystemExit(f"Пар не найдено в {d}")

    print(f"Папка: {d}")
    print(f"Найдено пар: {len(pairs)}")
    if args.dry_run:
        print("(dry-run)")

    for left, right in pairs:
        swap_one_pair(left, right, dry_run=args.dry_run)

    print("Готово.")


if __name__ == "__main__":
    main()
