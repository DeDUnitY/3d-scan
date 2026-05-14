"""
То же, что calibration/swap_stereo_filenames.py, но для записи в outputs/recorded.

Меняет местами имена *_left.* <-> *_right.* (только переименование, три шага).

По умолчанию: outputs/recorded — если есть подпапка frames/, обрабатывается она
(как в apps/main.py: outputs/recorded/frames).

Примеры:
  py apps/swap_recorded_stereo_filenames.py
  py apps/swap_recorded_stereo_filenames.py --dry-run
  py apps/swap_recorded_stereo_filenames.py C:/path/to/outputs/recorded
"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RECORDED_DIR = ROOT_DIR / "outputs" / "recorded"


def resolve_images_dir(raw: Path) -> Path:
    """Если передан корень recorded и есть recorded/frames — используем frames/."""
    raw = raw.resolve()
    frames = raw / "frames"
    if frames.is_dir():
        return frames
    return raw


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
        description="Поменять местами имена стереокадров в outputs/recorded (см. main.py INPUT_DIR)."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=DEFAULT_RECORDED_DIR,
        help=f"Корень recorded или папка с парами (по умолчанию: {DEFAULT_RECORDED_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать переименования, не выполнять",
    )
    args = parser.parse_args()
    raw = args.directory
    if not raw.is_absolute():
        raw = (ROOT_DIR / raw).resolve()
    else:
        raw = raw.resolve()
    if not raw.is_dir():
        raise SystemExit(f"Не папка: {raw}")

    d = resolve_images_dir(raw)
    pairs = collect_pairs(d)
    if not pairs:
        raise SystemExit(f"Пар не найдено в {d}")

    print(f"Обработка: {d}")
    if d != raw:
        print(f"  (указано: {raw}, используется подпапка frames/)")
    print(f"Найдено пар: {len(pairs)}")
    if args.dry_run:
        print("(dry-run)")

    for left, right in pairs:
        swap_one_pair(left, right, dry_run=args.dry_run)

    print("Готово.")


if __name__ == "__main__":
    main()
