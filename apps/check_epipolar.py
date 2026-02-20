"""
Проверка качества ректификации: эписопарные линии должны быть горизонтальны.
Сохраняет side-by-side ректифицированные кадры с горизонтальными линиями —
если ректификация корректна, соответствия на правом снимке лежат на тех же строках.
"""
from pathlib import Path
import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from object_config import get_calibration_file, get_frames_dir


def load_calib(path):
    d = np.load(path, allow_pickle=True)
    return {
        "K1": d["camera_matrix_left"], "D1": d["dist_coeffs_left"],
        "K2": d["camera_matrix_right"], "D2": d["dist_coeffs_right"],
        "R1": d["R1"], "R2": d["R2"], "P1": d["P1"], "P2": d["P2"],
        "image_size": tuple(int(x) for x in d["image_size"]),
    }


def main():
    calib_file = get_calibration_file()
    if not calib_file.exists():
        print(f"Calibration not found: {calib_file}")
        return
    calib = load_calib(calib_file)
    frames_dir = get_frames_dir()
    left_p = frames_dir / "capture_0000_left.png"
    right_p = frames_dir / "capture_0000_right.png"
    if not left_p.exists() or not right_p.exists():
        print(f"Frames not found in {frames_dir}")
        return

    size = calib["image_size"]
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_32FC1
    )

    left = cv2.imread(str(left_p))
    right = cv2.imread(str(right_p))
    if left.shape[:2][::-1] != size:
        left = cv2.resize(left, size)
    if right.shape[:2][::-1] != size:
        right = cv2.resize(right, size)

    rl = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rr = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    # Рисуем горизонтальные линии — после ректификации соответствия на той же строке
    h, w = rl.shape[:2]
    for y in range(100, h, 150):
        cv2.line(rl, (0, y), (w, y), (0, 255, 0), 2)
        cv2.line(rr, (0, y), (w, y), (0, 255, 0), 2)

    out = np.hstack([rl, rr])
    out_path = ROOT / "outputs" / "tri" / "reconstruction" / "epipolar_check.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"Saved: {out_path}")
    print("Если ректификация верна — соответствия слева и справа на одной горизонтали.")


if __name__ == "__main__":
    main()
