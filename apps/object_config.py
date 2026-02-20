"""
Общая конфигурация объекта для всех скриптов: capture, main, tune_cloud_pose_params.

- configs/           — общие конфиги: stereo_calib.npz, camera_params.json, pose_tuned_params.json, sgbm_tuned_params.json
- outputs/<OBJECT_NAME>/
  - frames/          — снимки (capture_*_left.png, capture_*_right.png) и capture_metadata.json
  - reconstruction/  — облако точек, bundle, HTML и т.д.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Папка конфигов: калибровка стереопары, параметры камер (exp/bri/con/gam), и т.д.
CONFIGS_DIR = ROOT_DIR / "configs"

# Имя объекта: меняйте на cube, sphere и т.д. — все скрипты используют эту папку
OBJECT_NAME = "tri"


def get_configs_dir() -> Path:
    """Папка конфигов: configs/ (калибровка, camera_params.json и т.д.)."""
    return CONFIGS_DIR


def get_camera_params_file() -> Path:
    """Файл параметров камер (exp, gain, bri, con, gam), сохраняемый из tune_camera_exposure."""
    return CONFIGS_DIR / "camera_params.json"


def get_calibration_file() -> Path:
    """Файл стереокалибровки (stereo_calib.npz)."""
    return CONFIGS_DIR / "stereo_calib.npz"


def get_pose_params_file() -> Path:
    """Параметры позы камеры/рига (ORBIT_RADIUS, TABLE_CENTER и т.д.) — общие для всех объектов."""
    return CONFIGS_DIR / "pose_tuned_params.json"


def get_sgbm_params_file() -> Path:
    """Параметры SGBM диспаратности (BLOCK_SIZE, WLS и т.д.) — общие для всех объектов."""
    return CONFIGS_DIR / "sgbm_tuned_params.json"


def get_object_dir() -> Path:
    """Корневая папка объекта: outputs/<OBJECT_NAME>/"""
    return ROOT_DIR / "outputs" / OBJECT_NAME


def get_frames_dir() -> Path:
    """Папка снимков: outputs/<OBJECT_NAME>/frames/"""
    return get_object_dir() / "frames"


def get_reconstruction_dir() -> Path:
    """Папка реконструкции: outputs/<OBJECT_NAME>/reconstruction/"""
    return get_object_dir() / "reconstruction"


def get_capture_metadata_file() -> Path:
    """Файл с метаданными съёмки (углы поворота, количество кадров и т.д.)."""
    return get_frames_dir() / "capture_metadata.json"
