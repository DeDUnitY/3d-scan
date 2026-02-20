# OpenCV + CUDA on Win10 (RTX 3070)

Этот гайд для проекта `diplom` и твоего `.venv`.

## 1) Что уже известно

- GPU и драйвер работают (`nvidia-smi` видит RTX 3070).
- Текущий `opencv-python` из pip собран без CUDA (`NVIDIA CUDA` нет в build info).

## 2) Автоскрипт

В репозиторий добавлен:

- `stereo/scripts/build_opencv_cuda_windows.ps1`

Он умеет:

- установить базовые prereqs через `winget` (опционально),
- скачать `opencv` + `opencv_contrib`,
- сконфигурировать CMake для CUDA + Python из `.venv`,
- собрать и установить OpenCV,
- скопировать `cv2.pyd` и OpenCV DLL в твой `.venv`.

## 3) Запуск

Открой **x64 Native Tools Command Prompt for VS 2022**  
или PowerShell, где доступны `cl`, `cmake`, `ninja`, `nvcc`.

### 3.1 Только конфигурация (быстрая проверка)

```powershell
powershell -ExecutionPolicy Bypass -File "c:\Users\ded_unity\PycharmProjects\diplom\stereo\scripts\build_opencv_cuda_windows.ps1" -ConfigureOnly
```

### 3.2 Полная сборка

```powershell
powershell -ExecutionPolicy Bypass -File "c:\Users\ded_unity\PycharmProjects\diplom\stereo\scripts\build_opencv_cuda_windows.ps1"
```

### 3.3 Если нужно поставить инструменты автоматически

```powershell
powershell -ExecutionPolicy Bypass -File "c:\Users\ded_unity\PycharmProjects\diplom\stereo\scripts\build_opencv_cuda_windows.ps1" -InstallPrereqs
```

## 4) Проверка результата

```powershell
& "c:/Users/ded_unity/PycharmProjects/diplom/.venv/Scripts/python.exe" -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2,'cuda') else -1); print('NVIDIA CUDA' in cv2.getBuildInformation())"
```

Ожидаемо:

- `cuda device count > 0`
- `NVIDIA CUDA` = `True`

## 5) Важные заметки

- Сборка OpenCV на Windows может занять 30-90+ минут.
- `backend=1` в `tune_disparity_params.py` даёт ускоренный CUDA-preview (StereoBM).
- Для качества ближе к текущему пайплайну финально сравнивай на CPU SGBM.
