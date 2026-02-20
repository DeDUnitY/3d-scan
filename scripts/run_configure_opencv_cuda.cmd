@echo off
setlocal
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" (
  set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%"
)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 (
  echo Failed to initialize VS build environment.
  exit /b 1
)
powershell -ExecutionPolicy Bypass -File "c:\Users\ded_unity\PycharmProjects\diplom\stereo\scripts\build_opencv_cuda_windows.ps1" -ConfigureOnly
exit /b %errorlevel%
