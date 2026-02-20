param(
    [string]$WorkspaceRoot = "c:\Users\ded_unity\PycharmProjects\diplom",
    [string]$OpenCvVersion = "4.13.0",
    [string]$CudaArchBin = "8.6",
    [switch]$InstallPrereqs,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = "Stop"

function Write-Section($msg) {
    Write-Host ""
    Write-Host "==== $msg ====" -ForegroundColor Cyan
}

function Ensure-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found in PATH."
    }
}

function Run($cmd) {
    Write-Host ">> $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

$workspace = Resolve-Path $WorkspaceRoot
$venvPython = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Python venv not found: $venvPython"
}

$thirdParty = Join-Path $workspace "stereo\third_party"
$opencvSrc = Join-Path $thirdParty "opencv"
$opencvContribSrc = Join-Path $thirdParty "opencv_contrib"
$buildDir = Join-Path $workspace "stereo\build\opencv_cuda"
$installDir = Join-Path $workspace "stereo\build\opencv_cuda_install"

if (-not (Test-Path $thirdParty)) {
    New-Item -ItemType Directory -Path $thirdParty | Out-Null
}

if ($InstallPrereqs) {
    Write-Section "Installing build prerequisites with winget"
    Run "winget install --id Kitware.CMake --silent --accept-package-agreements --accept-source-agreements"
    Run "winget install --id Ninja-build.Ninja --silent --accept-package-agreements --accept-source-agreements"
    Run "winget install --id Microsoft.VisualStudio.2022.BuildTools --override `"--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`" --accept-package-agreements --accept-source-agreements"
    Write-Host "If CUDA Toolkit is missing, install it manually from NVIDIA (or via winget if available)." -ForegroundColor Yellow
}

Write-Section "Checking required commands"
Ensure-Command "git"
Ensure-Command "cmake"
Ensure-Command "ninja"
Ensure-Command "cl"
Ensure-Command "nvcc"

Write-Section "Reading Python paths from venv"
$pyInfoJson = & $venvPython -c "import json,sys,sysconfig,site; print(json.dumps({'exe':sys.executable,'include':sysconfig.get_path('include'),'platlib':site.getsitepackages()[0],'base_prefix':sys.base_prefix,'ver':f'{sys.version_info.major}{sys.version_info.minor}'}))"
$pyInfo = $pyInfoJson | ConvertFrom-Json
$pythonExe = $pyInfo.exe
$pythonInclude = $pyInfo.include
$pythonSite = $pyInfo.platlib
$pythonLib = $null

if (-not (Test-Path $pythonInclude)) { throw "Python include path not found: $pythonInclude" }
if (-not (Test-Path $pythonSite)) { throw "Python site-packages path not found: $pythonSite" }

# Python import library is usually in the base interpreter, not in venv.
$pyVer = [string]$pyInfo.ver
$basePrefix = [string]$pyInfo.base_prefix
$candidateBase = Join-Path $basePrefix "libs\python$pyVer.lib"
$candidateVenv = Join-Path (Split-Path $pythonExe -Parent | Split-Path -Parent) "libs\python$pyVer.lib"
if (Test-Path $candidateBase) {
    $pythonLib = $candidateBase
} elseif (Test-Path $candidateVenv) {
    $pythonLib = $candidateVenv
}
if (-not (Test-Path $pythonLib)) {
    throw "Python import library not found. Checked: $candidateBase and $candidateVenv"
}

# CMake parses backslashes in some string contexts; use forward slashes.
$pythonExeCMake = ($pythonExe -replace "\\", "/")
$pythonIncludeCMake = ($pythonInclude -replace "\\", "/")
$pythonLibCMake = ($pythonLib -replace "\\", "/")
$pythonSiteCMake = ($pythonSite -replace "\\", "/")
$opencvSrcCMake = ($opencvSrc -replace "\\", "/")
$opencvContribModulesCMake = ((Join-Path $opencvContribSrc "modules") -replace "\\", "/")
$buildDirCMake = ($buildDir -replace "\\", "/")
$installDirCMake = ($installDir -replace "\\", "/")

Write-Section "Cloning OpenCV repositories"
if (-not (Test-Path $opencvSrc)) {
    Run "git clone --branch $OpenCvVersion --depth 1 https://github.com/opencv/opencv.git `"$opencvSrc`""
}
if (-not (Test-Path $opencvContribSrc)) {
    Run "git clone --branch $OpenCvVersion --depth 1 https://github.com/opencv/opencv_contrib.git `"$opencvContribSrc`""
}

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir | Out-Null
}

Write-Section "Configuring CMake (CUDA + Python)"
$cmakeArgs = @(
    "-S `"$opencvSrcCMake`"",
    "-B `"$buildDirCMake`"",
    "-G Ninja",
    "-D CMAKE_BUILD_TYPE=Release",
    "-D CMAKE_INSTALL_PREFIX=`"$installDirCMake`"",
    "-D OPENCV_EXTRA_MODULES_PATH=`"$opencvContribModulesCMake`"",
    "-D BUILD_TESTS=OFF",
    "-D BUILD_PERF_TESTS=OFF",
    "-D BUILD_EXAMPLES=OFF",
    "-D BUILD_DOCS=OFF",
    "-D BUILD_opencv_world=OFF",
    "-D WITH_CUDA=ON",
    "-D CUDA_ARCH_BIN=$CudaArchBin",
    "-D WITH_CUBLAS=ON",
    "-D CUDA_FAST_MATH=ON",
    "-D ENABLE_FAST_MATH=ON",
    "-D WITH_CUDNN=OFF",
    "-D BUILD_opencv_python3=ON",
    "-D PYTHON3_EXECUTABLE=`"$pythonExeCMake`"",
    "-D PYTHON3_INCLUDE_DIR=`"$pythonIncludeCMake`"",
    "-D PYTHON3_LIBRARY=`"$pythonLibCMake`"",
    "-D PYTHON3_PACKAGES_PATH=`"$pythonSiteCMake`"",
    "-D CMAKE_CUDA_ARCHITECTURES=86"
)
Run ("cmake " + ($cmakeArgs -join " "))

if ($ConfigureOnly) {
    Write-Section "Configure-only mode complete"
    exit 0
}

Write-Section "Building OpenCV (this takes a long time)"
Run "cmake --build `"$buildDir`" --config Release -j 8"

Write-Section "Installing OpenCV"
Run "cmake --install `"$buildDir`" --config Release"

Write-Section "Deploying cv2 module into venv"
$cv2SearchRoots = @(
    $installDir,
    (Join-Path $workspace ".venv\cv2"),
    $pythonSite
)
$cv2Pyd = $null
foreach ($root in $cv2SearchRoots) {
    if (Test-Path $root) {
        $cv2Pyd = Get-ChildItem -Path $root -Recurse -Filter "cv2*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cv2Pyd) { break }
    }
}
if ($cv2Pyd) {
    Copy-Item $cv2Pyd.FullName (Join-Path $pythonSite "cv2.pyd") -Force
} else {
    Write-Host "cv2*.pyd was not found in expected locations; skip explicit copy." -ForegroundColor Yellow
}

Write-Section "Copying OpenCV DLLs into venv\Scripts"
$dllCandidates = Get-ChildItem -Path $installDir -Recurse -Filter "opencv_*.dll"
$venvScripts = Join-Path (Split-Path $pythonExe -Parent) ""
foreach ($dll in $dllCandidates) {
    Copy-Item $dll.FullName (Join-Path $venvScripts $dll.Name) -Force
}

Write-Section "Validation"
& $venvPython -c "import cv2; print('cv2', cv2.__version__); print('has_cuda', hasattr(cv2,'cuda')); print('cuda_devices', cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2,'cuda') else -1); print('cuda_enabled_in_build', 'NVIDIA CUDA' in cv2.getBuildInformation())"

Write-Host ""
Write-Host "Done. If cuda_devices > 0, your OpenCV CUDA build is active." -ForegroundColor Green
