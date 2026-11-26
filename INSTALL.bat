@echo off
rem One-time setup. Creates .\venv_svc (dedicated environment), installs the light packages and lets you
rem choose the engine for your graphics card. The choice is stored in engine.txt and used by run.bat.
cd /d "%~dp0"
set PYTHONUTF8=1

set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY (
  echo Python 3.10 - 3.13 was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH", then run this file again.
  pause
  exit /b 1
)

if not exist "venv_svc\Scripts\python.exe" (
  echo Creating the environment venv_svc ...
  %PY% -m venv venv_svc
  if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)
set "VPY=%~dp0venv_svc\Scripts\python.exe"

"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements-llama.txt
if errorlevel 1 ( echo Package installation failed. & pause & exit /b 1 )

echo.
echo ============================================================================
echo  Which graphics card do you have?
echo.
echo   [1] AMD or NVIDIA - llama-server on Vulkan (no PyTorch). Voice modes A and B-clone.
echo       RECOMMENDED start; the GUI downloads everything else (~3 GB) by itself.
echo   [2] NVIDIA + PyTorch CUDA            (adds mode C and the VoiceDesign sample)
echo   [3] AMD Radeon RX 6000 (RDNA2) + PyTorch ROCm runtime "TheRock"
echo       (~6 GB download, ~25 GB free on C:, runs the installer of vllm-rocm-windows-rdna2)
echo   [4] AMD Radeon RX 7000 / RX 9000 + PyTorch ROCm, official AMD wheels (needs Python 3.12)
echo.
echo  PyTorch is only needed for mode C and the VoiceDesign sample. Options 2-4 also keep
echo  the llama-server engine, so you can compare speeds in benchmarks.csv.
echo ============================================================================
choice /c 1234 /n /m "Choose 1-4: "
set SEL=%errorlevel%

echo svc>engine.txt
if %SEL%==2 "%VPY%" install_helper.py nvidia
if %SEL%==3 "%VPY%" install_helper.py rocm-rdna2
if %SEL%==4 "%VPY%" install_helper.py rocm-official

echo.
"%VPY%" check_project.py
echo.
echo Done. Start the program with run.bat
pause
