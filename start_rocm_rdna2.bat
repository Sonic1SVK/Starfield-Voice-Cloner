@echo off
rem ===========================================================================
rem  start_rocm_rdna2.bat - Starfield Voice Cloner on AMD Radeon RX 6000 (RDNA2), Windows
rem
rem  Uses the prebuilt ROCm 7.x "TheRock" runtime installed by
rem  https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2  (INSTALL.bat)
rem  and runs qwen_gui.py with the Python of its venv ^(%%THEROCK_ROOT%%\.venv^).
rem ===========================================================================
setlocal
cd /d "%~dp0"

rem ---- configurable ---------------------------------------------------------
rem Set THEROCK_ROOT here, or as an environment variable before starting this file.
rem If it is empty the default install locations are tried: C:\TheRock, then D:\ROCmStack\TheRock.
rem set "THEROCK_ROOT=D:\ROCmStack\TheRock"
rem ---------------------------------------------------------------------------

if not defined THEROCK_ROOT if exist "C:\TheRock\.venv\Scripts\python.exe" set "THEROCK_ROOT=C:\TheRock"
if not defined THEROCK_ROOT if exist "D:\ROCmStack\TheRock\.venv\Scripts\python.exe" set "THEROCK_ROOT=D:\ROCmStack\TheRock"

if not defined THEROCK_ROOT goto :notfound
if not exist "%THEROCK_ROOT%\.venv\Scripts\python.exe" goto :notfound

rem ROCm runtime DLLs and the venv scripts first on PATH
set "PATH=%THEROCK_ROOT%\build\dist\rocm\bin;%THEROCK_ROOT%\.venv\Scripts;%PATH%"

rem RDNA2 override used by this runtime (rocBLAS / Tensile are built for gfx1031) + UTF-8 console output
set "HSA_OVERRIDE_GFX_VERSION=10.3.1"
set "PYTHONUTF8=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"

rem Same extras the runtime installer sets for its own benchmark ^(only when the folders exist^)
if exist "%THEROCK_ROOT%\build\dist\rocm" set "ROCM_PATH=%THEROCK_ROOT%\build\dist\rocm"
if exist "%THEROCK_ROOT%\build\dist\rocm" set "HIP_PATH=%THEROCK_ROOT%\build\dist\rocm"
if exist "%THEROCK_ROOT%\ROCM_VLLM_RUNTIME\bin\rocblas\library" set "ROCBLAS_TENSILE_LIBPATH=%THEROCK_ROOT%\ROCM_VLLM_RUNTIME\bin\rocblas\library"
if exist "%THEROCK_ROOT%\ROCM_VLLM_RUNTIME\bin" set "PATH=%THEROCK_ROOT%\ROCM_VLLM_RUNTIME\bin;%PATH%"

echo TheRock ROCm runtime : %THEROCK_ROOT%
echo HSA_OVERRIDE_GFX_VERSION=%HSA_OVERRIDE_GFX_VERSION%

"%THEROCK_ROOT%\.venv\Scripts\python.exe" qwen_gui.py %*
if errorlevel 1 pause
endlocal
exit /b %errorlevel%

:notfound
echo.
echo TheRock ROCm runtime not found.
echo Looked in: C:\TheRock and D:\ROCmStack\TheRock
echo Install it first with INSTALL.bat -SkipModel from
echo   https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2
echo or set THEROCK_ROOT ^(environment variable, or the line near the top of this file^).
echo See README.md, section "AMD Radeon RX 6000 on Windows".
echo.
pause
exit /b 1
