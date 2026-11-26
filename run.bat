@echo off
rem Starts the GUI with the engine chosen in INSTALL.bat (engine.txt), see _env.bat.
cd /d "%~dp0"
set PYTHONUTF8=1
call "%~dp0_env.bat" || exit /b 1
if /i "%ENGINE%"=="rocm_rdna2" (
  call "%~dp0start_rocm_rdna2.bat" %*
  exit /b %errorlevel%
)
"%PYEXE%" qwen_gui.py %*
if errorlevel 1 pause
