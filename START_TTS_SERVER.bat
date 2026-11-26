@echo off
rem Starts only the Qwen3-TTS backend (llama-server). First run downloads ~3 GB.
rem Backend: auto = CUDA on Nvidia, Vulkan otherwise. Use "vulkan" or "cuda" to force one: START_TTS_SERVER.bat vulkan
cd /d "%~dp0"
set PYTHONUTF8=1
call "%~dp0_env.bat" || exit /b 1
set BE=%1
if not defined BE set BE=auto
"%PYEXE%" server_manager.py %BE%
pause
