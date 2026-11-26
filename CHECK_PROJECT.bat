@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call "%~dp0_env.bat" || exit /b 1
"%PYEXE%" check_project.py
pause
