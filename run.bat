@echo off
rem Windows launcher: activates the virtual environment and starts the GUI.
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" goto :venv
if exist ".venv\Scripts\activate.bat" goto :dotvenv
echo No virtual environment found (venv\ or .venv\).
echo Create one first:  py -m venv venv  then  venv\Scripts\activate
echo Then install PyTorch for your hardware and: pip install -r requirements.txt  (see README.md)
pause
exit /b 1

:venv
call "venv\Scripts\activate.bat"
goto :run

:dotvenv
call ".venv\Scripts\activate.bat"

:run
python qwen_gui.py %*
if errorlevel 1 pause
