@echo off
rem Chooses the Python interpreter for the launchers (sets PYEXE). engine.txt (written by INSTALL.bat):
rem   svc            -> venv_svc            (llama-server, optionally with Nvidia PyTorch)
rem   rocm_official  -> venv_rocm          (AMD RX 7000/9000, official wheels)
rem   rocm_rdna2     -> handled by run.bat through start_rocm_rdna2.bat (TheRock runtime)
rem Without engine.txt: venv_svc, then the "python" of the console (e.g. conda), then old venv\ / .venv\
set "PYEXE="
set "ENGINE=svc"
if exist "%~dp0engine.txt" set /p ENGINE=<"%~dp0engine.txt"
if /i "%ENGINE%"=="rocm_official" if exist "%~dp0venv_rocm\Scripts\python.exe" set "PYEXE=%~dp0venv_rocm\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0venv_svc\Scripts\python.exe" set "PYEXE=%~dp0venv_svc\Scripts\python.exe"
if not defined PYEXE (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    echo %%P | find /i "WindowsApps" >nul
    if errorlevel 1 if not defined PYEXE set "PYEXE=%%P"
  )
)
if not defined PYEXE if exist "%~dp0venv\Scripts\python.exe" set "PYEXE=%~dp0venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE (
  echo No Python found. Run INSTALL.bat first.
  pause
  exit /b 1
)
echo Engine: %ENGINE%  ^|  Python: %PYEXE%
exit /b 0
