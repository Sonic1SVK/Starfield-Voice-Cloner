@echo off
rem EXPERIMENTAL: builds dist\StarfieldVoiceCloner\StarfieldVoiceCloner.exe for the llama-server edition
rem (modes A and B-clone, no PyTorch). Mode C and the VoiceDesign sample need the normal Python install.
cd /d "%~dp0"
set PYTHONUTF8=1
call "%~dp0_env.bat" || exit /b 1
"%PYEXE%" -m pip install pyinstaller
"%PYEXE%" -m PyInstaller --noconfirm --clean --onedir --windowed --name StarfieldVoiceCloner ^
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all _soundfile_data ^
  --collect-data librosa --collect-submodules librosa --collect-submodules numba ^
  --hidden-import generate_voice --hidden-import finalize_audio --hidden-import common --hidden-import server_manager ^
  --exclude-module torch --exclude-module torchaudio --exclude-module qwen_tts ^
  qwen_gui.py
if errorlevel 1 ( echo Build failed. & pause & exit /b 1 )
copy /y README.md dist\StarfieldVoiceCloner\ >nul
copy /y LICENSE dist\StarfieldVoiceCloner\ >nul
echo.
echo Built: dist\StarfieldVoiceCloner\StarfieldVoiceCloner.exe
echo Work files, output and the downloaded llama.cpp + model are created next to the .exe.
pause
