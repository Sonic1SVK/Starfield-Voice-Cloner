#!/usr/bin/env python3
"""Project self-check: files, Python packages, TTS backend, GPU. Run:  python check_project.py"""
import importlib.util
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OK, BAD, WARN = "[ OK ]", "[FAIL]", "[warn]"
problems = 0


def line(tag, msg):
    global problems
    if tag == BAD:
        problems += 1
    print(f"{tag} {msg}")


def version(mod):
    try:
        from importlib import metadata
        return metadata.version(mod)
    except Exception:
        return "?"


print(f"Starfield Voice Cloner - project check\nFolder: {HERE}\nPython {platform.python_version()} ({sys.executable})\n")

# --- chosen engine
eng = HERE / "engine.txt"
print("engine.txt: " + (eng.read_text().strip() if eng.is_file() else "(not set - run INSTALL.bat)") + "\n")

# --- files
for f in ("qwen_gui.py", "generate_voice.py", "finalize_audio.py", "common.py", "server_manager.py"):
    line(OK if (HERE / f).is_file() else BAD, f"file {f}")
for f in ("run.bat", "_env.bat", "INSTALL.bat", "install_helper.py", "START_TTS_SERVER.bat", "requirements-llama.txt"):
    line(OK if (HERE / f).is_file() else WARN, f"file {f}")

# --- python
if sys.version_info < (3, 9):
    line(BAD, "Python 3.9 or newer is required")
if sys.version_info >= (3, 14):
    line(WARN, "Python 3.14: some audio packages (PyAV, numba) may lack wheels - 3.11 to 3.13 is safest")

# --- packages
need = {"PySide6": "PySide6", "psutil": "psutil", "numpy": "numpy", "soundfile": "soundfile", "librosa": "librosa",
        "faster_whisper": "faster-whisper"}
for mod, pip in need.items():
    if importlib.util.find_spec(mod):
        line(OK, f"{pip} {version(pip)}")
    else:
        line(BAD, f"{pip} is missing  ->  pip install {pip}")

# --- PyTorch (optional)
if importlib.util.find_spec("torch"):
    try:
        import torch
        gpu = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if gpu else "-"
        line(OK if gpu else WARN, f"PyTorch {torch.__version__}; GPU available: {gpu} {name}")
    except Exception as e:
        line(WARN, f"PyTorch found but not usable: {e}")
    line(OK if importlib.util.find_spec("qwen_tts") else WARN,
         "qwen-tts " + ("installed" if importlib.util.find_spec("qwen_tts") else "missing (only for modes C and the VoiceDesign sample)"))
else:
    line(WARN, "PyTorch not installed - fine for llama-server mode (A, B-clone); needed for mode C and the VoiceDesign sample")

# --- llama-server backend
try:
    sys.path.insert(0, str(HERE))
    import server_manager as sm
    for be in ("vulkan", "cuda"):
        exe = sm.find_server_exe(be)
        line(OK if exe else WARN, f"llama.cpp {be} build: " + (str(exe) if exe else "not downloaded yet (downloaded on first 'Start server')"))
    for fn in (sm.MODEL_FILE, sm.MMPROJ_FILE):
        p = sm.MODEL_DIR / fn
        line(OK if p.is_file() else WARN, f"model {fn}: " + (f"{p.stat().st_size / 1e6:.0f} MB" if p.is_file() else "not downloaded yet"))
    if sm.server_alive(sm.PORT):
        line(OK, f"server answers on port {sm.PORT}")
    else:
        line(WARN, f"no server on port {sm.PORT} (press 'Start server' in the GUI or run START_TTS_SERVER.bat)")
    if sm.LOG_FILE.is_file():
        scan = sm.LogScan()
        for ln in sm.LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            scan.feed(ln)
        line(OK if scan.gpu_in_use else WARN, "last server log: " + scan.summary())
    has_nv = sm.has_nvidia()
    line(OK, "Nvidia driver found - Auto backend will use CUDA" if has_nv else "no Nvidia driver found - Auto backend will use Vulkan")
except Exception as e:
    line(BAD, f"server_manager failed: {e}")

# --- Windows GPU monitor probe
if sys.platform == "win32":
    try:
        out = sm.run_powershell(sm.WIN_GPU_ONCE_PS, 40)
        parts = out.splitlines()[-1].split("|") if out else []
        if len(parts) == 4:
            used = float(parts[1]) / 2**30
            total = float(parts[2]) / 2**30
            line(OK, f"GPU monitor: {parts[3]} | load {parts[0]} % | VRAM {used:.1f}" + (f" / {total:.1f} GB" if total else " GB (total unknown)"))
        else:
            line(WARN, f"GPU monitor returned: {out[:200]!r}")
        print("      --- GPU details (send this if the monitor shows wrong values) ---")
        for ln in sm.run_powershell(sm.WIN_GPU_DEBUG_PS, 40).splitlines():
            print("      " + ln)
    except Exception as e:
        line(WARN, f"GPU monitor probe failed: {e}")

# --- write access
try:
    t = HERE / ".write_test"
    t.write_text("x")
    t.unlink()
    line(OK, "folder is writable")
except Exception as e:
    line(BAD, f"folder is not writable: {e}")

state = HERE / "gui_state.json"
if state.is_file():
    try:
        json.loads(state.read_text(encoding="utf-8"))
        line(OK, "gui_state.json readable")
    except Exception:
        line(BAD, "gui_state.json is corrupt - delete it (settings will reset)")

print(f"\n{'All required checks passed.' if not problems else str(problems) + ' problem(s) found - see [FAIL] lines.'}")
sys.exit(1 if problems else 0)
