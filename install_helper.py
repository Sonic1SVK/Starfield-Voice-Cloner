#!/usr/bin/env python3
"""
Optional PyTorch engines, called by INSTALL.bat (needed only for mode C and the VoiceDesign sample of mode B):

    install_helper.py nvidia          PyTorch CUDA into venv_svc
    install_helper.py rocm-rdna2      AMD RX 6000 (RDNA2): ROCm runtime "TheRock" (vllm-rocm-windows-rdna2, -SkipModel)
    install_helper.py rocm-official   AMD RX 7000 / 9000 (RDNA3/4): official AMD ROCm wheels, Python 3.12 -> venv_rocm

It writes engine.txt, which run.bat reads to start the GUI with the right environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_FILE = HERE / "engine.txt"
BACKEND_DIR = HERE / "tts_backend"

RDNA2_ZIP = "https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2/archive/refs/heads/main.zip"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"          # check pytorch.org for a newer CUDA build

# Official AMD wheels for Windows (RX 7000 / RX 9000), copied from AMD's "PyTorch via PIP installation" page.
# AMD changes the release folder from time to time - if a download fails, open DOC_URL and update these lines.
AMD_REL = "rocm-rel-7.2.1"
AMD_BASE = f"https://repo.radeon.com/rocm/windows/{AMD_REL}/"
AMD_SDK = [AMD_BASE + n for n in (
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "rocm-7.2.1.tar.gz")]
AMD_TORCH = [AMD_BASE + n for n in (
    "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl")]
DOC_URL = "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html"


def run(cmd, **kw) -> int:
    print("> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], **kw)


def pip(py: str, *args) -> int:
    return run([py, "-m", "pip", *args])


def verify_torch(py: str, env=None) -> bool:
    code = "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| cuda', torch.version.cuda, '| GPU available:', torch.cuda.is_available())"
    return run([py, "-c", code], env=env) == 0


def set_engine(name: str) -> None:
    ENGINE_FILE.write_text(name + "\n", encoding="ascii")
    print(f"engine.txt = {name}")


# ---------------------------------------------------------------------------
def nvidia() -> int:
    py = sys.executable
    if pip(py, "install", "torch", "torchaudio", "--index-url", CUDA_INDEX) != 0:
        return 1
    if pip(py, "install", "qwen-tts") != 0:
        return 1
    if not verify_torch(py):
        return 1
    set_engine("svc")
    return 0


def fetch_rdna2_repo() -> Path:
    dest = BACKEND_DIR / "rocm_rdna2"
    if (dest / "INSTALL.ps1").is_file():
        return dest
    BACKEND_DIR.mkdir(exist_ok=True)
    zpath = BACKEND_DIR / "rocm_rdna2.zip"
    print("Downloading the ROCm runtime installer (vllm-rocm-windows-rdna2) ...", flush=True)
    req = urllib.request.Request(RDNA2_ZIP, headers={"User-Agent": "StarfieldVoiceCloner"})
    with urllib.request.urlopen(req, timeout=60) as r, open(zpath, "wb") as f:
        f.write(r.read())
    tmp = BACKEND_DIR / "rocm_rdna2_tmp"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    zpath.unlink(missing_ok=True)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    inner.rename(dest)
    tmp.rmdir()
    return dest


def find_therock() -> Path | None:
    for cand in (os.environ.get("THEROCK_ROOT"), r"C:\TheRock", r"D:\ROCmStack\TheRock"):
        if cand and (Path(cand) / ".venv" / "Scripts" / "python.exe").is_file():
            return Path(cand)
    return None


def rocm_rdna2() -> int:
    if find_therock() is None:
        repo = fetch_rdna2_repo()
        print("\nStarting the ROCm runtime installer (~6 GB download, ~25 GB free space on C: needed, asks for admin rights).")
        print("The LLM model of that project is skipped (-SkipModel), it is not needed here.\n", flush=True)
        run(["cmd", "/c", "INSTALL.bat", "-SkipModel"], cwd=str(repo))
    root = find_therock()
    if root is None:
        print("ERROR: TheRock runtime not found (C:\\TheRock or D:\\ROCmStack\\TheRock). Run INSTALL.bat -SkipModel of "
              "https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2 manually and start this again.")
        return 1
    py = str(root / ".venv" / "Scripts" / "python.exe")
    print(f"\nInstalling this project's packages into {root}\\.venv (torch is kept) ...", flush=True)
    if pip(py, "install", "-r", str(HERE / "requirements.txt"), "--upgrade-strategy", "only-if-needed") != 0:
        print("pip reported a problem. If it tried to replace torch, run:  pip install qwen-tts --no-deps  and install the "
              "remaining packages of requirements.txt by hand (see README, section 4).")
        return 1
    env = dict(os.environ, HSA_OVERRIDE_GFX_VERSION="10.3.1", PYTHONUTF8="1", KMP_DUPLICATE_LIB_OK="TRUE")
    env["PATH"] = f"{root}\\build\\dist\\rocm\\bin;{root}\\.venv\\Scripts;" + env.get("PATH", "")
    if not verify_torch(py, env):
        return 1
    set_engine("rocm_rdna2")
    return 0


def rocm_official() -> int:
    venv = HERE / "venv_rocm"
    if not (venv / "Scripts" / "python.exe").is_file():
        if run(["py", "-3.12", "-m", "venv", str(venv)]) != 0:
            print("ERROR: Python 3.12 is required for AMD's wheels. Install it from https://www.python.org/downloads/ "
                  "(py launcher included) and run this again.")
            return 1
    py = str(venv / "Scripts" / "python.exe")
    pip(py, "install", "--upgrade", "pip")
    if pip(py, "install", "--no-cache-dir", *AMD_SDK) != 0 or pip(py, "install", "--no-cache-dir", *AMD_TORCH) != 0:
        print(f"\nERROR: AMD's download failed. AMD may have published a newer release - open\n  {DOC_URL}\n"
              "and copy the current wheel names into install_helper.py (AMD_REL / AMD_SDK / AMD_TORCH).")
        return 1
    if pip(py, "install", "-r", str(HERE / "requirements.txt"), "--upgrade-strategy", "only-if-needed") != 0:
        return 1
    if not verify_torch(py):
        return 1
    set_engine("rocm_official")
    return 0


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    funcs = {"nvidia": nvidia, "rocm-rdna2": rocm_rdna2, "rocm-official": rocm_official}
    if choice not in funcs:
        sys.exit("usage: install_helper.py nvidia | rocm-rdna2 | rocm-official")
    rc = funcs[choice]()
    print("\nSUCCESS" if rc == 0 else "\nFAILED - see the messages above.")
    sys.exit(rc)
