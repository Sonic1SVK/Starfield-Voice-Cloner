#!/usr/bin/env python3
"""
One-click TTS backend for Starfield Voice Cloner (Windows).

Downloads (once) and starts llama.cpp's llama-server with Qwen3-TTS Base GGUF.

Backends
  vulkan : works on AMD *and* Nvidia (and Intel) GPUs  - one download
  cuda   : Nvidia only, usually the fastest on GeForce cards
  auto   : cuda when an Nvidia driver is found (nvidia-smi), otherwise vulkan

The log of llama-server is scanned, so the app can tell you whether the model
really runs on the GPU or silently fell back to the CPU.

Standard library only (no pip install needed).
Files are kept in ./tts_backend/
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

BASE = Path(__file__).resolve().parent
BACKEND_DIR = BASE / "tts_backend"
BIN_DIR = BACKEND_DIR / "llama"
MODEL_DIR = BACKEND_DIR / "models"
LOG_FILE = BACKEND_DIR / "server.log"

PORT = 8080
BACKENDS = ("auto", "vulkan", "cuda")
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=12"

HF_REPO = "moonbox-dev/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
MODEL_FILE = "qwen3-tts-1-7b-Q8_0.gguf"          # 1.85 GB
MMPROJ_FILE = "mmproj-qwen3-tts-1-7b-F16.gguf"   # 840 MB (shared by all precisions)

LogFn = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------
def _get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "StarfieldVoiceCloner"})
    return urllib.request.urlopen(req, timeout=timeout)


def _download(url: str, dest: Path, log: LogFn = _print) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _get(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 1.0 and total:
                log(f"   {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
                last = time.time()
    tmp.replace(dest)


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------
def has_nvidia() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        cand = Path(r"C:\Windows\System32\nvidia-smi.exe")
        exe = str(cand) if cand.exists() else None
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    backend = (backend or "auto").lower()
    if backend == "auto":
        return "cuda" if has_nvidia() else "vulkan"
    return backend if backend in ("vulkan", "cuda") else "vulkan"


def find_server_exe(backend: str) -> Optional[Path]:
    d = BIN_DIR / backend
    if d.exists():
        for p in d.rglob("llama-server.exe"):
            return p
    return None


def ensure_llama_server(backend: str, log: LogFn = _print) -> Path:
    exe = find_server_exe(backend)
    if exe:
        return exe
    log(f"Downloading llama.cpp ({backend}, Windows x64) ...")
    with _get(RELEASES_API, 30) as resp:
        releases = json.loads(resp.read().decode("utf-8"))

    if backend == "cuda":
        main_pat = re.compile(r"^llama.*win.*cuda.*x64.*\.zip$", re.I)
    else:
        main_pat = re.compile(r"^llama.*win.*vulkan.*x64.*\.zip$", re.I)
    rt_pat = re.compile(r"^cudart.*win.*cuda.*x64.*\.zip$", re.I)

    def pick(assets, pat):
        found = [a for a in assets if pat.search(a["name"]) and "arm64" not in a["name"].lower()]
        # prefer CUDA 12 (works with more drivers than 13)
        found.sort(key=lambda a: 0 if "cuda-12" in a["name"].lower() or "cuda12" in a["name"].lower() else 1)
        return found[0] if found else None

    for rel in releases:
        if rel.get("draft"):
            continue
        assets = rel.get("assets", [])
        main = pick(assets, main_pat)
        if not main:
            continue
        log(f"   release {rel['tag_name']}: {main['name']}")
        dest_dir = BIN_DIR / backend
        zpath = BACKEND_DIR / main["name"]
        _download(main["browser_download_url"], zpath, log)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest_dir)
        zpath.unlink(missing_ok=True)
        exe = find_server_exe(backend)
        if not exe:
            continue
        if backend == "cuda":
            # CUDA runtime DLLs (cudart/cublas) ship as a separate zip
            rt = pick(assets, rt_pat)
            if rt:
                log(f"   CUDA runtime: {rt['name']}")
                rpath = BACKEND_DIR / rt["name"]
                _download(rt["browser_download_url"], rpath, log)
                with zipfile.ZipFile(rpath) as z:
                    z.extractall(exe.parent)
                rpath.unlink(missing_ok=True)
            else:
                log("   (no separate CUDA runtime zip in this release - "
                    "install the CUDA toolkit or use Vulkan if the server fails to start)")
        return exe
    raise RuntimeError(f"No Windows {backend} build of llama.cpp found on GitHub.")


def ensure_models(log: LogFn = _print):
    out = []
    for fn in (MODEL_FILE, MMPROJ_FILE):
        p = MODEL_DIR / fn
        if not p.exists():
            log(f"Downloading {fn} from Hugging Face ...")
            _download(f"https://huggingface.co/{HF_REPO}/resolve/main/{fn}", p, log)
        out.append(p)
    return out[0], out[1]


def prepare(backend: str = "auto", log: LogFn = _print) -> Dict:
    """Download everything that is missing. Returns what is needed to start the server."""
    used = resolve_backend(backend)
    log(f"Backend: {used}" + (" (auto-detected)" if backend == "auto" else ""))
    exe = ensure_llama_server(used, log)
    model, mmproj = ensure_models(log)
    return {"exe": exe, "model": model, "mmproj": mmproj, "backend": used}


CPU_PORT = 8081


CTX_PER_SLOT = 8192          # a TTS line + reference sample needs far less than the 32768 default


def build_cmd(prep: Dict, port: int = PORT, extra: Optional[List[str]] = None, ngl: int = 99,
              parallel: Optional[int] = None, threads: Optional[int] = None, poll: Optional[int] = None) -> List[str]:
    """parallel = number of slots (= simultaneous requests this server will get). Without it llama-server opens 4
    slots with a 32768 context each, which wastes memory and lets several requests slow each other down.
    threads  = CPU threads (-t / -tb). Without it llama-server uses ALL physical cores, which in hybrid mode
               collides with the second (CPU) server.
    poll     = 0..100, how aggressively idle threads spin-wait. 0 = sleep (leaves the cores to the other server)."""
    cmd = [str(prep["exe"]), "-m", str(prep["model"]), "-mm", str(prep["mmproj"]),
           "--host", "127.0.0.1", "--port", str(port), "-ngl", str(ngl)]
    if parallel:
        cmd += ["-np", str(parallel), "-c", str(CTX_PER_SLOT * parallel)]
    if threads:
        cmd += ["-t", str(threads), "-tb", str(threads)]
    if poll is not None:
        cmd += ["--poll", str(poll)]
    return cmd + (extra or [])


# ---------------------------------------------------------------------------
# server state / GPU check
# ---------------------------------------------------------------------------
def server_alive(port: int = PORT) -> bool:
    """True once llama-server answers /health with 200 (503 = still loading)."""
    try:
        with _get(f"http://127.0.0.1:{port}/health", 2):
            return True
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


class LogScan:
    """Reads llama-server output and decides whether a GPU is really being used."""
    RE_VK = re.compile(r"ggml_vulkan:\s*\d+\s*=\s*(.+?)\s*(?:\(|\|)")
    RE_CUDA = re.compile(r"Device\s+\d+:\s*(.+?),\s*compute capability")
    RE_OFFLOAD = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.I)
    RE_BUF = re.compile(r"\b(Vulkan|CUDA|HIP|ROCm)\d+\b.*(model|compute|KV)\s+buffer", re.I)
    RE_USING = re.compile(r"using device\s+(\w+?)\d*\s*\((.+?)\)", re.I)
    RE_BACKEND = re.compile(r"loaded\s+(\w+)\s+backend", re.I)
    RE_ASSIGN = re.compile(r"layer\s+\d+\s+is assigned to device\s+(?!CPU)(\w+?\d+)", re.I)   # newer llama.cpp builds

    def __init__(self) -> None:
        self.devices: List[str] = []
        self.gpu_buffers = False
        self.gpu_device = False
        self.backends: List[str] = []
        self.offloaded = 0
        self.cpu_only_hint = False
        self.vram_gain = 0          # bytes of VRAM that appeared after the model was loaded (set by the GUI)

    def feed(self, line: str) -> bool:
        """Feed one line; returns True when the line is worth showing to the user."""
        for rx in (self.RE_VK, self.RE_CUDA):
            m = rx.search(line)
            if m and m.group(1) not in self.devices:
                self.devices.append(m.group(1).strip())
                return True
        m = self.RE_USING.search(line)
        if m:
            kind, name = m.group(1), m.group(2).strip()
            if kind.lower() not in ("cpu",):
                self.gpu_device = True
                if name not in self.devices:
                    self.devices.append(name)
            return True
        m = self.RE_ASSIGN.search(line)
        if m:
            self.gpu_device = True
            if m.group(1) not in self.devices:
                self.devices.append(m.group(1))
            return True
        m = self.RE_BACKEND.search(line)
        if m:
            if m.group(1) not in self.backends:
                self.backends.append(m.group(1))
            return True
        m = self.RE_OFFLOAD.search(line)
        if m:
            self.offloaded = max(self.offloaded, int(m.group(1)))
            return True
        if self.RE_BUF.search(line):
            self.gpu_buffers = True
        low = line.lower()
        if "error" in low or "failed" in low or "unknown model architecture" in low:
            return True
        return False

    @property
    def gpu_in_use(self) -> bool:
        return self.offloaded > 0 or self.gpu_buffers or self.gpu_device

    def summary(self) -> str:
        if self.gpu_in_use:
            dev = ", ".join(self.devices) if self.devices else "GPU"
            if self.vram_gain and not self.devices:
                dev = f"VRAM +{self.vram_gain / 2**30:.1f} GB"
            return f"GPU acceleration ON ({dev})"
        if self.devices:
            return f"WARNING: GPU found ({', '.join(self.devices)}) but the model seems to run on the CPU"
        be = f" (loaded backends: {', '.join(self.backends)})" if self.backends else ""
        return "WARNING: no GPU detected in the server log - running on CPU (slow). Try the other backend." + be


# ---------------------------------------------------------------------------
# Windows GPU monitor script (AMD / Nvidia / Intel) - one line per 2 s:  load|vram_used|vram_total|name
# ---------------------------------------------------------------------------
_PS_COMMON = r"""
$ErrorActionPreference = 'SilentlyContinue'
$best = [int64]0; $name = ''
foreach ($k in Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11cd-bfc1-08002be10318}') {
  $p = Get-ItemProperty $k.PSPath
  foreach ($prop in 'HardwareInformation.qwMemorySize', 'HardwareInformation.MemorySize') {
    $v = $p.$prop
    if ($v -is [byte[]]) {
      if ($v.Length -ge 8) { $v = [BitConverter]::ToInt64($v, 0) }
      elseif ($v.Length -ge 4) { $v = [int64][BitConverter]::ToUInt32($v, 0) } else { $v = 0 }
    }
    $v = [int64]$v
    if ($v -gt $best) { $best = $v; $name = [string]$p.DriverDesc }
  }
}
if (-not $name) {
  $vc = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Microsoft|Basic' } |
        Sort-Object AdapterRAM -Descending | Select-Object -First 1
  if ($vc) { $name = $vc.Name }
}
if (-not $name) { $name = 'GPU' }
function Get-GpuLine {
  $eng = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine
  $util = 0
  foreach ($g in ($eng | Group-Object { if ($_.Name -match '(eng_\d+_engtype_\w+)') { $Matches[1] } else { 'x' } })) {
    $sum = ($g.Group | Measure-Object UtilizationPercentage -Sum).Sum
    if ($sum -gt $util) { $util = $sum }
  }
  $mem = (Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Measure-Object DedicatedUsage -Sum).Sum
  return ('{0}|{1}|{2}|{3}' -f [math]::Min([int]$util, 100), [int64]$mem, $best, $name)
}
"""
WIN_GPU_PS = _PS_COMMON + r"""
while ($true) { [Console]::Out.WriteLine((Get-GpuLine)); Start-Sleep -Seconds 2 }
"""
WIN_GPU_ONCE_PS = _PS_COMMON + r"""
[Console]::Out.WriteLine((Get-GpuLine))
"""
WIN_GPU_DEBUG_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
foreach ($k in Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11cd-bfc1-08002be10318}') {
  $p = Get-ItemProperty $k.PSPath
  if ($p.DriverDesc) {
    "{0}: {1}" -f $k.PSChildName, $p.DriverDesc
    foreach ($prop in 'HardwareInformation.qwMemorySize', 'HardwareInformation.MemorySize') {
      $v = $p.$prop
      if ($v -ne $null) { "   {0} = {1} ({2})" -f $prop, ($v -join ','), $v.GetType().Name }
    }
  }
}
Get-CimInstance Win32_VideoController | ForEach-Object { "VideoController: {0}, AdapterRAM={1}" -f $_.Name, $_.AdapterRAM }
$e = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine
"GPUEngine counters: {0} instances, e.g. {1}" -f @($e).Count, (($e | Select-Object -First 1).Name)
"""


def run_powershell(script: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                           text=True, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (r.stdout or "").strip() or (r.stderr or "").strip()
    except Exception as e:
        return f"powershell failed: {e}"


# ---------------------------------------------------------------------------
# VRAM size fallback (Windows): dxdiag report, language independent
# ---------------------------------------------------------------------------
def parse_dxdiag_vram(text: str) -> int:
    """Largest 'Dedicated Memory' in bytes. dxdiag prints three MB lines per adapter - Display, Dedicated, Shared -
    with Display = Dedicated + Shared, which identifies them whatever the Windows language is."""
    vals = [int(v) for ln in text.splitlines() for v in re.findall(r":\s*(\d+)\s*MB\b", ln)]
    best = 0
    for i in range(len(vals) - 2):
        a, b, c = vals[i:i + 3]
        if b > 0 and abs(a - (b + c)) <= 2:
            best = max(best, b)
    return best * 1024 * 1024


def dxdiag_vram_total(timeout: int = 60) -> int:
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "svc_dxdiag.txt")
    try:
        if os.path.exists(path):
            os.remove(path)
        subprocess.Popen(["dxdiag", "/whql:off", "/t", path], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        end = time.time() + timeout
        while time.time() < end:
            if os.path.exists(path) and os.path.getsize(path) > 2000:
                time.sleep(1.0)                                   # let dxdiag finish writing
                raw = Path(path).read_bytes().replace(b"\x00", b"")
                return parse_dxdiag_vram(raw.decode("latin-1", errors="replace"))
            time.sleep(1.0)
    except Exception:
        pass
    return 0


def cached_vram_total(key: str) -> int:
    """Total VRAM for a GPU name; asks dxdiag once and remembers the answer in tts_backend/gpu_vram.json."""
    cache_file = BACKEND_DIR / "gpu_vram.json"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    if cache.get(key):
        return int(cache[key])
    total = dxdiag_vram_total()
    if total:
        cache[key] = total
        try:
            BACKEND_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass
    return total


# ---------------------------------------------------------------------------
# command-line use (START_TTS_SERVER.bat)
# ---------------------------------------------------------------------------
def run_cli(backend: str) -> int:
    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    if server_alive(PORT):
        _print(f"Server already running on port {PORT}.")
        return 0
    prep = prepare(backend)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    for attempt, extra in enumerate(([], ["--no-mmproj-offload"])):
        if extra:
            _print("Server did not start - retrying with --no-mmproj-offload (codec on CPU) ...")
        cmd = build_cmd(prep, PORT, extra)
        _print("Starting: " + " ".join(cmd))
        scan = LogScan()
        proc = subprocess.Popen(cmd, cwd=str(prep["exe"].parent), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                creationflags=flags)

        def reader():
            with open(LOG_FILE, "w", encoding="utf-8") as lf:
                for line in proc.stdout:               # type: ignore[union-attr]
                    lf.write(line); lf.flush()
                    if scan.feed(line):
                        _print("   " + line.rstrip())
        threading.Thread(target=reader, daemon=True).start()

        ready = False
        for _ in range(240):
            if proc.poll() is not None:
                break
            if server_alive(PORT):
                ready = True
                break
            time.sleep(1)
        if ready:
            _print(f"Ready: http://127.0.0.1:{PORT}/v1/audio/speech")
            _print(scan.summary())
            _print("Server is running. Close this window to stop it.")
            proc.wait()
            return 0
        proc.kill()
    _print(f"ERROR: server failed to start. See {LOG_FILE}")
    return 1


if __name__ == "__main__":
    be = sys.argv[1] if len(sys.argv) > 1 else "auto"
    try:
        rc = run_cli(be)
    except Exception as e:
        _print(f"ERROR: {e}")
        rc = 1
    if rc:
        input("Press Enter to close...")
    sys.exit(rc)
