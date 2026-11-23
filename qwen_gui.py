#!/usr/bin/env python3
"""Starfield voice cloner GUI (Qwen3-TTS) - Linux and Windows, NVIDIA CUDA / AMD ROCm / CPU.

Run inside the virtual environment:  python qwen_gui.py     (or ./run.sh, run.bat, start_rocm_rdna2.bat)
Needs generate_voice.py, finalize_audio.py and common.py next to this file.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from common import (BASE, HSA_VALUES, apply_export_to_manifest, auto_hsa_candidates, find_emotion_refs,
                    load_manifest, parse_export_all, preview_export, safe_component, save_manifest, scope_paths)

os.environ.setdefault("HF_HOME", str(BASE / "hf-cache"))     # model cache (Qwen3-TTS, faster-whisper)

import numpy as np
import psutil
import soundfile as sf
from PySide6.QtCore import (QLocale, QProcess, QProcessEnvironment, QStandardPaths, QThread, QTimer, QUrl, Qt,
                            Signal)
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QCompleter, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)
try:
    from PySide6.QtMultimedia import QSoundEffect
except Exception:                      # without QtMultimedia the system player is used
    QSoundEffect = None

STATE_PATH = BASE / "gui_state.json"
MAX_ATTEMPTS = 3         # how many times silent/broken files are regenerated
ETA_WINDOW = 16          # ETA is computed from the last N finished files
PHYS = psutil.cpu_count(logical=False) or 4
GPU_THREADS = 2          # CPU threads per GPU worker (kernel launches)
REF_MIN, REF_MAX, REF_TARGET = 6.0, 12.0, 8.5     # reference sample length in seconds
REF_POOL = 120           # how many closest candidates are scored for cleanliness
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
EXPORT_NAME_KEYS = ("export_ck", "dialogueexport")     # export files are found by name, any extension (.txt, .xt ...)

SPEAKERS = ("Ryan", "Aiden", "Vivian", "Serena", "Dylan", "Eric", "Ono_Anna", "Sohee", "Uncle_Fu")
VMODES = ("clone", "design", "custom")

# --- voice designer (mode B): dropdown choices -> phrases of the English voice description
GENDERS = ("Female", "Male")
AGES = {"Young 20s": "in their early twenties, youthful",
        "Adult 30s": "in their thirties, mature and steady",
        "Middle-aged 45s": "around 45 years old, seasoned and experienced",
        "Older 65+": "over 65 years old, aged and weathered"}
TIMBRES = {"Natural": "a natural mid-range timbre",
           "Deep": "a deep, low-pitched, resonant timbre",
           "Raspy & Gravelly": "a raspy, gravelly, rough timbre",
           "Soft & Warm": "a soft, warm, gentle timbre",
           "Sharp & Commanding": "a sharp, clear, commanding timbre"}
CADENCES = {"Calm & Neutral": "an even, calm, neutral delivery",
            "Unhinged & Manic": "an unhinged, manic, erratic delivery with sudden shifts in pace and pitch",
            "Cynical Frontier Spacer": "a cynical, dry, world-weary frontier-spacer attitude with sarcastic undertones",
            "Military & Disciplined": "a clipped, disciplined, military delivery",
            "Friendly": "a friendly, upbeat, approachable delivery",
            "Mysterious & Whispery": "a mysterious, hushed, whispery delivery"}
# two expressive sentences: a question, a pause, an exclamation -> the sample captures intonation instead of flat reading
DEFAULT_DESIGN_TEXT = ("Wait... you actually think that's going to work? "
                       "Fine, let's find out - because I am not backing down now!")


def compose_design(gender, age, timbre, cadence):
    return (f"{gender} voice, {AGES.get(age, '')}, with {TIMBRES.get(timbre, '')} and "
            f"{CADENCES.get(cadence, '')}. Clean, close-microphone studio recording.")


def preset(mode):
    if mode == "gpu":
        return dict(gpu_w=1, cpu_w=0, gpu_b=4, cpu_b=1, cpu_t=max(PHYS - GPU_THREADS, 1))
    if mode == "hybrid":
        return dict(gpu_w=1, cpu_w=1, gpu_b=4, cpu_b=1, cpu_t=max(PHYS - GPU_THREADS, 1))
    return dict(gpu_w=0, cpu_w=2, gpu_b=1, cpu_b=1, cpu_t=max(PHYS // 2, 1))


DEFAULT_CFG = dict(mode="gpu", dtype="auto", hsa="auto", **preset("gpu"))
HSA_KEYS = ("hsa_auto", "hsa_none", "hsa_1030", "hsa_1031", "hsa_1032", "hsa_1034")
DTYPE_VALUES = ("auto", "float16", "bfloat16", "float32")

GPU_TEST_CODE = (
    "import torch\n"
    "print('torch', torch.__version__, '| hip', torch.version.hip, '| cuda', torch.version.cuda, "
    "'| available', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    p = torch.cuda.get_device_properties(0)\n"
    "    print(p.name, round(p.total_memory/2**30,1), 'GB', getattr(p, 'gcnArchName', ''))\n"
    "    ok = False\n"
    "    for name in ('float16', 'bfloat16'):\n"
    "        try:\n"
    "            x = torch.randn(1024, 1024).to(device='cuda', dtype=getattr(torch, name))\n"
    "            y = x @ x; torch.cuda.synchronize()\n"
    "            good = bool(torch.isfinite(y.float().cpu()).all())\n"
    "            print(name, 'OK' if good else 'FAILED (NaN/inf)'); ok = ok or good\n"
    "        except Exception as e:\n"
    "            print(name, 'FAILED', type(e).__name__)\n"
    "    if ok: print('GPU_TEST_OK')\n"
)

T = {
    "sk": dict(
        title="Qwen3-TTS – Starfield hlasový klon", export="Export CK:", browse="Prehľadať…", rescan="Znovu hľadať",
        start="▶ Štart", stop="■ Stop", open_out="Otvoriť výstup", lang="Jazyk:", voice="Postava (VOICE TYPE):",
        plugin="Mod / plugin:", no_plugin="(bez FULL PATH – neznámy plugin)",
        total="Replík celkom", done="Hotových", finals="Finálnych", todo="Zostáva", new="Nových", changed="Zmenených",
        progress="Celkový postup", eta="Zostáva", finish_at="hotovo o", per_min="replík/min",
        calc="ETA sa počíta (treba aspoň 3 hotové súbory)…", elapsed="Uplynulo",
        idle="Nečinný", ph_gen="Krok 1: generovanie replík (pokus {a}/{m})", ph_fin="Krok 2: kontrola a finalizácia",
        ph_done="✔ Všetko hotové", ph_problems="⚠ Dokončené, ostali chybné súbory (pozri report.txt)",
        ph_stopped="Zastavené", ph_error="✖ Chyba pri generovaní",
        sysbox="Systém", cpu="CPU", ram="RAM", gpu="GPU", vram="VRAM", na="nedostupné", noload="bez údaja o záťaži",
        worker="Worker {n} · {k}", st_idle="čaká", st_load="načítava model / prvá veta", st_gen="generuje",
        st_done="hotovo", cnt="Hotových týmto workerom: {i}", spd="Priemer: {s:.1f} s/veta", cur="Teraz:",
        rss="RAM procesu", cores="CPU procesu", logbox="Log", processed="✓ spracovaný",
        no_export="Nevybral si žiadny export_ck súbor.", no_voice="V exporte nie je vybraná žiadna postava.",
        need_ref="Vyber referenčnú vzorku (.wav) a over jej text.", need_speaker="Vyber vstavaný hlas (speaker).",
        busy="Beží generovanie. Zastaviť a ukončiť?",
        retry="Nájdené tiché/chybné súbory – opakujem generovanie…",
        none_found="(žiadny export_ck nenájdený – použi Prehľadať…)", no_voices="(v exporte nie je VOICE TYPE)",
        perf="Výkon", mode="Režim:", m_cpu="Len CPU", m_gpu="Čisto GPU (bez CPU workera)", m_hybrid="Hybrid GPU + CPU",
        gpu_w="GPU workerov:", cpu_w="CPU workerov:", gpu_b="Batch GPU:", cpu_b="Batch CPU:",
        cpu_t="Vlákna / CPU worker:", dtype="Presnosť GPU:", dtype_auto="auto (float16 → float32)",
        hsa="AMD ROCm override:", hsa_auto="Auto (detekcia + overenie)",
        hsa_none="Bez override – NVIDIA CUDA, RX 7000/9000, RX 6800/6900 XT (gfx1030)",
        hsa_1030="10.3.0 – gfx1030: RX 6700 XT / 6750 XT / 6700 na štandardných ROCm wheelách (Linux)",
        hsa_1031="10.3.1 – gfx1031: RX 6700 XT / 6750 XT / 6700; TheRock runtime na Windows (štandard)",
        hsa_1032="10.3.2 – Navi 23: RX 6600 / 6600 XT / 6650 XT / 6600M",
        hsa_1034="10.3.4 – Navi 24: RX 6400 / 6500 XT",
        hsa_env="Prostredie už nastavuje HSA_OVERRIDE_GFX_VERSION={v} – Auto ho ponechá.",
        hsa_det="Detegované {g} → Auto skúsi {c} (a overí, že fungujú)",
        hsa_det_none="Detegované {g} → Auto: bez override",
        hsa_nodet="AMD GPU sa nedetegovala (detekcia funguje len na Linuxe) → Auto: bez override",
        test_gpu="Otestovať GPU", testing="Testujem GPU…", est_threads="Vlákien spolu: {n} / {m} fyzických jadier",
        hint="Tip: pri GPU je lepší 1 worker s väčším batchom než viac workerov. Sleduj VRAM.",
        ref_box="Referenčný hlas", pick_dir="Vybrať priečinok…", pick_wav="Vybrať .wav…", play="▶ Prehrať vzorku",
        other="↻ Iná vzorka", ref_text="Text vzorky:", ref_none="(žiadna vzorka)",
        ref_text_ph="Prepis vzorky (Whisper) – oprav, ak sa pomýlil", working="Hľadám vzorku a prepisujem (Whisper)…",
        sample_info="{i}/{n} · {d:.1f} s", sample_single="vlastný súbor · {d:.1f} s",
        ref_loaded="Vzorka uložená pre túto postavu (platí pre všetky mody).",
        emo_found="Emočné vzorky podľa FACE: {names}", emo_none="V priečinku nie sú emočné vzorky (angry.wav, sad.wav, …).",
        emo_prep="Prepisujem emočné vzorky (Whisper)…",
        err_nocand="V priečinku nie je žiadny .wav s dĺžkou 6–12 s.",
        err_import="Chýba faster-whisper. Nainštaluj: pip install faster-whisper",
        err_empty="Whisper nič nerozpoznal, skús „Iná vzorka“.", err_other="Chyba: {e}",
        pick_dir_t="Priečinok s hlasom (.wav)", pick_wav_t="Referenčný .wav",
        vmode="Režim hlasu:", vm_clone="A – Klon z referenčného WAV", vm_design="B – Nový hlas (VoiceDesign → klon)",
        vm_custom="C – Vstavaný hlas + emócie z CK (CustomVoice)",
        design_box="Tvorba hlasu", gender="Pohlavie:", age="Vek:", timbre="Farba hlasu:", cadence="Povaha / prejav:",
        desc="Popis hlasu:", dtext="Text vzorky:", gen_sample="Generate Voice Sample",
        design_working="Generujem vzorku hlasu (model VoiceDesign)…",
        design_ready="Vzorka hotová a nastavená ako referencia. Prehraj ju, v prípade potreby generuj znova a potom Štart.",
        design_fail="Generovanie vzorky zlyhalo – pozri log.", design_need="Zadaj popis hlasu aj text vzorky.",
        custom_box="Vstavaný hlas", speaker="Speaker:", use_notes="Pridať SCRIPT NOTES do inštrukcie",
        custom_hint="Inštrukcia pre každú vetu sa berie zo stĺpca FACE (napr. VeryAngry → „Speak very angrily…“). "
                    "Ryan a Aiden sú rodení anglickí hovoriaci, ostatné hlasy budú po anglicky s prízvukom.",
        clear="Pregenerovať všetko", clear_q="Zmazať všetko vygenerované audio tohto modu a hlasu a začať odznova?",
    ),
    "en": dict(
        title="Qwen3-TTS – Starfield voice cloner", export="CK export:", browse="Browse…", rescan="Rescan",
        start="▶ Start", stop="■ Stop", open_out="Open output", lang="Language:", voice="Character (VOICE TYPE):",
        plugin="Mod / plugin:", no_plugin="(no FULL PATH – unknown plugin)",
        total="Total lines", done="Done", finals="Final", todo="Remaining", new="New", changed="Changed",
        progress="Overall progress", eta="Remaining", finish_at="done at", per_min="lines/min",
        calc="Calculating ETA (needs at least 3 finished files)…", elapsed="Elapsed",
        idle="Idle", ph_gen="Step 1: generating lines (attempt {a}/{m})", ph_fin="Step 2: checking and finalizing",
        ph_done="✔ All done", ph_problems="⚠ Finished, bad files remain (see report.txt)",
        ph_stopped="Stopped", ph_error="✖ Generation error",
        sysbox="System", cpu="CPU", ram="RAM", gpu="GPU", vram="VRAM", na="unavailable", noload="no load info",
        worker="Worker {n} · {k}", st_idle="waiting", st_load="loading model / first line", st_gen="generating",
        st_done="finished", cnt="Done by this worker: {i}", spd="Average: {s:.1f} s/line", cur="Now:",
        rss="Process RAM", cores="Process CPU", logbox="Log", processed="✓ processed",
        no_export="No export_ck file selected.", no_voice="No character selected in the export.",
        need_ref="Pick a reference sample (.wav) and check its text.", need_speaker="Pick a preset speaker.",
        busy="Generation is running. Stop and quit?",
        retry="Silent/bad files found – regenerating…",
        none_found="(no export_ck found – use Browse…)", no_voices="(no VOICE TYPE in export)",
        perf="Performance", mode="Mode:", m_cpu="CPU only", m_gpu="GPU only (no CPU worker)", m_hybrid="Hybrid GPU + CPU",
        gpu_w="GPU workers:", cpu_w="CPU workers:", gpu_b="GPU batch:", cpu_b="CPU batch:",
        cpu_t="Threads / CPU worker:", dtype="GPU precision:", dtype_auto="auto (float16 → float32)",
        hsa="AMD ROCm override:", hsa_auto="Auto (detect + verify)",
        hsa_none="No override – NVIDIA CUDA, RX 7000/9000, RX 6800/6900 XT (gfx1030)",
        hsa_1030="10.3.0 – gfx1030: RX 6700 XT / 6750 XT / 6700 on stock ROCm wheels (Linux)",
        hsa_1031="10.3.1 – gfx1031: RX 6700 XT / 6750 XT / 6700; TheRock runtime on Windows (default)",
        hsa_1032="10.3.2 – Navi 23: RX 6600 / 6600 XT / 6650 XT / 6600M",
        hsa_1034="10.3.4 – Navi 24: RX 6400 / 6500 XT",
        hsa_env="The environment already sets HSA_OVERRIDE_GFX_VERSION={v} – Auto keeps it.",
        hsa_det="Detected {g} → Auto tries {c} (and verifies that it works)",
        hsa_det_none="Detected {g} → Auto: no override",
        hsa_nodet="No AMD GPU detected (detection works on Linux only) → Auto: no override",
        test_gpu="Test GPU", testing="Testing GPU…", est_threads="Total threads: {n} / {m} physical cores",
        hint="Tip: on GPU one worker with a bigger batch beats several workers. Watch VRAM.",
        ref_box="Reference voice", pick_dir="Choose folder…", pick_wav="Choose .wav…", play="▶ Play sample",
        other="↻ Other sample", ref_text="Sample text:", ref_none="(no sample)",
        ref_text_ph="Transcript of the sample (Whisper) – fix it if it got something wrong",
        working="Finding sample and transcribing (Whisper)…",
        sample_info="{i}/{n} · {d:.1f} s", sample_single="custom file · {d:.1f} s",
        ref_loaded="Sample saved for this character (shared by all modes and mods).",
        emo_found="Emotion samples matched to FACE: {names}", emo_none="No emotion samples in the folder (angry.wav, sad.wav, …).",
        emo_prep="Transcribing emotion samples (Whisper)…",
        err_nocand="No .wav of 6–12 s found in the folder.",
        err_import="faster-whisper is missing. Install: pip install faster-whisper",
        err_empty="Whisper recognised nothing, try “Other sample”.", err_other="Error: {e}",
        pick_dir_t="Voice folder (.wav)", pick_wav_t="Reference .wav",
        vmode="Voice mode:", vm_clone="A – Clone from reference WAV", vm_design="B – Design a new voice (VoiceDesign → clone)",
        vm_custom="C – Preset speaker + CK emotions (CustomVoice)",
        design_box="Voice designer", gender="Gender:", age="Age:", timbre="Timbre:", cadence="Personality / cadence:",
        desc="Voice description:", dtext="Sample text:", gen_sample="Generate Voice Sample",
        design_working="Generating the voice sample (VoiceDesign model)…",
        design_ready="Sample generated and set as the reference. Play it, regenerate if needed, then press Start.",
        design_fail="Voice sample generation failed – see the log.", design_need="Enter a voice description and a sample text.",
        custom_box="Preset speaker", speaker="Speaker:", use_notes="Add SCRIPT NOTES to the instruction",
        custom_hint="The per-line instruction comes from the CK FACE column (e.g. VeryAngry → “Speak very angrily…”). "
                    "Ryan and Aiden are native English speakers; the other speakers will speak English with an accent.",
        clear="Regenerate everything", clear_q="Delete all generated audio of this mod/voice and start from scratch?",
    ),
}


# ---------------------------------------------------------------- data helpers
def find_exports():
    """Look for *export_ck* / *DialogueExport* files (depth <= 1) in the usual user folders and next to the scripts."""
    roots = []
    for loc in (QStandardPaths.DownloadLocation, QStandardPaths.DocumentsLocation, QStandardPaths.DesktopLocation):
        roots += [Path(p) for p in QStandardPaths.standardLocations(loc)]
    roots += [Path.home() / "Downloads", BASE]
    found, seen = {}, set()
    for root in roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for dp, dn, files in os.walk(root):
            for name in files:
                if any(key in name.lower() for key in EXPORT_NAME_KEYS):
                    p = Path(dp) / name
                    found[str(p)] = p.stat().st_mtime
            if len(Path(dp).relative_to(root).parts) >= 1:
                dn[:] = []
    return sorted(found.items(), key=lambda kv: kv[1], reverse=True)


def new_voice():
    return {"ref_wav": "", "ref_text": "", "ref_dir": "", "ref_idx": 0, "transcripts": {},
            "vmode": "clone", "speaker": "Ryan", "use_notes": False,
            "design": {"gender": "Female", "age": "Adult 30s", "timbre": "Natural", "cadence": "Calm & Neutral",
                       "desc": "", "text": DEFAULT_DESIGN_TEXT}}


def default_lang():
    return "sk" if QLocale.system().name().lower().startswith(("sk", "cs")) else "en"


def load_state():
    try:
        st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        st = {}
    st.setdefault("voices", {})
    st.setdefault("lang", default_lang())
    st.setdefault("cfg", dict(DEFAULT_CFG))
    st.setdefault("last_voice", "")
    st.setdefault("last_plugin", "")
    return st


def save_state(st):
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def fmt(sec):
    sec = int(max(sec, 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def gb(b):
    return f"{b / 1024 ** 3:.1f} GB"


def count_wavs(directory, valid):
    n = 0
    try:
        with os.scandir(directory) as it:
            for e in it:
                if e.name.endswith(".wav") and e.name[:-4] in valid:
                    n += 1
    except FileNotFoundError:
        pass
    return n


def make_searchable(combo):
    """Editable combo with a contains-filter popup (useful with hundreds of VOICE TYPE values)."""
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    comp = combo.completer()
    comp.setFilterMode(Qt.MatchContains)
    comp.setCaseSensitivity(Qt.CaseInsensitive)
    comp.setCompletionMode(QCompleter.PopupCompletion)


# ---------------------------------------------------------------- GPU monitoring (AMD sysfs / nvidia-smi / torch)
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def amd_sysfs_info():
    """Linux amdgpu driver: load, VRAM, temperature, power. No extra tools needed."""
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            busy = int(Path(d, "gpu_busy_percent").read_text())
            used = int(Path(d, "mem_info_vram_used").read_text())
            tot = int(Path(d, "mem_info_vram_total").read_text())
        except (OSError, ValueError):
            continue
        temp = power = None
        try:
            hw = glob.glob(d + "/hwmon/hwmon*")[0]
            temp = int(Path(hw, "temp1_input").read_text()) / 1000
            power = int(Path(hw, "power1_average").read_text()) / 1e6
        except (OSError, ValueError, IndexError):
            pass
        return dict(name="AMD", busy=busy, used=used, total=tot, temp=temp, power=power)
    return None


def nvidia_smi_info():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4,
            creationflags=NO_WINDOW).stdout.strip().splitlines()
        if not out:
            return None
        name, busy, used, total, temp, power = [x.strip() for x in out[0].split(",")]
        return dict(name=name, busy=num(busy), used=int(float(used) * 2 ** 20), total=int(float(total) * 2 ** 20),
                    temp=num(temp), power=num(power))
    except Exception:
        return None


TORCH_HELPER = """
import sys, time
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(3)
    name = torch.cuda.get_device_name(0)
except Exception:
    sys.exit(3)
while True:
    free, total = torch.cuda.mem_get_info()
    print(total - free, total, name, flush=True)
    time.sleep(1.5)
"""


class GpuMonitor(QThread):
    """Polls GPU stats in the background: amdgpu sysfs -> nvidia-smi -> torch.cuda.mem_get_info() helper process.

    The last one also covers AMD on Windows (TheRock ROCm runtime), where it reports VRAM only.
    """

    def __init__(self):
        super().__init__()
        self.latest = None
        self._stop = False
        self._helper = None

    def stop(self):
        self._stop = True

    def _torch_info(self):
        if self._helper is None:
            if importlib.util.find_spec("torch") is None:
                return None
            self._helper = subprocess.Popen([sys.executable, "-u", "-c", TORCH_HELPER], stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True, creationflags=NO_WINDOW)
        if self._helper.poll() is not None:
            return None
        line = self._helper.stdout.readline().strip()
        try:
            used, total, name = line.split(" ", 2)
            return dict(name=name, busy=None, used=int(used), total=int(total), temp=None, power=None)
        except ValueError:
            return None

    def run(self):
        mode = None
        try:
            while not self._stop:
                info = None
                if mode in (None, "amd"):
                    info = amd_sysfs_info()
                    mode = "amd" if info else mode
                if info is None and mode in (None, "smi"):
                    info = nvidia_smi_info()
                    mode = "smi" if info else mode
                if info is None and mode in (None, "torch"):
                    info = self._torch_info()
                    mode = "torch" if info else mode
                self.latest = info
                if mode == "torch" and info:
                    continue                       # readline() already paces the loop
                self.msleep(1500)
        finally:
            if self._helper and self._helper.poll() is None:
                self._helper.kill()


# ---------------------------------------------------------------- reference sample + Whisper
def quality_score(path, dur):
    """Lower = better. Heuristic: close to 8.5 s, clean (SNR), no clipping, not too quiet, no shouting/reverb."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) < sr or float(np.max(np.abs(x))) < 0.02:
        return 99.0
    fl = int(sr * 0.025)
    n = len(x) // fl
    fr = np.sqrt(np.mean(x[:n * fl].reshape(n, fl) ** 2, axis=1)) + 1e-9
    speech = float(np.percentile(fr, 90))
    noise = float(np.percentile(fr, 10))
    snr = 20 * np.log10(speech / max(noise, 1e-5))
    active = float(np.mean(fr > speech * 0.1))
    clip = float(np.mean(np.abs(x) > 0.98))
    tail = float(np.sqrt(np.mean(x[-int(sr * 0.25):] ** 2)) / max(speech, 1e-6))
    s = abs(dur - REF_TARGET) * 0.5
    s += min(clip * 500, 5)                 # clipping
    s += max(0.0, 25 - snr) * 0.15          # noise / echo in the pauses
    s += max(0.0, 0.6 - active) * 8         # too much silence
    s += max(0.0, speech - 0.18) * 25       # too loud (battle cries)
    s += max(0.0, tail - 0.08) * 10         # reverb tail or cut-off ending
    return s


def rank_folder(folder):
    """.wav files of 6-12 s ordered from the best reference sample to the worst."""
    cands = []
    for dp, _dn, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".wav"):
                p = os.path.join(dp, name)
                try:
                    info = sf.info(p)
                except Exception:
                    continue
                d = info.frames / info.samplerate if info.samplerate else 0
                if REF_MIN <= d <= REF_MAX:
                    cands.append((abs(d - REF_TARGET), d, p))
    cands.sort()
    scored = []
    for _gap, d, p in cands[:REF_POOL]:
        try:
            scored.append((quality_score(p, d), p))
        except Exception:
            continue
    scored.sort()
    return [p for s, p in scored if s < 90]


_WHISPER = None


def whisper_transcribe(path):
    """Audio is loaded with librosa as a 16 kHz mono NumPy array and passed straight to Whisper (no PyAV decoding)."""
    global _WHISPER
    import librosa
    from faster_whisper import WhisperModel
    if _WHISPER is None:
        _WHISPER = WhisperModel("base.en", device="cpu", compute_type="int8")
    audio_16k, _ = librosa.load(path, sr=16000, mono=True)
    segments, _info = _WHISPER.transcribe(audio_16k, language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


class RefWorker(QThread):
    done = Signal(str, str, object, int)      # wav, text, ranked, idx
    texts_done = Signal(object)               # {path: transcript}
    failed = Signal(str)                      # error code or text

    def __init__(self, mode, target, start=0, ranked=None, cache=None):
        super().__init__()
        self.mode, self.target, self.start_idx = mode, target, start
        self.ranked, self.cache = ranked or [], dict(cache or {})

    def run(self):
        try:
            if self.mode == "texts":                    # transcripts for several files (emotion samples)
                self.texts_done.emit({p: self.cache.get(p) or whisper_transcribe(p) for p in self.target})
                return
            if self.mode == "single":
                text = self.cache.get(self.target) or whisper_transcribe(self.target)
                self.done.emit(self.target, text, [], 0)
                return
            ranked = self.ranked
            if self.mode == "scan":
                ranked = rank_folder(self.target)
            if not ranked:
                self.failed.emit("nocand")
                return
            n = len(ranked)
            for step in range(min(n, 6)):
                idx = (self.start_idx + step) % n
                path = ranked[idx]
                text = self.cache.get(path) or whisper_transcribe(path)
                if len(text.strip()) >= 3:
                    self.done.emit(path, text, ranked, idx)
                    return
            self.failed.emit("empty")
        except ImportError:
            self.failed.emit("import")
        except Exception as e:
            self.failed.emit(f"other:{type(e).__name__}: {e}")


class DesignWorker(QThread):
    """Runs generate_voice.py --design-sample (VoiceDesign model) in a child process and streams its log.

    A separate process keeps the GUI free of torch and returns all GPU memory when the sample is done.
    """
    line = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, out_path, instruct, text, env):
        super().__init__()
        self.out_path, self.instruct, self.text, self.env = out_path, instruct, text, env

    def run(self):
        cmd = [sys.executable, "-u", str(BASE / "generate_voice.py"), f"--design-sample={self.out_path}",
               f"--design-instruct={self.instruct}", f"--design-text={self.text}"]
        try:
            p = subprocess.Popen(cmd, cwd=str(BASE), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
            for ln in p.stdout:
                self.line.emit(ln.rstrip())
            rc = p.wait()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        if rc == 0 and os.path.isfile(self.out_path):
            self.done.emit(self.out_path)
        else:
            self.failed.emit(f"exit code {rc}")


# ---------------------------------------------------------------- widgets
class WorkerCard(QGroupBox):
    def __init__(self):
        super().__init__()
        g = QGridLayout(self)
        self.status = QLabel()
        self.cnt = QLabel()
        self.spd = QLabel()
        self.cpu_l = QLabel()
        self.cpu = QProgressBar()
        self.cpu.setFormat("%v %")
        self.ram = QLabel()
        self.cur = QLabel()
        self.cur.setWordWrap(True)
        self.cur.setMinimumHeight(40)
        self.cur.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        for r, w in enumerate((self.status, self.cnt, self.spd, self.cpu_l, self.cpu, self.ram, self.cur)):
            g.addWidget(w, r, 0)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.st = load_state()
        self.lang = self.st.get("lang", "en")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_output)
        self.proc.finished.connect(self.on_finished)
        self.tproc = QProcess(self)
        self.tproc.setProcessChannelMode(QProcess.MergedChannels)
        self.tproc.finished.connect(self.on_gpu_test)
        self.test_queue = []
        self.running = False
        self.stopping = False
        self.step = None
        self.attempt = 1
        self.run_start = 0.0
        self.gen_start = 0.0
        self.buf = ""
        self.pcache = {}
        self.valid = set()
        self.manifest = None
        self.n_new = self.n_changed = 0
        self.phase_key = "idle"
        self.workers, self.cards, self.exports = [], [], []
        self.export_cache = {}                # path -> (mtime, {plugin: {voice: {fn: line}}})
        self.ranked = []                      # ranked reference candidates (in memory)
        self.ref_worker = None
        self.design_worker = None
        self.ref_vt = None
        self.sfx = None
        self.sfx_path = ""
        self.cfg = dict(DEFAULT_CFG)
        self.run = None                       # snapshot of a run: scope, paths, mode, refs ...
        self.gpu_mon = GpuMonitor()
        self.gpu_mon.start()
        self.build_ui()
        self.load_cfg(self.st["cfg"])
        psutil.cpu_percent()
        self.rescan()
        self.retranslate()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    # ---- helpers
    def tr(self, key, /, **kw):
        s = T[self.lang][key]
        return s.format(**kw) if kw else s

    def scope(self):
        """(plugin, voice_type) of the selection, or None."""
        if self.c_plugin.currentIndex() < 0 or self.c_voice.currentIndex() < 0:
            return None
        vt = self.c_voice.currentData()
        if vt is None:
            return None
        return (self.c_plugin.currentData() or "", vt)

    def voice(self):
        sc = self.scope()
        return sc[1] if sc else None

    def vstate(self, vt=None):
        """Per-VOICE TYPE state: reference sample and voice settings, shared by every mod using that VOICE TYPE."""
        vt = vt or self.voice()
        if not vt:
            return new_voice()
        vs = self.st["voices"].setdefault(vt, new_voice())
        for k, v in new_voice().items():
            vs.setdefault(k, v)
        for k, v in new_voice()["design"].items():
            vs["design"].setdefault(k, v)
        return vs

    def paths(self):
        if self.running and self.run:
            return self.run["paths"]
        sc = self.scope()
        return scope_paths(*sc) if sc else None

    def export_data(self):
        return self.get_export(self.combo.currentData())

    def get_export(self, path):
        if not path or not os.path.isfile(path):
            return {}
        mt = os.path.getmtime(path)
        c = self.export_cache.get(path)
        if c and c[0] == mt:
            return c[1]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data = parse_export_all(path)
        except Exception:
            data = {}
        finally:
            QApplication.restoreOverrideCursor()
        self.export_cache[path] = (mt, data)
        return data

    def scope_lines(self):
        sc = self.scope()
        return self.export_data().get(sc[0], {}).get(sc[1], {}) if sc else {}

    # ---- UI
    def build_ui(self):
        c = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(c)
        self.setCentralWidget(scroll)
        v = QVBoxLayout(c)

        r1 = QHBoxLayout()
        self.l_export = QLabel()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(420)
        self.combo.currentIndexChanged.connect(self.export_changed)
        self.b_browse = QPushButton()
        self.b_browse.clicked.connect(self.browse)
        self.b_rescan = QPushButton()
        self.b_rescan.clicked.connect(self.rescan)
        self.l_lang = QLabel()
        self.c_lang = QComboBox()
        self.c_lang.addItem("Slovenčina", "sk")
        self.c_lang.addItem("English", "en")
        self.c_lang.setCurrentIndex(0 if self.lang == "sk" else 1)
        self.c_lang.currentIndexChanged.connect(self.change_lang)
        for w in (self.l_export, self.combo, self.b_browse, self.b_rescan, self.l_lang, self.c_lang):
            r1.addWidget(w)
        r1.setStretch(1, 1)
        v.addLayout(r1)

        r2 = QHBoxLayout()
        self.l_plugin = QLabel()
        self.c_plugin = QComboBox()
        make_searchable(self.c_plugin)
        self.c_plugin.currentIndexChanged.connect(self.plugin_changed)
        self.l_voice = QLabel()
        self.c_voice = QComboBox()
        make_searchable(self.c_voice)
        self.c_voice.currentIndexChanged.connect(self.voice_changed)
        for w in (self.l_plugin, self.c_plugin, self.l_voice, self.c_voice):
            r2.addWidget(w)
        r2.setStretch(1, 1)
        r2.setStretch(3, 1)
        v.addLayout(r2)

        r3 = QHBoxLayout()
        self.l_vmode = QLabel()
        self.c_vmode = QComboBox()
        for key in VMODES:
            self.c_vmode.addItem("", key)
        self.c_vmode.currentIndexChanged.connect(self.vmode_changed)
        r3.addWidget(self.l_vmode)
        r3.addWidget(self.c_vmode, 1)
        v.addLayout(r3)

        # reference voice (modes A and B)
        self.g_ref = QGroupBox()
        gr = QGridLayout(self.g_ref)
        self.b_pdir = QPushButton()
        self.b_pdir.clicked.connect(self.pick_dir)
        self.b_pwav = QPushButton()
        self.b_pwav.clicked.connect(self.pick_wav)
        self.l_refpath = QLabel()
        self.l_refpath.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.l_reft = QLabel()
        self.e_text = QLineEdit()
        self.e_text.editingFinished.connect(self.ref_text_edited)
        self.b_play = QPushButton()
        self.b_play.clicked.connect(lambda: self.play_wav(self.vstate()["ref_wav"]))
        self.b_other = QPushButton()
        self.b_other.clicked.connect(self.next_sample)
        self.l_refinfo = QLabel()
        self.l_refinfo.setStyleSheet("color:gray")
        self.l_emo = QLabel()
        self.l_emo.setStyleSheet("color:gray")
        self.l_emo.setWordWrap(True)
        gr.addWidget(self.b_pdir, 0, 0)
        gr.addWidget(self.b_pwav, 0, 1)
        gr.addWidget(self.l_refpath, 0, 2, 1, 2)
        gr.addWidget(self.l_reft, 1, 0)
        gr.addWidget(self.e_text, 1, 1, 1, 3)
        gr.addWidget(self.b_play, 2, 0)
        gr.addWidget(self.b_other, 2, 1)
        gr.addWidget(self.l_refinfo, 2, 2, 1, 2)
        gr.addWidget(self.l_emo, 3, 0, 1, 4)
        gr.setColumnStretch(3, 1)
        v.addWidget(self.g_ref)

        # voice designer (mode B)
        self.g_design = QGroupBox()
        gd = QGridLayout(self.g_design)
        self.l_gender, self.c_gender = QLabel(), QComboBox()
        self.l_age, self.c_age = QLabel(), QComboBox()
        self.l_timbre, self.c_timbre = QLabel(), QComboBox()
        self.l_cadence, self.c_cadence = QLabel(), QComboBox()
        for combo, values in ((self.c_gender, GENDERS), (self.c_age, AGES), (self.c_timbre, TIMBRES),
                              (self.c_cadence, CADENCES)):
            for val in values:
                combo.addItem(val, val)
            combo.currentIndexChanged.connect(self.design_combo_changed)
        self.l_desc = QLabel()
        self.e_desc = QPlainTextEdit()
        self.e_desc.setFixedHeight(70)
        self.e_desc.textChanged.connect(self.design_text_edited)
        self.l_dtext = QLabel()
        self.e_dtext = QPlainTextEdit()
        self.e_dtext.setFixedHeight(70)
        self.e_dtext.textChanged.connect(self.design_text_edited)
        self.b_gen = QPushButton()
        self.b_gen.clicked.connect(self.generate_design_sample)
        self.l_design_info = QLabel()
        self.l_design_info.setStyleSheet("color:gray")
        self.l_design_info.setWordWrap(True)
        gd.addWidget(self.l_gender, 0, 0); gd.addWidget(self.c_gender, 0, 1)
        gd.addWidget(self.l_age, 0, 2); gd.addWidget(self.c_age, 0, 3)
        gd.addWidget(self.l_timbre, 1, 0); gd.addWidget(self.c_timbre, 1, 1)
        gd.addWidget(self.l_cadence, 1, 2); gd.addWidget(self.c_cadence, 1, 3)
        gd.addWidget(self.l_desc, 2, 0); gd.addWidget(self.e_desc, 2, 1, 1, 3)
        gd.addWidget(self.l_dtext, 3, 0); gd.addWidget(self.e_dtext, 3, 1, 1, 3)
        gd.addWidget(self.b_gen, 4, 0, 1, 2); gd.addWidget(self.l_design_info, 4, 2, 1, 2)
        gd.setColumnStretch(1, 1)
        gd.setColumnStretch(3, 1)
        v.addWidget(self.g_design)

        # preset speaker (mode C)
        self.g_custom = QGroupBox()
        gc = QGridLayout(self.g_custom)
        self.l_speaker = QLabel()
        self.c_speaker = QComboBox()
        for sp in SPEAKERS:
            self.c_speaker.addItem(sp, sp)
        self.c_speaker.currentIndexChanged.connect(self.custom_changed)
        self.k_notes = QCheckBox()
        self.k_notes.toggled.connect(self.custom_changed)
        self.l_custom_hint = QLabel()
        self.l_custom_hint.setStyleSheet("color:gray")
        self.l_custom_hint.setWordWrap(True)
        gc.addWidget(self.l_speaker, 0, 0); gc.addWidget(self.c_speaker, 0, 1)
        gc.addWidget(self.k_notes, 0, 2, 1, 2)
        gc.addWidget(self.l_custom_hint, 1, 0, 1, 4)
        gc.setColumnStretch(3, 1)
        v.addWidget(self.g_custom)

        # performance
        self.g_perf = QGroupBox()
        gp = QGridLayout(self.g_perf)
        self.l_mode = QLabel()
        self.c_mode = QComboBox()
        for key in ("cpu", "gpu", "hybrid"):
            self.c_mode.addItem("", key)
        self.c_mode.currentIndexChanged.connect(self.mode_changed)
        self.l_dtype = QLabel()
        self.c_dtype = QComboBox()
        for d in DTYPE_VALUES:
            self.c_dtype.addItem(d, d)
        self.l_hsa = QLabel()
        self.c_hsa = QComboBox()
        for val in HSA_VALUES:
            self.c_hsa.addItem("", val)
        self.c_hsa.currentIndexChanged.connect(self.update_cfg_hint)
        self.l_hsa_info = QLabel()
        self.l_hsa_info.setStyleSheet("color:gray")
        self.l_hsa_info.setWordWrap(True)
        self.l_gw, self.s_gw = QLabel(), QSpinBox()
        self.l_cw, self.s_cw = QLabel(), QSpinBox()
        self.l_gb, self.s_gb = QLabel(), QSpinBox()
        self.l_cb, self.s_cb = QLabel(), QSpinBox()
        self.l_ct, self.s_ct = QLabel(), QSpinBox()
        self.s_gw.setRange(0, 3)
        self.s_cw.setRange(0, 4)
        self.s_gb.setRange(1, 16)
        self.s_cb.setRange(1, 8)
        self.s_ct.setRange(1, psutil.cpu_count() or 12)
        self.b_test = QPushButton()
        self.b_test.clicked.connect(self.test_gpu)
        self.l_thr = QLabel()
        self.l_hint = QLabel()
        self.l_hint.setStyleSheet("color:gray")
        for s in (self.s_gw, self.s_cw, self.s_gb, self.s_cb, self.s_ct):
            s.valueChanged.connect(self.update_cfg_hint)
        gp.addWidget(self.l_mode, 0, 0); gp.addWidget(self.c_mode, 0, 1)
        gp.addWidget(self.l_dtype, 0, 2); gp.addWidget(self.c_dtype, 0, 3)
        gp.addWidget(self.b_test, 0, 4, 1, 2)
        gp.addWidget(self.l_hsa, 1, 0); gp.addWidget(self.c_hsa, 1, 1, 1, 5)
        gp.addWidget(self.l_hsa_info, 2, 1, 1, 5)
        gp.addWidget(self.l_gw, 3, 0); gp.addWidget(self.s_gw, 3, 1)
        gp.addWidget(self.l_gb, 3, 2); gp.addWidget(self.s_gb, 3, 3)
        gp.addWidget(self.l_cw, 4, 0); gp.addWidget(self.s_cw, 4, 1)
        gp.addWidget(self.l_cb, 4, 2); gp.addWidget(self.s_cb, 4, 3)
        gp.addWidget(self.l_ct, 4, 4); gp.addWidget(self.s_ct, 4, 5)
        gp.addWidget(self.l_thr, 5, 0, 1, 4); gp.addWidget(self.l_hint, 6, 0, 1, 6)
        v.addWidget(self.g_perf)

        r4 = QHBoxLayout()
        self.b_start = QPushButton()
        self.b_start.clicked.connect(self.start_run)
        self.b_stop = QPushButton()
        self.b_stop.clicked.connect(self.stop_run)
        self.b_stop.setEnabled(False)
        self.b_open = QPushButton()
        self.b_open.clicked.connect(self.open_output)
        self.b_clear = QPushButton()
        self.b_clear.clicked.connect(self.clear_generated)
        self.l_phase = QLabel()
        self.l_phase.setStyleSheet("font-weight:bold")
        for w in (self.b_start, self.b_stop, self.b_open, self.b_clear):
            r4.addWidget(w)
        r4.addWidget(self.l_phase, 1)
        v.addLayout(r4)

        self.g_prog = QGroupBox()
        gpr = QGridLayout(self.g_prog)
        self.l_stats = QLabel()
        self.bar = QProgressBar()
        self.bar.setMinimumHeight(26)
        self.l_eta = QLabel()
        self.l_eta.setStyleSheet("font-size:15px")
        gpr.addWidget(self.l_stats, 0, 0)
        gpr.addWidget(self.bar, 1, 0)
        gpr.addWidget(self.l_eta, 2, 0)
        v.addWidget(self.g_prog)

        self.g_sys = QGroupBox()
        gs = QGridLayout(self.g_sys)
        self.l_cpu, self.b_cpu = QLabel(), QProgressBar()
        self.l_ram, self.b_ram = QLabel(), QProgressBar()
        self.l_gpu, self.b_gpu = QLabel(), QProgressBar()
        self.l_vram, self.b_vram = QLabel(), QProgressBar()
        for r, (a, b) in enumerate(((self.l_cpu, self.b_cpu), (self.l_ram, self.b_ram),
                                    (self.l_gpu, self.b_gpu), (self.l_vram, self.b_vram))):
            b.setRange(0, 100)
            gs.addWidget(a, r, 0)
            gs.addWidget(b, r, 1)
        gs.setColumnStretch(1, 1)
        v.addWidget(self.g_sys)

        self.cards_l = QHBoxLayout()
        v.addLayout(self.cards_l)

        self.g_log = QGroupBox()
        gl = QVBoxLayout(self.g_log)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(160)
        self.log.setStyleSheet("font-family:monospace")
        gl.addWidget(self.log)
        v.addWidget(self.g_log, 1)
        self.resize(1060, 1000)

    def retranslate(self):
        self.setWindowTitle(self.tr("title"))
        for w, k in ((self.l_export, "export"), (self.b_browse, "browse"), (self.b_rescan, "rescan"),
                     (self.l_lang, "lang"), (self.l_plugin, "plugin"), (self.l_voice, "voice"),
                     (self.l_vmode, "vmode"), (self.b_start, "start"), (self.b_stop, "stop"),
                     (self.b_open, "open_out"), (self.b_clear, "clear"), (self.l_mode, "mode"),
                     (self.l_dtype, "dtype"), (self.l_hsa, "hsa"), (self.l_gw, "gpu_w"), (self.l_cw, "cpu_w"),
                     (self.l_gb, "gpu_b"), (self.l_cb, "cpu_b"), (self.l_ct, "cpu_t"),
                     (self.b_test, "test_gpu"), (self.l_hint, "hint"), (self.b_pdir, "pick_dir"),
                     (self.b_pwav, "pick_wav"), (self.b_play, "play"), (self.b_other, "other"),
                     (self.l_reft, "ref_text"), (self.l_gender, "gender"), (self.l_age, "age"),
                     (self.l_timbre, "timbre"), (self.l_cadence, "cadence"), (self.l_desc, "desc"),
                     (self.l_dtext, "dtext"), (self.b_gen, "gen_sample"), (self.l_speaker, "speaker"),
                     (self.k_notes, "use_notes"), (self.l_custom_hint, "custom_hint")):
            w.setText(self.tr(k))
        self.e_text.setPlaceholderText(self.tr("ref_text_ph"))
        for i, k in enumerate(("vm_clone", "vm_design", "vm_custom")):
            self.c_vmode.setItemText(i, self.tr(k))
        for i, k in enumerate(("m_cpu", "m_gpu", "m_hybrid")):
            self.c_mode.setItemText(i, self.tr(k))
        for i, k in enumerate(HSA_KEYS):
            self.c_hsa.setItemText(i, self.tr(k))
        self.c_dtype.setItemText(0, self.tr("dtype_auto"))
        self.g_ref.setTitle(self.tr("ref_box"))
        self.g_design.setTitle(self.tr("design_box"))
        self.g_custom.setTitle(self.tr("custom_box"))
        self.g_perf.setTitle(self.tr("perf"))
        self.g_prog.setTitle(self.tr("progress"))
        self.g_sys.setTitle(self.tr("sysbox"))
        self.g_log.setTitle(self.tr("logbox"))
        self.fill_combo()
        self.update_cfg_hint()

    def change_lang(self):
        self.lang = self.c_lang.currentData()
        self.st["lang"] = self.lang
        save_state(self.st)
        self.retranslate()

    # ---- performance configuration
    def get_cfg(self):
        return dict(mode=self.c_mode.currentData(), dtype=self.c_dtype.currentData(), hsa=self.c_hsa.currentData(),
                    gpu_w=self.s_gw.value(), cpu_w=self.s_cw.value(), gpu_b=self.s_gb.value(),
                    cpu_b=self.s_cb.value(), cpu_t=self.s_ct.value())

    def load_cfg(self, c):
        c = {**DEFAULT_CFG, **{k: v for k, v in c.items() if k in DEFAULT_CFG}}
        if c["hsa"] not in HSA_VALUES:                      # old or unknown value (e.g. a boolean) -> back to Auto
            c["hsa"] = "auto"
        widgets = (self.c_mode, self.c_dtype, self.c_hsa, self.s_gw, self.s_cw, self.s_gb, self.s_cb, self.s_ct)
        for w in widgets:
            w.blockSignals(True)
        self.c_mode.setCurrentIndex(max(self.c_mode.findData(c["mode"]), 0))
        self.c_dtype.setCurrentIndex(max(self.c_dtype.findData(c["dtype"]), 0))
        self.c_hsa.setCurrentIndex(max(self.c_hsa.findData(c["hsa"]), 0))
        self.s_gw.setValue(c["gpu_w"]); self.s_cw.setValue(c["cpu_w"])
        self.s_gb.setValue(c["gpu_b"]); self.s_cb.setValue(c["cpu_b"]); self.s_ct.setValue(c["cpu_t"])
        for w in widgets:
            w.blockSignals(False)

    def mode_changed(self):
        p = preset(self.c_mode.currentData())
        for w, k in ((self.s_gw, "gpu_w"), (self.s_cw, "cpu_w"), (self.s_gb, "gpu_b"),
                     (self.s_cb, "cpu_b"), (self.s_ct, "cpu_t")):
            w.blockSignals(True)
            w.setValue(p[k])
            w.blockSignals(False)
        self.update_cfg_hint()

    def update_cfg_hint(self):
        c = self.get_cfg()
        n = c["gpu_w"] * GPU_THREADS + c["cpu_w"] * c["cpu_t"]
        self.l_thr.setText(self.tr("est_threads", n=n, m=PHYS))
        self.l_thr.setStyleSheet("color:#c0392b" if n > PHYS else "")
        self.l_hint.setVisible(c["gpu_w"] > 0)
        for w in (self.l_hsa, self.c_hsa, self.l_dtype, self.c_dtype, self.l_gw, self.s_gw, self.l_gb, self.s_gb):
            w.setEnabled(not self.running)
        cands, gfx = auto_hsa_candidates()
        env_val = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
        if c["hsa"] == "auto":
            if env_val:
                self.l_hsa_info.setText(self.tr("hsa_env", v=env_val))
            elif gfx:
                self.l_hsa_info.setText(self.tr("hsa_det", g=gfx, c=" → ".join(cands)) if cands
                                        else self.tr("hsa_det_none", g=gfx))
            else:
                self.l_hsa_info.setText(self.tr("hsa_nodet"))
        else:
            self.l_hsa_info.setText(f"HSA_OVERRIDE_GFX_VERSION={c['hsa']}" if c["hsa"] != "none" else "")

    def set_cfg_enabled(self, on):
        for w in (self.c_mode, self.c_dtype, self.c_hsa, self.s_gw, self.s_cw, self.s_gb, self.s_cb, self.s_ct,
                  self.combo, self.b_browse, self.b_rescan, self.c_plugin, self.c_voice, self.c_vmode,
                  self.b_pdir, self.b_pwav, self.b_other, self.e_text, self.c_gender, self.c_age, self.c_timbre,
                  self.c_cadence, self.e_desc, self.e_dtext, self.b_gen, self.c_speaker, self.k_notes,
                  self.b_clear):
            w.setEnabled(on)

    def hsa_candidates_for_test(self):
        """Override values the GPU test should try, in order ([''] = run without an override)."""
        setting = self.c_hsa.currentData()
        if setting == "auto":
            if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
                return [os.environ["HSA_OVERRIDE_GFX_VERSION"]]
            cands, _ = auto_hsa_candidates()
            return cands or [""]
        return [""] if setting == "none" else [setting]

    def test_gpu(self):
        if self.tproc.state() != QProcess.NotRunning:
            return
        self.test_queue = self.hsa_candidates_for_test()
        self.log.appendPlainText("--- " + self.tr("testing"))
        self.b_test.setEnabled(False)
        self.run_gpu_test()

    def run_gpu_test(self):
        ov = self.test_queue.pop(0)
        env = QProcessEnvironment.systemEnvironment()
        if ov:
            env.insert("HSA_OVERRIDE_GFX_VERSION", ov)
        elif self.c_hsa.currentData() == "none":
            env.remove("HSA_OVERRIDE_GFX_VERSION")
        env.insert("PYTHONUTF8", "1")
        self.tproc.setProcessEnvironment(env)
        self.log.appendPlainText(f"HSA_OVERRIDE_GFX_VERSION={ov or '(not set)'}")
        self.tproc.start(sys.executable, ["-c", GPU_TEST_CODE])

    def on_gpu_test(self):
        out = bytes(self.tproc.readAllStandardOutput()).decode("utf-8", "replace").strip()
        self.log.appendPlainText(out)
        if "GPU_TEST_OK" not in out and self.test_queue:
            self.run_gpu_test()                       # try the next override candidate
            return
        self.b_test.setEnabled(True)

    # ---- exports, mods (plugins) and VOICE TYPEs
    def rescan(self):
        self.exports = find_exports()
        self.fill_combo()

    def fill_combo(self):
        cur = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        if not self.exports:
            self.combo.addItem(self.tr("none_found"), None)
        proc = self.manifest.get("processed", {}) if self.manifest else {}
        for path, mt in self.exports:
            mark = f"  [{self.tr('processed')}]" if abs(proc.get(path, 0) - mt) < 1 else ""
            self.combo.addItem(f"{datetime.fromtimestamp(mt):%d.%m.%Y %H:%M}  {path}{mark}", path)
        if cur:
            i = self.combo.findData(cur)
            if i >= 0:
                self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        self.export_changed()

    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, self.tr("export"), str(Path.home()),
                                           "All files (*);;Text (*.txt *.xt *.tsv *.csv)")
        if p:
            if p not in dict(self.exports):
                self.exports.insert(0, (p, os.path.getmtime(p)))
            self.fill_combo()
            self.combo.setCurrentIndex(self.combo.findData(p))

    def export_changed(self):
        """Fill the mod/plugin menu (the folder after Sound\\Voice\\ in FULL PATH) from the selected export."""
        data = self.export_data()
        prev = self.c_plugin.currentData() if self.c_plugin.currentIndex() >= 0 else self.st.get("last_plugin")
        self.c_plugin.blockSignals(True)
        self.c_plugin.clear()
        for plugin, voices in sorted(data.items(), key=lambda kv: (-sum(len(x) for x in kv[1].values()), kv[0])):
            n = sum(len(x) for x in voices.values())
            self.c_plugin.addItem(f"{plugin or self.tr('no_plugin')}  ({n})", plugin)
        if not data:
            self.c_plugin.addItem(self.tr("no_voices"), None)
        i = self.c_plugin.findData(prev)
        self.c_plugin.setCurrentIndex(i if i >= 0 else 0)
        self.c_plugin.blockSignals(False)
        self.plugin_changed()

    def plugin_changed(self):
        """Fill the character (VOICE TYPE) menu of the selected mod."""
        plugin = self.c_plugin.currentData() if self.c_plugin.currentIndex() >= 0 else None
        voices = self.export_data().get(plugin, {}) if plugin is not None else {}
        prev = self.c_voice.currentData() if self.c_voice.currentIndex() >= 0 else self.st.get("last_voice")
        self.c_voice.blockSignals(True)
        self.c_voice.clear()
        for vt, lines in sorted(voices.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            self.c_voice.addItem(f"{vt}  ({len(lines)})", vt)
        if not voices:
            self.c_voice.addItem(self.tr("no_voices"), None)
        i = self.c_voice.findData(prev)
        self.c_voice.setCurrentIndex(i if i >= 0 else 0)
        self.c_voice.blockSignals(False)
        self.voice_changed()

    def voice_changed(self):
        if self.running:
            return
        sc = self.scope()
        if sc:
            self.st["last_plugin"], self.st["last_voice"] = sc
            self.manifest = load_manifest(*sc)
        else:
            self.manifest = None
        self.ranked = []
        self.load_voice_ui()
        self.refresh_preview()

    def refresh_preview(self):
        """Compare the export with the scope's manifest (no writes) and refresh the counters."""
        if self.running:
            return self.tick()
        self.valid, self.n_new, self.n_changed = set(), 0, 0
        if self.manifest:
            lines = self.scope_lines()
            self.valid = set(self.manifest["lines"]) | set(lines)
            self.n_new, self.n_changed = preview_export(self.manifest, lines)
        self.tick()

    def open_output(self):
        p = self.paths()
        if p:
            p["final"].mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p["final"])))

    def clear_generated(self):
        p = self.paths()
        if not p or self.running:
            return
        if QMessageBox.question(self, self.tr("title"), self.tr("clear_q")) != QMessageBox.Yes:
            return
        for d in (p["out"], p["final"]):
            if d.is_dir():
                for f in d.glob("*.wav"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
        self.refresh_preview()

    # ---- voice mode UI (A clone / B design / C custom)
    def load_voice_ui(self):
        vs = self.vstate()
        self.c_vmode.blockSignals(True)
        self.c_vmode.setCurrentIndex(max(self.c_vmode.findData(vs["vmode"]), 0))
        self.c_vmode.blockSignals(False)
        self.c_speaker.blockSignals(True)
        self.c_speaker.setCurrentIndex(max(self.c_speaker.findData(vs["speaker"]), 0))
        self.c_speaker.blockSignals(False)
        self.k_notes.blockSignals(True)
        self.k_notes.setChecked(bool(vs["use_notes"]))
        self.k_notes.blockSignals(False)
        d = vs["design"]
        for combo, key in ((self.c_gender, "gender"), (self.c_age, "age"), (self.c_timbre, "timbre"),
                           (self.c_cadence, "cadence")):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(d[key]), 0))
            combo.blockSignals(False)
        self.e_desc.blockSignals(True)
        self.e_desc.setPlainText(d["desc"] or compose_design(d["gender"], d["age"], d["timbre"], d["cadence"]))
        self.e_desc.blockSignals(False)
        self.e_dtext.blockSignals(True)
        self.e_dtext.setPlainText(d["text"] or DEFAULT_DESIGN_TEXT)
        self.e_dtext.blockSignals(False)
        self.l_design_info.setText("")
        self.load_ref_ui()
        self.apply_vmode_visibility()

    def apply_vmode_visibility(self):
        vm = self.c_vmode.currentData()
        self.g_ref.setVisible(vm in ("clone", "design"))
        self.g_design.setVisible(vm == "design")
        self.g_custom.setVisible(vm == "custom")

    def vmode_changed(self):
        if self.voice():
            self.vstate()["vmode"] = self.c_vmode.currentData()
            save_state(self.st)
        self.apply_vmode_visibility()

    def custom_changed(self):
        if self.voice():
            vs = self.vstate()
            vs["speaker"] = self.c_speaker.currentData()
            vs["use_notes"] = self.k_notes.isChecked()
            save_state(self.st)

    def design_combo_changed(self):
        """A dropdown changed -> rebuild the editable voice description."""
        if not self.voice():
            return
        d = self.vstate()["design"]
        d.update(gender=self.c_gender.currentData(), age=self.c_age.currentData(), timbre=self.c_timbre.currentData(),
                 cadence=self.c_cadence.currentData())
        d["desc"] = compose_design(d["gender"], d["age"], d["timbre"], d["cadence"])
        self.e_desc.blockSignals(True)
        self.e_desc.setPlainText(d["desc"])
        self.e_desc.blockSignals(False)
        save_state(self.st)

    def design_text_edited(self):
        if self.voice():
            d = self.vstate()["design"]
            d["desc"] = self.e_desc.toPlainText().strip()
            d["text"] = self.e_dtext.toPlainText().strip()
            save_state(self.st)

    def run_env(self, cfg=None):
        """Environment of the generator processes (TTS_* variables)."""
        c = cfg or self.get_cfg()
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")                    # Windows: log output as UTF-8
        env.insert("PYTHONIOENCODING", "utf-8")
        for k, val in dict(TTS_GPU_WORKERS=c["gpu_w"], TTS_CPU_WORKERS=c["cpu_w"], TTS_GPU_THREADS=GPU_THREADS,
                           TTS_CPU_THREADS=c["cpu_t"], TTS_GPU_BATCH=c["gpu_b"], TTS_CPU_BATCH=c["cpu_b"],
                           TTS_DTYPE=c["dtype"], TTS_HSA=c["hsa"]).items():
            env.insert(k, str(val))
        return env

    def generate_design_sample(self):
        sc = self.scope()
        if not sc:
            return QMessageBox.warning(self, self.tr("title"), self.tr("no_voice"))
        if self.design_worker and self.design_worker.isRunning():
            return
        desc, text = self.e_desc.toPlainText().strip(), self.e_dtext.toPlainText().strip()
        if not desc or not text:
            return QMessageBox.warning(self, self.tr("title"), self.tr("design_need"))
        vs = self.vstate()
        folder = Path(vs["ref_dir"]) if vs["ref_dir"] and os.path.isdir(vs["ref_dir"]) \
            else BASE / "voices" / safe_component(sc[1])
        folder.mkdir(parents=True, exist_ok=True)
        out = str(folder / "design_sample.wav")
        self.ref_vt = sc[1]
        env = self.run_env()
        pyenv = {k: env.value(k) for k in env.keys()}
        self.design_worker = DesignWorker(out, desc, text, pyenv)
        self.design_worker.line.connect(self.log.appendPlainText)
        self.design_worker.done.connect(lambda p, f=str(folder), t=text: self.on_design_done(p, f, t))
        self.design_worker.failed.connect(self.on_design_failed)
        self.ref_busy(True)
        self.l_refinfo.setText(self.tr("design_working"))
        self.l_design_info.setText(self.tr("design_working"))
        self.design_worker.start()

    def on_design_done(self, path, folder, text):
        vs = self.vstate(self.ref_vt)
        vs["ref_wav"], vs["ref_text"] = path, text
        vs["transcripts"][path] = text
        if not vs["ref_dir"]:
            vs["ref_dir"] = folder
        save_state(self.st)
        self.ref_busy(False)
        if self.voice() == self.ref_vt:
            self.load_ref_ui()
            self.l_design_info.setText(self.tr("design_ready"))

    def on_design_failed(self, why):
        self.ref_busy(False)
        self.l_design_info.setText(self.tr("design_fail"))
        self.log.appendPlainText(f"[DESIGN] {why}")
        QMessageBox.warning(self, self.tr("title"), self.tr("design_fail"))

    # ---- reference voice (shared per VOICE TYPE across all mods)
    def load_ref_ui(self):
        vs = self.vstate()
        wav = vs["ref_wav"]
        self.e_text.blockSignals(True)
        self.e_text.setText(vs["ref_text"])
        self.e_text.blockSignals(False)
        self.l_refpath.setText(wav if wav else self.tr("ref_none"))
        self.l_refinfo.setText(self.tr("ref_loaded") if wav and vs["ref_text"] else "")
        self.b_play.setEnabled(bool(wav) and os.path.isfile(wav))
        emo = find_emotion_refs(vs["ref_dir"]) if vs["ref_dir"] else {}
        self.l_emo.setText(self.tr("emo_found", names=", ".join(sorted(emo))) if emo else self.tr("emo_none"))

    def ref_busy(self, busy):
        for w in (self.b_pdir, self.b_pwav, self.b_other, self.b_play, self.e_text, self.c_voice, self.c_plugin,
                  self.b_gen, self.b_start):
            w.setEnabled(not busy)
        if busy:
            self.l_refinfo.setText(self.tr("working"))
        elif not self.running:
            self.b_play.setEnabled(bool(self.vstate()["ref_wav"]))

    def start_ref_worker(self, mode, target, start=0, ranked=None):
        if self.ref_worker and self.ref_worker.isRunning():
            return
        vs = self.vstate()
        self.ref_vt = self.voice()
        self.ref_worker = RefWorker(mode, target, start, ranked, vs["transcripts"])
        self.ref_worker.done.connect(self.on_ref_done)
        self.ref_worker.texts_done.connect(self.on_texts_done)
        self.ref_worker.failed.connect(self.on_ref_failed)
        self.ref_busy(True)
        self.ref_worker.start()

    def pick_dir(self):
        if not self.voice():
            return QMessageBox.warning(self, self.tr("title"), self.tr("no_voice"))
        vs = self.vstate()
        d = QFileDialog.getExistingDirectory(self, self.tr("pick_dir_t"), vs["ref_dir"] or str(Path.home()))
        if d:
            vs["ref_dir"], vs["ref_idx"], self.ranked = d, 0, []
            self.start_ref_worker("scan", d, 0)

    def pick_wav(self):
        if not self.voice():
            return QMessageBox.warning(self, self.tr("title"), self.tr("no_voice"))
        vs = self.vstate()
        start = os.path.dirname(vs["ref_wav"]) if vs["ref_wav"] else (vs["ref_dir"] or str(Path.home()))
        p, _ = QFileDialog.getOpenFileName(self, self.tr("pick_wav_t"), start, "WAV (*.wav *.WAV)")
        if p:
            self.ranked = []
            self.start_ref_worker("single", p)

    def next_sample(self):
        vs = self.vstate()
        if self.ranked:
            self.start_ref_worker("next", "", vs["ref_idx"] + 1, self.ranked)
        elif vs["ref_dir"] and os.path.isdir(vs["ref_dir"]):
            self.start_ref_worker("scan", vs["ref_dir"], vs["ref_idx"] + 1)
        else:
            self.pick_dir()

    def on_ref_done(self, wav, text, ranked, idx):
        vs = self.vstate(self.ref_vt)
        vs["ref_wav"], vs["ref_text"] = wav, text
        vs["transcripts"][wav] = text
        if ranked:
            self.ranked, vs["ref_idx"] = ranked, idx
        save_state(self.st)
        self.ref_busy(False)
        if self.voice() == self.ref_vt:
            self.load_ref_ui()
            try:
                d = sf.info(wav).duration
            except Exception:
                d = 0.0
            self.l_refinfo.setText(self.tr("sample_info", i=idx + 1, n=len(ranked), d=d) if ranked
                                   else self.tr("sample_single", d=d))

    def on_texts_done(self, texts):
        """Emotion-sample transcripts are ready -> continue with Start."""
        vs = self.vstate(self.ref_vt)
        vs["transcripts"].update(texts)
        save_state(self.st)
        self.ref_busy(False)
        self.start_run()

    def on_ref_failed(self, code):
        self.ref_busy(False)
        if code.startswith("other:"):
            msg = self.tr("err_other", e=code[6:])
        else:
            msg = self.tr({"nocand": "err_nocand", "import": "err_import", "empty": "err_empty"}[code])
        self.l_refinfo.setText(msg)
        QMessageBox.warning(self, self.tr("title"), msg)

    def ref_text_edited(self):
        if not self.voice():
            return
        vs = self.vstate()
        vs["ref_text"] = self.e_text.text().strip()
        if vs["ref_wav"]:
            vs["transcripts"][vs["ref_wav"]] = vs["ref_text"]
        save_state(self.st)

    def play_wav(self, path):
        if not path or not os.path.isfile(path):
            return
        self.sfx_path = path
        if QSoundEffect is None:
            return self.play_external(path)
        if self.sfx is None:
            self.sfx = QSoundEffect(self)
            self.sfx.statusChanged.connect(self.sfx_status)
        self.sfx.stop()
        self.sfx.setSource(QUrl())                     # force a reload (the design sample reuses its file name)
        self.sfx.setSource(QUrl.fromLocalFile(path))
        self.sfx.setVolume(0.9)
        self.sfx.play()

    def sfx_status(self):
        if self.sfx and self.sfx.status() == QSoundEffect.Status.Error:
            self.play_external(self.sfx_path)

    @staticmethod
    def play_external(path):
        if sys.platform.startswith("win"):
            os.startfile(path)                                   # noqa: S606 (Windows only)
            return
        for cmd in ("paplay", "aplay", "afplay", "xdg-open", "open"):
            if shutil.which(cmd):
                subprocess.Popen([cmd, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

    # ---- running
    def build_workers(self, c):
        kinds = ["GPU"] * c["gpu_w"] + ["CPU"] * c["cpu_w"]
        if not kinds:
            kinds = ["CPU", "CPU"]
        while self.cards_l.count():
            it = self.cards_l.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.workers, self.cards = [], []
        for k in kinds:
            maxc = (GPU_THREADS if k == "GPU" else c["cpu_t"]) * 100
            self.workers.append(dict(kind=k, status="idle", idx=0, cur="", times=[], cpu=None, rss=None, maxcpu=maxc))
            card = WorkerCard()
            card.cpu.setRange(0, maxc)
            self.cards_l.addWidget(card)
            self.cards.append(card)

    def start_run(self):
        sc = self.scope()
        path = self.combo.currentData()
        if not path or not os.path.isfile(path):
            return QMessageBox.warning(self, self.tr("title"), self.tr("no_export"))
        if not sc:
            return QMessageBox.warning(self, self.tr("title"), self.tr("no_voice"))
        self.ref_text_edited()
        vs = self.vstate()
        vm = self.c_vmode.currentData()
        mode = "custom" if vm == "custom" else "clone"
        refs_file = None
        paths = scope_paths(*sc)
        if mode == "clone":
            if not vs["ref_wav"] or not os.path.isfile(vs["ref_wav"]) or not vs["ref_text"]:
                return QMessageBox.warning(self, self.tr("title"), self.tr("need_ref"))
            emo = find_emotion_refs(vs["ref_dir"]) if vs["ref_dir"] else {}
            missing = [p for p in emo.values() if p not in vs["transcripts"]]
            if missing:                                          # transcribe emotion samples first, then restart
                self.l_refinfo.setText(self.tr("emo_prep"))
                return self.start_ref_worker("texts", missing)
            refs = {k: {"wav": p, "text": vs["transcripts"][p]} for k, p in emo.items()
                    if vs["transcripts"].get(p) and os.path.abspath(p) != os.path.abspath(vs["ref_wav"])}
            if refs:
                paths["work"].mkdir(parents=True, exist_ok=True)
                refs_file = paths["work"] / "emotion_refs.json"
                refs_file.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
        elif not vs["speaker"]:
            return QMessageBox.warning(self, self.tr("title"), self.tr("need_speaker"))

        self.cfg = self.get_cfg()
        self.st["cfg"] = self.cfg
        # merge the export into the scope's manifest (changed lines lose their old audio, nothing else is deleted)
        paths["work"].mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(*sc)
        apply_export_to_manifest(self.manifest, self.scope_lines(), path)
        save_manifest(self.manifest)
        self.valid = set(self.manifest["lines"])
        save_state(self.st)
        self.run = dict(scope=sc, paths=paths, export=path, mode=mode, ref_wav=vs["ref_wav"], ref_text=vs["ref_text"],
                        speaker=vs["speaker"], use_notes=vs["use_notes"], refs_file=refs_file)
        self.running, self.stopping, self.attempt = True, False, 1
        self.run_start = time.time()
        self.b_start.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.set_cfg_enabled(False)
        self.update_cfg_hint()
        self.log.clear()
        self.launch("gen")

    def launch(self, step):
        self.step = step
        self.phase_key = step
        r = self.run
        env = self.run_env(self.cfg)
        if step == "gen":
            self.gen_start = time.time()
            self.build_workers(self.cfg)
            self.pcache.clear()
            env.insert("TTS_USE_NOTES", "1" if r["use_notes"] else "0")
        self.proc.setProcessEnvironment(env)
        manifest = f"--manifest={r['paths']['manifest']}"
        if step == "gen":
            args = ["-u", str(BASE / "generate_voice.py"), manifest, f"--mode={r['mode']}"]
            if r["mode"] == "clone":
                args += [f"--ref-wav={r['ref_wav']}", f"--ref-text={r['ref_text']}"]
                if r["refs_file"]:
                    args.append(f"--emotion-refs={r['refs_file']}")
            else:
                args.append(f"--speaker={r['speaker']}")
        else:
            args = ["-u", str(BASE / "finalize_audio.py"), manifest]
        self.proc.setWorkingDirectory(str(BASE))
        self.proc.start(sys.executable, args)

    def stop_run(self):
        if not self.running:
            return
        self.stopping = True
        try:
            parent = psutil.Process(self.proc.processId())
            for ch in parent.children(recursive=True):
                ch.kill()
            parent.kill()
        except psutil.Error:
            self.proc.kill()

    def finish(self, key):
        self.running = False
        self.phase_key = key
        self.b_start.setEnabled(True)
        self.b_stop.setEnabled(False)
        self.set_cfg_enabled(True)
        self.update_cfg_hint()
        for wk in self.workers:
            if wk["status"] in ("load", "gen"):
                wk["status"] = "done"
        if key in ("done", "problems") and self.run and self.manifest:
            self.manifest.setdefault("processed", {})[self.run["export"]] = os.path.getmtime(self.run["export"])
            save_manifest(self.manifest)
        save_state(self.st)
        self.fill_combo()                                   # refreshes the ✓ marks and counters
        self.refresh_preview()                              # immediately up-to-date Done / Remaining
        QTimer.singleShot(600, self.refresh_preview)        # safety net until the last files hit the disk

    def on_finished(self, code, _status):
        if self.stopping:
            return self.finish("stopped")
        if self.step == "gen":
            return self.launch("fin") if code == 0 else self.finish("error")
        if code == 0:
            self.finish("done")
        elif self.attempt < MAX_ATTEMPTS:
            self.attempt += 1
            self.log.appendPlainText(self.tr("retry"))
            self.launch("gen")
        else:
            self.finish("problems")

    # ---- script output
    def on_output(self):
        self.buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        *lines, self.buf = self.buf.replace("\r", "").split("\n")
        for ln in lines:
            ln = ln.rstrip()
            if ln:
                self.log.appendPlainText(ln)
                for part in re.split(r"(?=\[Worker \d+\])", ln):      # parallel workers can interleave on one line
                    if part:
                        self.parse_line(part)

    def parse_line(self, ln):
        m = re.match(r"\[Worker (\d+)\] \[(\d+)/(\d+)\] (\S+?)\.wav -> (.*)", ln)
        if m and int(m[1]) < len(self.workers):
            w = self.workers[int(m[1])]
            w.update(status="gen", idx=int(m[2]), cur=f"{m[4]}: {m[5]}")
            w["times"].append(time.time())
            return
        m = re.match(r"\[Worker (\d+)\] Loading", ln)
        if m and int(m[1]) < len(self.workers):
            self.workers[int(m[1])]["status"] = "load"
            return
        m = re.match(r"\[Worker (\d+)\] (Done!|Nothing to generate)", ln)
        if m and int(m[1]) < len(self.workers):
            self.workers[int(m[1])]["status"] = "done"

    # ---- monitoring
    def worker_procs(self):
        """Worker processes in start order. On Windows venvs a launcher stub duplicates each process -> keep leaves."""
        if self.proc.state() == QProcess.NotRunning or self.step != "gen":
            return []
        try:
            kids = psutil.Process(self.proc.processId()).children(recursive=True)
        except psutil.Error:
            return []
        res = []
        for k in kids:
            try:
                if "spawn_main" in " ".join(k.cmdline()):
                    res.append(self.pcache.setdefault(k.pid, k))
            except psutil.Error:
                pass
        try:
            pids = {p.pid for p in res}
            res = [p for p in res if not any(q.ppid() == p.pid for q in res if q.pid in pids)]
            res.sort(key=lambda p: p.create_time())
        except psutil.Error:
            pass
        return res

    def scan_done(self):
        """Finished raw wavs, final wavs and completion times in this run (for the ETA)."""
        p = self.paths()
        if not p:
            return 0, 0, []
        done, times = 0, []
        try:
            with os.scandir(p["out"]) as it:
                for e in it:
                    if e.name.endswith(".wav") and e.name[:-4] in self.valid:
                        done += 1
                        if self.running:
                            mt = e.stat().st_mtime
                            if mt >= self.gen_start:
                                times.append(mt)
        except FileNotFoundError:
            pass
        return done, count_wavs(p["final"], self.valid), sorted(times)

    def tick(self):
        self.l_cpu.setText(f"{self.tr('cpu')}  ({psutil.cpu_count()} threads)")
        self.b_cpu.setValue(int(psutil.cpu_percent()))
        vm = psutil.virtual_memory()
        self.l_ram.setText(f"{self.tr('ram')}  {gb(vm.used)} / {gb(vm.total)}")
        self.b_ram.setValue(int(vm.percent))
        g = self.gpu_mon.latest
        if g:
            extra = "".join([f"  {g['temp']:.0f} °C" if g.get("temp") is not None else "",
                             f"  {g['power']:.0f} W" if g.get("power") is not None else ""])
            noload = f"  ({self.tr('noload')})" if g.get("busy") is None else ""
            self.l_gpu.setText(f"{self.tr('gpu')}  {g['name']}{extra}{noload}")
            self.b_gpu.setValue(int(g["busy"] or 0))
            self.l_vram.setText(f"{self.tr('vram')}  {gb(g['used'])} / {gb(g['total'])}")
            self.b_vram.setValue(int(100 * g["used"] / g["total"]) if g["total"] else 0)
        else:
            self.b_gpu.setValue(0)
            self.b_vram.setValue(0)
            self.l_gpu.setText(f"{self.tr('gpu')}  ({self.tr('na')})")
            self.l_vram.setText(self.tr("vram"))

        procs = self.worker_procs()
        for i, (wk, cd) in enumerate(zip(self.workers, self.cards)):
            wk["cpu"] = wk["rss"] = None
            if i < len(procs):
                try:
                    wk["cpu"] = procs[i].cpu_percent(None)
                    wk["rss"] = procs[i].memory_info().rss
                except psutil.Error:
                    pass
            cd.setTitle(self.tr("worker", n=i, k=wk["kind"]))
            cd.status.setText(f"<b>{self.tr('st_' + wk['status'])}</b>")
            cd.cnt.setText(self.tr("cnt", i=wk["idx"]))
            ts = wk["times"][-10:]
            avg = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else None
            cd.spd.setText(self.tr("spd", s=avg) if avg else "")
            cd.cpu_l.setText(self.tr("cores"))
            cd.cpu.setValue(int(min(wk["cpu"] or 0, wk["maxcpu"])))
            cd.ram.setText(f"{self.tr('rss')}: {gb(wk['rss']) if wk['rss'] else '–'}")
            cd.cur.setText(f"{self.tr('cur')} {wk['cur'][:140]}" if wk["cur"] else "")

        # progress + ETA (from FINISHED files only)
        done, finals, times = self.scan_done()
        total = len(self.valid)
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(min(done, total))
        remaining = max(total - done, 0)
        self.l_stats.setText(
            f"{self.tr('total')}: <b>{total}</b>   |   {self.tr('done')}: <b>{done}</b>   |   "
            f"{self.tr('finals')}: <b>{finals}</b>   |   {self.tr('todo')}: <b>{remaining}</b>   |   "
            f"{self.tr('new')}: <b>{self.n_new}</b>   |   {self.tr('changed')}: <b>{self.n_changed}</b>")
        eta_txt = ""
        if self.running and self.step == "gen" and remaining > 0:
            recent = times[-ETA_WINDOW:]
            if len(recent) >= 3 and recent[-1] > recent[0]:
                rate = (len(recent) - 1) / (recent[-1] - recent[0])
                eta = remaining / rate
                fin = datetime.now() + timedelta(seconds=eta)
                eta_txt = (f"{self.tr('eta')}: <b>{fmt(eta)}</b>   ·   {self.tr('finish_at')} {fin:%H:%M}   ·   "
                           f"{rate * 60:.1f} {self.tr('per_min')}")
            else:
                eta_txt = self.tr("calc")
        if self.running:
            eta_txt += f"   ·   {self.tr('elapsed')}: {fmt(time.time() - self.run_start)}"
        self.l_eta.setText(eta_txt)

        text = {"gen": self.tr("ph_gen", a=self.attempt, m=MAX_ATTEMPTS), "fin": self.tr("ph_fin"),
                "idle": self.tr("idle"), "done": self.tr("ph_done"), "problems": self.tr("ph_problems"),
                "stopped": self.tr("ph_stopped"), "error": self.tr("ph_error")}[self.phase_key]
        self.l_phase.setText(text)

    def closeEvent(self, e):
        if self.running:
            if QMessageBox.question(self, self.tr("title"), self.tr("busy")) != QMessageBox.Yes:
                return e.ignore()
            self.stop_run()
            self.proc.waitForFinished(3000)
        self.gpu_mon.stop()
        self.gpu_mon.wait(2500)
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Main()
    w.show()
    sys.exit(app.exec())
