# Starfield Voice Cloner

Voice new or missing **Starfield** dialogue with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Feed it the dialogue
export of the Creation Kit, pick a character, pick a voice mode and press **Start**. A Qt GUI drives the whole pipeline
(generation, quality check, finalisation) and keeps the result in the same `Sound\Voice\<Plugin>\<VoiceType>\` layout the
game uses, so lines of different mods never mix.

- **Linux and Windows**, **NVIDIA (CUDA)**, **AMD Radeon (ROCm)** or **plain CPU**.
- Lightweight repository: no binaries, models are downloaded from Hugging Face on first use.

> 🇸🇰 Krátke zhrnutie v slovenčine je na konci ([Slovenčina](#slovenčina)).

---
<img width="1074" height="1057" alt="image" src="https://github.com/user-attachments/assets/1b2d962c-ad6f-4d3b-bba2-4b262101540b" />

## Contents

1. [Features](#features) · 2. [Output layout](#output-layout) · 3. [Voice modes](#voice-modes)
4. [Installation](#installation) · 5. [Creation Kit workflow](#creation-kit-workflow) · 6. [Using the GUI](#using-the-gui)
7. [GPU settings](#gpu-settings) · 8. [Command line](#command-line) · 9. [Reference](#reference) · 10. [Troubleshooting](#troubleshooting)

## Features

- **Three voice modes**: clone from a reference WAV, design a brand-new voice from a description, or use a preset
  speaker with the **emotion of every line** taken from the Creation Kit `FACE` column.
- **Per-mod output**: scope = `<mod plugin>/<VOICE TYPE>`, taken from the `FULL PATH` column. Resumption, manifests and
  counters are kept per scope, so two mods that share a `VOICE TYPE` (or a `FILENAME`) never overwrite each other.
- **CPU, GPU or Hybrid** workers sharing one queue, batching on GPU, automatic `float16` → `float32` fallback,
  AMD RDNA 2 override (`HSA_OVERRIDE_GFX_VERSION`) with automatic verification.
- **Reference sample picker**: choose a folder of voice files; the tool picks a clean 6–12 s sample, transcribes it with
  `faster-whisper` and lets you edit, listen and swap it. The reference is stored once per `VOICE TYPE`.
- Live monitoring of CPU, RAM, per-worker load, GPU load / VRAM / temperature; ETA from lines that are really finished.
- Quality check and finalisation to **32 kHz mono** WAV with padding and loudness normalisation; silent or broken files
  are detected and regenerated automatically. GUI in **English and Slovak**.

## Output layout

```
work/<Plugin>/<VoiceType>/            raw generated lines, manifest.json, report.txt   (git-ignored)
output/Sound/Voice/<Plugin>/<VoiceType>/<FILENAME>.wav      finalized 32 kHz mono WAV  ← your result
```

Example: `output/Sound/Voice/Amber Personal quest.esm/NPCFJaedaHuang/00924DDD.wav` for the CK path
`Data\Sound\Voice\Amber Personal quest.esm\NPCFJaedaHuang\00924DDD.wem`. The plugin folder is the path segment right after
`Sound\Voice\`. Rows without a `FULL PATH` go to `_no_plugin`. Converting the WAVs to the game's audio format (and
packaging the mod) is not done by this tool.

Reference samples (`ref_wav` / `ref_text`), the voice mode and the voice settings are saved **per `VOICE TYPE`** in
`gui_state.json`, so once a voice is configured it applies to every mod that uses that `VOICE TYPE`.

## Voice modes

| | Mode | Model | What it does |
|---|---|---|---|
| **A** | Clone from reference WAV | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Clones the voice of a reference sample (6–12 s, with its transcript). |
| **B** | Design a new voice | `…-1.7B-VoiceDesign` → `…-1.7B-Base` | Builds a reference sample from a text description, then clones it for all lines like mode A. |
| **C** | Preset speaker + CK emotions | `…-1.7B-CustomVoice` | Uses a built-in speaker and gives every line its own instruction made from the CK `FACE` column. |

**Mode A – emotion samples (optional).** If the reference folder contains files named after a `FACE` value (for example
`veryangry.wav` or `angry.wav`, `sad.wav`, `serious.wav`, `grateful.wav`, `tired.wav` …), lines with that `FACE` are cloned
from the matching sample; everything else uses the default reference. Matching ignores case; `veryangry.wav` wins over
`angry.wav`. These samples are transcribed with Whisper automatically before the first run.

**Mode B – voice designer.** Choose *Gender* (Female / Male), *Age* (Young 20s / Adult 30s / Middle-aged 45s / Older 65+),
*Timbre* (Natural / Deep / Raspy & Gravelly / Soft & Warm / Sharp & Commanding) and *Personality / cadence* (Calm & Neutral /
Unhinged & Manic / Cynical Frontier Spacer / Military & Disciplined / Friendly / Mysterious & Whispery). The English
*Voice description* is filled in and stays editable. The *Sample text* is pre-filled with two expressive sentences (a
question, a pause, an exclamation) so the sample has real intonation. **Generate Voice Sample** runs the VoiceDesign
model in a background thread, saves `design_sample.wav` into the character's reference folder, sets it as the reference
WAV and text, and you can **▶ Play** it and regenerate until you like it – then press **Start** to clone all lines.

**Mode C – preset speakers.** `Ryan`, `Aiden`, `Vivian`, `Serena`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`, `Uncle_Fu`.
`Ryan` and `Aiden` are native English speakers; the others speak English with an accent. Every line gets its own instruction
from `FACE`, e.g.:

| FACE | Instruction |
|---|---|
| `Neutral` | Speak in a natural, calm, conversational tone. |
| `Stern` | Speak in a firm, stern, commanding tone. |
| `VeryAngry` | Speak very angrily with sharp, intense emotion. |
| `Somber` | Speak in a heavy, somber, mournful tone. |
| `SinisterSmile` | Speak with a sly, darkly amused, sinister smirk in your voice. |
| `Worried` | Speak with anxious worry and concern. |
| `Grateful` | Speak warmly with genuine gratitude. |
| `Tired` | Speak in an exhausted, weary, drained voice. |
| `Thinking` | Speak thoughtfully, as if weighing your words. |

All 16 CK values are mapped (`Neutral, Serious, Stern, Thinking, Surprised, Grateful, Worried, SinisterSmile, Somber,
Sad, VeryAngry, Tired, Happy, Fear, Disgust, Puzzled`); see `common.py`. Lines without `FACE` use the neutral instruction.
Optionally the `SCRIPT NOTES` column is appended as an acting direction (checkbox in the GUI).

## Installation

Needs **Python 3.10+** (3.11 or 3.12 recommended), several GB of disk space (models are cached in `hf-cache/`) and enough
RAM: the 1.7B model is loaded once **per worker**. Install **PyTorch first**, matching your hardware, then
`pip install -r requirements.txt` (the file deliberately does not list `torch`). Get the repository with
`git clone https://github.com/<you>/starfield-voice-cloner.git` and `cd` into it.

### 1. NVIDIA GPUs (Windows and Linux)

```bash
python -m venv .venv
# Linux:   source .venv/bin/activate        Windows:   .venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126     # check pytorch.org for the current command
pip install -r requirements.txt
python qwen_gui.py                                                                  # or ./run.sh, run.bat
```

Choose **GPU only** in the GUI. Old cards (GTX 10-series) run `float16` very slowly – select `float32` manually.
Linux needs `libxcb-cursor0` for Qt on some distributions (`sudo apt install libxcb-cursor0`).

### 2. AMD Radeon on Linux (ROCm)

```bash
python3 -m venv .venv && source .venv/bin/activate
# official AMD wheels (ROCm 7.2; versions as listed in AMD's release notes) …
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
# … or the PyTorch index
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
sudo usermod -aG video,render $USER      # then log out and in; /dev/kfd must exist
./run.sh
```

RDNA 2 cards that ROCm does not list officially (RX 6700 XT, 6750 XT, 6700, 6600, 6500 XT …) need
`HSA_OVERRIDE_GFX_VERSION`. Leave the GUI on **Auto**: it detects `gfx1031` and tries `10.3.1` first, then `10.3.0`, and
keeps the first value that really runs a GPU kernel (checked once per run). On the author's RX 6700 XT with stock ROCm 7.2
wheels `10.3.0` is the value that works. See [GPU settings](#gpu-settings).

### 3. AMD Radeon RX 7000 / RX 9000 on Windows (official AMD ROCm wheels)

AMD publishes PyTorch wheels for supported Radeon cards on Windows at `https://repo.radeon.com/rocm/windows/`
(ROCm 7.2 or newer, **Python 3.12**). Follow AMD's current instructions – the exact wheel names change per release –
*“PyTorch via PIP installation (Windows)”* in the
[ROCm for Radeon docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html):
install the ROCm SDK wheels and then the matching `torch` / `torchaudio` wheels into a Python 3.12 venv. Check AMD's
compatibility list for your exact card. Then:

```bat
pip install -r requirements.txt --upgrade-strategy only-if-needed
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
python qwen_gui.py
```

`torch.version.hip` must not be `None` and `is_available()` must be `True` (if `pip` replaced torch, reinstall the AMD
wheels). Leave the ROCm override on **Auto** or **No override**. RX 6000 cards are not covered by these wheels – use section 4.

### 4. AMD Radeon RX 6000 series (RDNA 2: RX 6400 – RX 6950 XT) on Windows

*Verified on an RX 6700 XT at roughly 80 lines/minute.* Official ROCm wheels do not support RDNA 2 on Windows, so this
project uses the prebuilt **ROCm 7.15 + PyTorch 2.12** runtime ("TheRock") from
**<https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2>** as a prerequisite.

1. **Install the runtime** from a clone of that repository (see its README for the requirements, e.g. Windows 11):

   ```bat
   INSTALL.bat -SkipModel
   ```

   `-SkipModel` skips the large LLM model download, which this project does not need. If your `C:` drive is short on space
   you can place the stack elsewhere: `INSTALL.bat -SkipModel -Prefix D:\ROCmStack` (installs to `D:\ROCmStack\TheRock`).
   Note: the installer's pre-flight check and its staging folder still use `C:` (it expects roughly 25 GB free there), so
   free some space on `C:` even when `-Prefix` points to another drive. Read that project's README for its requirements.
2. **Install this project's dependencies into the runtime's venv** (`TheRock\.venv`, Python 3.11) without touching its torch:

   ```bat
   C:\TheRock\.venv\Scripts\python.exe -m pip install -r requirements.txt --upgrade-strategy only-if-needed
   C:\TheRock\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
   ```

   (use `D:\ROCmStack\TheRock\.venv\…` if you installed with `-Prefix`). The second command must show a ROCm build and `True`.
   If `pip` tries to replace `torch`, stop, and install the rest without dependency resolution for the conflicting package
   (`pip install qwen-tts --no-deps`, then the remaining requirements by hand).
3. **Launch with `start_rocm_rdna2.bat`** (repository root). It finds the runtime at `C:\TheRock` or
   `D:\ROCmStack\TheRock` (or the folder in the `THEROCK_ROOT` variable – set it in the file or in your environment),
   puts `%THEROCK_ROOT%\build\dist\rocm\bin` and `%THEROCK_ROOT%\.venv\Scripts` first on `PATH`, sets
   `HSA_OVERRIDE_GFX_VERSION=10.3.1` and `PYTHONUTF8=1`, and starts `qwen_gui.py` with the runtime's Python.
4. In the GUI choose **GPU only**; leave the ROCm override on **Auto** (it keeps the `10.3.1` set by the launcher).

### CPU only (any machine)

Install the CPU PyTorch (`--index-url https://download.pytorch.org/whl/cpu`), then `pip install -r requirements.txt`.
CPU mode works everywhere but is much slower; 2 workers × half of your physical cores is the default.

## Creation Kit workflow

1. **Export the dialogue** from the Creation Kit (look for *Export Dialogue*; menu names differ between CK versions). The
   result is a tab-separated text file. Needed columns: `VOICE TYPE`, `FILENAME`, `RESPONSE TEXT`. Used when present:
   `FULL PATH` (e.g. `Data\Sound\Voice\Amber Personal quest.esm\NPCFJaedaHuang\00924DDD.wem` → plugin folder),
   `FACE` (emotion) and `SCRIPT NOTES` (acting direction). The text is expected as Windows-1252 (`SVC_EXPORT_ENCODING`
   changes that).
2. **Name it** so the file name contains `export_ck` (any extension, e.g. `export_ck.txt` or `export_ck.xt`) or
   `DialogueExport`, and put it in Downloads, Documents, Desktop or next to the scripts – the GUI finds it. Otherwise
   use **Browse…**.
3. **Get reference voices** (modes A and B): extract the voice archives (`Starfield - Voices*.ba2`, files below
   `sound\voice\<plugin>\<voice type>`) with an archive tool such as Bethesda Archive Extractor, convert the audio
   (`.wem`) to `.wav` with a tool of your choice (for example vgmstream) and point the GUI at that folder. For mode C no
   reference is needed.

## Using the GUI

1. **Export CK** – pick the export. Then pick the **Mod / plugin** (from `FULL PATH`) and the **Character (VOICE TYPE)**.
   Both lists are searchable; the number is the count of lines.
2. Choose the **Voice mode** (A, B or C) and set it up (reference folder / designer / speaker).
3. **Performance** – CPU only, GPU only or Hybrid; **Test GPU** checks that PyTorch sees the card.
4. **▶ Start**. Results appear in `output/Sound/Voice/<Plugin>/<VoiceType>/` (**Open output**). A newer export only
   generates new or changed lines. **Regenerate everything** deletes the generated audio of the current scope (use it after
   switching voice mode or speaker).

## GPU settings

| Mode | What it does |
|---|---|
| **CPU only** | 2 CPU workers (threads = physical cores / 2). Works everywhere. |
| **GPU only** | 1 GPU worker, batch 4, no CPU worker – usually the best choice. One worker with a bigger batch beats many. |
| **Hybrid** | GPU worker + 1 CPU worker; helps only when the CPU has cores to spare. |

**Precision** `auto` tries `float16` and falls back to `float32` when the card cannot do it (checked at start, and again if a
batch produces NaN/silent audio). Watch the VRAM bar: `float32` needs about twice the memory; 4 GB cards are best used in
CPU mode.

**AMD ROCm override (`HSA_OVERRIDE_GFX_VERSION`)** – makes ROCm treat a GPU as a target it has kernels for:

| GUI option | Use for |
|---|---|
| **Auto** (default) | Keeps `HSA_OVERRIDE_GFX_VERSION` if your environment already sets it (e.g. `start_rocm_rdna2.bat`). Otherwise, on Linux it reads the GPU target from the kernel: `gfx1031` → tries **10.3.1**, then **10.3.0**, and verifies each in a throw-away process; other unofficial RDNA 2 → 10.3.0; everything else → no override. Windows: no detection. |
| **No override** | NVIDIA CUDA, RX 7000 / 9000, RX 6800 / 6900 XT (`gfx1030`, natively supported). |
| **10.3.0** | `gfx1030` kernels – what stock ROCm wheels ship; the safe fallback for any RDNA 2 on Linux. |
| **10.3.1** | `gfx1031`: RX 6700 XT, 6750 XT, 6700; the value used by the TheRock Windows runtime. |
| **10.3.2** | Navi 23: RX 6600, 6600 XT, 6650 XT, 6600M. |
| **10.3.4** | Navi 24: RX 6400, 6500 XT. |

The `10.3.2` / `10.3.4` options only work if your ROCm/PyTorch build contains kernels for that exact target; on
*“invalid device function”* or HIP errors switch to `10.3.1` (TheRock) or `10.3.0` (stock wheels). **Test GPU** tries the
Auto candidates one after another.

GPU monitor: AMD on Linux via sysfs (load, VRAM, temperature, power), NVIDIA via `nvidia-smi`, otherwise a small helper
using `torch.cuda.mem_get_info()` (VRAM only – this is what you get on Windows with AMD).

## Command line

```bash
# clone (mode A / B)
python generate_voice.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm" \
       --mode clone --ref-wav sample.wav --ref-text "Exact transcript." [--emotion-refs refs.json]
# preset speaker + FACE emotions (mode C)
python generate_voice.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm" --mode custom --speaker Ryan
# voice design sample (mode B, step 1)
python generate_voice.py --design-sample sample.wav --design-instruct "Female voice, ..." --design-text "Wait... really? Fine!"
# check + finalize
python finalize_audio.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm"
```

`--plugin` may be left out when the `VOICE TYPE` occurs in only one mod. `--help` lists everything. Both scripts also accept
`--manifest work/<Plugin>/<VoiceType>/manifest.json` (what the GUI uses).

## Reference

| Variable | Meaning | Default |
|---|---|---|
| `TTS_GPU_WORKERS` / `TTS_CPU_WORKERS` | worker processes | `0` / `2` |
| `TTS_GPU_BATCH` / `TTS_CPU_BATCH` | lines generated at once per worker | `1` / `1` |
| `TTS_GPU_THREADS` / `TTS_CPU_THREADS` | CPU threads per worker | `2` / `6` |
| `TTS_DTYPE` | `auto`, `float16`, `bfloat16`, `float32` (GPU) | `auto` |
| `TTS_HSA` | `auto`, `none`, `10.3.0`, `10.3.1`, `10.3.2`, `10.3.4` | `auto` |
| `TTS_GPU_INDEX` | which GPU to use | `0` |
| `TTS_USE_NOTES` | `1` = add `SCRIPT NOTES` to the instruction (mode C) | `0` |
| `SVC_GFX` | force the detected GPU target, e.g. `gfx1031` (containers, WSL) | – |
| `SVC_DATA_DIR` | where `work/`, `output/` and the state are stored | the script folder |
| `SVC_EXPORT_ENCODING` | encoding of the export file | `cp1252` |
| `HF_HOME` | Hugging Face cache | `<repo>/hf-cache` |

Files: `qwen_gui.py` (GUI), `generate_voice.py` (generation), `finalize_audio.py` (check + finalisation), `common.py` (paths,
export parsing, FACE prompts, manifests), `run.sh` / `run.bat` / `start_rocm_rdna2.bat` (launchers). Folders from older
versions (`vystup_<VT>`, `FINAL_<VT>`) are not used any more – move the WAVs into `work/<Plugin>/<VoiceType>/` if you want
to keep them.

## Troubleshooting

- **`cuda_available False` on AMD** – you installed a CUDA build (`+cu…`) or the ROCm environment is incomplete. Reinstall
  PyTorch from a ROCm source, check `/dev/kfd` and the `video` / `render` groups (Linux), use **Test GPU**.
- **“invalid device function” / HIP errors** – wrong HSA override for your card and runtime, see [GPU settings](#gpu-settings).
- **`start_rocm_rdna2.bat` says the runtime was not found** – install it first, or set `THEROCK_ROOT`.
- **Out of memory** – lower GPU batch, use 1 GPU worker or CPU mode. Silent/garbled GPU output → precision `float32`.
- **Mode C sounds foreign** – use `Ryan` or `Aiden` for English; the other speakers are trained for other languages.
- **Whisper does nothing** – `pip install faster-whisper`; the first transcription downloads `base.en` (internet needed once).
  Audio is loaded with `librosa` as a 16 kHz array, so PyAV is not used for decoding.
- **`sox: not found`, `flash-attn is not installed`** – harmless warnings.
- **Export not listed** – the GUI looks for names containing `export_ck` or `DialogueExport` in Downloads, Documents,
  Desktop and the script folder; otherwise use **Browse…**.

## Responsible use

This tool clones and designs voices. Use it for personal, non-commercial modding, respect the Bethesda / Microsoft terms and
the rights of voice actors, and do **not** redistribute extracted game audio or imitations of real people's voices without
permission. Check the licence of the Qwen3-TTS models on Hugging Face. Not affiliated with Bethesda, AMD, NVIDIA or Alibaba.

## License

[MIT](LICENSE) for the code in this repository. Models and game assets have their own licences.

---

## Slovenčina

**Starfield Voice Cloner** namluví repliky zo Starfieldu hlasom, ktorý si vyberieš. Funguje na Linuxe aj Windowse (NVIDIA,
AMD ROCm, CPU).

- **Výstup** je rozdelený podľa módu a postavy z `FULL PATH`:
  `output/Sound/Voice/<Plugin>/<VoiceType>/<FILENAME>.wav` (priebežné súbory sú v `work/…`). Rovnaký `VOICE TYPE` z dvoch
  módov sa nikdy nepomieša. Referenčná vzorka sa ukladá raz pre každý `VOICE TYPE`.
- **Tri režimy hlasu:** **A** klon z referenčného WAV (voliteľne emočné vzorky `angry.wav`, `sad.wav` … podľa stĺpca `FACE`),
  **B** nový hlas z popisu (VoiceDesign → klon), **C** vstavaný hlas (Ryan, Aiden, Vivian …) s emóciou každej vety zo stĺpca `FACE`.
- **Inštalácia:** najprv PyTorch pre tvoj hardvér (CUDA `cu126`, ROCm, alebo CPU), potom `pip install -r requirements.txt`,
  spustenie `./run.sh` / `run.bat`. **AMD RX 6000 (RDNA 2) na Windowse:** nainštaluj runtime TheRock z
  <https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2> (`INSTALL.bat -SkipModel`, voliteľne `-Prefix D:\ROCmStack`;
  inštalátor aj tak kontroluje voľné miesto na `C:`), závislosti daj do `TheRock\.venv` príkazom
  `python.exe -m pip install -r requirements.txt --upgrade-strategy only-if-needed` a spúšťaj cez `start_rocm_rdna2.bat`.
- **AMD na Linuxe:** v GUI nechaj ROCm override na *Auto* – pre `gfx1031` vyskúša `10.3.1`, potom `10.3.0` a ponechá prvý, ktorý
  naozaj funguje.
- **Postup:** export z Creation Kitu (`export_ck…`) → vyber mod, postavu a režim hlasu → **▶ Štart**.
