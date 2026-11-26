# Starfield Voice Cloner

Voice new or missing **Starfield** dialogue locally with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Load a dialogue export from the Creation Kit, choose a mod/plugin and character, select a voice mode, and press **Start**. The Qt GUI generates, checks and finalises the audio while keeping output separated by mod and `VOICE TYPE`.

- Windows and Linux; NVIDIA CUDA, AMD ROCm, or CPU (PyTorch).
- Three voice modes: clone a reference WAV, design a voice then clone it, or use a preset speaker with CK emotions.
- Models and llama.cpp files are downloaded when needed; no model binaries are included in this repository.
- The GUI is available in **English and Slovak**. Change it with the **Language/Jazyk** selector.

> 🇸🇰 Slovenský prehľad je na konci ([Slovenčina](#slovenčina)).

---

<img width="1074" height="1057" alt="Starfield Voice Cloner GUI" src="https://github.com/user-attachments/assets/1b2d962c-ad6f-4d3b-bba2-4b262101540b" />

## Contents

1. [Features](#features)
2. [Output layout](#output-layout)
3. [Voice modes](#voice-modes)
4. [Windows installation](#windows-installation)
5. [Linux installation](#linux-installation)
6. [Creation Kit workflow](#creation-kit-workflow)
7. [Using the GUI](#using-the-gui)
8. [Performance and GPU settings](#performance-and-gpu-settings)
9. [Building the Windows executable](#building-the-windows-executable)
10. [Command line and environment](#command-line-and-environment)
11. [Troubleshooting](#troubleshooting)

## Features

- **Three voice modes:** clone a reference WAV; design a new voice from a description and clone it; or use a preset speaker with the emotion of each line taken from the CK `FACE` column.
- **Per-mod output:** the scope is `<Plugin>/<VOICE TYPE>`, derived from `FULL PATH`. Manifests, progress and output are kept per scope, so mods that share a `VOICE TYPE` or filename do not overwrite each other.
- **Reference sample picker:** choose a folder of WAVs; the GUI ranks clean samples around 6–12 seconds, transcribes a candidate with faster-whisper, and lets you listen, edit its transcript or choose another. References and settings are saved per `VOICE TYPE`.
- **Emotion references:** optional reference WAVs named for emotion/FACE values can be used in clone mode.
- **Progress and monitoring:** line counts, ETA, logs, CPU/RAM, per-worker activity, GPU and VRAM information.
- **Audio validation:** final 32 kHz mono WAVs include padding and loudness adjustment; silent or broken files are removed and retried.
- **English and Slovak GUI.**

## Output layout

```text
work/<Plugin>/<VOICE TYPE>/                 raw generated WAVs, manifest.json, report.txt (git-ignored)
output/Sound/Voice/<Plugin>/<VOICE TYPE>/    final 32 kHz mono WAVs
```

Example: `output/Sound/Voice/Amber Personal quest.esm/NPCFJaedaHuang/00924DDD.wav` corresponds to a CK path like `Data\Sound\Voice\Amber Personal quest.esm\NPCFJaedaHuang\00924DDD.wem`. Rows without a usable `FULL PATH` are placed under `_no_plugin`. Converting WAVs to the game's final audio format or packaging the mod is not done by this tool.

The GUI state file `gui_state.json` stores reference audio, transcripts, voice mode and settings per `VOICE TYPE`; settings for that character are shared across plugins using the same voice type.

## Voice modes

| Mode | Model | What it does |
|---|---|---|
| **A — Clone from reference WAV** | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Clones a voice from a reference sample and its transcript. |
| **B — Design a new voice** | `…-1.7B-VoiceDesign` → `…-1.7B-Base` | Creates a reference sample from a voice description, then clones the dialogue like mode A. |
| **C — Preset speaker + CK emotions** | `…-1.7B-CustomVoice` | Uses a built-in speaker and creates a per-line emotion instruction from CK `FACE`. |

**Mode A — emotion samples (optional).** Put emotion WAVs in the selected reference folder with names such as `veryangry.wav`, `angry.wav`, `sad.wav`, `serious.wav`, `grateful.wav` or `tired.wav`. Matching ignores case, and the most specific match wins (`veryangry.wav` before `angry.wav`). Unmatched lines use the default reference. The GUI transcribes emotion samples with Whisper before generation.

**Mode B — voice designer.** Choose Gender, Age, Timbre and Personality/cadence. The GUI builds an editable English voice description. The sample text is also editable and is pre-filled with an expressive question, pause and exclamation. **Generate Voice Sample** creates `design_sample.wav`, makes it the reference and associates the sample text with it. Listen and regenerate if needed, then press **Start**. Generating this sample requires PyTorch. If a sample WAV already exists, you can select it and use llama-server for the clone step.

**Mode C — preset speakers.** Available speakers: `Ryan`, `Aiden`, `Vivian`, `Serena`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`, `Uncle_Fu`. Ryan and Aiden are native English speakers; other speakers produce English with an accent. The 16 mapped CK values are `Neutral, Serious, Stern, Thinking, Surprised, Grateful, Worried, SinisterSmile, Somber, Sad, VeryAngry, Tired, Happy, Fear, Disgust, Puzzled`. Lines without `FACE` use Neutral. Enable **Add SCRIPT NOTES to the instruction** to append notes as an acting direction. Mode C requires PyTorch.

## Windows installation

### Recommended: `INSTALL.bat`

1. Install 64-bit **Python 3.10–3.13** from [python.org](https://www.python.org/downloads/). During setup, tick **Add python.exe to PATH**. Choice 4 needs Python 3.12.
2. Download the repository ZIP and extract the **whole project folder**, or clone the repository. Keep the scripts together.
3. Double-click **`INSTALL.bat`**. It creates `venv_svc`, installs the lightweight `requirements-llama.txt` packages and displays the engine choices:

   | Choice | Setup | Modes/features |
   |---|---|---|
   | **1 — AMD or NVIDIA (recommended first try)** | llama-server dependencies only; downloads llama.cpp and the model on first use | A and B-clone; Vulkan on AMD/NVIDIA or CUDA on NVIDIA |
   | **2 — NVIDIA** | Adds PyTorch CUDA and Qwen3-TTS to `venv_svc` | A, B and C; VoiceDesign sample |
   | **3 — AMD RX 6000 / RDNA2** | Uses the external TheRock ROCm runtime and this project's dependencies | A, B and C; VoiceDesign sample |
   | **4 — AMD RX 7000 / RX 9000** | AMD ROCm PyTorch wheels in `venv_rocm`; Python 3.12 | A, B and C; VoiceDesign sample |

4. When installation finishes, double-click **`run.bat`** to open the GUI. `INSTALL.bat` saves the choice to `engine.txt`; run it again to change engines.
5. Select the CK export, plugin, character and voice mode. Press **Start**. For llama-server modes, the GUI starts/prepares the server automatically if it is not running and continues when it is ready. You do not need to launch a server in a separate terminal.

The **Start server** button in the GUI is optional, for starting the backend in advance. `START_TTS_SERVER.bat` is an optional utility that starts only the Windows server without opening the GUI.

### AMD RX 6000 / RDNA2 on Windows

The RX 6000 PyTorch route uses [vllm-rocm-windows-rdna2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2) and its TheRock ROCm runtime. Installer choice 3 downloads and runs its installer. The corresponding manual installer command is:

```bat
INSTALL.bat -SkipModel
```

`-SkipModel` skips that project's LLM model, which this voice cloner does not use. The runtime installer downloads several GB, requires administrator approval and may need roughly 25 GB of temporary free space on `C:` even if the runtime is installed elsewhere. Follow the runtime project's README for current requirements. `start_rocm_rdna2.bat` looks for `C:\TheRock`, `D:\ROCmStack\TheRock` or `THEROCK_ROOT`, sets the runtime paths and starts the GUI with the runtime's Python. `run.bat` delegates to this launcher when choice 3 is selected.

### AMD RX 7000 / RX 9000 on Windows

Choose **4** in `INSTALL.bat`. It requires Python 3.12 and installs AMD's official ROCm PyTorch wheels into `venv_rocm`. Supported cards, wheel names and URLs can change; consult AMD's current [Windows PyTorch/ROCm instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html) if installation fails. The installer installs the project requirements with `--upgrade-strategy only-if-needed` to avoid replacing the AMD PyTorch build.

### NVIDIA manual PyTorch installation

The automatic installer is recommended. For a manual setup, install a CUDA-compatible PyTorch build first and then the project requirements. Example; check pytorch.org for the current CUDA wheel command:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python qwen_gui.py
```

## Linux installation

`run.sh` does **not** create an environment or install dependencies. It activates an existing `venv/` or `.venv/` in the repository folder and then starts `qwen_gui.py`. Install Python, the correct PyTorch build for your hardware, and the project requirements first.

### 1. Install system prerequisites

Use Python **3.10 or newer** (Python 3.11 or 3.12 recommended), `python3-venv`, `pip`, and standard build/runtime libraries. On Ubuntu/Debian, a starting point is:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git libxcb-cursor0
```

`libxcb-cursor0` is needed by Qt on some Linux distributions; package names differ on Fedora, Arch and other distributions. Use your distribution's package manager for the equivalent packages.

### 2. Get the project and create a virtual environment

Clone the repository, or use the directory where you extracted its ZIP. Run these commands from the project root (the directory containing `qwen_gui.py` and `run.sh`):

```bash
git clone https://github.com/Sonic1SVK/Starfield-Voice-Cloner.git
cd Starfield-Voice-Cloner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Keep using this activated `.venv` for the PyTorch and project package installation. If you close the terminal, activate it again before running manual `pip` or `python` commands. The `./run.sh` launcher activates `.venv` itself, so after setup you can also launch it from a terminal in the project directory without manually activating first.

### 3. Install PyTorch for the GPU (or CPU) **before** project requirements

`requirements.txt` intentionally does not list PyTorch. Install the build matching your GPU and driver **before** installing the requirements; otherwise pip may replace your CUDA/ROCm build with an incompatible package.

#### NVIDIA CUDA

Use the command shown by [PyTorch's installation selector](https://pytorch.org/get-started/locally/) for your Linux, Python and CUDA version. For example, the project has documented this wheel command:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

CUDA wheel versions evolve; if this command no longer matches the current PyTorch release or your driver, copy the current Linux/Pip/CUDA command from pytorch.org instead.

#### AMD Radeon ROCm

First check that your exact GPU and Linux distribution are supported by the ROCm/PyTorch release you intend to install. Use AMD's current compatibility matrix and installation instructions. The following are example ROCm 7.2 commands from the project documentation; versions and package availability may change:

```bash
# Option A: AMD's ROCm wheel repository (example ROCm 7.2 package versions)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/

# Option B: PyTorch ROCm index (use the ROCm version supported by your system)
# If you installed Option A successfully, do not install Option B on top of it.
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

RDNA2 cards such as RX 6700 XT/6750 XT/6700 and some RX 6000 models may not be listed as official ROCm targets. The GUI's **AMD ROCm override** has **Auto** and explicit target choices. Auto detection works on Linux; for a `gfx1031` card it tries `10.3.1` and then `10.3.0`, verifying whether the GPU can actually run a test kernel. Stock ROCm wheels may work with a different value than a custom runtime. An override cannot add kernels absent from the installed PyTorch/ROCm build.

Make sure your Linux user can access the AMD compute device. On Ubuntu/Debian, if those groups exist, add the user and then log out and back in:

```bash
sudo usermod -aG video,render "$USER"
```

Check that `/dev/kfd` exists. Consult the ROCm installation guide for your distribution if the device or permissions are missing.

#### CPU-only PyTorch

For machines without a supported GPU, install PyTorch's CPU build:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CPU generation is supported but substantially slower than GPU generation, especially with PyTorch float32.

### 4. Install this project's Python packages

Choose **one** requirements file:

```bash
# Full PyTorch install: enables PyTorch generation and modes A, B and C
pip install -r requirements.txt

# OR, lightweight llama-server GUI dependencies: modes A and B-clone only; no PyTorch/Qwen package
# Use this only when you intend to use llama-server and do not need mode C or to create a VoiceDesign sample.
# pip install -r requirements-llama.txt
```

The full `requirements.txt` includes the GUI, audio processing, faster-whisper and `qwen-tts`, but deliberately does not install torch. The lightweight `requirements-llama.txt` omits `qwen-tts` and torch because the llama-server engine handles Base-model clone requests. Linux can start the GUI with `run.sh`; however, the repository's managed llama.cpp downloader is Windows-specific, so on Linux use the PyTorch engine unless you have independently configured a compatible llama-server endpoint/setup.

### 5. Start the GUI

From the project directory:

```bash
./run.sh
```

If the script is not executable, run `chmod +x run.sh` once, or start it with `bash run.sh`. It uses `.venv` (or `venv`) and launches the GUI; it does not install packages. The first faster-whisper transcription downloads its `base.en` model and needs an internet connection.

### Linux installation checklist

Before reporting a problem, activate the environment and check that Python can import the packages and that PyTorch sees the GPU:

```bash
source .venv/bin/activate
python -c "import PySide6, librosa, soundfile, faster_whisper; print('GUI/audio dependencies OK')"
python -c "import torch; print('torch:', torch.__version__, 'HIP:', torch.version.hip, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
./run.sh
```

For NVIDIA, `torch.cuda.is_available()` should be `True` when the driver and CUDA wheel are compatible. For AMD ROCm, `torch.version.hip` should not be `None` and `torch.cuda.is_available()` should be `True`; PyTorch uses the `torch.cuda` API for ROCm too. In the GUI, use **Test GPU** and leave the ROCm override on **Auto** initially.

## Creation Kit workflow

1. Export dialogue from the Creation Kit as a tab-separated file. Required columns are `VOICE TYPE`, `FILENAME` and `RESPONSE TEXT`.
2. When present, `FULL PATH` determines the plugin folder; `FACE` provides emotion; `SCRIPT NOTES` provides an optional acting direction. Default export encoding is Windows-1252; change it with `SVC_EXPORT_ENCODING` if needed.
3. Name the file so it contains `export_ck` or `DialogueExport` (any extension, such as `.txt`, `.xt` or `.tsv`). Put it in Downloads, Documents, Desktop or next to the scripts, or use **Browse…**. Press **Rescan** to search again.
4. For modes A and B, prepare reference audio. Extract the relevant voice archive (`Starfield - Voices*.ba2`, under `sound\voice\<plugin>\<voice type>`) with an archive tool, convert `.wem` files to `.wav` with an audio tool such as vgmstream, then select the folder or WAV in the GUI. Mode C does not need reference audio.

## Using the GUI

1. Select the CK export. Select **Mod / plugin** and **Character (VOICE TYPE)**; the number beside each item is its line count. Both lists are searchable.
2. Choose voice mode A, B or C and configure its reference, voice design or preset speaker.
3. Choose a **Performance** engine and settings. For llama-server, select Auto/Vulkan/CUDA if needed. The first server start downloads llama.cpp and the model into `tts_backend/`.
4. Press **Start**. For llama-server modes, the GUI automatically starts/prepares the server if it is not running and continues when ready; no separate terminal launch is required. You may use the GUI's **Start server** button to start it in advance.
5. The GUI displays total, completed, finalised and remaining lines; progress, ETA, CPU/RAM/GPU/VRAM, worker status and logs. After generation it validates and finalises the audio.
6. Use **Open output** to open the final WAV folder. **Regenerate everything** deletes generated WAVs for the selected plugin/character scope and starts that scope again.

## GPU and performance settings

### Engines

| GUI option | Behaviour |
|---|---|
| **PyTorch GPU only** | Default PyTorch configuration: one GPU worker, batch 4 and no CPU worker. |
| **llama-server — GPU** | Uses llama.cpp and the Base GGUF model; choose Vulkan or CUDA backend. No PyTorch required for modes A/B-clone. |
| **llama-server — GPU + CPU hybrid** | GPU server plus a separate CPU-only server on port 8081. Uses additional system RAM; useful only if the CPU has spare capacity. |
| **llama-server — CPU only** | CPU-only llama-server; slower than GPU but does not require PyTorch or a GPU. |

The GUI measures completed runs and records results in `benchmarks.csv`. After runs with more than one engine, it may suggest which engine was faster on that computer. Treat benchmark rates as hardware/workload-specific, not universal promises.

PyTorch precision options are `auto`, `float16`, `bfloat16` and `float32`. `auto` tests float16 and falls back to float32 if the GPU cannot use it or output is invalid. Float32 can use substantially more VRAM. Lower GPU workers or batch size if memory is insufficient. **Test GPU** requires PyTorch.

### AMD ROCm override (`HSA_OVERRIDE_GFX_VERSION`)

Start with **Auto**. It preserves an override provided by the environment/launcher; on Linux it detects RDNA2 targets and verifies candidate values. The GUI also offers `No override`, `10.3.0`, `10.3.1`, `10.3.2` and `10.3.4`. These select kernel targets only; they do not add kernels missing from the installed ROCm/PyTorch runtime. Use **Test GPU** and the generation log to check support.

### RX 6700 XT: ROCm vs Vulkan example

A run on an RX 6700 XT reported 40.4 lines/min with PyTorch/ROCm/GPU and 15.1 lines/min with llama-server/Vulkan for a 526-line set. This is a measurement from one computer and workload, not a general guarantee. RX 6000 Windows users can try the [TheRock ROCm runtime](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2) and compare results on their system.

## Output and saved files

```text
work/<Plugin>/<VOICE TYPE>/                 raw WAVs, manifest.json, report.txt
output/Sound/Voice/<Plugin>/<VOICE TYPE>/    final 32 kHz mono WAVs
voices/                                      optional reference/design WAVs
hf-cache/                                    Hugging Face model cache
tts_backend/                                 downloaded llama.cpp/model files and server logs
gui_state.json                               GUI settings and per-VOICE TYPE state
benchmarks.csv                               measured engine performance
```

Rows without a usable `FULL PATH` are stored under `_no_plugin`. Old `vystup_<VT>` and `FINAL_<VT>` folders are not used; move wanted WAVs into the current `work/<Plugin>/<VOICE TYPE>/` layout if you need to reuse them.

## Building the Windows executable

`build_exe.bat` is a **build script**, not a Python installer or normal launcher. First run `INSTALL.bat`, then run `build_exe.bat`. It uses `_env.bat` to find the selected Python, installs PyInstaller and creates an **onedir** build:

```text
dist/StarfieldVoiceCloner/StarfieldVoiceCloner.exe
```

Keep the entire `dist/StarfieldVoiceCloner/` directory together; this is not a standalone single-file executable. The build excludes PyTorch and `qwen-tts`, so it is the llama-server edition: modes A and B-clone work, but mode C, **Test GPU** and creating a new VoiceDesign sample require the regular Python installation with PyTorch. A VoiceDesign WAV created earlier can still be selected for B-clone. The build script copies `README.md` and `LICENSE` into the output directory.

## Command line

```bash
# Clone from a reference WAV (mode A / B-clone)
python generate_voice.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm" \
  --mode clone --ref-wav sample.wav --ref-text "Exact transcript."

# Clone with optional emotion references
python generate_voice.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm" \
  --mode clone --ref-wav sample.wav --ref-text "Exact transcript." --emotion-refs refs.json

# Preset speaker + CK emotions (mode C; PyTorch required)
python generate_voice.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm" \
  --mode custom --speaker Ryan

# Create a VoiceDesign sample (PyTorch required)
python generate_voice.py --design-sample sample.wav \
  --design-instruct "Female voice, in their thirties, with a natural timbre and calm delivery." \
  --design-text "Wait... really? Fine!"

# Validate/finalise
python finalize_audio.py export_ck.txt --voice-type NPCFAndreja --plugin "My Mod.esm"
```

Use `--help` for all options. `--plugin` can be omitted when a `VOICE TYPE` occurs in only one plugin. Both scripts accept `--manifest work/<Plugin>/<VOICE TYPE>/manifest.json` (used by the GUI).

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `TTS_ENGINE` | Set to `llama` to use llama-server generation. | unset |
| `TTS_API_URL` | llama-server speech endpoint. | `http://127.0.0.1:8080/v1/audio/speech` |
| `TTS_API_URL_CPU` | Optional CPU server endpoint for hybrid mode. | unset |
| `TTS_GPU_WORKERS` / `TTS_CPU_WORKERS` | Worker counts; in llama-server mode, parallel request counts. | `0` / `2` |
| `TTS_GPU_BATCH` / `TTS_CPU_BATCH` | Lines generated per batch. | `1` / `1` |
| `TTS_GPU_THREADS` / `TTS_CPU_THREADS` | CPU threads per worker. | `2` / `6` |
| `TTS_DTYPE` | GPU precision: `auto`, `float16`, `bfloat16`, `float32`. | `auto` |
| `TTS_HSA` | ROCm override: `auto`, `none`, `10.3.0`, `10.3.1`, `10.3.2`, `10.3.4`. | `auto` |
| `TTS_GPU_INDEX` | GPU index. | `0` |
| `TTS_USE_NOTES` | `1` appends `SCRIPT NOTES` to CustomVoice instructions. | `0` |
| `SVC_GFX` | Force detected GPU target, e.g. `gfx1031`. | unset |
| `SVC_DATA_DIR` | Base folder for work/output/state. | project folder (or executable folder when frozen) |
| `SVC_EXPORT_ENCODING` | CK export encoding. | `cp1252` |
| `HF_HOME` | Hugging Face cache. | `<project>/hf-cache` |
| `THEROCK_ROOT` | Optional external TheRock runtime path. | auto-detected standard paths |

## Troubleshooting

- **Linux: `./run.sh` says no virtual environment found:** create `.venv` or `venv` in the project folder, install dependencies, and try again. `run.sh` activates an existing environment; it does not create one.
- **AMD PyTorch cannot see the GPU:** verify this is a ROCm PyTorch build, not CUDA. On Linux check `/dev/kfd` and `video`/`render` groups; use **Test GPU**.
- **HIP / invalid device function:** the ROCm runtime may not contain kernels for the GPU/override. Try **Auto** or another supported target.
- **Out of GPU memory:** reduce GPU workers or batch size and watch VRAM; try llama-server if available.
- **llama-server appears to use CPU:** try the other backend, restart the server and inspect `tts_backend/server.log`.
- **Mode C or VoiceDesign sample unavailable:** these need PyTorch. The PyInstaller executable intentionally does not include it.
- **Whisper transcription fails:** install `faster-whisper`; the first `base.en` transcription downloads a model and requires internet access.
- **TheRock runtime not found:** install it separately or set `THEROCK_ROOT`; use `start_rocm_rdna2.bat`.
- **Executable build fails:** run `INSTALL.bat` first and check that `_env.bat` finds the selected Python environment.
- **Old `vystup_*` / `FINAL_*` folders:** current versions use `work/<Plugin>/<VOICE TYPE>/` and `output/Sound/Voice/<Plugin>/<VOICE TYPE>/`.

## Responsible use and license

Only clone voices you are authorised to use. Respect Bethesda/Microsoft terms, voice actors' rights and local laws. Do not redistribute extracted game audio or imitate real people's voices without permission. The code in this repository is licensed under the [MIT License](LICENSE); models and game assets have separate licences. This project is not affiliated with Bethesda, AMD, NVIDIA or Alibaba.

---

## Slovenčina

GUI je dostupné **v slovenčine aj angličtine**; jazyk zmeníš v prepínači **Jazyk/Language**.

- **A — Klon z referenčného WAV:** použije nahrávku a jej prepis; voliteľne možno pridať emočné vzorky podľa `FACE`.
- **B — Nový hlas:** vytvorí ukážku z popisu hlasu a použije ju na klonovanie replík.
- **C — Vstavaný hlas:** vyberieš speakera a emóciu riadkov určuje CK `FACE`; možno pridať aj `SCRIPT NOTES`.

Na Windowse spusti `INSTALL.bat`, vyber engine a potom `run.bat`. Na Linuxe priprav Python prostredie podľa návodu vyššie a spusti `./run.sh`. V oboch prípadoch sa otvorí GUI; pri režime llama-server sa server automaticky spustí po stlačení **Štart**, ak ešte nebeží. Netreba ho samostatne štartovať z konzoly. Vyber CK export, plugin, postavu a hlasový režim, potom stlač **Štart**. Výsledné WAV súbory nájdeš v `output/Sound/Voice/<Plugin>/<VOICE TYPE>/`.

### Linux: stručný postup

Použi Python 3.10+ (odporúčame 3.11 alebo 3.12), nainštaluj systémové balíky `python3-venv`, `python3-pip` a prípadne `libxcb-cursor0`. Z koreňového priečinka projektu vytvor virtuálne prostredie, najprv nainštaluj PyTorch podľa svojej GPU a potom závislosti projektu:

```bash
cd Starfield-Voice-Cloner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Najprv nainštaluj PyTorch pre NVIDIA CUDA, AMD ROCm alebo CPU; príkazy sú v návode vyššie.
pip install -r requirements.txt
./run.sh
```

Ak chceš iba GUI a režimy A/B-klon cez vlastný/externý llama-server bez PyTorchu, namiesto plných requirements môžeš použiť `pip install -r requirements-llama.txt`. `run.sh` aktivuje existujúci `.venv` alebo `venv`, neinštaluje ho. Pri AMD ROCm over `/dev/kfd`, členstvo v skupinách `video`/`render` a použi **AMD ROCm override: Auto**; testuje sa cez **Otestovať GPU**.

### Vytvorenie `.exe`

`build_exe.bat` cez **PyInstaller** zostaví Windows aplikáciu z Python prostredia pripraveného cez `INSTALL.bat`. Výsledkom je celý priečinok `dist/StarfieldVoiceCloner/`, nie jeden samostatný `.exe` súbor. Build neobsahuje PyTorch ani `qwen-tts`; podporuje llama-server klonovanie v režimoch A a B-klon, ale nie režim C ani vytvorenie novej VoiceDesign vzorky.

### RX 6700 XT: ROCm a Vulkan

V jednom meraní bolo nameraných 40,4 replík/min pri PyTorch/ROCm/GPU a 15,1 replík/min pri llama-server/Vulkan. Je to výsledok konkrétneho počítača; výkon sa môže líšiť. Používatelia RX 6000 na Windowse môžu skúsiť [TheRock ROCm runtime](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2).
