# Starfield Voice Cloner

Create new or missing **Starfield** dialogue locally with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Load a Creation Kit dialogue export, select a mod/plugin and `VOICE TYPE`, choose a voice mode, and generate audio in the GUI.

The GUI is available in **English and Slovak** (change it with **Language/Jazyk**). Models are downloaded when needed; they are not included in this repository.

## Voice modes

| Mode | What it does | Requirement |
|---|---|---|
| **A — Clone from reference WAV** | Clones a voice from a WAV and its transcript. Optional emotion WAVs can match CK `FACE` values. | llama-server or PyTorch |
| **B — Design a new voice** | Creates a voice sample from an editable description, then uses it to clone dialogue. | PyTorch to create the sample; an existing sample can be used with llama-server |
| **C — Preset speaker + CK emotions** | Uses a built-in speaker and maps each line's `FACE` to an emotion instruction. `SCRIPT NOTES` can optionally be appended. | PyTorch |

Preset speakers: `Ryan`, `Aiden`, `Vivian`, `Serena`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`, `Uncle_Fu`. The llama-server engine supports clone modes A and B-clone, not mode C or VoiceDesign sample creation.

## Install and launch

### Windows

1. Run `INSTALL.bat` and choose an engine:

   | Choice | Setup |
   |---|---|
   | **1 — AMD or NVIDIA** | Lightweight llama-server setup; no PyTorch needed for A and B-clone. |
   | **2 — NVIDIA** | Adds PyTorch CUDA for mode C and VoiceDesign. |
   | **3 — AMD RX 6000 / RDNA2** | Uses the external TheRock ROCm runtime. |
   | **4 — AMD RX 7000 / RX 9000** | Uses AMD ROCm PyTorch wheels; Python 3.12 required. |

2. Start the application with `run.bat`.

### Linux

Install Python dependencies and a PyTorch build suitable for your hardware, then start the application with `./run.sh`. The launcher activates an existing `venv/` or `.venv/` automatically. See [PyTorch's installation selector](https://pytorch.org/get-started/locally/) and AMD's current ROCm documentation for compatible packages.

### Starting generation and the TTS server

For llama-server modes, **you do not need to start the server from a terminal or run a separate server launcher**. Start the GUI with `run.bat` on Windows or `./run.sh` on Linux, select the export, character and voice settings, then press **Start**. If the required llama-server is not already running, the GUI prepares/starts it automatically and continues with generation when it is ready. The first server start downloads llama.cpp and the model (about 3 GB).

The GUI also has a separate **Start server** button if you want to start the backend before beginning a generation run. `START_TTS_SERVER.bat` is an optional Windows utility for starting only the server without the GUI.

For llama-server, **Auto** selects CUDA on NVIDIA and Vulkan otherwise. You can also choose Vulkan or CUDA explicitly. Mode C and generating a new VoiceDesign sample require the PyTorch engine.

### AMD RX 6000 / RDNA2 on Windows

This project can use the separate [vllm-rocm-windows-rdna2 TheRock runtime](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2). Follow that repository's current requirements and installation instructions. Its `INSTALL.bat -SkipModel` option skips that project's LLM download, which this voice cloner does not use; the runtime installer may still need substantial temporary free space on `C:`. The project installer can set up this runtime as choice 3. `start_rocm_rdna2.bat` launches the GUI using the runtime's Python and GPU libraries; `run.bat` delegates to it when that engine is selected.

One run on an RX 6700 XT reported 40.4 lines/min with PyTorch/ROCm/GPU and 15.1 lines/min with llama-server/Vulkan. This is a measurement from one machine and workload, not a general performance guarantee. RX 6000 Windows users can try ROCm and compare it with Vulkan on their own system.

## Creation Kit export and workflow

The export is expected to be tab-separated and include:

- Required: `VOICE TYPE`, `FILENAME`, `RESPONSE TEXT`
- Optional: `FULL PATH` (plugin/output folder), `FACE` (emotion), `SCRIPT NOTES` (acting direction)

Name the export with `export_ck` or `DialogueExport` so the GUI can find it, or select it with **Browse…**. The GUI also searches Downloads, Documents, Desktop and the project folder. Default text encoding is Windows-1252; use `SVC_EXPORT_ENCODING` to override it.

1. Select the export, mod/plugin and character (`VOICE TYPE`).
2. Select voice mode A, B or C and configure its reference, design or speaker settings.
3. Choose performance settings.
4. Press **Start**. The GUI generates missing/changed lines, checks the audio and finalises it. Use **Open output** to view the result.

For modes A and B, provide a reference WAV. Game voice assets may need to be extracted and converted from `.wem` to `.wav` with an audio conversion tool. Mode C does not need a reference recording.

Reference audio and voice settings are saved per `VOICE TYPE` in `gui_state.json` and shared across plugins using that voice type.

## Performance and output

**llama-server** uses the Qwen3-TTS Base GGUF model. Available performance choices are GPU, GPU + CPU hybrid, and CPU-only. The first server start downloads llama.cpp and the model. The GUI reports server status and warns if the backend appears to use the CPU instead of the GPU.

**PyTorch** provides GPU generation and configurable worker/batch settings. The default GPU configuration is one GPU worker with batch 4. Precision `auto` tries float16 and falls back to float32 if needed. Watch VRAM when changing worker or batch settings. **Test GPU** checks the installed PyTorch GPU; it is not available in a llama-only install.

Final output is 32 kHz mono WAV, stored per plugin and character:

```text
work/<Plugin>/<VOICE TYPE>/                 raw WAVs, manifest.json, report.txt
output/Sound/Voice/<Plugin>/<VOICE TYPE>/    final WAVs
```

The tool does not convert WAVs into Starfield's final game/archive audio format.

## Building the Windows executable

`build_exe.bat` is a **build script**, not a Python installer or the normal application launcher. First set up the project with `INSTALL.bat`, then run `build_exe.bat`. It uses `_env.bat` to find the selected Python environment, installs PyInstaller into it and creates this **onedir** build:

```text
dist/StarfieldVoiceCloner/StarfieldVoiceCloner.exe
```

Keep the entire `dist/StarfieldVoiceCloner/` directory together; this is not a standalone single-file executable. The build intentionally excludes PyTorch and `qwen-tts`, so it is the llama-server edition: modes A and B-clone work, but mode C, **Test GPU**, and creating a new VoiceDesign sample require the regular Python installation with PyTorch. An already-created VoiceDesign WAV can still be used for B-clone. The build script copies `README.md` and `LICENSE` into the output directory.

## Useful files

| File | Purpose |
|---|---|
| `INSTALL.bat` | Windows environment and engine setup. |
| `run.bat`, `run.sh` | Start the GUI in the configured Python environment. |
| `build_exe.bat` | Build the PyInstaller Windows onedir executable. |
| `START_TTS_SERVER.bat` | Optional: start only the llama-server backend on Windows. |
| `CHECK_PROJECT.bat`, `check_project.py` | Check the local setup and GPU information. |
| `start_rocm_rdna2.bat` | Launch the GUI with the external TheRock runtime. |
| `qwen_gui.py` | GUI and generation workflow. |
| `generate_voice.py` | Generate dialogue audio. |
| `finalize_audio.py` | Validate and finalise audio. |
| `common.py` | CK export parsing, emotion mapping and output paths. |
| `server_manager.py` | Download and run llama.cpp and its model. |
| `install_helper.py` | Install optional PyTorch engines selected by `INSTALL.bat`. |
| `requirements.txt` | Audio/GUI/Qwen dependencies; PyTorch is installed separately. |
| `requirements-llama.txt` | Lightweight dependencies for llama-server mode. |

## Troubleshooting

- **No export found:** use **Browse…**, or include `export_ck`/`DialogueExport` in the filename and press **Rescan**.
- **AMD PyTorch cannot see the GPU:** verify that the installed PyTorch build is ROCm, and check the ROCm runtime and driver. On Linux, check `/dev/kfd` and `video`/`render` groups. Use **Test GPU** where available.
- **HIP / invalid device function:** the ROCm runtime may not contain kernels for the selected GPU/override. Try **Auto** or another supported setting.
- **Out of GPU memory:** reduce GPU workers or batch size; watch VRAM. Try llama-server if available.
- **llama-server appears to use CPU:** try the other backend, restart the server and inspect `tts_backend/server.log`.
- **Mode C or VoiceDesign is unavailable:** install PyTorch using the matching `INSTALL.bat` choice or a compatible manual setup.
- **Whisper transcription fails:** install `faster-whisper`; the first transcription downloads the `base.en` model and needs internet access.
- **TheRock runtime not found:** install it separately or set `THEROCK_ROOT`, then start with `start_rocm_rdna2.bat`.
- **Executable build fails:** run `INSTALL.bat` first and make sure `_env.bat` can find its selected Python environment.

## Responsible use and license

Only clone voices you are authorised to use. Respect Bethesda/Microsoft terms, voice actors' rights and local laws. Do not redistribute extracted game audio or imitate real people without permission. The code in this repository is licensed under the [MIT License](LICENSE); models and game assets have separate licences. This project is not affiliated with Bethesda, AMD, NVIDIA or Alibaba.

---

## Slovenčina

GUI je dostupné **v slovenčine aj angličtine**; jazyk zmeníš v prepínači **Jazyk/Language**.

- **A — Klon z referenčného WAV:** použije nahrávku a jej prepis; voliteľne možno pridať emočné vzorky podľa `FACE`.
- **B — Nový hlas:** vytvorí ukážku z popisu hlasu a použije ju na klonovanie replík.
- **C — Vstavaný hlas:** vyberieš speakera a emóciu riadkov určuje CK `FACE`; možno pridať aj `SCRIPT NOTES`.

Na Windowse spusti `INSTALL.bat`, vyber engine a potom `run.bat`. Na Linuxe priprav Python prostredie a spusti `./run.sh`. Pri generovaní v režime llama-server GUI server spustí automaticky, ak ešte nebež�� — netreba ho samostatne štartovať z konzoly. Vyber CK export, plugin, postavu a hlasový režim, potom stlač **Štart**. Výsledné WAV súbory nájdeš v `output/Sound/Voice/<Plugin>/<VOICE TYPE>/`.

### Vytvorenie `.exe`

`build_exe.bat` cez **PyInstaller** zostaví Windows aplikáciu z Python prostredia pripraveného cez `INSTALL.bat`. Výsledkom je celý priečinok `dist/StarfieldVoiceCloner/`, nie jeden samostatný `.exe` súbor. Build neobsahuje PyTorch ani `qwen-tts`; podporuje llama-server klonovanie v režimoch A a B-klon, ale nie režim C ani vytvorenie novej VoiceDesign vzorky.

Na RX 6700 XT bolo v jednom meraní nameraných 40,4 replík/min pri PyTorch/ROCm/GPU a 15,1 replík/min pri llama-server/Vulkan. Je to výsledok konkrétneho počítača; výkon sa môže líšiť. Používatelia RX 6000 na Windowse môžu skúsiť [TheRock ROCm runtime](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2).
