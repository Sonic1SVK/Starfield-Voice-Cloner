"""Generate dialogue lines with Qwen3-TTS for one <mod plugin>/<VOICE TYPE> scope of a Starfield Creation Kit export.

Voice modes
  clone    Qwen3-TTS-12Hz-1.7B-Base        clone a reference WAV (--ref-wav/--ref-text). Emotion-tagged samples such as
                                            angry.wav / sad.wav (--emotion-refs JSON) are used for lines with that FACE.
  custom   Qwen3-TTS-12Hz-1.7B-CustomVoice built-in speaker (--speaker); the CK FACE column becomes a per-line
                                            `instruct` prompt (VeryAngry -> "Speak very angrily ...").
  (a new voice is designed with --design-sample, then cloned with mode "clone")

Usage
  python generate_voice.py export_ck.txt --voice-type NPCFAndreja [--plugin "My Mod.esm"] \\
         --mode clone --ref-wav sample.wav --ref-text "Sample transcript"
  python generate_voice.py export_ck.txt --voice-type NPCFAndreja --mode custom --speaker Ryan
  python generate_voice.py --design-sample sample.wav --design-instruct "Female voice, ..." --design-text "..."
  python generate_voice.py --manifest work/<Plugin>/<VoiceType>/manifest.json --mode clone ...   (what the GUI does)

Raw audio goes to work/<Plugin>/<VoiceType>/ ; finalize_audio.py writes output/Sound/Voice/<Plugin>/<VoiceType>/.

Performance is configured through TTS_* environment variables (the GUI sets them):
  TTS_GPU_WORKERS / TTS_CPU_WORKERS   number of worker processes (GPU workers get IDs first)
  TTS_GPU_BATCH / TTS_CPU_BATCH       lines generated at once per worker
  TTS_GPU_THREADS / TTS_CPU_THREADS   CPU threads per worker
  TTS_DTYPE      auto | float16 | bfloat16 | float32   (GPU only; always falls back to float32 if the card fails)
  TTS_HSA        auto | none | 10.3.0 | 10.3.1 | 10.3.2 | 10.3.4   AMD ROCm HSA_OVERRIDE_GFX_VERSION
  TTS_GPU_INDEX  which GPU to use (default 0)
  TTS_USE_NOTES  1 = append the SCRIPT NOTES column to the per-line instruction (custom mode)
Without any TTS_* variables: 2 CPU workers, 6 threads each, float32.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import gc
import json
import multiprocessing as mp
import queue as pyqueue
import sys
import tempfile

from common import (BASE, add_scope_args, auto_hsa_candidates, build_instruct, face_key, probe_hsa, resolve_hsa,
                    resolve_scope)

os.environ.setdefault("HF_HOME", str(BASE / "hf-cache"))
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAMPLE_RATE_OUT = 32000


def env_int(name, default):
    try:
        return max(int(os.environ.get(name, default)), 0)
    except ValueError:
        return default


GPU_WORKERS = env_int("TTS_GPU_WORKERS", 0)
CPU_WORKERS = env_int("TTS_CPU_WORKERS", 2)
GPU_THREADS = max(env_int("TTS_GPU_THREADS", 2), 1)
CPU_THREADS = max(env_int("TTS_CPU_THREADS", 6), 1)
GPU_BATCH = max(env_int("TTS_GPU_BATCH", 1), 1)
CPU_BATCH = max(env_int("TTS_CPU_BATCH", 1), 1)
GPU_INDEX = env_int("TTS_GPU_INDEX", 0)
REQUESTED_DTYPE = os.environ.get("TTS_DTYPE", "auto").lower()
USE_NOTES = os.environ.get("TTS_USE_NOTES", "0") == "1"


# ------------------------------------------------------------------ GPU environment
def resolve_auto_hsa():
    """HSA 'auto' on RDNA 2: pick the first override that really works (e.g. 10.3.1, else 10.3.0). Runs once in the parent."""
    if os.environ.get("TTS_HSA", "auto").strip().lower() != "auto" or "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        return
    candidates, gfx = auto_hsa_candidates()
    if not candidates:
        return
    chosen = candidates[0]
    if len(candidates) > 1:
        for c in candidates:
            ok = probe_hsa(c)
            print(f"[HSA] {gfx}: HSA_OVERRIDE_GFX_VERSION={c} -> {'works' if ok else 'failed'}", flush=True)
            if ok:
                chosen = c
                break
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = chosen


def prepare_gpu_env():
    """Environment for processes that will initialise the GPU (must run before `import torch`)."""
    setting = os.environ.get("TTS_HSA", "auto").strip().lower()
    if setting == "none":                                   # explicit "No override": ignore an inherited variable
        os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    hsa = resolve_hsa(setting)
    if hsa and (setting != "auto" or "HSA_OVERRIDE_GFX_VERSION" not in os.environ):
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def pick_device(torch, want_gpu, label):
    if want_gpu:
        if torch.cuda.is_available():
            return "cuda", f"cuda:{GPU_INDEX}"
        print(f"[{label}] GPU not available (torch.cuda.is_available() is False) -> using CPU", flush=True)
    return "cpu", "cpu"


def probe_dtype(torch, name, device):
    """True if the GPU can run a matmul in this dtype and produces finite numbers."""
    try:
        if name == "bfloat16" and hasattr(torch.cuda, "is_bf16_supported"):
            try:
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    return False
            except TypeError:
                if not torch.cuda.is_bf16_supported():
                    return False
        x = torch.randn(256, 256).to(device=device, dtype=getattr(torch, name))
        y = x @ x
        return bool(torch.isfinite(y.float().cpu()).all().item())
    except Exception:
        return False


def choose_dtype(torch, requested, device):
    """GPU precision with automatic fallback to float32. Returns (dtype_name, reason or '')."""
    order = ["float16"] if requested == "auto" else [requested]
    for name in order:
        if name == "float32":
            return "float32", ""
        if probe_dtype(torch, name, device):
            return name, ""
    return "float32", f"{order[0]} not usable on this GPU"


def load_qwen(Qwen3TTSModel, torch, name, kind, dtype_name):
    kw = dict(device_map=f"cuda:{GPU_INDEX}" if kind == "cuda" else "cpu", dtype=getattr(torch, dtype_name))
    if kind == "cuda":
        kw["attn_implementation"] = "sdpa"
    try:
        return Qwen3TTSModel.from_pretrained(name, **kw)
    except Exception:
        if "attn_implementation" not in kw:
            raise
        kw.pop("attn_implementation")
        return Qwen3TTSModel.from_pretrained(name, **kw)


# ------------------------------------------------------------------ queue + generation
def take_batch(queue, n):
    """Take up to n lines from the shared queue. Returns (batch, finished)."""
    first = queue.get()
    if first is None:
        return [], True
    batch = [first]
    while len(batch) < n:
        try:
            nxt = queue.get(timeout=0.2)
        except pyqueue.Empty:
            break
        if nxt is None:
            return batch, True
        batch.append(nxt)
    return batch, False


def call_model(model, mode, items, prompt, speaker):
    """One Qwen3-TTS call for a list of line dicts that share the same reference prompt."""
    texts = [it["text"] for it in items]
    if mode == "custom":
        if len(items) == 1:
            kw = {"instruct": items[0]["instruct"]} if items[0]["instruct"] else {}
            return model.generate_custom_voice(text=texts[0], language="English", speaker=speaker, **kw)
        return model.generate_custom_voice(text=texts, language=["English"] * len(items),
                                           speaker=[speaker] * len(items), instruct=[it["instruct"] for it in items])
    if len(items) == 1:
        return model.generate_voice_clone(text=texts[0], language="English", voice_clone_prompt=prompt)
    return model.generate_voice_clone(text=texts, language=["English"] * len(items), voice_clone_prompt=prompt)


def generate_items(model, mode, items, prompt, speaker, torch, cuda):
    """Returns (list of waveforms or None per line, sample rate). Falls back to line-by-line if a batch fails."""
    try:
        wavs, sr = call_model(model, mode, items, prompt, speaker)
        return list(wavs), sr
    except Exception as e:
        if len(items) == 1:
            raise
        print(f"   batch failed ({type(e).__name__}: {str(e)[:80]}), retrying line by line...", flush=True)
        if cuda:
            torch.cuda.empty_cache()
        out, sr = [], SAMPLE_RATE_OUT
        for it in items:
            try:
                w, sr = call_model(model, mode, [it], prompt, speaker)
                out.append(w[0])
            except Exception as e2:
                print(f"   line failed: {type(e2).__name__}: {str(e2)[:80]}", flush=True)
                out.append(None)
        return out, sr


def run_worker(worker_id, device, threads, batch_size, total, queue, cfg):
    worker_temp = BASE / f"temp_clone_{worker_id}"
    worker_temp.mkdir(parents=True, exist_ok=True)
    for k in ("TMPDIR", "TEMP", "TMP"):
        os.environ[k] = str(worker_temp)
    tempfile.tempdir = str(worker_temp)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    if device == "gpu":
        prepare_gpu_env()

    # Nothing to do -> do not even load the model
    batch, finished = take_batch(queue, batch_size)
    if not batch:
        print(f"[Worker {worker_id}] Nothing to generate, skipping model load.", flush=True)
        return

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(threads)
    kind, dev = pick_device(torch, device == "gpu", f"Worker {worker_id}")
    ctx = {"kind": kind, "batch": batch_size}
    if kind == "cpu":
        ctx["batch"] = min(batch_size, CPU_BATCH)
        torch.set_num_threads(CPU_THREADS)
        dtype_name = "float32"
    else:
        dtype_name, why = choose_dtype(torch, REQUESTED_DTYPE, dev)
        if why:
            print(f"[Worker {worker_id}] dtype fallback -> float32 ({why})", flush=True)
    backend = "CPU" if kind == "cpu" else ("ROCm" if getattr(torch.version, "hip", None) else "CUDA")
    mode = cfg["mode"]
    model_name = MODEL_CUSTOM if mode == "custom" else MODEL_BASE
    holder = {}

    def load(dtype):
        holder.clear()
        gc.collect()
        if kind == "cuda":
            torch.cuda.empty_cache()
        holder["model"] = load_qwen(Qwen3TTSModel, torch, model_name, kind, dtype)
        holder["prompts"] = {}
        holder["dtype"] = dtype

    def get_prompt(key):
        """Voice-clone prompt of a reference ('default' or an emotion key), created on first use."""
        if mode != "clone":
            return None
        if key not in cfg["refs"]:
            key = "default"
        if key not in holder["prompts"]:
            ref = cfg["refs"][key]
            holder["prompts"][key] = holder["model"].create_voice_clone_prompt(
                ref_audio=ref["wav"], ref_text=ref["text"], x_vector_only_mode=False)
        return holder["prompts"][key]

    print(f"[Worker {worker_id}] Loading {model_name.split('/')[-1]} ({backend}, {dtype_name}, batch {ctx['batch']}) "
          f"lines to generate: {total}", flush=True)
    load(dtype_name)
    get_prompt("default")

    def synthesize(items):
        """Generate a batch (grouped by reference sample) and convert to 32 kHz mono float32 arrays."""
        groups = {}
        for idx, it in enumerate(items):
            groups.setdefault(it["refkey"] if mode == "clone" else "default", []).append((idx, it))
        audios = [None] * len(items)
        for key, grp in groups.items():
            with torch.inference_mode():
                wavs, sr = generate_items(holder["model"], mode, [g[1] for g in grp], get_prompt(key),
                                          cfg.get("speaker"), torch, kind == "cuda")
            for (idx, _), w in zip(grp, wavs):
                if w is None:
                    continue
                if hasattr(w, "detach"):
                    w = w.detach().float().cpu().numpy()
                a = np.asarray(w, dtype=np.float32)
                if a.ndim > 1:
                    a = np.mean(a, axis=-1)
                if sr != SAMPLE_RATE_OUT:
                    a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE_OUT)
                audios[idx] = a
        return audios

    def audio_ok(a):
        return len(a) > 0 and bool(np.isfinite(a).all()) and float(np.sqrt(np.mean(np.square(a)))) > 1e-4

    done = 0
    while True:
        if batch:
            audios = None
            for _attempt in range(2):
                try:
                    audios = synthesize(batch)
                    if kind == "cuda" and holder["dtype"] != "float32" \
                            and any(a is not None and not audio_ok(a) for a in audios):
                        raise FloatingPointError(f"invalid audio (NaN/silence) in {holder['dtype']}")
                    break
                except Exception as e:
                    audios = None
                    msg = f"{type(e).__name__}: {str(e)[:100]}"
                    oom = "out of memory" in str(e).lower()
                    if kind == "cuda" and holder.get("dtype") != "float32" and not oom:
                        try:
                            print(f"[Worker {worker_id}] {msg} -> falling back to float32", flush=True)
                            load("float32")
                            get_prompt("default")
                            continue
                        except Exception as e2:
                            msg = f"{type(e2).__name__}: {str(e2)[:100]}"
                    print(f"[Worker {worker_id}] ERROR: {msg}", flush=True)
                    if kind == "cuda":
                        torch.cuda.empty_cache()
                    break
            for it, a in zip(batch, audios or []):
                if a is None:
                    continue
                tmp_wav = worker_temp / f"{it['fn']}.wav"
                sf.write(str(tmp_wav), a, SAMPLE_RATE_OUT, format="WAV", subtype="PCM_16")
                os.replace(tmp_wav, os.path.join(cfg["out_dir"], f"{it['fn']}.wav"))
                done += 1
                print(f"[Worker {worker_id}] [{done}/{total}] {it['fn']}.wav -> {it['text'][:45]}...", flush=True)
            gc.collect()
        if finished:
            break
        batch, finished = take_batch(queue, ctx["batch"])
        if not batch:
            break

    print(f"[Worker {worker_id}] Done!", flush=True)


# ------------------------------------------------------------------ voice design sample (mode B)
def run_design_sample(args):
    """Generate one sample WAV with the VoiceDesign model; the GUI then clones it with the Base model."""
    want_gpu = GPU_WORKERS > 0
    threads = GPU_THREADS if want_gpu else CPU_THREADS
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)
    if want_gpu:
        resolve_auto_hsa()
        prepare_gpu_env()

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(threads)
    kind, dev = pick_device(torch, want_gpu, "DESIGN")
    dtype_name = "float32"
    if kind == "cuda":
        dtype_name, why = choose_dtype(torch, REQUESTED_DTYPE, dev)
        if why:
            print(f"[DESIGN] dtype fallback -> float32 ({why})", flush=True)

    tries = [dtype_name] + (["float32"] if dtype_name != "float32" else [])
    audio, last_err = None, ""
    for i, dt in enumerate(tries):
        try:
            print(f"[DESIGN] Loading {MODEL_DESIGN.split('/')[-1]} ({kind}, {dt})", flush=True)
            model = load_qwen(Qwen3TTSModel, torch, MODEL_DESIGN, kind, dt)
            with torch.inference_mode():
                wavs, sr = model.generate_voice_design(text=args.design_text, language="English",
                                                       instruct=args.design_instruct)
            w = wavs[0]
            if hasattr(w, "detach"):
                w = w.detach().float().cpu().numpy()
            a = np.asarray(w, dtype=np.float32)
            if a.ndim > 1:
                a = np.mean(a, axis=-1)
            if sr != SAMPLE_RATE_OUT:
                a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE_OUT)
            if not (len(a) and np.isfinite(a).all() and float(np.sqrt(np.mean(np.square(a)))) > 1e-4):
                raise FloatingPointError(f"invalid audio (NaN/silence) in {dt}")
            audio = a
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"[DESIGN] {dt} failed: {last_err}", flush=True)
            gc.collect()
            if kind == "cuda":
                torch.cuda.empty_cache()
    if audio is None:
        print(f"[DESIGN] ERROR: {last_err}", flush=True)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.design_sample)), exist_ok=True)
    sf.write(args.design_sample, audio, SAMPLE_RATE_OUT, format="WAV", subtype="PCM_16")
    print(f"[DESIGN] saved {args.design_sample} ({len(audio) / SAMPLE_RATE_OUT:.1f} s)", flush=True)
    return 0


# ------------------------------------------------------------------ main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scope_args(ap)
    ap.add_argument("--mode", choices=("clone", "custom"), default="clone",
                    help="clone = Base model + reference WAV; custom = CustomVoice model + --speaker")
    ap.add_argument("--ref-wav", default=None, help="reference voice sample (.wav, ~6-12 s) for --mode clone")
    ap.add_argument("--ref-text", default=None, help="exact transcript of the reference sample")
    ap.add_argument("--emotion-refs", default=None,
                    help='JSON {"veryangry": {"wav": "...", "text": "..."}, ...} reference samples per FACE (clone mode)')
    ap.add_argument("--speaker", default=None,
                    help="CustomVoice speaker: Ryan, Aiden, Vivian, Serena, Dylan, Eric, Ono_Anna, Sohee, Uncle_Fu")
    ap.add_argument("--design-sample", default=None, help="write a VoiceDesign sample WAV to this path and exit")
    ap.add_argument("--design-instruct", default=None, help="voice description for --design-sample")
    ap.add_argument("--design-text", default=None, help="text spoken in the design sample")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.design_sample:
        if not (args.design_instruct and args.design_text):
            print("ERROR: --design-sample needs --design-instruct and --design-text", flush=True)
            return 2
        return run_design_sample(args)

    manifest, paths = resolve_scope(args)
    out_dir = str(paths["out"])
    os.makedirs(out_dir, exist_ok=True)

    refs = {}
    if args.mode == "clone":
        if not (args.ref_wav and args.ref_text and os.path.isfile(args.ref_wav)):
            print("ERROR: clone mode needs an existing --ref-wav and its --ref-text", flush=True)
            return 2
        refs["default"] = {"wav": args.ref_wav, "text": args.ref_text}
        if args.emotion_refs and os.path.isfile(args.emotion_refs):
            with open(args.emotion_refs, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if v.get("wav") and v.get("text") and os.path.isfile(v["wav"]):
                        refs[k] = {"wav": v["wav"], "text": v["text"]}
    elif not args.speaker:
        print("ERROR: custom mode needs --speaker", flush=True)
        return 2

    scope = f"{manifest['plugin'] or '(no plugin path)'} / {manifest['voice_type']}"
    lines = manifest["lines"]
    print(f"Unique lines for {scope}: {len(lines)}", flush=True)
    items = []
    for fn, ln in lines.items():
        if os.path.exists(os.path.join(out_dir, f"{fn}.wav")):
            continue
        fk = face_key(ln.get("face", ""))
        items.append({"fn": fn, "text": ln["text"], "face": ln.get("face", ""),
                      "instruct": build_instruct(ln.get("face", ""), ln.get("notes", ""), USE_NOTES),
                      "refkey": fk if fk in refs else "default"})
    # clone: group lines by reference sample first, longest first inside a group (batches share one prompt)
    items.sort(key=lambda it: (it["refkey"], -len(it["text"])) if args.mode == "clone" else (0, -len(it["text"])))
    print(f"Missing, to generate: {len(items)}", flush=True)
    if args.mode == "clone" and len(refs) > 1:
        used = sorted({it["refkey"] for it in items if it["refkey"] != "default"})
        print(f"Emotion reference samples in use: {', '.join(used) if used else 'none for the missing lines'}", flush=True)

    # GPU workers get IDs 0..g-1, CPU workers g..g+c-1 (the GUI relies on this order)
    workers = [("gpu", GPU_THREADS, GPU_BATCH)] * GPU_WORKERS + [("cpu", CPU_THREADS, CPU_BATCH)] * CPU_WORKERS
    if not workers:
        workers = [("cpu", CPU_THREADS, CPU_BATCH)] * 2
    if items and any(w[0] == "gpu" for w in workers):
        resolve_auto_hsa()

    cfg = {"mode": args.mode, "refs": refs, "speaker": args.speaker, "out_dir": out_dir}
    queue = mp.Queue()
    for it in items:
        queue.put(it)
    for _ in workers:
        queue.put(None)

    procs = []
    for wid, (device, threads, batch) in enumerate(workers):
        p = mp.Process(target=run_worker, args=(wid, device, threads, batch, len(items), queue, cfg))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
