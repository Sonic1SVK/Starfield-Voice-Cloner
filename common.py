"""Shared helpers for the Starfield voice cloner: portable paths, Creation Kit export parsing (incl. FULL PATH, FACE and
SCRIPT NOTES), per-mod manifests and AMD ROCm override detection.

No heavy imports here (no torch / Qt), so it is cheap to import from every script.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# All working data (work/, output/, state, caches) lives next to the scripts unless SVC_DATA_DIR is set.
BASE = Path(os.environ.get("SVC_DATA_DIR")
            or (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent))

# Creation Kit exports are Windows-1252 tab-separated text files.
EXPORT_ENCODING = os.environ.get("SVC_EXPORT_ENCODING", "cp1252")

NO_PLUGIN = "_no_plugin"       # folder used when a row has no usable FULL PATH


# ------------------------------------------------------------------ path handling
def safe_component(name):
    """Make one path component safe on Windows and Linux while keeping spaces and dots ('Amber Personal quest.esm')."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    s = s.strip(" .")
    return s or "_"


def plugin_from_full_path(full_path):
    """'Data\\Sound\\Voice\\Amber Personal quest.esm\\NPCFJaedaHuang\\00924DDD.wem' -> 'Amber Personal quest.esm'.

    Returns '' when the path has no Sound\\Voice\\<plugin> part.
    """
    parts = [p for p in re.split(r"[\\/]+", (full_path or "").strip()) if p]
    low = [p.lower() for p in parts]
    for i in range(len(low) - 2):
        if low[i] == "sound" and low[i + 1] == "voice":
            return parts[i + 2]
    return ""


def scope_paths(plugin, voice_type):
    """Folders of one <plugin>/<voice type> scope.

    work   work/<Plugin>/<VoiceType>/                       raw generated wavs, manifest.json, report, refs
    final  output/Sound/Voice/<Plugin>/<VoiceType>/         finalized 32 kHz mono wavs (mirrors the game path)
    """
    p = safe_component(plugin) if plugin else NO_PLUGIN
    v = safe_component(voice_type)
    work = BASE / "work" / p / v
    return dict(work=work, out=work, final=BASE / "output" / "Sound" / "Voice" / p / v,
                report=work / "report.txt", manifest=work / "manifest.json")


# ------------------------------------------------------------------ FACE (emotion) -> instruction prompt
FACE_INSTRUCTIONS = {
    "Neutral": "Speak in a natural, calm, conversational tone.",
    "Serious": "Speak in a serious, focused, no-nonsense tone.",
    "Stern": "Speak in a firm, stern, commanding tone.",
    "Thinking": "Speak thoughtfully, as if weighing your words.",
    "Surprised": "Speak with genuine surprise and astonishment.",
    "Grateful": "Speak warmly with genuine gratitude.",
    "Worried": "Speak with anxious worry and concern.",
    "SinisterSmile": "Speak with a sly, darkly amused, sinister smirk in your voice.",
    "Somber": "Speak in a heavy, somber, mournful tone.",
    "Sad": "Speak with sadness, softly and with a heavy heart.",
    "VeryAngry": "Speak very angrily with sharp, intense emotion.",
    "Tired": "Speak in an exhausted, weary, drained voice.",
    "Happy": "Speak cheerfully with a bright, happy, upbeat tone.",
    "Fear": "Speak with fear, tense and trembling, almost breathless.",
    "Disgust": "Speak with clear disgust and distaste.",
    "Puzzled": "Speak with puzzlement and confusion, as if trying to understand.",
}

# File names (without extension) that count as an emotion-tagged reference sample for a FACE value.
EMOTION_ALIASES = {
    "Neutral": ("neutral", "calm"),
    "Serious": ("serious",),
    "Stern": ("stern", "serious"),
    "Thinking": ("thinking", "thoughtful"),
    "Surprised": ("surprised", "surprise"),
    "Grateful": ("grateful", "thankful"),
    "Worried": ("worried", "concerned", "anxious"),
    "SinisterSmile": ("sinistersmile", "sinister", "smirk"),
    "Somber": ("somber", "sombre"),
    "Sad": ("sad", "sorrow"),
    "VeryAngry": ("veryangry", "angry", "furious"),
    "Tired": ("tired", "weary"),
    "Happy": ("happy", "joy", "cheerful"),
    "Fear": ("fear", "scared", "afraid"),
    "Disgust": ("disgust", "disgusted"),
    "Puzzled": ("puzzled", "confused"),
}


def face_key(face):
    """Case/space-insensitive key: 'Very Angry' / 'veryangry' / 'VeryAngry' -> 'veryangry'."""
    return re.sub(r"[^a-z0-9]", "", (face or "").lower())


_FACE_BY_KEY = {face_key(k): k for k in FACE_INSTRUCTIONS}


def canonical_face(face):
    face = (face or "").strip()
    return _FACE_BY_KEY.get(face_key(face), face)


def face_instruct(face):
    """Natural-language instruction for a CK FACE value ('' for an empty value)."""
    face = canonical_face(face)
    if not face:
        return ""
    if face in FACE_INSTRUCTIONS:
        return FACE_INSTRUCTIONS[face]
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", face).lower()
    return f"Speak with a {words} expression in your voice."


def build_instruct(face, notes="", use_notes=False):
    """Per-line instruction for the CustomVoice model: FACE prompt (neutral if missing) + optional SCRIPT NOTES."""
    text = face_instruct(face) or FACE_INSTRUCTIONS["Neutral"]
    if use_notes and (notes or "").strip():
        text += f" Acting direction: {notes.strip()}"
    return text


def find_emotion_refs(folder):
    """{face_key: wav path} for emotion-tagged samples in a reference folder (angry.wav, sad.wav, serious.wav ...).

    Matches file stems against EMOTION_ALIASES; the shallowest file wins when a name occurs more than once.
    """
    if not folder or not os.path.isdir(folder):
        return {}
    alias_to_face = {}
    for face, aliases in EMOTION_ALIASES.items():
        for a in aliases:
            alias_to_face.setdefault(a, face)
    best = {}                                   # face -> (priority, depth, path)
    root_depth = str(folder).rstrip("\\/").count(os.sep)
    for dp, _dn, files in os.walk(folder):
        depth = dp.rstrip("\\/").count(os.sep) - root_depth
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() != ".wav":
                continue
            stem_key = face_key(stem)
            face = alias_to_face.get(stem_key)
            if not face:
                continue
            exact = 0 if stem_key == face_key(face) else 1       # 'veryangry.wav' beats 'angry.wav'
            cand = (exact, depth, os.path.join(dp, name))
            if face not in best or cand < best[face]:
                best[face] = cand
    return {face_key(f): v[2] for f, v in best.items()}


# ------------------------------------------------------------------ export parsing
_AUDIO_EXT = re.compile(r"\.(wem|wav|fuz|xwm|lip|ogg|mp3)$", re.I)


def clean_text(raw):
    text = (raw or "").strip().lstrip("> ").strip()
    if text.endswith('"') and text.count('"') == 1:
        text = text[:-1].strip()
    return text


def iter_export(path):
    """Yield one dict per usable row of a Creation Kit dialogue export.

    Keys: plugin (from FULL PATH, '' if missing), voice_type, filename, text, face, notes.
    """
    import csv
    with open(path, "r", encoding=EXPORT_ENCODING, errors="replace", newline="") as f:
        for raw in csv.DictReader(f, delimiter="\t"):
            row = {(k or "").strip().upper(): (v or "") for k, v in raw.items()}
            vt = row.get("VOICE TYPE", "").strip()
            fn = _AUDIO_EXT.sub("", row.get("FILENAME", "").strip())
            text = clean_text(row.get("RESPONSE TEXT"))
            if not (vt and fn and text):
                continue
            yield dict(plugin=plugin_from_full_path(row.get("FULL PATH", "")), voice_type=vt, filename=fn, text=text,
                       face=canonical_face(row.get("FACE", "")), notes=row.get("SCRIPT NOTES", "").strip())


def _line(d):
    return {"text": d["text"], "face": d["face"], "notes": d["notes"], "instruct": face_instruct(d["face"])}


def parse_export_all(path):
    """{plugin: {voice_type: {filename: {text, face, notes, instruct}}}} for the whole export.

    `plugin` is the folder right after Sound\\Voice\\ in the FULL PATH column ('' when the column is missing).
    The first occurrence of a filename wins inside one (plugin, voice type) scope.
    """
    out = {}
    for d in iter_export(path):
        out.setdefault(d["plugin"], {}).setdefault(d["voice_type"], {}).setdefault(d["filename"], _line(d))
    return out


def parse_export(path, plugin, voice_type):
    return parse_export_all(path).get(plugin, {}).get(voice_type, {})


# ------------------------------------------------------------------ manifests (per <plugin>/<voice type>)
def new_manifest(plugin, voice_type):
    return {"version": 2, "plugin": plugin, "voice_type": voice_type, "lines": {}, "exports": {}}


def load_manifest(plugin, voice_type):
    p = scope_paths(plugin, voice_type)["manifest"]
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("plugin") == plugin and m.get("voice_type") == voice_type:
            m.setdefault("lines", {})
            m.setdefault("exports", {})
            return m
    except Exception:
        pass
    return new_manifest(plugin, voice_type)


def save_manifest(m):
    p = scope_paths(m["plugin"], m["voice_type"])["manifest"]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def preview_export(m, parsed):
    """(new, changed) counts of an export versus the manifest, without writing anything."""
    paths = scope_paths(m["plugin"], m["voice_type"])
    new = changed = 0
    for fn, line in parsed.items():
        old = m["lines"].get(fn)
        if old is None:
            if not (paths["out"] / f"{fn}.wav").exists():
                new += 1
        elif old.get("text") != line["text"] or old.get("face", "") != line["face"]:
            changed += 1
    return new, changed


def apply_export_to_manifest(m, parsed, export_path):
    """Merge an export into the manifest. Lines whose text/FACE changed lose their old audio so they are regenerated.

    Returns (new_lines, changed_filenames). Lines that vanished from the export stay in the manifest, so older results
    are never deleted by a smaller export.
    """
    paths = scope_paths(m["plugin"], m["voice_type"])
    new, changed = 0, []
    for fn, line in parsed.items():
        old = m["lines"].get(fn)
        if old is None:
            new += 1
        elif old.get("text") != line["text"] or old.get("face", "") != line["face"]:
            changed.append(fn)
            for d in (paths["out"], paths["final"]):
                try:
                    (d / f"{fn}.wav").unlink()
                except FileNotFoundError:
                    pass
        m["lines"][fn] = {"text": line["text"], "face": line["face"], "notes": line["notes"],
                          "seen": (old or {}).get("seen", time.time())}
    try:
        m["exports"][str(export_path)] = {"mtime": os.path.getmtime(export_path), "ts": time.time()}
    except OSError:
        pass
    return new, changed


def add_scope_args(ap):
    ap.add_argument("export", nargs="?", default=None, help="export_ck.txt (tab separated Creation Kit export)")
    ap.add_argument("--plugin", default=None,
                    help="mod/plugin folder from FULL PATH, e.g. 'Amber Personal quest.esm' (optional if unambiguous)")
    ap.add_argument("--voice-type", default=None, help="value of the VOICE TYPE column, e.g. NPCFJaedaHuang")
    ap.add_argument("--manifest", default=None, help="manifest.json of a scope (written by the GUI)")


def resolve_scope(args):
    """(manifest, paths) from --manifest, or from an export + --voice-type [+ --plugin]; merges the export."""
    if args.manifest and not args.export:
        m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        m.setdefault("lines", {})
        return m, scope_paths(m["plugin"], m["voice_type"])
    if not (args.export and args.voice_type):
        raise SystemExit("ERROR: give either --manifest, or an export file together with --voice-type")
    data = parse_export_all(args.export)
    found = {p: d[args.voice_type] for p, d in data.items() if args.voice_type in d}
    if not found:
        raise SystemExit(f"ERROR: VOICE TYPE '{args.voice_type}' not found in {args.export}")
    plugin = args.plugin
    if plugin is None:
        if len(found) > 1:
            raise SystemExit(f"ERROR: '{args.voice_type}' occurs in several mods, pass --plugin: "
                             + ", ".join(repr(p) for p in found))
        plugin = next(iter(found))
    if plugin not in found:
        raise SystemExit(f"ERROR: no lines of '{args.voice_type}' in mod {plugin!r}")
    m = load_manifest(plugin, args.voice_type)
    apply_export_to_manifest(m, found[plugin], args.export)
    save_manifest(m)
    return m, scope_paths(plugin, args.voice_type)


# ------------------------------------------------------------------ AMD ROCm: HSA_OVERRIDE_GFX_VERSION
HSA_VALUES = ("auto", "none", "10.3.0", "10.3.1", "10.3.2", "10.3.4")

# RDNA 2 chips other than Navi 21 (gfx1030) are not native ROCm targets. The override makes the runtime use the kernels
# of another target: 10.3.1 = gfx1031 (RX 6700 XT / 6750 XT / 6700; also what the TheRock Windows runtime is built
# for), 10.3.0 = gfx1030 (what stock ROCm wheels ship).
RDNA2_UNOFFICIAL = {"gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"}


def gfx_name(target_version):
    """KFD gfx_target_version (e.g. 100301) -> 'gfx1031'."""
    major, minor, step = target_version // 10000, (target_version // 100) % 100, target_version % 100
    return f"gfx{major}{minor:x}{step:x}"


def detect_amd_gfx():
    """Linux only (reads the amdkfd topology, no torch needed). Returns e.g. ['gfx1031']; [] elsewhere.

    SVC_GFX=gfx1031 forces the result (containers, WSL, or when detection does not work).
    """
    forced = os.environ.get("SVC_GFX", "").strip().lower()
    if forced:
        return [forced]
    found = []
    for p in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            for line in Path(p).read_text().splitlines():
                if line.startswith("gfx_target_version"):
                    v = int(line.split()[1])
                    if v > 0:
                        found.append(gfx_name(v))
        except (OSError, ValueError, IndexError):
            continue
    return found


def auto_hsa_candidates():
    """(candidate override values in order of preference, gfx). [] when no override is needed.

    gfx1031 -> 10.3.1 first, 10.3.0 as the fallback (validated by probe_hsa); other unofficial RDNA 2 -> 10.3.0.
    """
    for g in detect_amd_gfx():
        if g == "gfx1031":
            return ["10.3.1", "10.3.0"], g
        if g in RDNA2_UNOFFICIAL:
            return ["10.3.0"], g
    gfxs = detect_amd_gfx()
    return [], (gfxs[0] if gfxs else "")


def auto_hsa_override():
    """(override, gfx) used for display: the preferred candidate, '' if none is needed."""
    cands, gfx = auto_hsa_candidates()
    return (cands[0] if cands else ""), gfx


def resolve_hsa(setting):
    """'auto' -> preferred detected value, 'none'/'' -> '', otherwise the explicit value such as '10.3.1'."""
    setting = (setting or "auto").strip().lower()
    if setting == "auto":
        return auto_hsa_override()[0]
    if setting in ("none", ""):
        return ""
    return setting


def probe_hsa(value, python=None, timeout=180):
    """Does torch see the GPU and run a matmul with HSA_OVERRIDE_GFX_VERSION=value? Runs in a throw-away process."""
    code = ("import torch\n"
            "ok = torch.cuda.is_available()\n"
            "if ok:\n"
            "    x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item(); torch.cuda.synchronize()\n"
            "print('PROBE_OK' if ok else 'PROBE_NO_GPU')\n")
    env = dict(os.environ, HSA_OVERRIDE_GFX_VERSION=value)
    try:
        r = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                           env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return "PROBE_OK" in r.stdout
    except Exception:
        return False
