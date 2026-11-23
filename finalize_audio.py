"""Check and finalize generated lines of one <mod plugin>/<VOICE TYPE> scope to 32 kHz mono WAV.

Reads   work/<Plugin>/<VoiceType>/<FILENAME>.wav                       (raw output of generate_voice.py)
Writes  output/Sound/Voice/<Plugin>/<VoiceType>/<FILENAME>.wav         (padding + loudness normalisation)
        work/<Plugin>/<VoiceType>/report.txt

Usage:
  python finalize_audio.py export_ck.txt --voice-type NPCFAndreja [--plugin "My Mod.esm"]
  python finalize_audio.py --manifest work/<Plugin>/<VoiceType>/manifest.json
Exit code 1 means some files were silent/broken and were deleted so the next generation run recreates them.
"""
import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf

from common import BASE, add_scope_args, resolve_scope

TARGET_SR = 32000
LEAD_IN_SEC = 0.15
LEAD_OUT_SEC = 0.25
TARGET_RMS = 0.13
MAX_GAIN = 3.0
MAX_PEAK = 0.90


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scope_args(ap)
    args = ap.parse_args()

    manifest, paths = resolve_scope(args)
    src_dir, final_dir, report_txt = str(paths["out"]), str(paths["final"]), str(paths["report"])
    os.makedirs(final_dir, exist_ok=True)
    valid = {fn: ln["text"] for fn, ln in manifest["lines"].items()}

    # Remove stale finals of this scope that are no longer part of the manifest
    for existing in os.listdir(final_dir):
        if existing.endswith(".wav") and existing[:-4] not in valid:
            os.remove(os.path.join(final_dir, existing))

    ok_count = 0
    problems = []
    lead_in = np.zeros(int(TARGET_SR * LEAD_IN_SEC), dtype=np.float32)
    lead_out = np.zeros(int(TARGET_SR * LEAD_OUT_SEC), dtype=np.float32)

    for filename, text in valid.items():
        src = os.path.join(src_dir, f"{filename}.wav")
        dst = os.path.join(final_dir, f"{filename}.wav")

        if not os.path.exists(src):
            problems.append((filename, "MISSING_FILE", 0.0, text))
            continue

        try:
            audio, sr = sf.read(src, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            length = len(audio) / TARGET_SR
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0
            peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

            if not np.isfinite(rms) or length < 0.35 or rms < 0.002 or peak < 0.015:
                problems.append((filename, f"SILENT_OR_TOO_SHORT (length={length:.2f}s, rms={rms:.5f})", 0.0, text))
                os.remove(src)  # delete the broken file so the next run regenerates it
                continue

            desired_gain = min(TARGET_RMS / rms, MAX_GAIN)
            gain = min(desired_gain, MAX_PEAK / peak)
            audio = audio * gain

            sf.write(dst, np.concatenate([lead_in, audio, lead_out]), TARGET_SR, format="WAV", subtype="PCM_16")
            ok_count += 1

        except Exception as e:
            problems.append((filename, f"READ_ERROR ({e})", 0.0, text))
            if os.path.exists(src):
                os.remove(src)

    rel_final = os.path.relpath(final_dir, str(BASE))
    with open(report_txt, "w", encoding="utf-8") as rep:
        rep.write(f"=== CHECK AND FINALIZATION REPORT ({manifest['plugin'] or 'no plugin path'} / "
                  f"{manifest['voice_type']}) ===\n")
        rep.write(f"Valid lines in manifest: {len(valid)}\n")
        rep.write(f"Finalized in {rel_final}: {ok_count}\n")
        rep.write(f"Broken / silent (deleted for regeneration): {len(problems)}\n\n")
        if problems:
            rep.write("FILENAME\tREASON\tRMS\tTEXT\n")
            for fn, reason, rms, txt in problems:
                rep.write(f"{fn}\t{reason}\t{rms:.5f}\t{txt}\n")

    print(f"\n[FINALIZATION] Done: {ok_count}/{len(valid)} | Problems: {len(problems)}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
