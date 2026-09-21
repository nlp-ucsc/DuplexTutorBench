"""Build the HCRC Map Task human-reference run for the speech_eval vap pipeline.

The Map Task corpus (CC-BY 4.0) is a task-oriented dyadic corpus: an instruction
**giver** describes a route to a **follower** — an asymmetric, goal-directed
interaction that mirrors our tutor↔student setting. We use it as the human
anchor for the distributional turn-taking realism metric.

This script turns the corpus into a synthetic ``duplex_output/`` run so it flows
through the *unchanged* vap scorer:

    <out_dir>/
      audio/<idx>/combined.wav     16 kHz stereo, L=giver(tutor), R=follower(student)
      aligned/maptask_gold.jsonl   conv_index + role-tagged word timings (NXT gold)

Then, from the Mac:

    rsync <remote>:<out_dir>/ duplex_output/maptask_ref/
    uv run python -m speech_eval --run-name maptask_ref --component vap \
        --vap-aligned-variant maptask_gold --vap-gpu <free>

Run this on the GPU box (audio + ffmpeg + numpy live there), e.g.

    python build_maptask.py --raw-dir /data/jhe516/maptask_raw \
        --out-dir /data/jhe516/duplex_output/maptask_ref

Self-contained: stdlib + numpy + ffmpeg only (no torch, no speech_eval import),
so it runs under any python3 on the remote with numpy available.

Data sources (downloaded on first run if absent):
  * NXT annotations zip: https://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip
  * dialogue audio:      https://groups.inf.ed.ac.uk/maptask/signals/dialogues/<id>.mix.wav

The .mix.wav files are genuine stereo with one speaker per channel (~20 dB
separation); the giver/follower→L/R order is auto-detected per dialogue by
correlating each channel's energy envelope with that role's word activity.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import urllib.request
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_maptask")

NXT_ZIP_URL = "https://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip"
SIGNALS_URL = "https://groups.inf.ed.ac.uk/maptask/signals/dialogues"
NXT_DIRNAME = "nxt"
TIMED_UNITS_REL = "nxt/maptaskv2-1/Data/timed-units"

# Group words into "segments" when the same-role silence between them exceeds
# this (cosmetic only — the vap scorer re-merges words by its own gap).
SEG_SPLIT_GAP_S = 0.5
ENV_HZ = 50  # energy-envelope rate for channel detection


# --- download ----------------------------------------------------------------
def ensure_nxt(raw_dir: Path) -> Path:
    """Ensure the NXT annotations are extracted; return the timed-units dir."""
    tu_dir = raw_dir / TIMED_UNITS_REL
    if tu_dir.is_dir():
        return tu_dir
    zip_path = raw_dir / "hcrcmaptask.nxtformatv2-1.zip"
    if not zip_path.is_file():
        logger.info("downloading NXT annotations -> %s", zip_path)
        raw_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NXT_ZIP_URL, zip_path)
    logger.info("extracting NXT annotations")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir / NXT_DIRNAME)
    if not tu_dir.is_dir():
        raise FileNotFoundError(f"timed-units dir not found after extract: {tu_dir}")
    return tu_dir


def list_dialogues(tu_dir: Path) -> list[str]:
    """Sorted distinct dialogue ids (e.g. q1ec1) from the timed-units files."""
    ids = set()
    for p in tu_dir.glob("*.timed-units.xml"):
        # <id>.<g|f>.timed-units.xml
        ids.add(p.name.split(".")[0])
    return sorted(ids)


def ensure_audio(raw_dir: Path, dialogue: str) -> Path:
    """Ensure ``<dialogue>.mix.wav`` is present; download it if not."""
    audio_dir = raw_dir / "signals" / "dialogues"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dst = audio_dir / f"{dialogue}.mix.wav"
    if not dst.is_file():
        url = f"{SIGNALS_URL}/{dialogue}.mix.wav"
        logger.info("downloading audio %s", url)
        urllib.request.urlretrieve(url, dst)
    return dst


# --- timed-units parsing -----------------------------------------------------
def parse_timed_units(path: Path) -> list[dict]:
    """Return word units ``{start, end, word, utt}`` (skipping <sil>/<noi>)."""
    if not path.is_file():
        return []
    root = ET.parse(path).getroot()
    units: list[dict] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "tu":
            continue
        word = (el.text or "").strip()
        if not word:
            continue
        try:
            s = float(el.attrib["start"])
            e = float(el.attrib["end"])
        except (KeyError, ValueError):
            continue
        if e <= s:
            continue
        units.append({"start": s, "end": e, "word": word, "utt": el.attrib.get("utt")})
    units.sort(key=lambda u: u["start"])
    return units


def units_to_segments(units: list[dict], role: str) -> list[dict]:
    """Group word units into aligned-schema segments (split on long pauses)."""
    segs: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if not cur:
            return
        words = [
            {
                "word": u["word"],
                "start_time": round(u["start"], 4),
                "end_time": round(u["end"], 4),
                "confidence": 1.0,
            }
            for u in cur
        ]
        segs.append(
            {
                "role": role,
                "text": " ".join(u["word"] for u in cur),
                "start_time": words[0]["start_time"],
                "end_time": words[-1]["end_time"],
                "words": words,
            }
        )

    for u in units:
        if cur and (u["start"] - cur[-1]["end"] > SEG_SPLIT_GAP_S):
            flush()
            cur = []
        cur.append(u)
    flush()
    return segs


# --- audio + channel detection ----------------------------------------------
def read_stereo(path: Path) -> tuple[np.ndarray, int]:
    """Read a 2-channel PCM wav -> (float32 [N, 2] in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        n_ch, sampwidth, sr, n = (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getnframes(),
        )
        raw = w.readframes(n)
    if n_ch != 2 or sampwidth != 2:
        raise ValueError(
            f"expected 16-bit stereo, got {n_ch}ch/{8*sampwidth}bit: {path}"
        )
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, 2), sr


def _envelope(ch: np.ndarray, hop: int) -> np.ndarray:
    n_frames = ch.size // hop
    if n_frames == 0:
        return np.zeros(0)
    blocks = ch[: n_frames * hop].reshape(n_frames, hop)
    return np.sqrt((blocks**2).mean(axis=1) + 1e-12)


def _vad_vector(units: list[dict], n_frames: int, hop: int, sr: int) -> np.ndarray:
    t = np.arange(n_frames) * (hop / sr)
    v = np.zeros(n_frames)
    for u in units:
        v[(t >= u["start"]) & (t < u["end"])] = 1.0
    return v


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / d)


def detect_giver_channel(
    samples: np.ndarray, sr: int, g_units: list[dict], f_units: list[dict]
) -> int:
    """Return which source channel (0 or 1) carries the giver, by correlating
    each channel's energy envelope with each role's word activity."""
    hop = max(1, sr // ENV_HZ)
    e0, e1 = _envelope(samples[:, 0], hop), _envelope(samples[:, 1], hop)
    n_frames = min(e0.size, e1.size)
    e0, e1 = e0[:n_frames], e1[:n_frames]
    vg = _vad_vector(g_units, n_frames, hop, sr)
    vf = _vad_vector(f_units, n_frames, hop, sr)
    # Score = how much better each channel tracks the giver than the follower.
    score_ch0_giver = (_corr(e0, vg) - _corr(e0, vf)) + (_corr(e1, vf) - _corr(e1, vg))
    return 0 if score_ch0_giver >= 0 else 1


def write_combined_wav(src: Path, dst: Path, giver_ch: int, sr_out: int) -> None:
    """ffmpeg: 16 kHz stereo with L=giver(tutor), R=follower(student)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if giver_ch == 1:  # source has follower on L, giver on R -> swap
        cmd += ["-af", "pan=stereo|c0=c1|c1=c0"]
    cmd += ["-ar", str(sr_out), "-ac", "2", str(dst)]
    subprocess.run(cmd, check=True)


# --- main --------------------------------------------------------------------
def build(raw_dir: Path, out_dir: Path, *, sample_rate: int, limit: int | None) -> None:
    tu_dir = ensure_nxt(raw_dir)
    dialogues = list_dialogues(tu_dir)
    if limit is not None:
        dialogues = dialogues[:limit]
    logger.info("building %d dialogues -> %s", len(dialogues), out_dir)

    aligned_path = out_dir / "aligned" / "maptask_gold.jsonl"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for idx, dialogue in enumerate(dialogues):
        g_units = parse_timed_units(tu_dir / f"{dialogue}.g.timed-units.xml")
        f_units = parse_timed_units(tu_dir / f"{dialogue}.f.timed-units.xml")
        if not g_units and not f_units:
            logger.warning("conv %d (%s): no words, skipping", idx, dialogue)
            continue
        src = ensure_audio(raw_dir, dialogue)
        samples, sr = read_stereo(src)
        giver_ch = detect_giver_channel(samples, sr, g_units, f_units)
        write_combined_wav(
            src, out_dir / "audio" / str(idx) / "combined.wav", giver_ch, sample_rate
        )
        segments = units_to_segments(g_units, "tutor") + units_to_segments(
            f_units, "student"
        )
        segments.sort(key=lambda s: s["start_time"])
        rows.append(
            {
                "pid": dialogue,
                "conv_index": idx,
                "giver_source_channel": giver_ch,
                "duration": round(samples.shape[0] / sr, 3),
                "segments": segments,
            }
        )
        logger.info(
            "conv %d (%s): giver=ch%d, %d tutor + %d follower words",
            idx,
            dialogue,
            giver_ch,
            len(g_units),
            len(f_units),
        )

    with aligned_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("wrote %s (%d conversations)", aligned_path, len(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="/data/jhe516/maptask_raw")
    p.add_argument("--out-dir", default="/data/jhe516/duplex_output/maptask_ref")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--limit", type=int, default=None, help="First N dialogues (smoke).")
    args = p.parse_args()
    build(
        Path(args.raw_dir),
        Path(args.out_dir),
        sample_rate=args.sample_rate,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
