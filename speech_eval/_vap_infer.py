"""Remote-side VAP forward pass. Runs under the VAP venv on ucsc_lab_sv11.

This is the ONLY file in speech_eval that imports torch / torchaudio / vap. The
local orchestrator (``vap_remote.py``) rsyncs it into
``~/repo/VoiceActivityProjection/`` and runs it with that repo's venv, so
``import vap`` resolves and the bundled checkpoint is on hand.

Per conversation it: loads the stereo ``combined.wav`` (L=tutor, R=student),
resamples 24k->16k, trims the leading silence (everything before
``first_onset - PREROLL_S``), runs ``model.probs`` (50 Hz), and writes a compact
``frames/{conv_index}.npz`` with the per-channel ``p_now``/``p_future``/``vad``
trajectories plus ``frame_times_original`` (frame times mapped back onto the
original conversation timeline). The metric logic stays on the Mac
(``vap_events.py``) so it can re-score without re-running this.

This file is the source of truth; ``vap_remote`` rsyncs it up to the remote
VAP repo on every run.

Usage (invoked by vap_remote):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python _vap_infer.py \
        --input-dir /tmp/vap_runs/<run> \
        --checkpoint example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt
"""

import argparse
import json
import os

# Reduce CUDA allocator fragmentation before any torch/CUDA init. The GPU box is
# shared; long (chunked) reference dialogues can peak a few GB, and a neighbor
# spiking free memory was enough to OOM an otherwise-fitting forward pass.
# expandable_segments lets the allocator grow into freed space instead of
# reserving fixed blocks — purely a memory-layout change, no effect on outputs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

# torch >= 2.6 flipped torch.load's default to weights_only=True, which rejects
# the VAP + CPC checkpoints (plain pickles). Force the legacy behavior before any
# checkpoint load (incl. the CPC load inside VapGPT construction). No-op on the
# older sv11 torch where False was already the default.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from vap.model import VapConfig, VapGPT  # noqa: E402

PREROLL_S = 0.8  # context kept before the first speech onset (VAP warm-up)
DIRECT_MAX_S = 160.0  # process in one forward pass below this; chunk above


def _load_wav(path):
    """Load an audio file as ([C, N] float tensor, sample_rate).

    Prefers ``torchaudio.load``; falls back to ``soundfile`` when torchaudio's
    backend is unavailable (torch >= 2.11 routes through TorchCodec, which needs
    ffmpeg — absent on the DGX Spark box). soundfile uses bundled libsndfile.
    """
    try:
        return torchaudio.load(path)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [N, C]
        return torch.from_numpy(data.T).contiguous(), sr


def _chunked_probs(waveform, model, device, context_time=20, step_time=5):
    """Sliding-window probs for long audio (self-contained; mirrors run.py).

    ``waveform`` is [1, 2, N]. Concatenates only the fresh ``step`` frames of
    each window, then patches the final uncovered segment.
    """
    n_samples = waveform.shape[-1]
    duration = n_samples / model.sample_rate
    chunk_time = context_time + step_time
    step_samples = int(step_time * model.sample_rate)
    chunk_samples = int(chunk_time * model.sample_rate)
    step_frames = int(step_time * model.frame_hz)

    folds = waveform.unfold(
        dimension=-1, size=chunk_samples, step=step_samples
    ).permute(2, 0, 1, 3)
    with torch.no_grad():
        out = model.probs(folds[0].to(device))
        for w in folds[1:]:
            o = model.probs(w.to(device))
            for k in ("vad", "p_now", "p_future", "probs", "H"):
                out[k] = torch.cat([out[k], o[k][:, -step_frames:]], dim=1)

        expected = round(duration * model.frame_hz)
        processed = out["p_now"].shape[1]
        if expected != processed:
            omitted = expected - processed
            o = model.probs(waveform[..., -chunk_samples:].to(device))
            for k in ("vad", "p_now", "p_future", "probs", "H"):
                out[k] = torch.cat([out[k], o[k][:, -omitted:]], dim=1)
    return out


def _process_one(model, device, wav_path, first_onset_s):
    wav, sr = _load_wav(wav_path)  # [C, N]
    if wav.shape[0] != 2:
        raise ValueError(
            f"{wav_path}: expected stereo [tutor, student], got {wav.shape}"
        )
    if sr != model.sample_rate:
        wav = AF.resample(wav, orig_freq=sr, new_freq=model.sample_rate)
    sr16 = model.sample_rate

    trim_start = max(0.0, float(first_onset_s) - PREROLL_S)
    i0 = int(round(trim_start * sr16))
    wav = wav[:, i0:].contiguous()
    waveform = wav.unsqueeze(0)  # [1, 2, T]
    duration = waveform.shape[-1] / sr16

    if duration > DIRECT_MAX_S:
        out = _chunked_probs(waveform, model, device)
    else:
        with torch.no_grad():
            out = model.probs(waveform.to(device))

    p_now = out["p_now"][0].cpu().numpy()  # [T, 2] = (tutor, student)
    p_future = out["p_future"][0].cpu().numpy()
    vad = out["vad"][0].cpu().numpy()
    n_frames = p_now.shape[0]
    frame_times = trim_start + np.arange(n_frames) / model.frame_hz
    header = {
        "sr_in": int(sr),
        "first_onset_s": float(first_onset_s),
        "preroll_s": PREROLL_S,
        "trim_start_s": float(trim_start),
        "frame_hz": int(model.frame_hz),
        "sample_rate": int(sr16),
        "channel_order": ["tutor", "student"],
        "n_frames": int(n_frames),
        "chunked": bool(duration > DIRECT_MAX_S),
        "torch_version": torch.__version__,
    }
    return {
        "p_now0": p_now[:, 0].astype(np.float32),
        "p_now1": p_now[:, 1].astype(np.float32),
        "p_future0": p_future[:, 0].astype(np.float32),
        "p_future1": p_future[:, 1].astype(np.float32),
        "vad0": vad[:, 0].astype(np.float32),
        "vad1": vad[:, 1].astype(np.float32),
        "frame_times_original": frame_times.astype(np.float32),
        "header": json.dumps(header),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="remote scratch dir for the run")
    ap.add_argument(
        "--checkpoint",
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
    )
    ap.add_argument("--conv-index", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    input_dir = args.input_dir
    frames_dir = os.path.join(input_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    with open(os.path.join(input_dir, "jobs.json")) as f:
        jobs = json.load(f)["jobs"]
    if args.conv_index is not None:
        jobs = [j for j in jobs if int(j["conv_index"]) == args.conv_index]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vap_infer] device={device} jobs={len(jobs)}", flush=True)

    model = VapGPT(VapConfig())
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(sd)
    model = model.eval().to(device)
    print(
        f"[vap_infer] model loaded sr={model.sample_rate} frame_hz={model.frame_hz}",
        flush=True,
    )

    for j in jobs:
        ci = int(j["conv_index"])
        out_path = os.path.join(frames_dir, f"{ci}.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"[vap_infer] conv {ci}: cached, skip", flush=True)
            continue
        wav_path = os.path.join(input_dir, j["wav"])
        data = _process_one(model, device, wav_path, j["first_onset_s"])
        np.savez_compressed(out_path, **data)
        print(
            f"[vap_infer] conv {ci}: {data['header']} -> {out_path}",
            flush=True,
        )

    print("[vap_infer] DONE", flush=True)


if __name__ == "__main__":
    main()
