# speech_eval quickstart

## Install

Already done via `uv add` — the two top-level dependencies are
`audiobox-aesthetics` and `espnet` (plus `espnet-model-zoo` and
`openai-whisper`, which `espnet` pulls in as transitives but does not declare
strongly enough for inference). All four are pinned in `pyproject.toml`.

If you're setting up a fresh worktree, `uv sync` is enough — everything is
in this project's venv.

## Run

```bash
# Default: both components, all conversations, judge processes the full
# conversation length
uv run python -m speech_eval --run-name <duplex_run>

# Smoke test: one conversation, judge processes only the first 30 s
uv run python -m speech_eval --run-name <duplex_run> --conv-index 0 \
    --max-windows 1

# Audiobox only (fast, ~3 s per conversation on MPS)
uv run python -m speech_eval --run-name <duplex_run> --component audiobox

# First five conversations
uv run python -m speech_eval --run-name <duplex_run> --limit 5

# VAP only (forward pass on the remote GPU box; needs aligned/whisper_mfa.jsonl)
uv run python -m speech_eval --run-name <duplex_run> --component vap --vap-gpu 0

# VAP smoke test: one conversation, keep the remote scratch for inspection
uv run python -m speech_eval --run-name <duplex_run> --component vap \
    --conv-index 0 --vap-gpu 0 --keep-remote-tmp
```

Outputs land in `speech_eval_output/<duplex_run>/`:

- `audiobox.jsonl` — one row per conversation, three channels (tutor /
  student / mixed) × four axes (PQ / PC / CE / CU).
- `turntaking_judge.jsonl` — one row per conversation, each holding one or
  more 30 s window dicts with per-frame (~40 ms) labels and 5-class
  probabilities.
- `vap.jsonl` (+ `vap_frames/<idx>.npz`) — one row per conversation with the
  turn-taking cue + responder metrics, the timing `distributions` block
  (FTO/gap/overlap/pause/turn-dur for the distributional-realism metric), plus
  the raw 50 Hz trajectories.
- `manifest.json` — CLI args, git SHA, weight cache paths, total runtime.

For the **distributional turn-taking realism** metric (KS / EMD / JS vs a human
reference) and how the Map Task `maptask_ref` reference is built, see
[`distributional.md`](distributional.md) and [`../reference/README.md`](../reference/README.md).

## VAP component (one-time remote setup)

The `vap` component runs its forward pass on `ucsc_lab_sv11` and needs the
VoiceActivityProjection repo + a venv there. Set up once:

```bash
ssh ucsc_lab_sv11 'cd ~/repo && \
  git clone https://github.com/ErikEkstedt/VoiceActivityProjection.git && \
  cd VoiceActivityProjection && \
  ~/.local/bin/uv venv .venv --python 3.12 && \
  ~/.local/bin/uv pip install --python .venv/bin/python \
      torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
  ~/.local/bin/uv pip install --python .venv/bin/python einops numpy && \
  ~/.local/bin/uv pip install --python .venv/bin/python -e . --no-deps'
```

Notes:
- The bundled checkpoint `example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`
  ships with the repo (no separate download) and includes the local CPC encoder
  weights under `assets/`.
- The sv11 install above uses **torch 2.5.1 / cu121** (proven on the RTX 6000
  Ada), but torch is **no longer pinned**: `_vap_infer.py` forces
  `weights_only=False` itself (the VAP + CPC checkpoints are plain pickles, which
  torch ≥ 2.6 rejects by default), and falls back to `soundfile` when
  `torchaudio.load` has no backend. So the script also runs unchanged on a newer
  stack — e.g. an aarch64 / Blackwell box on **torch 2.11 + cu128** (see the
  alternate-host note below).
- `speech_eval/_vap_infer.py` is the source of truth and is rsynced up to the
  remote VAP repo on every run (so edits to it take effect without a manual
  redeploy).
- The run needs a forced-alignment timeline at
  `duplex_output/<run>/aligned/<variant>.jsonl`. Default variant is
  `whisper_mfa` — generate it with `uv run python -m alignment --run-name <run>
  --mode whisper_mfa`. Pass `--vap-aligned-variant <name>` to read a different
  one (e.g. `maptask_gold` for the human reference, whose NXT word times are
  already gold and need no alignment).

### Alternate host (DGX Spark)

A second VAP-capable box exists: `ssh ucsc_dgx_spark_1` (NVIDIA **GB10 Grace
Blackwell**, aarch64, Ubuntu 24.04, ~128 GB **unified** memory, no ffmpeg). Set
up the same way but with a Blackwell-compatible stack:

```bash
ssh ucsc_dgx_spark_1 'cd ~/repo/VoiceActivityProjection && \
  ~/.local/bin/uv venv --python 3.12 .venv && \
  ~/.local/bin/uv pip install --python .venv torch torchaudio \
      --index-url https://download.pytorch.org/whl/cu128 && \
  ~/.local/bin/uv pip install --python .venv numpy einops soundfile && \
  ~/.local/bin/uv pip install --python .venv -e .'
```

Then score with `--vap-host ucsc_dgx_spark_1`. The repo (minus `.venv`) is copied
from sv11; `soundfile` covers audio loading since there's no ffmpeg/torchcodec.
**Caveat:** unified memory means VAP never hits a classic GPU OOM, but a runaway
allocation can starve the whole box (observed wedging the machine mid-run, needing
a reboot). **sv11 is always the default box — confirm with the user before using
the Spark;** never switch to it on your own initiative.

## Compute device

| Component | Default | Notes |
|---|---|---|
| `audiobox` | auto (`cuda → mps → cpu`) | The library auto-detects internally; we don't override. ~3 s forward over a 2-minute conversation on M-series MPS. |
| `judge` | auto (`cuda → mps → cpu`) | CUDA wins on Linux/GPU boxes; MPS on Apple Silicon; CPU as last resort. MPS is verified label-for-label equal to CPU on this checkpoint (max prob diff ≈ 1e-4), with ~3.7× speedup over CPU. The MPS path exports `PYTORCH_ENABLE_MPS_FALLBACK=1`; the CUDA path does not. Force a device with `--device cpu`, `--device mps`, or `--device cuda`. On a multi-GPU host, pick a specific card with `CUDA_VISIBLE_DEVICES=N`. |
| `vap` | remote CUDA (`ucsc_lab_sv11`) | Forward pass always runs remotely; `--device` does not apply. Pick the card with `--vap-gpu N`, the host with `--vap-host`. The ~5.8M-param model is tiny — all 30 conversations of a run finish in ~12 s. Metric re-scoring (threshold sweeps) runs locally off the cached `vap_frames/*.npz` with no GPU. |

The actually-resolved device for each component is recorded in
`manifest.json` under `devices_used`.

**Judge runtime per 30 s window**: ~7 min on M-series CPU, ~2 min on MPS,
~24 s on CUDA (RTX 6000 Ada; first-window measurement on
`gemini_gpt_v2`/conv 0 — subsequent windows are faster once CUDA kernels
are warm). A typical 2-minute duplex conversation has four 30 s windows,
so expect ~8 min per conversation on MPS and well under a minute on
CUDA. Use `--max-windows 1` for a fast smoke test (just the first 30 s).

## Model weight cache paths (where the big files live)

| Weights | Path | Approx size |
|---|---|---|
| Audiobox checkpoint | `~/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/` | ~300 MB |
| ESPnet judge head + config | `<venv>/lib/python3.13/site-packages/espnet_model_zoo/models--espnet--Turn_taking_prediction_SWBD/` | ~70 MB |
| Whisper-medium encoder | `~/.cache/whisper/medium.pt` | 1.4 GB |

> Note: `espnet_model_zoo` caches **inside the venv** rather than under
> `~/.cache/espnet/`. If you blow away `.venv`, the judge head will
> re-download (~70 MB). The Whisper encoder lives in `~/.cache/whisper/`
> and survives venv rebuilds.

`manifest.json` records the resolved paths on every run for traceability.
