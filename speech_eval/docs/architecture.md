# speech_eval architecture

```
duplex_output/<run>/                speech_eval_output/<run>/
├── conversations.jsonl             ├── audiobox.jsonl
├── aligned/whisper_mfa.jsonl       ├── turntaking_judge.jsonl
├── manifest.json                   ├── vap.jsonl
└── audio/<idx>/                    ├── vap_frames/<idx>.npz
    ├── tutor_full.wav (24k mono)   └── manifest.json
    ├── student_full.wav (24k mono)
    └── combined.wav (24k stereo)
```

`speech_eval.runner.run()` is the single orchestration entry point. It
iterates conversation indices found under `duplex_output/<run>/audio/` and
dispatches to the requested components.

## Component A — `audiobox_scorer.py`

Wraps `audiobox_aesthetics.infer.initialize_predictor`. For each
conversation we load three channels (tutor mono, student mono, mono mixdown
of `combined.wav`) and call `predictor.forward(batch)` once. The library
handles its own resampling and device placement — it auto-detects MPS on
Apple Silicon and we just log what it picked.

Output: one JSONL row per conversation, with a nested
`{channel: {axis: score}}` for PQ / PC / CE / CU on each channel.

## Component B — `turntaking_judge.py`

Wraps `espnet2.bin.slu_inference.Speech2Understand` loaded from the
HuggingFace tag `espnet/Turn_taking_prediction_SWBD`.

The model:

- Whisper-medium encoder (frozen at training time).
- A small classifier head emitting one of five labels per ~40 ms encoder
  frame: `C` (continuation), `T` (turn change), `BC` (backchannel),
  `I` (interruption), `NA` (silence).
- Trained on Switchboard 2-channel data mixed to mono.

### Two compatibility patches we apply

1. **`whisper.audio.N_MELS = 80` monkey-patch.** ESPnet's
   `whisper_encoder.py` imports `N_MELS` from `whisper.audio`, but
   openai-whisper >= 20250625 removed that constant (it became per-model when
   v3-large introduced 128-mel variants). We set it back to 80 before any
   ESPnet import; 80 is correct for Whisper-medium.
2. **`ctc_weight=0.0, beam_size=1` at construction.** The model was trained
   without CTC, so the default beam-search CTC scorer crashes (`'NoneType'
   object has no attribute 'log_softmax'`). The recipe's
   `decode_asr_chunk.yaml` uses `ctc_weight: 0.3, beam_size: 1`, but
   inspecting the loaded model shows no CTC head; clamping `ctc_weight=0`
   avoids the scorer entirely.

### Inference shape

One call to `predictor(audio_30s)` returns a single result tuple. Its first
element is a *string* of whitespace-separated frames; each frame is
comma-separated 5-float probabilities in label order `('C', 'NA', 'I', 'BC',
'T')`. A 30 s window produces ~745 frames (≈ 40 ms each), matching the
paper.

We slide non-overlapping 30 s windows across the conversation and parse each
result into a list of `frame_probs` + greedy `frame_labels`. We don't
implement the paper's overlapping-window scheme (where every 40 ms decision
sees a fresh 30 s of preceding context) because each window costs ~7 min on
M-series CPU. With non-overlapping windows the first 30 s of each window may
see less than full context — acceptable for v1 inspection, would need
revisiting for a research-grade comparison.

### Device

`auto` by default, with priority `cuda → mps → cpu`: CUDA on Linux/GPU
boxes, MPS on Apple Silicon, CPU as last resort. `--device cuda` is
straightforward — `Speech2Understand` takes the device string and we let
PyTorch place tensors. On a multi-GPU host, set `CUDA_VISIBLE_DEVICES=N`
to pin a specific card rather than expanding the CLI surface. `--device
mps` exports `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back
to CPU; expect possible silent-correctness issues on non-contiguous
tensors. The CUDA path does not set the MPS fallback env var. Apple
Neural Engine via CoreML would need the model re-exported — not done.

## Component C — `vap_scorer.py` + `vap_remote.py` + `_vap_infer.py`

Voice Activity Projection (VAP; Ekstedt & Skantze) is a ~5.8M-param model that
takes a **stereo 16 kHz waveform** (two separate speaker channels) and predicts
near-future voice activity at 50 Hz, emitting `p_now` (0–600 ms horizon) and
`p_future` (600–2000 ms), each `[T, 2]` and normalized so `[...,0]+[...,1]=1` —
a relative who-holds-the-floor signal. `combined.wav` is exactly that input
(L=tutor, R=student, cleanly separated), so no diarization is needed.

**Split across three files** (so the GPU pass is dumped once and re-scored
locally forever):

- `_vap_infer.py` — the only file importing torch/vap. Runs under a VAP venv on
  the remote (`~/repo/VoiceActivityProjection`). Loads `combined.wav`, resamples
  24k→16k, **trims the leading silence** (everything before
  `first_onset - 0.8 s`; the leading dead air is model-loading time and would
  inject a spurious mutual-silence state), runs `model.probs`, and writes
  `vap_frames/<idx>.npz` with the 50 Hz trajectories on the **original**
  conversation timeline.
- `vap_remote.py` — rsync/ssh orchestrator mirroring `alignment/remote_runner.py`
  (push `combined.wav` + jobs.json + the infer script → run → pull frames →
  cleanup).
- `vap_events.py` — pure-Python (no torch) event detection + metrics.
- `vap_scorer.py` — the `score_run` entry the runner calls; builds the jobs spec,
  drives the remote round-trip, and joins frames + timeline into `vap.jsonl`.

### Why the aligned timeline (not `conversations.jsonl`)

Event detection needs accurate word/turn boundaries. The raw
`conversations.jsonl` timestamps are *text-delta arrival* times (a 34-word
student turn can read as a 37 ms segment), so VAP consumes
`aligned/whisper_mfa.jsonl` (MFA models silence explicitly → trustworthy
boundaries) for both the trim and the timeline. The component errors if that
file is missing rather than silently falling back to the raw timestamps.

### The three readouts (`vap_events.detect_and_score`)

Events come from the two-role word-interval timeline (words merged into speech
runs; adjacent runs yield a **turn-end** when the floor changes hands or an
**intra-turn pause** when the same speaker resumes). A causality guard drops
events where the responder was already speaking just before `t_evt` (VAP is
causal, so its prediction there would be contaminated).

1. **Cue production** (read the holder's own channel): Early/Late-Yield at
   turn-ends, Strong/Weak-Hold at pauses — the 2023 VAP-evaluator metric.
2. **Responder appropriateness** (read the *other* channel's `p_future`): a 2×2
   of VAP's prediction (yield/hold) × what the responder actually did (took the
   floor / stayed silent within `W_DECISION_S=1.5 s`). Headline:
   **false-interruption rate** + uptake-latency distribution, computed
   symmetrically per role.
3. **Distributional**: `compute_distributions` emits per-conversation timing
   lists (floor-transfer offset / gap / overlap / within-pause / turn duration)
   into the `vap.jsonl` `distributions` block. The analyze module pools them per
   run and scores KS / Wasserstein-EMD / JS distance against a human reference
   (`maptask_ref`) — see [`distributional.md`](distributional.md). (The raw 50 Hz
   trajectories in the `vap_frames/` sidecar remain the substrate for the
   deferred VAP-*trajectory* distributional extension.)

All thresholds are module constants in `vap_events.py`, embedded per row, and
calibration-tunable. Channel order is fixed `[tutor=0, student=1]`; flipping it
flips every index.

### Device

The forward pass always runs remotely on CUDA (`--vap-host` / `--vap-gpu`); the
`--device` flag (audiobox/judge) does not apply. The metric logic runs locally
on the Mac. A re-score reuses cached `vap_frames/*.npz` and never touches the
GPU, which is what makes threshold sweeps cheap.

## Output schemas

See [`outputs.md`](outputs.md).

## What we deliberately skipped

See [`deferred.md`](deferred.md).
