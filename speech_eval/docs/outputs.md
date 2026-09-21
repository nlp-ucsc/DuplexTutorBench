# speech_eval outputs

Everything lands in `speech_eval_output/<run>/`. Three files:

## `audiobox.jsonl`

One row per conversation. Both PQ and CE are useful for our clean dialogue
case; PC and CU are kept for completeness because the model emits all four
in one forward pass.

```jsonc
{
  "conv_index": 0,
  "channels": {
    "tutor":   {"PQ": 7.73, "PC": 1.63, "CE": 4.97, "CU": 6.96},
    "student": {"PQ": 7.83, "PC": 1.54, "CE": 5.75, "CU": 7.18},
    "mixed":   {"PQ": 8.26, "PC": 1.44, "CE": 6.37, "CU": 7.58}
  },
  "device": "mps",
  "audio_sample_rate": 24000
}
```

Score range is roughly 1-10; higher is better. PC measures *complexity* not
quality, so single-speaker clean dialogue legitimately scores low.

## `turntaking_judge.jsonl`

One row per conversation. Each row contains a list of 30 s windows. Each
window has per-frame outputs at ~40 ms resolution.

```jsonc
{
  "conv_index": 0,
  "labels_order": ["C", "NA", "I", "BC", "T"],
  "device": "cpu",
  "n_windows_processed": 1,
  "max_windows": 1,
  "windows": [
    {
      "window_index": 0,
      "window_start_s": 0.0,
      "window_end_s": 30.0,
      "n_frames": 745,
      "frame_s": 0.04,
      "label_counts": {"NA": 540, "C": 200, "T": 5},
      "frame_labels": ["NA", "NA", "NA", "...", "C", "C"],
      "frame_probs": [[0.07, 0.87, 0.00, 0.03, 0.02], "..."]
    }
  ]
}
```

`labels_order` documents the column order of `frame_probs` so downstream
code never has to guess. `label_counts` is a cheap aggregate of
`frame_labels` for quick inspection.

## `vap.jsonl`

One row per conversation, from the `vap` component (Voice Activity Projection).
The forward pass runs on the remote GPU box; the metric logic runs locally on
the forced-alignment timeline (`aligned/whisper_mfa.jsonl`). Channel order is
fixed `[tutor=0, student=1]`. See [`architecture.md`](architecture.md) for the
three readouts and the event-detection algorithm.

```jsonc
{
  "conv_index": 0,
  "channel_order": ["tutor", "student"],
  "frame_hz": 50,
  "sample_rate": 16000,
  "device": "ucsc_lab_sv11:cuda:0",
  "checkpoint": "VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
  "aligned_variant": "whisper_mfa",
  "first_onset_s": 11.41,        // earliest aligned word (original timeline)
  "preroll_s": 0.8,              // context kept before first onset
  "trim_start_s": 10.61,         // wav cut point = first_onset - preroll
  "n_frames": 5478,
  "thresholds": {"yield_thresh": 0.5, "w_decision_s": 1.5, "...": "..."},
  "n_events": {"turn_end": 7, "intra_pause": 13, "dropped_guard": 0},

  // Distributional realism — raw per-conversation timing lists (seconds), pure
  // timeline, no VAP frames. Pooled across a run + scored (KS/EMD/JS) against a
  // human reference by the analyze module. See docs/distributional.md.
  "distributions": {
    "fto_s": [0.5, -0.2, 1.7],   // signed floor-transfer offsets (- = overlap)
    "gap_s": [0.5, 1.7],         // positive FTOs
    "overlap_s": [0.2],          // |negative FTOs|
    "within_pause_s": [0.4],     // same-speaker resume silences
    "turn_dur_s": [1.0, 0.7]     // merged turn durations, both roles
  },

  // Readout 1 — cue production, per holder role:
  "cues": {
    "tutor":   {"Early-Yield": 1, "Late-Yield": 0, "Yield-NotCued": 2,
                "Strong-Hold": 0, "Weak-Hold": 2, "Hold-Misread": 1},
    "student": {"...": 0}
  },

  // Readout 2 — responder 2x2, per responder role + pooled:
  "responder": {
    "tutor": {
      "appropriate_uptake": 2, "missed_yield": 1,
      "false_interruption": 2, "appropriate_restraint": 9,
      "false_interruption_rate": 0.182,        // FI / (FI + restraint); null if 0
      "uptake_latency_s": [0.91, 1.0]          // for appropriate-uptake events
    },
    "student": {"...": 0},
    "pooled":  {"...": 0}
  },

  // Per-event audit detail (one entry per scored event):
  "events": [
    {"type": "turn_end", "t_evt": 19.97, "holder": "student", "responder": "tutor",
     "gap_s": 1.0, "p_now": [0.064, 0.936], "p_future": [0.93, 0.07],
     "vap_pred": "YIELD", "responder_action": "took_floor", "latency_s": 1.0,
     "cue_label": "Late-Yield", "cell": "appropriate_uptake"}
  ]
}
```

`p_now`/`p_future` in each event are `[tutor, student]` at `t_evt`, normalized so
the pair sums to 1. The `thresholds` block is embedded so a row stays
reproducible after constants change. Re-scoring (e.g. sweeping `w_decision_s`)
reuses the cached frames and never touches the GPU.

### `vap_frames/{conv_index}.npz` (sidecar)

The raw 50 Hz trajectories pulled back from the remote, one compressed npz per
conversation: float32 arrays `p_now0`/`p_now1`, `p_future0`/`p_future1`,
`vad0`/`vad1`, `frame_times_original` (frame times on the original conversation
timeline), plus a `header` JSON string. The substrate for the deferred
VAP-*trajectory* distributional work (e.g. the distribution of `p_future` at
turn-ends); the *timing*-based distributional metric ships now and reads the
`distributions` block above, not these frames. See
[`distributional.md`](distributional.md).

## `manifest.json`

```jsonc
{
  "run_name": "gemini_gpt_v2",
  "components": ["audiobox", "judge"],
  "started_at_utc": "2026-...",
  "finished_at_utc": "2026-...",
  "elapsed_seconds": 432.1,
  "device_requested": "auto",
  "devices_used": {"audiobox": "mps", "judge": "mps"},
  "conv_indices": [0],
  "max_windows": 1,
  "overwrite": false,
  "duplex_root": "duplex_output",
  "out_root": "speech_eval_output",
  "argv": ["python", "-m", "speech_eval", "..."],
  "git_sha": "abc1234...",
  "artifacts": {
    "audiobox": "audiobox.jsonl",
    "turntaking_judge": "turntaking_judge.jsonl"
  },
  "weight_paths": {
    "audiobox_dir": "/Users/.../models--facebook--audiobox-aesthetics",
    "espnet_judge_file": "/Users/.../valid.loss.ave.pth",
    "whisper_medium_pt": "/Users/.../medium.pt"
  },
  "env": {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
}
```

The manifest is rewritten on every run; older artifacts on disk are not
touched. When the `vap` component runs, the manifest also carries
`artifacts.vap`, `devices_used.vap` (`"<host>:cuda:<gpu>"`), the
`weight_paths.vap_*` remote pointers, and a top-level `vap` block recording
`frame_hz`, `sample_rate`, `channel_order`, `aligned_variant`, host, and gpu.
