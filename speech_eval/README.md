# speech_eval

Audio-side evaluation of duplex generations. Independent of the text-only
`evaluation/` module — outputs land in their own `speech_eval_output/`
directory.

Three scoring components plus a cross-system analyzer, each a thin wrapper over
released code:

| Component | What it scores | Model | Per conversation |
|---|---|---|---|
| `audiobox` | Audio quality (4 axes) | Meta Audiobox Aesthetics | 3 channels × 4 axes |
| `judge` | Turn-taking labels | ESPnet `Turn_taking_prediction_SWBD` | ≈25 labels / sec |
| `vap` | Turn-taking appropriateness + timing distributions | Voice Activity Projection (Ekstedt & Skantze) | yield/hold cues + responder 2×2 + FTO/gap/overlap dists |
| `analyze` | Cross-system comparison + distributional realism | n/a (pure aggregation) | summary.csv + plots + report.md + KS/EMD/JS vs human ref |

Quick run:

```bash
uv run python -m speech_eval --run-name <duplex_run>             # audiobox + judge + vap
uv run python -m speech_eval --run-name <duplex_run> --component vap --vap-gpu 0
```

The `vap` component runs its forward pass on the remote GPU box and needs a
forced-alignment timeline (`aligned/whisper_mfa.jsonl`) for the run. See
[`docs/quickstart.md`](docs/quickstart.md) for setup, weight paths, the
`--device` (auto: cuda → mps → cpu) / `--max-windows` knobs, and the one-time
VAP remote install.

## What lives where

- [`docs/quickstart.md`](docs/quickstart.md) — install, run, weight cache paths.
- [`docs/architecture.md`](docs/architecture.md) — how the components plug in, the N_MELS monkey-patch, judge device priority (cuda → mps → cpu) and the MPS-fallback caveat, and the VAP local-metric / remote-forward split.
- [`docs/outputs.md`](docs/outputs.md) — schema of `audiobox.jsonl`, `turntaking_judge.jsonl`, `vap.jsonl` (+ `vap_frames/`), and `manifest.json`.
- [`docs/deferred.md`](docs/deferred.md) — methods from the source papers we deliberately did *not* implement in v1 (the 5 capability metrics, corpus-level IPU / Pause / Gap / Overlap stats, the VAD → decisions adapter, the VAP human-human calibration baseline + distributional distances).
- [`docs/analysis.md`](docs/analysis.md) — cross-system comparison module (`python -m speech_eval.analyze`): means + CI, paired Wilcoxon, label shares, plots.
- [`docs/distributional.md`](docs/distributional.md) — distributional turn-taking realism: KS / Wasserstein-EMD / JS distance of each system's timing distributions from a human reference (`--reference`).
- [`reference/README.md`](reference/README.md) — building the human-human reference run (`maptask_ref`, HCRC Map Task) that anchors the distributional metric.
- [`web_integration.py`](web_integration.py) — read-only VAP figure (spectrogram + aligned words + P(now)/P(future) ribbons) rendered in the duplex web UI's History view from `vap_frames/` + `vap.jsonl`. See [`../duplex/docs/web-ui.md`](../duplex/docs/web-ui.md#vap-turn-taking-figure).

## Source papers

- **Meta Audiobox Aesthetics** — Tjandra et al., 2025. <https://arxiv.org/abs/2502.05139>. Code: <https://github.com/facebookresearch/audiobox-aesthetics>.
- **Talking Turns** — Arora et al., ICLR 2025. <https://arxiv.org/abs/2503.01174>. Code shipped in [espnet PR #5948](https://github.com/espnet/espnet/pull/5948), checkpoint at <https://huggingface.co/espnet/Turn_taking_prediction_SWBD>.
- **Voice Activity Projection (VAP)** — Ekstedt & Skantze, Interspeech 2022; used as an evaluator in Ekstedt et al., Interspeech 2023 (<https://arxiv.org/abs/2305.17971>). Code + checkpoint: <https://github.com/ErikEkstedt/VoiceActivityProjection>.
