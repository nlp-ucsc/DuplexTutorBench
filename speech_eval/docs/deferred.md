# Deferred — what we intentionally did *not* implement in v1

v1 of `speech_eval` is strictly **what the released code ships**. The list
below is everything from the two source papers that *would* be valuable but
needs work we deliberately skipped. Picking any of these up is a separate
decision.

---

## 1. The five capability metrics from Talking Turns §4

`compute_turn_take_metrics.py` (shipped by espnet PR #5948 at
`egs2/TEMPLATE/asr1/pyscripts/utils/compute_turn_take_metrics.py`)
implements all five paper metrics as methods on `ScoreResult`:

- `turn_change_metric` — when the user pauses, did the AI speak up?
- `make_backchannel_metric` — did the AI backchannel at appropriate moments?
- `make_interruption_metric` — were the AI's interruptions well-timed?
- `turn_willingness_metric` — when the AI paused, did it signal turn-yielding?
- `handle_interruption_metric` — when the user interrupted, did the AI yield?

**Why not in v1**: each metric needs a "decisions" input file giving the
turn-taking label per chunk (NA/BC/I/T/C). The paper derives this from
pyannote VAD on each speaker channel plus simple filler-word heuristics. In
our setting we can derive it more cheaply from existing per-role audio
files + segment annotations in `conversations.jsonl` (~50 LoC), but the
user has scoped this out of v1.

When picked up, the script lifts cleanly: `ScoreResult` reads two plain
text files and writes a confusion matrix + agreement scores; nothing about
it requires the rest of ESPnet at runtime.

---

## 2. Corpus-level turn-taking statistics — Talking Turns §3

From the paper:

- **IPU** count and total duration (per speaker).
- **Pause** count and % cumulated duration (silence inside one speaker's turn).
- **Gap** count and % cumulated duration (silence between speakers' turns).
- **Overlap** count and % cumulated duration.
- Backchannel rate, speaking rate (words / min).

These describe how a conversation flows at the macro level (does the system
leave huge gaps? does it overlap as much as humans do?).

**Why not in v1**: the espnet PR doesn't ship a builder for these. The
paper computes them from per-channel pyannote VAD output. We can either
plug in pyannote ourselves or derive the same vectors from amplitude on
the per-role channels we already write. Both are small (~50 LoC) but
involve writing our own code rather than wrapping released code.

---

## 3. VAD-to-decisions adapter

The thing that would unlock both blocks above. The paper's labeling logic
takes per-chunk voice activity vectors `Y^AI` and `Y^Human` plus an ASR
transcript and a filler-word list, and produces the `L = {C, T, BC, I, NA}`
sequence required for the 5 metrics.

In *our* setting this is much simpler than the paper's pipeline:
`duplex_output/<run>/audio/<idx>/{tutor,student}_full.wav` are already
separated by speaker, and `conversations.jsonl` has start/end timestamps
per role. We can derive `Y^*` either from amplitude on the per-role files
or directly from the segment annotations.

The implementation is small but it is our code, not theirs, so v1 omits it.

---

## 4. Sliding-window overlapping inference for the judge

The paper uses a sliding 30 s context window stepping every 40 ms, so every
chunk-level prediction sees a fresh 30 s of preceding audio.

v1 uses **non-overlapping 30 s windows** for runtime reasons (each window is
~7 min of inference on M-series CPU). The first 30 s of each conversation
sees correct context; later windows see only 0-30 s of their own audio,
not the full preceding 30 s.

**Why not in v1**: cost. With overlapping windows, a 2 min conversation
would take ~60 hours of CPU inference. To match the paper exactly we'd
either need GPU access (the remote `ucsc_lab_sv11` box, per the
`alignment/` precedent) or a fundamentally different inference strategy
(e.g., extract per-frame encoder activations once and reuse them).

---

## 5. Remote-GPU offload for the judge

`alignment/` already SSHes to `ucsc_lab_sv11` for the heavy lifting. The
judge could ride the same rails. Out of scope for v1 — local CPU works,
just slowly. (The `vap` component now demonstrates exactly this pattern:
`vap_remote.py` is a direct sibling of `alignment/remote_runner.py`.)

---

## 6. Comparison-mode metrics

The paper compares Moshi vs. cascaded vs. human reference on each of the
five capability axes. We could trivially do the same once Block 1 is in
place — `speech_eval` already iterates many runs.

---

## 7. VAP calibration baseline + distributional distances

- **Distributional distance — DONE (2026-06-20).** Floor-transfer-offset / gap /
  overlap / pause / turn-duration distributions are scored KS / Wasserstein-EMD /
  JS against the **HCRC Map Task** human reference (`maptask_ref`). See
  [`distributional.md`](distributional.md). Built on the aligned-timeline
  `distributions` block (`vap_events.compute_distributions`), not the `vap_frames`
  trajectories — no new GPU work.
- **Human-human calibration baseline — still deferred.** VAP scores are
  synthesizer-dependent and the 0.5 decision thresholds are uncalibrated, so the
  responder 2×2's absolute rates (e.g. "FI rate 0.29") stay descriptive, not
  pass/fail. The Map Task reference now exists; the remaining work is to derive
  the threshold boundaries from VAP-on-Map-Task and replace the 0.5 defaults.
- **VAP-trajectory distributional extension — still deferred.** The
  `vap_frames/*.npz` sidecars hold the raw 50 Hz `p_now`/`p_future`/`vad`. Scoring
  the distribution of e.g. `p_future` at turn-ends vs the human reference (rather
  than the timing-only distances above) is the differentiated "VAP as a
  distributional evaluator" claim.
