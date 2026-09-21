# Distributional turn-taking realism

Does a system's turn-taking *look human in aggregate*? This is the complement to
the VAP responder 2×2 (which asks whether each individual decision is *locally*
right). It compares the **distribution** of a system's turn-taking timing against
a human-human reference corpus scored through the identical vap pipeline.

> Closeness here is **necessary, not sufficient**: a model can match the marginal
> histogram while still making locally wrong calls. Read it next to §11 (the
> responder 2×2) in the analysis report.

## The timing metrics

Computed by `vap_events.compute_distributions(segments)` — pure timing off the
aligned word/turn boundaries, **no VAP frames**. Speech runs are the per-role
merged IPUs from `build_timeline`; walking the onset-sorted stream, each adjacent
pair is either a floor change or a same-speaker resume:

| key | meaning |
|---|---|
| `fto_s` | **Floor-transfer offset**: signed gap at each role change. Positive = between-turn silence, negative = overlap (next speaker started early). The headline. |
| `gap_s` | positive FTOs (hand-off silences) |
| `overlap_s` | magnitudes of negative FTOs (overlap durations) |
| `within_pause_s` | same-speaker resume silences (intra-turn pauses) |
| `turn_dur_s` | merged speech-run (turn) durations, both roles |

These lists are stored **raw and unclipped** per conversation in the
`distributions` block of `vap.jsonl`.

Human reference points (Levinson & Torreira 2015; Heldner & Edlund 2010): modal
gap ≈ 200 ms, 70–82 % of transitions < 500 ms. A system whose FTO mass sits at
1–2 s is distributionally *slow*.

## The distances

`analyze/distributional_stats.reference_distances(input_root, systems, reference)`
pools each run's per-conversation lists, clips each metric to a sensible support
(FTO ∈ ±2 s; gaps/overlaps/pauses ∈ [0, 2 s]; turn duration ∈ [0, 20 s] — keeps
the tail from dominating), and scores each system against the reference on:

- **EMD** — Wasserstein-1, in **seconds**. The interpretable headline ("this
  system's gaps run ~180 ms longer than human"). Lower = more human.
- **KS** — Kolmogorov-Smirnov statistic (max CDF gap).
- **JS** — Jensen-Shannon distance on shared-bin histograms, ∈ [0, 1].

Each carries a percentile **bootstrap 95 % CI** (the system distribution is
resampled; the reference is held fixed as the anchor). The clip windows live in
`DIST_CLIP`; `FTO_CLIP_S` (= 2.0) is the tunable headline constant in
`vap_events.py`.

## The human reference

The reference is a **human-human corpus scored as a synthetic vap run**, so it
flows through the unchanged pipeline. We use the **HCRC Map Task** corpus
(CC-BY 4.0): a task-oriented dyadic corpus where an instruction *giver* guides a
*follower* — asymmetric and goal-directed, mirroring tutor↔student. See
[`../reference/README.md`](../reference/README.md) for how `maptask_ref` is built.

## Running it

```bash
# 1. Build + score the reference once (giver->tutor, follower->student):
#    (build on the GPU box, then:)
uv run python -m speech_eval --run-name maptask_ref --component vap \
    --vap-aligned-variant maptask_gold --vap-gpu 0

# 2. Make sure the systems carry the distributions block. Runs scored before
#    this metric existed are backfilled GPU-free (cached frames are reused):
uv run python -m speech_eval --run-name <system> --component vap --overwrite

# 3. Compare systems against the human reference:
uv run python -m speech_eval.analyze \
    --runs <system1> <system2> ... --reference maptask_ref
```

Outputs land in the analysis report as **§14** with `distributional.csv`
(per-(system, metric) KS/EMD/JS + CIs), an EMD bar chart (`12_dist_emd.png`), and
an ECDF overlay per metric (`11_dist_ecdf.png`, each system vs the bold human
reference). When the compared runs are named `<student>_<tutor>_<suffix>` (e.g.
the v3 set), §14 also includes a **student×tutor FTO-EMD heatmap**
(`13_student_tutor_emd.png`) — rows = student backend, columns = tutor backend —
which surfaces how *both* roles' backends shape the turn-taking distribution
(v2's fixed-student set renders as a 1×4 row).

## Status / follow-ups

- **Done**: the distributional distances above (this was `deferred.md` §7's
  "distributional distance" item).
- **Still deferred**: using the 0.5-threshold *calibration* against the same
  human reference (turn the responder 2×2's absolute rates into pass/fail), and
  the VAP-trajectory distributional extension (compare the distribution of
  `p_future` at turn-ends, not just timing). See [`deferred.md`](deferred.md) §7.
