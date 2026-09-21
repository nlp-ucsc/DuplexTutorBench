# speech_eval cross-system analysis

Aggregates the per-conversation JSONL outputs from one or more
`speech_eval_output/<run>/` directories into a single comparison report
(summary CSV + plots + markdown). Independent of the per-run scorer; reads
existing JSONL only, writes no new audio inferences.

## In scope

Seven analyses across both components:

**Audiobox** (`audiobox.jsonl`):
1. Per-system means ± 95% CI on PQ / CE / CU across tutor / student / mixed channels.
2. Per-system score distributions (3×3 violin facet grid).
3. Pairwise paired Wilcoxon signed-rank tests with Holm-Bonferroni correction.
4. Cross-axis Pearson correlation (replicates Audiobox paper Fig 2 as a sanity check).
5. Tutor vs Student PQ gap — explicitly a control panel (student backend held fixed).

**Turn-taking judge** (`turntaking_judge.jsonl`):
6. Per-system frame-label shares for {C, NA, I, BC, T}.
7. Pairwise paired Wilcoxon on per-conv event rates for {I, T, BC, NA}.

**VAP** (`vap.jsonl`, optional — only for runs scored by the `vap` component):
8. Responder false-interruption rate per system (tutor- and student-as-responder, mean ± 95% CI), with paired Wilcoxon on per-conv tutor FI rate across systems.
9. Cue-production composition (Early/Late-Yield, Strong/Weak-Hold) per holder role.
10. Responder uptake-latency distribution per system.

VAP sections appear only when at least one analyzed run has `vap.jsonl`; with a
single VAP-scored run the cross-system contrast is noted as not-yet-populated.
Absolute VAP rates use the uncalibrated 0.5 thresholds (descriptive, not
pass/fail — see `deferred.md` item 7).

**Distributional realism vs a human reference** (`vap.jsonl` `distributions`
block, optional — only with `--reference`):
11. Distance of each system's floor-transfer-offset / gap / overlap / pause /
    turn-duration **distributions** from a human-human reference run (KS / EMD /
    JS, each with a bootstrap 95% CI). Report §14, `distributional.csv`, an EMD
    bar chart, an ECDF overlay per metric, and — when run names decompose as
    `<student>_<tutor>` — a student×tutor FTO-EMD heatmap. The aggregate
    complement to the responder 2×2 above — see
    [`distributional.md`](distributional.md).

## Out of scope (deliberately skipped)

- **Comparison against the Switchboard human baseline** from Talking Turns Table 2. The paper used a 40 ms sliding hop; our judge uses non-overlapping 30 s windows (cost-driven, see `deferred.md` item 4). Absolute numbers from the paper are therefore not directly comparable. Cross-system comparisons among our runs remain valid because the same windowing artifact applies uniformly.
- **The 5 capability metrics from Talking Turns §4** (`deferred.md` items 1-3). Need a VAD-to-decisions adapter.
- **Corpus-level IPU / Pause / Gap / Overlap statistics** (`deferred.md` item 2).

## Invocation

```bash
uv run python -m speech_eval.analyze \
    [--runs RUN [RUN ...]] \
    [--group NAME] \
    [--reference RUN] \
    [--input-root speech_eval_output] \
    [--out-root speech_eval_output/_analysis]
```

Defaults: `--runs` = every non-`_*`-prefixed subdir of `--input-root`;
`--group` = `YYYY-MM-DD_<N>runs`. `--reference` (e.g. `maptask_ref`) enables the
distributional-realism section (§14); it is *not* one of `--runs` and needs only
a `vap.jsonl` (no audiobox/judge) — omit it to skip §14.

## Inputs expected

For every run in `--runs`, `<input-root>/<run>/` must contain both:

- `audiobox.jsonl` — one row per conversation × 4 axes × 3 channels.
- `turntaking_judge.jsonl` — one row per conversation × N windows × per-frame labels and probabilities.

## Output layout

```
<out-root>/<group>/
├── summary.csv              # one row per (run, conv_index) with all metrics
├── system_table.md          # headline metrics table (rendered into report.md too)
├── report.md                # human-readable report with embedded plots
├── vap_summary.csv          # per-(run, conv_index) VAP metrics   (if VAP runs)
├── distributional.csv       # per-(system, metric) KS/EMD/JS + CIs (if --reference)
├── figures/                 # PNGs at 150 dpi
│   ├── 01_means_ci_<channel>.png   ×3
│   ├── 02_violin_grid.png
│   ├── 03_pairwise_heatmap_<channel>.png  ×3
│   ├── 04_axis_corr_grid.png
│   ├── 05_tutor_student_gap.png
│   ├── 06_label_shares.png         # JUDGE_CAVEAT baked in
│   ├── 07_event_rates.png          # JUDGE_CAVEAT baked in
│   ├── 08_vap_fi_rate.png / 09_vap_cue_dist.png / 10_vap_uptake_latency.png  (if VAP)
│   ├── 11_dist_ecdf.png / 12_dist_emd.png                          (if --reference)
│   └── 13_student_tutor_emd.png             (if --reference + <student>_<tutor> names)
└── pairwise/                # per-comparison CSVs
    ├── <channel>_<axis>.csv  ×9    # Audiobox: 3 channels × {PQ, CE, CU}
    ├── judge_event_<label>.csv ×4  # judge: {I, T, BC, NA}
    └── vap_fi_rate_tutor.csv       # VAP FI-rate contrast (if ≥2 VAP runs)
```

Each pairwise CSV has columns `system_a, system_b, n_pairs, W, p_raw, p_holm, significant_at_05`.
`distributional.csv` has `system, metric, n_system, n_reference, {ks,emd,js}{,_lo,_hi}`.

## Statistical notes

- **Pairing**: tests use `scipy.stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")` on per-conv values inner-joined on `conv_index`. Each MathVista question is shared across all runs, so n_pairs = 30 per pair in the current dataset.
- **Multiple-comparison correction**: Holm-Bonferroni applied within each family of pairwise tests (one family per `(channel, axis)` for Audiobox; one family per event label for the judge). Implemented inline; no `statsmodels` dep. Chosen over BH-FDR because the families are small (`C(4,2)=6` pairs) and we want FWER control.
- **CIs**: `scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)` on per-conv values.
- **Correlations**: `df.pivot(...).corr(method="pearson")` per system on the mixed channel.

## Caveats

- **Non-overlapping judge windows** — see the dedicated `JUDGE_CAVEAT` constant defined in `judge_stats.py`. Surfaced in every judge figure caption and as §8 of `report.md`.
- **PC axis** — Audiobox PC is a complexity, not quality, measure; the Audiobox paper notes it is anti-correlated with quality only for music, weakly so for clean speech. Kept out of headline panels; available in `summary.csv` and the cross-axis correlation grid (fig 04).
- **Student channel is held fixed** — only the tutor backend varies across the 4 v2 runs. Flat student PQ across systems is by construction; large tutor-vs-student deltas isolate the backend swap.

## Pointers

- `speech_eval/docs/outputs.md` — schema of the per-run JSONL files this module consumes.
- `speech_eval/docs/deferred.md` — the work explicitly NOT in v1 (item 4 = the windowing caveat; items 1-3 = the VAD-decisions adapter needed for the 5 capability metrics).
- `speech_eval/README.md` — module overview.
