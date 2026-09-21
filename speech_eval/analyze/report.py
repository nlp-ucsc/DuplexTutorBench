"""Write the analysis artifacts: summary.csv, system_table.md, pairwise/, report.md.

Pure I/O. Statistics live in ``audiobox_stats`` / ``judge_stats``; figure
building lives in ``plots``. This module just assembles their outputs.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from speech_eval.analyze.audiobox_stats import (
    HEADLINE_AXES,
    all_pairwise_tables,
    axis_correlations,
    system_means_ci,
    tutor_student_gap,
)
from speech_eval.analyze.distributional_stats import (
    DIST_LABELS,
    DIST_METRICS,
    load_distributions,
    reference_distances,
)
from speech_eval.analyze.judge_stats import (
    EVENT_LABELS,
    JUDGE_CAVEAT,
    event_rate_contrasts,
    label_shares,
)
from speech_eval.analyze.loader import (
    AUDIOBOX_AXES,
    CHANNELS,
    JUDGE_LABELS,
    AnalysisData,
)
from speech_eval.analyze.plots import (
    fig01_means_ci,
    fig02_violin_grid,
    fig03_pairwise_heatmap,
    fig04_axis_corr_grid,
    fig05_tutor_student_gap,
    fig06_label_shares,
    fig07_event_rates,
    fig08_vap_fi_rate,
    fig09_vap_cue_dist,
    fig10_vap_uptake_latency,
    fig11_dist_ecdf,
    fig12_dist_emd,
    fig13_student_tutor_emd,
)
from speech_eval.analyze.vap_stats import (
    VAP_CAVEAT,
    cue_shares,
    fi_rate_contrasts,
    fi_rate_summary,
    runs_with_vap,
    uptake_latencies,
)


def _git_sha(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip()[:12]
    except Exception:
        return None


def write_summary_csv(data: AnalysisData, out_path: Path) -> None:
    """Wide per-(run, conv_index) table with all audiobox + judge columns."""
    df = data.audiobox.merge(
        data.judge, on=["run", "conv_index"], how="outer"
    ).sort_values(["run", "conv_index"])
    df.to_csv(out_path, index=False)


def _fmt_mean_ci(m: float, h: float) -> str:
    if pd.isna(m):
        return "n/a"
    if pd.isna(h):
        return f"{m:.2f}"
    return f"{m:.2f} ± {h:.2f}"


def write_system_table(
    data: AnalysisData,
    means_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Wide table: rows=systems; columns include the headline audiobox + judge metrics."""
    rows = []
    for run in data.runs:
        row: dict[str, object] = {"system": run}
        for ch in ("mixed", "tutor"):
            for ax in HEADLINE_AXES:
                sel = means_df[
                    (means_df["run"] == run)
                    & (means_df["channel"] == ch)
                    & (means_df["axis"] == ax)
                ]
                if len(sel):
                    row[f"{ch}_{ax}"] = _fmt_mean_ci(
                        float(sel["mean"].iloc[0]), float(sel["ci_half"].iloc[0])
                    )
                else:
                    row[f"{ch}_{ax}"] = "n/a"
        sel = means_df[
            (means_df["run"] == run)
            & (means_df["channel"] == "student")
            & (means_df["axis"] == "PQ")
        ]
        row["student_PQ"] = (
            _fmt_mean_ci(float(sel["mean"].iloc[0]), float(sel["ci_half"].iloc[0]))
            if len(sel)
            else "n/a"
        )
        for lbl in ("I", "T", "BC"):
            sel = shares_df[(shares_df["run"] == run) & (shares_df["label"] == lbl)]
            if len(sel):
                row[f"%{lbl}"] = _fmt_mean_ci(
                    float(sel["mean_pct"].iloc[0]),
                    float(sel["ci_half"].iloc[0]),
                )
            else:
                row[f"%{lbl}"] = "n/a"
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("system")
    out_path.write_text(tbl.to_markdown())
    return tbl


def write_pairwise_csvs(
    pairwise: dict[tuple[str, str], pd.DataFrame], pairwise_dir: Path
) -> list[str]:
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for (ch, ax), tbl in pairwise.items():
        name = f"{ch}_{ax}.csv"
        tbl.to_csv(pairwise_dir / name, index=False)
        written.append(name)
    return sorted(written)


def write_event_pairwise_csvs(
    pairwise: dict[str, pd.DataFrame], pairwise_dir: Path
) -> list[str]:
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for lbl, tbl in pairwise.items():
        name = f"judge_event_{lbl}.csv"
        tbl.to_csv(pairwise_dir / name, index=False)
        written.append(name)
    return sorted(written)


def write_report(
    data: AnalysisData,
    out_root: Path,
    *,
    group: str,
    means_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    pairwise_audiobox: dict[tuple[str, str], pd.DataFrame],
    pairwise_judge: dict[str, pd.DataFrame],
    fig_files: list[str],
    pairwise_files: list[str],
    vap_runs: list[str] | None = None,
    vap_fi_df: pd.DataFrame | None = None,
    reference: str | None = None,
    dist_df: pd.DataFrame | None = None,
) -> Path:
    """Assemble the human-readable markdown report."""
    n_convs_per_run = data.audiobox.groupby("run")["conv_index"].nunique().to_dict()
    sha = _git_sha(Path(__file__).resolve().parents[2])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    L = lines.append
    L(f"# speech_eval cross-system comparison — `{group}`")
    L("")
    L(f"_Generated: {now}{f' (git {sha})' if sha else ''}_")
    L("")

    # 1. Overview
    L("## 1. Overview")
    L("")
    L(f"- Input root: `{data.input_root}`")
    L(f"- Runs analyzed ({len(data.runs)}):")
    for run in data.runs:
        L(f"  - `{run}` — {n_convs_per_run.get(run, 0)} conversations")
    L("")
    L(
        "Audiobox scores axes: PQ (production quality), CE (content enjoyment), "
        "CU (content usefulness), PC (production complexity — see §11)."
    )
    L(
        "Judge labels: C (continuation), NA (silence), I (interruption), "
        "BC (backchannel), T (turn change)."
    )
    L("")

    # 2. System table
    L("## 2. Headline system table")
    L("")
    L("All entries are `mean ± half-95%-CI` across each system's conversations.")
    L("")
    L((out_root / "system_table.md").read_text())
    L("")

    # 3. Audiobox quality (figs 01)
    L("## 3. Audiobox quality — means + 95% CI")
    L("")
    for ch in CHANNELS:
        fname = f"01_means_ci_{ch}.png"
        if fname in fig_files:
            L(f"**Channel: {ch}**")
            L("")
            L(f"![{ch} means CI](figures/{fname})")
            L("")
    L(
        "Mixed channel is the joint mixdown both speakers contribute to; tutor "
        "and student are the per-role mono channels. Only the tutor backend "
        "varies across these runs (student is always Gemini), so any tutor-"
        "channel difference comes from the backend swap; student-channel "
        "differences reflect cross-conversation variation in what the student "
        "said in response."
    )
    L("")

    # 4. Distributions
    L("## 4. Audiobox score distributions")
    L("")
    L("![violin grid](figures/02_violin_grid.png)")
    L("")
    L(
        "3×3 facet grid (rows = channel, columns = headline axis). **Vertical "
        "extent** = range of scores across conversations; **horizontal width "
        "at a given y** = density of conversations at that score (a fat bulge "
        "means many convs clustered there). Inner lines mark the 25/50/75th "
        "percentiles. A tall violin with no clear bulge = inconsistent system; "
        "a short, fat violin = very consistent."
    )
    L("")

    # 5. Pairwise comparisons
    L("## 5. Paired Wilcoxon comparisons")
    L("")
    L(
        "Tests are paired by `conv_index` (each MathVista question is "
        "shared across all runs). Holm-Bonferroni correction applied within "
        "each `(channel, axis)` family of 6 pairwise tests. Lower p (darker "
        "cells) = stronger evidence of a difference."
    )
    L("")
    for ch in CHANNELS:
        fname = f"03_pairwise_heatmap_{ch}.png"
        if fname in fig_files:
            L(f"![pairwise {ch}](figures/{fname})")
            L("")
    L("Per-pair p-values, W-statistic, and n_pairs for each test are in `pairwise/`:")
    for name in pairwise_files:
        L(f"- [`pairwise/{name}`](pairwise/{name})")
    L("")

    # 6. Cross-axis correlations
    L("## 6. Audiobox cross-axis correlations")
    L("")
    L("![axis corr grid](figures/04_axis_corr_grid.png)")
    L("")
    L(
        "Per-system Pearson correlation across the four Audiobox axes "
        "(channel=mixed). Replicates the Audiobox paper's Figure 2 as a "
        "sanity check that axes are not collinear."
    )
    L("")

    # 7. Tutor vs student gap
    L("## 7. Tutor vs Student PQ gap")
    L("")
    L("![tutor-student gap](figures/05_tutor_student_gap.png)")
    L("")
    L(
        "**Control note.** The student backend is held fixed (Gemini) across "
        "all runs. Roughly flat student PQ across systems is therefore "
        "expected; large tutor-vs-student deltas indicate that the tutor "
        "backend swap moved the tutor channel only."
    )
    L("")

    # 8. Caveat
    L("## 8. Turn-taking judge — windowing caveat")
    L("")
    L(JUDGE_CAVEAT)
    L("")
    L(
        "All judge-derived figures below carry this caveat. Cross-system "
        "comparisons among the 4 runs remain valid (the same windowing "
        "artifact applies to all systems), but absolute label percentages "
        "are not directly comparable to the paper's Switchboard numbers."
    )
    L("")

    # 9. Label shares
    L("## 9. Judge label shares")
    L("")
    L("> " + JUDGE_CAVEAT)
    L("")
    L("![label shares](figures/06_label_shares.png)")
    L("")

    # 10. Event-rate contrasts
    L("## 10. Judge event-rate contrasts")
    L("")
    L("> " + JUDGE_CAVEAT)
    L("")
    L("![event rates](figures/07_event_rates.png)")
    L("")
    L(
        "Per-conversation % of frames labeled as each event type, with "
        "95% CI error bars. Paired Wilcoxon comparisons of these per-conv "
        "percentages are written to `pairwise/judge_event_<label>.csv`."
    )
    L("")

    # 11. VAP turn-taking appropriateness (optional)
    if vap_runs:
        L("## 11. VAP turn-taking appropriateness")
        L("")
        L("> " + VAP_CAVEAT)
        L("")
        L(
            f"VAP-scored runs ({len(vap_runs)}): "
            + ", ".join(f"`{r}`" for r in vap_runs)
            + "."
        )
        if len(vap_runs) < 2:
            L("")
            L(
                "_Only one VAP-scored run so far, so the cross-system contrast is "
                "not yet populated. Score the other runs (`python -m speech_eval "
                "--run-name <run> --component vap`) and re-run analyze to compare._"
            )
        L("")
        L(
            "**Responder false-interruption rate** — at moments VAP predicts the "
            "floor-holder keeps the turn, how often the *other* role grabs it "
            "anyway (a barge-in). `tutor`-as-responder is the model under test "
            "(student is always Gemini)."
        )
        L("")
        L("![vap fi rate](figures/08_vap_fi_rate.png)")
        L("")
        if vap_fi_df is not None and len(vap_fi_df):
            tut = vap_fi_df[vap_fi_df["scope"] == "tutor"].set_index("run")
            rows = ["| system | tutor FI-rate | n_convs |", "|---|---|---|"]
            for run in vap_runs:
                if run in tut.index:
                    m = float(tut.loc[run, "mean_fi_rate"])
                    h = float(tut.loc[run, "ci_half"])
                    n = int(tut.loc[run, "n"])
                    rows.append(f"| `{run}` | {_fmt_mean_ci(m, h)} | {n} |")
            L("\n".join(rows))
            L("")
        L(
            "**Cue production** — at each turn-end the holder's Early/Late-Yield, "
            "at each intra-turn pause its Strong/Weak-Hold (the 2023 VAP-evaluator "
            "metric)."
        )
        L("")
        L("![vap cue dist](figures/09_vap_cue_dist.png)")
        L("")
        L(
            "**Uptake latency** — for appropriate-uptake events, how long the "
            "responder took to take the floor after VAP's predicted yield point."
        )
        L("")
        L("![vap uptake latency](figures/10_vap_uptake_latency.png)")
        L("")

    # 14. Distributional turn-taking realism vs a human reference (optional).
    if reference and dist_df is not None and not dist_df.empty:
        L("## 14. Distributional turn-taking realism")
        L("")
        L(
            f"Distance of each system's timing distributions from the human "
            f"reference `{reference}` (a human-human corpus scored through the "
            f"identical vap pipeline). This is the *aggregate* counterpart to the "
            f"VAP responder 2×2 in §11: §11 asks whether each turn-taking "
            f"decision is locally right; this asks whether the system's "
            f"turn-taking **looks human in distribution**. Closeness here is "
            f"necessary, not sufficient — a model can match the marginal "
            f"histogram while still making locally wrong calls."
        )
        L("")
        L(
            "Metrics, each over a clipped support (FTO ∈ ±2 s): **EMD** = "
            "Wasserstein-1 in seconds (interpretable headline, lower = more "
            "human), **KS** = max CDF gap, **JS** = Jensen-Shannon distance ∈ "
            "[0, 1]. Brackets are percentile bootstrap 95% CIs (system resampled, "
            "reference fixed)."
        )
        L("")
        L("![dist emd](figures/12_dist_emd.png)")
        L("")
        if "13_student_tutor_emd.png" in fig_files:
            L("![student x tutor emd](figures/13_student_tutor_emd.png)")
            L("")
            L(
                "FTO Wasserstein-EMD vs human by **student × tutor** backend "
                "(rows × columns), lower = more human — surfaces how both roles' "
                "backends shape the turn-taking distribution."
            )
            L("")
        L("![dist ecdf](figures/11_dist_ecdf.png)")
        L("")
        # Headline table: the floor-transfer-offset (fto_s) distances per system.
        fto = dist_df[dist_df["metric"] == "fto_s"].set_index("system")
        if len(fto):
            rows = [
                "| system | FTO EMD (s) | FTO KS | FTO JS | n_events |",
                "|---|---|---|---|---|",
            ]
            for run in vap_runs or []:
                if run in fto.index:
                    r = fto.loc[run]
                    rows.append(
                        f"| `{run}` | {r['emd']:.3f} "
                        f"[{r['emd_lo']:.3f}, {r['emd_hi']:.3f}] "
                        f"| {r['ks']:.3f} | {r['js']:.3f} | {int(r['n_system'])} |"
                    )
            L("\n".join(rows))
            L("")
        L(
            "Per-(system, metric) distances for all five timing metrics "
            f"({', '.join(DIST_LABELS[m] for m in DIST_METRICS)}) are in "
            "[`distributional.csv`](distributional.csv)."
        )
        L("")

    # 12. PC note
    L("## 12. Note on the PC axis")
    L("")
    L(
        "PC (production complexity) measures structural complexity, not "
        "quality. The Audiobox paper (§3.2) reports it as anti-correlated "
        "with speech quality for music but only weakly related for clean "
        "single-speaker speech. We deliberately keep PC out of headline "
        "panels and the system table. The raw per-conv PC values are still "
        "available in `summary.csv` and the cross-axis correlation grid "
        "(fig 04) shows how it tracks the other axes per system."
    )
    L("")

    # 13. Files index
    L("## 13. Files index")
    L("")
    L(
        "- [`summary.csv`](summary.csv) — one row per (run, conv_index) with all metrics."
    )
    L("- [`system_table.md`](system_table.md) — headline metric table.")
    if vap_runs:
        L(
            "- [`vap_summary.csv`](vap_summary.csv) — per-(run, conv_index) VAP "
            "turn-taking metrics."
        )
    if reference and dist_df is not None and not dist_df.empty:
        L(
            "- [`distributional.csv`](distributional.csv) — per-(system, metric) "
            f"KS/EMD/JS distances vs the `{reference}` human reference."
        )
    L("- `figures/` — all PNGs referenced above.")
    L("- `pairwise/` — per-comparison Wilcoxon results.")
    L("")

    report_path = out_root / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def build(
    data: AnalysisData,
    *,
    out_root: Path,
    group: str,
    reference: str | None = None,
) -> Path:
    """End-to-end pipeline: write CSV, run stats, build plots, write report.

    ``reference`` is an optional human-reference run name (e.g. ``maptask_ref``)
    whose ``vap.jsonl`` distributions anchor the distributional-realism distances
    (§14). It need not be one of ``data.runs`` and need not carry audiobox/judge.

    Returns the group output directory.
    """
    group_dir = out_root / group
    figures_dir = group_dir / "figures"
    pairwise_dir = group_dir / "pairwise"
    group_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw join CSV.
    write_summary_csv(data, group_dir / "summary.csv")

    # 2. Stats.
    means_df = system_means_ci(data.audiobox)
    shares_df = label_shares(data.judge)
    gap_df = tutor_student_gap(data.audiobox, axis="PQ")
    pairwise_ab = all_pairwise_tables(data.audiobox, runs=data.runs)
    pairwise_jg = event_rate_contrasts(data.judge, runs=data.runs)
    corrs = axis_correlations(data.audiobox, channel="mixed")

    # 3. System table (writes its own MD file too).
    write_system_table(data, means_df, shares_df, group_dir / "system_table.md")

    # 4. Pairwise CSVs.
    pw_files = write_pairwise_csvs(pairwise_ab, pairwise_dir)
    pw_files += write_event_pairwise_csvs(pairwise_jg, pairwise_dir)

    # 5. Figures.
    fig_files: list[str] = []
    fig_files += fig01_means_ci(means_df, data.runs, figures_dir)
    fig_files.append(fig02_violin_grid(data.audiobox, data.runs, figures_dir))
    fig_files += fig03_pairwise_heatmap(pairwise_ab, data.runs, figures_dir)
    fig_files.append(fig04_axis_corr_grid(corrs, data.runs, figures_dir))
    fig_files.append(fig05_tutor_student_gap(gap_df, data.runs, figures_dir))
    fig_files.append(fig06_label_shares(shares_df, data.runs, figures_dir))
    fig_files.append(fig07_event_rates(shares_df, data.runs, figures_dir))

    # 5b. VAP (optional — only for runs scored by the vap component).
    vap_runs: list[str] = runs_with_vap(data.vap, data.runs)
    vap_fi_df = None
    if vap_runs:
        vap_fi_df = fi_rate_summary(data.vap)
        cue_df = cue_shares(data.vap)
        lat_df = uptake_latencies(data.vap, scope="pooled")
        # Drop the per-conv latency list columns before the flat CSV.
        list_cols = [c for c in data.vap.columns if c.endswith("_uptake_lat")]
        data.vap.drop(columns=list_cols).to_csv(
            group_dir / "vap_summary.csv", index=False
        )
        fi_contrasts = fi_rate_contrasts(data.vap, scope="tutor", runs=vap_runs)
        if not fi_contrasts.empty:
            fi_contrasts.to_csv(pairwise_dir / "vap_fi_rate_tutor.csv", index=False)
            pw_files.append("vap_fi_rate_tutor.csv")
        fig_files.append(fig08_vap_fi_rate(vap_fi_df, vap_runs, figures_dir))
        fig_files.append(fig09_vap_cue_dist(cue_df, vap_runs, figures_dir))
        fig_files.append(fig10_vap_uptake_latency(lat_df, vap_runs, figures_dir))

    # 5c. Distributional realism vs a human reference (optional).
    dist_df: pd.DataFrame | None = None
    if reference and vap_runs:
        dist_df = reference_distances(data.input_root, vap_runs, reference)
        if not dist_df.empty:
            dist_df.to_csv(group_dir / "distributional.csv", index=False)
            dist_by_run = {
                r: load_distributions(data.input_root, r)
                for r in [*vap_runs, reference]
            }
            fig_files.append(
                fig11_dist_ecdf(dist_by_run, vap_runs, reference, figures_dir)
            )
            fig_files.append(fig12_dist_emd(dist_df, vap_runs, figures_dir))
            f13 = fig13_student_tutor_emd(dist_df, vap_runs, figures_dir)
            if f13:
                fig_files.append(f13)

    # 6. Report.
    write_report(
        data,
        group_dir,
        group=group,
        means_df=means_df,
        shares_df=shares_df,
        gap_df=gap_df,
        pairwise_audiobox=pairwise_ab,
        pairwise_judge=pairwise_jg,
        fig_files=fig_files,
        pairwise_files=pw_files,
        vap_runs=vap_runs,
        vap_fi_df=vap_fi_df,
        reference=reference,
        dist_df=dist_df,
    )

    return group_dir
