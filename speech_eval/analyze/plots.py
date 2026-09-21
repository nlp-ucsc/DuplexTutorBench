"""Figure builders for the cross-system speech_eval comparison report.

All plotting helpers take pre-computed DataFrames (from ``audiobox_stats`` /
``judge_stats``) plus the raw long DataFrames, write a single PNG to the
target directory, and return its filename. Filenames are designed to sort
into the report order.

Figures using judge data (06, 07) prepend ``JUDGE_CAVEAT`` via
``fig.text`` so the caveat is baked into the PNG itself, not just the
markdown caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from speech_eval.analyze.audiobox_stats import HEADLINE_AXES
from speech_eval.analyze.distributional_stats import (
    DIST_CLIP,
    DIST_LABELS,
    DIST_METRICS,
)
from speech_eval.analyze.judge_stats import EVENT_LABELS, JUDGE_CAVEAT
from speech_eval.analyze.loader import AUDIOBOX_AXES, CHANNELS, JUDGE_LABELS

SAVE_KW = dict(dpi=150, bbox_inches="tight")

# Don't strip if the common affix is shorter than this — for short run names,
# the saved characters aren't worth the loss of context.
_MIN_AFFIX_STRIP = 3


def _set_style() -> None:
    sns.set_theme(context="notebook", style="whitegrid")


def _common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""
    s1, s2 = min(strs), max(strs)
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def _common_suffix(strs: list[str]) -> str:
    return _common_prefix([s[::-1] for s in strs])[::-1]


def short_labels(runs: list[str]) -> list[str]:
    """Strip the longest common prefix and suffix from a list of run names.

    Returns a list aligned with the input order. Falls back to the original
    names if the saved affixes are shorter than ``_MIN_AFFIX_STRIP`` or if
    stripping would produce duplicate / empty labels.
    """
    if len(runs) < 2:
        return list(runs)
    pre = _common_prefix(runs)
    suf = _common_suffix(runs)
    if len(pre) < _MIN_AFFIX_STRIP:
        pre = ""
    if len(suf) < _MIN_AFFIX_STRIP:
        suf = ""
    if not pre and not suf:
        return list(runs)
    stripped = [r[len(pre) : len(r) - len(suf) if suf else len(r)] for r in runs]
    if any(not s for s in stripped) or len(set(stripped)) != len(stripped):
        return list(runs)
    return stripped


def _add_caveat(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        -0.04,
        JUDGE_CAVEAT,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        wrap=True,
    )


def fig01_means_ci(means_df: pd.DataFrame, runs: list[str], out_dir: Path) -> list[str]:
    """One PNG per channel: grouped bars of headline axes with 95% CI."""
    _set_style()
    labels = short_labels(runs)
    written: list[str] = []
    for ch in CHANNELS:
        sub = means_df[
            (means_df["channel"] == ch) & (means_df["axis"].isin(HEADLINE_AXES))
        ].copy()
        sub["run"] = pd.Categorical(sub["run"], categories=runs, ordered=True)
        sub["axis"] = pd.Categorical(
            sub["axis"], categories=list(HEADLINE_AXES), ordered=True
        )
        sub = sub.sort_values(["axis", "run"])
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        axes_list = list(HEADLINE_AXES)
        n_axes = len(axes_list)
        n_runs = len(runs)
        bar_w = 0.8 / n_runs
        for i, run in enumerate(runs):
            run_rows = sub[sub["run"] == run]
            xs = np.arange(n_axes) + i * bar_w - 0.4 + bar_w / 2
            means = [
                float(run_rows[run_rows["axis"] == ax_name]["mean"].iloc[0])
                for ax_name in axes_list
            ]
            errs = [
                float(run_rows[run_rows["axis"] == ax_name]["ci_half"].iloc[0])
                for ax_name in axes_list
            ]
            ax.bar(xs, means, width=bar_w, yerr=errs, capsize=3, label=labels[i])
        ax.set_xticks(np.arange(n_axes))
        ax.set_xticklabels(axes_list)
        ax.set_ylabel("Score (Audiobox, 1-10)")
        ax.set_title(f"Audiobox means ± 95% CI — channel: {ch}")
        ax.legend(loc="lower right", fontsize=8)
        name = f"01_means_ci_{ch}.png"
        fig.savefig(out_dir / name, **SAVE_KW)
        plt.close(fig)
        written.append(name)
    return written


def fig02_violin_grid(audiobox_df: pd.DataFrame, runs: list[str], out_dir: Path) -> str:
    """3×3 facet (rows=channel, cols=headline axis) of per-run violins."""
    _set_style()
    labels = short_labels(runs)
    label_map = dict(zip(runs, labels))
    long_rows = []
    for ch in CHANNELS:
        for ax in HEADLINE_AXES:
            col = f"ab_{ch}_{ax}"
            for _, r in audiobox_df.iterrows():
                long_rows.append(
                    {
                        "run": label_map[r["run"]],
                        "channel": ch,
                        "axis": ax,
                        "score": float(r[col]),
                    }
                )
    long_df = pd.DataFrame(long_rows)
    long_df["run"] = pd.Categorical(long_df["run"], categories=labels, ordered=True)
    g = sns.catplot(
        data=long_df,
        kind="violin",
        x="run",
        y="score",
        row="channel",
        col="axis",
        row_order=list(CHANNELS),
        col_order=list(HEADLINE_AXES),
        inner="quartile",
        cut=0,
        height=2.4,
        aspect=1.4,
        sharey=False,
    )
    g.set_titles("{row_name} — {col_name}")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
    g.fig.suptitle("Audiobox score distributions per system", y=1.02)
    name = "02_violin_grid.png"
    g.savefig(out_dir / name, **SAVE_KW)
    plt.close(g.fig)
    return name


def fig03_pairwise_heatmap(
    pairwise: dict[tuple[str, str], pd.DataFrame],
    runs: list[str],
    out_dir: Path,
) -> list[str]:
    """One PNG per channel; 1×3 panels for PQ/CE/CU showing Holm-adj p-values."""
    _set_style()
    labels = short_labels(runs)
    label_map = dict(zip(runs, labels))
    written: list[str] = []
    for ch in CHANNELS:
        fig, axarr = plt.subplots(
            1, len(HEADLINE_AXES), figsize=(13, 5.2), constrained_layout=True
        )
        if len(HEADLINE_AXES) == 1:
            axarr = [axarr]
        for i, ax_name in enumerate(HEADLINE_AXES):
            tbl = pairwise[(ch, ax_name)]
            mat = pd.DataFrame(np.nan, index=labels, columns=labels)
            for _, row in tbl.iterrows():
                a = label_map[row["system_a"]]
                b = label_map[row["system_b"]]
                p = row["p_holm"]
                mat.loc[a, b] = p
                mat.loc[b, a] = p
            sns.heatmap(
                mat.astype(float),
                annot=True,
                fmt=".3f",
                cmap="rocket_r",
                vmin=0.0,
                vmax=1.0,
                cbar=(i == len(HEADLINE_AXES) - 1),
                ax=axarr[i],
                annot_kws={"fontsize": 10},
            )
            # Panel name goes below (as an xlabel) and the run names go above
            # (as top-placed x-ticks).
            axarr[i].set_ylabel("")
            axarr[i].xaxis.tick_top()
            axarr[i].tick_params(axis="x", labelrotation=30)
            for tl in axarr[i].get_xticklabels():
                tl.set_ha("left")
            axarr[i].set_xlabel(ax_name, fontsize=12, fontweight="bold", labelpad=8)
            if i > 0:
                axarr[i].set_yticklabels([])
                axarr[i].tick_params(left=False)
        fig.suptitle(
            f"Paired Wilcoxon (Holm-adjusted p) — channel: {ch}",
        )
        name = f"03_pairwise_heatmap_{ch}.png"
        fig.savefig(out_dir / name, **SAVE_KW)
        plt.close(fig)
        written.append(name)
    return written


def fig04_axis_corr_grid(
    corrs: dict[str, pd.DataFrame], runs: list[str], out_dir: Path
) -> str:
    """1×N facet of 4×4 Pearson heatmaps (one per system)."""
    _set_style()
    labels = short_labels(runs)
    n = len(runs)
    fig, axarr = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    if n == 1:
        axarr = [axarr]
    for i, run in enumerate(runs):
        sns.heatmap(
            corrs[run].loc[list(AUDIOBOX_AXES), list(AUDIOBOX_AXES)],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            cbar=(i == n - 1),
            square=True,
            ax=axarr[i],
        )
        axarr[i].set_title(labels[i], fontsize=10)
    fig.suptitle("Audiobox cross-axis Pearson correlations (channel=mixed)", y=1.02)
    name = "04_axis_corr_grid.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def fig05_tutor_student_gap(
    gap_df: pd.DataFrame, runs: list[str], out_dir: Path
) -> str:
    """Grouped bar: tutor vs student PQ per system + Δ annotation."""
    _set_style()
    labels = short_labels(runs)
    gap_df = gap_df.set_index("run").loc[runs].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xs = np.arange(len(runs))
    w = 0.36
    ax.bar(xs - w / 2, gap_df["tutor"], width=w, label="tutor")
    ax.bar(xs + w / 2, gap_df["student"], width=w, label="student")
    for i, d in enumerate(gap_df["delta_tutor_minus_student"]):
        ax.text(
            xs[i],
            max(gap_df["tutor"].iloc[i], gap_df["student"].iloc[i]) + 0.05,
            f"Δ={d:+.2f}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("PQ score")
    ax.set_title("Tutor vs Student PQ per system — student is always Gemini (control)")
    ax.legend(loc="lower right", fontsize=8)
    name = "05_tutor_student_gap.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def fig06_label_shares(shares_df: pd.DataFrame, runs: list[str], out_dir: Path) -> str:
    """Stacked bar of 5 labels × N systems."""
    _set_style()
    labels = short_labels(runs)
    pivot = shares_df.pivot(index="run", columns="label", values="mean_pct")
    pivot = pivot.loc[runs, list(JUDGE_LABELS)]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bottom = np.zeros(len(runs))
    palette = sns.color_palette("Set2", n_colors=len(JUDGE_LABELS))
    for j, lbl in enumerate(JUDGE_LABELS):
        vals = pivot[lbl].to_numpy()
        ax.bar(range(len(runs)), vals, bottom=bottom, label=lbl, color=palette[j])
        bottom = bottom + vals
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% of frames (per-conv mean)")
    ax.set_title("Judge label shares per system")
    ax.legend(title="label", loc="upper right", fontsize=8)
    _add_caveat(fig)
    name = "06_label_shares.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def fig07_event_rates(shares_df: pd.DataFrame, runs: list[str], out_dir: Path) -> str:
    """2×2 facet: one panel per {I, T, BC, NA}, bar of N systems with CI."""
    _set_style()
    labels = short_labels(runs)
    fig, axarr = plt.subplots(2, 2, figsize=(9.0, 6.4))
    flat = axarr.flatten()
    for i, lbl in enumerate(EVENT_LABELS):
        ax = flat[i]
        sub = (
            shares_df[shares_df["label"] == lbl]
            .set_index("run")
            .loc[runs]
            .reset_index()
        )
        xs = np.arange(len(runs))
        ax.bar(
            xs,
            sub["mean_pct"],
            yerr=sub["ci_half"],
            capsize=3,
            color=sns.color_palette("Set2")[i],
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"% frames labeled {lbl}")
        ax.set_title(f"{lbl}")
    fig.suptitle("Judge event-rate contrasts (per-conv mean ± 95% CI)", y=1.02)
    fig.tight_layout()
    _add_caveat(fig)
    name = "07_event_rates.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def _add_vap_caveat(fig: plt.Figure) -> None:
    from speech_eval.analyze.vap_stats import VAP_CAVEAT

    fig.text(
        0.5,
        -0.04,
        VAP_CAVEAT,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        wrap=True,
    )


def fig08_vap_fi_rate(fi_df: pd.DataFrame, runs: list[str], out_dir: Path) -> str:
    """Responder false-interruption rate per system, tutor vs student."""
    _set_style()
    labels = short_labels(runs)
    scopes = ("tutor", "student")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bar_w = 0.8 / len(scopes)
    for i, scope in enumerate(scopes):
        sub = fi_df[fi_df["scope"] == scope].set_index("run")
        xs = np.arange(len(runs)) + i * bar_w - 0.4 + bar_w / 2
        means = [
            float(sub.loc[r, "mean_fi_rate"]) if r in sub.index else float("nan")
            for r in runs
        ]
        errs = [float(sub.loc[r, "ci_half"]) if r in sub.index else 0.0 for r in runs]
        ax.bar(
            xs, means, width=bar_w, yerr=errs, capsize=3, label=f"{scope}-as-responder"
        )
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("False-interruption rate")
    ax.set_title("VAP responder false-interruption rate (per-conv mean ± 95% CI)")
    ax.legend(loc="upper right", fontsize=8)
    _add_vap_caveat(fig)
    name = "08_vap_fi_rate.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def fig09_vap_cue_dist(cue_df: pd.DataFrame, runs: list[str], out_dir: Path) -> str:
    """Stacked composition of cue labels per system, one panel per holder role."""
    _set_style()
    labels = short_labels(runs)
    roles = ("tutor", "student")
    cue_labels = list(cue_df["label"].drop_duplicates())
    palette = sns.color_palette("Set3", len(cue_labels))
    fig, axarr = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ri, role in enumerate(roles):
        ax = axarr[ri]
        bottoms = np.zeros(len(runs))
        for li, lbl in enumerate(cue_labels):
            sub = cue_df[(cue_df["role"] == role) & (cue_df["label"] == lbl)].set_index(
                "run"
            )
            vals = [float(sub.loc[r, "pct"]) if r in sub.index else 0.0 for r in runs]
            ax.bar(
                np.arange(len(runs)),
                vals,
                bottom=bottoms,
                color=palette[li],
                label=lbl if ri == 0 else None,
            )
            bottoms += np.array(vals)
        ax.set_xticks(np.arange(len(runs)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("% of cue events")
        ax.set_title(f"holder = {role}")
    fig.suptitle("VAP cue-production composition (Yield/Hold cue quality)", y=1.02)
    fig.legend(loc="center right", fontsize=8, bbox_to_anchor=(1.12, 0.5))
    fig.tight_layout()
    _add_vap_caveat(fig)
    name = "09_vap_cue_dist.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def fig10_vap_uptake_latency(
    lat_df: pd.DataFrame, runs: list[str], out_dir: Path
) -> str:
    """Distribution of responder uptake latency per system (pooled scope)."""
    _set_style()
    have = [r for r in runs if r in set(lat_df["run"].unique())]
    labels = short_labels(runs)
    label_for = dict(zip(runs, labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if have:
        order = have
        sns.violinplot(
            data=lat_df[lat_df["run"].isin(have)],
            x="run",
            y="latency_s",
            order=order,
            ax=ax,
            cut=0,
            inner="quartile",
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [label_for[r] for r in order], rotation=20, ha="right", fontsize=8
        )
    ax.set_xlabel("")
    ax.set_ylabel("Uptake latency (s)")
    ax.set_title("VAP responder uptake latency — appropriate-uptake events (pooled)")
    _add_vap_caveat(fig)
    name = "10_vap_uptake_latency.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def _ecdf(ax: plt.Axes, x: np.ndarray, **kw) -> None:
    if x.size == 0:
        return
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    ax.plot(xs, ys, **kw)


def fig11_dist_ecdf(
    dist_by_run: dict[str, dict[str, np.ndarray]],
    systems: list[str],
    reference: str,
    out_dir: Path,
) -> str:
    """ECDF overlay per timing metric: each system against the human reference.

    The reference is drawn bold/black; a system tracking it closely is
    distributionally human-like on that metric (closeness here is necessary, not
    sufficient — see the responder 2×2)."""
    _set_style()
    labels = short_labels(systems)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    flat = axes.ravel()
    for ax, metric in zip(flat, DIST_METRICS):
        lo, hi = DIST_CLIP[metric]

        def clip(x: np.ndarray) -> np.ndarray:
            return x[(x >= lo) & (x <= hi)] if x.size else x

        ref = dist_by_run.get(reference, {}).get(metric, np.array([]))
        _ecdf(ax, clip(ref), label="human ref", color="black", lw=2.5, zorder=5)
        for run, lab in zip(systems, labels):
            arr = dist_by_run.get(run, {}).get(metric, np.array([]))
            _ecdf(ax, clip(arr), label=lab, lw=1.5, alpha=0.9)
        ax.set_title(DIST_LABELS[metric], fontsize=10)
        ax.set_xlabel("seconds")
        ax.set_ylabel("ECDF")
        ax.set_xlim(lo, hi)
    for ax in flat[len(DIST_METRICS) :]:
        ax.axis("off")
    flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Timing distributions vs human reference (`{reference}`)")
    fig.tight_layout()
    name = "11_dist_ecdf.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def fig12_dist_emd(dist_df: pd.DataFrame, systems: list[str], out_dir: Path) -> str:
    """Grouped bars of Wasserstein-EMD (seconds, lower = more human) per metric,
    one bar per system, with bootstrap 95% CI whiskers."""
    _set_style()
    labels = short_labels(systems)
    metrics = list(DIST_METRICS)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    n_sys = len(systems)
    bar_w = 0.8 / max(n_sys, 1)
    for i, run in enumerate(systems):
        sub = dist_df[dist_df["system"] == run].set_index("metric")
        xs = np.arange(len(metrics)) + i * bar_w - 0.4 + bar_w / 2
        emd = [float(sub.loc[m, "emd"]) if m in sub.index else np.nan for m in metrics]
        lo = [
            float(sub.loc[m, "emd_lo"]) if m in sub.index else np.nan for m in metrics
        ]
        hi = [
            float(sub.loc[m, "emd_hi"]) if m in sub.index else np.nan for m in metrics
        ]
        yerr = np.array(
            [
                [max(e - l, 0) for e, l in zip(emd, lo)],
                [max(h - e, 0) for e, h in zip(emd, hi)],
            ]
        )
        ax.bar(xs, emd, width=bar_w, yerr=yerr, capsize=3, label=labels[i])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([DIST_LABELS[m] for m in metrics], rotation=15, ha="right")
    ax.set_ylabel("Wasserstein-EMD vs human (s)")
    ax.set_title("Distributional distance from human reference — lower is more human")
    ax.legend(fontsize=8)
    name = "12_dist_emd.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name


def _student_tutor(run: str) -> tuple[str, str] | None:
    """Parse a `<student>_<tutor>_<suffix>` run name into (student, tutor)."""
    parts = run.split("_")
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def fig13_student_tutor_emd(
    dist_df: pd.DataFrame, systems: list[str], out_dir: Path, metric: str = "fto_s"
) -> str | None:
    """Heatmap of FTO Wasserstein-EMD vs the reference, decomposed by
    student × tutor backend (rows × cols), for run names shaped
    ``<student>_<tutor>_<suffix>``. Returns None (no figure) if no run parses."""
    sub = dist_df[dist_df["metric"] == metric].set_index("system")
    parsed = {
        r: st
        for r in systems
        if (st := _student_tutor(r)) is not None and r in sub.index
    }
    if not parsed:
        return None
    students = sorted({s for s, _ in parsed.values()})
    tutors = sorted({t for _, t in parsed.values()})
    mat = pd.DataFrame(index=students, columns=tutors, dtype=float)
    for run, (s, t) in parsed.items():
        mat.loc[s, t] = float(sub.loc[run, "emd"])

    _set_style()
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(tutors), 1.4 + 0.7 * len(students)))
    sns.heatmap(
        mat.astype(float),
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        vmin=0.0,
        cbar_kws={"label": f"{DIST_LABELS[metric]} EMD vs human (s)"},
        ax=ax,
    )
    ax.set_xlabel("tutor backend")
    ax.set_ylabel("student backend")
    ax.set_title(f"{DIST_LABELS[metric]} distance from human — lower is more human")
    name = "13_student_tutor_emd.png"
    fig.savefig(out_dir / name, **SAVE_KW)
    plt.close(fig)
    return name
