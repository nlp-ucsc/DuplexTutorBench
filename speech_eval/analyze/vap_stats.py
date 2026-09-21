"""Statistics over the VAP turn-taking scorer outputs.

Inputs are the long ``vap`` DataFrame from ``loader.AnalysisData`` (one row per
(run, conv_index); see ``loader._vap_row`` for columns). Mirrors
``judge_stats`` in shape: per-run means ± 95% CI, per-run cue shares, and
paired-by-conversation Wilcoxon contrasts across systems.

The headline metric is the **responder false-interruption rate** — at moments
VAP predicts the floor-holder keeps the turn, how often the other role grabs it
anyway (a barge-in). The ``tutor`` scope is the model-under-test (student is
always Gemini).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from speech_eval.analyze.audiobox_stats import _ci95_halfwidth, _holm_bonferroni
from speech_eval.analyze.loader import VAP_CELLS, VAP_CUE_LABELS, VAP_SCOPES

VAP_CAVEAT = (
    "Note: thresholds default to VAP's natural 0.5 decision boundary and are "
    "not yet calibrated against a human-human baseline, so absolute rates are "
    "descriptive, not pass/fail. Cross-system comparisons (same thresholds for "
    "all runs) are valid. See speech_eval/docs/deferred.md."
)


def runs_with_vap(vap_df: pd.DataFrame, runs: list[str]) -> list[str]:
    """Subset of ``runs`` that actually have VAP rows, preserving order."""
    if vap_df is None or vap_df.empty:
        return []
    have = set(vap_df["run"].unique())
    return [r for r in runs if r in have]


def fi_rate_summary(vap_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run, per-scope mean ± 95% CI of the per-conversation FI rate."""
    rows = []
    for run, g in vap_df.groupby("run", sort=False):
        for scope in VAP_SCOPES:
            x = g[f"vap_{scope}_FI_rate"].dropna().to_numpy()
            m = float(np.mean(x)) if len(x) else float("nan")
            h = _ci95_halfwidth(x)
            rows.append(
                {
                    "run": run,
                    "scope": scope,
                    "n": int(len(x)),
                    "mean_fi_rate": m,
                    "ci_half": h,
                    "lo": m - h,
                    "hi": m + h,
                }
            )
    return pd.DataFrame(rows)


def cell_totals(vap_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run, per-scope summed 2x2 cell counts across all conversations."""
    rows = []
    for run, g in vap_df.groupby("run", sort=False):
        for scope in VAP_SCOPES:
            counts = {c: int(g[f"vap_{scope}_{c}"].sum()) for c in VAP_CELLS}
            fi, ar = counts["false_interruption"], counts["appropriate_restraint"]
            up, mi = counts["appropriate_uptake"], counts["missed_yield"]
            rows.append(
                {
                    "run": run,
                    "scope": scope,
                    **counts,
                    "fi_rate": (fi / (fi + ar)) if (fi + ar) else float("nan"),
                    "uptake_rate": (up / (up + mi)) if (up + mi) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def cue_shares(vap_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run, per-holder-role share (%) of each cue label."""
    rows = []
    for run, g in vap_df.groupby("run", sort=False):
        for role in ("tutor", "student"):
            totals = {
                lbl: int(g[f"vap_cue_{role}_{lbl}"].sum()) for lbl in VAP_CUE_LABELS
            }
            grand = sum(totals.values())
            for lbl in VAP_CUE_LABELS:
                rows.append(
                    {
                        "run": run,
                        "role": role,
                        "label": lbl,
                        "count": totals[lbl],
                        "pct": (100.0 * totals[lbl] / grand) if grand else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def uptake_latencies(vap_df: pd.DataFrame, scope: str = "pooled") -> pd.DataFrame:
    """Long (run, latency_s) table exploded from per-conversation lists."""
    rows = []
    col = f"vap_{scope}_uptake_lat"
    for _, r in vap_df.iterrows():
        for v in r.get(col) or []:
            rows.append({"run": r["run"], "latency_s": float(v)})
    return pd.DataFrame(rows)


def fi_rate_contrasts(
    vap_df: pd.DataFrame,
    *,
    scope: str = "tutor",
    runs: list[str] | None = None,
) -> pd.DataFrame:
    """Pairwise paired Wilcoxon on per-conv FI rate across systems (Holm).

    Paired by ``conv_index`` (shared MathVista question). Returns an empty
    frame when fewer than two runs have VAP data.
    """
    if runs is None:
        runs = list(dict.fromkeys(vap_df["run"].tolist()))
    runs = runs_with_vap(vap_df, runs)
    if len(runs) < 2:
        return pd.DataFrame(
            columns=["system_a", "system_b", "n_pairs", "W", "p_raw", "p_holm"]
        )

    col = f"vap_{scope}_FI_rate"
    per_run = {
        r: g.set_index("conv_index")[col].dropna()
        for r, g in vap_df[vap_df["run"].isin(runs)].groupby("run", sort=False)
    }
    rows = []
    for a, b in combinations(runs, 2):
        sa, sb = per_run.get(a), per_run.get(b)
        joined = pd.concat([sa, sb], axis=1, join="inner").dropna()
        if len(joined) < 2:
            rows.append(
                {
                    "system_a": a,
                    "system_b": b,
                    "n_pairs": int(len(joined)),
                    "W": float("nan"),
                    "p_raw": float("nan"),
                }
            )
            continue
        x, y = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
        try:
            res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            W, p = float(res.statistic), float(res.pvalue)
        except ValueError:
            W, p = float("nan"), 1.0
        rows.append(
            {
                "system_a": a,
                "system_b": b,
                "n_pairs": int(len(joined)),
                "W": W,
                "p_raw": p,
            }
        )

    tbl = pd.DataFrame(rows)
    p_for_holm = [1.0 if (p is None or np.isnan(p)) else p for p in tbl["p_raw"]]
    p_holm = _holm_bonferroni(p_for_holm)
    tbl["p_holm"] = [
        float("nan") if np.isnan(raw) else adj for raw, adj in zip(tbl["p_raw"], p_holm)
    ]
    tbl["significant_at_05"] = tbl["p_holm"].apply(
        lambda v: bool(v <= 0.05) if not (v is None or np.isnan(v)) else False
    )
    return tbl
