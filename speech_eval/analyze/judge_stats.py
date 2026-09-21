"""Statistics over the ESPnet turn-taking judge frame labels.

All inputs are the long ``judge`` DataFrame from ``loader.AnalysisData``:
columns ``run, conv_index, judge_n_windows, judge_n_frames,
judge_pct_C, judge_pct_NA, judge_pct_I, judge_pct_BC, judge_pct_T``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from speech_eval.analyze.audiobox_stats import _ci95_halfwidth, _holm_bonferroni
from speech_eval.analyze.loader import JUDGE_LABELS

JUDGE_CAVEAT = (
    "Note: judge predictions use non-overlapping 30 s windows, not the "
    "paper's 40 ms sliding hop. Frames after t=30 s see only their own "
    "window's preceding context. See speech_eval/docs/deferred.md item 4."
)

# Headline event labels (silence + the three event types the paper anchors on).
EVENT_LABELS = ("I", "T", "BC", "NA")


def label_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run, per-label mean ± 95% CI over per-conversation percentages."""
    rows = []
    for run, g in df.groupby("run", sort=False):
        for lbl in JUDGE_LABELS:
            col = f"judge_pct_{lbl}"
            x = g[col].dropna().to_numpy()
            m = float(np.mean(x)) if len(x) else float("nan")
            h = _ci95_halfwidth(x)
            rows.append(
                {
                    "run": run,
                    "label": lbl,
                    "n": int(len(x)),
                    "mean_pct": m,
                    "ci_half": h,
                    "lo": m - h,
                    "hi": m + h,
                }
            )
    return pd.DataFrame(rows)


def event_rate_contrasts(
    df: pd.DataFrame,
    *,
    labels: Iterable[str] = EVENT_LABELS,
    runs: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Pairwise paired Wilcoxon on per-conv label percentages.

    One DataFrame per label, with Holm correction applied within that label's
    family of pairwise tests.
    """
    from itertools import combinations

    if runs is None:
        runs = list(dict.fromkeys(df["run"].tolist()))

    out: dict[str, pd.DataFrame] = {}
    for lbl in labels:
        col = f"judge_pct_{lbl}"
        per_run = {
            r: g.set_index("conv_index")[col].dropna()
            for r, g in df[df["run"].isin(runs)].groupby("run", sort=False)
        }
        rows = []
        for a, b in combinations(runs, 2):
            sa = per_run.get(a)
            sb = per_run.get(b)
            if sa is None or sb is None or sa.empty or sb.empty:
                rows.append(
                    {
                        "system_a": a,
                        "system_b": b,
                        "n_pairs": 0,
                        "W": float("nan"),
                        "p_raw": float("nan"),
                    }
                )
                continue
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
            x = joined.iloc[:, 0].to_numpy()
            y = joined.iloc[:, 1].to_numpy()
            try:
                res = stats.wilcoxon(
                    x, y, zero_method="wilcox", alternative="two-sided"
                )
                W = float(res.statistic)
                p = float(res.pvalue)
            except ValueError:
                W = float("nan")
                p = 1.0
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
        if tbl.empty:
            # <2 runs -> no pairs -> empty rows -> no columns. Emit a typed empty
            # frame for this label rather than KeyError-ing on tbl["p_raw"].
            out[lbl] = pd.DataFrame(
                columns=[
                    "system_a",
                    "system_b",
                    "n_pairs",
                    "W",
                    "p_raw",
                    "p_holm",
                    "significant_at_05",
                ]
            )
            continue
        p_raws = tbl["p_raw"].tolist()
        p_for_holm = [1.0 if (p is None or np.isnan(p)) else p for p in p_raws]
        p_holm = _holm_bonferroni(p_for_holm)
        tbl["p_holm"] = [
            float("nan") if np.isnan(raw) else adj for raw, adj in zip(p_raws, p_holm)
        ]
        tbl["significant_at_05"] = tbl["p_holm"].apply(
            lambda v: bool(v <= 0.05) if not (v is None or np.isnan(v)) else False
        )
        out[lbl] = tbl
    return out
