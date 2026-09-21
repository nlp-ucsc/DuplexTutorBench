"""Unit tests for the distributional turn-taking realism metric.

Plain-assert tests (no pytest dependency). Run from anywhere:

    uv run python tests/test_distributional.py

Covers:
* ``vap_events.compute_distributions`` on a hand-built timeline with known
  gaps / overlaps / pauses / turn durations.
* ``distributional_stats.reference_distances`` on synthetic vap.jsonl runs
  (identical distribution -> ~0 distance; shifted -> clearly positive), plus the
  no-system / no-reference empty-frame guards.
* The previously-crashing single-run guards in ``audiobox_stats`` /
  ``judge_stats`` (KeyError 'p_raw').
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from speech_eval import vap_events  # noqa: E402
from speech_eval.analyze import (  # noqa: E402
    audiobox_stats,
    distributional_stats,
    judge_stats,
)


def _word(s: float, e: float) -> dict:
    return {"word": "x", "start_time": s, "end_time": e, "confidence": 1.0}


def _seg(role: str, s: float, e: float) -> dict:
    return {
        "role": role,
        "text": "x",
        "start_time": s,
        "end_time": e,
        "words": [_word(s, e)],
    }


def test_compute_distributions():
    # tutor [0,1]; student [1.5,2.5] -> gap 0.5; tutor [2.3,3.0] -> overlap 0.2;
    # tutor [3.4,4.0] -> within-turn pause 0.4. Turn durs: 1.0,0.7,0.6 (tutor),1.0.
    segs = [
        _seg("tutor", 0.0, 1.0),
        _seg("student", 1.5, 2.5),
        _seg("tutor", 2.3, 3.0),
        _seg("tutor", 3.4, 4.0),
    ]
    d = vap_events.compute_distributions(segs)
    assert d["fto_s"] == [0.5, -0.2], d["fto_s"]
    assert d["gap_s"] == [0.5], d["gap_s"]
    assert d["overlap_s"] == [0.2], d["overlap_s"]
    assert d["within_pause_s"] == [0.4], d["within_pause_s"]
    assert sorted(d["turn_dur_s"]) == [0.6, 0.7, 1.0, 1.0], d["turn_dur_s"]


def test_compute_distributions_empty():
    d = vap_events.compute_distributions([])
    assert all(d[k] == [] for k in vap_events.DIST_KEYS), d


def _write_vap_jsonl(run_dir: pathlib.Path, fto_values: list[float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # One conversation row carrying just the distributions block the metric reads.
    row = {
        "conv_index": 0,
        "distributions": {
            "fto_s": fto_values,
            "gap_s": [v for v in fto_values if v >= 0],
            "overlap_s": [-v for v in fto_values if v < 0],
            "within_pause_s": [],
            "turn_dur_s": [],
        },
    }
    (run_dir / "vap.jsonl").write_text(json.dumps(row) + "\n")


def test_reference_distances():
    rng = np.random.default_rng(0)
    ref_vals = list(rng.normal(0.2, 0.1, 400))
    same_vals = list(rng.normal(0.2, 0.1, 400))  # same distribution as ref
    shift_vals = list(rng.normal(0.7, 0.1, 400))  # shifted +0.5 s
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _write_vap_jsonl(root / "ref", ref_vals)
        _write_vap_jsonl(root / "sys_same", same_vals)
        _write_vap_jsonl(root / "sys_shift", shift_vals)

        df = distributional_stats.reference_distances(
            root, ["sys_same", "sys_shift"], "ref", n_bootstrap=100
        )
        fto = df[df["metric"] == "fto_s"].set_index("system")
        emd_same = float(fto.loc["sys_same", "emd"])
        emd_shift = float(fto.loc["sys_shift", "emd"])
        assert emd_same < 0.05, emd_same
        assert emd_shift > 0.3, emd_shift
        # CI brackets the point estimate.
        assert (
            fto.loc["sys_shift", "emd_lo"]
            <= emd_shift
            <= fto.loc["sys_shift", "emd_hi"]
        )
        # KS also orders them.
        assert float(fto.loc["sys_same", "ks"]) < float(fto.loc["sys_shift", "ks"])


def test_reference_distances_guards():
    cols = distributional_stats._RESULT_COLUMNS
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _write_vap_jsonl(root / "ref", [0.1, 0.2, 0.3])
        # No systems.
        e1 = distributional_stats.reference_distances(root, [], "ref")
        assert e1.empty and list(e1.columns) == cols
        # No reference.
        e2 = distributional_stats.reference_distances(root, ["ref"], None)
        assert e2.empty and list(e2.columns) == cols
        # Reference == only system (filtered out -> no systems left).
        e3 = distributional_stats.reference_distances(root, ["ref"], "ref")
        assert e3.empty and list(e3.columns) == cols


def test_single_run_guards():
    # audiobox: one run -> paired_wilcoxon_table must not KeyError on 'p_raw'.
    ab = pd.DataFrame(
        [
            {"run": "only", "conv_index": 0, "ab_tutor_PQ": 5.0},
            {"run": "only", "conv_index": 1, "ab_tutor_PQ": 6.0},
        ]
    )
    out = audiobox_stats.paired_wilcoxon_table(ab, channel="tutor", axis="PQ")
    assert out.empty and "p_raw" in out.columns

    # judge: one run -> event_rate_contrasts returns per-label empty frames.
    jg = pd.DataFrame(
        [
            {"run": "only", "conv_index": 0, "judge_pct_I": 1.0, "judge_pct_T": 2.0},
            {"run": "only", "conv_index": 1, "judge_pct_I": 1.5, "judge_pct_T": 2.5},
        ]
    )
    res = judge_stats.event_rate_contrasts(jg, labels=("I", "T"))
    for lbl in ("I", "T"):
        assert res[lbl].empty and "p_raw" in res[lbl].columns


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} passed")


if __name__ == "__main__":
    main()
