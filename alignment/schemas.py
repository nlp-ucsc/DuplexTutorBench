"""Read/write helpers for per-variant aligned conversation files.

Each `(mode, no_bias)` combination is stored in its own JSONL file under
`<run_dir>/aligned/<variant_key>.jsonl`, so biased and un-biased outputs can
coexist for the same conversation. The record shape inside each file still
mirrors `DuplexConversation.to_dict()` plus per-segment `words[]` so existing
evaluators read it without changes.

Valid variant keys:
    whisper_mms, whisper_mms_no_bias,
    whisper_mfa, whisper_mfa_no_bias,
    text_mms
"""

from __future__ import annotations

import json
from pathlib import Path

ALIGNED_SUBDIR = "aligned"


def variant_key(mode: str, no_bias: bool) -> str:
    """Stable filename stem for a (mode, no_bias) variant.

    `text_mms` has no bias dimension — the aligner doesn't take Whisper's
    prompt — so it gets a single key with no `_no_bias` partner.
    """
    if mode == "text_mms":
        return "text_mms"
    return f"{mode}_no_bias" if no_bias else mode


def aligned_dir_for(run_dir: Path) -> Path:
    return run_dir / ALIGNED_SUBDIR


def aligned_path_for(run_dir: Path, variant: str) -> Path:
    return aligned_dir_for(run_dir) / f"{variant}.jsonl"


def list_aligned_variants(run_dir: Path) -> list[str]:
    """Sorted variant keys present on disk for a run."""
    d = aligned_dir_for(run_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))


def load_aligned_conversations(jsonl_path: Path) -> list[dict]:
    """Same shape as `duplex.storage.load_conversations`."""
    if not jsonl_path.is_file():
        return []
    out: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def aligned_by_index(jsonl_path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for rec in load_aligned_conversations(jsonl_path):
        idx = rec.get("conv_index")
        if idx is None:
            continue
        out[int(idx)] = rec
    return out
