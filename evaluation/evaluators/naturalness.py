"""Speech naturalness MOS prediction (DNSMOS) per role.

Uses the `speechmos` package, which ships ONNX-based DNSMOS
(SIG / BAK / OVRL / P.808 MOS — all non-intrusive, no reference audio).
The ONNX weights are bundled inside the wheel itself (~2.8 MB total under
`site-packages/speechmos/dnsmos_models/`), so there's no runtime download.

If `speechmos` is missing for some reason, this evaluator self-skips
rather than crashing the whole run — but it is a hard dependency in
`pyproject.toml`, so a normal `uv sync` is sufficient.

Note: `speechmos` does NOT bundle UTMOS, only DNSMOS / PLCMOS / AECMOS.
DNSMOS is similarly standard for non-intrusive MOS prediction (Microsoft
DNS Challenge series); swapping in another predictor later is a one-line
change inside `_score`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evaluation.base import Evaluator

logger = logging.getLogger(__name__)


def _try_import() -> tuple[Any, str | None]:
    try:
        from speechmos import dnsmos  # type: ignore[import-not-found]

        return dnsmos, None
    except ImportError as e:
        return None, str(e)


def _load_resampled(path: Path, target_sr: int = 16000):
    """Load a WAV and resample to `target_sr` if needed.

    DNSMOS expects 16 kHz; our duplex audio is 24 kHz, so we always
    have to convert. Imported lazily so the framework stays usable
    without the mos extras installed.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == target_sr:
        return audio, sr
    n_in = audio.shape[0]
    n_out = int(round(n_in * target_sr / sr))
    if n_out <= 0:
        return audio, sr
    x_in = np.linspace(0.0, 1.0, n_in, endpoint=False, dtype=np.float64)
    x_out = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float64)
    out = np.interp(x_out, x_in, audio).astype(np.float32)
    return out, target_sr


def _score(dnsmos, path: Path) -> dict[str, float] | None:
    """Run DNSMOS on a wav path; return SIG / BAK / OVRL / P808."""
    audio, sr = _load_resampled(path, target_sr=16000)
    if audio.size == 0:
        return None
    result = dnsmos.run(audio, sr=sr)
    # `dnsmos.run` returns a dict with keys ovrl_mos, sig_mos, bak_mos, p808_mos
    return {
        "ovrl": float(result["ovrl_mos"]),
        "sig": float(result["sig_mos"]),
        "bak": float(result["bak_mos"]),
        "p808": float(result["p808_mos"]),
    }


class SpeechNaturalness(Evaluator):
    name = "naturalness"
    requires_audio = True

    def __init__(self):
        self._dnsmos, self._import_err = _try_import()

    def evaluate(self, conv: dict, audio_dir: Path) -> dict[str, Any]:
        if self._dnsmos is None:
            return {
                "status": "skipped",
                "reason": f"speechmos not installed: {self._import_err}. "
                "Run `uv sync` to install.",
            }

        out: dict[str, Any] = {"status": "ok", "metric": "dnsmos"}
        for role in ("tutor", "student"):
            wav = audio_dir / f"{role}_full.wav"
            if not wav.is_file():
                out[f"{role}_dnsmos"] = None
                continue
            try:
                scores = _score(self._dnsmos, wav)
                if scores is None:
                    out[f"{role}_dnsmos"] = None
                else:
                    out[f"{role}_dnsmos"] = {k: round(v, 3) for k, v in scores.items()}
            except Exception as e:
                logger.warning("DNSMOS failed for %s: %s", wav, e)
                out[f"{role}_dnsmos"] = None
                out[f"{role}_error"] = str(e)
        return out
