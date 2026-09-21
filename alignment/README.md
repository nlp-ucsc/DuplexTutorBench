# `alignment/` — ASR-based transcript alignment

Cleans up duplex transcripts by re-deriving them from the saved per-role
audio with word-level timestamps. The original `conversations.jsonl` is
kept as-is; this module produces one file per `(mode, no_bias)` variant
under `duplex_output/{run_name}/aligned/<variant>.jsonl` in the same
shape as the original plus `words[]` per segment. Valid variant keys:
`whisper_mms`, `whisper_mms_no_bias`, `whisper_mfa`,
`whisper_mfa_no_bias`, `text_mms` (no bias dimension — the aligner
doesn't take a Whisper prompt). Multiple variants coexist side-by-side
so biased vs un-biased and MMS vs MFA can be A/B-compared without
re-running.

## Why

The duplex relay flushes a text segment when it sees ≥1.0 s of token
silence, but for cloud backends (`gpt-realtime`, `gemini-live`) the model
streams its full intended text far faster than the audio plays. So the
saved segment timestamps are essentially "when the text deltas arrived",
not "when the audio was heard". On top of that, when a barge-in happens
the audio is truncated but the already-buffered text is not.

ASR over the saved WAVs gives ground truth.

## Modes

- `whisper_mms` (default) — faster-whisper transcribes the audio (with the
  original overlong text passed in as `initial_prompt` for soft bias on
  proper nouns / math), then `MahmoudAshraf/mms-300m-1130-forced-aligner`
  produces word timestamps inside each Whisper VAD segment. Pass
  `--no-bias` to skip the `initial_prompt` and let Whisper transcribe
  purely from audio (useful as a baseline against the bias-on run).
- `text_mms` — single-pass: feed the original (overlong) text per
  segment into the aligner with `<star>` between every word; un-spoken
  words get absorbed and only the spoken ones come back with timestamps.
  Preserves the original segment count + intended text per segment.
- `whisper_mfa` — same Whisper-first shape as `whisper_mms` but swaps the
  MMS-CTC aligner for Montreal Forced Aligner (Kaldi HMM-GMM, CPU-only).
  MFA models silence as an explicit phone, so word boundaries don't
  absorb leading/trailing silence the way CTC alignment does. Tradeoffs:
  per-word `confidence` is reported as a constant `1.0` (TextGrids carry
  no per-word likelihood), and very short or unalignable per-role
  batches may produce no output for that role — the other role is
  unaffected. Whisper runs with `word_timestamps=True` in this mode so
  the pipeline can re-split each Whisper segment on per-word gaps
  > 1.5 s before sending it to MFA (faster-whisper's `vad_filter` can
  otherwise produce single segments whose timestamps span tens of
  seconds covering speech + removed silence on the original timeline,
  which MFA cannot align across).

## Run

Heavy lifting happens on `ucsc_lab_sv11`; the local CLI rsyncs audio +
the source jsonl to the remote, invokes `~/repo/asr_postproc/align.py`
over SSH, and rsyncs the resulting `aligned/<variant>.jsonl` back.

```
uv run python -m alignment --run-name <run> [--mode whisper_mms|text_mms|whisper_mfa]
                                            [--no-bias]
                                            [--conv-index N] [--force]
                                            [--keep-remote-tmp]
```

Idempotent — each variant lives in its own file (`aligned/whisper_mms.jsonl`
vs `aligned/whisper_mms_no_bias.jsonl`, etc.), so re-running the same
mode + bias skips already-processed conversations. Switching `--mode` or
toggling `--no-bias` writes to a *different* file and leaves the old one
untouched — variants accumulate side-by-side. `--force` recomputes only
the targeted variant file.

The remote project lives at `~/repo/asr_postproc/`; a reference copy is
under `server/asr_postproc/` (same convention as `server/omni_server.py`).

## Web UI integration

`alignment.web_integration.register_routes(app, get_run_dir=lambda: self.run_dir)`
wires `/api/aligned/variants` + `/api/aligned/{variant}/{idx}` into the
duplex web app. `get_run_dir` is re-read per request so the picker
follows the user's run selection. The template has `<!--ALIGN_TOGGLE-->`
and `<!--ALIGN_SCRIPT-->` markers that get replaced at request time
with a variant `<select>` next to the Segments tab — when at least one
aligned variant exists, the picker lists `(original)` plus every
`aligned/*.jsonl` variant (e.g. `whisper_mms`, `whisper_mms_no_bias`,
`whisper_mfa`, `text_mms`). Selecting a variant lazily fetches its
record, swaps the segments view to that variant's words (with
hover-tooltips on each word showing per-word timestamps + confidence,
and click-to-seek into `combined.wav`); `(original)` flips back to the
overlong view for comparison.

## Evaluation integration

`uv run python -m evaluation run --run-name <run> --align-variant <variant>`
loads `aligned/<variant>.jsonl` instead of the original and writes
scores to `eval_output/{run}__<variant>/scores.jsonl`, so original and
each aligned variant accumulate side-by-side and you can A/B them.
Replaces the previous boolean `--aligned` flag.

## Output schema

One JSON object per line, mirroring `DuplexConversation.to_dict()`:

```json
{
  "pid": "...",
  "attempt_index": 0,
  "conv_index": 0,
  "duration": 60.5,
  "tutor_voice": "...", "student_voice": "...",
  "tutor_backend": "gpt-realtime", "student_backend": "gemini-live",
  "alignment": {
    "mode": "whisper_mms",
    "asr_model": "large-v3-turbo",
    "aligner": "MahmoudAshraf/mms-300m-1130-forced-aligner",
    "language": "en",
    "prompt_bias": true,
    "timestamp": "...",
    "source_segments_count": {"tutor": 6, "student": 7},
    "aligned_segments_count": {"tutor": 6, "student": 8}
  },
  "segments": [
    {
      "role": "tutor",
      "text": "Let's think step by step.",
      "start_time": 13.12,
      "end_time": 14.40,
      "words": [
        {"word": "Let's", "start_time": 13.12, "end_time": 13.36, "confidence": -0.02},
        ...
      ]
    }
  ]
}
```
