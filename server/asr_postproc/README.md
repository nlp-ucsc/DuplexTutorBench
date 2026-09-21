# asr_postproc

Reference copy of the project that lives at `~/repo/asr_postproc/` on
`ucsc_lab_sv11`. The local Mac client at `alignment/` rsyncs runs to that
remote folder and invokes `align.py` over SSH.

## What it does

For each `(conversation, role)` pair in
`<input_dir>/conversations.jsonl`, runs ASR + forced alignment on
`<input_dir>/audio/{idx}/{role}_full.wav` and writes word-timestamped
segments to `<input_dir>/aligned/<variant>.jsonl`, where `<variant>` is
`<mode>` (default) or `<mode>_no_bias` when `--no-bias` is set on a
Whisper-first mode (`text_mms` has no bias dimension so it always
writes `text_mms.jsonl`). Each `(mode, no_bias)` combination has its
own file so biased / un-biased / cross-aligner variants coexist without
clobbering. Re-segmentation uses the same 1.0 s silence threshold as
`duplex/relay.py` so aligned segments are directly comparable to the
original ones. The output schema mirrors `DuplexConversation.to_dict()`
plus `words[]` per segment plus an `alignment` block recording
`{mode, prompt_bias, asr_model, aligner, ...}`.

## Modes

- `whisper_mms` (default): faster-whisper transcribes the audio (using the
  saved overlong text as `initial_prompt` for soft bias on math/proper
  nouns), then `MahmoudAshraf/mms-300m-1130-forced-aligner` produces
  word timestamps on the ASR output.
- `text_mms`: skip the Whisper pass; feed the saved (overlong) text
  directly to the aligner with `<star>` tokens between every word so
  un-spoken words get absorbed.
- `whisper_mfa`: same Whisper-first shape as `whisper_mms` but the per-segment
  aligner is Montreal Forced Aligner (Kaldi HMM-GMM, CPU-only). MFA
  models silence as a first-class phone, so word boundaries don't absorb
  surrounding silence the way the MMS-CTC path does. Implemented in
  `pipeline_whisper_mfa.py`; Whisper runs with `word_timestamps=True` and
  segments are re-split on per-word gaps > 1.5 s before being handed to
  MFA — required because faster-whisper's `vad_filter` can re-project a
  single emitted segment to span tens of seconds (speech + removed
  silence) on the original timeline, which MFA cannot align across.

## Install

```
cd ~/repo/asr_postproc
uv sync
uv add praatio   # required for whisper_mfa mode (parses MFA's TextGrid output)
```

**`ctc-forced-aligner` must be installed from GitHub, not PyPI.** The
PyPI name `ctc-forced-aligner` is squatted by an unrelated `deskpai`
fork that does not expose the `<star>` skip-token primitive used by
`text_mms` mode — installing it succeeds silently but `text_mms`
produces wrong output. Always use:

```
uv add "git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
```

`whisper_mfa` mode additionally requires a separate conda env containing
Montreal Forced Aligner — it's Kaldi-based and ships with bundled
binaries via conda-forge:

```
~/miniconda3/bin/conda create -y -n mfa-env -c conda-forge montreal-forced-aligner
~/miniconda3/envs/mfa-env/bin/mfa model download acoustic english_mfa
~/miniconda3/envs/mfa-env/bin/mfa model download dictionary english_mfa
```

`pipeline_whisper_mfa.py` invokes `~/miniconda3/envs/mfa-env/bin/mfa` as a
subprocess and injects the env's `bin/` first on `PATH` so MFA can find
`fstcompile` and the other Kaldi tools by name.

## Usage

```
uv run python align.py --input-dir /tmp/asr_postproc_runs/<run> \\
    --mode {whisper_mms|text_mms|whisper_mfa} --device cuda \\
    [--no-bias] [--conv-index N] [--force] [--whisper-model large-v3-turbo]
```

Each variant lives in its own `aligned/<variant>.jsonl`, so re-running
the same `(mode, no_bias)` is idempotent (skips conversations already
in *that* file). Use `--force` to recompute only the targeted variant;
switching `--mode` or toggling `--no-bias` writes to a different file
and leaves the others intact. Output is checkpointed after every
conversation (atomic `.tmp` -> rename).
