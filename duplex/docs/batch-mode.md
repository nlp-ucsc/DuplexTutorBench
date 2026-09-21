# Batch mode: scaling, resume, sweeps

Batch mode is built for large sweeps. All flags shown here are batch-only — the web UI ignores them with a warning.

## Multiple attempts per question

`--attempts N` generates N independent conversations per question. With `--n 50 --attempts 5` you get 250 rows; each row is keyed by `(pid, attempt_index)` in the JSONL.

## Auto-resume

Rerunning the same `--run-name` skips `(pid, attempt)` pairs already in `conversations.jsonl`. No flag to opt in — interrupted runs just pick up where they left off. Audio dirs orphaned by an interrupt mid-conversation are wiped on the next run before being reused. Delete the run dir to start over.

## Manifest

The first batch invocation writes `duplex_output/{run_name}/manifest.json` with the full args, inlined prompt contents, the resolved `target_pids`, the command line, and the git SHA. Subsequent invocations validate against it and **error out** on any output-affecting mismatch (backends, models, voices, prompts, `max_duration`, `split`, `attempts`); pass `--force` to override. Operational flags (`--web`, `--port`, `--concurrency`, hosts/ports) are not validated.

## Replay

```bash
python -m duplex --replay duplex_output/run_a/manifest.json --run-name run_a_v2
```

Reproduces a dataset using the inlined prompts and resolved pid list — independent of MathVista row-order changes or local prompt-file edits.

## Concurrency

`--concurrency 8` fans out parallel conversations through `asyncio.Semaphore` and `asyncio.gather`. **Cloud-API backends only**; PersonaPlex / MoshiVis hold per-port locks and are rejected at startup. JSONL idx allocation + audio-dir creation are serialized under a lock so concurrent rows don't collide.

## Sweep helper

`scripts/run_sweep.py experiments.yaml` invokes `python -m duplex` per row of a YAML/JSON config, sequentially. Each child run is independently resumable; a killed sweep can be re-run and only the in-progress child needs to catch up.

```yaml
# experiments.yaml
runs:
  - name: gpt_v1
    tutor_backend: gpt-realtime
    student_backend: gpt-realtime
    sample: 50
    seed: 0
    attempts: 5
    concurrency: 8
  - name: gemini_v1
    tutor_backend: gemini-live
    student_backend: gemini-live
    sample: 50
    seed: 0
    attempts: 5
    concurrency: 8
```

```bash
uv run python scripts/run_sweep.py experiments.yaml             # run
uv run python scripts/run_sweep.py experiments.yaml --dry-run   # show commands only
```
