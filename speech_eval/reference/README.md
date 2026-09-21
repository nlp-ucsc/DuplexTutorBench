# speech_eval reference corpora

Human-human reference runs that anchor the distributional turn-taking realism
metric (see [`../docs/distributional.md`](../docs/distributional.md)). A
reference is just a synthetic `duplex_output/` run, so it flows through the
**unchanged** vap scorer.

## `maptask_ref` — HCRC Map Task

[HCRC Map Task](https://groups.inf.ed.ac.uk/maptask/) (CC-BY 4.0): 128 dyadic,
task-oriented dialogues. An instruction **giver** guides a **follower** along a
route — asymmetric and goal-directed, mirroring tutor↔student. One mic per
speaker on separate channels (~20 dB separation → clean VAP stereo input) and
NXT **gold** word timings (no forced alignment needed).

`build_maptask.py` turns the corpus into `duplex_output/maptask_ref/`:

- `audio/<idx>/combined.wav` — 16 kHz stereo, **L = giver (tutor)**, **R =
  follower (student)**. The giver/follower→L/R order is auto-detected per
  dialogue by correlating each channel's energy with that role's word activity.
- `aligned/maptask_gold.jsonl` — `conv_index` + role-tagged word timings, in the
  schema the vap scorer consumes (read via `--vap-aligned-variant maptask_gold`).

### Reproduce

Run on the GPU box — audio (~5 GB), `ffmpeg`, and disk live there; the script is
self-contained (stdlib + numpy, no torch, no `speech_eval` import). It downloads
the NXT zip and the 128 `.mix.wav` on first run.

```bash
# on ucsc_lab_sv11 (raw data + processed run kept under /data/jhe516):
python build_maptask.py --raw-dir /data/jhe516/maptask_raw \
    --out-dir /data/jhe516/duplex_output/maptask_ref
# (--limit N for a quick smoke build)

# then, from the Mac, pull the compact processed run down and score it:
rsync -a ucsc_lab_sv11:/data/jhe516/duplex_output/maptask_ref/ duplex_output/maptask_ref/
uv run python -m speech_eval --run-name maptask_ref --component vap \
    --vap-aligned-variant maptask_gold --vap-gpu 0
```

The raw corpus (`/data/jhe516/maptask_raw/`) and the processed run are **not**
committed (the run dir is gitignored like other `duplex_output/` runs); only this
build script is, for reproducibility.
