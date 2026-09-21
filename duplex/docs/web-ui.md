# Web UI

Launch with `--web` (no `--run-name` required). Open <http://localhost:5002>.

## Run picker

The header has a `Run:` dropdown that lists all subdirs under `duplex_output/` (mtime-desc, with conversation counts) plus a `+ New run...` entry that creates a fresh dataset directory via `POST /api/runs` and reloads into it. Switching runs uses `?run=<name>` (page reload, mirrors `viewer.py`) so history, alignment, and evaluation panels rebuild against the new run with no shared state to invalidate.

The dropdown locks while a recording is active to prevent the in-flight save from landing in the wrong dataset; the server also rejects `/api/start` calls whose `run` body disagrees with the active selection.

## Monitor (live conversation)

- Pick the tutor and student backends from the two **dropdowns in the header**. The CLI `--tutor-backend` / `--student-backend` values seed the initial selection; the dropdowns are authoritative from there on and can be changed between runs without restarting the server.
- For self-hosted backends (`personaplex`, `moshivis`), a small server widget shows a coloured dot + **Start Server** / **Check** / **Kill** buttons.
    - **Start Server** SSHes into `ucsc_lab_sv11` and launches the right `moshi.server` / MoshiVis process (after a TCP pre-check that skips SSH if the port is already listening); the dot auto-polls every 3 s for ~2 min and flips green once the port is up.
    - **Check** is a manual one-shot probe.
    - **Kill** stops only this role's instance — the narrow `pkill -f … --port N` pattern is per-port so the other role's server keeps running.
- Cloud backends (`gpt-realtime`, `gemini-live`) hide the server widget — they only need the API key.
- Select a MathVista question from the dropdown.
- Click **Start** to begin a live full-duplex conversation.
- Watch real-time text transcription from both agents.
- Listen to live audio playback with per-role volume controls.

## History (saved conversation playback)

- Browse all saved conversations in the left panel.
- Play synchronized tutor/student audio with a progress bar.
- Click any segment to seek to that point.
- Adjust playback speed (0.5×–2×) and per-role volume.
- If an aligned variant exists for the run, a variant `<select>` next to the *Segments* tab lets you switch between `(original)` and each `aligned/*.jsonl` (`whisper_mms`, `whisper_mfa`, `text_mms`, …). See [`../../alignment/README.md`](../../alignment/README.md).

## Evaluation panel

The evaluation framework mounts into the same web app — per-conversation scores appear in the History panel, and a run-level summary is at `/eval-summary`. See [`../../evaluation/README.md`](../../evaluation/README.md).

## VAP turn-taking figure

When a conversation has been scored by `python -m speech_eval … vap`, the History view shows a Figure-2-style panel under the audio controls (reproducing the VAP evaluator paper, [arXiv:2305.17971](https://arxiv.org/abs/2305.17971)): a mel-spectrogram of `combined.wav`, the forced-aligned words, and two stacked **P(now)** / **P(future)** ribbons (blue = tutor holds the floor, red = student) with the detected turn-taking events marked. The playback cursor sweeps the whole figure in sync with the audio; click anywhere on it to seek.

Because a conversation runs for minutes, the figure can be **zoomed to the second level**: scroll the wheel over it to zoom (anchored at the pointer), drag to pan, or use the **− / + / Reset** buttons; the readout shows the visible window. While zoomed, the figure auto-pages to keep the playhead in view, and seeking re-centres on the target. The ribbons are drawn from the full 50 Hz trajectories, so they stay crisp at any zoom.

It's a read-only overlay — the per-frame trajectories come from `speech_eval_output/<run>/vap_frames/<idx>.npz` and `vap.jsonl`, so the panel only appears for runs that have a matching VAP score (it stays hidden otherwise). The model is never run from the web app. See [`../../speech_eval/README.md`](../../speech_eval/README.md).
