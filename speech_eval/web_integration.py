"""Hooks for the duplex web UI to render a VAP turn-taking figure.

Reproduces Figure 2 of the VAP evaluator paper (arXiv:2305.17971) inside the
History view: a mel-spectrogram of ``combined.wav`` with the forced-aligned
words above two stacked probability ribbons — P(now) and P(future) — that show
who holds the floor moment-to-moment. A playback cursor sweeps the whole figure
in sync with the audio.

The per-frame ``p_now`` / ``p_future`` trajectories already live in
``speech_eval_output/<run>/vap_frames/<idx>.npz`` (dumped by ``_vap_infer.py``);
the words come from the same ``aligned/<variant>.jsonl`` the scorer used, and the
turn-taking events from ``vap.jsonl``. So this is purely a read-side overlay — it
never runs the model.

Wired identically to ``alignment.web_integration`` / ``evaluation.web_integration``:

    from speech_eval.web_integration import (
        register_routes, render_panel_html, render_panel_script,
    )
    register_routes(
        app,
        get_run_dir=lambda: self.run_dir,             # duplex_output/<run>
        get_vap_dir=lambda: self._vap_dir(),          # speech_eval_output/<run>
    )
    html = HTML_TEMPLATE.replace("<!--VAP_PANEL-->", render_panel_html())
    html = html.replace("<!--VAP_SCRIPT-->", render_panel_script())

The browser calls ``window.loadVap(idx)`` from ``selectConversation`` and
``window.vapUpdateCursor(t)`` from the playback loop's ``highlightSegments``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np
from aiohttp import web

from speech_eval import vap_events

logger = logging.getLogger(__name__)

# Cap the number of trajectory points shipped to the browser. Kept high enough
# to preserve the full 50 Hz frame rate for a typical run (~300 s ⇒ 15 000
# frames) so the ribbons stay crisp when the user zooms in to the second level;
# longer conversations are downsampled by an integer stride.
_MAX_POINTS = 15000

# Rendered spectrogram PNGs are cached in-process, keyed by (path, mtime), so
# re-opening a conversation doesn't recompute. Small, single-user UI; cap the
# dict so a long session can't grow it unbounded.
_spec_cache: dict[tuple[str, float], bytes] = {}
_SPEC_CACHE_MAX = 24


def _vap_row(vap_dir: Path, idx: int) -> dict:
    """The ``vap.jsonl`` row for ``idx`` (events + aligned_variant), or {}."""
    path = vap_dir / "vap.jsonl"
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if int(row.get("conv_index", -1)) == idx:
                    return row
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _load_words(run_dir: Path, variant: str, idx: int) -> tuple[list[dict], float]:
    """Flatten the aligned timeline into role-tagged word boxes for ``idx``.

    Returns ``(words, duration)``; ``duration`` is the conversation length from
    the aligned record (0.0 if unknown).
    """
    path = run_dir / "aligned" / f"{variant}.jsonl"
    if not path.is_file():
        return [], 0.0
    try:
        rows = vap_events.load_aligned_conversations(path)
    except (OSError, json.JSONDecodeError, KeyError):
        return [], 0.0
    row = rows.get(idx)
    if row is None:
        return [], 0.0
    words: list[dict] = []
    for seg in row.get("segments", []):
        role = seg.get("role", "")
        for w in seg.get("words") or []:
            s, e = w.get("start_time"), w.get("end_time")
            if s is None or e is None:
                continue
            words.append(
                {
                    "w": w.get("word", ""),
                    "s": round(float(s), 3),
                    "e": round(float(e), 3),
                    "role": role,
                }
            )
    return words, float(row.get("duration", 0.0) or 0.0)


def _build_payload(vap_dir: Path, run_dir: Path, idx: int) -> dict:
    """Read frames + words + events and assemble the JSON for the figure."""
    npz_path = vap_dir / "vap_frames" / f"{idx}.npz"
    data = np.load(npz_path)
    t = data["frame_times_original"].astype(float)
    n = len(t)
    stride = max(1, math.ceil(n / _MAX_POINTS))
    sel = slice(0, n, stride)

    def col(key: str) -> list[float]:
        return [round(float(v), 4) for v in data[key][sel]]

    try:
        header = json.loads(str(data["header"]))
    except (KeyError, json.JSONDecodeError):
        header = {}

    row = _vap_row(vap_dir, idx)
    variant = row.get("aligned_variant") or "whisper_mfa"
    channel_order = row.get("channel_order") or ["tutor", "student"]
    words, duration = _load_words(run_dir, variant, idx)

    events = [
        {
            "t": round(float(e.get("t_evt", 0.0)), 3),
            "type": e.get("type", ""),
            "pred": e.get("vap_pred", ""),
            "cell": e.get("cell", ""),
            "holder": e.get("holder", ""),
            "responder": e.get("responder", ""),
        }
        for e in row.get("events", [])
        if e.get("t_evt") is not None
    ]

    return {
        "available": True,
        "conv_index": idx,
        "frame_hz": int(header.get("frame_hz", row.get("frame_hz", 50)) or 50),
        "channel_order": channel_order,
        "aligned_variant": variant,
        "duration": round(duration, 3),
        "t": [round(float(v), 3) for v in t[sel]],
        "p_now": [col("p_now0"), col("p_now1")],
        "p_future": [col("p_future0"), col("p_future1")],
        "words": words,
        "events": events,
    }


def _render_spectrogram_png(wav_path: Path) -> bytes:
    """Full-bleed log-mel spectrogram of the mixed (mono) audio, as PNG bytes.

    Axes fill the figure edge-to-edge with ``extent=[0, dur]`` so the browser
    can stretch the image to the timeline and have x map linearly to seconds.
    """
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    dur = len(y) / sr if sr else 0.0
    if y.size == 0:
        y = np.zeros(sr or 16000, dtype=np.float32)
        dur = 1.0

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=400, hop_length=160, n_mels=80, fmax=sr / 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(14, 1.8), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(0.0, dur, 0.0, mel_db.shape[0]),
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def register_routes(
    app: web.Application,
    get_run_dir: Callable[[], Path | None],
    get_vap_dir: Callable[[], Path | None],
) -> None:
    async def handle_vap(request: web.Request) -> web.Response:
        vap_dir = get_vap_dir()
        run_dir = get_run_dir()
        if vap_dir is None or run_dir is None:
            return web.json_response({"available": False})
        idx = int(request.match_info["idx"])
        if not (vap_dir / "vap_frames" / f"{idx}.npz").is_file():
            return web.json_response({"available": False})
        loop = asyncio.get_event_loop()
        try:
            payload = await loop.run_in_executor(
                None, _build_payload, vap_dir, run_dir, idx
            )
        except Exception:  # noqa: BLE001 — surface as "no data", don't 500 the UI
            logger.exception("VAP: failed to build payload for conv %d", idx)
            return web.json_response({"available": False})
        return web.json_response(payload)

    async def handle_spectrogram(request: web.Request) -> web.Response:
        run_dir = get_run_dir()
        if run_dir is None:
            return web.Response(status=404, text="no run selected")
        idx = int(request.match_info["idx"])
        wav = run_dir / "audio" / str(idx) / "combined.wav"
        if not wav.is_file():
            return web.Response(status=404, text="no combined.wav")
        key = (str(wav), wav.stat().st_mtime)
        png = _spec_cache.get(key)
        if png is None:
            loop = asyncio.get_event_loop()
            try:
                png = await loop.run_in_executor(None, _render_spectrogram_png, wav)
            except Exception:  # noqa: BLE001
                logger.exception("VAP: spectrogram render failed for conv %d", idx)
                return web.Response(status=500, text="spectrogram render failed")
            if len(_spec_cache) >= _SPEC_CACHE_MAX:
                _spec_cache.clear()
            _spec_cache[key] = png
        return web.Response(
            body=png,
            content_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    app.router.add_get(r"/api/vap/{idx:\d+}", handle_vap)
    app.router.add_get(r"/api/vap/spectrogram/{idx:\d+}.png", handle_spectrogram)


def render_panel_html() -> str:
    """Collapsible figure container — hidden until loadVap finds VAP data."""
    return """
<div id="vap-panel" style="display:none">
  <div class="vap-head" onclick="window.vapToggle && window.vapToggle()">
    <span class="vap-title">VAP turn-taking</span>
    <span id="vap-meta" class="vap-meta"></span>
    <span style="margin-left:auto; color:var(--muted); font-size:11px;">P(now) top · P(future) bottom — blue=tutor, red=student.</span>
    <span id="vap-caret" class="vap-caret">&#9662;</span>
  </div>
  <div id="vap-body">
    <div class="vap-controls">
      <button type="button" class="vap-btn" data-vap-zoom="out" title="Zoom out">&minus;</button>
      <button type="button" class="vap-btn" data-vap-zoom="in" title="Zoom in">&plus;</button>
      <button type="button" class="vap-btn" data-vap-zoom="reset" title="Fit whole conversation">Reset</button>
      <span id="vap-range" class="vap-range"></span>
      <span class="vap-hint">scroll to zoom · drag to pan · click to seek</span>
    </div>
    <div id="vap-figure" class="vap-figure">
      <img id="vap-spec" class="vap-spec" alt="mel-spectrogram" draggable="false"/>
      <canvas id="vap-canvas" class="vap-canvas"></canvas>
      <div id="vap-cursor" class="vap-cursor"></div>
    </div>
  </div>
</div>
<style>
  #vap-panel { margin:8px 12px; border:1px solid var(--border, #2a2a2a); border-radius:8px; overflow:hidden; }
  .vap-head { display:flex; align-items:center; gap:10px; padding:6px 12px; cursor:pointer;
              background:rgba(127,127,127,0.08); user-select:none; font-size:13px; }
  .vap-title { font-weight:600; }
  .vap-meta { color:var(--muted, #999); font-size:11px; }
  .vap-caret { transition:transform 120ms linear; }
  #vap-panel.collapsed .vap-caret { transform:rotate(-90deg); }
  #vap-panel.collapsed #vap-body { display:none; }
  .vap-figure { position:relative; width:100%; overflow:hidden; touch-action:none;
                cursor:ew-resize; user-select:none; }
  .vap-spec { display:block; width:100%; height:120px; image-rendering:auto;
              transform-origin:left top; will-change:transform,width;
              -webkit-user-drag:none; user-select:none; }
  .vap-canvas { display:block; width:100%; }
  .vap-cursor { position:absolute; top:0; bottom:0; width:1px; background:#ffd24a;
                box-shadow:0 0 3px #ffd24a; pointer-events:none; display:none; }
  .vap-controls { display:flex; align-items:center; gap:6px; padding:6px 12px 0; font-size:11px; }
  .vap-btn { font:600 12px system-ui, sans-serif; line-height:1; padding:3px 9px; cursor:pointer;
             color:var(--text, #222); background:rgba(127,127,127,0.12);
             border:1px solid var(--border, #ccc); border-radius:4px; }
  .vap-btn:hover { background:rgba(127,127,127,0.22); }
  .vap-range { color:var(--muted, #999); font-variant-numeric:tabular-nums; min-width:118px; }
  .vap-hint { margin-left:auto; color:var(--muted, #999); }
</style>
"""


def render_panel_script() -> str:
    """Inline JS — exposes window.loadVap(idx) and window.vapUpdateCursor(t)."""
    return """
<script>
(function () {
  const TUTOR = '#3a7bd5', STUDENT = '#e0563a';
  const WORDS_H = 26, GAP = 6, RIBBON_H = 78;  // canvas lane geometry (CSS px)
  const MIN_SPAN = 0.5;   // tightest zoom: seconds of audio shown across full width
  let data = null;        // last fetched payload
  let dur = 0;            // conversation duration (s) — the full x-axis span
  let viewStart = 0;      // left edge of the visible window (s)
  let viewSpan = 0;       // width of the visible window (s); == dur when zoomed out
  let cursorT = -1;       // last playhead time (s); -1 = no cursor

  const $ = (id) => document.getElementById(id);
  const figW = () => { const f = $('vap-figure'); return (f && f.clientWidth) || 1; };

  function laneTops() {
    const wordsY = 0;
    const nowY = wordsY + WORDS_H + GAP;
    const futY = nowY + RIBBON_H + GAP;
    return { wordsY, nowY, futY, total: futY + RIBBON_H };
  }

  // Keep the view window inside [0, dur] with a sane minimum span.
  function clampView() {
    if (!(dur > 0)) { viewStart = 0; viewSpan = 0; return; }
    viewSpan = Math.min(dur, Math.max(Math.min(MIN_SPAN, dur), viewSpan));
    viewStart = Math.max(0, Math.min(viewStart, dur - viewSpan));
  }

  function fmtClock(t) {
    if (!(t >= 0)) t = 0;
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  // Stretch/offset the spectrogram <img> (whose extent is [0, dur]) so it lines
  // up with the visible window; the figure's overflow:hidden clips the rest.
  function applySpecTransform(W) {
    const spec = $('vap-spec');
    if (!spec || !(dur > 0) || !(viewSpan > 0)) return;
    const z = dur / viewSpan;                         // horizontal zoom factor (>= 1)
    spec.style.width = (z * 100) + '%';
    spec.style.transform = 'translateX(' + (-(viewStart / viewSpan) * W) + 'px)';
  }

  function updateRange() {
    const el = $('vap-range');
    if (!el) return;
    if (!(dur > 0)) { el.textContent = ''; return; }
    if (viewSpan >= dur - 1e-6) {
      el.textContent = 'full · ' + fmtClock(dur);
    } else {
      el.textContent = fmtClock(viewStart) + '–' + fmtClock(viewStart + viewSpan)
        + ' · ' + viewSpan.toFixed(1) + 's';
    }
  }

  // Re-zoom to `newSpan` seconds, pinning audio time `anchorT` at screen
  // fraction `frac` so zooming stays anchored where the user pointed.
  function zoomTo(newSpan, anchorT, frac) {
    if (!(dur > 0)) return;
    viewSpan = Math.min(dur, Math.max(Math.min(MIN_SPAN, dur), newSpan));
    viewStart = anchorT - frac * viewSpan;
    clampView();
    draw();
  }

  function draw() {
    const fig = $('vap-figure');
    const canvas = $('vap-canvas');
    if (!fig || !canvas || !data) return;
    const W = fig.clientWidth || 1;
    const L = laneTops();
    const H = L.total;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!(dur > 0) || !(viewSpan > 0)) return;
    applySpecTransform(W);
    updateRange();
    const xOf = (t) => ((t - viewStart) / viewSpan) * W;

    // --- Word lane: role-colored boxes with labels -----------------------
    const words = data.words || [];
    ctx.font = '10px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (const wd of words) {
      const x0 = xOf(wd.s), x1 = xOf(wd.e);
      const w = Math.max(1, x1 - x0);
      const col = wd.role === 'tutor' ? TUTOR : STUDENT;
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = col;
      ctx.fillRect(x0, L.wordsY, w, WORDS_H);
      ctx.globalAlpha = 1;
      if (w > 16) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + 1, L.wordsY, w - 2, WORDS_H);
        ctx.clip();
        // Dark text — the host page is a light theme, so light labels vanished
        // into the near-white word lane.
        ctx.fillStyle = '#1a1a1a';
        ctx.fillText(wd.w, x0 + 2, L.wordsY + WORDS_H / 2);
        ctx.restore();
      }
    }

    // --- Probability ribbons: two-tone fill split by P(tutor) ------------
    const ts = data.t || [];
    function ribbon(probTutor, y0) {
      if (!probTutor || !probTutor.length) return;
      // Whole lane = student (red); the tutor (blue) fills the top down to a
      // boundary at height p_tutor, so a fully-blue lane == P(tutor) = 1.
      ctx.fillStyle = STUDENT;
      ctx.globalAlpha = 0.82;
      ctx.fillRect(0, y0, W, RIBBON_H);
      ctx.beginPath();
      ctx.moveTo(xOf(ts[0]), y0);
      ctx.lineTo(xOf(ts[ts.length - 1]), y0);
      for (let i = probTutor.length - 1; i >= 0; i--) {
        const yb = y0 + probTutor[i] * RIBBON_H;
        ctx.lineTo(xOf(ts[i]), yb);
      }
      ctx.closePath();
      ctx.fillStyle = TUTOR;
      ctx.fill();
      ctx.globalAlpha = 1;
      // 0.5 reference line
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y0 + RIBBON_H / 2);
      ctx.lineTo(W, y0 + RIBBON_H / 2);
      ctx.stroke();
    }
    ribbon(data.p_now ? data.p_now[0] : null, L.nowY);
    ribbon(data.p_future ? data.p_future[0] : null, L.futY);

    // lane labels
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = '#e8e8e8';
    ctx.font = '600 10px system-ui, sans-serif';
    ctx.fillText('P(now)', 4, L.nowY + 8);
    ctx.fillText('P(future)', 4, L.futY + 8);
    ctx.globalAlpha = 1;

    // --- Turn-taking event markers ---------------------------------------
    for (const ev of (data.events || [])) {
      const x = xOf(ev.t);
      const good = (ev.cell || '').startsWith('appropriate');
      ctx.strokeStyle = good ? 'rgba(120,220,140,0.7)' : 'rgba(255,170,60,0.85)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, L.nowY);
      ctx.lineTo(x, L.futY + RIBBON_H);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    placeCursor();  // keep the playhead pinned to its time as the view changes
  }

  // Position the cursor for the remembered playhead time under the current
  // view; called on every redraw so panning/zooming keeps it anchored in time.
  function placeCursor() {
    const cur = $('vap-cursor');
    if (!cur || !data || !(dur > 0) || !(viewSpan > 0)) return;
    if (cursorT < 0) { cur.style.display = 'none'; return; }
    const W = figW();
    const x = ((cursorT - viewStart) / viewSpan) * W;
    if (x < 0 || x > W) { cur.style.display = 'none'; return; }
    cur.style.left = x + 'px';
    cur.style.display = 'block';
  }

  window.vapUpdateCursor = function (t) {
    cursorT = t;
    if (!data || !(dur > 0) || !(viewSpan > 0)) return;
    // When zoomed in, page the window to follow the playhead so it stays visible.
    if (t >= 0 && viewSpan < dur - 1e-6 && (t < viewStart || t > viewStart + viewSpan)) {
      viewStart = t - viewSpan / 2;
      clampView();
      draw();  // draw() calls placeCursor()
      return;
    }
    placeCursor();
  };

  window.vapToggle = function () {
    const p = $('vap-panel');
    if (p) p.classList.toggle('collapsed');
  };

  window.loadVap = async function (idx) {
    const panel = $('vap-panel');
    if (panel) panel.style.display = 'none';
    data = null;
    viewStart = 0; viewSpan = 0; cursorT = -1;
    if ($('vap-cursor')) $('vap-cursor').style.display = 'none';

    let payload;
    try {
      const resp = await fetch('/api/vap/' + idx);
      if (!resp.ok) return;
      payload = await resp.json();
    } catch (e) { return; }
    if (!payload || !payload.available) return;

    data = payload;
    dur = (window.selectedConversation && window.selectedConversation.duration)
      || payload.duration || 0;
    viewStart = 0; viewSpan = dur;   // start fully zoomed out
    $('vap-spec').src = '/api/vap/spectrogram/' + idx + '.png';
    $('vap-meta').textContent =
      `${(payload.events || []).length} events · ${payload.frame_hz} Hz · ${payload.aligned_variant}`;
    if (panel) { panel.style.display = ''; panel.classList.remove('collapsed'); }
    // Defer one frame so clientWidth is valid after display flips to visible.
    requestAnimationFrame(draw);
  };

  const figEl = $('vap-figure');
  if (figEl) {
    // Scroll wheel: zoom, anchored at the pointer's time position.
    figEl.addEventListener('wheel', (e) => {
      if (!data || !(dur > 0)) return;
      e.preventDefault();
      const rect = figEl.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const anchorT = viewStart + frac * viewSpan;
      const factor = e.deltaY < 0 ? (1 / 1.25) : 1.25;  // up = zoom in
      zoomTo(viewSpan * factor, anchorT, frac);
    }, { passive: false });

    // Drag to pan; a press that doesn't move is a click-to-seek.
    let panning = false, panStartX = 0, panStartView = 0, panMoved = false;
    figEl.addEventListener('pointerdown', (e) => {
      if (!data || !(dur > 0)) return;
      panning = true; panMoved = false;
      panStartX = e.clientX; panStartView = viewStart;
      try { figEl.setPointerCapture(e.pointerId); } catch (_) {}
    });
    figEl.addEventListener('pointermove', (e) => {
      if (!panning) return;
      const dx = e.clientX - panStartX;
      if (Math.abs(dx) > 3) panMoved = true;
      viewStart = panStartView - (dx / figW()) * viewSpan;
      clampView();
      draw();
    });
    const endPan = (e) => {
      if (!panning) return;
      panning = false;
      try { figEl.releasePointerCapture(e.pointerId); } catch (_) {}
      if (!panMoved && dur > 0) {
        const rect = figEl.getBoundingClientRect();
        const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const t = viewStart + frac * viewSpan;
        if (typeof window.seekToTime === 'function') window.seekToTime(t);
      }
    };
    figEl.addEventListener('pointerup', endPan);
    figEl.addEventListener('pointercancel', endPan);
  }

  // Zoom buttons (centered on the current view).
  document.querySelectorAll('[data-vap-zoom]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!data || !(dur > 0)) return;
      const kind = btn.getAttribute('data-vap-zoom');
      if (kind === 'reset') { viewStart = 0; viewSpan = dur; clampView(); draw(); return; }
      const factor = kind === 'in' ? (1 / 1.5) : 1.5;
      zoomTo(viewSpan * factor, viewStart + viewSpan / 2, 0.5);
    });
  });

  let resizeRaf = null;
  window.addEventListener('resize', () => {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(draw);
  });
})();
</script>
"""
