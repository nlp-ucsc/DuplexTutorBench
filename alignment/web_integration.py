"""Hooks for the duplex web UI to expose aligned-vs-original transcript.

Each `(mode, no_bias)` variant is stored in its own
`<run_dir>/aligned/<variant>.jsonl` file. The UI surfaces a `<select>` so
the user can switch between variants (or `(original)`) without rerunning.

Wired identically to `evaluation.web_integration`:

    from alignment.web_integration import (
        register_routes, render_toggle_html, render_toggle_script,
    )
    register_routes(app, get_run_dir=lambda: self.run_dir)
    html = HTML_TEMPLATE.replace("<!--ALIGN_TOGGLE-->", render_toggle_html())
    html = html.replace("<!--ALIGN_SCRIPT-->", render_toggle_script())
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from aiohttp import web

from alignment.schemas import (
    aligned_by_index,
    aligned_dir_for,
    aligned_path_for,
    list_aligned_variants,
)

logger = logging.getLogger(__name__)


# Preference order when picking a default variant for the UI dropdown.
_DEFAULT_PRIORITY = ("whisper_mms", "whisper_mfa", "text_mms")


def _variant_metadata(path: Path) -> dict:
    """Cheaply peek at the first record to surface mode/bias for the UI."""
    n = 0
    mode: str | None = None
    prompt_bias: bool | None = None
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if mode is None:
                    try:
                        rec = json.loads(line)
                        align = rec.get("alignment", {}) or {}
                        mode = align.get("mode")
                        prompt_bias = align.get("prompt_bias")
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return {"mode": mode, "prompt_bias": prompt_bias, "n_records": n}


def _default_variant(variants: list[str]) -> str | None:
    for name in _DEFAULT_PRIORITY:
        if name in variants:
            return name
    return variants[0] if variants else None


def register_routes(
    app: web.Application,
    get_run_dir: Callable[[], Path | None],
) -> None:
    async def handle_variants(request: web.Request) -> web.Response:
        run_dir = get_run_dir()
        if run_dir is None:
            return web.json_response({"variants": [], "default": None})
        variants = list_aligned_variants(run_dir)
        records: list[dict] = []
        for key in variants:
            meta = _variant_metadata(aligned_path_for(run_dir, key))
            records.append({"key": key, **meta})
        return web.json_response(
            {"variants": records, "default": _default_variant(variants)}
        )

    async def handle_one(request: web.Request) -> web.Response:
        run_dir = get_run_dir()
        if run_dir is None:
            return web.json_response({"error": "no run selected"}, status=400)
        variant = request.match_info["variant"]
        # Reject path traversal / unknown variants quickly.
        path = aligned_path_for(run_dir, variant)
        if not path.is_file() or path.parent != aligned_dir_for(run_dir):
            return web.json_response(
                {"error": "unknown variant", "variant": variant}, status=404
            )
        idx = int(request.match_info["idx"])
        index = aligned_by_index(path)
        rec = index.get(idx)
        if rec is None:
            return web.json_response(
                {"error": "no aligned record", "conv_index": idx, "variant": variant},
                status=404,
            )
        return web.json_response(rec)

    app.router.add_get("/api/aligned/variants", handle_variants)
    app.router.add_get(r"/api/aligned/{variant}/{idx:\d+}", handle_one)


def render_toggle_html() -> str:
    """Variant picker that lives next to the segments tab. Hidden until loaded."""
    return """
<span id="align-toggle" style="display:none; margin-left:auto; align-items:center; gap:8px; font-size:12px; color:var(--muted); padding:0 12px;">
  <span id="align-toggle-meta" style="color:#999;"></span>
  <label style="cursor:pointer;">
    View:
    <select id="align-toggle-sel" style="margin-left:4px; font-size:12px;">
      <option value="__original__">(original)</option>
    </select>
  </label>
</span>
<style>
  #align-toggle.visible { display:inline-flex !important; }
  .align-word { padding:0 1px; border-radius:2px; cursor:pointer; transition: background-color 60ms linear; }
  .align-word:hover { background:#fff59d; }
  .align-word.low-conf { color:#888; font-style:italic; }
  .align-word.now { background:#ffc107; color:#1a1a1a; font-weight:600; }
</style>
"""


def render_toggle_script() -> str:
    """Inline JS — exposes window.loadAligned(idx) called from selectConversation."""
    return """
<script>
(function () {
  const ORIGINAL = '__original__';
  let variantsMeta = null;        // [{key, mode, prompt_bias, n_records}]
  let defaultVariant = null;      // server-suggested default key
  let loadedRecords = {};         // variantKey -> aligned record for current idx
  let originalSegments = null;    // saved copy of the original segments[]
  let currentIdx = null;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function renderSegments(segs, withWords) {
    const container = document.getElementById('transcript-segments');
    container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.className = 'segment-wrapper';
    segs.forEach((seg, i) => {
      const div = document.createElement('div');
      div.className = `segment ${seg.role}`;
      div.id = `seg-${i}`;
      div.dataset.start = seg.start_time;
      div.dataset.end = seg.end_time;
      let body = '';
      if (withWords && Array.isArray(seg.words) && seg.words.length) {
        body = seg.words.map(w => {
          const conf = (typeof w.confidence === 'number') ? w.confidence : 0;
          const lowConf = conf < -2 ? ' low-conf' : '';
          const t = `${w.start_time.toFixed(2)}–${w.end_time.toFixed(2)}s`;
          return `<span class="align-word${lowConf}" title="${t} (conf ${conf.toFixed(2)})" data-t="${w.start_time}" data-t-end="${w.end_time}">${escapeHtml(w.word)}</span>`;
        }).join(' ');
      } else {
        body = escapeHtml(seg.text);
      }
      div.innerHTML = `<div class="seg-header">${seg.role} [${seg.start_time.toFixed(1)}s – ${seg.end_time.toFixed(1)}s]</div>${body}`;
      div.style.cursor = 'pointer';
      div.onclick = (ev) => {
        const w = ev.target.closest('.align-word');
        const t = w ? parseFloat(w.dataset.t) : seg.start_time;
        if (typeof window.seekToTime === 'function') window.seekToTime(t);
      };
      wrapper.appendChild(div);
    });
    container.appendChild(wrapper);
  }

  function variantLabel(meta) {
    const bias = (meta.prompt_bias === false) ? ' (no bias)' :
                 (meta.prompt_bias === true) ? ' (biased)' : '';
    return `${meta.key}${bias}`;
  }

  function populateSelect() {
    const sel = document.getElementById('align-toggle-sel');
    if (!sel) return;
    sel.innerHTML = '<option value="' + ORIGINAL + '">(original)</option>';
    (variantsMeta || []).forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.key;
      opt.textContent = variantLabel(v);
      sel.appendChild(opt);
    });
  }

  function metaForKey(key) {
    return (variantsMeta || []).find(v => v.key === key) || null;
  }

  async function applySelection() {
    const sel = document.getElementById('align-toggle-sel');
    const meta = document.getElementById('align-toggle-meta');
    const key = sel.value;

    if (key === ORIGINAL) {
      meta.textContent = '';
      if (originalSegments) {
        renderSegments(originalSegments, /*withWords=*/false);
        window.selectedConversation && (window.selectedConversation.segments = originalSegments);
      }
      return;
    }

    let rec = loadedRecords[key];
    if (rec === undefined) {
      try {
        const resp = await fetch('/api/aligned/' + encodeURIComponent(key) + '/' + currentIdx);
        if (!resp.ok) {
          meta.textContent = `${key}: HTTP ${resp.status}`;
          return;
        }
        rec = await resp.json();
        loadedRecords[key] = rec;
      } catch (e) {
        meta.textContent = `${key}: error`;
        return;
      }
    }

    const m = metaForKey(key);
    meta.textContent = m ? variantLabel(m) : key;
    if (rec && Array.isArray(rec.segments) && rec.segments.length) {
      renderSegments(rec.segments, /*withWords=*/true);
      window.selectedConversation && (window.selectedConversation.segments = rec.segments);
    } else if (originalSegments) {
      renderSegments(originalSegments, /*withWords=*/false);
    }
  }

  let lastActiveWords = new Set();
  window.highlightAlignedWords = function (currentTime) {
    const activeSegs = document.querySelectorAll('#transcript-segments .segment.highlight');
    const newActive = new Set();
    activeSegs.forEach(seg => {
      const spans = seg.getElementsByClassName('align-word');
      for (let i = 0; i < spans.length; i++) {
        const sp = spans[i];
        const ws = parseFloat(sp.dataset.t);
        const we = parseFloat(sp.dataset.tEnd);
        if (currentTime >= ws && currentTime <= we) { newActive.add(sp); break; }
      }
    });
    lastActiveWords.forEach(sp => { if (!newActive.has(sp)) sp.classList.remove('now'); });
    newActive.forEach(sp => { if (!lastActiveWords.has(sp)) sp.classList.add('now'); });
    lastActiveWords = newActive;
  };

  async function fetchVariantsOnce() {
    if (variantsMeta !== null) return;
    try {
      const resp = await fetch('/api/aligned/variants');
      if (!resp.ok) { variantsMeta = []; return; }
      const data = await resp.json();
      variantsMeta = Array.isArray(data.variants) ? data.variants : [];
      defaultVariant = data.default || null;
    } catch (e) {
      variantsMeta = [];
    }
  }

  window.loadAligned = async function (idx) {
    currentIdx = idx;
    loadedRecords = {};   // per-conv cache; reset on new conversation
    originalSegments = (window.selectedConversation && window.selectedConversation.segments)
      ? window.selectedConversation.segments.map(s => Object.assign({}, s))
      : null;

    const toggle = document.getElementById('align-toggle');
    const sel = document.getElementById('align-toggle-sel');
    const meta = document.getElementById('align-toggle-meta');
    if (!toggle || !sel) return;

    await fetchVariantsOnce();
    if (!variantsMeta || !variantsMeta.length) {
      toggle.classList.remove('visible');
      return;
    }
    populateSelect();
    sel.onchange = applySelection;
    toggle.classList.add('visible');

    // Default to the server-suggested variant if available for this conv;
    // otherwise stay on (original).
    if (defaultVariant) {
      sel.value = defaultVariant;
      await applySelection();
    } else {
      sel.value = ORIGINAL;
      meta.textContent = '';
    }
  };
})();
</script>
"""
