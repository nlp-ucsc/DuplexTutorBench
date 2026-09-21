"""Hooks for embedding evaluation views into the duplex web UI.

Usage from `duplex/web.py`:

    from evaluation.web_integration import (
        register_routes, render_panel_html, render_panel_script,
    )

    # in create_app():
    register_routes(app, get_run_dir=lambda: self.run_dir)

    # in _handle_index() before returning:
    html = HTML_TEMPLATE.replace("<!--EVAL_PANEL-->", render_panel_html())
    html = html.replace("<!--EVAL_SCRIPT-->", render_panel_script())

The injected JS exposes `window.loadEvalScores(idx)` which the existing
`selectConversation(idx)` should call after rendering its own panes.

`get_run_dir` is read on every request so the active run can change at
runtime (e.g. when the user picks a different run from the dropdown).
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Callable

from aiohttp import web

logger = logging.getLogger(__name__)


def _eval_root_for(run_dir: Path, eval_root: Path | None) -> Path:
    """Map a duplex run dir to the parallel eval_output dir."""
    if eval_root is not None:
        return eval_root / run_dir.name
    return Path("eval_output") / run_dir.name


def _load_scores(scores_path: Path) -> list[dict[str, Any]]:
    if not scores_path.is_file():
        return []
    out = []
    with scores_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-evaluator metrics across all conversations.

    Numeric leaves (int/float) are averaged. Non-numeric values (status
    strings, error blobs, rubric rationales) are dropped from the summary.
    """

    def _walk(d: Any, path: list[str], buckets: dict[str, list[float]]) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                _walk(v, path + [k], buckets)
        elif isinstance(d, (int, float)) and not isinstance(d, bool):
            key = ".".join(path)
            buckets.setdefault(key, []).append(float(d))

    by_evaluator: dict[str, dict[str, list[float]]] = {}
    for rec in scores:
        for ev_name, ev_result in rec.get("evaluator_results", {}).items():
            buckets = by_evaluator.setdefault(ev_name, {})
            _walk(ev_result, [], buckets)

    summary: dict[str, Any] = {"n_conversations": len(scores), "evaluators": {}}
    for ev, buckets in by_evaluator.items():
        ev_out = {}
        for key, vals in buckets.items():
            if not vals:
                continue
            ev_out[key] = {
                "mean": round(statistics.fmean(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "n": len(vals),
            }
        summary["evaluators"][ev] = ev_out
    return summary


def register_routes(
    app: web.Application,
    get_run_dir: Callable[[], Path | None],
    eval_root: Path | None = None,
) -> None:
    """Add /api/eval/* and /eval-summary routes to the duplex web app.

    `get_run_dir` is invoked per request so a runtime run switch picks up
    the new eval_output/<run>/ dir without re-registering routes.
    """

    def _scores_path() -> Path | None:
        run_dir = get_run_dir()
        if run_dir is None:
            return None
        return _eval_root_for(run_dir, eval_root) / "scores.jsonl"

    async def handle_scores_all(request: web.Request) -> web.Response:
        p = _scores_path()
        return web.json_response(_load_scores(p) if p else [])

    async def handle_scores_one(request: web.Request) -> web.Response:
        idx = int(request.match_info["idx"])
        p = _scores_path()
        if p:
            for rec in _load_scores(p):
                if int(rec.get("conv_index", -1)) == idx:
                    return web.json_response(rec)
        return web.json_response({"error": "not found", "conv_index": idx}, status=404)

    async def handle_summary(request: web.Request) -> web.Response:
        p = _scores_path()
        return web.json_response(_aggregate(_load_scores(p) if p else []))

    async def handle_summary_page(request: web.Request) -> web.Response:
        run_dir = get_run_dir()
        if run_dir is None:
            return web.Response(
                text="No run selected.", status=400, content_type="text/plain"
            )
        scores = _load_scores(_eval_root_for(run_dir, eval_root) / "scores.jsonl")
        agg = _aggregate(scores)
        html = _summary_page_html(run_dir.name, scores, agg)
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/api/eval/scores", handle_scores_all)
    app.router.add_get(r"/api/eval/scores/{idx:\d+}", handle_scores_one)
    app.router.add_get("/api/eval/summary", handle_summary)
    app.router.add_get("/eval-summary", handle_summary_page)


# ---------------------------------------------------------------------------
# HTML/JS injection for the per-conversation scores panel.
# ---------------------------------------------------------------------------


def render_panel_html() -> str:
    """Static HTML inserted into the duplex history left-panel.

    Includes a small "Run summary →" link to the per-run summary page.
    """
    return """
<div id="eval-panel" style="margin-top:12px; padding:10px 12px; border:1px solid var(--border); border-radius:8px; background:#fafafa; font-size:13px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
    <strong style="color:var(--muted); text-transform:uppercase; font-size:12px;">Evaluation</strong>
    <a href="/eval-summary" target="_blank" style="font-size:12px; color:var(--accent); text-decoration:none;">Run summary &rarr;</a>
  </div>
  <div id="eval-panel-body" style="color:var(--muted);">Select a conversation to view scores.</div>
</div>
"""


def render_panel_script() -> str:
    """Inline <script> block defining `window.loadEvalScores(idx)`."""
    return """
<script>
(function () {
  function fmt(v) {
    if (v === null || v === undefined) return '<span style="color:#999">—</span>';
    if (typeof v === 'number') return v.toFixed(3).replace(/\\.?0+$/, '');
    return String(v);
  }

  function row(label, val) {
    return '<div style="display:flex; justify-content:space-between; padding:2px 0;">'
      + '<span style="color:#666">' + label + '</span><span>' + fmt(val) + '</span></div>';
  }

  function renderStats(s) {
    if (!s) return '';
    return '<div style="margin-top:6px;"><div style="font-weight:600; margin-bottom:2px;">Conversation stats</div>'
      + row('Duration (s)', s.duration_s)
      + row('Segments', s.num_segments)
      + row('Tutor talk ratio', s.tutor_talk_ratio)
      + row('Student talk ratio', s.student_talk_ratio)
      + row('Tutor wpm', s.tutor_words_per_min)
      + row('Student wpm', s.student_words_per_min)
      + '</div>';
  }

  function renderTurnTaking(t) {
    if (!t) return '';
    var lat = t.response_latency_s || {};
    return '<div style="margin-top:6px;"><div style="font-weight:600; margin-bottom:2px;">Turn-taking</div>'
      + row('Latency mean (s)', lat.mean)
      + row('Latency median (s)', lat.median)
      + row('Latency p90 (s)', lat.p90)
      + row('Overlap (s)', t.overlap_s)
      + row('Silence ratio', t.silence_ratio)
      + row('Backchannels', t.backchannel_count)
      + '</div>';
  }

  function renderNaturalness(n) {
    if (!n) return '';
    if (n.status === 'skipped') {
      return '<div style="margin-top:6px;"><div style="font-weight:600;">Naturalness (UTMOS)</div>'
        + '<div style="color:#999; font-size:12px;">' + (n.reason || 'skipped') + '</div></div>';
    }
    return '<div style="margin-top:6px;"><div style="font-weight:600; margin-bottom:2px;">Naturalness (UTMOS)</div>'
      + row('Tutor', n.tutor_utmos)
      + row('Student', n.student_utmos)
      + '</div>';
  }

  function renderJudge(j) {
    if (!j) return '';
    if (j.status !== 'ok') {
      return '<div style="margin-top:6px;"><div style="font-weight:600;">LLM judge</div>'
        + '<div style="color:#c33; font-size:12px;">' + (j.error || j.status) + '</div></div>';
    }
    var html = '<div style="margin-top:6px;"><div style="font-weight:600; margin-bottom:2px;">LLM judge'
      + ' <span style="font-weight:400; color:#999; font-size:11px;">(' + (j.model || '') + ')</span></div>';
    var keys = ['answer_correctness', 'scaffolding_quality', 'student_realism', 'phrasing_naturalness', 'overall'];
    var labels = {
      answer_correctness: 'Correctness',
      scaffolding_quality: 'Scaffolding',
      student_realism: 'Student realism',
      phrasing_naturalness: 'Naturalness',
      overall: 'Overall'
    };
    keys.forEach(function (k) {
      var v = (j.scores || {})[k];
      if (!v) return;
      var line = '<div style="display:flex; justify-content:space-between; padding:2px 0;" title="'
        + (v.rationale || '').replace(/"/g, '&quot;') + '">'
        + '<span style="color:#666">' + labels[k] + '</span>'
        + '<span><strong>' + v.score + '</strong> / 5</span></div>';
      html += line;
    });
    html += '</div>';
    return html;
  }

  window.loadEvalScores = async function (idx) {
    var body = document.getElementById('eval-panel-body');
    if (!body) return;
    body.innerHTML = '<span style="color:#999">Loading...</span>';
    try {
      var resp = await fetch('/api/eval/scores/' + idx);
      if (resp.status === 404) {
        body.innerHTML = '<span style="color:#999">No scores yet. Run <code>uv run python -m evaluation run --run-name &lt;run&gt;</code>.</span>';
        return;
      }
      if (!resp.ok) {
        body.innerHTML = '<span style="color:#c33">Failed to load scores (HTTP ' + resp.status + ').</span>';
        return;
      }
      var rec = await resp.json();
      var er = rec.evaluator_results || {};
      var html = renderStats(er.stats)
        + renderTurnTaking(er.turn_taking)
        + renderNaturalness(er.naturalness)
        + renderJudge(er.llm_judge);
      body.innerHTML = html || '<span style="color:#999">No metrics in record.</span>';
    } catch (e) {
      body.innerHTML = '<span style="color:#c33">Error: ' + e + '</span>';
    }
  };
})();
</script>
"""


# ---------------------------------------------------------------------------
# Run-level summary page (standalone HTML)
# ---------------------------------------------------------------------------


def _summary_page_html(
    run_name: str, scores: list[dict[str, Any]], aggregate: dict[str, Any]
) -> str:
    """Render the standalone /eval-summary page."""
    rows_json = json.dumps(scores)
    agg_json = json.dumps(aggregate)
    # Page is fully self-contained — no shared CSS dependency on duplex/web.py
    # so we don't have to thread template context through.
    return (
        _SUMMARY_TEMPLATE.replace("__RUN_NAME__", run_name)
        .replace("__SCORES_JSON__", rows_json)
        .replace("__AGG_JSON__", agg_json)
    )


_SUMMARY_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Evaluation Summary &mdash; __RUN_NAME__</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f5f5; color: #222; }
  header { background: #fff; padding: 16px 24px; border-bottom: 1px solid #e0e0e0; }
  h1 { margin: 0; font-size: 18px; }
  h1 small { color: #999; font-weight: 400; margin-left: 8px; font-size: 13px; }
  main { padding: 16px 24px; max-width: 1200px; margin: 0 auto; }
  section { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  section h2 { margin: 0 0 12px; font-size: 14px; text-transform: uppercase; color: #555; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #f0f0f0; }
  th { font-weight: 600; color: #555; cursor: pointer; user-select: none; }
  th.sort-asc::after { content: " ▲"; color: #999; }
  th.sort-desc::after { content: " ▼"; color: #999; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .muted { color: #999; }
  .pid-cell { font-family: ui-monospace, monospace; font-size: 12px; color: #555; }
  .agg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .agg-block h3 { margin: 0 0 8px; font-size: 13px; color: #444; }
  .agg-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; }
  .agg-row .k { color: #666; }
</style>
</head>
<body>
<header>
  <h1>Evaluation Summary <small>__RUN_NAME__</small></h1>
</header>
<main>
  <section>
    <h2>Aggregate</h2>
    <div id="agg"></div>
  </section>
  <section>
    <h2>Per-conversation</h2>
    <div id="table-wrapper">Loading...</div>
  </section>
</main>
<script>
const SCORES = __SCORES_JSON__;
const AGG = __AGG_JSON__;

function fmt(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return Number(v.toFixed(3)).toString();
  return String(v);
}

function renderAgg() {
  const root = document.getElementById('agg');
  const ev = AGG.evaluators || {};
  if (!Object.keys(ev).length) {
    root.innerHTML = '<div class="muted">No scores yet for this run.</div>';
    return;
  }
  let html = '<div class="muted" style="margin-bottom:8px;">' + AGG.n_conversations + ' conversations</div>';
  html += '<div class="agg-grid">';
  for (const [name, metrics] of Object.entries(ev)) {
    html += '<div class="agg-block"><h3>' + name + '</h3>';
    for (const [k, s] of Object.entries(metrics)) {
      html += '<div class="agg-row"><span class="k">' + k + '</span>'
            + '<span>mean ' + fmt(s.mean) + ' &middot; min ' + fmt(s.min) + ' &middot; max ' + fmt(s.max) + '</span></div>';
    }
    html += '</div>';
  }
  html += '</div>';
  root.innerHTML = html;
}

function flatten(obj, prefix) {
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? prefix + '.' + k : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(out, flatten(v, key));
    } else if (typeof v === 'number' || typeof v === 'string') {
      out[key] = v;
    }
  }
  return out;
}

function pickColumns() {
  // Prefer a small curated set; otherwise show whatever evaluators emitted.
  const preferred = [
    'stats.duration_s', 'stats.num_segments',
    'turn_taking.response_latency_s.mean', 'turn_taking.overlap_s', 'turn_taking.silence_ratio',
    'llm_judge.scores.overall.score', 'llm_judge.mean_component_score',
    'naturalness.tutor_utmos', 'naturalness.student_utmos'
  ];
  const seen = new Set();
  const cols = [];
  for (const r of SCORES) {
    const flat = flatten(r.evaluator_results);
    for (const k of Object.keys(flat)) seen.add(k);
  }
  for (const c of preferred) if (seen.has(c)) cols.push(c);
  if (!cols.length) {
    // Fallback: any 3 columns
    [...seen].slice(0, 5).forEach(c => cols.push(c));
  }
  return cols;
}

let sortState = { col: 'conv_index', dir: 'asc' };

function renderTable() {
  const cols = pickColumns();
  const headers = ['conv_index', 'pid'].concat(cols);
  const rows = SCORES.map(r => {
    const flat = flatten(r.evaluator_results);
    return Object.assign({ conv_index: r.conv_index, pid: r.pid }, flat);
  });

  rows.sort((a, b) => {
    const av = a[sortState.col], bv = b[sortState.col];
    if (av === undefined && bv === undefined) return 0;
    if (av === undefined) return 1;
    if (bv === undefined) return -1;
    if (typeof av === 'number' && typeof bv === 'number') {
      return sortState.dir === 'asc' ? av - bv : bv - av;
    }
    return sortState.dir === 'asc'
      ? String(av).localeCompare(String(bv))
      : String(bv).localeCompare(String(av));
  });

  let html = '<table><thead><tr>';
  for (const h of headers) {
    const cls = h === sortState.col ? ('sort-' + sortState.dir) : '';
    html += '<th class="' + cls + '" data-col="' + h + '">' + h + '</th>';
  }
  html += '</tr></thead><tbody>';
  for (const r of rows) {
    html += '<tr>';
    for (const h of headers) {
      const v = r[h];
      const cls = (typeof v === 'number') ? 'num' : (h === 'pid' ? 'pid-cell' : '');
      html += '<td class="' + cls + '">' + fmt(v) + '</td>';
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('table-wrapper').innerHTML = html;

  document.querySelectorAll('th').forEach(th => {
    th.onclick = () => {
      const col = th.dataset.col;
      if (sortState.col === col) {
        sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
      } else {
        sortState.col = col;
        sortState.dir = 'asc';
      }
      renderTable();
    };
  });
}

renderAgg();
renderTable();
</script>
</body>
</html>
"""
