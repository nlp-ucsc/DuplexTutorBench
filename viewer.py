"""Web UI for browsing generated conversation datasets."""

import argparse
import json
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)

app = Flask(__name__)
OUTPUT_DIR: Path = Path("output")


def discover_datasets() -> list[str]:
    """Return sorted list of dataset names (subdirs of OUTPUT_DIR with data)."""
    if not OUTPUT_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "conversations.jsonl").exists()
    )


def load_conversations(dataset: str) -> list[dict]:
    """Load conversations from a dataset directory.

    Handles both JSONL (one JSON object per line) and pretty-printed JSON
    (single object or array).
    """
    path = OUTPUT_DIR / dataset / "conversations.jsonl"
    if not path.exists():
        return []

    text = path.read_text()
    # Try parsing as a single JSON value first (object or array)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL: one JSON object per line
    convos = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            convos.append(json.loads(line))
    return convos


def detect_dataset_type(conversations: list[dict]) -> str:
    """Auto-detect dataset type from fields present."""
    if not conversations:
        return "empty"
    first = conversations[0]
    has_image = bool(first.get("image_path"))
    has_audio = any(u.get("audio_path") for u in first.get("utterances", []))
    if has_image and has_audio:
        return "text + image + audio"
    if has_image:
        return "text + image"
    if has_audio:
        return "text + audio"
    return "text only"


# -- Routes ------------------------------------------------------------------


@app.route("/")
def index() -> Response | str:
    datasets = discover_datasets()
    if not datasets:
        return "No datasets found in output/ directory", 404

    # Pick dataset from query param, default to first available
    dataset = request.args.get("dataset", datasets[0])
    if dataset not in datasets:
        return redirect(url_for("index", dataset=datasets[0]))

    conversations = load_conversations(dataset)
    dataset_type = detect_dataset_type(conversations)
    return render_template_string(
        TEMPLATE,
        conversations=conversations,
        dataset_type=dataset_type,
        datasets=datasets,
        current_dataset=dataset,
    )


@app.route("/api/conversations")
def api_conversations() -> Response:
    dataset = request.args.get("dataset", "")
    return jsonify(load_conversations(dataset))


@app.route("/api/conversation/<dataset>/<pid>")
def api_conversation(dataset: str, pid: str) -> tuple[Response, int] | Response:
    for c in load_conversations(dataset):
        if str(c.get("pid")) == pid:
            return jsonify(c)
    return jsonify({"error": "not found"}), 404


@app.route("/files/<dataset>/<path:filepath>")
def serve_file(dataset: str, filepath: str) -> Response:
    return send_from_directory(OUTPUT_DIR / dataset, filepath)


# -- Template ----------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Viewer</title>
<style>
  :root {
    --bg: #f5f5f5; --sidebar-bg: #fff; --chat-bg: #fff;
    --student: #e3f2fd; --tutor: #f3e5f5;
    --border: #ddd; --text: #222; --muted: #888;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); display: flex; height: 100vh; }

  /* Sidebar */
  .sidebar { width: 300px; min-width: 300px; background: var(--sidebar-bg);
             border-right: 1px solid var(--border); display: flex; flex-direction: column; }
  .sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
  .sidebar-header h2 { font-size: 16px; margin-bottom: 8px; }
  .dataset-select { width: 100%; padding: 6px 8px; border: 1px solid var(--border);
                    border-radius: 6px; font-size: 13px; margin-bottom: 8px;
                    background: var(--bg); cursor: pointer; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 600; background: #e8eaf6; color: #3949ab; }
  .sidebar-list { flex: 1; overflow-y: auto; }
  .sidebar-item { padding: 12px 16px; border-bottom: 1px solid var(--border);
                  cursor: pointer; transition: background .15s; }
  .sidebar-item:hover { background: #f0f0f0; }
  .sidebar-item.active { background: #e3f2fd; border-left: 3px solid #1976d2; }
  .sidebar-item .pid { font-weight: 600; font-size: 13px; }
  .sidebar-item .preview { font-size: 12px; color: var(--muted);
                           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Main */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .main-empty { display: flex; align-items: center; justify-content: center;
                flex: 1; color: var(--muted); font-size: 18px; }

  /* Metadata */
  .meta { padding: 16px 24px; border-bottom: 1px solid var(--border);
          background: var(--sidebar-bg); display: none; }
  .meta.visible { display: block; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 8px; font-size: 13px; }
  .meta-grid dt { font-weight: 600; color: var(--muted); }
  .meta-grid dd { margin-bottom: 4px; }

  /* Chat */
  .chat { flex: 1; overflow-y: auto; padding: 24px; display: none; }
  .chat.visible { display: block; }
  .question-image { max-width: 480px; border-radius: 8px; margin-bottom: 16px;
                    border: 1px solid var(--border); }
  .bubble { max-width: 70%; padding: 12px 16px; border-radius: 16px;
            margin-bottom: 12px; line-height: 1.5; font-size: 14px; position: relative; }
  .bubble.student { background: var(--student); align-self: flex-start; border-bottom-left-radius: 4px; }
  .bubble.tutor   { background: var(--tutor); align-self: flex-end; border-bottom-right-radius: 4px; }
  .bubble .role { font-size: 11px; font-weight: 700; text-transform: uppercase;
                  color: var(--muted); margin-bottom: 4px; }
  .bubble .turn-num { font-size: 10px; color: var(--muted); position: absolute;
                      top: 12px; right: 12px; }
  .bubble audio { margin-top: 8px; width: 100%; height: 32px; }
  .chat-wrapper { display: flex; flex-direction: column; }

  /* Question text */
  .question-text { background: #fffde7; padding: 12px 16px; border-radius: 8px;
                   margin-bottom: 16px; font-size: 13px; border: 1px solid #fff9c4; }
  .question-text strong { color: #f57f17; }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-header">
    <h2>Dataset Viewer</h2>
    <select class="dataset-select" id="dataset-select">
      {% for ds in datasets %}
      <option value="{{ ds }}" {{ "selected" if ds == current_dataset else "" }}>{{ ds }}</option>
      {% endfor %}
    </select>
    <span class="badge" id="type-badge">{{ dataset_type }}</span>
    <span class="badge" style="margin-left:4px" id="count-badge">{{ conversations|length }} conversations</span>
  </div>
  <div class="sidebar-list" id="sidebar-list"></div>
</div>

<div class="main">
  <div class="meta" id="meta">
    <div class="meta-grid" id="meta-grid"></div>
  </div>
  <div class="chat" id="chat"></div>
  <div class="main-empty" id="empty-msg">Select a conversation from the sidebar</div>
</div>

<script>
const CURRENT_DATASET = {{ current_dataset | tojson }};
let DATA = {{ conversations | tojson }};

const sidebar = document.getElementById("sidebar-list");
const metaEl = document.getElementById("meta");
const metaGrid = document.getElementById("meta-grid");
const chatEl = document.getElementById("chat");
const emptyMsg = document.getElementById("empty-msg");

// Dataset switcher — navigate via query param so the server reloads data
document.getElementById("dataset-select").addEventListener("change", function() {
  window.location.href = "/?dataset=" + encodeURIComponent(this.value);
});

// Build sidebar
function buildSidebar() {
  sidebar.innerHTML = "";
  DATA.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "sidebar-item";
    div.dataset.index = i;
    div.innerHTML = `<div class="pid">PID ${c.pid}</div>
      <div class="preview">${(c.question || "").substring(0, 80)}…</div>`;
    div.addEventListener("click", () => selectConversation(i));
    sidebar.appendChild(div);
  });
}

function selectConversation(idx) {
  // Update sidebar active state
  document.querySelectorAll(".sidebar-item").forEach(el => el.classList.remove("active"));
  document.querySelector(`.sidebar-item[data-index="${idx}"]`).classList.add("active");

  const c = DATA[idx];
  emptyMsg.style.display = "none";

  // Metadata
  metaEl.classList.add("visible");
  metaGrid.innerHTML = `
    <div><dt>PID</dt><dd>${c.pid}</dd></div>
    <div><dt>Answer</dt><dd>${c.answer}</dd></div>
    <div><dt>Tutor Model</dt><dd>${c.tutor_model || "—"}</dd></div>
    <div><dt>Student Model</dt><dd>${c.student_model || "—"}</dd></div>
    <div><dt>Turns</dt><dd>${c.num_turns || c.utterances.length}</dd></div>
  `;

  // Chat
  chatEl.classList.add("visible");
  let html = "";

  // Question image
  if (c.image_path) {
    html += `<img class="question-image" src="/files/${CURRENT_DATASET}/${c.image_path}" alt="Question image">`;
  }

  // Question text
  html += `<div class="question-text"><strong>Question:</strong> ${escapeHtml(c.question)}</div>`;

  html += `<div class="chat-wrapper">`;
  c.utterances.forEach(u => {
    html += `<div class="bubble ${u.role}">
      <div class="role">${u.role}</div>
      <span class="turn-num">#${u.turn}</span>
      <div>${escapeHtml(u.content)}</div>
      ${u.audio_path ? `<audio controls preload="none" src="/files/${CURRENT_DATASET}/${u.audio_path}"></audio>` : ""}
    </div>`;
  });
  html += `</div>`;

  chatEl.innerHTML = html;
  chatEl.scrollTop = 0;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

buildSidebar();
// Auto-select first conversation
if (DATA.length > 0) selectConversation(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Viewer")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Parent directory containing dataset subdirectories (default: output)",
    )
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir).resolve()
    datasets = discover_datasets()
    if not datasets:
        raise SystemExit(f"No datasets found in {OUTPUT_DIR}")

    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    print(f"Serving at http://localhost:{args.port}")
    app.run(debug=True, port=args.port)
