"""aiohttp web server for live full-duplex conversation monitoring."""

import asyncio
import io
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from aiohttp import web

from alignment.web_integration import register_routes as register_align_routes
from alignment.web_integration import render_toggle_html as render_align_toggle_html
from alignment.web_integration import render_toggle_script as render_align_toggle_script
from duplex import remote
from duplex.backend import (
    BACKEND_CHOICES,
    FRAME_SIZE,
    IMAGE_CAPABLE_BACKENDS,
    ModelBackend,
    pcm_int16_bytes_to_float32,
    uses_image,
)
from duplex.backend_human import HumanBackend
from duplex.relay import DuplexRelay
from duplex.schemas import DuplexConversation, DuplexSegment
from duplex.storage import (
    load_conversations,
    next_attempt_index,
    save_audio,
    save_audio_wav,
    save_audio_wav_stereo,
    save_conversation,
)
from evaluation.web_integration import register_routes as register_eval_routes
from evaluation.web_integration import (
    render_panel_html,
    render_panel_script,
)
from speech_eval.web_integration import register_routes as register_vap_routes
from speech_eval.web_integration import render_panel_html as render_vap_panel_html
from speech_eval.web_integration import render_panel_script as render_vap_panel_script
from src.conversation import _format_prompt
from src.data import load_mathvista
from src.schemas import MathVistaQuestion

logger = logging.getLogger(__name__)


class DuplexWebApp:
    """Web application for monitoring and controlling full-duplex conversations."""

    def __init__(
        self,
        backend_factory: Callable[[str, str, str, bytes | None], ModelBackend],
        voice_factory: Callable[[str, str], str],
        tutor_backend_type: str,
        student_backend_type: str,
        tutor_prompt_template: str,
        student_prompt_template: str,
        run_dir: Path | None,
        duplex_root: Path = Path("duplex_output"),
        speech_eval_root: Path = Path("speech_eval_output"),
        max_duration: float = 300.0,
        split: str = "testmini",
        n_questions: int | None = None,
    ):
        self._backend_factory = backend_factory
        self._voice_factory = voice_factory
        # Defaults from CLI; live values are whatever the UI sent on the last
        # /api/start and are used when saving the conversation record.
        self.default_tutor_backend_type = tutor_backend_type
        self.default_student_backend_type = student_backend_type
        self.tutor_backend_type = tutor_backend_type
        self.student_backend_type = student_backend_type
        self.tutor_voice = voice_factory(tutor_backend_type, "tutor")
        self.student_voice = voice_factory(student_backend_type, "student")
        self.tutor_prompt_template = tutor_prompt_template
        self.student_prompt_template = student_prompt_template
        # `run_dir` is mutable: the index handler resets it from ?run=<name>
        # so the user can switch the active dataset without restarting the
        # server. None means "no run selected" — handlers that need it 400.
        self.duplex_root = duplex_root
        # VAP frames + vap.jsonl live in a parallel tree under the same run name,
        # written post-hoc by `python -m speech_eval`. None of it is required for
        # the live flow; the VAP panel just stays hidden when it's absent.
        self.speech_eval_root = speech_eval_root
        self.run_dir: Path | None = run_dir
        self.max_duration = max_duration

        self._questions: list[MathVistaQuestion] = load_mathvista(
            split=split, n=n_questions
        )
        self._questions_by_pid: dict[str, MathVistaQuestion] = {
            q.pid: q for q in self._questions
        }
        self._image_cache: dict[str, bytes] = {}

        self._relay: DuplexRelay | None = None
        self._relay_task: asyncio.Task | None = None
        self._backends: dict[str, ModelBackend] = {}
        self._ws_clients: set[web.WebSocketResponse] = set()

        self._human_role: str | None = None
        # Per-WS residual buffer for re-chunking mic int16 bytes into
        # FRAME_SIZE samples; browser AudioWorklet batches may not align.
        self._mic_residual: dict[int, bytearray] = {}

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/questions", self._handle_questions)
        app.router.add_get("/api/defaults", self._handle_defaults)
        app.router.add_get("/api/runs", self._handle_runs_list)
        app.router.add_post("/api/runs", self._handle_runs_create)
        app.router.add_post("/api/start", self._handle_start)
        app.router.add_post("/api/stop", self._handle_stop)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_post("/api/remote/start", self._handle_remote_start)
        app.router.add_post("/api/remote/kill", self._handle_remote_kill)
        app.router.add_get("/api/remote/health", self._handle_remote_health)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/api/history", self._handle_history_list)
        app.router.add_get(r"/api/history/{idx:\d+}", self._handle_history_detail)
        app.router.add_get(r"/api/audio/{idx:\d+}/{role}", self._handle_audio)
        app.router.add_get("/api/image/{pid}", self._handle_image)
        app.router.add_get("/files/{path:.*}", self._handle_files)
        register_eval_routes(app, get_run_dir=lambda: self.run_dir)
        register_align_routes(app, get_run_dir=lambda: self.run_dir)
        register_vap_routes(
            app,
            get_run_dir=lambda: self.run_dir,
            get_vap_dir=self._vap_dir,
        )
        return app

    def _vap_dir(self) -> Path | None:
        """`speech_eval_output/<run>/` for the active run, or None."""
        if self.run_dir is None:
            return None
        return self.speech_eval_root / self.run_dir.name

    # -- Run discovery / selection -------------------------------------------

    # Run names live under `duplex_output/`; restrict to a safe alphabet to
    # block path traversal and weird shell-quoting issues with the run-name
    # propagated into manifest.json, scoring dirs, etc. The leading-char
    # requirement also blocks dot-prefixed names — _list_runs hides those,
    # so allowing them through here would create invisible runs.
    _RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
    _RUNS_EXCLUDE = frozenset({"__pycache__"})

    def _resolve_run_dir(self, name: str) -> Path | None:
        """Return `duplex_output/<name>/` if `name` is a valid run, else None.

        Validates the name against `_RUN_NAME_RE` and confirms the resolved
        path's parent is exactly the duplex_root (defeats `..` traversal).
        """
        if not self._RUN_NAME_RE.match(name):
            return None
        candidate = (self.duplex_root / name).resolve()
        if candidate.parent != self.duplex_root.resolve():
            return None
        return candidate if candidate.is_dir() else None

    def _list_runs(self) -> list[dict]:
        """Return existing runs sorted by mtime descending."""
        if not self.duplex_root.is_dir():
            return []
        out = []
        for p in self.duplex_root.iterdir():
            if not p.is_dir() or p.name.startswith(".") or p.name in self._RUNS_EXCLUDE:
                continue
            jsonl = p / "conversations.jsonl"
            n = 0
            if jsonl.is_file():
                try:
                    with jsonl.open() as f:
                        n = sum(1 for line in f if line.strip())
                except OSError:
                    pass
            out.append(
                {
                    "name": p.name,
                    "mtime": p.stat().st_mtime,
                    "n_conversations": n,
                }
            )
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return out

    async def _handle_runs_list(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "runs": self._list_runs(),
                "current": self.run_dir.name if self.run_dir else None,
            }
        )

    @staticmethod
    async def _parse_json_dict(
        request: web.Request,
    ) -> tuple[dict | None, web.Response | None]:
        """Parse the request body as a JSON object.

        request.json() only raises ValueError on malformed JSON; valid but
        non-dict bodies (null, [], "foo", 42) parse successfully and then
        crash any downstream `body.get(...)` with AttributeError → 500.
        Returns (body, None) on success or (None, 400-response).
        """
        try:
            body = await request.json()
        except ValueError:
            return None, web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(body, dict):
            return None, web.json_response({"error": "invalid JSON"}, status=400)
        return body, None

    async def _handle_runs_create(self, request: web.Request) -> web.Response:
        if self._relay and self._relay.is_running:
            return web.json_response(
                {"error": "Cannot create a run while a recording is in progress"},
                status=409,
            )
        body, err = await self._parse_json_dict(request)
        if err is not None:
            return err
        name = str(body.get("name", "")).strip()
        if not self._RUN_NAME_RE.match(name):
            return web.json_response(
                {
                    "error": (
                        "invalid run name (must start with A-Z a-z 0-9, "
                        "then A-Z a-z 0-9 . _ -, up to 64 chars)"
                    )
                },
                status=400,
            )
        candidate = (self.duplex_root / name).resolve()
        if candidate.parent != self.duplex_root.resolve():
            return web.json_response({"error": "invalid run name"}, status=400)
        # mkdir(exist_ok=False) is the only atomic way to detect a collision;
        # an exists()-then-mkdir pair races with concurrent POSTs.
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return web.json_response(
                {"error": f"run already exists: {name}"}, status=409
            )
        return web.json_response({"name": name, "redirect": f"/?run={name}"})

    # -- HTTP handlers -------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.Response:
        # ?run=<name> switches the active run. The dropdown reload pattern
        # (matches viewer.py) means we mutate state here, then the page
        # re-fetches history/eval/align against the new run_dir.
        # Refuse the switch while a relay is active — otherwise a second tab
        # or a typed URL would flip self.run_dir mid-recording and
        # _on_conversation_done would land the JSONL + audio in the wrong
        # dataset. The dropdown lock only guards the originating tab, and
        # the /api/start body check only fires on conversation start.
        requested = request.query.get("run", "").strip()
        if requested and not (self._relay and self._relay.is_running):
            resolved = self._resolve_run_dir(requested)
            if resolved is not None:
                self.run_dir = resolved
            # Silently fall back to the existing run_dir if the requested
            # name is unknown — the dropdown JS will surface the available
            # options. Don't 404 the page itself.
        html = HTML_TEMPLATE.replace("<!--EVAL_PANEL-->", render_panel_html())
        html = html.replace("<!--EVAL_SCRIPT-->", render_panel_script())
        html = html.replace("<!--ALIGN_TOGGLE-->", render_align_toggle_html())
        html = html.replace("<!--ALIGN_SCRIPT-->", render_align_toggle_script())
        html = html.replace("<!--VAP_PANEL-->", render_vap_panel_html())
        html = html.replace("<!--VAP_SCRIPT-->", render_vap_panel_script())
        return web.Response(text=html, content_type="text/html")

    async def _handle_questions(self, request: web.Request) -> web.Response:
        questions = [
            {
                "pid": q.pid,
                "question": q.question,
                "answer": q.answer,
                "has_image": q.image is not None,
            }
            for q in self._questions
        ]
        return web.json_response(questions)

    async def _handle_start(self, request: web.Request) -> web.Response:
        if self.run_dir is None:
            return web.json_response(
                {"error": "No run selected. Pick or create one in the dropdown."},
                status=400,
            )
        if self._relay and self._relay.is_running:
            return web.json_response(
                {"error": "A conversation is already running"}, status=409
            )

        body, err = await self._parse_json_dict(request)
        if err is not None:
            return err
        # Belt-and-suspenders: the dropdown locks while a run is active, but
        # if the client sends an explicit `run` it must match the server's
        # current selection. Avoids saving into the wrong dataset if the UI
        # state ever drifts.
        explicit_run = str(body.get("run", "")).strip()
        if explicit_run and explicit_run != self.run_dir.name:
            return web.json_response(
                {
                    "error": (
                        f"Server's active run is {self.run_dir.name!r}, "
                        f"not {explicit_run!r}. Reload the page after switching."
                    )
                },
                status=409,
            )
        pid = str(body.get("pid", ""))
        question = self._questions_by_pid.get(pid)
        if not question:
            return web.json_response({"error": f"Unknown PID: {pid}"}, status=404)

        # Backend types come from the UI dropdowns; fall back to CLI defaults.
        tutor_backend_type = str(
            body.get("tutor_backend") or self.default_tutor_backend_type
        )
        student_backend_type = str(
            body.get("student_backend") or self.default_student_backend_type
        )
        if tutor_backend_type not in BACKEND_CHOICES:
            return web.json_response(
                {"error": f"Unknown tutor_backend: {tutor_backend_type}"}, status=400
            )
        if student_backend_type not in BACKEND_CHOICES:
            return web.json_response(
                {"error": f"Unknown student_backend: {student_backend_type}"},
                status=400,
            )
        if tutor_backend_type == "human" and student_backend_type == "human":
            return web.json_response(
                {"error": "At most one role can be 'human' (not both)."},
                status=400,
            )

        # Format prompts
        tutor_prompt = _format_prompt(self.tutor_prompt_template, question)
        student_prompt = _format_prompt(self.student_prompt_template, question)

        # MoshiVis requires an image; serve a clear error if missing.
        image_bytes = self._get_image_png(pid)
        needs_image = "moshivis" in (tutor_backend_type, student_backend_type)
        if needs_image and image_bytes is None:
            return web.json_response(
                {"error": "Selected question has no image, required by MoshiVis."},
                status=400,
            )

        # Create backends via factory
        tutor_backend = self._backend_factory(
            "tutor", tutor_prompt, tutor_backend_type, image_bytes
        )
        student_backend = self._backend_factory(
            "student", student_prompt, student_backend_type, image_bytes
        )
        self._backends = {"tutor": tutor_backend, "student": student_backend}

        if isinstance(tutor_backend, HumanBackend):
            self._human_role = "tutor"
        elif isinstance(student_backend, HumanBackend):
            self._human_role = "student"
        else:
            self._human_role = None

        # Record the live selection so _on_conversation_done saves the right
        # backend type + voice for this conversation.
        self.tutor_backend_type = tutor_backend_type
        self.student_backend_type = student_backend_type
        self.tutor_voice = self._voice_factory(tutor_backend_type, "tutor")
        self.student_voice = self._voice_factory(student_backend_type, "student")

        tutor_uses_image = uses_image(tutor_backend_type, image_bytes)
        student_uses_image = uses_image(student_backend_type, image_bytes)

        # Create relay with callbacks
        self._relay = DuplexRelay(
            tutor_backend=tutor_backend,
            student_backend=student_backend,
            max_duration=self.max_duration,
            on_token=self._broadcast_token,
            on_segment=self._broadcast_segment,
            on_audio=self._broadcast_audio,
            on_interrupt=self._broadcast_interrupt,
            on_done=lambda dur: self._on_conversation_done(pid, question, dur),
        )

        self._relay_task = asyncio.create_task(self._run_conversation(pid, question))
        await self._broadcast(
            {
                "type": "status",
                "state": "starting",
                "pid": pid,
                "tutor_backend": tutor_backend_type,
                "student_backend": student_backend_type,
                "human_role": self._human_role,
                "tutor_uses_image": tutor_uses_image,
                "student_uses_image": student_uses_image,
            }
        )
        return web.json_response(
            {
                "status": "started",
                "pid": pid,
                "tutor_backend": tutor_backend_type,
                "student_backend": student_backend_type,
                "human_role": self._human_role,
                "tutor_uses_image": tutor_uses_image,
                "student_uses_image": student_uses_image,
            }
        )

    async def _handle_stop(self, request: web.Request) -> web.Response:
        if self._relay:
            self._relay.stop()
            return web.json_response({"status": "stopping"})
        return web.json_response({"status": "not_running"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        if self._relay and self._relay.is_running:
            return web.json_response(
                {"state": "running", "elapsed": self._relay.elapsed}
            )
        return web.json_response({"state": "idle"})

    async def _handle_defaults(self, request: web.Request) -> web.Response:
        """Return the initial dropdown defaults + the list of backend choices."""
        return web.json_response(
            {
                "backends": BACKEND_CHOICES,
                "cloud_backends": sorted(remote.CLOUD_BACKENDS),
                "image_capable_backends": sorted(IMAGE_CAPABLE_BACKENDS),
                "tutor_backend": self.default_tutor_backend_type,
                "student_backend": self.default_student_backend_type,
            }
        )

    def _resolve_remote(
        self, backend: str, role: str, cloud_response: dict
    ) -> tuple[dict[str, object] | None, web.Response | None]:
        """Validate (backend, role) for remote endpoints.

        Returns ``(target, None)`` on success or ``(None, response)`` where
        ``response`` short-circuits the handler with either a 400 (invalid
        role/pair) or a 200 cloud-backend response.
        """
        if role not in ("tutor", "student"):
            return None, web.json_response(
                {"error": "role must be 'tutor' or 'student'"}, status=400
            )
        if not remote.has_remote_server(backend):
            return None, web.json_response(cloud_response)
        target = remote.get_target(backend, role)
        if target is None:
            return None, web.json_response(
                {"error": f"No remote target for {backend}/{role}"}, status=400
            )
        return target, None

    async def _handle_remote_start(self, request: web.Request) -> web.Response:
        """Start the remote model server for ``(backend, role)`` if not already up."""
        body, err = await self._parse_json_dict(request)
        if err is not None:
            return err
        backend = str(body.get("backend", ""))
        role = str(body.get("role", ""))
        target, err = self._resolve_remote(
            backend,
            role,
            {"ok": True, "already_running": True, "reason": "no_remote_server_needed"},
        )
        if err is not None:
            return err
        result = await remote.start_remote_server(backend, role)
        result["host"] = target["probe_host"]
        result["port"] = target["probe_port"]
        return web.json_response(result)

    async def _handle_remote_kill(self, request: web.Request) -> web.Response:
        """Stop the remote model server for ``(backend, role)``."""
        body, err = await self._parse_json_dict(request)
        if err is not None:
            return err
        backend = str(body.get("backend", ""))
        role = str(body.get("role", ""))
        target, err = self._resolve_remote(
            backend, role, {"ok": True, "reason": "no_remote_server_needed"}
        )
        if err is not None:
            return err
        result = await remote.kill_remote_server(backend, role)
        result["host"] = target["probe_host"]
        result["port"] = target["probe_port"]
        return web.json_response(result)

    async def _handle_remote_health(self, request: web.Request) -> web.Response:
        """TCP probe the expected host:port for ``(backend, role)``."""
        backend = request.query.get("backend", "")
        role = request.query.get("role", "")
        target, err = self._resolve_remote(
            backend, role, {"reachable": True, "reason": "no_remote_server_needed"}
        )
        if err is not None:
            return err
        host = str(target["probe_host"])
        port = int(target["probe_port"])  # type: ignore[arg-type]
        reachable = await remote.probe_tcp(host, port)
        return web.json_response({"reachable": reachable, "host": host, "port": port})

    async def _handle_files(self, request: web.Request) -> web.Response:
        if self.run_dir is None:
            return web.Response(status=404)
        path = self.run_dir / request.match_info["path"]
        if not path.is_file():
            return web.Response(status=404)
        return web.FileResponse(path)

    def _has_question_image(self, pid: str) -> bool:
        q = self._questions_by_pid.get(pid)
        return q is not None and q.image is not None

    def _get_image_png(self, pid: str) -> bytes | None:
        """Return cached PNG bytes for a question image, or None."""
        if pid in self._image_cache:
            return self._image_cache[pid]
        q = self._questions_by_pid.get(pid)
        if not q or q.image is None:
            return None
        buf = io.BytesIO()
        q.image.save(buf, format="PNG")
        self._image_cache[pid] = buf.getvalue()
        return self._image_cache[pid]

    async def _handle_image(self, request: web.Request) -> web.Response:
        """Serve the question image as PNG for a given PID."""
        pid = request.match_info["pid"]
        png = self._get_image_png(pid)
        if png is None:
            return web.Response(status=404, text="Image not found")
        return web.Response(
            body=png,
            content_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # -- History handlers ----------------------------------------------------

    def _load_history(self) -> list[dict]:
        """Load conversations from JSONL, adding an index field to each."""
        if self.run_dir is None:
            return []
        jsonl_path = self.run_dir / "conversations.jsonl"
        conversations = load_conversations(jsonl_path)
        for idx, conv in enumerate(conversations):
            conv["idx"] = idx
        return conversations

    def _has_audio(self, idx: int, role: str) -> bool:
        """Check if audio exists for a conversation (index-based only)."""
        if self.run_dir is None:
            return False
        audio_dir = self.run_dir / "audio" / str(idx)
        return (audio_dir / f"{role}_full.wav").is_file() or (
            audio_dir / f"{role}_full.opus"
        ).is_file()

    def _has_combined_audio(self, idx: int) -> bool:
        if self.run_dir is None:
            return False
        return (self.run_dir / "audio" / str(idx) / "combined.wav").is_file()

    async def _handle_history_list(self, request: web.Request) -> web.Response:
        conversations = self._load_history()
        summaries = []
        for conv in conversations:
            pid = conv["pid"]
            idx = conv["idx"]
            summaries.append(
                {
                    "idx": idx,
                    "pid": pid,
                    "attempt_index": int(conv.get("attempt_index", 0)),
                    "question": conv["question"][:100],
                    "answer": conv.get("answer", ""),
                    "duration": conv.get("duration", 0),
                    "num_segments": len(conv.get("segments", [])),
                    "has_tutor_audio": self._has_audio(idx, "tutor"),
                    "has_student_audio": self._has_audio(idx, "student"),
                    "has_combined_audio": self._has_combined_audio(idx),
                    "has_image": self._has_question_image(pid),
                    "tutor_backend": conv.get("tutor_backend", ""),
                    "student_backend": conv.get("student_backend", ""),
                    "tutor_uses_image": conv.get("tutor_uses_image", False),
                    "student_uses_image": conv.get("student_uses_image", False),
                }
            )
        return web.json_response(summaries)

    async def _handle_history_detail(self, request: web.Request) -> web.Response:
        idx = int(request.match_info["idx"])
        conversations = self._load_history()
        if idx < 0 or idx >= len(conversations):
            return web.json_response({"error": "Conversation not found"}, status=404)
        conv = conversations[idx]
        conv["has_tutor_audio"] = self._has_audio(idx, "tutor")
        conv["has_student_audio"] = self._has_audio(idx, "student")
        conv["has_combined_audio"] = self._has_combined_audio(idx)
        conv["has_image"] = self._has_question_image(conv["pid"])
        conv.setdefault("tutor_uses_image", False)
        conv.setdefault("student_uses_image", False)
        return web.json_response(conv)

    async def _handle_audio(self, request: web.Request) -> web.Response:
        if self.run_dir is None:
            return web.Response(status=404, text="No run selected")
        idx = int(request.match_info["idx"])
        role = request.match_info["role"]
        audio_dir = self.run_dir / "audio" / str(idx)

        def _serve(p: Path, mime: str) -> web.FileResponse:
            return web.FileResponse(
                p, headers={"Content-Type": mime, "Cache-Control": "no-cache"}
            )

        if role == "combined":
            combined_path = audio_dir / "combined.wav"
            if combined_path.is_file():
                return _serve(combined_path, "audio/wav")
            return web.Response(status=404, text="Audio file not found")

        if role not in ("tutor", "student"):
            return web.Response(
                status=400, text="Role must be tutor, student, or combined"
            )

        # Prefer WAV (universal), fall back to Opus (legacy/native)
        wav_path = audio_dir / f"{role}_full.wav"
        opus_path = audio_dir / f"{role}_full.opus"
        if wav_path.is_file():
            return _serve(wav_path, "audio/wav")
        elif opus_path.is_file():
            return _serve(opus_path, "audio/ogg")
        return web.Response(status=404, text="Audio file not found")

    # -- WebSocket handler ---------------------------------------------------

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        ws_key = id(ws)
        logger.info(
            "Browser WebSocket connected. Total clients: %d", len(self._ws_clients)
        )
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    self._handle_mic_frame(ws_key, msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except ValueError:
                        continue
                    if payload.get("type") == "mic_error":
                        logger.warning(
                            "Browser reported mic error: %s",
                            payload.get("message", ""),
                        )
        finally:
            self._ws_clients.discard(ws)
            self._mic_residual.pop(ws_key, None)
        return ws

    def _handle_mic_frame(self, ws_key: int, data: bytes) -> None:
        """Decode int16 mic bytes from the browser and forward as FRAME_SIZE frames."""
        if self._human_role is None:
            return
        backend = self._backends.get(self._human_role)
        if not isinstance(backend, HumanBackend):
            return
        buf = self._mic_residual.setdefault(ws_key, bytearray())
        buf.extend(data)
        frame_bytes = FRAME_SIZE * 2
        while len(buf) >= frame_bytes:
            chunk = bytes(buf[:frame_bytes])
            del buf[:frame_bytes]
            backend.push_mic_frame(pcm_int16_bytes_to_float32(chunk))

    # -- Broadcast helpers ---------------------------------------------------

    async def _broadcast(self, data: dict):
        """Send JSON message to all connected browser clients."""
        text = json.dumps(data)
        closed = set()
        for ws in self._ws_clients:
            try:
                await ws.send_str(text)
            except Exception:
                closed.add(ws)
        self._ws_clients -= closed

    async def _broadcast_token(self, role: str, text: str, timestamp: float):
        await self._broadcast(
            {"type": "token", "role": role, "text": text, "time": round(timestamp, 3)}
        )

    async def _broadcast_segment(self, segment: DuplexSegment):
        await self._broadcast(
            {
                "type": "segment",
                "role": segment.role,
                "text": segment.text,
                "start": round(segment.start_time, 3),
                "end": round(segment.end_time, 3),
            }
        )

    async def _broadcast_interrupt(self, role: str):
        """Tell browsers to flush any scheduled audio for `role`.

        The backend's server cancelled its response mid-utterance; any audio
        frames already sent to the browser are still queued in the Web Audio
        graph and would otherwise keep playing for hundreds of ms.
        """
        await self._broadcast({"type": "interrupt", "role": role})

    async def _broadcast_audio(self, role: str, pcm_bytes: bytes):
        """Send PCM audio to all browser clients as binary WebSocket frames.

        Format: 1 byte role tag (0x01=tutor, 0x02=student) + float32 PCM samples.
        """
        if not self._ws_clients:
            return
        role_tag = b"\x01" if role == "tutor" else b"\x02"
        frame = role_tag + pcm_bytes
        logger.debug(
            "Broadcasting audio %s: %d bytes to %d clients",
            role,
            len(frame),
            len(self._ws_clients),
        )
        closed = set()
        for ws in self._ws_clients:
            try:
                await ws.send_bytes(frame)
            except Exception:
                closed.add(ws)
        self._ws_clients -= closed

    # -- Conversation lifecycle ----------------------------------------------

    async def _run_conversation(self, pid: str, question: MathVistaQuestion):
        """Run the relay and save results when done."""
        try:
            await self._broadcast({"type": "status", "state": "running", "pid": pid})
            await self._relay.run()
        except Exception:
            logger.exception("Conversation pid=%s failed.", pid)
            await self._broadcast({"type": "error", "message": "Conversation failed"})
        finally:
            self._relay_task = None
            self._human_role = None

    async def _on_conversation_done(
        self, pid: str, question: MathVistaQuestion, duration: float
    ):
        """Save the completed conversation."""
        # `run_dir` cannot become None mid-conversation — the dropdown is
        # locked while a relay runs and `/api/start` rejects calls with a
        # mismatched run — but assert it explicitly for clarity.
        assert self.run_dir is not None, "run_dir disappeared during conversation"
        segments = self._relay.segments
        pcm_bufs = self._relay.audio_buffers
        native_bufs = self._relay.native_audio_buffers
        audio_offsets = self._relay.audio_start_offsets

        # Empty conversation (e.g. gemini-live silently produced nothing
        # before silence_timeout): broadcast completion so the UI updates,
        # but don't write a JSONL row.
        if not segments and not pcm_bufs.get("tutor") and not pcm_bufs.get("student"):
            logger.warning(
                "pid=%s -- empty conversation (no audio, no segments) "
                "after %.1fs; not saving.",
                pid,
                duration,
            )
            await self._broadcast(
                {
                    "type": "done",
                    "pid": pid,
                    "duration": round(duration, 1),
                    "num_segments": 0,
                }
            )
            return

        output_path = self.run_dir / "conversations.jsonl"
        idx = 0
        if output_path.is_file():
            with open(output_path) as f:
                idx = sum(1 for line in f if line.strip())

        attempt_index = next_attempt_index(output_path, pid)

        audio_dir = self.run_dir / "audio" / str(idx)
        for role in ("tutor", "student"):
            if pcm_bufs.get(role):
                save_audio_wav(
                    pcm_bufs[role],
                    audio_dir / f"{role}_full.wav",
                    total_duration_s=duration,
                    offset_s=audio_offsets[role],
                )
            if native_bufs.get(role) and role in self._backends:
                ext = self._backends[role].get_native_audio_ext()
                save_audio(native_bufs[role], audio_dir / f"{role}_full{ext}")

        save_audio_wav_stereo(
            pcm_bufs.get("tutor", b""),
            pcm_bufs.get("student", b""),
            audio_dir / "combined.wav",
            total_duration_s=duration,
            left_offset_s=audio_offsets["tutor"],
            right_offset_s=audio_offsets["student"],
        )

        metadata = dict(question.metadata)

        image_bytes = self._get_image_png(pid)
        conversation = DuplexConversation(
            pid=pid,
            question=question.question,
            answer=question.answer,
            tutor_voice=self.tutor_voice,
            student_voice=self.student_voice,
            duration=duration,
            segments=segments,
            metadata=metadata,
            tutor_backend=self.tutor_backend_type,
            student_backend=self.student_backend_type,
            tutor_uses_image=uses_image(self.tutor_backend_type, image_bytes),
            student_uses_image=uses_image(self.student_backend_type, image_bytes),
            attempt_index=attempt_index,
        )

        save_conversation(conversation, output_path)

        await self._broadcast(
            {
                "type": "done",
                "pid": pid,
                "duration": round(duration, 1),
                "num_segments": len(segments),
            }
        )


# -- HTML Template -----------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full-Duplex Conversation Monitor</title>
<style>
  :root {
    --bg: #f5f5f5; --panel-bg: #fff;
    --student: #e3f2fd; --tutor: #f3e5f5;
    --border: #ddd; --text: #222; --muted: #888;
    --accent: #1976d2; --danger: #d32f2f; --success: #388e3c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }

  /* Header */
  .header { padding: 12px 24px; background: var(--panel-bg); border-bottom: 1px solid var(--border);
            display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .header h1 { font-size: 16px; margin-right: 16px; }
  .header select { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }
  .header button { padding: 6px 16px; border: none; border-radius: 6px; font-size: 13px;
                   cursor: pointer; font-weight: 600; }
  .btn-start { background: var(--accent); color: #fff; }
  .btn-start:disabled { background: #90caf9; cursor: not-allowed; }
  .btn-stop { background: var(--danger); color: #fff; }
  .btn-stop:disabled { background: #ef9a9a; cursor: not-allowed; }
  .status-badge { padding: 4px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .status-idle { background: #e8eaf6; color: #3949ab; }
  .status-running { background: #e8f5e9; color: #2e7d32; }
  .status-starting { background: #fff3e0; color: #e65100; }

  /* Mode toggle */
  .mode-toggle { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; margin-right: 12px; }
  .mode-btn { padding: 5px 14px; font-size: 12px; font-weight: 600; border: none; cursor: pointer;
              background: var(--panel-bg); color: var(--muted); }
  .mode-btn.active { background: var(--accent); color: #fff; }
  .mode-btn:not(.active):hover { background: #f0f0f0; }

  /* Main layout */
  .main { flex: 1; display: flex; overflow: hidden; }

  /* Left panel */
  .left-panel { width: 420px; min-width: 360px; border-right: 1px solid var(--border);
                background: var(--panel-bg); padding: 16px; overflow-y: auto; }
  .left-panel h3 { font-size: 13px; color: var(--muted); margin-bottom: 8px; text-transform: uppercase; }
  .question-text { background: #fffde7; padding: 12px; border-radius: 8px; font-size: 13px;
                   border: 1px solid #fff9c4; margin-bottom: 12px; line-height: 1.5; }
  .question-text strong { color: #f57f17; }
  .question-image { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 12px; display: block; }
  .image-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-bottom: 8px; }
  .image-badge.not-used { background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }
  .image-badge.used { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
  .image-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .meta-item { font-size: 12px; margin-bottom: 4px; }
  .meta-item label { font-weight: 600; color: var(--muted); }

  /* History list */
  .history-item { padding: 10px 12px; border: 1px solid var(--border); border-radius: 8px;
                  margin-bottom: 8px; cursor: pointer; transition: background 0.15s; }
  .history-item:hover { background: #f5f5f5; }
  .history-item.active { background: #e3f2fd; border-color: var(--accent); }

  /* Transcript panel */
  .transcript { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .transcript-header { padding: 12px 24px; border-bottom: 1px solid var(--border);
                       background: var(--panel-bg); font-size: 13px; font-weight: 600; }
  .transcript-body { flex: 1; overflow-y: auto; padding: 16px 24px; }
  .token-line { margin-bottom: 2px; font-size: 13px; font-family: "SF Mono", Monaco, monospace; }
  .token-line .time { color: var(--muted); font-size: 11px; margin-right: 8px; }
  .token-line .role-student { color: #1565c0; font-weight: 600; }
  .token-line .role-tutor { color: #7b1fa2; font-weight: 600; }
  .token-line .text { }

  /* Segment bubbles */
  .segment { max-width: 80%; padding: 10px 14px; border-radius: 12px; margin-bottom: 8px;
             font-size: 13px; line-height: 1.5; transition: background 0.2s, box-shadow 0.2s; }
  .segment.student { background: var(--student); align-self: flex-start; border-bottom-left-radius: 4px; }
  .segment.tutor { background: var(--tutor); align-self: flex-end; border-bottom-right-radius: 4px; }
  .segment .seg-header { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .segment-wrapper { display: flex; flex-direction: column; }
  .segment.highlight { box-shadow: 0 0 0 2px var(--accent); }
  .segment.highlight.student { background: #bbdefb; }
  .segment.highlight.tutor { background: #e1bee7; }

  /* Tabs */
  .tab-bar { display: flex; gap: 0; border-bottom: 1px solid var(--border); background: var(--panel-bg); }
  .tab { padding: 8px 20px; font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tab.active { border-bottom-color: var(--accent); font-weight: 600; color: var(--accent); }

  .empty-state { display: flex; align-items: center; justify-content: center;
                 flex: 1; color: var(--muted); font-size: 15px; }

  /* Audio controls */
  .audio-controls { padding: 10px 24px; background: var(--panel-bg); border-top: 1px solid var(--border);
                    display: flex; align-items: center; gap: 16px; font-size: 13px; }
  .audio-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .volume-slider { width: 80px; }
  .audio-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                     background: #ccc; transition: background 0.1s; }
  .audio-indicator.active { background: #4caf50; }

  /* Playback controls */
  .playback-controls { padding: 10px 24px; background: var(--panel-bg); border-top: 1px solid var(--border);
                       display: flex; align-items: center; gap: 14px; font-size: 13px; }
  .playback-controls button { padding: 6px 16px; border: none; border-radius: 6px; font-size: 13px;
                              cursor: pointer; font-weight: 600; background: var(--accent); color: #fff; }
  .playback-controls button:disabled { background: #90caf9; cursor: not-allowed; }
  .progress-bar-wrapper { flex: 1; height: 8px; background: #e0e0e0; border-radius: 4px;
                          cursor: pointer; position: relative; }
  .progress-bar-fill { height: 100%; background: var(--accent); border-radius: 4px; width: 0%;
                       transition: width 0.1s linear; pointer-events: none; }
  .time-display { font-size: 12px; color: var(--muted); min-width: 100px; text-align: center; font-variant-numeric: tabular-nums; }
  .speed-select { padding: 4px 8px; border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }

  /* Backend selectors + remote-server widgets */
  .backend-selector { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: var(--muted); }
  .backend-selector select { padding: 4px 8px; font-size: 12px; }
  .server-widget { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; }
  .server-widget.hidden { display: none; }
  .server-dot { width: 10px; height: 10px; border-radius: 50%; background: #ccc; transition: background 0.2s; }
  .server-dot.red { background: var(--danger); }
  .server-dot.yellow { background: #f9a825; }
  .server-dot.green { background: var(--success); }
  .server-btn { padding: 3px 8px; font-size: 11px; border: 1px solid var(--border); background: #fafafa; color: var(--text); border-radius: 4px; cursor: pointer; font-weight: 500; }
  .server-btn:hover:not(:disabled) { background: #f0f0f0; }
  .server-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .server-btn.kill { border-color: var(--danger); color: var(--danger); }
  .server-btn.kill:hover:not(:disabled) { background: #ffebee; }
</style>
</head>
<body>

<div class="header">
  <h1>Full-Duplex Conversations</h1>
  <span class="run-picker" style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted);">
    Run:
    <select id="run-select" title="Switch active run (reloads the page)"></select>
    <span id="new-run-form" style="display:none; gap:6px; align-items:center;">
      <input id="new-run-name" type="text" placeholder="new run name"
             style="padding:5px 8px; border:1px solid var(--border); border-radius:6px; font-size:12px;" />
      <button id="new-run-create" class="btn-start" type="button" style="padding:4px 10px;">Create</button>
      <button id="new-run-cancel" type="button"
              style="padding:4px 8px; border:1px solid var(--border); background:#fff; border-radius:6px; cursor:pointer; font-size:12px;">Cancel</button>
    </span>
    <span id="run-error" style="color:var(--danger); font-size:12px;"></span>
  </span>
  <div class="mode-toggle">
    <button class="mode-btn active" data-mode="monitor" onclick="setMode('monitor')">Monitor</button>
    <button class="mode-btn" data-mode="history" onclick="setMode('history')">History</button>
  </div>
  <span id="monitor-controls">
    <span class="backend-selector">Tutor <select id="tutor-backend"></select></span>
    <span id="tutor-server" class="server-widget hidden"></span>
    <span class="backend-selector">Student <select id="student-backend"></select></span>
    <span id="student-server" class="server-widget hidden"></span>
    <select id="question-select"></select>
    <button class="btn-start" id="btn-start" onclick="startConversation()">Start</button>
    <button class="btn-stop" id="btn-stop" onclick="stopConversation()" disabled>Stop</button>
    <span class="status-badge status-idle" id="status-badge">Idle</span>
    <span id="elapsed" style="font-size:12px; color:var(--muted)"></span>
  </span>
  <span id="history-controls" style="display:none">
    <button class="btn-start" onclick="loadHistoryList()">Refresh</button>
  </span>
</div>

<div class="main">
  <div class="left-panel" id="left-panel">
    <div id="monitor-left">
      <h3>Question</h3>
      <div id="question-info">
        <div class="empty-state" style="height:200px">Select a question and start</div>
      </div>
    </div>
    <div id="history-left" style="display:none">
      <h3>Saved Conversations</h3>
      <div id="history-list"></div>
      <div id="history-question-info" style="margin-top:12px"></div>
      <!--EVAL_PANEL-->
    </div>
  </div>

  <div class="transcript">
    <div class="tab-bar">
      <div class="tab active" data-tab="tokens" onclick="switchTab('tokens')">Live Tokens</div>
      <div class="tab" data-tab="segments" onclick="switchTab('segments')">Segments</div>
      <!--ALIGN_TOGGLE-->
    </div>
    <div class="transcript-body" id="transcript-tokens"></div>
    <div class="transcript-body" id="transcript-segments" style="display:none"></div>
  </div>
</div>

<div class="audio-controls" id="live-audio-controls">
  <label>
    <input type="checkbox" id="audio-enabled" checked> Live Audio
  </label>
  <span>Tutor <span class="audio-indicator" id="ind-tutor"></span></span>
  <input type="range" class="volume-slider" id="vol-tutor" min="0" max="100" value="80" title="Tutor volume">
  <span>Student <span class="audio-indicator" id="ind-student"></span></span>
  <input type="range" class="volume-slider" id="vol-student" min="0" max="100" value="80" title="Student volume">
</div>

<div class="playback-controls" id="playback-controls" style="display:none">
  <button id="btn-play" onclick="togglePlayback()" disabled>Play</button>
  <div class="progress-bar-wrapper" id="progress-bar">
    <div class="progress-bar-fill" id="progress-fill"></div>
  </div>
  <span class="time-display" id="time-display">0:00 / 0:00</span>
  <select class="speed-select" id="speed-select" onchange="setPlaybackSpeed(this.value)">
    <option value="0.5">0.5x</option>
    <option value="1" selected>1x</option>
    <option value="1.5">1.5x</option>
    <option value="2">2x</option>
  </select>
  <span>Tutor</span>
  <input type="range" class="volume-slider" id="pb-vol-tutor" min="0" max="100" value="80" title="Tutor volume">
  <span>Student</span>
  <input type="range" class="volume-slider" id="pb-vol-student" min="0" max="100" value="80" title="Student volume">
</div>

<div class="playback-controls" id="combined-controls" style="display:none">
  <span>Combined (stereo: tutor=L, student=R)</span>
  <audio id="combined-audio" controls preload="none" style="flex:1; min-width:0;"></audio>
</div>

<!--VAP_PANEL-->

<script>
let ws = null;
let questions = [];
let currentPid = null;
let elapsedTimer = null;
let startTime = null;
let currentMode = 'monitor';

// ---- Utility ----
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m + ':' + String(s).padStart(2, '0');
}

let IMAGE_CAPABLE_BACKENDS = new Set();

function wouldUseImage(backend, hasImage) {
  return !!hasImage && IMAGE_CAPABLE_BACKENDS.has(backend);
}

function imageBadge(role, used) {
  const cls = used ? 'used' : 'not-used';
  const state = used ? 'used by model' : 'not used by model';
  return '<span class="image-badge ' + cls + '">' + role + ': image ' + state + '</span>';
}

function resolveImageUsed(role, conv) {
  // Prefer the server-provided flag; fall back to the current dropdown
  // selection so the pre-start preview is still informative.
  const key = role + '_uses_image';
  return (key in conv)
    ? !!conv[key]
    : wouldUseImage(getBackend(role), conv.has_image);
}

function buildQuestionInfoHtml(conv) {
  let html = '<div class="question-text"><strong>Question:</strong> ' + escapeHtml(conv.question) + '</div>';
  if (conv.has_image) {
    html += '<div class="image-badges">';
    html += imageBadge('tutor', resolveImageUsed('tutor', conv));
    html += imageBadge('student', resolveImageUsed('student', conv));
    html += '</div>';
    html += '<img class="question-image" src="/api/image/' + conv.pid + '" alt="Question image">';
  }
  html += '<div class="meta-item"><label>PID:</label> ' + conv.pid + '</div>';
  html += '<div class="meta-item"><label>Answer:</label> ' + escapeHtml(conv.answer || '') + '</div>';
  if (conv.tutor_backend || conv.student_backend) {
    html += '<div class="meta-item"><label>Backends:</label> tutor=' + escapeHtml(conv.tutor_backend || '?') + ', student=' + escapeHtml(conv.student_backend || '?') + '</div>';
  }
  return html;
}

function renderMonitorQuestionInfo(extra) {
  const sel = document.getElementById('question-select');
  const q = questions.find(x => x.pid === sel.value);
  if (!q) return;
  const conv = Object.assign({}, q, extra || {});
  document.getElementById('question-info').innerHTML = buildQuestionInfoHtml(conv);
}

// ---- Mode Switching ----
function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  const isHistory = mode === 'history';
  document.getElementById('monitor-controls').style.display = isHistory ? 'none' : '';
  document.getElementById('history-controls').style.display = isHistory ? '' : 'none';
  document.getElementById('monitor-left').style.display = isHistory ? 'none' : '';
  document.getElementById('history-left').style.display = isHistory ? '' : 'none';
  document.getElementById('live-audio-controls').style.display = isHistory ? 'none' : '';
  document.getElementById('playback-controls').style.display = isHistory ? '' : 'none';
  if (!isHistory) resetCombinedAudio();

  if (isHistory) {
    stopHistoryPlayback();
    loadHistoryList();
    switchTab('segments');
  } else {
    stopHistoryPlayback();
  }
}

// ---- Run picker -----------------------------------------------------------
// Reload-on-change pattern (mirrors viewer.py): switching active dataset is a
// hard navigation so the entire UI state (history, eval, alignment, audio) is
// rebuilt against the new run_dir.

const NEW_RUN_SENTINEL = '__new__';

async function loadRunsAndPopulate() {
  const sel = document.getElementById('run-select');
  const errBox = document.getElementById('run-error');
  errBox.textContent = '';
  try {
    const resp = await fetch('/api/runs');
    if (!resp.ok) { errBox.textContent = 'Failed to load runs'; return; }
    const data = await resp.json();
    const runs = Array.isArray(data.runs) ? data.runs : [];
    const current = data.current || null;

    sel.innerHTML = '';
    if (!runs.length && !current) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(no run selected)';
      opt.disabled = true;
      opt.selected = true;
      sel.appendChild(opt);
    }
    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.name;
      opt.textContent = `${r.name} (${r.n_conversations})`;
      if (r.name === current) opt.selected = true;
      sel.appendChild(opt);
    });
    if (current && !runs.some(r => r.name === current)) {
      // Defensive: current run isn't in listing (race or filesystem oddity).
      const opt = document.createElement('option');
      opt.value = current;
      opt.textContent = current;
      opt.selected = true;
      sel.appendChild(opt);
    }
    const newOpt = document.createElement('option');
    newOpt.value = NEW_RUN_SENTINEL;
    newOpt.textContent = '+ New run...';
    sel.appendChild(newOpt);
  } catch (e) {
    errBox.textContent = 'Error loading runs';
  }
}

function showNewRunForm(show) {
  document.getElementById('new-run-form').style.display = show ? 'inline-flex' : 'none';
  document.getElementById('run-select').style.display = show ? 'none' : '';
  if (show) document.getElementById('new-run-name').focus();
}

async function createNewRun() {
  const errBox = document.getElementById('run-error');
  const nameEl = document.getElementById('new-run-name');
  const name = (nameEl.value || '').trim();
  if (!name) { errBox.textContent = 'name required'; return; }
  errBox.textContent = '';
  try {
    const resp = await fetch('/api/runs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name}),
    });
    const data = await resp.json();
    if (!resp.ok) { errBox.textContent = data.error || `HTTP ${resp.status}`; return; }
    window.location.href = data.redirect || ('/?run=' + encodeURIComponent(name));
  } catch (e) {
    errBox.textContent = 'Network error';
  }
}

(function initRunPicker() {
  const sel = document.getElementById('run-select');
  sel.addEventListener('change', () => {
    const v = sel.value;
    if (v === NEW_RUN_SENTINEL) {
      showNewRunForm(true);
      return;
    }
    if (!v) return;
    window.location.href = '/?run=' + encodeURIComponent(v);
  });
  document.getElementById('new-run-create').addEventListener('click', createNewRun);
  document.getElementById('new-run-cancel').addEventListener('click', () => {
    document.getElementById('new-run-name').value = '';
    document.getElementById('run-error').textContent = '';
    showNewRunForm(false);
    loadRunsAndPopulate();   // restore the previously selected option
  });
  document.getElementById('new-run-name').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') createNewRun();
    else if (ev.key === 'Escape') {
      showNewRunForm(false);
      loadRunsAndPopulate();
    }
  });
  loadRunsAndPopulate();
})();

function setRunPickerLocked(locked) {
  document.getElementById('run-select').disabled = !!locked;
}

// ---- Monitor Mode (existing) ----

fetch('/api/questions')
  .then(r => r.json())
  .then(data => {
    questions = data;
    const sel = document.getElementById('question-select');
    data.forEach(q => {
      const opt = document.createElement('option');
      opt.value = q.pid;
      opt.textContent = `PID ${q.pid}${q.has_image ? ' [img]' : ''}: ${q.question.substring(0, 55)}...`;
      sel.appendChild(opt);
    });
    // Show question preview on select change
    sel.addEventListener('change', () => renderMonitorQuestionInfo());
    // Trigger initial preview
    if (data.length > 0) sel.dispatchEvent(new Event('change'));
  });

function connectWS() {
  if (ws && ws.readyState <= 1) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      handleAudioFrame(e.data);
    } else {
      handleMessage(JSON.parse(e.data));
    }
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}

function handleMessage(msg) {
  if (msg.type === 'token') {
    appendToken(msg.role, msg.text, msg.time);
  } else if (msg.type === 'segment') {
    appendSegment(msg.role, msg.text, msg.start, msg.end);
  } else if (msg.type === 'status') {
    if (msg.human_role !== undefined) pendingHumanRole = msg.human_role;
    if (msg.state === 'running' && pendingHumanRole) {
      startMicCapture(pendingHumanRole);
    }
    setStatus(msg.state);
  } else if (msg.type === 'interrupt') {
    flushRoleAudio(msg.role);
  } else if (msg.type === 'done') {
    setStatus('done');
    stopMicCapture();
    pendingHumanRole = null;
    clearInterval(elapsedTimer);
    document.getElementById('elapsed').textContent = `Done: ${msg.duration}s, ${msg.num_segments} segments`;
  } else if (msg.type === 'error') {
    setStatus('error');
    stopMicCapture();
    pendingHumanRole = null;
    clearInterval(elapsedTimer);
  }
}

function appendToken(role, text, time) {
  const container = document.getElementById('transcript-tokens');
  let line = container.querySelector(`.token-line[data-role="${role}"]:last-child`);
  const lastLine = container.lastElementChild;
  if (!lastLine || lastLine.dataset.role !== role) {
    line = document.createElement('div');
    line.className = 'token-line';
    line.dataset.role = role;
    const roleClass = role === 'student' ? 'role-student' : 'role-tutor';
    line.innerHTML = `<span class="time">[${time.toFixed(1)}s]</span><span class="${roleClass}">${role}: </span><span class="text"></span>`;
    container.appendChild(line);
  }
  line.querySelector('.text').textContent += text;
  container.scrollTop = container.scrollHeight;
}

function appendSegment(role, text, start, end) {
  const container = document.getElementById('transcript-segments');
  const wrapper = container.querySelector('.segment-wrapper') || (() => {
    const w = document.createElement('div');
    w.className = 'segment-wrapper';
    container.appendChild(w);
    return w;
  })();
  const div = document.createElement('div');
  div.className = `segment ${role}`;
  div.innerHTML = `<div class="seg-header">${role} [${start.toFixed(1)}s – ${end.toFixed(1)}s]</div>${escapeHtml(text)}`;
  wrapper.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function startConversation() {
  const sel = document.getElementById('question-select');
  const pid = sel.value;
  if (!pid) return;
  document.getElementById('transcript-tokens').innerHTML = '';
  document.getElementById('transcript-segments').innerHTML = '';
  renderMonitorQuestionInfo();
  connectWS();
  currentPid = pid;
  startTime = Date.now();
  fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      pid: pid,
      tutor_backend: getBackend('tutor'),
      student_backend: getBackend('student'),
    })
  }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }
    // Re-render the info panel with the authoritative per-role flags from
    // the server so the badges reflect what was actually sent to the model.
    renderMonitorQuestionInfo({
      tutor_backend: data.tutor_backend,
      student_backend: data.student_backend,
      tutor_uses_image: data.tutor_uses_image,
      student_uses_image: data.student_uses_image,
    });
    setStatus('starting');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    elapsedTimer = setInterval(updateElapsed, 1000);
  });
}

function stopConversation() {
  fetch('/api/stop', {method: 'POST'});
  stopMicCapture();
  pendingHumanRole = null;
}

function updateElapsed() {
  if (!startTime) return;
  const secs = ((Date.now() - startTime) / 1000).toFixed(0);
  document.getElementById('elapsed').textContent = `${secs}s`;
}

function setStatus(state) {
  const badge = document.getElementById('status-badge');
  badge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
  badge.className = 'status-badge';
  if (state === 'running') badge.classList.add('status-running');
  else if (state === 'starting') badge.classList.add('status-starting');
  else badge.classList.add('status-idle');
  if (state === 'done' || state === 'idle' || state === 'error') {
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
  }
  // Lock the run dropdown while anything but idle/done — switching dataset
  // mid-recording would orphan the in-flight save.
  setRunPickerLocked(state === 'running' || state === 'starting');
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${tab}"]`).classList.add('active');
  document.getElementById('transcript-tokens').style.display = tab === 'tokens' ? 'block' : 'none';
  document.getElementById('transcript-segments').style.display = tab === 'segments' ? 'block' : 'none';
}

connectWS();

// ---- Backend Selection + Remote Server Control ----
let cloudBackends = new Set();
const serverState = {
  tutor:   { polling: null, attempts: 0, probing: false },
  student: { polling: null, attempts: 0, probing: false },
};

function getBackend(role) {
  return document.getElementById(role + '-backend').value;
}

function isCloud(backend) {
  return cloudBackends.has(backend);
}

function setDot(role, color) {
  const dot = document.getElementById(role + '-dot');
  if (!dot) return;
  dot.classList.remove('red', 'yellow', 'green');
  if (color === 'red' || color === 'yellow' || color === 'green') {
    dot.classList.add(color);
  }
}

function hasNoRemoteServer(backend) {
  return isCloud(backend) || backend === 'human';
}

function applyBackendMutex() {
  for (const role of ['tutor', 'student']) {
    const other = role === 'tutor' ? 'student' : 'tutor';
    const otherSel = document.getElementById(other + '-backend');
    if (!otherSel) continue;
    const humanOpt = otherSel.querySelector('option[value="human"]');
    if (!humanOpt) continue;
    humanOpt.disabled = getBackend(role) === 'human';
  }
}

function renderServerWidget(role) {
  const widget = document.getElementById(role + '-server');
  const backend = getBackend(role);
  stopAutoPoll(role);
  if (hasNoRemoteServer(backend)) {
    widget.classList.add('hidden');
    widget.innerHTML = '';
    return;
  }
  widget.classList.remove('hidden');
  widget.innerHTML =
    '<span class="server-dot" id="' + role + '-dot" title="Remote server status"></span>' +
    '<button class="server-btn" id="' + role + '-start-btn" onclick="startRemoteServer(\'' + role + '\')">Start Server</button>' +
    '<button class="server-btn" id="' + role + '-check-btn" onclick="probeServer(\'' + role + '\')">Check</button>' +
    '<button class="server-btn kill" id="' + role + '-kill-btn" onclick="killRemoteServer(\'' + role + '\')">Kill</button>';
  setDot(role, null);
}

async function probeServer(role) {
  const backend = getBackend(role);
  if (hasNoRemoteServer(backend)) return true;
  const s = serverState[role];
  // Guard against overlapping probes when the auto-poll interval fires before
  // the previous probe has returned (e.g. a timed-out TCP connect).
  if (s.probing) return false;
  s.probing = true;
  setDot(role, 'yellow');
  try {
    const resp = await fetch('/api/remote/health?role=' + role + '&backend=' + encodeURIComponent(backend));
    const data = await resp.json();
    const reachable = !!data.reachable;
    setDot(role, reachable ? 'green' : 'red');
    return reachable;
  } catch (e) {
    console.error('probeServer failed', e);
    setDot(role, 'red');
    return false;
  } finally {
    s.probing = false;
  }
}

function stopAutoPoll(role) {
  const s = serverState[role];
  if (s.polling) {
    clearInterval(s.polling);
    s.polling = null;
  }
  s.attempts = 0;
}

function startAutoPoll(role) {
  stopAutoPoll(role);
  const s = serverState[role];
  const MAX_ATTEMPTS = 40; // 3s * 40 = ~2 min
  s.polling = setInterval(async () => {
    s.attempts++;
    const reachable = await probeServer(role);
    if (reachable || s.attempts >= MAX_ATTEMPTS) {
      stopAutoPoll(role);
      const btn = document.getElementById(role + '-start-btn');
      if (btn) btn.disabled = false;
    }
  }, 3000);
}

async function startRemoteServer(role) {
  const backend = getBackend(role);
  if (hasNoRemoteServer(backend)) return;
  const btn = document.getElementById(role + '-start-btn');
  if (btn) btn.disabled = true;
  setDot(role, 'yellow');
  try {
    const resp = await fetch('/api/remote/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role: role, backend: backend}),
    });
    const data = await resp.json();
    if (data.error || data.ok === false) {
      alert('Failed to start server: ' + (data.error || data.stderr || 'unknown error'));
      setDot(role, 'red');
      if (btn) btn.disabled = false;
      return;
    }
    if (data.already_running) {
      setDot(role, 'green');
      if (btn) btn.disabled = false;
      return;
    }
    // SSH fired the nohup command; model takes ~1-2 min to load. Keep the
    // button disabled until the auto-poll succeeds or times out.
    startAutoPoll(role);
  } catch (e) {
    alert('Failed to start server: ' + e.message);
    setDot(role, 'red');
    if (btn) btn.disabled = false;
  }
}

async function killRemoteServer(role) {
  const backend = getBackend(role);
  if (hasNoRemoteServer(backend)) return;
  if (!confirm('Kill the ' + backend + ' server for ' + role + '? The other role\'s instance is unaffected.')) return;
  const killBtn  = document.getElementById(role + '-kill-btn');
  const startBtn = document.getElementById(role + '-start-btn');
  if (killBtn) killBtn.disabled = true;
  stopAutoPoll(role);
  setDot(role, 'yellow');
  let errorMsg = null;
  try {
    const resp = await fetch('/api/remote/kill', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role: role, backend: backend}),
    });
    const data = await resp.json();
    if (data.error || data.ok === false) {
      errorMsg = data.error || data.stderr || 'unknown error';
    }
  } catch (e) {
    errorMsg = e.message;
  }
  if (errorMsg) alert('Failed to kill server: ' + errorMsg);
  if (killBtn)  killBtn.disabled = false;
  if (startBtn) startBtn.disabled = false;
  // Give the OS a moment to release the listening socket, then probe to
  // reflect the true state — whether the kill succeeded or we hit an error.
  setTimeout(() => probeServer(role), 500);
}

async function loadBackendDefaults() {
  let data = {};
  try {
    const resp = await fetch('/api/defaults');
    data = await resp.json();
  } catch (e) {
    console.error('loadBackendDefaults failed', e);
  }
  const choices = data.backends || [];
  cloudBackends = new Set(data.cloud_backends || []);
  IMAGE_CAPABLE_BACKENDS = new Set(data.image_capable_backends || []);
  for (const role of ['tutor', 'student']) {
    const sel = document.getElementById(role + '-backend');
    sel.innerHTML = '';
    choices.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = b;
      sel.appendChild(opt);
    });
    const preferred = data[role + '_backend'];
    sel.value = choices.includes(preferred) ? preferred : (choices[0] || '');
    sel.addEventListener('change', () => {
      applyBackendMutex();
      renderServerWidget(role);
      probeServer(role);
      renderMonitorQuestionInfo();
    });
    renderServerWidget(role);
  }
  applyBackendMutex();
  await Promise.all([probeServer('tutor'), probeServer('student')]);
}

loadBackendDefaults();

// ---- Live Audio Playback Engine ----
const SAMPLE_RATE = 24000;
let audioCtx = null;
const audioState = {
  tutor:   { gain: null, nextTime: 0, sources: new Set() },
  student: { gain: null, nextTime: 0, sources: new Set() },
};

function initAudio() {
  if (audioCtx) return;
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  for (const role of ['tutor', 'student']) {
    const gain = audioCtx.createGain();
    gain.gain.value = document.getElementById('vol-' + role).value / 100;
    gain.connect(audioCtx.destination);
    audioState[role].gain = gain;
    audioState[role].nextTime = 0;
  }
  document.getElementById('vol-tutor').oninput = (e) => {
    if (audioState.tutor.gain) audioState.tutor.gain.gain.value = e.target.value / 100;
  };
  document.getElementById('vol-student').oninput = (e) => {
    if (audioState.student.gain) audioState.student.gain.gain.value = e.target.value / 100;
  };
}

function handleAudioFrame(buffer) {
  try {
    if (!document.getElementById('audio-enabled').checked) return;
    if (!audioCtx) initAudio();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const view = new DataView(buffer);
    const roleTag = view.getUint8(0);
    const role = roleTag === 1 ? 'tutor' : 'student';
    const numBytes = buffer.byteLength - 1;
    const numSamples = numBytes / 4;
    if (numSamples <= 0) return;
    const aligned = new ArrayBuffer(numBytes);
    new Uint8Array(aligned).set(new Uint8Array(buffer, 1));
    const pcm = new Float32Array(aligned);
    const audioBuf = audioCtx.createBuffer(1, numSamples, SAMPLE_RATE);
    audioBuf.getChannelData(0).set(pcm);
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuf;
    source.connect(audioState[role].gain);
    const state = audioState[role];
    const now = audioCtx.currentTime;
    if (state.nextTime < now) state.nextTime = now + 0.05;
    source.start(state.nextTime);
    state.nextTime += audioBuf.duration;
    // Track the scheduled source so a barge-in can stop it early.
    state.sources.add(source);
    source.onended = () => state.sources.delete(source);
    const ind = document.getElementById('ind-' + role);
    ind.classList.add('active');
    setTimeout(() => ind.classList.remove('active'), 100);
  } catch (err) {
    console.error('Audio playback error:', err);
  }
}

function flushRoleAudio(role) {
  const state = audioState[role];
  if (!state) return;
  for (const src of state.sources) {
    src.onended = null;
    try { src.stop(); } catch (e) {}
    try { src.disconnect(); } catch (e) {}
  }
  state.sources.clear();
  state.nextTime = 0;
  console.log('Flushed scheduled audio for', role);
}

document.addEventListener('click', () => { if (!audioCtx) initAudio(); }, { once: true });

// ---- Human Mic Capture ----
let pendingHumanRole = null;
const micState = {
  role: null,
  stream: null,
  source: null,
  node: null,
  silentSink: null,
  originalGain: null,
};

const MIC_WORKLET_SRC = `
class MicFramer extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(1920);
    this._pos = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    let i = 0;
    while (i < ch.length) {
      const n = Math.min(1920 - this._pos, ch.length - i);
      this._buf.set(ch.subarray(i, i + n), this._pos);
      this._pos += n;
      i += n;
      if (this._pos >= 1920) {
        const out = this._buf;
        this.port.postMessage(out, [out.buffer]);
        this._buf = new Float32Array(1920);
        this._pos = 0;
      }
    }
    return true;
  }
}
registerProcessor('mic-framer', MicFramer);
`;

async function startMicCapture(role) {
  if (micState.role) return;
  if (!role) return;
  try {
    if (!audioCtx) initAudio();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    micState.role = role;
    micState.stream = stream;

    // Mute playback of the human's own role so AI audio echoed via the
    // role's gain node doesn't feed back into the mic (browser AEC helps
    // but headphones are strongly recommended).
    const g = audioState[role] && audioState[role].gain;
    if (g) {
      micState.originalGain = g.gain.value;
      g.gain.value = 0;
    }

    const source = audioCtx.createMediaStreamSource(stream);
    micState.source = source;

    let node;
    if (typeof AudioWorkletNode !== 'undefined' && audioCtx.audioWorklet) {
      const blob = new Blob([MIC_WORKLET_SRC], {type: 'application/javascript'});
      await audioCtx.audioWorklet.addModule(URL.createObjectURL(blob));
      node = new AudioWorkletNode(audioCtx, 'mic-framer');
      node.port.onmessage = (ev) => sendMicFrame(ev.data);
      source.connect(node);
    } else {
      // Safari/legacy fallback: ScriptProcessor must connect to destination
      // to fire onaudioprocess. Route it through a muted gain so it stays silent.
      node = audioCtx.createScriptProcessor(1024, 1, 1);
      const accum = new Float32Array(1920);
      let pos = 0;
      node.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        let i = 0;
        while (i < ch.length) {
          const n = Math.min(1920 - pos, ch.length - i);
          accum.set(ch.subarray(i, i + n), pos);
          pos += n;
          i += n;
          if (pos >= 1920) {
            sendMicFrame(new Float32Array(accum));
            pos = 0;
          }
        }
      };
      source.connect(node);
      const silent = audioCtx.createGain();
      silent.gain.value = 0;
      node.connect(silent);
      silent.connect(audioCtx.destination);
      micState.silentSink = silent;
    }
    micState.node = node;

    const ind = document.getElementById('ind-' + role);
    if (ind) ind.classList.add('active');
    try { ws && ws.readyState === 1 && ws.send(JSON.stringify({type: 'mic_ready', role})); } catch (e) {}
    console.log('Mic capture started for', role);
  } catch (err) {
    console.error('startMicCapture failed', err);
    stopMicCapture();
    try {
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({type: 'mic_error', message: String(err && err.message || err)}));
      }
    } catch (e) {}
    alert('Microphone capture failed: ' + (err.message || err) + '\nThe conversation will be stopped.');
    fetch('/api/stop', {method: 'POST'});
  }
}

function sendMicFrame(float32) {
  if (!ws || ws.readyState !== 1) return;
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = float32[i];
    if (s > 1) s = 1; else if (s < -1) s = -1;
    int16[i] = Math.round(s * 32767);
  }
  ws.send(int16.buffer);
}

function stopMicCapture() {
  const role = micState.role;
  if (micState.node) {
    try { micState.node.disconnect(); } catch (e) {}
    micState.node = null;
  }
  if (micState.silentSink) {
    try { micState.silentSink.disconnect(); } catch (e) {}
    micState.silentSink = null;
  }
  if (micState.source) {
    try { micState.source.disconnect(); } catch (e) {}
    micState.source = null;
  }
  if (micState.stream) {
    try { micState.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    micState.stream = null;
  }
  if (role && audioState[role] && audioState[role].gain && micState.originalGain !== null) {
    audioState[role].gain.gain.value = micState.originalGain;
  }
  micState.originalGain = null;
  micState.role = null;
  if (role) {
    const ind = document.getElementById('ind-' + role);
    if (ind) ind.classList.remove('active');
  }
}

// ---- History Mode ----
let selectedConversation = null;
const historyAudio = {
  tutor:   { buffer: null, source: null, gain: null },
  student: { buffer: null, source: null, gain: null },
};
const pb = {
  playing: false,
  startedAt: 0,
  pausedAt: 0,
  duration: 0,
  rafId: null,
  rate: 1.0,
};
// Track which segments have been revealed in the live tokens view
let revealedSegIdx = -1;

async function loadHistoryList() {
  const resp = await fetch('/api/history');
  const conversations = await resp.json();
  const container = document.getElementById('history-list');
  if (conversations.length === 0) {
    container.innerHTML = '<div class="empty-state" style="height:200px">No saved conversations</div>';
    return;
  }
  container.innerHTML = conversations.map(c => `
    <div class="history-item" data-idx="${c.idx}" onclick="selectConversation(${c.idx})">
      <div style="font-weight:600">PID ${c.pid}${(c.attempt_index ?? 0) > 0 ? ` <span style="font-weight:400;color:var(--muted);font-size:11px">try ${c.attempt_index}</span>` : ''} <span style="font-weight:400;color:var(--muted);font-size:11px">#${c.idx}</span></div>
      <div style="font-size:12px;color:var(--muted);margin-top:2px;
           overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(c.question)}</div>
      <div style="font-size:11px;margin-top:4px;color:var(--muted)">
        ${c.duration.toFixed(1)}s · ${c.num_segments} segments
        ${(c.has_tutor_audio || c.has_student_audio) ? ' · has audio' : ''}
        ${c.tutor_backend ? '<br>' + c.tutor_backend + ' / ' + c.student_backend : ''}
      </div>
    </div>
  `).join('');
}

async function selectConversation(idx) {
  stopHistoryPlayback();

  document.querySelectorAll('.history-item').forEach(el => {
    el.classList.toggle('active', el.dataset.idx === String(idx));
  });

  // Fetch full detail
  const resp = await fetch(`/api/history/${idx}`);
  selectedConversation = await resp.json();

  // Show question info below history list
  const infoEl = document.getElementById('history-question-info');
  infoEl.innerHTML = buildQuestionInfoHtml(selectedConversation);

  // Render segments
  const container = document.getElementById('transcript-segments');
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'segment-wrapper';
  selectedConversation.segments.forEach((seg, i) => {
    const div = document.createElement('div');
    div.className = `segment ${seg.role}`;
    div.id = `seg-${i}`;
    div.dataset.start = seg.start_time;
    div.dataset.end = seg.end_time;
    div.innerHTML = `<div class="seg-header">${seg.role} [${seg.start_time.toFixed(1)}s – ${seg.end_time.toFixed(1)}s]</div>${escapeHtml(seg.text)}`;
    div.style.cursor = 'pointer';
    div.onclick = () => seekToTime(seg.start_time);
    wrapper.appendChild(div);
  });
  container.appendChild(wrapper);

  // Reset live tokens view
  resetLiveTokens();

  // Load audio
  const hasAudio = selectedConversation.has_tutor_audio || selectedConversation.has_student_audio;
  pb.duration = selectedConversation.duration;
  pb.pausedAt = 0;
  updateProgressUI(0);

  if (selectedConversation.has_combined_audio) {
    const combinedAudio = document.getElementById('combined-audio');
    combinedAudio.pause();
    combinedAudio.src = `/api/audio/${idx}/combined`;
    combinedAudio.load();
    document.getElementById('combined-controls').style.display = '';
  } else {
    resetCombinedAudio();
  }

  if (hasAudio) {
    document.getElementById('btn-play').disabled = true;
    document.getElementById('btn-play').textContent = 'Loading...';
    await loadHistoryAudio(idx, selectedConversation);
    document.getElementById('btn-play').disabled = false;
    document.getElementById('btn-play').textContent = 'Play';
  } else {
    document.getElementById('btn-play').disabled = true;
    document.getElementById('btn-play').textContent = 'No Audio';
  }

  // Expose to alignment.js so it can swap conv.segments in place.
  window.selectedConversation = selectedConversation;

  if (typeof window.loadEvalScores === 'function') {
    window.loadEvalScores(idx);
  }
  if (typeof window.loadAligned === 'function') {
    window.loadAligned(idx);
  }
  if (typeof window.loadVap === 'function') {
    window.loadVap(idx);
  }
}

// ---- Live Tokens (history playback) ----
function resetLiveTokens() {
  revealedSegIdx = -1;
  const container = document.getElementById('transcript-tokens');
  container.innerHTML = '';
}

function updateLiveTokens(currentTime) {
  if (!selectedConversation) return;
  const container = document.getElementById('transcript-tokens');
  const segs = selectedConversation.segments;

  while (revealedSegIdx + 1 < segs.length && segs[revealedSegIdx + 1].start_time <= currentTime) {
    revealedSegIdx++;
    const seg = segs[revealedSegIdx];
    // Find or create current line for this role (group consecutive same-role)
    const lastLine = container.lastElementChild;
    const roleClass = seg.role === 'student' ? 'role-student' : 'role-tutor';

    if (!lastLine || lastLine.dataset.role !== seg.role) {
      const line = document.createElement('div');
      line.className = 'token-line';
      line.dataset.role = seg.role;
      line.innerHTML = `<span class="time">[${seg.start_time.toFixed(1)}s]</span><span class="${roleClass}">${seg.role}: </span><span class="text">${escapeHtml(seg.text)}</span>`;
      container.appendChild(line);
    } else {
      lastLine.querySelector('.text').textContent += ' ' + seg.text;
    }
    container.scrollTop = container.scrollHeight;
  }
}

function revealLiveTokensUpTo(currentTime) {
  // Reset and re-reveal up to the given time (used on seek)
  resetLiveTokens();
  updateLiveTokens(currentTime);
}

// ---- Audio Loading ----
function initHistoryAudio() {
  if (!audioCtx) initAudio();
  for (const role of ['tutor', 'student']) {
    if (!historyAudio[role].gain) {
      const gain = audioCtx.createGain();
      gain.gain.value = document.getElementById('pb-vol-' + role).value / 100;
      gain.connect(audioCtx.destination);
      historyAudio[role].gain = gain;
    }
  }
}

// Wire playback volume sliders once
document.getElementById('pb-vol-tutor').oninput = (e) => {
  if (historyAudio.tutor.gain) historyAudio.tutor.gain.gain.value = e.target.value / 100;
};
document.getElementById('pb-vol-student').oninput = (e) => {
  if (historyAudio.student.gain) historyAudio.student.gain.gain.value = e.target.value / 100;
};

async function loadHistoryAudio(idx, conv) {
  initHistoryAudio();

  for (const role of ['tutor', 'student']) {
    historyAudio[role].buffer = null;
  }

  // Load audio files in parallel. Per-role WAVs are conversation-aligned
  // (same length as combined.wav, both start at conversation t=0), so audio
  // time === conversation time and no per-role offset is needed.
  const loads = [];
  for (const role of ['tutor', 'student']) {
    if (conv[`has_${role}_audio`]) {
      loads.push(
        fetch(`/api/audio/${idx}/${role}`)
          .then(r => r.arrayBuffer())
          .then(buf => audioCtx.decodeAudioData(buf))
          .then(decoded => {
            historyAudio[role].buffer = decoded;
            console.log(`${role} audio: ${decoded.duration.toFixed(1)}s`);
          })
          .catch(err => console.warn(`Failed to load ${role} audio:`, err))
      );
    }
  }
  await Promise.all(loads);
}

// ---- Playback Controls ----
function togglePlayback() {
  if (pb.playing) {
    pauseHistoryPlayback();
  } else {
    playHistoryAudio();
  }
}

function playHistoryAudio() {
  if (!audioCtx) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();

  // Stop any lingering sources first
  stopAllSources();

  const convTime = pb.pausedAt;  // conversation-relative time
  const now = audioCtx.currentTime;

  for (const role of ['tutor', 'student']) {
    if (historyAudio[role].buffer) {
      const source = audioCtx.createBufferSource();
      source.buffer = historyAudio[role].buffer;
      source.playbackRate.value = pb.rate;
      source.connect(historyAudio[role].gain);
      if (convTime < source.buffer.duration) {
        source.start(now, Math.max(0, convTime));
      }
      historyAudio[role].source = source;
    }
  }

  pb.playing = true;
  pb.startedAt = now - (convTime / pb.rate);
  document.getElementById('btn-play').textContent = 'Pause';
  startPlaybackLoop();
}

function stopAllSources() {
  for (const role of ['tutor', 'student']) {
    if (historyAudio[role].source) {
      historyAudio[role].source.onended = null;
      try { historyAudio[role].source.stop(); } catch(e) {}
      historyAudio[role].source = null;
    }
  }
}

function pauseHistoryPlayback() {
  if (!pb.playing) return;
  const currentTime = (audioCtx.currentTime - pb.startedAt) * pb.rate;
  pb.pausedAt = Math.min(currentTime, pb.duration);
  stopAllSources();
  pb.playing = false;
  cancelAnimationFrame(pb.rafId);
  document.getElementById('btn-play').textContent = 'Play';
}

function stopHistoryPlayback() {
  pauseHistoryPlayback();
  pb.pausedAt = 0;
  updateProgressUI(0);
  highlightSegments(-1);
  resetLiveTokens();
}

function resetCombinedAudio() {
  const ca = document.getElementById('combined-audio');
  ca.pause();
  ca.removeAttribute('src');
  ca.load();
  document.getElementById('combined-controls').style.display = 'none';
}

function seekToTime(t) {
  const wasPlaying = pb.playing;
  if (wasPlaying) pauseHistoryPlayback();
  pb.pausedAt = Math.max(0, Math.min(t, pb.duration));
  updateProgressUI(pb.pausedAt);
  highlightSegments(pb.pausedAt);
  revealLiveTokensUpTo(pb.pausedAt);
  if (wasPlaying) playHistoryAudio();
}
window.seekToTime = seekToTime;

function setPlaybackSpeed(val) {
  const newRate = parseFloat(val);
  if (pb.playing) {
    const currentTime = (audioCtx.currentTime - pb.startedAt) * pb.rate;
    pauseHistoryPlayback();
    pb.rate = newRate;
    pb.pausedAt = Math.min(currentTime, pb.duration);
    playHistoryAudio();
  } else {
    pb.rate = newRate;
  }
}

function onPlaybackEnd() {
  stopAllSources();
  pb.playing = false;
  pb.pausedAt = 0;
  cancelAnimationFrame(pb.rafId);
  document.getElementById('btn-play').textContent = 'Play';
  updateProgressUI(pb.duration);
  highlightSegments(-1);
}

function startPlaybackLoop() {
  function tick() {
    if (!pb.playing) return;
    const currentTime = (audioCtx.currentTime - pb.startedAt) * pb.rate;
    if (currentTime >= pb.duration) {
      onPlaybackEnd();
      return;
    }
    updateProgressUI(currentTime);
    highlightSegments(currentTime);
    updateLiveTokens(currentTime);
    pb.rafId = requestAnimationFrame(tick);
  }
  pb.rafId = requestAnimationFrame(tick);
}

function updateProgressUI(t) {
  const pct = pb.duration > 0 ? (t / pb.duration) * 100 : 0;
  document.getElementById('progress-fill').style.width = Math.min(pct, 100) + '%';
  document.getElementById('time-display').textContent =
    formatTime(Math.min(t, pb.duration)) + ' / ' + formatTime(pb.duration);
}

function highlightSegments(currentTime) {
  document.querySelectorAll('#transcript-segments .segment').forEach(el => {
    const start = parseFloat(el.dataset.start);
    const end = parseFloat(el.dataset.end);
    if (isNaN(start)) return;
    const isActive = currentTime >= start && currentTime <= end;
    const wasActive = el.classList.contains('highlight');
    el.classList.toggle('highlight', isActive);
    if (isActive && !wasActive) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
  if (typeof window.highlightAlignedWords === 'function') {
    window.highlightAlignedWords(currentTime);
  }
  if (typeof window.vapUpdateCursor === 'function') {
    window.vapUpdateCursor(currentTime);
  }
}

// Progress bar click-to-seek
document.getElementById('progress-bar').addEventListener('click', (e) => {
  if (!selectedConversation) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  seekToTime(fraction * pb.duration);
});
</script>
<!--EVAL_SCRIPT-->
<!--ALIGN_SCRIPT-->
<!--VAP_SCRIPT-->
</body>
</html>
"""
