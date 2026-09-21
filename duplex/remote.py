"""Remote server control for self-hosted duplex backends.

Responsible for:
- Mapping each ``(backend, role)`` pair to its ``ssh`` start command and the
  host/port to probe for readiness.
- Starting the remote process via ``ssh ucsc_lab_sv11 '<detached nohup cmd>'``.
  A cheap TCP probe runs first — if the port is already listening, we skip SSH
  entirely and report ``already_running``.
- A ``probe_tcp`` helper used by both the pre-check above and the UI's
  ``/api/remote/health`` endpoint.

Only PersonaPlex and MoshiVis need a remote process; ``gpt-realtime`` and
``gemini-live`` call cloud APIs and are deliberately absent from the table.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


REMOTE_SSH_HOST = "ucsc_lab_sv11"
PROBE_HOST = "100.116.140.1"
CLOUD_BACKENDS = {"gpt-realtime", "gemini-live"}


def _personaplex_target(role: str, port: int, gpu: int) -> dict[str, object]:
    return {
        "probe_host": PROBE_HOST,
        "probe_port": port,
        "remote_cmd": (
            f"cd ~/repo/personaplex/moshi && "
            f"CUDA_VISIBLE_DEVICES={gpu} nohup .venv/bin/python -m moshi.server "
            f"--host 0.0.0.0 --port {port} "
            f"> /tmp/moshi_{role}.log 2>&1 & disown; exit 0"
        ),
        "kill_cmd": f"pkill -f 'moshi.server --host 0.0.0.0 --port {port}'",
    }


def _moshivis_target(role: str, port: int, gpu: int) -> dict[str, object]:
    return {
        "probe_host": PROBE_HOST,
        "probe_port": port,
        "remote_cmd": (
            f"cd ~/repo/moshivis/kyuteye_pt && "
            f"CUDA_VISIBLE_DEVICES={gpu} nohup ~/.local/bin/uv run server "
            f"configs/moshika-vis.yaml --host 0.0.0.0 --port {port} --ssl False "
            f"> /tmp/moshivis_{role}.log 2>&1 & disown; exit 0"
        ),
        "kill_cmd": f"pkill -f 'moshika-vis.yaml --host 0.0.0.0 --port {port}'",
    }


# Tutor runs on GPU 0, student on GPU 1, each on its own port so a per-port
# ``pkill`` only takes out one role's instance.
REMOTE_TARGETS: dict[tuple[str, str], dict[str, object]] = {
    ("personaplex", "tutor"): _personaplex_target("tutor", port=8998, gpu=0),
    ("personaplex", "student"): _personaplex_target("student", port=8999, gpu=1),
    ("moshivis", "tutor"): _moshivis_target("tutor", port=8088, gpu=0),
    ("moshivis", "student"): _moshivis_target("student", port=8089, gpu=1),
}


def has_remote_server(backend: str) -> bool:
    """Return True if this backend needs a local/remote server (not a cloud API)."""
    return backend not in CLOUD_BACKENDS


def get_target(backend: str, role: str) -> dict[str, object] | None:
    """Return the REMOTE_TARGETS entry for ``(backend, role)``, or None."""
    return REMOTE_TARGETS.get((backend, role))


async def probe_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    """Cheap readiness check: can we open a TCP connection?

    We deliberately don't speak WebSocket here — PersonaPlex/MoshiVis servers
    hold a per-session lock and a real handshake would block the next real
    conversation.
    """
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    return True


async def _run_ssh(cmd: str, timeout: float = 20.0) -> dict:
    """Run ``ssh REMOTE_SSH_HOST <cmd>`` and return returncode + stdout + stderr.

    On timeout the subprocess is killed so it doesn't linger as a zombie.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o",
            "BatchMode=yes",
            REMOTE_SSH_HOST,
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "ssh binary not found on this machine"}

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        return {"ok": False, "error": f"ssh command timed out after {timeout:.0f}s"}

    return {
        "ok": True,
        "returncode": proc.returncode,
        "stdout": stdout_b.decode("utf-8", errors="replace").strip(),
        "stderr": stderr_b.decode("utf-8", errors="replace").strip(),
    }


async def start_remote_server(backend: str, role: str) -> dict:
    """Start the remote model server for ``(backend, role)``.

    Flow:
    1. Probe the target port. If reachable, return ``already_running=True`` —
       no SSH is attempted, which avoids stacking duplicate processes.
    2. Otherwise ``ssh REMOTE_SSH_HOST '<nohup ... & disown>'``. SSH itself
       returns in seconds; the model takes 1–2 min to load and will show up on
       the port later. The caller is expected to poll ``probe_tcp`` to detect
       readiness.

    Returns a dict: ``{ok, already_running, stdout?, stderr?, error?}``.
    """
    target = get_target(backend, role)
    if target is None:
        return {
            "ok": False,
            "already_running": False,
            "error": f"No remote target registered for {backend!r}/{role!r}",
        }

    host = str(target["probe_host"])
    port = int(target["probe_port"])  # type: ignore[arg-type]

    if await probe_tcp(host, port):
        logger.info(
            "Remote %s/%s already listening on %s:%d — skipping SSH start.",
            backend,
            role,
            host,
            port,
        )
        return {"ok": True, "already_running": True}

    logger.info(
        "SSHing to %s to start %s/%s (target %s:%d)",
        REMOTE_SSH_HOST,
        backend,
        role,
        host,
        port,
    )
    result = await _run_ssh(str(target["remote_cmd"]))
    if "error" in result:
        return {"ok": False, "already_running": False, "error": result["error"]}
    return {
        "ok": result["returncode"] == 0,
        "already_running": False,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


async def kill_remote_server(backend: str, role: str) -> dict:
    """Stop the remote model server for ``(backend, role)``.

    Uses ``pkill -f <narrow pattern>`` over SSH. The pattern matches on the
    process's own ``--port N`` argument so the student instance survives when
    you kill the tutor (and vice versa). Exit code 1 from ``pkill`` means no
    process matched — we treat that as success too (nothing was running).
    """
    target = get_target(backend, role)
    if target is None:
        return {
            "ok": False,
            "error": f"No remote target registered for {backend!r}/{role!r}",
        }

    logger.info(
        "SSHing to %s to stop %s/%s (pattern: %s)",
        REMOTE_SSH_HOST,
        backend,
        role,
        target["kill_cmd"],
    )
    result = await _run_ssh(str(target["kill_cmd"]))
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    rc = result["returncode"]
    return {
        "ok": rc in (0, 1),
        "matched": rc == 0,
        "returncode": rc,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }
