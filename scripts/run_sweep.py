"""Sequential sweep runner: invoke `python -m duplex` for each row in a YAML/JSON config.

Each child run gets its own `--run-name` and is independently resumable. Killing
this script mid-sweep leaves the in-progress child's run dir in a state that the
next sweep invocation will resume.

Example config (YAML):

    runs:
      - name: gpt_v1
        tutor_backend: gpt-realtime
        student_backend: gpt-realtime
        sample: 50
        seed: 0
        attempts: 5
        concurrency: 8
      - name: pp_v1
        tutor_backend: personaplex
        student_backend: personaplex
        sample: 50
        seed: 0
        attempts: 5

Usage:
    uv run python scripts/run_sweep.py experiments.yaml
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Keys whose CLI flag uses the same kebab-case as the snake_case config key.
# Anything not in this map is auto-derived: snake_case -> --kebab-case.
_FLAG_OVERRIDES: dict[str, str] = {}


def _to_flag(key: str) -> str:
    return _FLAG_OVERRIDES.get(key, "--" + key.replace("_", "-"))


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as e:
            raise SystemExit(
                "PyYAML is required for YAML configs. Install with `uv add pyyaml`."
            ) from e
        return yaml.safe_load(text)
    return json.loads(text)


def _build_command(run: dict[str, Any]) -> list[str]:
    name = run.get("name")
    if not name:
        raise SystemExit(f"Each sweep row needs a `name` field; got {run!r}.")

    cmd = [sys.executable, "-m", "duplex", "--run-name", str(name)]
    for key, value in run.items():
        if key == "name":
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(_to_flag(key))
            continue
        cmd.extend([_to_flag(key), str(value)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to sweep config (YAML/JSON)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run, then exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(args.config)
    runs = config.get("runs", [])
    if not runs:
        raise SystemExit(f"No `runs:` entries found in {args.config}")

    logger.info("Sweep: %d runs from %s", len(runs), args.config)
    for i, run in enumerate(runs, 1):
        cmd = _build_command(run)
        logger.info("[%d/%d] %s", i, len(runs), " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(
                "[%d/%d] run %s exited with code %d; continuing.",
                i,
                len(runs),
                run.get("name"),
                result.returncode,
            )

    logger.info("Sweep done.")


if __name__ == "__main__":
    main()
