"""Per-run directories: every independent run writes logs and results under runs/<run_id>/ and never
overwrites another run's files.

    RUN_ID=gate_v2 python s4ngram.py gate India Wn Wa Pn Cn

runs/<run_id>/
    config.json   argv, start time, git commit (+dirty flag), python
    logs/         stdout/stderr of the jobs (launch with `> runs/<run_id>/logs/<job>.out`)
    ...           whatever the script produces (gate/, output/, model.pkl, ...)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path(__file__).parent / "runs"


def run_id() -> str:
    rid = os.environ.get("RUN_ID")
    if not rid:
        raise SystemExit("Set RUN_ID (e.g. RUN_ID=gate_v2) so this run gets its own runs/<id>/ directory.")
    return rid


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent, timeout=20).stdout.strip()
    except Exception:
        return ""


def run_dir(rid: str | None = None, *, create: bool = True) -> Path:
    d = RUNS / (rid or run_id())
    if create:
        (d / "logs").mkdir(parents=True, exist_ok=True)
        cfg = d / "config.json"
        if not cfg.exists():
            cfg.write_text(json.dumps({
                "run_id": d.name, "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                "python": sys.executable,
            }, indent=2))
    return d


def safe_path(path: Path) -> Path:
    """Refuse to overwrite an existing result unless RUN_OVERWRITE=1, so a rerun cannot clobber a finished run."""
    if path.exists() and os.environ.get("RUN_OVERWRITE") != "1":
        raise SystemExit(f"{path} already exists; use a new RUN_ID (or RUN_OVERWRITE=1 to replace it).")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
