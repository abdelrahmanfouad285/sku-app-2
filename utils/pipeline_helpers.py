"""Shared pipeline helpers used by every entry point (CLI, Claude, local, UI).

Centralises things that used to be copy-pasted across main.py / main_claude.py /
main_local.py so behaviour stays consistent. Pure functions / IO only — no LLM,
no Excel, no Streamlit.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

# These paths must match the locations the Excel/CLI code already uses.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input_photos"
PROCESSED_DIR = INPUT_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_CSV_PATH = OUTPUT_DIR / "run_log.csv"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

# Default Ollama endpoint. Overridable via OLLAMA_BASE_URL env var in the
# CLI entry points; the Streamlit UI also exposes it as a sidebar input.
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434/v1"

# How much of an error/traceback to store in run_log.csv. Long stack traces
# bloat the log and most editors struggle past a few KB.
MAX_LOG_ERROR_CHARS = 2000


def ensure_dirs() -> None:
    for d in (INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_input_images() -> list[Path]:
    """All image files in INPUT_DIR, sorted by name for stable runs."""
    if not INPUT_DIR.exists():
        return []
    return sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def move_to_processed(path: Path) -> Path:
    """Move `path` to PROCESSED_DIR, never overwriting an existing file."""
    dest = PROCESSED_DIR / path.name
    counter = 1
    while dest.exists():
        dest = PROCESSED_DIR / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(dest))
    return dest


def append_run_log(filename: str, item_count: int, status: str, error: str = "") -> None:
    """Append one row to output/run_log.csv. Truncates long errors."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV_PATH.exists()
    err = (error or "").strip()
    if len(err) > MAX_LOG_ERROR_CHARS:
        err = err[: MAX_LOG_ERROR_CHARS - 3] + "..."
    with LOG_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "filename", "item_count", "status", "error"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            filename,
            item_count,
            status,
            err,
        ])


def filter_by_extension(files: Iterable[Path], allowed: set[str] = VALID_EXTS) -> list[Path]:
    """Return only files whose suffix (case-insensitive) is in `allowed`."""
    out: list[Path] = []
    for p in files:
        if p.is_file() and p.suffix.lower() in allowed:
            out.append(p)
    return out
