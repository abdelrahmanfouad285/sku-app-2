"""
SKU Data Entry Agent â€” shared core module.

Both CLI entry points (main_claude.py / main_local.py) and the Streamlit
UI (app.py) import from this module. It re-exports the common configuration,
helpers, and pipeline pieces so the call sites stay clean.

The active backend is determined by which entry point you run:
  - python main_claude.py    -> Anthropic Claude (cloud, paid)
  - python main_local.py     -> Ollama local model (free)
  - streamlit run app.py     -> Ollama local model (via sidebar dropdown)

Module-level constants and functions (re-exported here so app.py and the CLIs share them):
  INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR, LOG_CSV_PATH, EXCEL_COLUMNS,
  EXCEL_PATH, MODEL, OLLAMA_BASE_URL, ensure_dirs, list_input_images,
  process_image, append_run_log, move_to_processed,
  scan_barcodes, encode_image_as_base64, extract_with_ollama, append_product.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path



from excel_manager import (
    EXCEL_COLUMNS,
    EXCEL_PATH,
    append_product,
    append_rows_to_excel,
)
from image_utils import encode_image_as_base64
from llm import extract_with_ollama
from pipeline_helpers import (
    INPUT_DIR,
    LOG_CSV_PATH,
    OLLAMA_BASE_URL_DEFAULT,
    OUTPUT_DIR,
    PROCESSED_DIR,
    VALID_EXTS,
    append_run_log,
    ensure_dirs,
    list_input_images,
    move_to_processed,
)


# Default model â€” auto-overridden in the Streamlit UI from the sidebar.
MODEL = "llama3.2-vision:latest"

# Re-export for callers that still expect the old constant name.
OLLAMA_BASE_URL = OLLAMA_BASE_URL_DEFAULT


# Module-level logger â€” imported by app.py as core.log
log = logging.getLogger("sku_agent")
if not log.handlers:
    log.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_handler)
    log.propagate = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_NULL_LIKE = {"null", "none", "nil", "n/a", "na", "-"}


def _dedupe_items(items: list) -> list:
    """Collapse items that are identical across the key text fields.

    Weak vision models (e.g. llava:7b) sometimes repeat the same item many
    times. Identical (brand, product_name, flavor, qty) tuples are merged into
    one, with a note recording how many were collapsed. Non-dict entries pass
    through untouched.
    """
    seen: list[tuple] = []
    out: list = []
    dups = 0
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            continue
        key = (
            (it.get("brand") or "").strip().lower(),
            (it.get("product_name") or "").strip().lower(),
            (it.get("flavor_or_variant") or "").strip().lower(),
            (it.get("quantity_or_size") or "").strip().lower(),
        )
        if key in seen:
            dups += 1
            continue
        seen.append(key)
        out.append(it)
    if dups and out and isinstance(out[0], dict):
        existing = (out[0].get("notes") or "").strip()
        note = f"{dups} duplicate item(s) collapsed (model repetition)"
        out[0]["notes"] = (existing + "; " + note).strip("; ") if existing else note
    return out


def _coerce_null_strings(item: dict) -> dict:
    """Some local models (llava especially) return the literal string 'null'
    instead of JSON null for missing fields. Treat those, and obvious empties,
    as None so downstream code (validator, Excel, UI) doesn't store the word
    "null" in cells. Also normalises confidence to a known set.
    """
    out = dict(item)
    for k, v in list(out.items()):
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in _NULL_LIKE:
                out[k] = None
            else:
                out[k] = s
    conf = out.get("confidence")
    if isinstance(conf, str):
        c = conf.strip().lower()
        if c not in {"high", "medium", "low"}:
            out["confidence"] = "low"
        else:
            out["confidence"] = c
    return out

def process_image(
    client,
    path: Path,
    model: str | None = None,
    dry_run: bool = False,
) -> tuple[list[dict], int]:
    """
    Run the full pipeline for one image. Returns (rows, item_count) where rows
    is the list of row dicts that were appended (or would be appended) to the
    Excel file. `model` overrides the default MODEL constant (used by the UI
    dropdown). `dry_run=True` skips the Excel write and the move-to-processed
    step so the caller can inspect what would happen.
    """
    image_b64 = encode_image_as_base64(str(path))
    # Try the requested model, but if it's llama3.2-vision on an old Ollama that
    # doesn't know 'mllama', automatically fall back to llava:7b so the user
    # isn't blocked until they run winget upgrade.
    requested_model = model or MODEL
    try:
        if "Gemini" in str(requested_model) or client is None:
            from llm import extract_with_gemini
            parsed = extract_with_gemini("gemini-2.5-flash", image_b64)
        else:
            parsed = extract_with_ollama(
                client, requested_model, image_b64
            )
    except Exception as exc:
        msg = str(exc).lower()
        if "mllama" in msg or "unknown model architecture" in msg:
            fallback = "llava:7b"
            if fallback.lower() == (requested_model or "").lower():
                raise
            print(
                f"[lookup] Model {requested_model} failed with mllama error (Ollama too old) â€” "
                f"falling back to {fallback}. Please run: winget upgrade Ollama.Ollama",
                flush=True,
            )
            parsed = extract_with_ollama(
                client, fallback, image_b64
            )
        else:
            raise

    items = parsed.get("items") or []
    items = _dedupe_items(items)

    item_count = parsed.get("item_count")
    if not isinstance(item_count, int):
        item_count = len(items)

    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[dict] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        # Some local models (llava, etc.) return the literal string "null" instead
        # of JSON null. Coerce those to None so they don't pollute the workbook
        # and the validator/UI sees real empties, not the word "null".
        clean = _coerce_null_strings(item)
        row = {col: None for col in EXCEL_COLUMNS}
        row.update({
            "source_filename": path.name,
            "processed_at": timestamp,
            "item_index": idx,
            "item_count": item_count,
            "sku": clean.get("sku"),
            "product_name": clean.get("product_name"),
            "brand": clean.get("brand"),
            "flavor_or_variant": clean.get("flavor_or_variant"),
            "quantity_or_size": clean.get("quantity_or_size"),
            "visible_unit_count": clean.get("visible_unit_count"),
            "confidence": (clean.get("confidence") or "low"),
            "notes": clean.get("notes"),
        })
        rows.append(row)
        if dry_run:
            continue
        # Excel write â€” append_product is now idempotent on (filename, item_index)
        # so retries are safe. We still re-raise so callers can decide how to log.
        try:
            append_product(row)
        except Exception as e:
            print(
                f"[excel] Failed to write row for {path.name} item {idx}: {e}",
                flush=True,
            )
            raise RuntimeError(
                f"Excel write failed for {path.name} item {idx} (sku={row.get('sku')}): {e}"
            ) from e

    return rows, item_count


# Public re-exports so callers don't have to chase imports.
__all__ = [
    "INPUT_DIR",
    "PROCESSED_DIR",
    "OUTPUT_DIR",
    "LOG_CSV_PATH",
    "EXCEL_PATH",
    "EXCEL_COLUMNS",
    "MODEL",
    "OLLAMA_BASE_URL",
    "VALID_EXTS",
    "ensure_dirs",
    "list_input_images",
    "process_image",
    "append_run_log",
    "move_to_processed",
    "scan_barcodes",
    "encode_image_as_base64",
    "extract_with_ollama",
    "append_product",
    "append_rows_to_excel",
    "log",
    "OpenAI",
]
