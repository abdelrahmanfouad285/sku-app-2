"""
SKU Data Agent — Streamlit UI (Simple Design)
----------------------------------------------
A simple local web UI for the SKU data extraction pipeline. Wraps the core
functions in main.py so the user can:

  1. Drag-and-drop product photos into the browser.
  2. Watch live logs / progress as the agent processes each image.
  3. See the resulting rows in an editable table.
  4. Filter to rows that "need review" or by brand.
  5. Edit values inline; edits are saved back to the Excel file.
  6. Download the workbook (or a CSV) with one click.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

load_dotenv()

# ---------------------------------------------------------------------------
# Session state initialization — MUST happen before any st.* calls
# ---------------------------------------------------------------------------

# Files we'll accept in the uploader — must match pipeline_helpers.VALID_EXTS
_ACCEPTED_EXTS = ("jpg", "jpeg", "png", "heic", "heif")


def _init_state() -> None:
    if "initialized" not in st.session_state:
        st.session_state.setdefault("rows", [])
        st.session_state.setdefault("processing", False)
        st.session_state.setdefault("log_lines", [])
        st.session_state.setdefault("base_url", "http://localhost:11434/v1")
        st.session_state.setdefault("model", "llama3.2-vision:latest")
        st.session_state["initialized"] = True


_init_state()

# ---------------------------------------------------------------------------
# Import the core pipeline functions from main.py
# ---------------------------------------------------------------------------
import sys
import importlib
if 'main' in sys.modules:
    importlib.reload(sys.modules['main'])
if 'utils.llm' in sys.modules:
    importlib.reload(sys.modules['utils.llm'])
import main as core

import streamlit as st

def check_password():
    def password_entered():
        import os
        # Fallback to env var if running locally, otherwise use Streamlit secrets
        try:
            secret_pass = st.secrets.get('APP_PASSWORD')
        except Exception:
            secret_pass = os.environ.get('APP_PASSWORD', 'admin123')
            
        if not secret_pass:
            secret_pass = 'admin123'

        if st.session_state.get('password') == secret_pass:
            st.session_state['password_correct'] = True
            del st.session_state['password']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    st.markdown('# 🔒 Secure Login')
    st.text_input('Enter Password', type='password', on_change=password_entered, key='password')
    if 'password_correct' in st.session_state and not st.session_state['password_correct']:
        st.error('Incorrect password')
    return False

if False:
    st.stop()

from pipeline_helpers import VALID_EXTS
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sku_agent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_uploaded_files(uploaded_files) -> tuple[list[Path], list[str]]:
    """Save uploaded files to INPUT_DIR. Returns (saved, skipped_reasons)."""
    core.ensure_dirs()
    saved: list[Path] = []
    skipped: list[str] = []
    for up in uploaded_files:
        # The uploader already filters by `type=` on the client side, but be
        # defensive — a malicious or buggy client could still send anything.
        ext = Path(up.name).suffix.lower().lstrip(".")
        if ext not in _ACCEPTED_EXTS:
            skipped.append(f"{up.name}: unsupported type '.{ext}'")
            continue
        if not up.getbuffer():
            skipped.append(f"{up.name}: empty file")
            continue
        safe_name = Path(up.name).name
        dest = core.INPUT_DIR / safe_name
        counter = 1
        while dest.exists():
            dest = core.INPUT_DIR / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
            counter += 1
        dest.write_bytes(up.getbuffer())
        saved.append(dest)
    return saved, skipped



from sqlalchemy import create_engine
import os

def _get_engine():
    import streamlit as st
    try:
        url = st.secrets["DATABASE_URL"]
    except Exception:
        url = os.environ.get("DATABASE_URL")
    return create_engine(url)

def _load_existing_excel() -> pd.DataFrame:
    import pandas as pd
    import streamlit as st
    import main as core
    try:
        engine = _get_engine()
        df = pd.read_sql_table('sku_data', engine)
        for col in core.EXCEL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[core.EXCEL_COLUMNS]
    except Exception as exc:
        return pd.DataFrame(columns=core.EXCEL_COLUMNS)

def _write_dataframe_to_excel(df: pd.DataFrame) -> None:
    import streamlit as st
    try:
        engine = _get_engine()
        df.to_sql('sku_data', engine, if_exists='replace', index=False)
    except Exception as exc:
        st.error(f"Failed to save to Postgres: {exc}")

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="SKU Data")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Using Google Gemini API.")
    st.session_state.model = "Gemini 1.5 Flash (Cloud)"
    
    st.selectbox(
        "AI Vision Model",
        options=[st.session_state.model],
        disabled=True,
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("AI SKU Agent")
st.caption("Drop product photos into input_photos/. Google Gemini extracts SKU data directly into this editable table.")

tab_upload, tab_data, tab_log = st.tabs(["📤 Upload & Process", "📊 Results & Edit", "📜 Log"])


# ---- Tab 1: Upload & Process ---------------------------------------------

with tab_upload:
    # Show the most recent run's summary if present so the user can see
    # "Saved N rows" even after a rerun.
    if st.session_state.get("_last_summary"):
        st.success(st.session_state["_last_summary"])

    uploaded = st.file_uploader(
        "Drag and drop or browse (jpg, jpeg, png, heic)",
        type=list(_ACCEPTED_EXTS),
        accept_multiple_files=True,
    )

    cols = st.columns([1, 1, 2])
    with cols[0]:
        save_clicked = st.button("💾  Save uploads", width='stretch')
    with cols[1]:
        run_clicked = st.button(
            "🚀  Process all images in input_photos/",
            type="primary",
            width='stretch',
            disabled=st.session_state.processing or not True,
        )
    with cols[2]:
        pending = len(list(core.INPUT_DIR.iterdir())) if core.INPUT_DIR.exists() else 0
        st.caption(f"Folder: `{core.INPUT_DIR.resolve()}`  •  ⏳ Pending: {pending}")

    if save_clicked and uploaded:
        saved, skipped = _save_uploaded_files(uploaded)
        if saved:
            st.success(f"Saved {len(saved)} file(s) to {core.INPUT_DIR}")
        if skipped:
            for s in skipped:
                st.warning(f"Skipped {s}")
        st.rerun()

    if run_clicked:
        st.session_state.processing = True
        st.session_state.log_lines = []
        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        client = None
        core.MODEL = st.session_state.model

        images = core.list_input_images()
        if not images:
            status.warning("No images found in input_photos/.")
            st.session_state.processing = False
            st.session_state["_last_summary"] = "No images found in input_photos/."
        else:
            t0 = time.perf_counter()
            new_rows: list[dict] = []
            failures = 0
            for i, img_path in enumerate(images, start=1):
                status.info(f"Processing {img_path.name} ({i}/{len(images)})…")
                # --- LLM + validation stage ---
                try:
                    rows, item_count = core.process_image(None, img_path, st.session_state.model)
                except Exception as exc:
                    tb = traceback.format_exc()
                    failures += 1
                    core.append_run_log(img_path.name, 0, "error", tb)
                    st.session_state.log_lines.append(f"ERROR on {img_path.name} (LLM/validation stage):\n{tb}")
                    progress.progress(i / len(images), text=f"{i}/{len(images)} done")
                    continue

                # Always log a per-image summary line so the user can see WHY
                # item_count might be 0 (e.g. blank shelf, model returned 0).
                st.session_state.log_lines.append(
                    f"{img_path.name}: extracted {item_count} item(s) ({len(rows)} row(s) written)"
                )

                # --- Excel write stage (separate try so you can tell which stage failed) ---
                try:
                    if rows:
                        new_rows.extend(rows)
                    core.append_run_log(img_path.name, item_count, "ok")
                    core.move_to_processed(img_path)
                except Exception as exc:
                    tb = traceback.format_exc()
                    failures += 1
                    core.append_run_log(
                        img_path.name, item_count, "error",
                        f"Excel write failed for {img_path.name}: {tb}",
                    )
                    st.session_state.log_lines.append(f"ERROR on {img_path.name} (Excel write stage):\n{tb}")

                progress.progress(i / len(images), text=f"{i}/{len(images)} done")

            elapsed = time.perf_counter() - t0
            st.session_state.rows.extend(new_rows)
            summary = (
                f"Done in {elapsed:.1f}s — {len(new_rows)} row(s) added, {failures} failure(s). "
                f"Workbook: {core.EXCEL_PATH}"
            )
            st.session_state["_last_summary"] = summary
            status.success(summary)
            progress.empty()
            st.session_state.processing = False


# ---- Tab 2: Results & Edit -----------------------------------------------

with tab_data:
    df = _load_existing_excel()

    with st.expander("🔎  Filters", expanded=False):
        fcols = st.columns(3, gap="medium")
        with fcols[0]:
            needs_review = st.checkbox(
                "Show only rows needing review (yellow)",
                value=False,
                help="Low confidence OR any null in sku/barcode/product_name/brand/qty",
            )
        with fcols[1]:
            brands = sorted({b for b in df["brand"].dropna().unique() if str(b).strip()})
            brand_filter = st.multiselect("Filter by brand", options=brands, default=[])
        with fcols[2]:
            conf_options = ["high", "medium", "low"]
            conf_filter = st.multiselect("Filter by confidence", options=conf_options, default=[])

    # "Needs review" uses the same fields the Excel writer highlights on.
    required_for_review = ("sku", "barcode_number", "product_name", "brand", "quantity_or_size")

    view_df = df.copy()
    if needs_review:
        mask_null = view_df[list(required_for_review)].isna().any(axis=1) | (
            view_df[list(required_for_review)] == ""
        ).any(axis=1)
        mask_low = view_df["confidence"].fillna("low").astype(str).str.lower() == "low"
        view_df = view_df[mask_null | mask_low]
    if brand_filter:
        view_df = view_df[view_df["brand"].isin(brand_filter)]
    if conf_filter:
        view_df = view_df[view_df["confidence"].fillna("low").astype(str).str.lower().isin(conf_filter)]

    st.caption(f"Showing {len(view_df)} of {len(df)} row(s)")

    if view_df.empty:
        st.info("No rows match the current filters.")
    else:
        edited = st.data_editor(
            view_df,
            width='stretch',
            num_rows="fixed",
            hide_index=True,
            key="sku_editor",
        )

    bcols = st.columns([1, 1, 1, 2])
    with bcols[0]:
        save_edits = st.button("💾  Save edits", type="primary", disabled=view_df.empty)
    with bcols[1]:
        st.download_button(
            "⬇️  CSV",
            data=_to_csv_bytes(df),
            file_name=f"sku_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with bcols[2]:
        st.download_button(
            "⬇️  Excel",
            data=_to_excel_bytes(df),
            file_name=f"sku_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if save_edits and not view_df.empty:
        try:
            merged = df.copy()
            # When the user clears a cell in st.data_editor, pandas may return
            # the string "None", NaN, or the literal "null". Normalize all of
            # those to real None so we don't store the words "None"/"null" in
            # the workbook.
            def _clean(v):
                if v is None:
                    return None
                if isinstance(v, float) and pd.isna(v):
                    return None
                if isinstance(v, str):
                    s = v.strip()
                    if s == "" or s.lower() in {"none", "null", "nil", "n/a", "na"}:
                        return None
                    return s
                return v
            for _, row in edited.iterrows():
                key = (row.get("source_filename"), row.get("item_index"))
                mask = (merged["source_filename"] == key[0]) & (merged["item_index"] == key[1])
                if mask.any():
                    for col in core.EXCEL_COLUMNS:
                        merged.loc[mask, col] = _clean(row.get(col))
            _write_dataframe_to_excel(merged)
            st.success("Saved back to Excel.")
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    # --- Per-image log (current session) ---
    if st.session_state.get("log_lines"):
        with st.expander("📋  Per-image log (this session)", expanded=False):
            st.code("\n".join(st.session_state.log_lines[-50:]), language="log")


    with st.expander("⚠️  Danger zone", expanded=False):
        st.caption(
            "These actions permanently delete data. Download a backup first "
            "using the buttons above."
        )
        confirm = st.checkbox("I understand this will delete all data", key="clear_confirm")
        if st.button("🗑️  Clear table", type="secondary", disabled=not confirm):
            try:
                empty_df = pd.DataFrame(columns=core.EXCEL_COLUMNS)
                _write_dataframe_to_excel(empty_df)
                st.session_state.rows = []
                if "sku_editor" in st.session_state:
                    del st.session_state["sku_editor"]
                st.success("Table cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Clear failed: {exc}")


# ---- Tab 3: Log ----------------------------------------------------------

with tab_log:
    st.subheader("Run log")
    if st.session_state.log_lines:
        st.code("\n".join(st.session_state.log_lines[-200:]), language="log")
    else:
        st.info("No log output yet. Process some images to see live logs here.")
    st.divider()
    st.subheader("run_log.csv (per-image history)")
    if core.LOG_CSV_PATH.exists():
        try:
            log_df = pd.read_csv(core.LOG_CSV_PATH)
            st.dataframe(log_df, width='stretch', hide_index=True)
        except Exception as exc:
            st.warning(f"Could not read run log: {exc}")
    else:
        st.caption("No run log yet.")
