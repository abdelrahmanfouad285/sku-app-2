import sys
import importlib
if 'main' in sys.modules:
    importlib.reload(sys.modules['main'])
if 'llm' in sys.modules:
    importlib.reload(sys.modules['llm'])
import main as core

import streamlit as st
import pandas as pd
import time
import traceback
from datetime import datetime
from pathlib import Path
import io

st.set_page_config(page_title="AI SKU Agent", layout="wide", page_icon="🤖")

# --- Modern UI Styles ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #90949c;
        margin-bottom: 30px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AI SKU Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instantly extract products, variants, and quantities using Google Gemini.</div>', unsafe_allow_html=True)

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

tab_upload, tab_data = st.tabs(["🚀 Upload & Process", "📊 View & Edit Data"])

with tab_upload:
    st.write("### Upload Product Photos")
    
    uploaded_files = st.file_uploader(
        "Drag and drop shelf or fridge photos here...",
        type=["jpg", "jpeg", "png", "heic", "heif"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) ready to process.")
        
        if st.button("✨ Extract Data Now ✨", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing...")
            status = st.empty()
            
            failures = 0
            rows_added = 0
            
            core.ensure_dirs()
            temp_paths = []
            for uf in uploaded_files:
                path = core.INPUT_DIR / uf.name
                with open(path, "wb") as f:
                    f.write(uf.getvalue())
                temp_paths.append(path)
                
            for i, img_path in enumerate(temp_paths, start=1):
                progress_bar.progress(i / len(temp_paths), text=f"Processing {i}/{len(temp_paths)}...")
                status.info(f"Analyzing {img_path.name} with Gemini...")
                
                try:
                    rows, item_count = core.process_image(None, img_path, "Gemini 3.6 Flash (Cloud)")
                    rows_added += len(rows)
                    st.session_state.log_lines.append(f"SUCCESS: {img_path.name} -> {item_count} items")
                    
                    core.append_run_log(img_path.name, item_count, "ok")
                    core.move_to_processed(img_path)
                    
                except Exception as exc:
                    failures += 1
                    tb = traceback.format_exc()
                    core.append_run_log(img_path.name, 0, "error", tb)
                    st.session_state.log_lines.append(f"ERROR on {img_path.name}: {exc}")
                    
            status.success(f"✅ Finished! Added {rows_added} row(s) with {failures} failure(s).")
            progress_bar.empty()
            st.balloons()

with tab_data:
    st.write("### Extracted Inventory Data")
    
    if not core.EXCEL_PATH.exists():
        st.info("No data yet. Upload some images to get started!")
    else:
        df = pd.read_excel(core.EXCEL_PATH, engine="openpyxl")
        
        edited_df = st.data_editor(
            df, 
            use_container_width=True, 
            num_rows="dynamic",
            hide_index=True,
            height=500
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Edits to Excel", use_container_width=True):
                # We save back to the original format
                edited_df.to_excel(core.EXCEL_PATH, index=False, engine="openpyxl")
                st.toast("✅ Saved changes!")
                
        with col2:
            csv = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sku_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
