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

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="SKU Vision AI", page_icon="📦", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
        [data-testid="stButton"] > button {
            background: linear-gradient(135deg, #2E1A47 0%, #170B2E 100%);
            color: #E2D9F3; border: 1px solid #6339A6; border-radius: 8px; font-weight: 600;
            transition: all 0.3s ease-in-out; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            height: 3rem;
        }
        [data-testid="stButton"] > button:hover {
            border-color: #9D72FF; background: linear-gradient(135deg, #3A2359 0%, #1E0F3D 100%);
            box-shadow: 0 0 15px rgba(157, 114, 255, 0.4); transform: translateY(-2px); color: #FFFFFF;
        }
        [data-testid="stFileUploadDropzone"] {
            background-color: #12141A !important; border: 2px dashed #3A3F58 !important; border-radius: 12px;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #9D72FF !important; background-color: #1A1C24 !important; box-shadow: 0 0 20px rgba(157, 114, 255, 0.1) inset;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #2B2F42; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# 2. HERO SECTION
# ==========================================
st.title("📦 SKU Vision AI")
st.markdown("#### Automated Inventory & SKU Extraction Agent")
st.divider()

# ==========================================
# 3. MAIN WORKSPACE
# ==========================================
col_upload, col_data = st.columns([1, 2], gap="large")

with col_upload:
    st.subheader("📤 1. Upload Images")
    uploaded_files = st.file_uploader("Drop shelf photos here", accept_multiple_files=True, type=["png", "jpg", "jpeg", "heic", "heif"])
    process_btn = st.button("✨ Extract Data Now", use_container_width=True)

with col_data:
    st.subheader("📊 2. Extracted Inventory")
    
    # TRIGGER EXTRACTION
    if process_btn and uploaded_files:
        with st.status("🧠 Initializing Gemini Vision AI...", expanded=True) as status:
            failures = 0
            rows_added = 0
            
            # --- RUN BACKEND PIPELINE ---
            core.ensure_dirs()
            temp_paths = []
            for uf in uploaded_files:
                path = core.INPUT_DIR / uf.name
                with open(path, "wb") as f:
                    f.write(uf.getvalue())
                temp_paths.append(path)
                
            for img_path in temp_paths:
                st.write(f"👁️ Scanning {img_path.name}...")
                
                try:
                    # Extract with Gemini
                    rows, item_count = core.process_image(None, img_path, "Gemini 1.5 Flash (Cloud)")
                    rows_added += len(rows)
                    
                    core.append_run_log(img_path.name, item_count, "ok")
                    core.move_to_processed(img_path)
                    
                except Exception as exc:
                    failures += 1
                    tb = traceback.format_exc()
                    core.append_run_log(img_path.name, 0, "error", tb)
                    st.write(f"❌ Error on {img_path.name}: {exc}")
                    
            st.write("📝 Formatting final spreadsheet...")
            status.update(label=f"Extraction Complete! Added {rows_added} rows.", state="complete", expanded=False)
            
        st.balloons()

    # RENDER DATA EDITOR
    if core.EXCEL_PATH.exists():
        st.markdown("**Review and edit the AI output before saving:**")
        df = pd.read_excel(core.EXCEL_PATH, engine="openpyxl")
        
        edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", hide_index=True)
        
        # Action Toolbar
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("💾 Save Edits to Excel", use_container_width=True):
                edited_df.to_excel(core.EXCEL_PATH, index=False, engine="openpyxl")
                st.success("Successfully saved to Excel!")
                
        with c2:
            st.download_button(
                label="📥 Download CSV", 
                data=edited_df.to_csv(index=False).encode('utf-8'), 
                file_name=f"sku_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                mime="text/csv", 
                use_container_width=True
            )
        with c3:
            with open(core.EXCEL_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Excel",
                    data=f.read(),
                    file_name=f"sku_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    else:
        st.info("👈 Upload images on the left and click 'Extract Data Now' to generate data.")
