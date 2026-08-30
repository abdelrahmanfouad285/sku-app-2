import os
import pandas as pd
from sqlalchemy import create_engine
from pipeline_helpers import OUTPUT_DIR

EXCEL_COLUMNS = ["source_filename", "processed_at", "item_index", "item_count", "date", "bi_weekly_round", "wk", "rtm", "tdm", "area_name", "client", "client_code", "sku", "product_name", "brand", "flavor_or_variant", "size", "quantity", "confidence", "notes"]
EXCEL_PATH = OUTPUT_DIR / "sku_data.xlsx"

def _get_engine():
    import streamlit as st
    try:
        url = st.secrets["DATABASE_URL"]
    except Exception:
        url = os.environ.get("DATABASE_URL")
    
    if not url:
        raise ValueError("DATABASE_URL is not set in environment or secrets!")
    return create_engine(url)

def _needs_review(row: dict) -> bool:
    required = ("sku", "product_name", "brand", "size", "quantity", "flavor_or_variant", "date", "bi_weekly_round", "wk", "rtm", "tdm", "area_name", "client", "client_code")
    for field in required:
        v = row.get(field)
        if v is None or str(v).strip() == "":
            return True
    conf = str(row.get("confidence", ""))[:3].lower()
    if conf == "low":
        return True
    return False

def append_product(row: dict) -> bool:
    engine = _get_engine()
    df = pd.DataFrame([row])
    # Ensure all columns exist
    for col in EXCEL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EXCEL_COLUMNS]
    
    try:
        # Read existing to check for duplicates
        existing_df = pd.read_sql_table('sku_data', engine)
        if 'barcode_number' in existing_df.columns:
            existing_df = existing_df.drop(columns=['barcode_number'])
        if 'quantity_or_size' in existing_df.columns:
            existing_df = existing_df.rename(columns={'quantity_or_size': 'size'})
        if 'visible_unit_count' in existing_df.columns:
            existing_df = existing_df.rename(columns={'visible_unit_count': 'quantity'})
        fn = row.get("source_filename") or ""
        idx = int(row.get("item_index") or 0)
        
        mask = (existing_df["source_filename"] == fn) & (existing_df["item_index"] == idx)
        if mask.any():
            # Update existing
            for col in EXCEL_COLUMNS:
                existing_df.loc[mask, col] = row.get(col)
            existing_df.to_sql('sku_data', engine, if_exists='replace', index=False)
            return False
        else:
            # Append new
            df.to_sql('sku_data', engine, if_exists='append', index=False)
            return True
    except ValueError:
        # Table doesn't exist yet
        df.to_sql('sku_data', engine, if_exists='replace', index=False)
        return True

def append_rows_to_excel(df: pd.DataFrame) -> None:
    engine = _get_engine()
    df = df.copy()
    for col in EXCEL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EXCEL_COLUMNS]
    
    try:
        existing_df = pd.read_sql_table('sku_data', engine)
        if 'barcode_number' in existing_df.columns:
            existing_df = existing_df.drop(columns=['barcode_number'])
        if 'quantity_or_size' in existing_df.columns:
            existing_df = existing_df.rename(columns={'quantity_or_size': 'size'})
        if 'visible_unit_count' in existing_df.columns:
            existing_df = existing_df.rename(columns={'visible_unit_count': 'quantity'})
        # Update existing or append
        for _, row in df.iterrows():
            fn = row.get("source_filename") or ""
            idx = int(row.get("item_index") or 0)
            mask = (existing_df["source_filename"] == fn) & (existing_df["item_index"] == idx)
            if mask.any():
                for col in EXCEL_COLUMNS:
                    existing_df.loc[mask, col] = row.get(col)
            else:
                existing_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
        existing_df.to_sql('sku_data', engine, if_exists='replace', index=False)
    except ValueError:
        df.to_sql('sku_data', engine, if_exists='replace', index=False)
