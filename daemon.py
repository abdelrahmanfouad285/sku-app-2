import time
import requests
import tempfile
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
import main as core
import traceback

def get_engine():
    db_uri = os.environ.get("DATABASE_URL")
    if not db_uri:
        # Fallback for Streamlit secrets if needed
        import streamlit as st
        try:
            db_uri = st.secrets["DATABASE_URL"]
        except:
            pass
    return create_engine(db_uri)

def download_image(url: str, dest_dir: Path) -> Path | None:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        filename = url.split("/")[-1]
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.heic')):
            filename += ".jpg"
        
        path = dest_dir / filename
        with open(path, 'wb') as f:
            f.write(response.content)
        return path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def run_sync():
    print("Starting SQL Sync...", flush=True)
    engine = get_engine()
    
    # Get all processed URLs
    try:
        sku_df = pd.read_sql_table('sku_data', engine)
        processed_urls = set(sku_df['source_filename'].dropna().tolist())
    except:
        processed_urls = set()
        
    print(f"Found {len(processed_urls)} already processed URLs.")

    # Get URLs from audit_records
    query = """
    SELECT 
        id, 
        photo_link1_mp, photo_link2_mp, 
        photo_link1_sp, photo_link2_sp, 
        photo_link1_ambient, photo_link2_ambient, 
        storefront_photo
    FROM audit_records
    ORDER BY id DESC LIMIT 200
    """
    audit_df = pd.read_sql_query(query, engine)
    
    columns_to_check = [
        'photo_link1_mp', 'photo_link2_mp', 
        'photo_link1_sp', 'photo_link2_sp', 
        'photo_link1_ambient', 'photo_link2_ambient', 
        'storefront_photo'
    ]
    
    new_urls = set()
    for _, row in audit_df.iterrows():
        for col in columns_to_check:
            url = row.get(col)
            if url and isinstance(url, str) and url.startswith("http"):
                if url not in processed_urls:
                    new_urls.add(url)
                    
    print(f"Found {len(new_urls)} new URLs to process.")
    
    core.ensure_dirs()
    for url in list(new_urls):
        print(f"Processing {url}...")
        img_path = download_image(url, core.INPUT_DIR)
        if img_path:
            try:
                # Process with Gemini (passing None for client)
                # We rename the img_path so the database records the URL as the source_filename!
                # Wait, process_image uses img_path.name. 
                # Let's temporarily mock the path name
                original_name = img_path.name
                
                rows, item_count = core.process_image(None, img_path, "gemini-1.5-flash")
                
                # Now we need to overwrite the source_filename to be the URL so it's marked as processed
                for r in rows:
                    r["source_filename"] = url
                    
                # Save to SQL
                for r in rows:
                    core.append_product(r)
                
                print(f"Success! Found {item_count} items.")
            except Exception as e:
                print(f"Error processing {url}: {e}")
                traceback.print_exc()
            finally:
                if img_path.exists():
                    os.remove(img_path)

if __name__ == "__main__":
    import sys
    import dotenv
    dotenv.load_dotenv()
    if "--once" in sys.argv:
        try:
            run_sync()
        except Exception as e:
            print(f"Sync error: {e}")
    else:
        while True:
            try:
                run_sync()
            except Exception as e:
                print(f"Sync error: {e}")
            print("Sleeping for 60 seconds...")
            time.sleep(60)
