import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)

def download_dataset_from_kaggle(config):
    dataset_identifier = config.datasets.demographics.kaggle_id
    download_path = config.datasets.demographics.base_dir
    expected_file = config.datasets.demographics.raw_file

    target_file_path = os.path.join(download_path, expected_file)

    # Check if the file already exists
    if os.path.exists(target_file_path):
        print(f"File {expected_file} already exists at {download_path}. Skipping download.")
        return

    try:
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading {dataset_identifier} to {download_path}...")
        
        os.makedirs(download_path, exist_ok=True)
        
        api.dataset_download_files(
            dataset_identifier, 
            path=download_path, 
            unzip=True
        )
        print("Download complete.")
        
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        raise e

    return

def ingest_demographics_to_silver(config):
    input_path = os.path.join(config.datasets.demographics.base_dir, config.datasets.demographics.raw_file)
    output_path = os.path.join(config.paths.staging_dir, config.datasets.demographics.staging_file)
    columns_to_keep = config.datasets.demographics.columns_to_keep
    
    df = pd.read_csv(input_path, sep=';', engine='python')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    df = df[columns_to_keep]
    # 4. Standardize Values
    df['city'] = df['city'].str.upper().str.strip()
    df['state'] = df['state_code'].str.upper().str.strip()
    df.drop(columns=['state_code'], inplace=True)
    df = df.drop_duplicates(subset=['city', 'state'], keep='first')

    # 5. Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, compression='snappy')
    
    print(f"Successfully processed {len(df)} rows to {output_path}")
