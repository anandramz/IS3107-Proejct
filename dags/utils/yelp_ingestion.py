import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

def validate_raw_yelp_data(config):
    input_path = os.path.join(config.datasets.yelp.base_dir, config.datasets.yelp.business.raw_file)
    if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
        return input_path
    raise FileNotFoundError("Raw Yelp file missing!")

def clean_and_parquet_yelp_data(config):
    biz_cfg = config.datasets.yelp.business

    output_path = os.path.join(config.paths.staging_dir, biz_cfg.staging_file)
    input_path = os.path.join(config.datasets.yelp.base_dir, config.datasets.yelp.business.raw_file)
    columns_to_keep = biz_cfg.columns_to_keep

    reader = pd.read_json(input_path, lines=True, chunksize=config.defaults.chunk_size)
    writer = None

    try:
        for i, chunk in enumerate(reader):
            
                
            # chunk = chunk[chunk['is_open'] == 1]                        # Filter for open businesses
            chunk = chunk[columns_to_keep]                              # Select columns to keep
            chunk['city'] = chunk['city'].str.upper().str.strip()       # Format city to uppercase
            chunk['state'] = chunk['state'].str.upper().str.strip()     # Format state to uppercase
            chunk['stars'] = chunk['stars'].astype(float)               # Ensure stars is float
            chunk['review_count'] = chunk['review_count'].astype(int)   # Ensure stars is float
            
            # Force to string first to handle any mixed types/NaNs
            chunk['postal_code'] = chunk['postal_code'].astype(str).str.zfill(5)
            is_numeric = chunk['postal_code'].str.match(r'^\d+$')
            chunk = chunk[is_numeric]

            # Convert categories from csv string to list
            chunk['categories'] = chunk['categories'].fillna('')
            chunk['categories'] = chunk['categories'].apply(lambda x: [i.strip() for i in x.split(',')])

            table = pa.Table.from_pandas(chunk)

            if writer is None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema)
            
            writer.write_table(table)
    finally:
        if writer: writer.close()
        
    return output_path

