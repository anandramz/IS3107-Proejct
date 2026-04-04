import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter
import json

def validate_raw_yelp_data(config):
    input_path = os.path.join(config.datasets.yelp.base_dir, config.datasets.yelp.business.raw_file)
    if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
        return input_path
    raise FileNotFoundError("Raw Yelp file missing!")

def build_consensus_map(config):
    input_path = os.path.join(config.datasets.yelp.base_dir, config.datasets.yelp.business.raw_file)
    output_path = os.path.join(config.paths.staging_dir, config.datasets.yelp.business.map_file)
    
    counts = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                biz = json.loads(line)
                
                # Extract and clean ZIP
                zip = str(biz.get('postal_code', '')).strip().zfill(5)
                raw_city = str(biz.get('city', '')).upper().strip()
                # Cleans city name by removing special characters
                city = "".join(char for char in raw_city if char.isalnum() or char.isspace())

                if zip and city:
                    if zip not in counts:
                        # add a new counter for this ZIP if it doesn't exist
                        counts[zip] = Counter()
                    counts[zip].update([city])
            except json.JSONDecodeError:
                continue

    # We ignore '00000' and '99999' here to keep the map clean
    invalid_zips = {'00000', '99999'}
    
    master_data = [
        {'postal_code': z, 'city_consensus': c.most_common(1)[0][0]} 
        for z, c in counts.items() 
        if c and z not in invalid_zips
    ]

    master_map = pd.DataFrame(master_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_map.to_parquet(output_path, index=False)
    
    return output_path

def clean_and_parquet_yelp_data(config):
    biz_cfg = config.datasets.yelp.business
    output_path = os.path.join(config.paths.staging_dir, biz_cfg.staging_file)
    input_path = os.path.join(config.datasets.yelp.base_dir, biz_cfg.raw_file)
    columns_to_keep = biz_cfg.columns_to_keep

    map_path = os.path.join(config.paths.staging_dir, biz_cfg.map_file)
    master_map_df = pd.read_parquet(map_path)
    
    consensus_map = dict(zip(master_map_df['postal_code'], master_map_df['city_consensus']))
    
    reader = pd.read_json(input_path, lines=True, chunksize=config.defaults.chunk_size)
    writer = None

    try:
        for i, chunk in enumerate(reader):
            # Format postal code
            chunk['postal_code'] = chunk['postal_code'].astype(str).str.zfill(5)
            
            # Overwrite the city column with the consensus name for that ZIP to avoid inconsistent city names
            chunk['city'] = chunk['postal_code'].map(consensus_map)
            chunk = chunk[~chunk['postal_code'].isin(['00000', '99999'])]
            
            chunk = chunk[columns_to_keep]
            chunk['state'] = chunk['state'].str.upper().str.strip()
            chunk['stars'] = chunk['stars'].astype(float)
            chunk['review_count'] = chunk['review_count'].astype(int)
            
            is_numeric = chunk['postal_code'].str.match(r'^\d+$')
            chunk = chunk[is_numeric]

            chunk['categories'] = chunk['categories'].fillna('')
            chunk['categories'] = chunk['categories'].apply(lambda x: [i.strip() for i in x.split(',')])

            table = pa.Table.from_pandas(chunk)

            if writer is None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema)
            
            writer.write_table(table)
            print(f"Processed chunk {i+1}")

    finally:
        if writer: writer.close()
        
    return output_path