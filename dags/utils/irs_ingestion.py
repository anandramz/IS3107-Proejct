import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def validate_raw_irs_data(config):
    input_path = os.path.join(config.datasets.irs.base_dir, config.datasets.irs.raw_file)
    if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
        return input_path
    raise FileNotFoundError("Raw IRS file missing!")


def ingest_irs_to_silver(config):
    input_path = os.path.join(config.datasets.irs.base_dir, config.datasets.irs.raw_file)
    output_path = os.path.join(config.paths.staging_dir, config.datasets.irs.staging_file)
    columns_to_keep = config.datasets.irs.columns_to_keep
    
    df = pd.read_csv(input_path, sep=',', engine='python')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    df = df[columns_to_keep]
    # Standardize Values
    df['postal_code'] = df['zipcode'].astype(str).str.zfill(5)
    df['marriage_ratio'] = 2*df['mars2'].astype(float) / df['n2'].astype(float)

    pivot_agi_df = df.pivot_table(index='postal_code', columns='agi_stub', values='n1').fillna(0)
    total_households = pivot_agi_df.sum(axis=1)
    agi_ratio_df = pivot_agi_df.div(total_households, axis=0).reset_index()
    agi_ratio_df.columns = [
        'postal_code',          # keep postal_code for merging
        'ratio_under_25k',      # agi_stub 1
        'ratio_25k_to_50k',     # agi_stub 2
        'ratio_50k_to_75k',     # agi_stub 3
        'ratio_75k_to_100k',    # agi_stub 4
        'ratio_100k_to_200k',   # agi_stub 5
        'ratio_over_200k',      # agi_stub 6
    ]
    
    pivot_agi_marriage_ratio_df = df.pivot_table(index='postal_code', columns='agi_stub', values='marriage_ratio').fillna(0).reset_index()
    pivot_agi_marriage_ratio_df.columns = [
        'postal_code',          # keep postal_code for merging
        'marriage_ratio_under_25k',      # agi_stub 1
        'marriage_ratio_25k_to_50k',     # agi_stub 2
        'marriage_ratio_50k_to_75k',     # agi_stub 3
        'marriage_ratio_75k_to_100k',    # agi_stub 4
        'marriage_ratio_100k_to_200k',   # agi_stub 5
        'marriage_ratio_over_200k',      # agi_stub 6
    ]

    final_df = agi_ratio_df.merge(pivot_agi_marriage_ratio_df, on='postal_code', how='inner')


    # Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False, compression='snappy')
    
    print(f"Successfully processed {len(df)} rows to {output_path}")
