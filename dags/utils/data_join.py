import os
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

def join_yelp_and_irs_data(config):
    # Get paths
    yelp_path = os.path.join(config.paths.staging_dir, config.datasets.yelp.business.staging_file)
    irs_path = os.path.join(config.paths.staging_dir, config.datasets.irs.staging_file)
    output_path = os.path.join(config.paths.staging_dir, config.datasets.yelp_irs.staging_file)
    
    # Load data
    yelp = pd.read_parquet(yelp_path)
    irs = pd.read_parquet(irs_path)

    # Join data on postal code and state
    joined = yelp.merge(irs, on=['postal_code', 'state'], how='inner')
    assert not joined['postal_code'].isnull().any(), "Join resulted in missing postal codes!" 
    assert not joined['state'].isnull().any(), "Join resulted in missing states!"

    joined.to_parquet(output_path, index=False)

    return output_path
