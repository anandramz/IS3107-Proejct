import os
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

def join_yelp_and_demographics_data(config):
    # Get paths
    yelp_path = os.path.join(config.paths.staging_dir, config.datasets.yelp.business.staging_file)
    demographics_path = os.path.join(config.paths.staging_dir, config.datasets.demographics.staging_file)
    output_path = os.path.join(config.paths.staging_dir, config.datasets.joined.staging_file)
    
    # Load data
    yelp = pd.read_parquet(yelp_path)
    demographics = pd.read_parquet(demographics_path)

    # Check for duplicate city-state pairs before joining
    assert not demographics.duplicated(subset=["city", "state"]).any()

    # Join data on city and state
    joined = demographics.merge(yelp, on=['city', 'state'], how='inner')

    joined.to_parquet(output_path, index=False)

    return output_path