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

def join_yelp_irs_and_demographics_data(config):
    # Get paths
    yelp_irs_path = os.path.join(config.paths.staging_dir, config.datasets.yelp_irs.staging_file)
    demographics_path = os.path.join(config.paths.staging_dir, config.datasets.demographics.staging_file)
    output_path = os.path.join(config.paths.staging_dir, config.datasets.yelp_irs_demographics.staging_file)
    
    # Load data
    yelp = pd.read_parquet(yelp_irs_path)
    demographics = pd.read_parquet(demographics_path)


    # Check for duplicate city-state pairs before joining
    assert not demographics.duplicated(subset=["city", "state"]).any()

    # Join data on city and state
    joined = demographics.merge(yelp, on=['city', 'state'], how='inner')

    joined.to_parquet(output_path, index=False)

    return output_path