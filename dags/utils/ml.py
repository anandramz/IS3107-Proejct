import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def cluster_zip_codes(config):

    joined_data_path = os.path.join(config.paths.staging_dir, config.datasets.yelp_irs_demographics.staging_file)
    output_path = os.path.join(config.paths.staging_dir, config.ml.cluster_staging_file)
    df = pd.read_parquet(joined_data_path)
    
    # Drop business-specific columns
    df = df.drop(
        ['business_id', 'name', 'categories', 'review_count', 'stars'], 
        axis=1
    ).drop_duplicates()

    # Scale the Data
    scaler = StandardScaler()
    features_to_cluster = config.ml.cluster_features
    scaled_features = scaler.fit_transform(df[features_to_cluster])
    
    # Run K-Means clustering
    num_clusters = config.ml.num_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    
    # Predict and assign the cluster ID to our dataframe
    df['cluster_id'] = kmeans.fit_predict(scaled_features)

    df.to_parquet(output_path, index=False, compression='snappy')
    
    return output_path



def generate_opportunity_mart(config):
    clustered_zips_path = os.path.join(config.paths.staging_dir, config.ml.cluster_staging_file)
    joined_data_path = os.path.join(config.paths.staging_dir, config.datasets.yelp_irs_demographics.staging_file)
    
    # Load data
    zip_profiles = pd.read_parquet(clustered_zips_path)
    df = pd.read_parquet(joined_data_path)

    for target_category in config.ml.target_categories:
        target_set = set(target_category)   # Convert to set for faster lookup
        mask = df['categories'].apply(lambda x: bool(set(x).intersection(target_set)))

        # Filter the DataFrame
        cat_df = df[mask]

        # Create success metric (stars * log(reviews))
        cat_df['business_success'] = cat_df['stars'] * np.log1p(cat_df['review_count'])

    
        # Aggregate to the ZIP level
        zip_supply = cat_df.groupby('postal_code').agg(
            current_supply=('business_id', 'count'),
            avg_success=('business_success', 'mean'),
        ).reset_index()

        # Merge Supply back into Demographic Profiles
        master_df = pd.merge(zip_profiles, zip_supply, on='postal_code', how='left')
        master_df['current_supply'] = master_df['current_supply'].fillna(0)
        master_df['avg_success'] = master_df['avg_success'].fillna(0)
        master_df['current_density'] = np.where(
            master_df['total_individuals'] > 0,
            master_df['current_supply'] / master_df['total_individuals'],
            0.0
        )
    
        # Define Anchor ZIPs that are succesful in their category
        successful_zips = master_df[master_df['current_supply'] > 0]
        top_threshold = successful_zips['avg_success'].quantile(config.ml.quantile)
        anchor_zips = successful_zips[successful_zips['avg_success'] >= top_threshold]
    
        exclude_cols = ['postal_code', 'cluster_id', 'current_supply', 'avg_success']
        feature_cols = [
            col for col in zip_profiles.select_dtypes(include=['number']).columns
            if col not in exclude_cols
        ]
        
        # The ideal demographic profile is the mean of the anchor ZIPs
        anchor_vector = anchor_zips[feature_cols].mean().values.reshape(1, -1)
        
        # Scale features and calculate cosine similarity
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(master_df[feature_cols])
        scaled_anchor = scaler.transform(anchor_vector)
    
        # Calculate similarity
        master_df['similarity_score'] = cosine_similarity(scaled_features, scaled_anchor)
    
        # # Format the final data mart
        # mart_cols = ['postal_code', 'city', 'state', 'cluster_id', 'current_supply', 'similarity_score', 'current_density', 'total_individuals']
        # opportunity_mart = master_df[mart_cols].copy()
        
        # # Sort to show the biggest gaps at the top!
        # # High similarity (>0.85) but Zero/Low Supply
        # opportunity_mart = opportunity_mart.sort_values(
        #     by=['similarity_score', 'current_supply'], 
        #     ascending=[False, True]
        # ).reset_index(drop=True)

        # suffix = "_".join(target_category).lower().replace(" & ", "_").replace(" ", "_")
        # output_dir = config.paths.output_dir
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(
        #     output_dir,
        #     f"opportunity_mart_{suffix}.parquet"
        # )
        # opportunity_mart.to_parquet(output_path)

        
        master_df['log_supply'] = np.log1p(master_df['current_supply'])
        master_df['log_density'] = np.log1p(master_df['current_density'])
        master_df['log_population'] = np.log1p(master_df['total_individuals'])

        # Normalize components to 0-1
        score_scaler = MinMaxScaler()

        norm_cols = ['similarity_score', 'log_supply', 'log_density', 'log_population']
        normalized = score_scaler.fit_transform(master_df[norm_cols])

        master_df['similarity_norm'] = normalized[:, 0]
        master_df['supply_norm'] = normalized[:, 1]
        master_df['density_norm'] = normalized[:, 2]
        master_df['population_norm'] = normalized[:, 3]

        # Higher is better:
        # - similarity_norm high is good
        # - low supply is good => 1 - supply_norm
        # - low density is good => 1 - density_norm
        # - high population is good => population_norm
        master_df['opportunity_score'] = (
            0.6 * master_df['similarity_norm'] +
            0.25 * (1 - master_df['density_norm']) +
            0.15 * master_df['population_norm']
        )

        # Final mart
        mart_cols = [
            'postal_code',
            'city',
            'state',
            'cluster_id',
            'total_individuals',
            'similarity_score',
            'opportunity_score',
            'current_supply',
            'current_density',
        ]

        opportunity_mart = master_df[mart_cols].copy()

        # Sort best opportunities to top
        opportunity_mart = opportunity_mart.sort_values(
            ascending=False,
            by='opportunity_score',
        ).reset_index(drop=True)

        suffix = "_".join(target_category).lower().replace(" & ", "_").replace(" ", "_")
        output_dir = config.paths.output_dir
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"opportunity_mart_{suffix}.parquet"
        )
        opportunity_mart.to_parquet(output_path, index=False)


    return 
