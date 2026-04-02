import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from airflow.decorators import dag, task
from utils.data_join import join_yelp_and_demographics_data
import pendulum
from utils.yelp_ingestion import validate_raw_yelp_data, clean_and_parquet_yelp_data
from utils.demographics_ingestion import ingest_demographics_to_silver, download_dataset_from_kaggle

# --- HELPER FUNCTIONS ---
def get_config():
    dag_path = os.path.dirname(__file__)
    config_path = os.path.join(dag_path, "config.yaml")
    return OmegaConf.load(config_path)

@dag(
    dag_id="main_dag",
    schedule=None,  # Manual trigger only
    start_date=pendulum.datetime(2024, 1, 1),
    catchup=False,
    tags=['test', 'yelp']
)
def main_dag():
    config = get_config()

    @task
    def validate_yelp_bronze():
        validate_raw_yelp_data(config)

    @task
    def stage_yelp_silver():
        clean_and_parquet_yelp_data(config)

    @task
    def validate_demographics_bronze():
        download_dataset_from_kaggle(config)

    @task
    def stage_demographics_silver():
        ingest_demographics_to_silver(config)

    @task
    def join_data_silver():
        join_yelp_and_demographics_data(config)

    validate_yelp = validate_yelp_bronze()
    validate_demographics = validate_demographics_bronze()

    stage_yelp = stage_yelp_silver()
    stage_demographics = stage_demographics_silver()

    join = join_data_silver()

    validate_yelp >> stage_yelp                     # Branch 1: Yelp
    validate_demographics >> stage_demographics     # Branch 2: Demographics
    [stage_yelp, stage_demographics] >> join
    
dag = main_dag()