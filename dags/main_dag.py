import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from airflow.decorators import dag, task
import pendulum
from utils.yelp_ingestion import validate_raw_yelp_data, clean_and_parquet_yelp_data

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

    validate_yelp_bronze() >> stage_yelp_silver()


dag = main_dag()