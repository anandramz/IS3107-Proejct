import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from airflow.decorators import dag, task
from utils.data_join import join_yelp_and_irs_data, join_yelp_irs_and_demographics_data
import pendulum
from utils.yelp_ingestion import validate_raw_yelp_data, clean_and_parquet_yelp_data, build_consensus_map
# from utils.demographics_ingestion import ingest_demographics_to_silver, download_dataset_from_kaggle
from utils.irs_ingestion import validate_raw_irs_data, ingest_irs_to_silver
from utils.ml import cluster_zip_codes, generate_opportunity_mart

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
    def build_consensus_map_task():
        build_consensus_map(config)

    @task
    def stage_yelp_silver():
        clean_and_parquet_yelp_data(config)

    # @task
    # def validate_demographics_bronze():
    #     download_dataset_from_kaggle(config)

    # @task
    # def stage_demographics_silver():
    #     ingest_demographics_to_silver(config)

    @task
    def validate_irs_bronze():
        validate_raw_irs_data(config)
    
    @task
    def stage_irs_silver():
        ingest_irs_to_silver(config)

    @task
    def join_yelp_irs_silver():
        join_yelp_and_irs_data(config)

    # @task
    # def join_yelp_irs_demographics_silver():
    #     join_yelp_irs_and_demographics_data(config)

    @task
    def cluster_zip_codes_gold():
        cluster_zip_codes(config)
    
    @task
    def generate_opportunity_mart_gold():
        generate_opportunity_mart(config)


    validate_yelp = validate_yelp_bronze()
    # validate_demographics = validate_demographics_bronze()

    build_consensus_map_yelp = build_consensus_map_task()
    stage_yelp = stage_yelp_silver()
    # stage_demographics = stage_demographics_silver()

    validate_irs = validate_irs_bronze()
    stage_irs = stage_irs_silver()

    join_yelp_irs = join_yelp_irs_silver()
    # join_yelp_irs_demographics = join_yelp_irs_demographics_silver()

    cluster_zip = cluster_zip_codes_gold()
    generate_marts = generate_opportunity_mart_gold()

    validate_yelp >> build_consensus_map_yelp >> stage_yelp # Branch 1: Yelp
    validate_irs >> stage_irs                               # Branch 2: IRS
    # validate_demographics >> stage_demographics             # Branch 3: Demographics
    
    [stage_yelp, stage_irs] >> join_yelp_irs                           
    join_yelp_irs >> cluster_zip >> generate_marts   

    
dag = main_dag()