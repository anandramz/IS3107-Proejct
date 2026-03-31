import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from airflow.decorators import dag, task
import pendulum

# --- HELPER FUNCTIONS ---
def get_config():
    dag_path = os.path.dirname(__file__)
    config_path = os.path.join(dag_path, "config.yaml")
    return OmegaConf.load(config_path)

@dag(
    dag_id="test_yelp_ingestion",
    schedule=None,  # Manual trigger only
    start_date=pendulum.datetime(2024, 1, 1),
    catchup=False,
    tags=['test', 'yelp']
)
def test_ingestion_dag():
    @task
    def validate_raw_data():
        config = get_config()
        input_path = os.path.join(config.datasets.yelp.base_dir, config.datasets.yelp.business.raw_file)
        if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
            return input_path
        raise FileNotFoundError("Raw Yelp file missing!")

    @task
    def clean_and_parquet(input_path):
        config = get_config()
        biz_cfg = config.datasets.yelp.business
        output_path = os.path.join(config.paths.staging_dir, biz_cfg.output_file)
        
        reader = pd.read_json(input_path, lines=True, chunksize=config.defaults.chunk_size)
        writer = None

        try:
            for chunk in reader:
                chunk = chunk[list(biz_cfg.columns_to_keep)]
                
                chunk['city'] = chunk['city'].str.upper().str.strip()

                table = pa.Table.from_pandas(chunk)
                if writer is None:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema)
                
                writer.write_table(table)
        finally:
            if writer: writer.close()
            
        return output_path

    # Flow
    raw_path = validate_raw_data()
    clean_and_parquet(raw_path)

# Instantiate the DAG
test_dag = test_ingestion_dag()