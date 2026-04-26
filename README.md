# IS3107 Project — Business Opportunity Analysis

An Airflow pipeline that cross-references Yelp business data with IRS income statistics to surface ZIP codes that look demographically similar to places where a given business category thrives, but currently have low supply.

## What it does

The DAG runs in three stages:

**Bronze → Silver**: Raw data lands as-is. The Yelp JSON is chunked and cleaned into Parquet (standardising city names via a consensus map over postal codes). The IRS CSV is pivoted from one row per income bracket into one row per ZIP, with income ratios and marriage rates as features.

**Silver → Join**: The two datasets are merged on `(postal_code, state)`.

**Gold**: ZIP codes are clustered (KMeans, k=5) on their income distribution and marriage-rate profiles. For each business category, the pipeline identifies "anchor ZIPs" — places that already host successful businesses in that category — and computes a cosine similarity between every other ZIP and that anchor profile. The final opportunity score weighs demographic similarity (60%), inverse supply density (25%), and population size (15%), then outputs a sorted Parquet mart per category.

Categories covered: Restaurants/Food, Shopping, Health & Medical, Beauty & Spas, Bars/Nightlife.

## Data

You need two datasets placed under `data/raw/`:

- **Yelp Open Dataset** (JSON) — `data/raw/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json`
- **IRS SOI ZIP Code Data 2022** — `data/raw/zipcode2022/22zpallagi.csv`

Both are available for free download (Yelp from their academic dataset page, IRS from the Statistics of Income division).

## Running it

Requires Docker.

```bash
docker compose up airflow-init   # first time only — initialises the DB and creates the admin user
docker compose up
```

The Airflow UI will be at `http://localhost:8080` (user: `airflow`, password: `airflow`). Find `main_dag` and trigger it manually.

Output Parquet marts land in `data/processed/`.

## Kaggle (optional)

If you want to pull data via the Kaggle API, drop a `kaggle.json` credentials file in the project root — the scheduler mounts it automatically.

## Dependencies

See `requirements.txt`. Key ones: `pandas`, `pyarrow`, `scikit-learn`, `omegaconf`, `rapidfuzz`.
