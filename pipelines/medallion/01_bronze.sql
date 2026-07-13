-- BRONZE: land the raw GCS JSON into BigQuery tables (idempotent OVERWRITE).
-- Run by the pipeline as a BigQuery job. Uses BQ's LOAD DATA ... FROM FILES (no bash bq load).
-- attributes/hours kept as JSON to absorb the schema drift in business.json; silver ignores them.

LOAD DATA OVERWRITE bronze.reviews_raw (
  review_id STRING, user_id STRING, business_id STRING,
  stars FLOAT64, useful INT64, funny INT64, cool INT64,
  text STRING, date STRING
)
FROM FILES (
  format = 'JSON',
  uris = ['gs://yelp-review-rating-prediction-mlops/bronze/review/*.json']
);

LOAD DATA OVERWRITE bronze.business_raw (
  business_id STRING, name STRING, address STRING, city STRING, state STRING,
  postal_code STRING, latitude FLOAT64, longitude FLOAT64,
  stars FLOAT64, review_count INT64, is_open INT64,
  categories STRING, attributes JSON, hours JSON
)
FROM FILES (
  format = 'JSON',
  uris = ['gs://yelp-review-rating-prediction-mlops/bronze/business/*.json']
);
