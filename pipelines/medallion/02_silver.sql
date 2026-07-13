-- SILVER: cleaned, typed, deduped. Runs with `bq query --project_id=<PROJECT>`.
-- Tables are dataset-qualified (no project prefix) so they resolve to the job's default project.

-- Reviews: type-cast, parse timestamp, drop empty/null text, dedup exact-duplicate text,
-- and stamp the leakage-safe business-level split (deterministic via FARM_FINGERPRINT).
CREATE OR REPLACE TABLE silver.reviews
CLUSTER BY split AS
WITH deduped AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY FARM_FINGERPRINT(text) ORDER BY date) AS rn
  FROM bronze.reviews_raw
  WHERE text IS NOT NULL AND LENGTH(TRIM(text)) > 0 AND stars IS NOT NULL
)
SELECT
  review_id,
  business_id,
  user_id,
  CAST(stars AS INT64)                                   AS stars,
  useful, funny, cool,
  text,
  LENGTH(text)                                          AS text_len,
  SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', date)       AS review_ts,
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(business_id)), 100) < 70 THEN 'train'
    WHEN MOD(ABS(FARM_FINGERPRINT(business_id)), 100) < 85 THEN 'val'
    ELSE 'test'
  END                                                   AS split
FROM deduped
WHERE rn = 1;

-- Business: keep the modelling-relevant columns; derive the restaurant flag (EDA: 73% of reviews).
CREATE OR REPLACE TABLE silver.business AS
SELECT
  business_id,
  name, city, state, postal_code,
  latitude, longitude,
  stars                                                 AS business_avg_stars,
  review_count,
  is_open,
  categories,
  (LOWER(categories) LIKE '%restaurant%' OR LOWER(categories) LIKE '%food%') AS is_restaurant
FROM bronze.business_raw
WHERE business_id IS NOT NULL;
