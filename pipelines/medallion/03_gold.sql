-- GOLD: ML-ready training set. Text -> stars (ordinal label), leakage-safe split, restaurant context.
-- Deterministic downsample to a tractable, class-preserving set (~100k/20k/20k) via FARM_FINGERPRINT.
-- Identifiers (review_id/business_id) are kept for traceability only, NOT as model features.
-- user_avg / business_avg stars are deliberately EXCLUDED (they leak the label — see notebook 02).
CREATE OR REPLACE TABLE gold.training_data
CLUSTER BY split AS
SELECT
  r.review_id,
  r.business_id,
  r.text,
  r.stars,
  r.text_len,
  COALESCE(b.is_restaurant, FALSE) AS is_restaurant,
  r.split
FROM silver.reviews r
LEFT JOIN silver.business b USING(business_id)
WHERE
  (r.split = 'train'          AND MOD(ABS(FARM_FINGERPRINT(r.review_id)), 100000) < 2100)
  OR (r.split IN ('val','test') AND MOD(ABS(FARM_FINGERPRINT(r.review_id)), 100000) < 1950);

-- Monitoring view: label distribution per split (feeds a data-quality/drift check).
CREATE OR REPLACE VIEW gold.label_distribution AS
SELECT split, stars, COUNT(*) AS n,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 2) AS pct
FROM gold.training_data
GROUP BY split, stars
ORDER BY split, stars;
