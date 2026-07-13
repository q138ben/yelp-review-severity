"""Build a request file for the rating API from held-out test data.

Draws a deterministic, class-balanced sample (N per star) from
gold.training_data (split='test') and writes:
  - sample_request.json   {"texts": [...]}       -> POST body for /predict
  - sample_expected.json  {"stars": [...], ...}  -> true labels for comparison

Requires: pip install google-cloud-bigquery
"""
import json
import os
from google.cloud import bigquery

PROJECT = "yelp-review-rating-prediction"
PER_STAR = int(os.environ.get("PER_STAR", "2"))
HERE = os.path.dirname(os.path.abspath(__file__))

sql = f"""
SELECT review_id, stars, text FROM `{PROJECT}.gold.training_data`
WHERE split='test' AND text_len BETWEEN 60 AND 240
QUALIFY ROW_NUMBER() OVER (PARTITION BY stars ORDER BY FARM_FINGERPRINT(CAST(review_id AS STRING))) <= {PER_STAR}
ORDER BY stars
"""

rows = list(bigquery.Client(project=PROJECT).query(sql).result())
texts = [r["text"] for r in rows]
stars = [int(r["stars"]) for r in rows]
ids = [r["review_id"] for r in rows]

with open(os.path.join(HERE, "sample_request.json"), "w") as f:
    json.dump({"texts": texts}, f, indent=2, ensure_ascii=False)

with open(os.path.join(HERE, "sample_expected.json"), "w") as f:
    json.dump({"review_id": ids, "stars": stars}, f, indent=2, ensure_ascii=False)

print(f"wrote {len(texts)} reviews ({PER_STAR} per star) to sample_request.json")
