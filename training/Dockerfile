FROM python:3.11-slim
WORKDIR /app
# CPU torch (no GPU quota needed on a fresh trial project)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    "transformers==4.46.3" "google-cloud-bigquery==3.27.0" "google-cloud-storage==2.19.0" \
    "pandas==2.2.3" "scikit-learn==1.5.2" "scipy==1.14.1" "db-dtypes==1.3.1" "numpy==1.26.4"
# build context is the repo ROOT (see cloudbuild.yaml -f + README) so the shared training core ships too
COPY src/ /app/src/
COPY training/train.py /app/train.py
ENTRYPOINT ["python", "train.py"]
