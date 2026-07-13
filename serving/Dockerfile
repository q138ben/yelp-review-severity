FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    "transformers==4.46.3" "google-cloud-storage==2.19.0" \
    "fastapi==0.115.5" "uvicorn==0.32.1" "numpy==1.26.4"
# build context is the repo ROOT (see cloudbuild.yaml -f + README) so the shared aspect lexicon ships too
COPY src/ /app/src/
COPY serving/app.py /app/app.py
ENV PORT=8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
