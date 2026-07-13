"""Vertex AI Pipeline (KFP v2) — Yelp ordinal star-rating MLOps.

train_model=True  : bronze -> silver -> gold -> train -> read-metrics -> [QWK gate] -> deploy Cloud Run
train_model=False : read-metrics(model_dir) -> [QWK gate] -> deploy      (redeploy an existing model, no retrain)

`model_dir` is a pipeline parameter; submit.py passes an IMMUTABLE, versioned path (models/<timestamp>) per run,
so every training run yields a distinct artifact and every Cloud Run deploy is a clean, rollback-able revision.
Downstream steps (read-metrics, deploy) all resolve the same path the training step wrote.

Compile:  python pipelines/pipeline.py   ->   pipelines/compiled/pipeline.yaml
"""
from pathlib import Path
from typing import NamedTuple
import yaml
from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SQL = ROOT / "pipelines" / "medallion"

PROJECT = CFG["project"]
REGION = CFG["region"]
BQLOC = CFG["bq_location"]
BUCKET = CFG["bucket"]
AR = CFG["artifact_registry"]
SA = CFG["service_account"]
GOLD_TABLE = f'{PROJECT}.{CFG["datasets"]["gold"]}.training_data'
TRAIN_IMAGE = f"{AR}/train:latest"
SERVE_IMAGE = f"{AR}/serve:latest"
M = CFG["model"]
DEFAULT_MODEL_DIR = f"{BUCKET}/models/manual"  # fallback for console runs; submit.py passes models/<timestamp>


@dsl.component(base_image="python:3.11-slim", packages_to_install=["google-cloud-storage==2.19.0"])
def read_metrics(model_dir: str) -> NamedTuple("Out", [("qwk", float), ("mae", float)]):
    """Read metrics.json the training job wrote to GCS; expose QWK for the gate."""
    import json
    from collections import namedtuple
    from google.cloud import storage
    bucket_name, prefix = model_dir[5:].split("/", 1)
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix.rstrip('/')}/metrics.json")
    metrics = json.loads(blob.download_as_text())
    print("metrics:", metrics)
    return namedtuple("Out", ["qwk", "mae"])(float(metrics["QWK"]), float(metrics["MAE"]))


@dsl.component(base_image="python:3.11-slim", packages_to_install=["google-cloud-run==0.10.11"])
def deploy_cloud_run(project: str, region: str, service: str, image: str, model_uri: str,
                     service_account: str, cpu: str, memory: str,
                     min_instances: int, max_instances: int):
    """Create/update the Cloud Run scoring service via the Admin API (auth via the pipeline SA).

    Uses run_v2 rather than shelling out to gcloud so it runs reliably as a KFP lightweight component.
    Service is private by default (no allUsers IAM binding).
    """
    from google.cloud import run_v2
    client = run_v2.ServicesClient()
    parent = f"projects/{project}/locations/{region}"
    name = f"{parent}/services/{service}"
    svc = run_v2.Service(template=run_v2.RevisionTemplate(
        service_account=service_account,
        scaling=run_v2.RevisionScaling(min_instance_count=min_instances, max_instance_count=max_instances),
        containers=[run_v2.Container(
            image=image,
            env=[run_v2.EnvVar(name="MODEL_URI", value=model_uri)],
            resources=run_v2.ResourceRequirements(limits={"cpu": cpu, "memory": memory}),
            ports=[run_v2.ContainerPort(container_port=8080)],
        )],
    ))
    try:
        client.get_service(name=name)
        svc.name = name
        op = client.update_service(service=svc)
    except Exception:
        op = client.create_service(parent=parent, service=svc, service_id=service)
    print("deployed:", op.result().uri)


@dsl.container_component
def train_container(model_dir: str):
    """Fine-tune the ordinal encoder from gold.training_data; writes model + metrics.json to model_dir."""
    return dsl.ContainerSpec(
        image=TRAIN_IMAGE,
        command=["python", "/app/train.py"],
        args=["--project", PROJECT, "--gold-table", GOLD_TABLE, "--model-dir", model_dir,
              "--base-encoder", M["base_encoder"], "--max-seq-len", str(M["max_seq_len"]),
              "--train-rows", str(M["train_rows"]), "--epochs", str(M["epochs"]),
              "--patience", str(M["patience"]), "--max-grad-norm", str(M["max_grad_norm"]),
              "--batch-size", str(M["batch_size"]), "--lr", str(M["lr"])],
    )


# wrap the container as a Vertex CustomJob so model_dir flows in as a normal pipeline parameter
train_job = create_custom_training_job_from_component(
    train_container, display_name="train-ordinal-encoder",
    machine_type="e2-highmem-8", replica_count=1)


@dsl.pipeline(name="yelp-ordinal-rating", pipeline_root=f"{BUCKET}/pipeline_root")
def pipeline(train_model: bool = True, model_dir: str = DEFAULT_MODEL_DIR,
             qwk_gate: float = CFG["evaluation"]["qwk_gate"]):

    def eval_and_deploy(mdir, upstream=None):
        m = read_metrics(model_dir=mdir)
        if upstream is not None:
            m.after(upstream)
        with dsl.If(m.outputs["qwk"] >= qwk_gate, name="qwk-gate"):
            deploy_cloud_run(
                project=PROJECT, region=REGION, service=CFG["serving"]["service_name"],
                image=SERVE_IMAGE, model_uri=mdir, service_account=SA,
                cpu=CFG["serving"]["cpu"], memory=CFG["serving"]["memory"],
                min_instances=int(CFG["serving"]["min_instances"]),
                max_instances=int(CFG["serving"]["max_instances"]),
            ).set_caching_options(False)

    with dsl.If(train_model == True, name="train-and-deploy"):
        bronze = BigqueryQueryJobOp(project=PROJECT, location=BQLOC, query=(SQL / "01_bronze.sql").read_text())
        bronze.set_display_name("bronze-load")
        silver = BigqueryQueryJobOp(project=PROJECT, location=BQLOC,
                                    query=(SQL / "02_silver.sql").read_text()).after(bronze)
        silver.set_display_name("silver-clean")
        gold = BigqueryQueryJobOp(project=PROJECT, location=BQLOC,
                                  query=(SQL / "03_gold.sql").read_text()).after(silver)
        gold.set_display_name("gold-features")
        train = train_job(project=PROJECT, location=REGION, model_dir=model_dir).after(gold)
        eval_and_deploy(model_dir, upstream=train)
    with dsl.Else(name="deploy-only"):
        eval_and_deploy(model_dir)


if __name__ == "__main__":
    out = str(ROOT / "pipelines" / "compiled" / "pipeline.yaml")
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=out)
    print("compiled ->", out)
