"""Submit a compiled Vertex AI Pipeline to run on GCP.

  python pipelines/submit.py                                   # full train+deploy pipeline
  python pipelines/submit.py --param train_model=False \
      --param model_dir=gs://.../models/<run>                  # deploy-only (no retrain)
"""
import argparse
import datetime
from pathlib import Path
import yaml
from google.cloud import aiplatform

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

ap = argparse.ArgumentParser()
ap.add_argument("--template", default=str(ROOT / "pipelines" / "compiled" / "pipeline.yaml"))
ap.add_argument("--param", action="append", default=[], help="key=value (repeatable)")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--dry-run", action="store_true", help="resolve params + print, do not submit")
args = ap.parse_args()

def coerce(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


params = {k: coerce(v) for k, v in (p.split("=", 1) for p in args.param)}

# Immutable, versioned model_dir: each training run writes a distinct artifact, so every Cloud Run deploy
# is a clean, rollback-able revision. Deploy-only runs must name the exact version to redeploy.
if "model_dir" not in params:
    if params.get("train_model") is False:
        raise SystemExit("deploy-only (train_model=False) needs --param model_dir=gs://.../models/<version>")
    params["model_dir"] = f'{CFG["bucket"]}/models/{datetime.datetime.now():%Y%m%d-%H%M%S}'
print("template:", Path(args.template).name, "| params:", params)
if args.dry_run:
    raise SystemExit(0)

# local auth = ambient gcloud ADC (gcloud auth application-default login); pipeline runs as the SA below
aiplatform.init(project=CFG["project"], location=CFG["region"], staging_bucket=CFG["bucket"])
job = aiplatform.PipelineJob(
    display_name=Path(args.template).stem,
    template_path=args.template,
    pipeline_root=f'{CFG["bucket"]}/pipeline_root',
    parameter_values=params,
    enable_caching=not args.no_cache,
)
job.submit(service_account=CFG["service_account"])
print("submitted:", job.resource_name)
