"""Poll a Vertex AI PipelineJob until terminal, then print per-task states."""
import sys
import time
from pathlib import Path
import yaml
from google.cloud import aiplatform

CFG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
NAME = sys.argv[1]
aiplatform.init(project=CFG["project"], location=CFG["region"])

TERMINAL = {4, 5, 7}  # PipelineState: SUCCEEDED=4, FAILED=5, CANCELLED=7
prev = None
while True:
    job = aiplatform.PipelineJob.get(NAME)
    s = int(job.state)
    if s != prev:
        print(time.strftime("%H:%M:%S"), f"state={s}", flush=True)
        prev = s
    if s in TERMINAL:
        break
    time.sleep(60)

print("=== per-task states ===", flush=True)
for t in (job.task_details or []):
    print(f"  {t.task_name:28} {str(t.state)}", flush=True)
print("PIPELINE_WATCH_DONE", flush=True)
