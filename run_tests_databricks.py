"""Run insurance-fairness tests on Databricks."""
import os
import time
import base64

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

w = WorkspaceClient()

notebook_content = r'''
import subprocess, sys

# Install package from workspace
r = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "/Workspace/insurance-fairness/",
     "insurance-fairness[intersectional]",
     "--quiet", "--no-deps"],
    capture_output=True, text=True
)
# Install deps separately
r2 = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "dcor", "polars", "scipy", "scikit-learn", "pandas",
     "numpy", "pytest", "--quiet"],
    capture_output=True, text=True
)
print("Install done")
print(r.stderr[-500:] if r.stderr else "")
print(r2.stderr[-500:] if r2.stderr else "")

# Run new tests first
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-fairness/tests/test_coverage_expansion.py",
     "-v", "--tb=short", "-q"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-fairness/"
)
print("=== NEW TESTS ===")
print(result.stdout[-6000:])
print(result.stderr[-1000:])
print("RC:", result.returncode)

# Run full suite
result2 = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-fairness/tests/",
     "-x", "-q", "--tb=line"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-fairness/"
)
print("=== FULL SUITE ===")
print(result2.stdout[-8000:])
print("RC:", result2.returncode)
'''

notebook_path = "/Workspace/Shared/run-fairness-tests"
w.workspace.import_(
    path=notebook_path,
    format="SOURCE",
    language="PYTHON",
    content=base64.b64encode(notebook_content.encode()).decode(),
    overwrite=True,
)
print(f"Notebook at {notebook_path}")

run = w.jobs.submit(
    run_name="fairness-test-expansion",
    tasks=[jobs.SubmitTask(
        task_key="tests",
        notebook_task=jobs.NotebookTask(notebook_path=notebook_path),
        new_cluster=compute.ClusterSpec(
            spark_version="15.4.x-scala2.12",
            node_type_id="m5d.large",
            num_workers=0,
            spark_conf={"spark.master": "local[*, 4]"},
            data_security_mode=compute.DataSecurityMode.SINGLE_USER,
        ),
    )],
).result()

print(f"run_id={run.run_id}")

for _ in range(60):
    rs = w.jobs.get_run(run_id=run.run_id)
    life = rs.state.life_cycle_state if rs.state else None
    print(f"  {life}")
    if life and life.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(15)

try:
    out = w.jobs.get_run_output(run_id=run.run_id)
    if out.notebook_output:
        print(out.notebook_output.result)
    if out.error:
        print("ERR:", out.error)
except Exception as e:
    print(f"output error: {e}")
