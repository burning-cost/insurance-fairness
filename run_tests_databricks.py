"""Run insurance-fairness tests on Databricks (serverless compute).

Usage:
    /home/ralph/burning-cost/.venv/bin/python run_tests_databricks.py
"""
from __future__ import annotations

import os
import sys
import time
import base64
import pathlib

# Load credentials
env_path = pathlib.Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, workspace as ws_svc

w = WorkspaceClient()

WORKSPACE_BASE = "/Workspace/insurance-fairness-tests"
REPO_ROOT = pathlib.Path(__file__).parent

# ── upload files ──────────────────────────────────────────────────────────────

def upload_file(local_path: pathlib.Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = str(pathlib.Path(remote_path).parent)
    try:
        w.workspace.mkdirs(parent)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ws_svc.ImportFormat.AUTO,
    )


def upload_directory(local_dir: pathlib.Path, remote_dir: str) -> None:
    for p in sorted(local_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in str(p) and ".pyc" not in p.name:
            rel = p.relative_to(local_dir)
            remote = f"{remote_dir}/{rel}"
            print(f"  {p.relative_to(REPO_ROOT)} -> {remote}")
            upload_file(p, remote)


print("Uploading source files...")
upload_directory(REPO_ROOT / "src", f"{WORKSPACE_BASE}/src")
upload_directory(REPO_ROOT / "tests", f"{WORKSPACE_BASE}/tests")
upload_file(REPO_ROOT / "pyproject.toml", f"{WORKSPACE_BASE}/pyproject.toml")
print("Files uploaded.")

# ── notebook content ──────────────────────────────────────────────────────────

NOTEBOOK_CONTENT = """\
# Databricks notebook source
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-fairness-tests/", "--quiet"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-fairness-tests"
)
if result.stdout:
    print(result.stdout[-2000:])
if result.stderr:
    print(result.stderr[-2000:])
if result.returncode != 0:
    raise SystemExit("pip install failed")

# COMMAND ----------
import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-fairness-tests/tests/",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-fairness-tests"
)
output = result.stdout or ""
if len(output) > 10000:
    print(output[:2000])
    print("... [truncated] ...")
    print(output[-8000:])
else:
    print(output)
if result.stderr:
    print(result.stderr[-2000:])
if result.returncode != 0:
    raise SystemExit(f"Tests failed with exit code {result.returncode}")
"""

NOTEBOOK_PATH = f"{WORKSPACE_BASE}/run_tests"

print("Uploading test notebook...")
encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
try:
    w.workspace.mkdirs(WORKSPACE_BASE)
except Exception:
    pass
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=encoded_nb,
    overwrite=True,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
)
print(f"Notebook at {NOTEBOOK_PATH}")

# ── create job and run ────────────────────────────────────────────────────────
# Serverless pattern: environment_key + JobEnvironment, no new_cluster

JOB_NAME = "insurance-fairness-seq-ot-tests"

print("Creating job...")
created = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs.Task(
            task_key="pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                source=jobs.Source.WORKSPACE,
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=jobs.compute.Environment(
                client="2",
                dependencies=[
                    "catboost",
                    "pymoo>=0.6.1",
                    "polars",
                    "scikit-learn",
                    "scipy",
                    "jinja2",
                    "pyarrow",
                    "pytest",
                    "networkx",
                ],
            ),
        )
    ],
)
job_id = created.job_id
print(f"Job ID: {job_id}")

print("Running job...")
run_resp = w.jobs.run_now(job_id=job_id)
run_id = run_resp.run_id
print(f"Run ID: {run_id}")
print(f"Tracking: {os.environ['DATABRICKS_HOST']}#job/{job_id}/run/{run_id}")

# ── poll until done ───────────────────────────────────────────────────────────

print("Waiting for run to complete...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state
    print(f"  state: {life_cycle}")
    if life_cycle in (
        jobs.RunLifeCycleState.TERMINATED,
        jobs.RunLifeCycleState.SKIPPED,
        jobs.RunLifeCycleState.INTERNAL_ERROR,
    ):
        break
    time.sleep(20)

result_state = run_state.state.result_state
print(f"Final result: {result_state}")

# Print task output
for task in (run_state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output:
            print("\n--- Notebook Output ---")
            print(output.notebook_output.result)
        if output.error:
            print(f"\n--- Error ---\n{output.error}")
        if output.error_trace:
            trace = output.error_trace
            print(f"\n--- Trace (last 5000 chars) ---\n{trace[-5000:]}")
    except Exception as e:
        print(f"Could not fetch output: {e}")

# Clean up job
try:
    w.jobs.delete(job_id=job_id)
    print(f"Deleted job {job_id}")
except Exception:
    pass

if result_state == jobs.RunResultState.SUCCESS:
    print("\nAll tests passed.")
    sys.exit(0)
else:
    print("\nTests FAILED.")
    sys.exit(1)
