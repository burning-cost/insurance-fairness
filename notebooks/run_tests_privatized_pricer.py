"""
Run insurance-fairness v0.8.0 tests (PrivatizedFairPricer) on Databricks serverless.

Usage (from repo root):
    python notebooks/run_tests_privatized_pricer.py
"""
import os
import sys
import time
import base64
import pathlib

env_path = pathlib.Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_DIR = "/Workspace/insurance-fairness"
REPO_ROOT = pathlib.Path(__file__).parent.parent


def upload_file(local_path: pathlib.Path, remote_path: str):
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ImportFormat.AUTO,
        overwrite=True,
    )


def upload_tree(local_dir: pathlib.Path, remote_dir: str, extensions=(".py", ".toml", ".cfg")):
    for fpath in sorted(local_dir.rglob("*")):
        if fpath.is_file() and fpath.suffix in extensions:
            rel = fpath.relative_to(local_dir)
            remote = f"{remote_dir}/{str(rel).replace(chr(92), '/')}"
            remote_parent = remote.rsplit("/", 1)[0]
            try:
                w.workspace.mkdirs(remote_parent)
            except Exception:
                pass
            upload_file(fpath, remote)


try:
    w.workspace.mkdirs(WORKSPACE_DIR)
except Exception:
    pass

print("Uploading source files...")
upload_tree(REPO_ROOT / "src", f"{WORKSPACE_DIR}/src")
upload_tree(REPO_ROOT / "tests", f"{WORKSPACE_DIR}/tests")
try:
    upload_file(REPO_ROOT / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")
    upload_file(REPO_ROOT / "README.md", f"{WORKSPACE_DIR}/README.md")
except Exception as e:
    print(f"Root file upload issue: {e}")
print("Upload complete.")

NOTEBOOK_CONTENT = '''
import subprocess, sys, shutil, os

lines = []

def run(cmd, desc="", env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (r.stdout or "")[-8000:]
    err = (r.stderr or "")[-1000:]
    lines.append(f"=== {desc} rc={r.returncode} ===")
    if out:
        lines.append(out)
    if err and r.returncode != 0:
        lines.append("STDERR: " + err[-2000:])
    return r.returncode

# Install dependencies
rc1 = run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "catboost>=1.2", "scikit-learn>=1.3", "numpy>=1.24",
     "polars>=1.0", "scipy>=1.11", "pyarrow>=12", "jinja2>=3.1",
     "statsmodels>=0.14", "networkx>=3.0", "POT>=0.9", "pytest>=7.4"],
    "pip install deps"
)

# Install package
rc2 = run(
    [sys.executable, "-m", "pip", "install", "--quiet", "-e",
     "/Workspace/insurance-fairness"],
    "pip install package"
)
if rc2 != 0:
    dbutils.notebook.exit("FAILED: package install\\n" + "\\n".join(lines))

# Copy tests to writable location
test_src = "/Workspace/insurance-fairness/tests"
test_dst = "/tmp/insurance_fairness_tests_v080"
if os.path.exists(test_dst):
    shutil.rmtree(test_dst)
shutil.copytree(test_src, test_dst)
lines.append(f"Copied tests to {test_dst}")

env = dict(os.environ)
env["PYTHONDONTWRITEBYTECODE"] = "1"

# Run just the new pricer tests first
rc_pricer = run(
    [sys.executable, "-m", "pytest",
     f"{test_dst}/test_privatized_pricer.py",
     "-v", "--tb=short",
     "-p", "no:cacheprovider",
     "--import-mode=importlib"],
    "pytest test_privatized_pricer",
    env=env,
)

# Then run full suite
rc_full = run(
    [sys.executable, "-m", "pytest",
     test_dst,
     "-x", "-q", "--tb=short",
     "-p", "no:cacheprovider",
     "--import-mode=importlib"],
    "pytest full suite",
    env=env,
)

output = "\\n".join(lines)
if rc_pricer != 0 or rc_full != 0:
    dbutils.notebook.exit("FAILED\\n" + output)
else:
    dbutils.notebook.exit("PASSED\\n" + output)
'''

notebook_path = f"{WORKSPACE_DIR}/run_pytest_v080"
encoded = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=notebook_path,
    content=encoded,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Uploaded test runner: {notebook_path}")

print("Submitting Databricks serverless job run...")
run_resp = w.jobs.submit(
    run_name="insurance-fairness-v080-privatized-pricer-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run-pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
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
            ),
        )
    ],
)

run_id = run_resp.run_id
print(f"Run submitted: run_id={run_id}")
host = os.environ["DATABRICKS_HOST"].rstrip("/")
print(f"View at: {host}#job/runs/{run_id}")

print("Polling for completion...")
while True:
    state = w.jobs.get_run(run_id=run_id)
    life_cycle = str(state.state.life_cycle_state)
    result_state = str(state.state.result_state)
    print(f"  status: {life_cycle} / {result_state}")
    if any(x in life_cycle for x in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR")):
        break
    time.sleep(20)

task_run_id = None
if state.tasks:
    task_run_id = state.tasks[0].run_id

if task_run_id:
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output and output.notebook_output.result:
            print("\n--- Test output ---")
            print(output.notebook_output.result[:20000])
        if output.error:
            print("ERROR:", output.error)
        if output.error_trace:
            print("TRACE:", output.error_trace[:2000])
    except Exception as e:
        print(f"Could not fetch output: {e}")

final_state = w.jobs.get_run(run_id=run_id)
result_state_str = str(final_state.state.result_state)
print(f"\nFinal result: {result_state_str}")

if "SUCCESS" not in result_state_str:
    sys.exit(1)

print("SUCCESS: All tests passed on Databricks.")
