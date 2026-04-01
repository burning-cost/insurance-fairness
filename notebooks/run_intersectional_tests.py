# Databricks notebook source
# MAGIC %pip install "git+https://github.com/burning-cost/insurance-fairness.git@main" "dcor>=0.6" polars scipy numpy pytest pandas tabulate

# COMMAND ----------

import subprocess, sys, os, tempfile, re

work_dir = tempfile.mkdtemp(prefix="insurance_fairness_intersectional_")
repo_url = "https://github.com/burning-cost/insurance-fairness.git"
clone = subprocess.run(
    ["git", "clone", repo_url, work_dir],
    capture_output=True, text=True, timeout=120,
)
if clone.returncode != 0:
    dbutils.notebook.exit("CLONE_FAILED: " + clone.stderr[-1000:])

# Install the package with dcor
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[intersectional]", "--quiet"],
    capture_output=True, text=True, timeout=180, cwd=work_dir,
)

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        os.path.join(work_dir, "tests", "test_intersectional.py"),
        "-v", "--tb=short",
    ],
    capture_output=True, text=True, timeout=600, cwd=work_dir,
)

output = result.stdout + result.stderr
clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)

exit_msg = f"EXIT_CODE:{result.returncode}\n" + clean_output[-10000:]
dbutils.notebook.exit(exit_msg)
