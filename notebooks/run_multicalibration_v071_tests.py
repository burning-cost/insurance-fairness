# Databricks notebook source
# MAGIC %pip install "git+https://github.com/burning-cost/insurance-fairness.git@main" polars scikit-learn scipy statsmodels pytest

# COMMAND ----------

import subprocess, sys, os, tempfile, re

work_dir = tempfile.mkdtemp(prefix="insurance_fairness_v071_")
repo_url = "https://github.com/burning-cost/insurance-fairness.git"
clone = subprocess.run(["git", "clone", repo_url, work_dir], capture_output=True, text=True, timeout=120)
if clone.returncode != 0:
    dbutils.notebook.exit("CLONE_FAILED: " + clone.stderr[-1000:])

# Run only the multicalibration tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(work_dir, "tests", "test_multicalibration.py"),
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True, timeout=600, cwd=work_dir,
)

output = result.stdout + result.stderr
clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
exit_msg = f"EXIT_CODE:{result.returncode}\n" + clean_output[-15000:]
dbutils.notebook.exit(exit_msg)
