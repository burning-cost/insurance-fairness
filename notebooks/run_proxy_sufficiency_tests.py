# Databricks notebook source
# MAGIC %pip install polars scikit-learn scipy pytest

# COMMAND ----------

import subprocess, sys, os, tempfile, shutil, re, base64

work_dir = tempfile.mkdtemp(prefix="proxy_sufficiency_tests_")

# Source is uploaded to the workspace; copy to a local temp dir
src_ws_path = "/Workspace/Users/pricing.frontier@gmail.com/proxy_sufficiency_src"
shutil.copytree(src_ws_path, work_dir, dirs_exist_ok=True)

# Install the package in development mode
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", work_dir, "--quiet"],
    capture_output=True, text=True, timeout=120, cwd=work_dir,
)
if install.returncode != 0:
    dbutils.notebook.exit("INSTALL_FAILED: " + install.stderr[-2000:])

# Run the multicalibration tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(work_dir, "tests", "test_multicalibration.py"),
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True, timeout=600, cwd=work_dir,
)

output = result.stdout + result.stderr
clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output)
exit_msg = f"EXIT_CODE:{result.returncode}\n" + clean_output[-12000:]
dbutils.notebook.exit(exit_msg)
