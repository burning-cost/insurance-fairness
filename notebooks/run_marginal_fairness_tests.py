# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-fairness v0.5.0 — MarginalFairnessPremium Tests

# COMMAND ----------

import subprocess, sys, os, shutil, tempfile, json

BASE = tempfile.mkdtemp(prefix="ins_fair_")
os.makedirs(f"{BASE}/src/insurance_fairness", exist_ok=True)
os.makedirs(f"{BASE}/tests", exist_ok=True)
print(f"Working in: {BASE}")

# Copy only the modules needed: marginal_fairness + _utils (if exists)
workspace_src = "/Workspace/insurance-fairness-v050/src/insurance_fairness"
local_src = f"{BASE}/src/insurance_fairness"

needed = {"marginal_fairness.py", "_utils.py", "__init__.py"}
for fname in needed:
    src_path = os.path.join(workspace_src, fname)
    if os.path.isfile(src_path):
        shutil.copy2(src_path, os.path.join(local_src, fname))
        print(f"  copied {fname}")

# Write a minimal __init__.py that ONLY imports marginal_fairness
# This avoids circular import issues with missing subpackages
minimal_init = '''from insurance_fairness.marginal_fairness import MarginalFairnessPremium, MarginalFairnessReport
__version__ = "0.5.0"
__all__ = ["MarginalFairnessPremium", "MarginalFairnessReport"]
'''
with open(f"{local_src}/__init__.py", "w") as f:
    f.write(minimal_init)
print("  wrote minimal __init__.py")

# Copy test
workspace_tests = "/Workspace/insurance-fairness-v050/tests"
test_file = "test_marginal_fairness.py"
shutil.copy2(os.path.join(workspace_tests, test_file), os.path.join(f"{BASE}/tests", test_file))
print(f"  copied tests/{test_file}")

# COMMAND ----------

# Write pyproject.toml
pyproject = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "insurance-fairness-test"
version = "0.5.0"
requires-python = ">=3.10"
dependencies = []

[tool.hatch.build.targets.wheel]
packages = ["src/insurance_fairness"]
"""
with open(f"{BASE}/pyproject.toml", "w") as f:
    f.write(pyproject)

# Install dependencies only (no package install needed — just sys.path)
r = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "numpy", "scipy", "scikit-learn", "pytest", "-q", "--disable-pip-version-check"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("pip stderr:", r.stderr[-1000:])
else:
    print("Dependencies installed")

# Add src to path
sys.path.insert(0, f"{BASE}/src")
print(f"sys.path prepended: {BASE}/src")

# COMMAND ----------

# Verify import works
from insurance_fairness.marginal_fairness import MarginalFairnessPremium, MarginalFairnessReport
print("Import OK:", MarginalFairnessPremium)

# COMMAND ----------

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{BASE}/tests/test_marginal_fairness.py",
     "-v", "--tb=long", "--no-header",
     f"--rootdir={BASE}"],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONPATH": f"{BASE}/src"},
    cwd=BASE
)
output = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr.strip() else "")
print(output[-10000:] if len(output) > 10000 else output)

dbutils.notebook.exit(json.dumps({"returncode": result.returncode, "summary": output[-4000:]}))
