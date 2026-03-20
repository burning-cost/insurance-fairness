# Databricks notebook source
# Test insurance-fairness v0.3.8 (MulticalibrationAudit)

# COMMAND ----------

# MAGIC %pip install --upgrade insurance-fairness==0.3.8 pytest

# COMMAND ----------

import insurance_fairness
print("version:", insurance_fairness.__version__)
assert insurance_fairness.__version__ == "0.3.8", f"Expected 0.3.8, got {insurance_fairness.__version__}"

from insurance_fairness import MulticalibrationAudit, MulticalibrationReport
print("MulticalibrationAudit imported OK")

# COMMAND ----------

import numpy as np

rng = np.random.default_rng(42)
n = 3000
exposure = rng.uniform(0.5, 2.0, n)
y_pred = rng.gamma(2.0, 50.0, n)
expected_claims = y_pred * exposure
y_true = rng.poisson(expected_claims) / exposure
protected = rng.integers(0, 2, n).astype(str)

audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20)
report = audit.audit(y_true, y_pred, protected, exposure)

print("is_multicalibrated:", report.is_multicalibrated)
print("overall_pvalue:", round(report.overall_calibration_pvalue, 4))
print("group_calibration:", {k: round(v, 4) for k, v in report.group_calibration.items()})

corrected = audit.correct(y_pred, protected, report, exposure)
print("Correction OK. Mean change:", round(float((corrected / y_pred - 1).mean()), 6))

# COMMAND ----------

# Functional tests inline (don't rely on test discovery)

import dataclasses

# Test 1: MulticalibrationReport is a proper dataclass
assert dataclasses.is_dataclass(report), "Report should be a dataclass"
assert hasattr(report, 'is_multicalibrated')
assert hasattr(report, 'bin_group_table')
assert hasattr(report, 'worst_cells')
assert hasattr(report, 'group_calibration')
print("Test 1 PASSED: report structure OK")

# Test 2: bin_group_table has correct columns
expected_cols = {"bin", "group", "n_obs", "exposure", "observed", "expected", "ae_ratio", "pvalue", "significant", "small_cell"}
assert expected_cols == set(report.bin_group_table.columns), f"Missing columns: {expected_cols - set(report.bin_group_table.columns)}"
print("Test 2 PASSED: bin_group_table columns OK")

# Test 3: worst_cells has <=10 rows
assert report.worst_cells.height <= 10
print("Test 3 PASSED: worst_cells length OK")

# Test 4: well-calibrated model should have high overall pvalue
assert report.overall_calibration_pvalue > 0.01, f"Expected high pvalue, got {report.overall_calibration_pvalue}"
print("Test 4 PASSED: overall calibration pvalue OK")

# Test 5: group_calibration keys match groups
groups_in_data = set(protected.tolist())
assert set(report.group_calibration.keys()) == groups_in_data
print("Test 5 PASSED: group calibration keys OK")

# Test 6: biased model is detected
rng2 = np.random.default_rng(7)
n2 = 4000
exp2 = rng2.uniform(0.5, 2.0, n2)
ypred2 = rng2.gamma(2.0, 50.0, n2)
prot2 = rng2.integers(0, 2, n2).astype(str)
# Group "1" under-reports by factor 1.5
expected2 = ypred2 * exp2
ytrue2 = rng2.poisson(expected2) / exp2
mask1 = prot2 == "1"
ytrue2[mask1] = rng2.poisson(expected2[mask1] / 1.5) / exp2[mask1]
report2 = audit.audit(ytrue2, ypred2, prot2, exp2)
assert not report2.is_multicalibrated, "Biased model should fail multicalibration"
print("Test 6 PASSED: biased model detected")

# Test 7: correction reduces mean abs deviation
audit_small = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20, min_credible=50)
report3 = audit_small.audit(ytrue2, ypred2, prot2, exp2)
corrected3 = audit_small.correct(ypred2, prot2, report3, exp2)
report4 = audit_small.audit(ytrue2, corrected3, prot2, exp2)

import polars as pl
before_dev = float(report3.bin_group_table.filter(~pl.col("small_cell")).select((pl.col("ae_ratio") - 1).abs().mean()).item())
after_dev = float(report4.bin_group_table.filter(~pl.col("small_cell")).select((pl.col("ae_ratio") - 1).abs().mean()).item())
assert after_dev < before_dev, f"Correction should reduce deviation: {before_dev:.4f} -> {after_dev:.4f}"
print(f"Test 7 PASSED: correction reduced deviation {before_dev:.4f} -> {after_dev:.4f}")

# Test 8: single group works
n3 = 300
ypred3 = np.random.default_rng(1).uniform(50, 200, n3)
ytrue3 = np.random.default_rng(1).uniform(50, 200, n3)
prot3 = np.array(["A"] * n3)
audit3 = MulticalibrationAudit(n_bins=5, min_bin_size=10)
report5 = audit3.audit(ytrue3, ypred3, prot3)
assert "A" in report5.group_calibration
assert len(report5.group_calibration) == 1
print("Test 8 PASSED: single group OK")

# Test 9: invalid params raise errors
try:
    MulticalibrationAudit(n_bins=1)
    assert False, "Should raise ValueError"
except ValueError:
    pass
print("Test 9 PASSED: validation errors OK")

# Test 10: n_bins reported correctly
assert report.n_bins == 5
assert report.alpha == 0.05
print("Test 10 PASSED: report metadata OK")

print("\nALL FUNCTIONAL TESTS PASSED: insurance-fairness 0.3.8")
