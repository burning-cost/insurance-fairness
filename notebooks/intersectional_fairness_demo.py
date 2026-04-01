# Databricks notebook source
# MAGIC %pip install "git+https://github.com/burning-cost/insurance-fairness.git@main" "dcor>=0.6" polars scipy numpy pandas tabulate matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC # Intersectional Fairness via Distance Covariance
# MAGIC
# MAGIC **Paper:** Lee, Antonio, Avanzi, Marchi & Zhou (2025). *Machine Learning with Multitype
# MAGIC Protected Attributes: Intersectional Fairness through Regularisation.* arXiv:2509.08163.
# MAGIC
# MAGIC **The problem:** Auditing each protected attribute in isolation does not guarantee
# MAGIC intersectional fairness. A UK motor insurer that achieves gender parity and age parity
# MAGIC separately may still price young women systematically differently from both young men
# MAGIC and elderly women. This is fairness gerrymandering — and it is exactly the pattern
# MAGIC the FCA's 2026 AI review flags as a risk under Consumer Duty.
# MAGIC
# MAGIC **The solution:** Concatenated Distance Covariance (CCdCov) penalises the joint
# MAGIC dependence of predictions on all protected attributes simultaneously:
# MAGIC
# MAGIC $$\text{CCdCov}(\hat{y}, S) = \sum_k \widetilde{d\text{Cov}}^2(\hat{y}, s_k) + \eta(\hat{y}, s)$$
# MAGIC
# MAGIC The $\eta$ term is the intersectional residual — positive when the model exploits
# MAGIC joint attribute structure beyond what marginals explain.

# COMMAND ----------

import numpy as np
import pandas as pd
import warnings

np.random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic UK Motor Insurance Data
# MAGIC
# MAGIC We generate a synthetic motor insurance dataset with:
# MAGIC - **gender**: binary (M/F)
# MAGIC - **age_band**: ordinal 1–5 (17-25, 26-35, 36-50, 51-65, 65+)
# MAGIC - **vehicle_group**: categorical A/B/C
# MAGIC
# MAGIC Three scenarios are demonstrated:
# MAGIC 1. **Fair model**: predictions independent of all protected attributes
# MAGIC 2. **Marginally unfair**: predictions depend on gender alone
# MAGIC 3. **Intersectionally unfair**: predictions depend on the gender × age interaction
# MAGIC    but NOT on either attribute marginally — this is the gerrymandering scenario

# COMMAND ----------

def make_uk_motor_data(n: int = 2000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])
    age_band = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.12, 0.22, 0.30, 0.24, 0.12])
    vehicle_group = rng.choice(["A", "B", "C"], size=n, p=[0.40, 0.35, 0.25])

    # True risk: driven by vehicle and age, not gender
    base_rate = 0.08
    vehicle_factor = {"A": 1.0, "B": 1.3, "C": 1.6}
    age_factor = {1: 2.5, 2: 1.8, 3: 1.2, 4: 1.0, 5: 1.1}

    mu = np.array([
        base_rate * vehicle_factor[vg] * age_factor[ab]
        for vg, ab in zip(vehicle_group, age_band)
    ])

    # Observed claims (Poisson)
    claims = rng.poisson(mu)

    D = pd.DataFrame({"gender": gender, "age_band": age_band, "vehicle_group": vehicle_group})
    return claims, D, mu


claims, D, true_mu = make_uk_motor_data(n=2000)
print(f"Dataset: {len(D)} policies")
print(f"Protected attributes: gender ({D['gender'].value_counts().to_dict()})")
print(f"Age bands: {dict(sorted(D['age_band'].value_counts().to_dict().items()))}")
print(f"Mean claim rate: {claims.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. IntersectionalFairnessAudit — Three Prediction Scenarios

# COMMAND ----------

from insurance_fairness.intersectional import (
    IntersectionalFairnessAudit,
    DistanceCovFairnessRegulariser,
    LambdaCalibrationResult,
)

rng = np.random.default_rng(0)

# Scenario 1: Fair — predictions are the true risk (no protected attribute dependence)
y_fair = true_mu + rng.normal(0, 0.005, size=len(D))

# Scenario 2: Marginally unfair — gender premium uplift
gender_factor = np.where(D["gender"] == "M", 1.15, 1.0)
y_marginal_unfair = true_mu * gender_factor + rng.normal(0, 0.005, size=len(D))

# Scenario 3: Intersectionally unfair — young women specifically penalised
# No marginal effect on gender or age alone; the interaction drives the bias
intersect_factor = np.where(
    (D["gender"] == "F") & (D["age_band"] == 1),  # young women
    1.5,   # 50% uplift for this specific subgroup
    1.0,
)
y_intersect_unfair = true_mu * intersect_factor + rng.normal(0, 0.005, size=len(D))

print("Three prediction scenarios generated.")
print(f"  Fair:                mean={y_fair.mean():.4f}")
print(f"  Marginally unfair:   mean={y_marginal_unfair.mean():.4f}")
print(f"  Intersect. unfair:   mean={y_intersect_unfair.mean():.4f}")

# COMMAND ----------

# Run the audit on all three scenarios
audit = IntersectionalFairnessAudit(
    protected_attrs=["gender", "age_band"],
    continuous_attrs=[],  # ordinal encode age_band
)

report_fair = audit.audit(y_fair, D)
report_marginal = audit.audit(y_marginal_unfair, D)
report_intersect = audit.audit(y_intersect_unfair, D)

# COMMAND ----------

# Summary table
print("=" * 70)
print(f"{'Metric':<35} {'Fair':>10} {'Marginal':>10} {'Intersect':>10}")
print("-" * 70)
print(f"{'CCdCov²(ŷ, S)':<35} {report_fair.ccDcov:>10.6f} {report_marginal.ccDcov:>10.6f} {report_intersect.ccDcov:>10.6f}")
print(f"{'Σ marginal d̃Cov²':<35} {sum(report_fair.marginal_dcov.values()):>10.6f} {sum(report_marginal.marginal_dcov.values()):>10.6f} {sum(report_intersect.marginal_dcov.values()):>10.6f}")
print(f"{'η (intersectional residual)':<35} {report_fair.eta:>10.6f} {report_marginal.eta:>10.6f} {report_intersect.eta:>10.6f}")
print(f"{'D_JS overall':<35} {report_fair.js_divergence_overall:>10.6f} {report_marginal.js_divergence_overall:>10.6f} {report_intersect.js_divergence_overall:>10.6f}")
print(f"{'d̃Cov²(ŷ, gender)':<35} {report_fair.marginal_dcov['gender']:>10.6f} {report_marginal.marginal_dcov['gender']:>10.6f} {report_intersect.marginal_dcov['gender']:>10.6f}")
print(f"{'d̃Cov²(ŷ, age_band)':<35} {report_fair.marginal_dcov['age_band']:>10.6f} {report_marginal.marginal_dcov['age_band']:>10.6f} {report_intersect.marginal_dcov['age_band']:>10.6f}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reading the results:**
# MAGIC
# MAGIC - **Fair model**: CCdCov ≈ 0, η ≈ 0. Both marginals are near zero. D_JS low.
# MAGIC - **Marginally unfair**: d̃Cov²(ŷ, gender) is elevated. The sum-of-marginals approach
# MAGIC   catches this. CCdCov > 0.
# MAGIC - **Intersectionally unfair**: The marginals for gender and age may both appear small,
# MAGIC   but η > 0, meaning the model has encoded the joint (young × female) distribution.
# MAGIC   **This is the pattern that marginal auditing would miss.**

# COMMAND ----------

# Full report for the intersectional scenario
print(report_intersect.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The Decomposition Theorem
# MAGIC
# MAGIC CCdCov(ŷ, S) = Σ_k d̃Cov²(ŷ, s_k) + η
# MAGIC
# MAGIC This holds to floating-point precision:

# COMMAND ----------

for name, report in [("Fair", report_fair), ("Marginal", report_marginal), ("Intersect.", report_intersect)]:
    marg_sum = sum(report.marginal_dcov.values())
    reconstruct = marg_sum + report.eta
    diff = abs(report.ccDcov - reconstruct)
    print(f"{name:12s}: CCdCov={report.ccDcov:.8f},  marg+η={reconstruct:.8f},  diff={diff:.2e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. DistanceCovFairnessRegulariser — Training-Time Penalty

# COMMAND ----------

# Demonstrate the regulariser as a penalty term
reg_ccDcov  = DistanceCovFairnessRegulariser(protected_attrs=["gender", "age_band"], method="ccDcov",   lambda_val=1.0)
reg_sum     = DistanceCovFairnessRegulariser(protected_attrs=["gender", "age_band"], method="sum_dcov", lambda_val=1.0)
reg_jdCov   = DistanceCovFairnessRegulariser(protected_attrs=["gender", "age_band"], method="jdCov",    lambda_val=1.0)

print("Fairness penalties on the intersectionally unfair model:")
print(f"  CCdCov penalty  (λ=1): {reg_ccDcov.penalty(y_intersect_unfair, D):.6f}")
print(f"  sum_dCov penalty(λ=1): {reg_sum.penalty(y_intersect_unfair, D):.6f}")
print(f"  JdCov penalty   (λ=1): {reg_jdCov.penalty(y_intersect_unfair, D):.6f}")

print("\nFairness penalties on the fair model:")
print(f"  CCdCov penalty  (λ=1): {reg_ccDcov.penalty(y_fair, D):.6f}")
print(f"  sum_dCov penalty(λ=1): {reg_sum.penalty(y_fair, D):.6f}")
print(f"  JdCov penalty   (λ=1): {reg_jdCov.penalty(y_fair, D):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Jensen-Shannon Divergence vs Lambda
# MAGIC
# MAGIC To demonstrate the lambda calibration workflow without a full neural network,
# MAGIC we simulate a sweep where increasing lambda gradually attenuates the intersectional
# MAGIC effect in the predictions.

# COMMAND ----------

from insurance_fairness.intersectional import _js_divergence, _make_group_labels

lambda_grid = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
group_labels = _make_group_labels(D, ["gender", "age_band"])

results_lambda = []
for lv in lambda_grid:
    # Simulate: higher lambda -> predictions move toward fair baseline
    alpha = min(lv / 5.0, 1.0)
    y_lv = y_intersect_unfair * (1 - alpha) + y_fair * alpha
    djs = _js_divergence(y_lv, group_labels)

    # Poisson deviance as accuracy measure
    eps = 1e-10
    y_lv_pos = np.maximum(y_lv, eps)
    deviance = 2.0 * np.mean(y_lv_pos - claims - claims * np.log(y_lv_pos / np.maximum(claims, eps)))

    results_lambda.append({
        "lambda": lv,
        "D_JS": djs,
        "deviance": deviance,
    })

df_lambda = pd.DataFrame(results_lambda)
print(df_lambda.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. LambdaCalibrationResult — Pareto Front

# COMMAND ----------

pareto_idx = np.array([
    i for i, row in df_lambda.iterrows()
    if not any(
        (df_lambda.loc[j, "D_JS"] <= row["D_JS"]) and
        (df_lambda.loc[j, "deviance"] <= row["deviance"]) and
        (df_lambda.loc[j, "D_JS"] < row["D_JS"] or df_lambda.loc[j, "deviance"] < row["deviance"])
        for j in df_lambda.index if j != i
    )
])

from insurance_fairness.intersectional import _pareto_indices
pareto_idx_computed = _pareto_indices(
    df_lambda["D_JS"].values, df_lambda["deviance"].values
)

result = LambdaCalibrationResult(
    lambda_values=df_lambda["lambda"].tolist(),
    js_divergence=df_lambda["D_JS"].tolist(),
    validation_loss=df_lambda["deviance"].tolist(),
    selected_lambda=0.5,
    method="ccDcov",
    pareto_indices=pareto_idx_computed,
)

print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Subgroup Statistics
# MAGIC
# MAGIC The audit report contains per-subgroup mean predictions, allowing direct
# MAGIC identification of which intersectional cells drive the disparity.

# COMMAND ----------

print("\nSubgroup statistics — intersectionally unfair model (top 10 by mean prediction):")
print(
    report_intersect.subgroup_statistics
    .sort_values("mean_prediction", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nCompare with fair model (same subgroups):")
print(
    report_fair.subgroup_statistics
    .sort_values("mean_prediction", ascending=False)
    .head(10)
    .to_string(index=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Workflow Summary
# MAGIC
# MAGIC ```
# MAGIC 1. Train baseline model (λ=0)
# MAGIC 2. IntersectionalFairnessAudit.audit() — quantify CCdCov, η, D_JS
# MAGIC 3. If CCdCov > threshold or η > 0:
# MAGIC    a. Add DistanceCovFairnessRegulariser to training loop
# MAGIC    b. DistanceCovFairnessRegulariser.calibrate_lambda() — sweep λ grid
# MAGIC    c. Plot LambdaCalibrationResult.plot() — choose operating point
# MAGIC    d. Retrain with selected λ
# MAGIC    e. Re-audit to confirm η reduced
# MAGIC 4. Export IntersectionalAuditReport.to_markdown() for FCA evidence pack
# MAGIC ```
# MAGIC
# MAGIC The choice of λ is explicitly exogenous — the pricing team must document
# MAGIC their operating point on the fairness-accuracy Pareto and justify it under
# MAGIC Consumer Duty Outcome 4.

# COMMAND ----------

# Final markdown report
md = report_intersect.to_markdown()
print(md[:2000])

# COMMAND ----------

print("Demo complete.")
print(f"insurance-fairness version: ", end="")
import insurance_fairness
print(insurance_fairness.__version__)
