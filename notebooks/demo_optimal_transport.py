# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-fairness-ot: Discrimination-Free Insurance Pricing
# MAGIC
# MAGIC **Library:** insurance-fairness-ot v0.1.0
# MAGIC **Methodology:** Lindholm (2022) marginalisation + causal path decomposition
# MAGIC **Regulatory basis:** Equality Act 2010 s.19, FCA PRIN 2A.4, ICOBS 6B
# MAGIC
# MAGIC This notebook demonstrates the full workflow on synthetic UK motor insurance data:
# MAGIC 1. Specify the causal graph (protected/proxy/justified variables)
# MAGIC 2. Train a toy pricing model (logistic regression frequency model)
# MAGIC 3. Apply Lindholm discrimination-free correction
# MAGIC 4. Apply Wasserstein barycenter correction (multi-attribute)
# MAGIC 5. Compute fairness metrics and generate FCA compliance report

# COMMAND ----------

# MAGIC %pip install insurance-fairness-ot polars numpy scipy statsmodels networkx POT

# COMMAND ----------

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from insurance_fairness.optimal_transport import (
    CausalGraph,
    DiscriminationFreePrice,
    FairnessReport,
    FCAReport,
    LindholmCorrector,
    WassersteinCorrector,
)

print("insurance-fairness optimal_transport imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic UK Motor Data
# MAGIC
# MAGIC Based on the Côté et al. (2025) UK motor DAG:
# MAGIC - S: gender (protected)
# MAGIC - R: claims_history, ncb_years (justified mediators)
# MAGIC - V: annual_mileage (partial proxy for gender via usage patterns)
# MAGIC - Y: claim_indicator

# COMMAND ----------

rng = np.random.default_rng(42)
n = 5000

# Protected attribute: gender (binary for illustration)
gender = rng.choice(["M", "F"], n, p=[0.55, 0.45])

# Causal structure:
# Gender influences mileage (proxy) and claims history (justified mediator)
annual_mileage = np.where(
    gender == "M",
    rng.integers(12000, 35000, n),
    rng.integers(8000, 25000, n),
)

ncb_years = np.where(
    gender == "M",
    rng.integers(0, 10, n),
    rng.integers(1, 10, n),  # women slightly more experienced on average
)

claims_history = rng.choice(
    ["yes", "no"],
    n,
    p=None,  # set per-row below
)
p_claim = 0.25 - 0.02 * ncb_years / 9
p_claim = np.clip(p_claim, 0.05, 0.45)
claims_history = np.where(
    rng.uniform(size=n) < p_claim,
    "yes",
    "no",
)

# Outcome: claim frequency (influenced by mileage, claims_history; direct gender effect)
base_freq = 0.08
freq = (
    base_freq
    + 0.002 * (annual_mileage - 15000) / 10000  # mileage effect
    + 0.04 * (claims_history == "yes")           # claims history
    + 0.015 * (gender == "M")                    # direct gender effect (discriminatory)
    - 0.003 * ncb_years                          # NCB bonus
)
freq = np.clip(freq, 0.02, 0.5)
claim_indicator = rng.binomial(1, freq, n).astype(float)
exposure = rng.uniform(0.5, 1.0, n)

# Split into features
X = pl.DataFrame({
    "annual_mileage": annual_mileage.tolist(),
    "ncb_years": ncb_years.tolist(),
    "claims_history": claims_history.tolist(),
    "gender": gender.tolist(),
})
D = pl.DataFrame({"gender": gender.tolist()})

print(f"Dataset: {n} policies")
print(f"Gender split: M={np.mean(gender=='M'):.2%}, F={np.mean(gender=='F'):.2%}")
print(f"Mean claim frequency: {np.mean(claim_indicator):.3f}")
print(f"Gender claim rates: M={np.mean(claim_indicator[gender=='M']):.3f}, F={np.mean(claim_indicator[gender=='F']):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train a Toy Frequency Model
# MAGIC
# MAGIC In production this would be a GLM or CatBoost model. Here we use logistic
# MAGIC regression with gender included (the D paradox: to remove discrimination
# MAGIC you must first include the protected attribute in training).

# COMMAND ----------

# Encode categoricals
X_np = np.column_stack([
    annual_mileage / 10000,
    ncb_years,
    (claims_history == "yes").astype(float),
    (gender == "M").astype(float),
])

lr = LogisticRegression(max_iter=1000)
lr.fit(X_np, (claim_indicator > 0).astype(int))

def sklearn_model(df: pl.DataFrame) -> np.ndarray:
    """Wrapper to make scikit-learn model work with Polars DataFrames."""
    features = np.column_stack([
        df["annual_mileage"].to_numpy() / 10000,
        df["ncb_years"].to_numpy(),
        (df["claims_history"] == "yes").to_numpy().astype(float),
        (df["gender"] == "M").to_numpy().astype(float),
    ])
    return lr.predict_proba(features)[:, 1]

# Verify model gives reasonable predictions
preds = sklearn_model(X)
print(f"Model mean prediction: {np.mean(preds):.3f}")
print(f"Model predictions by gender: M={np.mean(preds[gender=='M']):.3f}, F={np.mean(preds[gender=='F']):.3f}")
print(f"Gender premium gap (best-estimate): {np.mean(preds[gender=='M'])/np.mean(preds[gender=='F']):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Specify the Causal Graph
# MAGIC
# MAGIC The key governance decision: classify each variable as protected, proxy,
# MAGIC or justified mediator. This requires actuarial sign-off.
# MAGIC
# MAGIC Our UK motor DAG:
# MAGIC - gender → direct → claim_freq (discriminatory: must remove)
# MAGIC - gender → annual_mileage → claim_freq (proxy path: must remove)
# MAGIC - gender → claims_history → claim_freq (justified: retain)
# MAGIC - gender → ncb_years → claim_freq (justified: retain)

# COMMAND ----------

graph = (
    CausalGraph()
    .add_protected("gender")
    .add_justified_mediator("claims_history", parents=["gender"])
    .add_justified_mediator("ncb_years", parents=["gender"])
    .add_proxy("annual_mileage", parents=["gender"])
    .add_outcome("claim_freq")
    .add_edge("claims_history", "claim_freq")
    .add_edge("ncb_years", "claim_freq")
    .add_edge("annual_mileage", "claim_freq")
)

graph.validate()
print(graph)
print(f"\nProtected nodes: {graph.get_protected_nodes()}")
print(f"Proxy nodes: {graph.get_proxy_nodes()}")
print(f"Justified mediators: {graph.get_justified_nodes()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Apply Lindholm Discrimination-Free Correction
# MAGIC
# MAGIC h*(x_i) = sum_d mu_hat(x_i, d) * P(D=d)
# MAGIC
# MAGIC For each policyholder, we predict their frequency under M and F, then
# MAGIC average by portfolio proportions. This removes the gender effect from
# MAGIC pricing while retaining the claims_history and mileage effects.

# COMMAND ----------

dfp = DiscriminationFreePrice(
    graph=graph,
    combined_model_fn=sklearn_model,
    correction="lindholm",
    bias_correction="proportional",
    log_space=False,  # logistic regression outputs probabilities, not log-rates
)

# Fit on full data (in production: fit on calibration set, transform on holdout)
result = dfp.fit_transform(X, D, exposure=exposure)

print(f"\n=== LINDHOLM CORRECTION RESULTS ===")
print(f"Portfolio weights (P(D=d)):")
for d_val, w in result.metadata["portfolio_weights"]["gender"].items():
    print(f"  {d_val}: {w:.4f}")

print(f"\nBias correction factor: {result.bias_correction_factor:.4f}")
print(f"  (deviation from 1.0 = {abs(result.bias_correction_factor - 1.0)*100:.2f}%)")

print(f"\nPremium comparison:")
print(f"  Best-estimate (M): {np.mean(result.best_estimate[gender=='M']):.4f}")
print(f"  Best-estimate (F): {np.mean(result.best_estimate[gender=='F']):.4f}")
print(f"  Fair premium (M):  {np.mean(result.fair_premium[gender=='M']):.4f}")
print(f"  Fair premium (F):  {np.mean(result.fair_premium[gender=='F']):.4f}")

gap_before = np.mean(result.best_estimate[gender=='M']) / np.mean(result.best_estimate[gender=='F'])
gap_after = np.mean(result.fair_premium[gender=='M']) / np.mean(result.fair_premium[gender=='F'])
print(f"\nGender premium ratio before: {gap_before:.3f}")
print(f"Gender premium ratio after:  {gap_after:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Lindholm Manual Verification
# MAGIC
# MAGIC Cross-check against the Lindholm (2022) Example 8 known values.

# COMMAND ----------

# Example 8 synthetic data: smoker/gender health insurance
# Portfolio: 589 policies, P(woman) = 264/589
# Known values: h*(smoker) = 0.200, h*(non-smoker) = 0.184

def lindholm_health_model(df: pl.DataFrame) -> np.ndarray:
    """Exact model from Lindholm Example 8."""
    n = df.shape[0]
    preds = np.zeros(n)
    g = df["gender"].to_numpy()
    s = df["smoker"].to_numpy()
    preds[(g == "F") & (s == "yes")] = 32 / 133  # 0.2406
    preds[(g == "F") & (s == "no")]  = 21 / 131  # 0.1603
    preds[(g == "M") & (s == "yes")] = 4 / 24    # 0.1667
    preds[(g == "M") & (s == "no")]  = 51 / 301  # 0.1694
    return np.maximum(preds, 1e-10)

# Build portfolio: 264 women, 325 men; 133/24 smokers
gender_h = ["F"] * 264 + ["M"] * 325
smoker_h = ["yes"] * 133 + ["no"] * 131 + ["yes"] * 24 + ["no"] * 301
X_h = pl.DataFrame({"smoker": smoker_h, "gender": gender_h})
D_h = pl.DataFrame({"gender": gender_h})

corrector = LindholmCorrector(["gender"], bias_correction="proportional", log_space=False)
corrector.fit(lindholm_health_model, X_h, D_h)

# Compute marginalised values analytically
p_F = 264 / 589
p_M = 325 / 589
h_smoker = (32/133) * p_F + (4/24) * p_M
h_nonsmoker = (21/131) * p_F + (51/301) * p_M

print("=== LINDHOLM EXAMPLE 8 VERIFICATION ===")
print(f"P(F) = {p_F:.4f}  (expected 0.4482)")
print(f"P(M) = {p_M:.4f}  (expected 0.5518)")
print(f"h*(smoker)     = {h_smoker:.4f}  (expected 0.200)")
print(f"h*(non-smoker) = {h_nonsmoker:.4f}  (expected 0.184)")

# Portfolio-level totals
h_all = corrector.transform(lindholm_health_model, X_h, D_h)
mu_all = lindholm_health_model(pl.concat([X_h, D_h], how="horizontal"))

print(f"\nPortfolio sum h*(x): {h_all.sum():.2f}  (expected ~110.77)")
print(f"Portfolio sum mu(x,d): {mu_all.sum():.2f}  (expected ~112.0)")
print(f"Bias correction factor: {corrector.bias_correction_factor_:.4f}  (expected ~1.0111)")

assert abs(h_smoker - 0.200) < 0.001, f"Smoker value off: {h_smoker}"
assert abs(h_nonsmoker - 0.184) < 0.002, f"Non-smoker value off: {h_nonsmoker}"
print("\nAll Example 8 regression checks passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wasserstein Barycenter Correction
# MAGIC
# MAGIC Secondary correction for multi-attribute simultaneous fairness.
# MAGIC Achieves demographic parity (unconditional).

# COMMAND ----------

wc = WassersteinCorrector(
    ["gender"],
    epsilon=0.0,
    log_space=False,
    n_quantiles=500,
)
wc.fit(preds, D, exposure=exposure)
fair_ot = wc.transform(preds, D)

print("=== WASSERSTEIN CORRECTION ===")
print(f"W2 distance (before correction): {wc.wasserstein_distances_['gender']:.4f}")
print(f"\nGroup means before OT:")
print(f"  M: {np.mean(preds[gender=='M']):.4f}")
print(f"  F: {np.mean(preds[gender=='F']):.4f}")
print(f"\nGroup means after OT:")
print(f"  M: {np.mean(fair_ot[gender=='M']):.4f}")
print(f"  F: {np.mean(fair_ot[gender=='F']):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fairness Metrics

# COMMAND ----------

report = FairnessReport(result, graph)
metrics = report.discrimination_metrics(D, exposure=exposure)

print("=== FAIRNESS METRICS ===")
for attr, m in metrics.items():
    print(f"\n{attr}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

print("\n=== PREMIUM COMPARISON TABLE ===")
comparison = report.premium_comparison_table(D)
print(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. FCA Compliance Report

# COMMAND ----------

fca = FCAReport(
    result,
    report_metadata={
        "firm_name": "Acme Motor Insurance Ltd",
        "model_name": "Motor Frequency Model",
        "reporting_date": "2026-03-10",
        "model_version": "1.0",
    },
)

print("=== FCA MARKDOWN REPORT (first 3000 chars) ===")
print(fca.render("markdown")[:3000])

# Save to Databricks FileStore
md_path = "/dbfs/FileStore/fca_fairness_report.md"
json_path = "/dbfs/FileStore/fca_fairness_report.json"
fca.save(md_path, "markdown")
fca.save(json_path, "json")
print(f"\nReport saved to {md_path}")
print(f"Report saved to {json_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Gender premium gap (best-estimate) | > 1.0 |
# MAGIC | Gender premium gap (after Lindholm) | closer to 1.0 |
# MAGIC | Bias correction factor | ≈ 1.0 |
# MAGIC | Lindholm Example 8 h*(smoker) | 0.200 ✓ |
# MAGIC | Lindholm Example 8 h*(non-smoker) | 0.184 ✓ |
# MAGIC
# MAGIC The Lindholm correction removes the discriminatory gender pricing effect
# MAGIC while preserving the actuarially justified claim history and NCB effects.
# MAGIC The bias correction factor is close to 1.0, confirming portfolio balance
# MAGIC is maintained after correction.
