# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-fairness: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the `insurance-fairness` library on synthetic
# MAGIC UK motor insurance data. It covers:
# MAGIC
# MAGIC 1. Generating a realistic synthetic dataset
# MAGIC 2. Training a CatBoost frequency model
# MAGIC 3. Running a full fairness audit
# MAGIC 4. Interpreting the outputs
# MAGIC 5. Running proxy detection
# MAGIC 6. Counterfactual fairness testing
# MAGIC
# MAGIC **Regulatory context:** The audit is framed around FCA Consumer Duty
# MAGIC (PRIN 2A.4) and Equality Act 2010 Section 19 requirements. The outputs
# MAGIC are suitable for inclusion in a pricing committee pack or FCA file review.

# COMMAND ----------

# MAGIC %pip install insurance-fairness catboost polars scikit-learn scipy jinja2 pyarrow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Data Generation
# MAGIC
# MAGIC We generate 50,000 synthetic motor policies with realistic rating factors
# MAGIC and a known fairness issue: postcode is correlated with an ethnicity proxy,
# MAGIC and also contributes to the model's price prediction. This replicates the
# MAGIC Citizens Advice (2022) finding of a postcode-based ethnicity penalty.

# COMMAND ----------

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

np.random.seed(2024)
n = 50_000

# Protected characteristic: ethnicity proxy (ONS LSOA % non-white at LSOA level)
# In practice you would join this from ONS Census data via postcode.
# Here we simulate it as a continuous value in [0, 1].
ethnicity_proxy = np.random.beta(2, 8, n)  # Most LSOAs are majority white; right-skewed

# Postcode district: strongly correlated with ethnicity proxy
# (This is the mechanistic driver of the Citizens Advice finding)
postcode_percentile = np.argsort(np.argsort(ethnicity_proxy)) / n
postcode_category = np.digitize(postcode_percentile, np.linspace(0, 1, 21)) - 1
postcode = np.array([f"PC{d:02d}" for d in postcode_category])

# Other rating factors (not correlated with ethnicity proxy)
vehicle_age = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n)
ncd_years = np.random.choice(range(11), n)
driver_age_band = np.random.choice(
    ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    n,
    p=[0.08, 0.18, 0.22, 0.22, 0.18, 0.12],
)
vehicle_group = np.random.choice(list("ABCDEFGH"), n, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.12, 0.08, 0.05])
exposure = np.random.uniform(0.3, 1.0, n)

# Claim frequency model (Poisson)
# True log-frequency includes a postcode effect that contains ethnicity proxy information
log_freq_base = (
    -3.5                                          # intercept
    + 0.3 * (vehicle_age / 10)                   # vehicle age effect
    - 0.04 * ncd_years                           # NCD effect
    + np.where(driver_age_band == "17-24", 0.8, 0)   # young driver
    + np.where(driver_age_band == "25-34", 0.3, 0)
    + np.where(driver_age_band == "65+", 0.2, 0)
    + (postcode_category / 20) * 0.4             # postcode effect (risk gradient)
)

# The postcode effect has two components:
# 1. Legitimate: traffic density, crime rates (correlated with urban areas)
# 2. Proxy: ethnicity_proxy directly increases frequency (the discriminatory part)
log_freq = log_freq_base + ethnicity_proxy * 0.1  # small direct ethnicity effect

# Simulate claims
freq = np.exp(log_freq) * exposure
n_claims = np.random.poisson(freq)
# Severity: log-normal, mean ~£2,500
severity = np.where(
    n_claims > 0,
    np.random.lognormal(7.8, 0.6, n),
    0.0,
)
claim_amount = n_claims * severity

# Predicted premium from a model that doesn't know about ethnicity_proxy
# (simulates a production CatBoost model fit on the same data)
# For the demo, we use a simpler approximation of what the model would predict
predicted_premium = np.exp(log_freq_base) * 2500  # frequency * mean severity

df = pl.DataFrame({
    "ethnicity_proxy": ethnicity_proxy.tolist(),        # Protected characteristic
    "postcode_district": postcode.tolist(),             # Rating factor (proxy for ethnicity)
    "vehicle_age": vehicle_age.tolist(),
    "ncd_years": ncd_years.tolist(),
    "driver_age_band": driver_age_band.tolist(),
    "vehicle_group": vehicle_group.tolist(),
    "exposure": exposure.tolist(),
    "n_claims": n_claims.tolist(),
    "claim_amount": claim_amount.tolist(),
    "predicted_premium": predicted_premium.tolist(),
})

print(f"Dataset: {len(df):,} policies")
print(f"Total exposure: {df['exposure'].sum():,.1f} policy-years")
print(f"Claim frequency: {df['n_claims'].sum() / df['exposure'].sum():.4f}")
print(f"Average premium: £{df['predicted_premium'].mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train a CatBoost Frequency Model
# MAGIC
# MAGIC We train a Poisson CatBoost model on the rating factors (excluding the
# MAGIC ethnicity proxy, which would not be available in practice). This model
# MAGIC will implicitly proxy for ethnicity through postcode.

# COMMAND ----------

from sklearn.model_selection import train_test_split

feature_cols = ["postcode_district", "vehicle_age", "ncd_years", "driver_age_band", "vehicle_group"]
cat_cols = ["postcode_district", "driver_age_band", "vehicle_group"]

df_pd = df.select(feature_cols + ["n_claims", "exposure"]).to_pandas()

X_train, X_test, y_train, y_test = train_test_split(
    df_pd[feature_cols],
    df_pd["n_claims"],
    test_size=0.2,
    random_state=42,
)
w_train = df_pd.loc[X_train.index, "exposure"]
w_test = df_pd.loc[X_test.index, "exposure"]

train_pool = Pool(X_train, y_train, weight=w_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, weight=w_test, cat_features=cat_cols)

model = CatBoostRegressor(
    iterations=300,
    depth=5,
    loss_function="Poisson",
    learning_rate=0.05,
    random_seed=42,
    verbose=50,
    allow_writing_files=False,
)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)

# Add model predictions to the full dataset
full_pool = Pool(df.select(feature_cols).to_pandas(), cat_features=cat_cols)
df = df.with_columns(
    pl.Series("model_freq", model.predict(full_pool))
)
# Convert frequency to approximate pure premium
df = df.with_columns(
    (pl.col("model_freq") * 2500).alias("model_premium")
)

print(f"\nModel trained. Test Poisson deviance: {model.best_score_['validation']['Poisson']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Full Fairness Audit
# MAGIC
# MAGIC Now run the full `FairnessAudit`. We pass the ethnicity proxy as the
# MAGIC protected characteristic (in practice, this would come from ONS LSOA
# MAGIC data joined to postcodes).

# COMMAND ----------

from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["ethnicity_proxy"],
    prediction_col="model_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=feature_cols,
    model_name="Motor Frequency Model v1.0",
    run_proxy_detection=True,
    run_counterfactual=False,  # ethnicity_proxy not a direct model input
    n_calibration_deciles=10,
    proxy_catboost_iterations=150,
)

report = audit.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Audit Results

# COMMAND ----------

report.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demographic Parity

# COMMAND ----------

# For continuous protected characteristics, we group into quartiles for display
dp = report.results["ethnicity_proxy"].demographic_parity
print(f"Demographic parity log-ratio: {dp.log_ratio:+.4f}")
print(f"Ratio: {dp.ratio:.4f}")
print(f"RAG status: {dp.rag.upper()}")
print("\nGroup means (log-space):")
for g, m in sorted(dp.group_means.items(), key=lambda x: float(x[0])):
    print(f"  Ethnicity proxy = {float(g):.2f}: {m:.4f} (exposure: {dp.group_exposures[g]:,.0f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calibration by Group

# COMMAND ----------

cal = report.results["ethnicity_proxy"].calibration
print(f"Max calibration disparity (A/E deviation from 1.0): {cal.max_disparity:.4f}")
print(f"RAG status: {cal.rag.upper()}")
print("\nA/E ratios by decile and ethnicity quartile (first 5 deciles shown):")
for d in range(1, 6):
    d_vals = cal.actual_to_expected.get(d, {})
    vals_str = " | ".join(
        f"{g}: {v:.3f}" for g, v in sorted(d_vals.items(), key=lambda x: float(x[0]))
        if v == v  # skip nan
    )
    print(f"  Decile {d}: {vals_str}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proxy Detection

# COMMAND ----------

prox = report.results["ethnicity_proxy"].proxy_detection
proxy_df = prox.to_polars()
print(f"Factors flagged as proxies: {prox.flagged_factors}")
print("\nTop factors by proxy R-squared:")
display(proxy_df.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Disparate Impact Ratio

# COMMAND ----------

di = report.results["ethnicity_proxy"].disparate_impact
print(f"Disparate impact ratio: {di.ratio:.4f}")
print(f"RAG status: {di.rag.upper()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export Audit Report

# COMMAND ----------

# Write to Markdown for FCA file / pricing committee
report.to_markdown("/tmp/motor_fairness_audit.md")

# Also export as dict for downstream processing
import json
audit_dict = report.to_dict()
print(json.dumps(audit_dict, indent=2)[:2000])  # First 2000 chars

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Individual Metrics - Standalone Usage
# MAGIC
# MAGIC All metrics are importable independently for ad-hoc analysis.

# COMMAND ----------

from insurance_fairness import (
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    gini_by_group,
    theil_index,
)

# Bin ethnicity proxy into quartiles for binary/categorical analysis
df_binned = df.with_columns(
    pl.when(pl.col("ethnicity_proxy") > pl.col("ethnicity_proxy").median())
    .then(pl.lit("High ethnicity diversity"))
    .otherwise(pl.lit("Low ethnicity diversity"))
    .alias("ethnicity_group")
)

# Demographic parity (binary version)
dp_binary = demographic_parity_ratio(
    df_binned,
    protected_col="ethnicity_group",
    prediction_col="model_premium",
    exposure_col="exposure",
)
print("Binary ethnicity group demographic parity:")
print(f"  Log-ratio: {dp_binary.log_ratio:+.4f}")
print(f"  Ratio: {dp_binary.ratio:.4f} ({(dp_binary.ratio - 1)*100:+.1f}%)")
print(f"  Status: {dp_binary.rag.upper()}")

# COMMAND ----------

# Calibration
cal_binary = calibration_by_group(
    df_binned,
    protected_col="ethnicity_group",
    prediction_col="model_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    n_deciles=5,
)
print("\nCalibration (A/E) by ethnicity group and pricing decile:")
for d, d_vals in sorted(cal_binary.actual_to_expected.items()):
    vals = {g: f"{v:.3f}" for g, v in d_vals.items() if v == v}
    print(f"  Decile {d}: {vals}")

# COMMAND ----------

# Theil index decomposition
theil = theil_index(
    df_binned,
    protected_col="ethnicity_group",
    prediction_col="model_premium",
    exposure_col="exposure",
)
print("\nTheil Index decomposition:")
print(f"  Total Theil: {theil.theil_total:.6f}")
print(f"  Between-group: {theil.theil_between:.6f} ({theil.theil_between/theil.theil_total*100:.1f}% of total)")
print(f"  Within-group: {theil.theil_within:.6f} ({theil.theil_within/theil.theil_total*100:.1f}% of total)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. SHAP Proxy Scores
# MAGIC
# MAGIC SHAP proxy scores link each feature's contribution to the price prediction
# MAGIC to the protected characteristic. A high score means the feature is using
# MAGIC information correlated with the protected characteristic to set prices.

# COMMAND ----------

from insurance_fairness import shap_proxy_scores

shap_scores = shap_proxy_scores(
    df_binned,
    protected_col="ethnicity_group",
    factor_cols=feature_cols,
    model=model,
)

print("SHAP proxy scores (|Spearman(SHAP_j, S)|):")
for factor, score in sorted(shap_scores.items(), key=lambda x: -x[1]):
    bar = "#" * int(score * 30)
    print(f"  {factor:<25} {score:.4f}  {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Counterfactual Fairness (Where Protected Char is Direct Input)
# MAGIC
# MAGIC Here we demonstrate counterfactual testing for `driver_age_band`, which
# MAGIC is a direct model input. We flip "17-24" to "25-34" to measure the
# MAGIC premium impact of being a young driver.

# COMMAND ----------

from insurance_fairness import counterfactual_fairness

# Filter to just young drivers for the flip test
young_driver_df = df.filter(pl.col("driver_age_band") == "17-24")

cf_result = counterfactual_fairness(
    model=model,
    df=young_driver_df,
    protected_col="driver_age_band",
    feature_cols=feature_cols,
    exposure_col="exposure",
    flip_values={"17-24": "25-34"},
    method="direct_flip",
)

print(cf_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook has demonstrated:
# MAGIC
# MAGIC - Generating synthetic motor insurance data with a known proxy discrimination
# MAGIC   issue (postcode as proxy for ethnicity)
# MAGIC - Training a CatBoost Poisson frequency model
# MAGIC - Running a full `FairnessAudit` with exposure-weighted metrics
# MAGIC - Interpreting calibration, demographic parity, and disparate impact results
# MAGIC - Using proxy detection to identify which factors carry ethnicity information
# MAGIC - Exporting an FCA-ready Markdown audit report
# MAGIC - Running SHAP proxy scores to link feature contributions to protected characteristics
# MAGIC - Counterfactual testing for a direct rating factor
# MAGIC
# MAGIC The postcode factor should show the highest proxy R-squared and SHAP proxy
# MAGIC score. The calibration check will show whether this translates to systematic
# MAGIC over-pricing of high-ethnicity-diversity areas.
# MAGIC
# MAGIC **Next steps for a real audit:**
# MAGIC 1. Join ONS Census 2021 LSOA ethnicity data to your policy postcodes
# MAGIC 2. Run on your actual model and portfolio
# MAGIC 3. Document the legitimate actuarial justification for each flagged factor
# MAGIC 4. Consider whether less discriminatory alternatives exist
# MAGIC 5. Sign off the audit report and retain for FCA file
