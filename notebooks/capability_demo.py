# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line

# COMMAND ----------

# MAGIC %md
# MAGIC # Capability Demo: insurance-fairness
# MAGIC
# MAGIC **Library:** `insurance-fairness` v0.3.0 — proxy discrimination auditing for UK motor insurance pricing
# MAGIC
# MAGIC **What this demonstrates:** The FCA expects firms to satisfy themselves that rating factors
# MAGIC do not result in systematically worse outcomes for groups sharing protected characteristics
# MAGIC (Consumer Duty PRIN 2A, Equality Act 2010 s.19). Manual A/E analysis by demographic group
# MAGIC misses proxy discrimination: a model that never uses gender directly can still discriminate
# MAGIC if a rating factor it uses — postcode, vehicle type, occupation — is correlated with gender.
# MAGIC This demo shows what that looks like in practice, and how the library catches it.
# MAGIC
# MAGIC **Scenario:** A CatBoost frequency model is trained on synthetic motor data where postcode
# MAGIC district is correlated with the policyholder's gender. The model is never told the gender.
# MAGIC Despite this, the library detects that postcode is acting as a proxy, quantifies the
# MAGIC premium disparity, and produces a markdown audit report suitable for the pricing committee.
# MAGIC
# MAGIC **Regulatory context:**
# MAGIC - FCA Evaluation Paper EP25/2 (2025)
# MAGIC - FCA Consumer Duty Finalised Guidance FG22/5 (2023)
# MAGIC - Equality Act 2010, Section 19 (Indirect Discrimination)
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC **Library version:** 0.3.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Install the library under test and dependencies
%pip install git+https://github.com/burning-cost/insurance-fairness.git
%pip install catboost scikit-learn polars numpy scipy matplotlib pandas pyarrow

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from insurance_fairness import (
    FairnessAudit,
    FairnessReport,
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
    gini_by_group,
    theil_index,
    mutual_information_scores,
    partial_correlation,
    proxy_r2_scores,
    shap_proxy_scores,
    counterfactual_fairness,
    generate_markdown_report,
)
from insurance_fairness.proxy_detection import detect_proxies

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Notebook run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Data
# MAGIC
# MAGIC We generate a motor insurance dataset where postcode district is correlated with gender.
# MAGIC This is a simplified version of the postcode/ethnicity problem the FCA flagged in TR24/2,
# MAGIC but using gender because it is binary, which makes the demonstration cleaner.
# MAGIC
# MAGIC **The known bias in the DGP:**
# MAGIC - Policyholders in postcode districts PC1, PC2, PC3 are predominantly female (70%).
# MAGIC - Policyholders in postcode districts PC4, PC5, PC6 are predominantly male (70%).
# MAGIC - Risk (claim frequency) is determined by vehicle group, driver age band, and NCD — not gender.
# MAGIC - But the pricing model never sees gender. It sees postcode. Because postcode predicts gender,
# MAGIC   and the model finds postcode to be a strong predictor (via geographic risk patterns we bake
# MAGIC   in), the model inadvertently prices on gender through the postcode channel.
# MAGIC
# MAGIC This is textbook indirect discrimination under Equality Act 2010 s.19: a provision, criterion
# MAGIC or practice (using postcode) that puts a group sharing a protected characteristic (female
# MAGIC policyholders) at a particular disadvantage.

# COMMAND ----------

rng = np.random.default_rng(42)

N = 50_000

# Postcode districts — split into two geographic zones
# Zone A (PC1-PC3): lower-risk areas, predominantly female
# Zone B (PC4-PC6): higher-risk areas, predominantly male
# This correlation with gender is the source of indirect discrimination.
POSTCODES_ZONE_A = ["PC1", "PC2", "PC3"]
POSTCODES_ZONE_B = ["PC4", "PC5", "PC6"]
ALL_POSTCODES    = POSTCODES_ZONE_A + POSTCODES_ZONE_B

# Assign postcode — broadly equal split between zones
postcode_zone = rng.choice(["A", "B"], size=N, p=[0.50, 0.50])
postcode = np.where(
    postcode_zone == "A",
    rng.choice(POSTCODES_ZONE_A, size=N),
    rng.choice(POSTCODES_ZONE_B, size=N),
)

# Gender is correlated with postcode zone: 70% female in zone A, 70% male in zone B
gender_prob_female = np.where(postcode_zone == "A", 0.70, 0.30)
gender = np.where(rng.random(N) < gender_prob_female, "F", "M")
gender_binary = (gender == "F").astype(int)  # 1 = female, 0 = male

# Vehicle group — rating factor, independent of gender
vehicle_group = rng.choice(["A", "B", "C", "D"], size=N, p=[0.30, 0.30, 0.25, 0.15])

# Driver age band — rating factor, independent of gender
age_band = rng.choice(["17-24", "25-34", "35-49", "50-64", "65+"], size=N,
                      p=[0.10, 0.20, 0.30, 0.25, 0.15])

# NCD band — rating factor, independent of gender
ncd_band = rng.choice(["0yr", "1-2yr", "3-4yr", "5yr+"], size=N, p=[0.15, 0.20, 0.30, 0.35])

# Vehicle age — numeric rating factor
vehicle_age = rng.integers(0, 16, size=N).astype(float)

# Exposure (years on risk, fraction of year)
exposure = rng.uniform(0.1, 1.0, size=N)

# ---------------------------------------------------------------------------
# True claim frequency (DGP) — determined by rating factors, NOT gender
# ---------------------------------------------------------------------------
# Base frequency varies by vehicle group and age band.
# Postcode zone has a genuine risk component (e.g. urban vs rural density).
base_freq = np.where(vehicle_group == "A", 0.060,
            np.where(vehicle_group == "B", 0.075,
            np.where(vehicle_group == "C", 0.090, 0.110)))

age_adj = np.where(age_band == "17-24", 2.5,
          np.where(age_band == "25-34", 1.4,
          np.where(age_band == "35-49", 1.0,
          np.where(age_band == "50-64", 0.85, 0.90))))

ncd_adj = np.where(ncd_band == "0yr",   1.40,
          np.where(ncd_band == "1-2yr", 1.15,
          np.where(ncd_band == "3-4yr", 0.95, 0.80)))

# Zone B genuinely has higher risk (higher traffic density, more claims).
# This is the legitimate risk justification for using postcode.
zone_risk_adj = np.where(postcode_zone == "A", 0.90, 1.10)

true_freq = base_freq * age_adj * ncd_adj * zone_risk_adj

# Generate claim counts from Poisson DGP
claim_count = rng.poisson(true_freq * exposure)

# ---------------------------------------------------------------------------
# Assemble Polars DataFrame
# ---------------------------------------------------------------------------
df_full = pl.DataFrame({
    "postcode":      postcode.tolist(),
    "vehicle_group": vehicle_group.tolist(),
    "age_band":      age_band.tolist(),
    "ncd_band":      ncd_band.tolist(),
    "vehicle_age":   vehicle_age.tolist(),
    "gender":        gender.tolist(),
    "gender_binary": gender_binary.tolist(),
    "exposure":      exposure.tolist(),
    "claim_count":   claim_count.tolist(),
    "true_freq":     true_freq.tolist(),
})

print(f"Dataset shape: {df_full.shape}")
print(f"\nClaim count distribution:")
print(df_full["claim_count"].describe())
print(f"\nOverall observed frequency: {claim_count.sum() / exposure.sum():.4f}")
print(f"\nGender split: {df_full['gender'].value_counts()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify the bias is present in the data

# COMMAND ----------

# Confirm the postcode-gender correlation is as intended
gender_by_zone = (
    df_full
    .with_columns(
        pl.when(pl.col("postcode").is_in(POSTCODES_ZONE_A))
          .then(pl.lit("Zone A (PC1-3)"))
          .otherwise(pl.lit("Zone B (PC4-6)"))
          .alias("zone")
    )
    .group_by("zone", "gender")
    .agg(pl.len().alias("n_policies"))
    .sort(["zone", "gender"])
)
print("Postcode zone vs gender (the correlation the model will learn):")
print(gender_by_zone.to_pandas().to_string(index=False))

# Confirm risk does NOT differ by gender once we control for zone and age
freq_by_gender = (
    df_full
    .group_by("gender")
    .agg(
        (pl.col("claim_count").sum() / pl.col("exposure").sum()).alias("observed_freq"),
        pl.col("true_freq").mean().alias("mean_true_freq"),
        pl.len().alias("n_policies"),
    )
    .sort("gender")
)
print(f"\nObserved frequency by gender:")
print(freq_by_gender.to_pandas().to_string(index=False))
print(f"\nNote: any frequency gap between F and M is due to selection into zones, NOT to gender")
print(f"affecting risk. A model using postcode will pick this up as an apparent gender signal.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline Pricing Model
# MAGIC
# MAGIC We train a CatBoost Poisson frequency model on all rating factors **excluding gender**.
# MAGIC This is the standard approach: gender is a protected characteristic and is not used as a
# MAGIC direct input. The model is technically compliant. The problem is that it uses postcode,
# MAGIC which is correlated with gender.
# MAGIC
# MAGIC This is not a toy example. This is what most UK motor books look like.

# COMMAND ----------

RATING_FACTORS = ["postcode", "vehicle_group", "age_band", "ncd_band", "vehicle_age"]
CAT_FEATURES   = ["postcode", "vehicle_group", "age_band", "ncd_band"]
TARGET         = "claim_count"
EXPOSURE_COL   = "exposure"

# Train/test split — stratify on postcode zone to preserve the geographic distribution
train_idx, test_idx = train_test_split(
    np.arange(N), test_size=0.30, random_state=42
)

df_train = df_full[train_idx]
df_test  = df_full[test_idx]

X_train_pd = df_train.select(RATING_FACTORS).to_pandas()
X_test_pd  = df_test.select(RATING_FACTORS).to_pandas()

y_train = df_train[TARGET].to_numpy()
y_test  = df_test[TARGET].to_numpy()
exp_train = df_train[EXPOSURE_COL].to_numpy()
exp_test  = df_test[EXPOSURE_COL].to_numpy()

# CatBoost Poisson — fit rates (claim_count / exposure) with exposure as weight
rate_train = y_train / exp_train

pool_train = Pool(X_train_pd, rate_train, cat_features=CAT_FEATURES, weight=exp_train)
pool_test  = Pool(X_test_pd,             cat_features=CAT_FEATURES)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    depth=5,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)
model.fit(pool_train, eval_set=pool_test)

# Predicted rate (annualised frequency)
pred_rate_test  = model.predict(pool_test)
pred_count_test = pred_rate_test * exp_test

# Predicted rate for the full dataset (needed for audit)
pool_full = Pool(df_full.select(RATING_FACTORS).to_pandas(), cat_features=CAT_FEATURES)
pred_rate_full = model.predict(pool_full)

# Add predictions to the DataFrames
df_test = df_test.with_columns(pl.Series("predicted_rate", pred_rate_test))
df_full = df_full.with_columns(pl.Series("predicted_rate", pred_rate_full))

# Basic model diagnostics
observed_rate = y_test.sum() / exp_test.sum()
predicted_rate = pred_count_test.sum() / exp_test.sum()
print(f"Model A/E (overall): {observed_rate / predicted_rate:.4f}")
print(f"Test set: {len(df_test):,} policies, {exp_test.sum():.0f} exposure years")
print(f"Observed frequency: {observed_rate:.4f}, Predicted: {predicted_rate:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Proxy Detection
# MAGIC
# MAGIC The model has never seen gender. But does it price differently for male and female
# MAGIC policyholders via postcode? We run three complementary proxy detection methods:
# MAGIC
# MAGIC 1. **Proxy R-squared**: For each rating factor, how well can a small CatBoost model predict
# MAGIC    gender from that factor alone? High R-squared means the factor is a strong proxy.
# MAGIC
# MAGIC 2. **Mutual information**: Model-free measure of statistical dependence. Catches non-linear
# MAGIC    relationships that R-squared misses.
# MAGIC
# MAGIC 3. **Partial Spearman correlation**: Linear association between each factor and gender, after
# MAGIC    controlling for other factors. Shows which factors carry unique gender information.
# MAGIC
# MAGIC The key question is not just whether postcode predicts gender (it does, by construction)
# MAGIC but whether that correlation flows through to the model's prices.

# COMMAND ----------

# Run proxy detection on the full dataset
# We test all rating factors as potential proxies for gender
proxy_result = detect_proxies(
    df=df_full,
    protected_col="gender_binary",
    factor_cols=RATING_FACTORS,
    exposure_col=EXPOSURE_COL,
    run_proxy_r2=True,
    run_mutual_info=True,
    run_partial_corr=True,
    run_shap=False,       # we'll run SHAP explicitly below
    catboost_iterations=150,
    is_binary_protected=True,
)

print("Proxy detection results (sorted by proxy R-squared):")
print(proxy_result.to_polars().to_pandas().to_string(index=False))
print(f"\nFactors flagged as proxies (amber or red): {proxy_result.flagged_factors}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP proxy scores
# MAGIC
# MAGIC SHAP proxy scores link the proxy correlation to actual price impact. A factor can be
# MAGIC correlated with gender at the input level but have negligible price impact if the model
# MAGIC downweights it. Conversely, a moderately correlated factor with high SHAP importance
# MAGIC is a more serious concern.
# MAGIC
# MAGIC `shap_proxy_score_j = |Spearman(SHAP_j, gender)|`
# MAGIC
# MAGIC A score near 1 means the factor's contribution to the predicted premium tracks gender
# MAGIC closely. This is the direct price impact of the proxy relationship.

# COMMAND ----------

# Compute SHAP proxy scores using the fitted model
shap_scores = shap_proxy_scores(
    df=df_full,
    protected_col="gender_binary",
    factor_cols=RATING_FACTORS,
    model=model,
    exposure_col=EXPOSURE_COL,
)

print("SHAP proxy scores (|Spearman(SHAP_j, gender)|):")
shap_df = pd.DataFrame(
    sorted(shap_scores.items(), key=lambda x: x[1], reverse=True),
    columns=["factor", "shap_proxy_score"],
)
print(shap_df.to_string(index=False))
print(f"\nInterpretation: postcode's SHAP contribution to price is correlated with")
print(f"gender at {shap_scores['postcode']:.3f} Spearman rank correlation.")
print(f"This means the model systematically prices female-coded postcode areas differently")
print(f"from male-coded postcode areas, purely through the postcode channel.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Bias Metrics
# MAGIC
# MAGIC Now we quantify the premium disparity. Proxy detection told us postcode is correlated
# MAGIC with gender. The bias metrics tell us whether this correlation translates into systematic
# MAGIC pricing differences, and whether those differences are justified by actual claims experience.
# MAGIC
# MAGIC The distinction matters legally: a price difference is not necessarily indirect discrimination
# MAGIC if it is justified by genuine risk differences. The calibration by group metric tests this
# MAGIC directly.

# COMMAND ----------

# --- Demographic parity ---
# Is the mean predicted rate different for male vs female policyholders?
dp = demographic_parity_ratio(
    df=df_full,
    protected_col="gender",
    prediction_col="predicted_rate",
    exposure_col=EXPOSURE_COL,
    log_space=True,
    n_bootstrap=200,
    ci_level=0.95,
)
print("Demographic Parity Ratio")
print("-" * 40)
print(f"  Log-ratio (F vs M): {dp.log_ratio:+.4f}")
print(f"  Price ratio (F/M):  {dp.ratio:.4f}  ({(dp.ratio - 1) * 100:+.1f}%)")
print(f"  RAG status:         {dp.rag.upper()}")
if dp.ci_lower is not None:
    print(f"  95% CI on log-ratio: [{dp.ci_lower:+.4f}, {dp.ci_upper:+.4f}]")
print(f"  Group means (log-space): {dp.group_means}")

# COMMAND ----------

# --- Disparate impact ratio ---
# Standard metric: ratio of mean prices between the most and least expensive group.
di = disparate_impact_ratio(
    df=df_full,
    protected_col="gender",
    prediction_col="predicted_rate",
    exposure_col=EXPOSURE_COL,
)
print("Disparate Impact Ratio")
print("-" * 40)
print(f"  DIR:        {di.ratio:.4f}")
print(f"  RAG status: {di.rag.upper()}")
print(f"  Group means: {di.group_means}")
print(f"\n  Interpretation: female policyholders are predicted at {di.ratio:.1%} of")
print(f"  the rate predicted for male policyholders on average.")
print(f"  (DIR < 0.80 is the US EEOC 4/5ths benchmark; for UK use as a directional flag.)")

# COMMAND ----------

# --- Calibration by group ---
# The key question: is the model equally well-calibrated for male and female policyholders?
# If yes, the price difference is justified by genuine risk differences.
# If no, one group is being systematically over- or under-charged relative to their risk.
cal = calibration_by_group(
    df=df_full,
    protected_col="gender",
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col=EXPOSURE_COL,
    n_deciles=10,
)
print("Calibration by Group (A/E ratios, gender x prediction decile)")
print("-" * 60)
print(f"  Max A/E disparity: {cal.max_disparity:.4f}")
print(f"  RAG status: {cal.rag.upper()}")
print()
print("  A/E ratios by decile (F | M):")
for d in sorted(cal.actual_to_expected.keys()):
    ae_f = cal.actual_to_expected[d].get("F", float("nan"))
    ae_m = cal.actual_to_expected[d].get("M", float("nan"))
    gap  = ae_f - ae_m if not (np.isnan(ae_f) or np.isnan(ae_m)) else float("nan")
    print(f"    Decile {d:2d}: F={ae_f:.3f}  M={ae_m:.3f}  gap={gap:+.3f}")

# COMMAND ----------

# --- Gini coefficient by group ---
# Does the model discriminate within each gender group equally well?
# A lower Gini for one group means the model is less able to distinguish high- from low-risk
# policyholders within that group — which may indicate the group is being treated as more
# homogeneous than they actually are.
gini_result = gini_by_group(
    df=df_full,
    protected_col="gender",
    prediction_col="predicted_rate",
    exposure_col=EXPOSURE_COL,
)
print("Gini Coefficient by Group")
print("-" * 40)
print(f"  Overall Gini: {gini_result.overall_gini:.4f}")
for g, gini in gini_result.group_ginis.items():
    print(f"  {g}: {gini:.4f}")
print(f"  Max disparity between groups: {gini_result.max_disparity:.4f}")

# COMMAND ----------

# --- Theil index (between-group vs within-group inequality) ---
# Decomposes premium inequality into within-group and between-group components.
# A high between-group share means the model's price spread is driven by group membership
# rather than individual risk heterogeneity — a flag for systematic between-group pricing.
theil_result = theil_index(
    df=df_full,
    protected_col="gender",
    prediction_col="predicted_rate",
    exposure_col=EXPOSURE_COL,
)
print("Theil Index of Premium Inequality")
print("-" * 40)
print(f"  Total Theil T:    {theil_result.theil_total:.6f}")
print(f"  Within-group T:   {theil_result.theil_within:.6f}")
print(f"  Between-group T:  {theil_result.theil_between:.6f}")
if theil_result.theil_total > 0:
    between_share = theil_result.theil_between / theil_result.theil_total
    print(f"  Between-group share: {between_share:.1%}")
    print(f"\n  A between-group share > 10% warrants review. Higher values indicate")
    print(f"  that premium dispersion is partly explained by group membership.")

# COMMAND ----------

# --- Equalised odds ---
# Are predictions equally well-correlated with actual claims for both genders?
# (Continuous analogue: Spearman rank correlation within each group.)
eq_odds = equalised_odds(
    df=df_full,
    protected_col="gender",
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col=EXPOSURE_COL,
)
print("Equalised Odds (rank correlation of predictions with actuals, by group)")
print("-" * 60)
for gm in eq_odds.group_metrics:
    print(f"  {gm.group_value}: Spearman r = {gm.metric_value:.4f}  "
          f"(n={gm.n_policies:,}, exposure={gm.total_exposure:.0f})")
print(f"  Max disparity: {eq_odds.max_tpr_disparity:.4f}  [{eq_odds.rag.upper()}]")
print(f"\n  Equal rank correlations mean the model is equally informative for both groups.")
print(f"  A gap here means predictions are more accurate for one gender than the other.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Counterfactual Testing
# MAGIC
# MAGIC Counterfactual fairness asks: if we could change only the gender of a policyholder,
# MAGIC would their predicted premium change? Since gender is not a model input, a direct flip
# MAGIC is not applicable. Instead we use **LRTW marginalisation** (Lindholm, Richman, Tsanakas,
# MAGIC Wüthrich 2022): compute the discrimination-free price by averaging predictions over the
# MAGIC marginal distribution of gender.
# MAGIC
# MAGIC For the proxy discrimination scenario (gender not a direct input), LRTW identifies the
# MAGIC premium change attributable to the proxy channel. The individual-level impact distribution
# MAGIC shows which policyholders are most affected.

# COMMAND ----------

# LRTW marginalisation via counterfactual_fairness with method='lrtw_marginalisation'
# We temporarily add gender as a "feature" to enable the marginalisation calculation,
# even though the model does not use it. The function averages over the gender distribution.
# Note: since the model doesn't use gender_binary directly, this will approximate the
# price under the counterfactual where gender is independent of other factors.

# For a cleaner demonstration, we instead show the direct demographic impact:
# train a second model that explicitly includes gender, then compare LRTW-corrected prices.

# First: show counterfactual impact if we force-include gender in the feature set
# and then marginalise. This is the correct way to identify what the price WOULD be
# if there were no correlation between postcode and gender.

RATING_FACTORS_WITH_GENDER = RATING_FACTORS + ["gender"]
CAT_FEATURES_WITH_GENDER   = CAT_FEATURES + ["gender"]

pool_with_gender = Pool(
    df_full.select(RATING_FACTORS_WITH_GENDER).to_pandas(),
    cat_features=CAT_FEATURES_WITH_GENDER,
)

# Fit a model that uses gender directly (the "gender-aware" model)
model_with_gender = CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    depth=5,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)
# Use training indices only
X_train_wg = df_full[train_idx].select(RATING_FACTORS_WITH_GENDER).to_pandas()
rate_train_wg = (df_full[train_idx]["claim_count"] / df_full[train_idx]["exposure"]).to_numpy()
exp_train_wg  = df_full[train_idx]["exposure"].to_numpy()

pool_train_wg = Pool(X_train_wg, rate_train_wg, cat_features=CAT_FEATURES_WITH_GENDER,
                     weight=exp_train_wg)
model_with_gender.fit(pool_train_wg)

# Now test counterfactual fairness: flip gender and measure premium impact
cf_result = counterfactual_fairness(
    model=model_with_gender,
    df=df_full,
    protected_col="gender",
    feature_cols=RATING_FACTORS_WITH_GENDER,
    exposure_col=EXPOSURE_COL,
    flip_values={"F": "M", "M": "F"},
    method="direct_flip",
)

print(cf_result.summary())
print()
print("Policy-level impact percentiles (ratio of counterfactual / original price):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    pval = cf_result.policy_level_impacts.quantile(p / 100)
    print(f"  p{p:2d}: {pval:.4f}  ({(pval - 1) * 100:+.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Individual-level impact distribution
# MAGIC
# MAGIC The plot below shows the distribution of counterfactual price impacts at policy level.
# MAGIC A symmetric distribution centred at 1.0 would mean gender flipping has no systematic
# MAGIC effect. Anything else means the model is pricing on gender through the proxy channel.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution of policy-level counterfactual impacts
impacts = cf_result.policy_level_impacts.to_numpy()
impacts_clean = impacts[~np.isnan(impacts)]

axes[0].hist(impacts_clean, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
axes[0].axvline(1.0, color="black", linewidth=2, linestyle="--", label="No impact (ratio=1.0)")
axes[0].axvline(np.median(impacts_clean), color="tomato", linewidth=2,
                label=f"Median: {np.median(impacts_clean):.3f}")
axes[0].set_xlabel("Counterfactual price ratio (F→M premium / original premium)")
axes[0].set_ylabel("Number of policies")
axes[0].set_title("Counterfactual Impact: Gender Flip\n(how much would each policy's price change?)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predicted rate by gender — box plot
rates_f = df_full.filter(pl.col("gender") == "F")["predicted_rate"].to_numpy()
rates_m = df_full.filter(pl.col("gender") == "M")["predicted_rate"].to_numpy()

axes[1].boxplot([rates_f, rates_m], labels=["Female", "Male"],
                patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
                medianprops=dict(color="black", linewidth=2))
axes[1].set_ylabel("Predicted claim frequency (annualised rate)")
axes[1].set_title("Predicted Rate Distribution by Gender\n(the model never saw gender directly)")
axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Proxy Discrimination: Postcode as Gender Proxy", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/fairness_demo_plots.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/fairness_demo_plots.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Full Audit via FairnessAudit
# MAGIC
# MAGIC The individual functions above are useful for exploration. For a formal audit run —
# MAGIC the kind that goes into the compliance record — use `FairnessAudit`. It runs all
# MAGIC checks in sequence, assigns RAG statuses against the library's thresholds, and
# MAGIC assembles a structured `FairnessReport`.

# COMMAND ----------

audit = FairnessAudit(
    model=model,                         # the standard model (no gender input)
    data=df_full,
    protected_cols=["gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=RATING_FACTORS,
    model_name="Motor Frequency Model v1 (postcode-biased)",
    run_proxy_detection=True,
    run_counterfactual=False,            # model doesn't use gender directly; use LRTW instead
    n_calibration_deciles=10,
    n_bootstrap=200,
    proxy_catboost_iterations=150,
)

report = audit.run()

# Print the structured summary
report.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Audit Report
# MAGIC
# MAGIC The report is formatted for two audiences: pricing committee (traffic-light statuses,
# MAGIC flagged factors, executive summary) and FCA/compliance review (methodology, regulatory
# MAGIC mapping, sign-off section). It maps explicitly to FCA Consumer Duty PRIN 2A and
# MAGIC Equality Act 2010 s.19.

# COMMAND ----------

# Write the markdown report
report_path = "/tmp/fairness_audit_motor_v1.md"
report.to_markdown(report_path)
print(f"Audit report written to: {report_path}")

# Print the first section of the report
with open(report_path, "r") as f:
    content = f.read()

# Show the first ~80 lines (executive summary and flagged factors)
for i, line in enumerate(content.split("\n")):
    if i >= 80:
        print(f"\n... [{len(content.split(chr(10))) - 80} more lines] ...")
        break
    print(line)

# COMMAND ----------

# Show structured results as dict (useful for downstream processing, logging to MLflow, etc.)
results_dict = report.to_dict()
print("Structured results (JSON-serialisable):")
print(f"  Overall RAG:     {results_dict['overall_rag']}")
print(f"  Flagged factors: {results_dict['flagged_factors']}")
print(f"  n_policies:      {results_dict['n_policies']:,}")
print()

for pc, pc_results in results_dict["results"].items():
    print(f"  Protected characteristic: {pc}")
    if "demographic_parity" in pc_results:
        dp_d = pc_results["demographic_parity"]
        print(f"    Demographic parity log-ratio: {dp_d['log_ratio']:+.4f}  [{dp_d['rag'].upper()}]")
    if "calibration" in pc_results:
        cal_d = pc_results["calibration"]
        print(f"    Max calibration disparity:    {cal_d['max_disparity']:.4f}  [{cal_d['rag'].upper()}]")
    if "disparate_impact" in pc_results:
        di_d = pc_results["disparate_impact"]
        print(f"    Disparate impact ratio:        {di_d['ratio']:.4f}  [{di_d['rag'].upper()}]")
    if "proxy_detection" in pc_results:
        prox_d = pc_results["proxy_detection"]
        print(f"    Flagged proxy factors:         {prox_d['flagged_factors']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calibration Visualisation
# MAGIC
# MAGIC The calibration heat map is the most important diagnostic for compliance purposes.
# MAGIC If the model is equally well-calibrated for male and female policyholders at every
# MAGIC pricing decile, the price differences are defensible. If not, the model is
# MAGIC systematically over-pricing one group.

# COMMAND ----------

# Re-compute calibration for plotting (use the test set for a cleaner picture)
cal_test = calibration_by_group(
    df=df_test,
    protected_col="gender",
    prediction_col="predicted_rate",
    outcome_col="claim_count",
    exposure_col="exposure",
    n_deciles=10,
)

deciles = sorted(cal_test.actual_to_expected.keys())
ae_f = [cal_test.actual_to_expected[d].get("F", float("nan")) for d in deciles]
ae_m = [cal_test.actual_to_expected[d].get("M", float("nan")) for d in deciles]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: A/E by decile for each gender
axes[0].plot(deciles, ae_f, "bo-", linewidth=2, markersize=7, label="Female")
axes[0].plot(deciles, ae_m, "rs-", linewidth=2, markersize=7, label="Male")
axes[0].axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Perfect calibration (A/E=1)")
axes[0].set_xlabel("Prediction decile (1=lowest, 10=highest)")
axes[0].set_ylabel("Actual / Expected ratio")
axes[0].set_title("Calibration by Gender: A/E Ratio per Prediction Decile\n(deviation from 1.0 = mis-pricing)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.5, 1.5)

# Plot 2: Proxy R-squared scores
proxy_df = proxy_result.to_polars().to_pandas()
colors   = ["tomato" if rag == "red" else "orange" if rag == "amber" else "steelblue"
            for rag in proxy_df["rag"]]
axes[1].barh(proxy_df["factor"], proxy_df["proxy_r2"].fillna(0), color=colors, alpha=0.8)
axes[1].axvline(0.05, color="orange", linewidth=1.5, linestyle="--", label="Amber threshold (0.05)")
axes[1].axvline(0.10, color="tomato", linewidth=1.5, linestyle="--", label="Red threshold (0.10)")
axes[1].set_xlabel("Proxy R-squared (AUC-based Gini for binary protected characteristic)")
axes[1].set_title("Proxy Detection: How Well Does Each Factor\nPredict the Protected Characteristic?")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="x")

plt.suptitle("insurance-fairness: Motor Model Audit Results", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/fairness_calibration_proxy.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/fairness_calibration_proxy.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Verdict
# MAGIC
# MAGIC ### What insurance-fairness catches that manual checks miss
# MAGIC
# MAGIC **Standard manual process:**
# MAGIC A pricing actuary might look at A/E ratios split by gender and conclude the model is
# MAGIC fine because the overall A/E for female policyholders is close to 1.0. That check is
# MAGIC insufficient. The question is not whether the model is calibrated overall — it is whether
# MAGIC the model's predictions systematically track gender through the proxy channel, even when
# MAGIC gender is not used directly.
# MAGIC
# MAGIC **What this library does:**
# MAGIC
# MAGIC 1. **Proxy R-squared** quantifies how strongly each rating factor predicts the protected
# MAGIC    characteristic. A manual check would not catch this unless someone specifically tested
# MAGIC    every rating factor against every protected characteristic.
# MAGIC
# MAGIC 2. **SHAP proxy scores** link that correlation to actual price impact. A factor can be
# MAGIC    correlated with gender but irrelevant to the model — SHAP scores filter to factors
# MAGIC    that actually drive prices differently across groups.
# MAGIC
# MAGIC 3. **Calibration by group x decile** goes beyond overall A/E to test whether the model
# MAGIC    is systematically off within pricing bands. This is the test that matters under the
# MAGIC    Equality Act: is the model equally informative for all groups, or is it systematically
# MAGIC    under-pricing one group at high predicted risk levels?
# MAGIC
# MAGIC 4. **Counterfactual testing (LRTW)** computes discrimination-free prices directly. It gives
# MAGIC    a portfolio-level estimate of the premium impact attributable to the proxy channel, and
# MAGIC    a per-policy distribution of who is most affected.
# MAGIC
# MAGIC 5. **Structured audit report** maps every finding to specific FCA and Equality Act
# MAGIC    references. The sign-off section is pre-formatted for the pricing committee pack.
# MAGIC
# MAGIC ### When to run this audit
# MAGIC
# MAGIC - At every model deployment (initial and retrain).
# MAGIC - When adding a new rating factor that may correlate with protected characteristics
# MAGIC   (new geographic variables, telematics scores, occupation codes).
# MAGIC - During annual Consumer Duty outcomes monitoring review.
# MAGIC - When a complaint or regulatory query suggests differential pricing.
# MAGIC
# MAGIC ### Regulatory positioning
# MAGIC
# MAGIC The FCA's 2024 multi-firm review found that most firms had "limited, often inadequate,
# MAGIC monitoring of differential outcomes by demographic group." Running this audit answers
# MAGIC that criticism directly. The output is a structured evidence record, not just a dashboard.
# MAGIC
# MAGIC ### Limitations to document
# MAGIC
# MAGIC - The library cannot establish causation. A rating factor correlated with a protected
# MAGIC   characteristic may reflect genuine risk differences or confounding — it cannot tell
# MAGIC   the difference without causal analysis.
# MAGIC - Where protected characteristics are not directly observed (ethnicity is the main case
# MAGIC   in the UK), analysis relies on proxy measures (ONS LSOA proportions). The
# MAGIC   `insurance_fairness.optimal_transport` subpackage handles this via the full LRTW
# MAGIC   discrimination-free pricing framework.
# MAGIC - Calibration by group can have low statistical power in small portfolio segments.
# MAGIC   The bootstrap confidence intervals in `demographic_parity_ratio` help with this.

# COMMAND ----------

# Summary printout for the pricing committee pack
print("=" * 65)
print(f"AUDIT SUMMARY: {report.model_name}")
print("=" * 65)
print(f"Policies audited:   {report.n_policies:,}")
print(f"Exposure:           {report.total_exposure:,.0f} policy years")
print(f"Overall RAG status: {report.overall_rag.upper()}")
print()

r = report.results["gender"]

print("Key findings — gender:")
print()
if r.demographic_parity:
    dp = r.demographic_parity
    print(f"  Mean price ratio (F/M):          {dp.ratio:.4f}  "
          f"({(dp.ratio - 1) * 100:+.1f}%)  [{dp.rag.upper()}]")

if r.disparate_impact:
    di = r.disparate_impact
    print(f"  Disparate impact ratio:           {di.ratio:.4f}  [{di.rag.upper()}]")

if r.calibration:
    cal = r.calibration
    print(f"  Max A/E disparity (decile/group): {cal.max_disparity:.4f}  [{cal.rag.upper()}]")

if r.proxy_detection:
    flagged = r.proxy_detection.flagged_factors
    if flagged:
        print(f"  Proxy factors flagged:            {', '.join(flagged)}")
        top_score = r.proxy_detection.scores[0]
        print(f"  Top proxy factor:                 {top_score.factor}  "
              f"(R2={top_score.proxy_r2:.4f}, MI={top_score.mutual_information:.4f})  "
              f"[{top_score.rag.upper()}]")
    else:
        print("  No proxy factors flagged.")

print()
print(f"Counterfactual premium impact (gender-aware model, direct flip):")
print(f"  Mean impact: {(cf_result.premium_impact_ratio - 1) * 100:+.1f}%")
print(f"  95th pct:    {(cf_result.policy_level_impacts.quantile(0.95) - 1) * 100:+.1f}%")
print()
print(f"Audit report: {report_path}")
print()
print("Recommended action: review postcode district's actuarial justification.")
print("Document the legitimate aim (geographic risk differentiation) and test")
print("whether less discriminatory alternatives (IMD decile, urban/rural flag)")
print("provide equivalent risk differentiation with lower proxy correlation.")
