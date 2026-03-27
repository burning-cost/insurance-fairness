"""
bias_metrics.py — exposure-weighted fairness metrics for UK motor pricing
=========================================================================

Shows the main bias metrics on a synthetic UK motor portfolio with a known
gender and postcode-diversity signal injected into the claim DGP.

Each metric answers a different regulatory question:

  calibration_by_group  — Is the model correctly calibrated for each group?
                          The primary Equality Act s.19 test: a well-calibrated
                          model does not systematically overcharge any group.
                          A/E disparity > 0.10 is amber; > 0.20 is red.

  demographic_parity    — Do different groups pay different average premiums?
                          This does NOT control for risk differences — a large
                          log-ratio may reflect genuine risk, not discrimination.
                          Use calibration to distinguish the two.

  disparate_impact_ratio — Ratio of adverse outcome rates between groups.
                           FCA Consumer Duty Outcome 4 monitor: values below
                           0.80 are conventionally adverse (4/5ths rule), but
                           apply this contextually in the UK, not mechanically.

  theil_index           — Decomposes premium inequality into within-group and
                          between-group components. A large between-group share
                          means inequality is driven by protected-group membership,
                          not by individual risk characteristics.

Run locally:
    python examples/bias_metrics.py

Or on Databricks: upload and run as a notebook cell.
"""

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

from insurance_fairness import (
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    theil_index,
    gini_by_group,
)

# ---------------------------------------------------------------------------
# 1. Synthetic UK motor portfolio — 15,000 policies
# ---------------------------------------------------------------------------
rng = np.random.default_rng(99)
n = 15_000

gender = rng.choice(["M", "F"], size=n)
vehicle_age = rng.integers(1, 15, n).astype(float)
driver_age = rng.integers(21, 75, n).astype(float)
ncd_years = rng.integers(0, 9, n).astype(float)
vehicle_group = rng.choice(["A", "B", "C", "D", "E"], size=n)

# Postcode districts with diversity profiles matching ONS 2021 Census
postcode_district = rng.choice(
    ["E1", "N1", "B1", "LS1", "SW1", "EX1"],
    size=n,
    p=[0.18, 0.15, 0.17, 0.17, 0.17, 0.16],
)
high_diversity = np.isin(postcode_district, ["E1", "N1", "B1"]).astype(float)

exposure = rng.uniform(0.3, 1.0, n)

# True DGP: postcode diversity loading (~£80/year equivalent) injected
# alongside a gender signal. Both are in the claim-generation process, so
# a model that ignores protected characteristics will find them indirectly
# through postcode and vehicle_group.
log_mu = (
    4.6
    + 0.04 * vehicle_age
    - 0.008 * ncd_years
    - 0.003 * (driver_age - 40)
    + 0.09 * (gender == "M").astype(float)   # injected gender signal
    + 0.10 * high_diversity                  # injected postcode-proxy signal
    + rng.normal(0, 0.35, n)
)
claim_amount = np.exp(log_mu) * exposure

# ---------------------------------------------------------------------------
# 2. Fit a simple model (no protected characteristics as inputs)
# ---------------------------------------------------------------------------
X = np.column_stack([vehicle_age, driver_age, ncd_years])
y = claim_amount / exposure

model = CatBoostRegressor(iterations=300, verbose=0)
model.fit(X, y, sample_weight=exposure)
predicted_rate = model.predict(X)

df = pl.DataFrame({
    "gender": gender,
    "postcode_district": postcode_district,
    "high_diversity_area": high_diversity,
    "vehicle_age": vehicle_age,
    "driver_age": driver_age,
    "ncd_years": ncd_years,
    "vehicle_group": vehicle_group,
    "exposure": exposure,
    "claim_amount": claim_amount,
    "predicted_rate": predicted_rate,
    "predicted_premium": predicted_rate * exposure,
})

# ---------------------------------------------------------------------------
# 3. Calibration by group — the primary Equality Act metric
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. CALIBRATION BY GROUP (Equality Act s.19 primary test)")
print("=" * 60)

cal = calibration_by_group(
    df,
    protected_col="gender",
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    n_deciles=10,
)
print(f"Max A/E disparity (gender):    {cal.max_disparity:.4f}  [{cal.rag.upper()}]")
print()
print("  A/E by group (aggregate):")
for grp, decile_dict in cal.actual_to_expected.items():
    ae_vals = list(decile_dict.values())
    print(f"    {grp}: mean A/E = {np.mean(ae_vals):.3f}")
print()
print("  Interpretation: A/E = 1.0 means perfectly calibrated for that group.")
print("  Disparity > 0.10 (amber) means the model is systematically over- or")
print("  under-predicting for one group. It does not prove discrimination — it")
print("  is the trigger for deeper investigation.")
print()

# ---------------------------------------------------------------------------
# 4. Demographic parity ratio — log-space for multiplicative models
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. DEMOGRAPHIC PARITY RATIO (Consumer Duty Outcome 4 monitor)")
print("=" * 60)

dp = demographic_parity_ratio(
    df,
    protected_col="gender",
    prediction_col="predicted_premium",
    exposure_col="exposure",
    log_space=True,     # correct for multiplicative GLM/GBM models
)
print(f"Log-ratio M vs F:   {dp.log_ratio:+.4f}  (ratio: {dp.ratio:.4f})  [{dp.rag.upper()}]")
print(f"Group mean premiums: {dp.group_means}")
print()
print("  Interpretation: log_ratio > 0 means male policyholders pay more on")
print("  average. This does NOT prove discrimination — males may genuinely")
print("  have higher claim rates. Use calibration to check whether the")
print("  disparity is justified by actual loss experience.")
print()

# ---------------------------------------------------------------------------
# 5. Disparate impact ratio
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. DISPARATE IMPACT RATIO")
print("=" * 60)

di = disparate_impact_ratio(
    df,
    protected_col="gender",
    prediction_col="predicted_premium",
    exposure_col="exposure",
)
print(f"DIR (F/M): {di.ratio:.4f}  [{di.rag.upper()}]")
print(f"  < 0.80 is conventionally adverse (4/5ths rule).")
print(f"  In UK insurance, apply this contextually — motor gender rating")
print(f"  is prohibited under EU/Retained Law, but the DIR is a useful")
print(f"  secondary diagnostic for monitoring differential pricing.")
print()

# ---------------------------------------------------------------------------
# 6. Theil index — inequality decomposition
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. THEIL INDEX — premium inequality decomposition")
print("=" * 60)

# Create a categorical diversity group for the Theil decomposition
df = df.with_columns(
    pl.when(pl.col("high_diversity_area") == 1.0)
    .then(pl.lit("high_diversity"))
    .otherwise(pl.lit("low_diversity"))
    .alias("area_diversity_group")
)

theil = theil_index(
    df,
    protected_col="area_diversity_group",
    prediction_col="predicted_premium",
    exposure_col="exposure",
)
between_share = theil.theil_between / theil.theil_total if theil.theil_total > 0 else 0.0
print(f"Theil T (total):        {theil.theil_total:.6f}")
print(f"Theil T (within-group): {theil.theil_within:.6f}")
print(f"Theil T (between-group):{theil.theil_between:.6f}")
print(f"Between-group share:    {between_share:.1%}")
print()
print("  Interpretation: between-group share > 10% means a meaningful")
print("  fraction of premium inequality is driven by group membership")
print("  (here: high vs low diversity area). Where group = protected")
print("  characteristic, this is a flag for a deeper proxy audit.")
print()

# ---------------------------------------------------------------------------
# 7. Gini by group
# ---------------------------------------------------------------------------
print("=" * 60)
print("5. GINI COEFFICIENT BY GROUP")
print("=" * 60)

gini = gini_by_group(
    df,
    protected_col="gender",
    prediction_col="predicted_premium",
    exposure_col="exposure",
)
print(f"Overall Gini:    {gini.overall_gini:.4f}")
print(f"Gini by group:   {gini.group_ginis}")
print(f"Max disparity:   {gini.max_disparity:.4f}")
print()
print("  Interpretation: similar Gini values across groups suggests the")
print("  risk-spread within each group is comparable. A large disparity")
print("  means the model is spreading risk very differently across groups,")
print("  which may indicate that proxy factors are doing different work")
print("  for different demographic segments.")
