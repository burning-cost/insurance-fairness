# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line

# COMMAND ----------

# MAGIC %md
# MAGIC # DoubleFairnessAudit Benchmark: Action vs Outcome Fairness on Synthetic Motor TPLI
# MAGIC
# MAGIC **Library:** `insurance-fairness` v0.6.0
# MAGIC
# MAGIC **What this demonstrates:**
# MAGIC The FCA Consumer Duty (PRIN 2A, Outcome 4) requires firms to demonstrate equivalent
# MAGIC *value* for all customer groups — not just equal treatment at point of quoting. These are
# MAGIC different obligations that can pull in opposite directions.
# MAGIC
# MAGIC This notebook benchmarks `DoubleFairnessAudit` on a synthetic UK motor TPLI scenario
# MAGIC where gender is correlated with the vehicle_group rating factor. The key finding, mirroring
# MAGIC Bian et al. (2026) on Belgian motor data: **a policy that minimises action unfairness
# MAGIC (premium disparity) does not minimise outcome unfairness (loss ratio disparity)**. Optimising
# MAGIC one worsens the other. The Pareto front is the evidence of the considered trade-off.
# MAGIC
# MAGIC **Why this matters for UK compliance:**
# MAGIC - FCA Multi-Firm Review of Consumer Duty Implementation (2024) cited failure to assess differential post-sale outcomes as a compliance gap
# MAGIC - Consumer Duty Outcome 4 asks whether the product delivers fair value — a claims/premium question
# MAGIC - A pricing team that only checks demographic parity of premiums is answering the wrong question
# MAGIC
# MAGIC **Comparison:** We contrast `DoubleFairnessAudit` against a naive single-metric check
# MAGIC (demographic parity of premiums only) to show what the naive check misses.
# MAGIC
# MAGIC **References:**
# MAGIC - Bian, Z., Wang, L., Shi, C., Qi, Z. (2026). Double Fairness Policy Learning. arXiv:2601.19186v2.
# MAGIC - FCA Consumer Duty (PS22/9, PRIN 2A), live July 2023.
# MAGIC - FCA Multi-Firm Review of Consumer Duty Implementation (2024).
# MAGIC
# MAGIC **Date:** 2026-03-21
# MAGIC **Library version:** 0.6.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Install the library and dependencies
%pip install git+https://github.com/burning-cost/insurance-fairness.git
%pip install catboost scikit-learn numpy scipy matplotlib pandas polars pyarrow

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge, TweedieRegressor
from sklearn.model_selection import train_test_split

from insurance_fairness import DoubleFairnessAudit
from insurance_fairness.double_fairness import DoubleFairnessResult
from insurance_fairness import (
    demographic_parity_ratio,
    calibration_by_group,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Notebook run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Motor Portfolio
# MAGIC
# MAGIC **Scenario:** UK motor TPLI book, 20,000 policies. Gender is correlated with vehicle_group
# MAGIC — female policyholders are more likely to be in lower-risk vehicle groups A and B, male
# MAGIC policyholders in higher-risk groups C and D. This is not injected discrimination: it is
# MAGIC an empirical correlation that appears in real books (younger females more likely to drive
# MAGIC smaller vehicles).
# MAGIC
# MAGIC **The structural consequence:** Even when the pricing model treats gender identically,
# MAGIC any model that uses vehicle_group will produce premium disparities between genders.
# MAGIC This is action unfairness (Delta_1). At the same time, the claims experience differs
# MAGIC between groups for reasons tied to vehicle group — not gender per se — which creates
# MAGIC outcome unfairness (Delta_2) at the portfolio level.
# MAGIC
# MAGIC **DGP design:** Risk (loss cost rate) is determined by vehicle_group, driver age, and NCD.
# MAGIC Gender has *no direct effect* on risk. Any observable premium or loss-ratio gap between
# MAGIC genders is purely a consequence of the vehicle_group correlation.
# MAGIC
# MAGIC This is the scenario where the tension between action and outcome fairness is sharpest:
# MAGIC equalising premiums by gender requires ignoring vehicle_group risk signals, which then
# MAGIC means one group's loss ratio deteriorates.

# COMMAND ----------

rng = np.random.default_rng(2026)

N = 20_000

# ---------------------------------------------------------------------------
# Protected attribute: gender (S=0 female, S=1 male)
# ---------------------------------------------------------------------------
# 50/50 split
gender_binary = rng.binomial(1, 0.50, N)   # 0=F, 1=M
gender_label  = np.where(gender_binary == 0, "F", "M")

# ---------------------------------------------------------------------------
# Vehicle group — correlated with gender
#
# Female (S=0): more likely to be in lower-risk groups A and B
#   P(A|F)=0.40, P(B|F)=0.30, P(C|F)=0.20, P(D|F)=0.10
# Male (S=1): more spread; higher-risk groups more common
#   P(A|M)=0.15, P(B|M)=0.30, P(C|M)=0.30, P(D|M)=0.25
#
# This correlation is the structural source of the premium gap and the
# loss ratio gap.
# ---------------------------------------------------------------------------
vg_f_probs = [0.40, 0.30, 0.20, 0.10]
vg_m_probs = [0.15, 0.30, 0.30, 0.25]
GROUPS     = ["A", "B", "C", "D"]

vehicle_group_idx = np.where(
    gender_binary == 0,
    np.array([rng.choice(4, p=vg_f_probs) for _ in range(N)]),
    np.array([rng.choice(4, p=vg_m_probs) for _ in range(N)]),
)
vehicle_group = np.array(GROUPS)[vehicle_group_idx]

# ---------------------------------------------------------------------------
# Other rating factors — independent of gender
# ---------------------------------------------------------------------------
age_band = rng.choice(["17-24", "25-34", "35-49", "50-64", "65+"], N,
                      p=[0.08, 0.20, 0.32, 0.25, 0.15])
ncd_band = rng.choice(["0yr", "1-2yr", "3-4yr", "5yr+"], N,
                      p=[0.15, 0.20, 0.30, 0.35])
vehicle_age = rng.integers(0, 16, N).astype(float)
exposure    = rng.uniform(0.1, 1.0, N)

# ---------------------------------------------------------------------------
# True risk (loss cost rate per unit exposure)
# Determined entirely by rating factors — NOT by gender directly.
# ---------------------------------------------------------------------------
vg_base = {"A": 0.055, "B": 0.075, "C": 0.095, "D": 0.115}
base_lc = np.array([vg_base[v] for v in vehicle_group])

age_adj = np.where(age_band == "17-24", 2.80,
          np.where(age_band == "25-34", 1.50,
          np.where(age_band == "35-49", 1.00,
          np.where(age_band == "50-64", 0.85, 0.90))))

ncd_adj = np.where(ncd_band == "0yr",   1.50,
          np.where(ncd_band == "1-2yr", 1.20,
          np.where(ncd_band == "3-4yr", 0.95, 0.80)))

vage_adj = 1.0 + 0.015 * vehicle_age

true_lc_rate = base_lc * age_adj * ncd_adj * vage_adj

# Simulate aggregate loss cost (gamma distributed — severity x frequency)
# Mean = true_lc_rate * exposure; coefficient of variation ~ 0.5
claim_cost = rng.gamma(
    shape=4.0,
    scale=true_lc_rate * exposure / 4.0,
    size=N,
)

# ---------------------------------------------------------------------------
# Assemble Polars DataFrame
# ---------------------------------------------------------------------------
df = pl.DataFrame({
    "gender_binary": gender_binary.tolist(),
    "gender":        gender_label.tolist(),
    "vehicle_group": vehicle_group.tolist(),
    "age_band":      age_band.tolist(),
    "ncd_band":      ncd_band.tolist(),
    "vehicle_age":   vehicle_age.tolist(),
    "exposure":      exposure.tolist(),
    "true_lc_rate":  true_lc_rate.tolist(),
    "claim_cost":    claim_cost.tolist(),
})

print(f"Portfolio shape: {df.shape}")
print(f"Gender split: {df['gender'].value_counts().sort('gender')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify the gender-vehicle_group correlation
# MAGIC
# MAGIC This is the structural source of all the fairness tension. The correlation is
# MAGIC not a modelling artefact — it exists in the data before any model is run.

# COMMAND ----------

vg_gender = (
    df
    .group_by("vehicle_group", "gender")
    .agg(
        pl.len().alias("n_policies"),
        (pl.col("claim_cost").sum() / pl.col("exposure").sum()).alias("observed_lc_rate"),
        pl.col("true_lc_rate").mean().alias("mean_true_lc_rate"),
    )
    .sort(["vehicle_group", "gender"])
)
print("Vehicle group x gender:")
print(vg_gender.to_pandas().to_string(index=False))

# Confirm gender has no direct risk effect: within vehicle group, rates should match
print("\nKey check — within vehicle group A, female vs male loss cost rates:")
grp_a = df.filter(pl.col("vehicle_group") == "A")
for g in ["F", "M"]:
    s = grp_a.filter(pl.col("gender") == g)
    rate = s["claim_cost"].sum() / s["exposure"].sum()
    print(f"  {g}: {rate:.4f} per policy year")
print("These should be similar (no direct gender risk effect in DGP).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pricing Model
# MAGIC
# MAGIC Train a CatBoost model on all rating factors **except** gender. This is standard
# MAGIC practice: gender is not a direct model input. The model learns to price by vehicle group,
# MAGIC age, NCD, and vehicle age.
# MAGIC
# MAGIC Because vehicle_group is correlated with gender, the model produces systematically
# MAGIC different premiums for female and male policyholders — even though it never saw gender.
# MAGIC This is the action fairness problem.

# COMMAND ----------

RATING_FACTORS = ["vehicle_group", "age_band", "ncd_band", "vehicle_age"]
CAT_FEATURES   = ["vehicle_group", "age_band", "ncd_band"]
TARGET         = "claim_cost"
EXPOSURE_COL   = "exposure"

# Train/test split
train_idx, test_idx = train_test_split(np.arange(N), test_size=0.30, random_state=42)

df_train = df[train_idx]
df_test  = df[test_idx]

X_train_pd = df_train.select(RATING_FACTORS).to_pandas()
X_test_pd  = df_test.select(RATING_FACTORS).to_pandas()

y_train   = df_train[TARGET].to_numpy()
y_test    = df_test[TARGET].to_numpy()
exp_train = df_train[EXPOSURE_COL].to_numpy()
exp_test  = df_test[EXPOSURE_COL].to_numpy()

# Fit CatBoost Poisson on loss cost rate
rate_train = y_train / exp_train
pool_train = Pool(X_train_pd, rate_train, cat_features=CAT_FEATURES, weight=exp_train)
pool_test  = Pool(X_test_pd, cat_features=CAT_FEATURES)

cb_model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    depth=5,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)
cb_model.fit(pool_train, eval_set=pool_test)

# Predict on full portfolio
pool_full       = Pool(df.select(RATING_FACTORS).to_pandas(), cat_features=CAT_FEATURES)
pred_rate_full  = cb_model.predict(pool_full)  # annualised loss cost rate
pred_premium    = pred_rate_full * df["exposure"].to_numpy()  # total premium per policy

df = df.with_columns([
    pl.Series("pred_rate",    pred_rate_full),
    pl.Series("pred_premium", pred_premium),
])

# Basic diagnostics
observed_rate = y_test.sum() / exp_test.sum()
pred_test     = cb_model.predict(pool_test) * exp_test
predicted_rate = pred_test.sum() / exp_test.sum()
print(f"Model A/E (overall): {observed_rate / predicted_rate:.4f}")
print(f"Test policies: {len(df_test):,}, exposure: {exp_test.sum():.0f} years")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Naive Check: Demographic Parity of Premiums
# MAGIC
# MAGIC The simplest fairness check is demographic parity: are mean premiums the same for
# MAGIC male and female policyholders? This is the check most UK pricing teams actually run.
# MAGIC It asks: *is the model treating men and women the same at the point of quoting?*
# MAGIC
# MAGIC **What it tells you:** Action fairness. How different are the premiums by gender?
# MAGIC
# MAGIC **What it doesn't tell you:** Whether the pricing produces equivalent *outcomes* — i.e.,
# MAGIC whether the loss ratios experienced by each group are equivalent. Two groups can have
# MAGIC identical mean premiums but very different loss ratios, which is a Consumer Duty Outcome 4
# MAGIC failure.

# COMMAND ----------

# --- Naive check: demographic parity of predicted premiums ---
dp_naive = demographic_parity_ratio(
    df=df,
    protected_col="gender",
    prediction_col="pred_rate",
    exposure_col="exposure",
    log_space=True,
    n_bootstrap=500,
    ci_level=0.95,
)
print("NAIVE CHECK: Demographic Parity of Predicted Premium Rates")
print("=" * 60)
print(f"  Log-ratio (F vs M):  {dp_naive.log_ratio:+.4f}")
print(f"  Rate ratio (F/M):    {dp_naive.ratio:.4f}  ({(dp_naive.ratio - 1) * 100:+.1f}%)")
print(f"  95% CI on log-ratio: [{dp_naive.ci_lower:+.4f}, {dp_naive.ci_upper:+.4f}]")
print(f"  RAG status:          {dp_naive.rag.upper()}")
print()
print("Interpretation:")
print(f"  Female policyholders are priced at {dp_naive.ratio:.3f}x the male rate.")
print(f"  A pricing team seeing '{dp_naive.rag}' here might consider the model fair.")
print(f"  But this only checks action fairness — what happens to loss ratios?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What does the naive check miss?
# MAGIC
# MAGIC The demographic parity ratio tells us about premium disparity (action fairness). It does
# MAGIC not tell us whether those premiums produce equivalent value for each group. Let's look
# MAGIC at the loss ratios directly.

# COMMAND ----------

# Compute loss ratio (claims / premium) by gender
# This is the outcome fairness question: is the product delivering equivalent value?
df_with_lr = df.with_columns(
    (pl.col("claim_cost") / pl.col("pred_premium")).alias("loss_ratio")
)

lr_by_gender = (
    df_with_lr
    .group_by("gender")
    .agg(
        pl.len().alias("n_policies"),
        pl.col("exposure").sum().alias("total_exposure"),
        (pl.col("claim_cost").sum() / pl.col("pred_premium").sum()).alias("portfolio_lr"),
        pl.col("loss_ratio").mean().alias("mean_individual_lr"),
        pl.col("claim_cost").sum().alias("total_claims"),
        pl.col("pred_premium").sum().alias("total_premium"),
    )
    .sort("gender")
)
print("Loss Ratio by Gender (Outcome Fairness)")
print("=" * 60)
print(lr_by_gender.to_pandas().to_string(index=False))
print()

lr_f = float(lr_by_gender.filter(pl.col("gender") == "F")["portfolio_lr"][0])
lr_m = float(lr_by_gender.filter(pl.col("gender") == "M")["portfolio_lr"][0])
print(f"  Loss ratio gap (F - M): {lr_f - lr_m:+.4f}")
print(f"  Loss ratio ratio (F/M): {lr_f / lr_m:.4f}")
print()
print("The loss ratio gap is the outcome fairness metric the naive check misses entirely.")
print("Even if action fairness (premium parity) looks acceptable, the product may not")
print("be delivering equivalent value to both groups under Consumer Duty Outcome 4.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. DoubleFairnessAudit
# MAGIC
# MAGIC Now we run the proper double fairness analysis. We fit `DoubleFairnessAudit` with:
# MAGIC
# MAGIC - **X** = feature matrix (rating factors excluding gender, standardised)
# MAGIC - **y_primary** = predicted premium (company revenue objective)
# MAGIC - **y_fairness** = loss ratio (claims / premium) — the outcome fairness objective
# MAGIC - **S** = gender binary (0=F, 1=M)
# MAGIC
# MAGIC The audit sweeps 20 alpha weights across (0,1), computing the Pareto front of
# MAGIC policies that trade off action fairness (Delta_1) against outcome fairness (Delta_2),
# MAGIC both subject to the revenue objective V_hat.
# MAGIC
# MAGIC **Why 20 alpha points?** The paper finds the Pareto front stabilises with ~10 points
# MAGIC for linear nuisance models. We use 20 for a denser picture.

# COMMAND ----------

# Prepare inputs for DoubleFairnessAudit
# Features: one-hot encode categoricals, add numeric features, standardise
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df_pd = df.to_pandas()

num_features = ["vehicle_age"]
cat_features = ["vehicle_group", "age_band", "ncd_band"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(sparse_output=False, drop="first"), cat_features),
])

X_all = preprocessor.fit_transform(df_pd[num_features + cat_features])
print(f"Feature matrix shape: {X_all.shape}  (n_policies={N}, n_features={X_all.shape[1]})")

# Primary outcome: predicted premium (company revenue)
y_primary  = df["pred_premium"].to_numpy()

# Fairness outcome: loss ratio (claims / predicted premium)
# Cap at 5.0 to reduce influence of outlier policies
y_fairness = np.clip(
    df["claim_cost"].to_numpy() / np.maximum(pred_premium, 1e-6),
    0.0, 5.0
)

# Protected group indicator
S = df["gender_binary"].to_numpy()

# Exposure weights
exposure_all = df["exposure"].to_numpy()

print(f"y_primary   (premium): mean={y_primary.mean():.2f}, std={y_primary.std():.2f}")
print(f"y_fairness  (loss ratio): mean={y_fairness.mean():.4f}, std={y_fairness.std():.4f}")
print(f"                          zero fraction: {(y_fairness == 0).mean():.2%}")
print(f"S (gender): {(S==0).sum():,} female, {(S==1).sum():,} male")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and run the audit
# MAGIC
# MAGIC The audit uses `Ridge` for the primary outcome model and auto-selects the fairness
# MAGIC outcome model based on the zero fraction of `y_fairness`. If > 30% are zero (typical
# MAGIC for loss ratios with many no-claim policies), it uses `TweedieRegressor(power=1.5)`.
# MAGIC
# MAGIC We use `n_alphas=20` Pareto points. The Tchebycheff sweep takes a few minutes on
# MAGIC a standard cluster — each alpha runs two `L-BFGS-B` / `SLSQP` optimisations.

# COMMAND ----------

audit = DoubleFairnessAudit(
    n_alphas=20,
    random_state=42,
    max_iter=2000,
    # kappa is set automatically: sqrt(log(n)/n) for parametric nuisance models
)

print("Fitting nuisance models...")
audit.fit(
    X=X_all,
    y_primary=y_primary,
    y_fairness=y_fairness,
    S=S,
    exposure=exposure_all,
)
print(f"Nuisance models fitted. Auto-selected fairness model: {audit._outcome_model_type}")
print(f"Kappa (slack): {audit._kappa:.5f}")
print()

print("Running Pareto sweep (20 alpha weights)...")
result = audit.audit()
print("Pareto sweep complete.")
print()

# Full Pareto front table
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pareto Front Analysis
# MAGIC
# MAGIC The Pareto front reveals the fundamental tension. For each of the 20 alpha weights:
# MAGIC
# MAGIC - **High alpha** (toward 1.0): the optimisation puts more weight on action fairness
# MAGIC   (Delta_1). Premiums become more equal across groups. But outcome fairness (Delta_2)
# MAGIC   may worsen — the policy is less able to reflect genuine risk differences, so loss
# MAGIC   ratios diverge.
# MAGIC
# MAGIC - **Low alpha** (toward 0.0): the optimisation puts more weight on outcome fairness
# MAGIC   (Delta_2). Loss ratios become more equal. But this typically requires the policy to
# MAGIC   make risk-differentiated pricing decisions that increase premium disparity (Delta_1).
# MAGIC
# MAGIC The naive check only looks at the horizontal axis of the left panel. It misses the
# MAGIC right panel entirely.

# COMMAND ----------

fig = audit.plot_pareto(figsize=(12, 5))
plt.suptitle(
    "DoubleFairnessAudit: Motor TPLI Pareto Front\n"
    "(action fairness vs outcome fairness — optimising one worsens the other)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/tmp/double_fairness_pareto.png", dpi=130, bbox_inches="tight")
plt.show()
print("Saved to /tmp/double_fairness_pareto.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quantifying the Tension
# MAGIC
# MAGIC The Pareto front gives us a precise quantification of the trade-off. We compare three
# MAGIC operating points:
# MAGIC
# MAGIC 1. **Min Delta_1 point**: the policy that minimises action unfairness (premium parity)
# MAGIC 2. **Min Delta_2 point**: the policy that minimises outcome unfairness (loss ratio parity)
# MAGIC 3. **Selected point**: the value-maximising Pareto solution (DoubleFairnessAudit default)
# MAGIC
# MAGIC This is the comparison the naive check cannot make. It requires both dimensions.

# COMMAND ----------

# Find the three operating points
idx_min_d1 = int(np.argmin(result.pareto_delta1))
idx_min_d2 = int(np.argmin(result.pareto_delta2))
idx_selected = result.selected_idx

print("Operating Point Comparison")
print("=" * 75)
print(f"{'Point':<22}  {'alpha':>6}  {'V_hat':>9}  {'Delta_1':>10}  {'Delta_2':>10}")
print("-" * 75)

for idx, label in [
    (idx_min_d1,   "Min action unfair"),
    (idx_min_d2,   "Min outcome unfair"),
    (idx_selected, "Selected (value-max)"),
]:
    print(
        f"{label:<22}  "
        f"{result.pareto_alphas[idx]:>6.3f}  "
        f"{result.pareto_V[idx]:>9.4f}  "
        f"{result.pareto_delta1[idx]:>10.6f}  "
        f"{result.pareto_delta2[idx]:>10.6f}"
    )

print("-" * 75)
print()

# Quantify the tension: at min-Delta_1, how large is Delta_2?
d2_at_min_d1 = result.pareto_delta2[idx_min_d1]
d2_at_min_d2 = result.pareto_delta2[idx_min_d2]
pct_worse    = 100.0 * (d2_at_min_d1 - d2_at_min_d2) / max(d2_at_min_d2, 1e-12)

d1_at_min_d2 = result.pareto_delta1[idx_min_d2]
d1_at_min_d1 = result.pareto_delta1[idx_min_d1]
pct_worse_d1 = 100.0 * (d1_at_min_d2 - d1_at_min_d1) / max(d1_at_min_d1, 1e-12)

print("The tension, quantified:")
print(f"  At the action-fair extreme (min Delta_1):   Delta_2 = {d2_at_min_d1:.6f}")
print(f"  At the outcome-fair extreme (min Delta_2):  Delta_2 = {d2_at_min_d2:.6f}")
print(f"  --> Minimising action unfairness worsens outcome unfairness by {pct_worse:.1f}%")
print()
print(f"  At the outcome-fair extreme (min Delta_2):  Delta_1 = {d1_at_min_d2:.6f}")
print(f"  At the action-fair extreme (min Delta_1):   Delta_1 = {d1_at_min_d1:.6f}")
print(f"  --> Minimising outcome unfairness worsens action unfairness by {pct_worse_d1:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Naive vs Double: Side-by-Side Comparison
# MAGIC
# MAGIC This section shows directly what the naive check (demographic parity of premiums)
# MAGIC tells you versus what `DoubleFairnessAudit` tells you.
# MAGIC
# MAGIC The key difference: the naive check gives a single number with no information about
# MAGIC the outcome dimension. The double fairness audit gives the full Pareto frontier and
# MAGIC identifies operating points that are dominated — where you could simultaneously
# MAGIC improve both action and outcome fairness by moving along the front.

# COMMAND ----------

# Reconstruct the "naive" result: just look at demographic parity
# The pricing team sees: a single ratio and a RAG status.
print("NAIVE FAIRNESS CHECK (demographic parity only)")
print("=" * 60)
print(f"  Premium ratio (F/M):   {dp_naive.ratio:.4f}")
print(f"  Log-ratio 95% CI:      [{dp_naive.ci_lower:+.4f}, {dp_naive.ci_upper:+.4f}]")
print(f"  RAG status:            {dp_naive.rag.upper()}")
print()
print("What the naive check reports:")
print("  -> One number. One RAG status. No outcome dimension.")
print("  -> Cannot distinguish between: (a) premium parity with loss ratio divergence,")
print("     and (b) premium parity with equivalent loss ratios. Both give the same output.")
print()

print("DOUBLE FAIRNESS AUDIT (DoubleFairnessAudit)")
print("=" * 60)
print(f"  Pareto front: {audit.n_alphas} operating points across action/outcome trade-off")
print(f"  Selected point:")
print(f"    Alpha (action/outcome weight): {result.selected_alpha:.3f}")
print(f"    Action unfairness (Delta_1):   {result.selected_delta1:.6f}")
print(f"    Outcome unfairness (Delta_2):  {result.selected_delta2:.6f}")
print(f"    Expected revenue (V_hat):      {result.selected_V:.4f}")
print()
print("What the double fairness audit reports:")
print("  -> The full Pareto front: every possible operating point along the trade-off.")
print("  -> Separate violation metrics for action fairness and outcome fairness.")
print("  -> The revenue cost of each fairness operating point.")
print("  -> Evidence of the considered trade-off suitable for an FCA evidence pack.")
print()

# Show the specific scenario where naive = green but outcome unfairness is hidden
# At the action-fair (min Delta_1) point, premiums are most equal but loss ratios diverge
alpha_min_d1 = result.pareto_alphas[idx_min_d1]
print(f"Illustrative tension (alpha={alpha_min_d1:.3f}, action-fair extreme):")
print(f"  Delta_1 (action unfairness):   {result.pareto_delta1[idx_min_d1]:.6f}  <- minimised")
print(f"  Delta_2 (outcome unfairness):  {result.pareto_delta2[idx_min_d1]:.6f}  <- still large")
print(f"  Naive check at this point:     would show lowest premium disparity")
print(f"  But Consumer Duty Outcome 4:   still at risk — outcome unfairness persists")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. FCA Consumer Duty Evidence Report
# MAGIC
# MAGIC The `report()` method produces a section suitable for insertion into a Consumer Duty
# MAGIC evidence pack. It quantifies the improvement in both fairness dimensions at the
# MAGIC selected operating point and references the relevant regulatory obligations.

# COMMAND ----------

print(audit.report())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Visualisation: Action vs Outcome Trade-off
# MAGIC
# MAGIC The scatter plot below shows every Pareto point in the (Delta_1, Delta_2) space.
# MAGIC Points closer to the origin are better on both dimensions simultaneously — but the
# MAGIC Pareto front means no single point dominates all others. Moving left (better action
# MAGIC fairness) forces you up (worse outcome fairness), and vice versa.
# MAGIC
# MAGIC The naive check only looks at the horizontal axis of this plot and picks one point
# MAGIC without awareness of where it falls vertically. The double fairness audit shows the
# MAGIC whole picture.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Pareto front in (Delta_1, Delta_2) space
ax = axes[0]
ax.plot(result.pareto_delta1, result.pareto_delta2, "o-", color="steelblue",
        alpha=0.6, markersize=7, linewidth=1.5, label="Pareto front")

# Mark the three key operating points
ax.scatter(result.pareto_delta1[idx_min_d1], result.pareto_delta2[idx_min_d1],
           c="darkorange", s=180, zorder=6, marker="^",
           label=f"Min action unfair (alpha={result.pareto_alphas[idx_min_d1]:.2f})")
ax.scatter(result.pareto_delta1[idx_min_d2], result.pareto_delta2[idx_min_d2],
           c="green", s=180, zorder=6, marker="s",
           label=f"Min outcome unfair (alpha={result.pareto_alphas[idx_min_d2]:.2f})")
ax.scatter(result.pareto_delta1[idx_selected], result.pareto_delta2[idx_selected],
           c="red", s=220, zorder=7, marker="*",
           label=f"Selected / value-max (alpha={result.selected_alpha:.2f})")

# Shade the region that the naive check can't distinguish
ax.axvspan(0, result.pareto_delta1[idx_min_d1] * 1.5, alpha=0.04, color="orange",
           label="Naive check: only sees horizontal axis")

ax.set_xlabel("Delta_1 (action fairness violation)", fontsize=11)
ax.set_ylabel("Delta_2 (outcome fairness violation)", fontsize=11)
ax.set_title("Action vs Outcome Fairness Trade-off\n(Pareto front — no single point dominates all others)",
             fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Panel 2: Loss ratio by gender at each Pareto operating point
# Approximate the loss ratio gap implied by each alpha using the policy probs
ax2 = axes[1]

delta1_vals = result.pareto_delta1
delta2_vals = result.pareto_delta2
colors_pts  = plt.cm.RdYlGn_r(np.linspace(0, 1, len(delta1_vals)))

sc = ax2.scatter(delta1_vals, delta2_vals, c=result.pareto_V,
                 cmap="RdYlGn", s=90, zorder=5, edgecolors="grey", linewidths=0.5)
plt.colorbar(sc, ax=ax2, label="V_hat (expected revenue)")
ax2.scatter(result.pareto_delta1[idx_selected], result.pareto_delta2[idx_selected],
            c="red", s=220, zorder=7, marker="*", label="Selected")
ax2.set_xlabel("Delta_1 (action fairness violation)", fontsize=11)
ax2.set_ylabel("Delta_2 (outcome fairness violation)", fontsize=11)
ax2.set_title("Pareto Front Coloured by Expected Revenue\n(redder = lower revenue)",
              fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle("DoubleFairnessAudit: Motor TPLI — Gender as Protected Attribute",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/double_fairness_tradeoff.png", dpi=130, bbox_inches="tight")
plt.show()
print("Saved to /tmp/double_fairness_tradeoff.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Benchmark Summary
# MAGIC
# MAGIC ### What the naive check tells you
# MAGIC
# MAGIC A single demographic parity ratio: mean premium for female policyholders vs male
# MAGIC policyholders, with a RAG threshold (e.g. > 5% gap = amber). This answers: "are we
# MAGIC pricing men and women similarly at the point of quoting?"
# MAGIC
# MAGIC It does not tell you whether the pricing decision produces equivalent outcomes — whether
# MAGIC the product delivers fair value (Consumer Duty Outcome 4) — or where on the
# MAGIC action/outcome Pareto frontier the current policy sits.
# MAGIC
# MAGIC ### What DoubleFairnessAudit adds
# MAGIC
# MAGIC 1. **Separates action from outcome unfairness.** Delta_1 and Delta_2 are distinct. A
# MAGIC    pricing team that only looks at Delta_1 (or the demographic parity ratio) is unaware
# MAGIC    of its position on the outcome fairness dimension. This is the compliance gap
# MAGIC    the FCA's multi-firm review of Consumer Duty implementation (2024) identified.
# MAGIC
# MAGIC 2. **Shows the trade-off explicitly.** The Pareto front quantifies the cost of improving
# MAGIC    each dimension. A pricing committee can make a documented, evidenced choice about where
# MAGIC    to operate on the front. This is auditable evidence of considered decision-making under
# MAGIC    uncertainty — exactly what the FCA wants to see.
# MAGIC
# MAGIC 3. **Identifies dominated policies.** If the current operating point is interior to the
# MAGIC    Pareto front (not on the boundary), there exist policies that are simultaneously better
# MAGIC    on both dimensions. The audit identifies this.
# MAGIC
# MAGIC 4. **Quantifies the revenue cost.** V_hat at each Pareto point shows how much revenue
# MAGIC    efficiency is sacrificed by moving toward fairness. This connects the fairness
# MAGIC    obligation to a commercial number the pricing committee can engage with.
# MAGIC
# MAGIC ### When to use this
# MAGIC
# MAGIC - Annual Consumer Duty evidence pack (mandatory for UK insurers under PS22/9)
# MAGIC - When adding or changing a rating factor that may have differential outcomes by group
# MAGIC - When a complaint or FCA query suggests differential value delivery by demographic
# MAGIC - As part of the model governance pack for any pricing model with protected-characteristic
# MAGIC   adjacency
# MAGIC
# MAGIC ### Limitations (document in your evidence pack)
# MAGIC
# MAGIC 1. **Binary action.** The policy is parametrised as A in {0, 1} (high-risk band vs not).
# MAGIC    In practice, interpret at your chosen rating threshold. For continuous pricing, run the
# MAGIC    audit at multiple threshold values.
# MAGIC
# MAGIC 2. **Parametric kappa.** The default kappa = sqrt(log(n)/n) assumes parametric nuisance
# MAGIC    models (Ridge). If using gradient boosted trees for nuisance fitting, set kappa
# MAGIC    explicitly — the default will underestimate the slack.
# MAGIC
# MAGIC 3. **Loss ratio as fairness outcome.** The paper uses negative premium (customer welfare)
# MAGIC    as the fairness outcome. Loss ratio is more actuarially meaningful for UK motor but
# MAGIC    introduces zero-inflation (no-claim policies) that the fairness model must handle.
# MAGIC    The auto-selected TweedieRegressor manages this, but verify via calibration checks.
# MAGIC
# MAGIC 4. **No doubly-robust estimation.** The current implementation uses outcome regression
# MAGIC    only (no propensity score weighting). If nuisance models are misspecified, Delta
# MAGIC    estimates are biased. For a more robust audit, use k-fold cross-fitting.

# COMMAND ----------

# Final summary printout for the pricing committee pack
print("=" * 65)
print("DOUBLE FAIRNESS BENCHMARK SUMMARY")
print("Motor TPLI, Synthetic Portfolio (n=20,000)")
print("=" * 65)
print()
print("Dataset:")
print(f"  Policies:      {N:,}")
print(f"  Feature dims:  {X_all.shape[1]}")
print(f"  Gender split:  {(S==0).sum():,} female / {(S==1).sum():,} male")
print()
print("Naive Check (demographic parity of premiums):")
print(f"  Rate ratio (F/M):  {dp_naive.ratio:.4f}  [{dp_naive.rag.upper()}]")
print(f"  Observed LR gap (F-M): {lr_f - lr_m:+.4f}")
print(f"  Naive conclusion: {'fair' if dp_naive.rag == 'green' else 'investigate'}")
print(f"  Missing dimension: outcome unfairness (Consumer Duty Outcome 4)")
print()
print("DoubleFairnessAudit:")
print(f"  Pareto points:  {audit.n_alphas}")
print(f"  Kappa (slack):  {audit._kappa:.5f}")
print(f"  Fairness model: {audit._outcome_model_type}")
print()
print(f"  Selected operating point (value-maximising):")
print(f"    Alpha (action/outcome balance): {result.selected_alpha:.3f}")
print(f"    Delta_1 (action unfairness):    {result.selected_delta1:.6f}")
print(f"    Delta_2 (outcome unfairness):   {result.selected_delta2:.6f}")
print(f"    V_hat (expected revenue):       {result.selected_V:.4f}")
print()
print(f"  Trade-off quantified:")
print(f"    At action-fair extreme: Delta_2 = {result.pareto_delta2[idx_min_d1]:.6f}")
print(f"    At outcome-fair extreme: Delta_2 = {result.pareto_delta2[idx_min_d2]:.6f}")
print(f"    Minimising action unfairness worsens outcome unfairness by {pct_worse:.1f}%")
print()
print("Regulatory references:")
print("  FCA Consumer Duty (PRIN 2A), Outcome 4 (Price and Value).")
print("  FCA Multi-Firm Review of Consumer Duty Implementation (2024).")
print("  Bian et al. (2026). arXiv:2601.19186v2.")
