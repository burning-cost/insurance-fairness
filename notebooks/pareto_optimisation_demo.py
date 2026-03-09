# Databricks notebook source
# MAGIC # insurance-fairness: NSGA-II Pareto Front Demo
# MAGIC
# MAGIC This notebook demonstrates the full Pareto optimisation workflow on synthetic data.
# MAGIC It shows how to:
# MAGIC
# MAGIC 1. Train two CatBoost models with different fairness/accuracy trade-offs
# MAGIC 2. Run NSGA-II to find the full Pareto front of ensemble weights
# MAGIC 3. Use TOPSIS to select an operating point with explicit preference weights
# MAGIC 4. Plot the Pareto front
# MAGIC 5. Compute the LipschitzMetric for both models
# MAGIC
# MAGIC **Regulatory context:** This workflow produces an auditable record of the
# MAGIC fairness-accuracy trade-off considered during model selection. Under FCA
# MAGIC Consumer Duty, firms should demonstrate they have considered differential
# MAGIC outcomes for protected-characteristic groups. The Pareto front makes this
# MAGIC consideration explicit and documented.

# COMMAND ----------
# MAGIC %pip install insurance-fairness[pareto] pymoo>=0.6.1 catboost polars pyarrow scikit-learn

# COMMAND ----------
# DBTITLE 1,Imports and setup

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

from insurance_fairness.pareto import (
    NSGA2FairnessOptimiser,
    FairnessProblem,
    LipschitzMetric,
    ParetoResult,
    topsis_select,
)

print("insurance-fairness version:", __import__("insurance_fairness").__version__)

# COMMAND ----------
# DBTITLE 1,Generate synthetic insurance data

rng = np.random.default_rng(2024)

N = 5_000  # policies

# Protected characteristic: gender (0 = female, 1 = male)
gender = rng.integers(0, 2, size=N)

# Rating factors (legitimate risk factors)
age = rng.integers(18, 70, size=N)
vehicle_age = rng.integers(0, 15, size=N)
ncd_years = rng.integers(0, 10, size=N)
annual_mileage = rng.integers(5000, 30000, size=N)

# Risk score (what the model should learn)
# Gender has NO direct causal effect - but is correlated with age distribution
true_risk = np.exp(
    0.03 * (age - 40)       # older drivers: lower risk
    - 0.02 * ncd_years        # NCD discount
    + 0.01 * vehicle_age      # older vehicles: slightly higher risk
    + rng.normal(0, 0.3, N)   # individual variation
)

# Make male drivers slightly younger on average (a proxy correlation)
age_adjustment = np.where(gender == 1, -3, 3)
age_adjusted = np.clip(age + age_adjustment, 18, 70)

# Actual claims: Poisson frequency with log link
frequency = true_risk * rng.uniform(0.8, 1.2, N)  # add noise
claims = rng.poisson(frequency * 0.3)  # expected frequency ~0.3 claims/year

# Exposure: uniform
exposure = rng.uniform(0.5, 1.0, N)

df = pl.DataFrame({
    "gender": gender.tolist(),
    "age": age_adjusted.tolist(),
    "vehicle_age": vehicle_age.tolist(),
    "ncd_years": ncd_years.tolist(),
    "annual_mileage": annual_mileage.tolist(),
    "claims": claims.tolist(),
    "exposure": exposure.tolist(),
})

print(f"Dataset: {N:,} policies")
print(f"Mean claims per policy: {df['claims'].mean():.3f}")
print(f"Gender split: {df.filter(pl.col('gender')==0)['gender'].len()} female, "
      f"{df.filter(pl.col('gender')==1)['gender'].len()} male")
print(df.head(5))

# COMMAND ----------
# DBTITLE 1,Train a standard model (includes gender as a feature)

FEATURES_WITH_GENDER = ["gender", "age", "vehicle_age", "ncd_years", "annual_mileage"]
FEATURES_WITHOUT_GENDER = ["age", "vehicle_age", "ncd_years", "annual_mileage"]

X_train = df.select(FEATURES_WITH_GENDER).to_pandas()
y_train = df["claims"].to_numpy().astype(float)
w_train = df["exposure"].to_numpy()

model_base = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=4,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
)
model_base.fit(X_train, y_train, sample_weight=w_train)

# Standard model predictions
pred_base = model_base.predict(X_train)
df = df.with_columns(pl.Series("pred_base", pred_base.tolist()))

print("Base model (with gender):")
print(f"  Overall mean prediction: {pred_base.mean():.4f}")
male_mask = df["gender"] == 1
female_mask = df["gender"] == 0
print(f"  Male mean prediction:    {df.filter(male_mask)['pred_base'].mean():.4f}")
print(f"  Female mean prediction:  {df.filter(female_mask)['pred_base'].mean():.4f}")
ratio = df.filter(male_mask)['pred_base'].mean() / df.filter(female_mask)['pred_base'].mean()
print(f"  M/F ratio:               {ratio:.4f}")

# COMMAND ----------
# DBTITLE 1,Train a fairness-aware model (excludes gender)

X_train_fair = df.select(FEATURES_WITHOUT_GENDER).to_pandas()

model_fair = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=4,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
)
model_fair.fit(X_train_fair, y_train, sample_weight=w_train)

pred_fair = model_fair.predict(X_train_fair)
df = df.with_columns(pl.Series("pred_fair", pred_fair.tolist()))

print("Fair model (without gender):")
print(f"  Overall mean prediction: {pred_fair.mean():.4f}")
print(f"  Male mean prediction:    {df.filter(male_mask)['pred_fair'].mean():.4f}")
print(f"  Female mean prediction:  {df.filter(female_mask)['pred_fair'].mean():.4f}")
ratio_fair = df.filter(male_mask)['pred_fair'].mean() / df.filter(female_mask)['pred_fair'].mean()
print(f"  M/F ratio:               {ratio_fair:.4f}")

# COMMAND ----------
# DBTITLE 1,Evaluate FairnessProblem objectives directly

# Evaluate the three objectives for each pure model (no mixing)
problem = FairnessProblem(
    models={"base": model_base, "fair": model_fair},
    X=df.select(FEATURES_WITH_GENDER),
    y=y_train,
    exposure=w_train,
    protected_col="gender",
)

obj_base = problem.evaluate(np.array([1.0, 0.0]))
obj_fair = problem.evaluate(np.array([0.0, 1.0]))

print("Objective values (all minimisation — lower is better):")
print(f"{'Objective':<30} {'Base model':>15} {'Fair model':>15}")
print("-" * 60)
obj_names = ["Neg. Gini (accuracy)", "Group unfairness", "CF unfairness"]
for name, b, f in zip(obj_names, obj_base, obj_fair):
    print(f"  {name:<28} {b:>15.4f} {f:>15.4f}")

print("\nNote: base model has lower neg_gini (better accuracy) but higher group")
print("unfairness. The Pareto front explores the full range of trade-offs.")

# COMMAND ----------
# DBTITLE 1,Run NSGA-II to find the Pareto front

optimiser = NSGA2FairnessOptimiser(
    models={"base": model_base, "fair": model_fair},
    X=df.select(FEATURES_WITH_GENDER),
    y=y_train,
    exposure=w_train,
    protected_col="gender",
    cf_tolerance=0.05,  # 5% premium change threshold
)

print("Running NSGA-II (pop_size=50, n_gen=100)...")
result = optimiser.run(pop_size=50, n_gen=100, seed=42, verbose=False)

print(result.summary())

# COMMAND ----------
# DBTITLE 1,Select operating point using TOPSIS

# Equal weights: balanced trade-off
idx_equal = result.selected_point(weights=None)
print("TOPSIS selection with equal weights:")
print(f"  Selected index: {idx_equal}")
print(f"  Ensemble weights: base={result.weights[idx_equal, 0]:.3f}, "
      f"fair={result.weights[idx_equal, 1]:.3f}")
print(f"  Objectives: neg_gini={result.F[idx_equal, 0]:.4f}, "
      f"group_unfairness={result.F[idx_equal, 1]:.4f}, "
      f"cf_unfairness={result.F[idx_equal, 2]:.4f}")

print()

# Accuracy-weighted: prioritise Gini
idx_accuracy = result.selected_point(weights=[0.6, 0.2, 0.2])
print("TOPSIS selection with accuracy-weighted preferences [0.6, 0.2, 0.2]:")
print(f"  Selected index: {idx_accuracy}")
print(f"  Ensemble weights: base={result.weights[idx_accuracy, 0]:.3f}, "
      f"fair={result.weights[idx_accuracy, 1]:.3f}")

print()

# Fairness-weighted: prioritise group fairness
idx_fairness = result.selected_point(weights=[0.2, 0.6, 0.2])
print("TOPSIS selection with fairness-weighted preferences [0.2, 0.6, 0.2]:")
print(f"  Selected index: {idx_fairness}")
print(f"  Ensemble weights: base={result.weights[idx_fairness, 0]:.3f}, "
      f"fair={result.weights[idx_fairness, 1]:.3f}")

# COMMAND ----------
# DBTITLE 1,Plot the Pareto front

try:
    fig = result.plot_front(highlight=idx_equal)
    fig.savefig("/tmp/pareto_front.png", dpi=150, bbox_inches="tight")
    display(fig)
    print("Plot saved to /tmp/pareto_front.png")
except ImportError:
    print("matplotlib not available - skipping plot")

# COMMAND ----------
# DBTITLE 1,Serialise and restore the Pareto result

import json
import tempfile, os

# Serialise to dict
d = result.to_dict()
print("Serialised keys:", list(d.keys()))
print(f"n_solutions: {d['n_solutions']}")
print(f"First F row: {[round(x, 4) for x in d['F'][0]]}")

# Write to JSON and read back
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    path = f.name
result.to_json(path)

with open(path) as f:
    loaded = json.load(f)
recovered = ParetoResult.from_dict(loaded)

assert recovered.n_solutions == result.n_solutions
print(f"\nRound-trip check: {recovered.n_solutions} solutions recovered correctly")
os.unlink(path)

# COMMAND ----------
# DBTITLE 1,LipschitzMetric on both models

# Encode gender as float for numeric distance computation
X_numeric = df.select(FEATURES_WITH_GENDER).to_pandas().astype(float).values

def insurance_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Insurance-aware distance metric.

    Uses log-space difference for age (multiplicative risk effect) and
    linear difference for ncd_years and vehicle_age.
    Features: [gender, age, vehicle_age, ncd_years, annual_mileage]
    """
    gender_diff = abs(x1[0] - x2[0])
    age_diff = abs(np.log(max(x1[1], 1)) - np.log(max(x2[1], 1)))
    vehicle_age_diff = abs(x1[2] - x2[2]) / 15.0  # normalise by max
    ncd_diff = abs(x1[3] - x2[3]) / 10.0
    mileage_diff = abs(np.log(max(x1[4], 1000)) - np.log(max(x2[4], 1000)))
    return float(np.sqrt(gender_diff**2 + age_diff**2 + vehicle_age_diff**2
                         + ncd_diff**2 + mileage_diff**2))

metric = LipschitzMetric(
    distance_fn=insurance_distance,
    n_pairs=2000,
    log_predictions=True,
    random_seed=42,
)

print("Lipschitz constant estimation (2,000 random pairs):")
print()

result_base = metric.compute(X_numeric, pred_base)
print("Base model (with gender):")
print(f"  Lipschitz constant (sample max): {result_base.lipschitz_constant:.4f}")
print(f"  95th pct ratio:                  {result_base.p95_ratio:.4f}")
print(f"  Median ratio:                    {result_base.p50_ratio:.4f}")
print(f"  Pairs sampled:                   {result_base.n_pairs_sampled:,}")

print()

X_fair_numeric = df.select(FEATURES_WITHOUT_GENDER).to_pandas().astype(float).values

def fair_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Distance metric for the 4-feature fair model."""
    age_diff = abs(np.log(max(x1[0], 1)) - np.log(max(x2[0], 1)))
    vehicle_age_diff = abs(x1[1] - x2[1]) / 15.0
    ncd_diff = abs(x1[2] - x2[2]) / 10.0
    mileage_diff = abs(np.log(max(x1[3], 1000)) - np.log(max(x2[3], 1000)))
    return float(np.sqrt(age_diff**2 + vehicle_age_diff**2 + ncd_diff**2 + mileage_diff**2))

result_fair_lip = metric.__class__(
    distance_fn=fair_distance, n_pairs=2000, log_predictions=True, random_seed=42
).compute(X_fair_numeric, pred_fair)

print("Fair model (without gender):")
print(f"  Lipschitz constant (sample max): {result_fair_lip.lipschitz_constant:.4f}")
print(f"  95th pct ratio:                  {result_fair_lip.p95_ratio:.4f}")
print(f"  Median ratio:                    {result_fair_lip.p50_ratio:.4f}")

print("\nNote: LipschitzMetric is EXPERIMENTAL and requires a carefully designed")
print("distance metric. The values above are illustrative — do not compare")
print("models with different distance metrics.")

# COMMAND ----------
# DBTITLE 1,FairnessAudit integration with run_pareto=True

from insurance_fairness import FairnessAudit

df_audit = df.with_columns(pl.Series("pred_base", pred_base.tolist()))

audit = FairnessAudit(
    model=None,
    data=df_audit,
    protected_cols=["gender"],
    prediction_col="pred_base",
    outcome_col="claims",
    exposure_col="exposure",
    factor_cols=["age", "vehicle_age", "ncd_years", "annual_mileage"],
    model_name="Base Model (with gender)",
    run_proxy_detection=False,
    run_counterfactual=False,
    run_pareto=True,
    pareto_models={"base": model_base, "fair": model_fair},
    pareto_pop_size=30,
    pareto_n_gen=50,
    pareto_seed=42,
)

print("Running FairnessAudit with run_pareto=True...")
report = audit.run()
report.summary()

if report.pareto_result is not None:
    print(f"\nPareto result attached: {report.pareto_result.n_solutions} solutions")
    d = report.to_dict()
    assert "pareto" in d, "Pareto result should be in to_dict() output"
    print("to_dict() includes pareto key: OK")

# COMMAND ----------
print("\nDemo complete. All components working correctly.")
