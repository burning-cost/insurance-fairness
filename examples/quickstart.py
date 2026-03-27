"""
insurance-fairness quickstart
==============================

Self-contained example: synthetic UK motor portfolio, CatBoost frequency model,
FCA-ready fairness audit with proxy detection.

Run locally (requires insurance-fairness, catboost, polars, numpy):

    python examples/quickstart.py

Or on Databricks — upload this file and run as a notebook cell or %run it.

The example injects a known postcode-ethnicity proxy structure so the audit
has something to find. In a real audit you would pass your fitted model and
your portfolio DataFrame.
"""

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness import FairnessAudit

# ---------------------------------------------------------------------------
# 1. Synthetic UK motor portfolio — 10,000 policies
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 10_000

gender = rng.choice(["M", "F"], size=n)
vehicle_age = rng.integers(1, 15, n).astype(float)
driver_age = rng.integers(21, 75, n).astype(float)
ncd_years = rng.integers(0, 9, n).astype(float)
vehicle_group = rng.choice(["A", "B", "C", "D"], size=n)

# Postcode districts: proxy for ethnicity (high-diversity areas: E1, B1).
# This creates the indirect discrimination signal the audit should detect.
postcode_district = rng.choice(["SW1", "E1", "M1", "B1", "LS1"], size=n)
high_diversity = np.isin(postcode_district, ["E1", "B1"]).astype(float)

exposure = rng.uniform(0.3, 1.0, n)

# Claim cost: slight gender correlation injected via postcode proxy.
# True DGP includes a postcode loading (~£70-90/year) that correlates
# with ethnicity-diverse areas — replicating Citizens Advice (2022)
# finding structure.
claim_amount = (
    np.exp(
        4.5
        + 0.05 * vehicle_age
        - 0.01 * ncd_years
        + 0.08 * (gender == "M").astype(float)  # injected gender signal
        + 0.12 * high_diversity                 # injected postcode-proxy signal
        + rng.normal(0, 0.4, n)
    )
    * exposure
)

# ---------------------------------------------------------------------------
# 2. Fit a CatBoost frequency model (without protected characteristic)
# ---------------------------------------------------------------------------
feature_cols = ["vehicle_age", "driver_age", "ncd_years"]
X = np.column_stack([vehicle_age, driver_age, ncd_years])
y = claim_amount / exposure

model = CatBoostRegressor(iterations=200, verbose=0)
model.fit(X, y, sample_weight=exposure)

# predicted_rate is the model output on the rate scale (claims per unit
# exposure). FairnessAudit expects a rate here — it multiplies by exposure
# internally when computing sum(predicted * exposure).
predicted_rate = model.predict(X)

df = pl.DataFrame(
    {
        "gender": gender,
        "vehicle_age": vehicle_age,
        "driver_age": driver_age,
        "ncd_years": ncd_years,
        "vehicle_group": vehicle_group,
        "postcode_district": postcode_district,
        "exposure": exposure,
        "claim_amount": claim_amount,
        "predicted_rate": predicted_rate,
    }
)

# ---------------------------------------------------------------------------
# 3. Run the fairness audit
# ---------------------------------------------------------------------------
# factor_cols may include factors NOT used as model inputs. The audit checks
# for proxy contamination in both model inputs and non-model factors — a
# factor not in the model can still correlate with protected characteristics
# and inform future model development decisions.
audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=[
        "postcode_district",
        "vehicle_age",
        "ncd_years",
        "vehicle_group",
    ],
    model_name="Motor Model Q4 2024",
    run_proxy_detection=True,
)

report = audit.run()

# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------
report.summary()

# Write FCA-ready Markdown audit report
report.to_markdown("audit_q4_2024.md")
print("\nAudit report written to: audit_q4_2024.md")
print(f"Overall RAG status: {report.overall_rag.upper()}")
print(f"Flagged proxy factors: {report.flagged_factors or 'none'}")
