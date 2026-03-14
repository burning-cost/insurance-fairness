# insurance-fairness
[![Tests](https://github.com/burning-cost/insurance-fairness/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-fairness/actions/workflows/tests.yml)
[![CI](https://github.com/burning-cost/insurance-fairness/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-fairness/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-fairness)

Proxy discrimination auditing for UK insurance pricing models.

## The Problem

UK insurers face a genuine compliance obligation to demonstrate their pricing models do not discriminate against customers with protected characteristics. The FCA Consumer Duty (PRIN 2A, live July 2023) requires firms to monitor whether their products provide fair value for different groups of customers, and the FCA's multi-firm review (2024) found most insurers were doing this inadequately. The Equality Act 2010 Section 19 independently prohibits indirect discrimination through rating factors that act as proxies for protected characteristics.

The practical problem is well-documented. Citizens Advice (2022) found a £280/year ethnicity penalty in UK motor insurance in postcodes where more than 50% of residents are people of colour, estimated at £213m per year. The mechanism is straightforward: insurers use postcode as a rating factor; postcode correlates with ethnicity; the postcode effect on price therefore contains an ethnicity component that cannot be justified on pure risk grounds.

Every Python fairness library was built for binary classification or generic regression. None handles the multiplicative frequency/severity structure, exposure-weighted metrics, or the log-link world that pricing actuaries actually work in. This library fills that gap for the UK market.

**Blog post:** [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/07/your-pricing-model-might-be-discriminating/) — the Lindholm-Richman-Tsanakas-Wüthrich framework explained, the Citizens Advice data in full, and what a defensible audit trail looks like.

## What This Library Does

- Identifies which rating factors act as proxies for protected characteristics (mutual information, CatBoost proxy R-squared, partial correlations, SHAP proxy scores)
- Computes exposure-weighted fairness metrics appropriate for insurance: calibration by group, demographic parity ratio in log-space, disparate impact ratio, Gini by group, Theil index
- Runs counterfactual fairness tests by flipping protected characteristics and measuring premium impact
- Produces structured Markdown audit reports with explicit FCA regulatory mapping, suitable for pricing committee packs and FCA file reviews

## Installation

```bash
uv add insurance-fairness
# or
pip install insurance-fairness
```

**Dependencies:** polars, catboost, scikit-learn, scipy, numpy, jinja2, pyarrow

## Quick Start

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness import FairnessAudit

# Synthetic UK motor portfolio — 1,000 policies
rng = np.random.default_rng(42)
n = 1_000

# Protected characteristic: gender (binary)
gender       = rng.choice(["M", "F"], size=n)
vehicle_age  = rng.integers(1, 15, n).astype(float)
driver_age   = rng.integers(21, 75, n).astype(float)
ncd_years    = rng.integers(0, 9, n).astype(float)
vehicle_group = rng.choice(["A", "B", "C", "D"], size=n)
postcode_district = rng.choice(["SW1", "E1", "M1", "B1", "LS1"], size=n)
exposure     = rng.uniform(0.3, 1.0, n)

# Claim amount: slight gender correlation via vehicle_age proxy
claim_amount = np.exp(
    4.5
    + 0.05 * vehicle_age
    - 0.01 * ncd_years
    + 0.08 * (gender == "M").astype(float)  # injected disparity
    + rng.normal(0, 0.4, n)
) * exposure

# Fit a simple CatBoost frequency model inline
feature_cols = ["vehicle_age", "driver_age", "ncd_years"]
X = np.column_stack([vehicle_age, driver_age, ncd_years])
y = claim_amount / exposure

model = CatBoostRegressor(iterations=100, verbose=0)
model.fit(X, y)

predicted_premium = model.predict(X) * exposure

df = pl.DataFrame({
    "gender":            gender,
    "vehicle_age":       vehicle_age,
    "driver_age":        driver_age,
    "ncd_years":         ncd_years,
    "vehicle_group":     vehicle_group,
    "postcode_district": postcode_district,
    "exposure":          exposure,
    "claim_amount":      claim_amount,
    "predicted_premium": predicted_premium,
})

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],            # or ethnicity proxy from ONS LSOA data
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=[
        "postcode_district", "vehicle_age", "ncd_years",
        "vehicle_group",
    ],
    model_name="Motor Model Q4 2024",
    run_proxy_detection=True,
)

report = audit.run()
report.summary()                         # print to console
report.to_markdown("audit_q4_2024.md")  # write FCA-ready report
```

### Output example

```
============================================================
Fairness Audit: Motor Model Q4 2024
Date: 2024-12-01
Policies: 250,000 | Exposure: 187,432.1
Overall status: AMBER
============================================================

Protected characteristic: gender
----------------------------------------
  Demographic parity log-ratio: +0.0821 (ratio: 1.0855) [AMBER]
  Max calibration disparity: 0.0623 [GREEN]
  Disparate impact ratio: 0.9210 [AMBER]
  Flagged proxy factors (2): postcode_district, vehicle_group

Factors with proxy concerns (across all protected characteristics):
  - postcode_district
  - vehicle_group
```

## Modules

### `FairnessAudit` and `FairnessReport`

The main entry point. `FairnessAudit.run()` returns a `FairnessReport` with:

- `report.summary()` - plain-text console output
- `report.to_markdown(path)` - Markdown report with regulatory mapping and sign-off section
- `report.to_dict()` - JSON-serialisable dict for downstream processing
- `report.flagged_factors` - list of factors with proxy concerns
- `report.overall_rag` - 'green', 'amber', or 'red'
- `report.results["gender"]` - per-characteristic `ProtectedCharacteristicReport`

### `bias_metrics`

All metrics are exposure-weighted and work on Polars DataFrames.

```python
from insurance_fairness import (
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
    gini_by_group,
    theil_index,
)

# Calibration by group (sufficiency) - most defensible under Equality Act
cal = calibration_by_group(
    df,
    protected_col="ethnicity_group",
    prediction_col="model_freq",
    outcome_col="n_claims",
    exposure_col="exposure",
    n_deciles=10,
)
print(f"Max A/E disparity: {cal.max_disparity:.4f} [{cal.rag}]")

# Demographic parity ratio (log-space, multiplicative model)
dp = demographic_parity_ratio(df, "gender", "predicted_premium", "exposure")
print(f"Log-ratio: {dp.log_ratio:+.4f} (ratio: {dp.ratio:.4f})")

# Theil index decomposition
theil = theil_index(df, "ethnicity_group", "predicted_premium", "exposure")
print(f"Between-group share: {theil.theil_between / theil.theil_total:.1%}")
```

### `proxy_detection`

```python
from insurance_fairness import mutual_information_scores, proxy_r2_scores, shap_proxy_scores
from insurance_fairness.proxy_detection import detect_proxies

# Combined proxy detection report
result = detect_proxies(
    df,
    protected_col="ethnicity_proxy",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "driver_age_band"],
    run_proxy_r2=True,
    run_mutual_info=True,
    run_partial_corr=True,
)
print(result.flagged_factors)  # ['postcode_district']
print(result.to_polars())      # Polars DataFrame, sorted by proxy R-squared
```

### `counterfactual`

```python
from insurance_fairness import counterfactual_fairness

cf = counterfactual_fairness(
    model=model,
    df=df,
    protected_col="gender",
    feature_cols=["gender", "postcode_district", "vehicle_age", "ncd_years"],
    exposure_col="exposure",
    flip_values={"M": "F", "F": "M"},
    method="direct_flip",
)
cf.summary()
# "Counterfactual premium impact: +8.2%"
# (gender=M policyholders would pay 8.2% less if recorded as gender=F)
```

For models that do not use the protected characteristic directly, use
`method="lrtw_marginalisation"`: predictions are averaged over the marginal
distribution of the protected characteristic, approximating the
Lindholm-Richman-Tsanakas-Wüthrich discrimination-free price.

### `report`

```python
from insurance_fairness.report import generate_markdown_report

md = generate_markdown_report(report)
# Returns a Markdown string with:
# - Executive summary with RAG statuses
# - Per-characteristic metric tables and calibration grids
# - Proxy detection results
# - Regulatory compliance framework mapping
# - Methodology section with academic references
# - Sign-off table for senior actuary attestation
```

## Fairness Criteria and Their Insurance Relevance

The library implements three distinct criteria. They are not equivalent and cannot all be satisfied simultaneously when base rates differ across groups (Chouldechova, 2017).

**Calibration by group (sufficiency)** - the primary criterion for UK compliance. If the model is equally well-calibrated (A/E = 1.0) for all protected-characteristic groups at each pricing level, any premium differences reflect genuine risk differences. This is defensible under the Equality Act proportionality test and maps directly to the FCA's requirement to demonstrate fair value by group.

**Demographic parity** - equal average prices across groups. Not required by the Equality Act (which allows risk-based differences), but flagged because large disparities warrant investigation. Reported in log-space, which is the natural metric for multiplicative pricing models.

**Counterfactual fairness** - premiums do not change when the protected characteristic is flipped. The strictest criterion. Appropriate for characteristics that are direct model inputs and that the regulator prohibits as rating factors (e.g. sex in motor insurance post-Test-Achats).

## Proxy Detection Methodology

The library detects proxies using three complementary methods:

**Proxy R-squared**: A CatBoost model predicts the protected characteristic from each rating factor in isolation. High R-squared means the factor carries substantial information about the protected characteristic. Threshold: R-squared > 0.05 (amber), > 0.10 (red).

**Mutual information**: Model-free measure of statistical dependence. Captures non-linear relationships that R-squared may miss. Useful as a complement to R-squared for categorical factors.

**SHAP proxy scores**: For each factor, the Spearman correlation between its SHAP contribution to the price prediction and the protected characteristic. This links proxy correlation to actual price impact - a factor with high proxy R-squared but low SHAP correlation is correlated with the protected characteristic but not contributing to discriminatory prices.

These thresholds are not prescribed by the FCA. Treat them as triggers for investigation rather than bright-line compliance tests.

## Data Requirements

The protected characteristic column can be:

- Binary (0/1 or string): common for gender, disability indicator
- Multi-category string: e.g. driver age band as a protected characteristic
- Continuous proxy: ONS Census 2021 LSOA ethnicity proportion joined to postcode

For ethnicity, the recommended approach for UK insurers:

1. Download ONS Postcode Directory (ONSPD) from the ONS Geography Portal
2. Download 2021 Census Table TS021 (Ethnic group by LSOA) from NOMIS
3. Join postcode -> LSOA -> ethnicity proportion
4. Use the "% non-white British" at LSOA level as a continuous ethnicity proxy

The library does not bundle this data (it is large and updated quarterly). The join logic is straightforward and can be done in Polars before passing to `FairnessAudit`.

## Regulatory Context

**FCA Consumer Duty (PRIN 2A.4):** Firms must monitor and demonstrate fair value across groups of customers defined by protected characteristics. The `FairnessReport` output and its calibration by group metrics directly satisfy this monitoring requirement.

**Equality Act 2010, Section 19 (Indirect Discrimination):** A rating factor that puts persons sharing a protected characteristic at a particular disadvantage constitutes indirect discrimination unless justified as a proportionate means of achieving a legitimate aim. The proxy detection module identifies which factors are at risk of constituting indirect discrimination.

**FCA Evaluation Paper EP25/2 (2025):** Compliance requires written records demonstrating pricing does not systematically discriminate. The Markdown audit report is designed for inclusion in the pricing committee file and FCA supervisory review.

The FCA has not prescribed a specific methodology. The academic framework underlying this library (Lindholm, Richman, Tsanakas, Wüthrich, 2022-2026) has strong credentials - published in ASTIN Bulletin and the European Journal of Operational Research, and awarded by the American Academy of Actuaries. Using a published, peer-reviewed methodology is more defensible than a bespoke approach.

## Academic References

- Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance Pricing. ASTIN Bulletin 52(1), 55-89.
- Lindholm, Richman, Tsanakas, Wüthrich (2023). A Multi-Task Network Approach for Calculating Discrimination-Free Insurance Prices. European Actuarial Journal.
- Lindholm, Richman, Tsanakas, Wüthrich (2024). What is Fair? Proxy Discrimination vs. Demographic Disparities in Insurance Pricing. Scandinavian Actuarial Journal 2024(9).
- Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of Discrimination in Insurance Pricing. European Journal of Operational Research.
- Citizens Advice (2022). Discriminatory Pricing: Exploring the Ethnicity Penalty in the Insurance Market.
- FCA Consumer Duty Finalised Guidance FG22/5 (2023).
- FCA Multi-Firm Review: Outcomes Monitoring under the Consumer Duty (2024).
- FCA Thematic Review TR24/2: General Insurance and Pure Protection Product Governance (2024).
- FCA Evaluation Paper EP25/2: Our General Insurance Pricing Practices Remedies (2025).

## Running Tests on Databricks

Local test execution will crash a Raspberry Pi or similar low-memory device. Run tests on Databricks:

```python
# In a Databricks notebook:
# %pip install insurance-fairness pytest
# (In Databricks notebooks use %pip; outside Databricks: uv add insurance-fairness)

# !pytest /path/to/insurance_fairness/tests/ -v
```

Or via the Databricks Jobs API. See the `notebooks/fairness_audit_demo.py` for a full workflow demo that runs on Databricks serverless compute.

---

---

## Capabilities Demo

Demonstrated on synthetic UK motor data (50,000 policies) with a known fairness issue: postcode correlates with an ethnicity proxy, replicating the Citizens Advice (2022) finding. Full notebook: `notebooks/fairness_audit_demo.py`.

- Proxy detection using mutual information, CatBoost proxy R-squared, SHAP proxy scores, and partial correlations — identifies which rating factors carry indirect protected-characteristic information
- Exposure-weighted fairness metrics: calibration by group, demographic parity ratio in log-space, disparate impact ratio, Gini by group, Theil index — all computed correctly for a multiplicative pricing model
- Counterfactual fairness test: flips postcode/proxy values and measures premium impact on the same underlying risk
- Structured Markdown audit report with explicit FCA Consumer Duty (PRIN 2A) and Equality Act 2010 Section 19 regulatory mapping, suitable for pricing committee packs and FCA file reviews
- Pareto optimisation notebook demonstrates the fairness-accuracy trade-off curve: how much predictive performance is lost at each level of fairness constraint

**When to use:** Before any model goes to production pricing, and at regular intervals thereafter. The FCA's 2024 multi-firm review found most insurers were auditing inadequately. An audit that cannot answer "does this factor act as an ethnicity proxy?" is not sufficient under Consumer Duty.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py).

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger framework with ENBP audit logging |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal price elasticity and DML — includes elasticity subpackage |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries and blog posts →](https://burning-cost.github.io)

---

## Performance

No formal benchmark yet. Runtime depends primarily on the proxy detection methods enabled.

| Task | Time (n=50,000 policies) |
|------|--------------------------|
| Calibration by group (10 deciles) | < 2s |
| Demographic parity ratio | < 1s |
| Proxy R-squared (per factor, CatBoost) | 15–60s |
| Mutual information scores | < 5s |
| SHAP proxy scores (requires full SHAP run) | 1–5 min |
| Full FairnessAudit.run() with proxy detection | 2–10 min |
| Markdown report generation | < 1s |

For portfolios above 250,000 policies, the proxy R-squared fits run on a 50,000-row subsample by default (configurable). The metrics themselves use all rows.

The library adds value over manual fairness review when the portfolio has multiple protected characteristics, multiple rating factors, and a regulatory requirement for a documented audit trail. For a simple sanity check on two groups with three factors, a direct A/E comparison by group is sufficient and faster.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports — fairness audit outputs feed directly into the governance sign-off pack |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal inference — establishes whether a rating factor causally drives risk or is a proxy for a protected characteristic |
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Synthetic portfolio generation — generate test data with known proxy structure to validate the audit pipeline |

## Licence

MIT
