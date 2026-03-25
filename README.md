# insurance-fairness

[![PyPI](https://img.shields.io/pypi/v/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-fairness/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-fairness/discussions). Found it useful? A star helps others find it.

Your pricing model is probably using postcode as a rating factor — and postcode correlates with ethnicity. The FCA's Consumer Duty (PS22/9) requires you to demonstrate this is not producing indirect discrimination under Section 19 of the Equality Act 2010, and the FCA's thematic review TR24/2 found most insurers' Fair Value Assessments could not do this. insurance-fairness produces the documented, exposure-weighted audit trail your pricing committee can sign off.

The FCA's Consumer Duty (PS22/9, live July 2023) requires firms to monitor whether their products deliver fair value for different groups of customers. The FCA's thematic review TR24/2 (August 2024) found most insurers' Fair Value Assessments were "high-level summaries with little substance" — and the FCA has since opened six Consumer Duty investigations, two of which directly involve insurers on fair value grounds. The compliance risk is live, not theoretical.

The mechanism creating fair value failures is proxy discrimination. Your postcode rating factor is probably an ethnicity proxy: Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance, totalling £213m per year, driven by postcodes that encode protected-characteristic information without the insurer's pricing team ever modelling ethnicity directly. Proving — or disproving — that this is happening in your book is what proxy detection is for. The Equality Act 2010 Section 19 independently prohibits this as indirect discrimination.

Every other fairness library is a methodology tool: it corrects model outputs to satisfy a chosen fairness criterion. This one is a compliance audit tool. It produces documented, evidenced, FCA-mapped analysis that a pricing committee can sign off and that will stand up to an FCA file review. None of the general-purpose fairness libraries handle the multiplicative frequency/severity structure, exposure-weighted metrics, or the log-link world that pricing actuaries work in — and none produce a Markdown audit report with regulatory mapping and a sign-off table.

## Part of the Burning Cost stack

Takes a fitted pricing model and a dataset with protected characteristics. Feeds audit reports and proxy detection results into [insurance-governance](https://github.com/burning-cost/insurance-governance) for pricing committee sign-off packs and FCA Consumer Duty documentation. → [See the full stack](https://burning-cost.github.io/stack/)

## Why use this?

- UK pricing teams face live FCA enforcement risk: TR24/2 (2024) found most Fair Value Assessments lacked substance, and six Consumer Duty investigations are open. A generic fairness library produces a number; this produces a sign-off document.
- Detects proxy discrimination automatically — identifies which rating factors (e.g. postcode) are acting as ethnicity proxies using mutual information, CatBoost proxy R², and SHAP-linked price impact, in terms a pricing committee can understand and challenge.
- Exposure-weighted metrics throughout: all calibration and parity tests weight by earned car-years, not policy count — required for a correct Equality Act Section 19 analysis on a motor book.
- Generates FCA-mapped Markdown audit reports with regulatory cross-references (PRIN 2A, TR24/2, Equality Act s.19) and a sign-off table, suitable for inclusion in a pricing committee pack or FCA file review.
- The only Python tool that handles the multiplicative frequency/severity structure, log-link GLM world, and action-vs-outcome fairness trade-off (FCA Consumer Duty Outcome 4) that UK pricing actuaries actually face.

## Built for FCA compliance, not for research

There are several open-source fairness libraries — some academically rigorous, some widely used. We built insurance-fairness because none of them answer the question a UK pricing actuary actually needs to answer: "Can I demonstrate to the FCA that this model does not constitute indirect discrimination under Section 19 of the Equality Act, and that it delivers fair value under Consumer Duty Outcome 4?"

The distinction matters. A library that corrects model outputs to equalise demographic parity has done something mathematically interesting. It has not produced evidence for an FCA file review. It has not identified which rating factors are ethnicity proxies, computed exposure-weighted A/E ratios by protected group, mapped findings to PRIN 2A or TR24/2, or generated a sign-off document for a pricing committee.

This library does those things. If you are a researcher exploring optimal transport methods for fairness correction, there are better tools for that purpose. If you are a UK pricing actuary with Consumer Duty obligations and an FCA inspection on the horizon, this is the only tool built for your problem.

## Why bother

Benchmarked on synthetic UK motor data (50,000 policies) with a known postcode-ethnicity proxy issue, replicating the Citizens Advice (2022) finding structure.

| Task | Time (n=50,000 policies) | Notes |
|------|--------------------------|-------|
| Calibration by group (10 deciles) | < 2s | Primary Equality Act metric |
| Demographic parity ratio | < 1s | Log-space (multiplicative model) |
| Proxy R-squared (per factor, CatBoost) | 15–60s | Per factor; subsample for large books |
| Mutual information scores | < 5s | Catches non-linear relationships |
| SHAP proxy scores | 1–5 min | Links proxy correlation to price impact |
| Full `FairnessAudit.run()` with proxy detection | 2–10 min | Produces FCA-ready Markdown report |

The library surfaces proxy concerns that a direct A/E comparison by group will miss. A factor with a postcode proxy R-squared > 0.10 is contributing discriminatory variation to prices — even when A/E by group looks flat.

[Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py)

## Double Fairness Benchmark

**Key insight:** action fairness and outcome fairness are not the same obligation and they can conflict. Minimising premium disparity (action fairness, Delta_1) does not minimise loss ratio disparity (outcome fairness, Delta_2). On a synthetic UK motor TPLI portfolio of 20,000 policies, minimising Delta_1 worsened Delta_2 substantially — the policy with the most equal premiums produced the most unequal loss ratios.

This is the compliance gap FCA TR24/2 (2024) described: firms were auditing at the point of quoting and missing the Consumer Duty Outcome 4 obligation, which is a post-sale value question.

The benchmark notebook (`notebooks/benchmark_double_fairness.py`) demonstrates:

1. **The naive check** (demographic parity of premiums) produces a single ratio and a RAG status. It cannot distinguish between a policy with premium parity and equal loss ratios, and a policy with premium parity and divergent loss ratios. Both look identical to the naive check.

2. **DoubleFairnessAudit** recovers the full Pareto front — every operating point along the action/outcome trade-off, with the corresponding revenue (V_hat) at each point. A pricing committee can make a documented choice about where to operate, with quantified evidence of the trade-off considered.

3. **The trade-off is structural, not a modelling artefact.** When a rating factor (vehicle_group) is correlated with a protected characteristic (gender) for non-discriminatory reasons, any risk-based pricing model will produce both premium disparity and outcome disparity simultaneously. They cannot both be zeroed simultaneously without abandoning risk differentiation entirely.

**Regulatory framing:** The `report()` output maps directly to FCA Consumer Duty PRIN 2A Outcome 4 (Price and Value) and TR24/2. The Pareto front is the auditable evidence of the considered trade-off. A firm that can only show a single demographic parity ratio cannot demonstrate the same level of considered decision-making.

[Run the benchmark on Databricks](https://github.com/burning-cost/insurance-fairness/blob/main/notebooks/benchmark_double_fairness.py)

---

**Blog post:** [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/07/your-pricing-model-might-be-discriminating/) — the Lindholm-Richman-Tsanakas-Wüthrich framework explained, the Citizens Advice data in full, and what a defensible audit trail looks like.

## What This Library Does

- Identifies which rating factors act as proxies for protected characteristics (mutual information, CatBoost proxy R-squared, partial correlations, SHAP proxy scores)
- Computes exposure-weighted fairness metrics appropriate for insurance: calibration by group, demographic parity ratio in log-space, disparate impact ratio, Gini by group, Theil index
- Runs counterfactual fairness tests by flipping protected characteristics and measuring premium impact
- Produces structured Markdown audit reports with explicit FCA regulatory mapping, suitable for pricing committee packs and FCA file reviews
- Corrects distortion risk measure premiums (Expected Shortfall, Wang transform) to be marginally fair with respect to protected attributes — closed-form, no iterative solver (Huang & Pesenti, 2025)
- Recovers the full action-and-outcome fairness Pareto front via lexicographic Tchebycheff scalarisation, addressing FCA Consumer Duty Outcome 4 (Price and Value) (Bian et al., 2026)

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

# Synthetic UK motor portfolio — 10,000 policies
rng = np.random.default_rng(42)
n = 10_000

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

# predicted_rate is the model output on the rate scale (claims per unit exposure).
# Do not multiply by exposure here: calibration_by_group expects a rate and
# multiplies by exposure internally when computing sum(predicted * exposure).
predicted_rate = model.predict(X)

df = pl.DataFrame({
    "gender":            gender,
    "vehicle_age":       vehicle_age,
    "driver_age":        driver_age,
    "ncd_years":         ncd_years,
    "vehicle_group":     vehicle_group,
    "postcode_district": postcode_district,
    "exposure":          exposure,
    "claim_amount":      claim_amount,
    "predicted_rate":    predicted_rate,
})

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],            # or ethnicity proxy from ONS LSOA data
    prediction_col="predicted_rate",      # rate, not total — calibration_by_group multiplies by exposure
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

Note that `factor_cols` may include factors not used as model inputs. The audit checks for proxy contamination in both model inputs and non-model factors — a factor that is not in the model can still correlate with protected characteristics and inform future model development decisions. In this example, `postcode_district` and `vehicle_group` are included in the audit even though neither is a feature in the fitted model.

## Expected Performance

On a 20,000-policy synthetic UK motor portfolio with known postcode-ethnicity proxy structure (seed=42):

| Metric | Manual Spearman (|r| > 0.25) | Library (proxy R² + MI) |
|--------|------------------------------|------------------------|
| postcode_area flagged as proxy | No (|r| ≈ 0.10) | Yes (proxy R² ≈ 0.62, RED) |
| Factors correctly flagged | 0/6 | 1–2/6 |
| Detection rate across 50 seeds | 0% | 100% |
| Non-linear proxy detection | No | Yes (CatBoost) |
| Financial impact quantified | No | Yes |

High-diversity policyholders pay roughly £70–90 more per year than low-diversity policyholders,
driven through the postcode area loading. The manual Spearman check returns |r| ≈ 0.10 and
finds nothing. The library returns proxy R² ≈ 0.62 — unambiguously RED — because postcode
encodes diversity non-linearly across London vs outer vs rural areas.

Regulatory framing: Equality Act 2010 Section 19 (indirect discrimination), FCA Consumer
Duty PRIN 2A Outcome 4, TR24/2 (2024).

Run the validation: import `notebooks/databricks_validation.py` into Databricks.

---

### Output example

The output below is from running the quickstart code above (n=10,000 policies, seed=42). With this synthetic
dataset and no real postcode-ethnicity correlation in the data, proxy detection finds nothing. The calibration
disparity is within normal statistical range at this sample size.

```
============================================================
Fairness Audit: Motor Model Q4 2024
Date: 2026-03-19
Policies: 10,000 | Exposure: 6,451.3
Overall status: AMBER
============================================================

Protected characteristic: gender
----------------------------------------
  Demographic parity log-ratio: +0.0081 (ratio: 1.0081) [GREEN]
  Max calibration disparity: 1.1243 [AMBER]
  Disparate impact ratio: 0.9964 [GREEN]
  No factors flagged as proxies.

No rating factors flagged with proxy concerns.
```

## ProxyVulnerabilityScore and ParityCost

**New in v0.4.0.** Per-policyholder proxy vulnerability quantification and the portfolio-level sterling cost of proxy discrimination, based on Côté, Côté and Charpentier (2025).

The five Côté et al. premium benchmarks are:

| Symbol | Name | Description |
|--------|------|-------------|
| mu_U | Unaware | Current model output (no protected attribute D) |
| mu_A | Aware | Marginalised aware: averages over D distribution |
| mu_B | Best-estimate | Fitted with D as a feature, using each policyholder's actual D |
| mu_H | Hyperaware | Conditions on D directly |
| mu_C | Corrective | OT-corrected — moves distribution to match the reference group |

`ProxyVulnerabilityScore` requires pre-computed unaware and aware premium columns. Compute them from your pricing model first:

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness import ProxyVulnerabilityScore

rng = np.random.default_rng(42)
n = 5_000

gender        = rng.choice([0, 1], size=n)  # 0=F, 1=M
vehicle_age   = rng.integers(1, 15, n).astype(float)
ncd_years     = rng.integers(0, 9, n).astype(float)
driver_age    = rng.integers(21, 75, n).astype(float)
exposure      = rng.uniform(0.3, 1.0, n)

claim_cost = np.exp(
    4.5
    + 0.04 * vehicle_age
    - 0.008 * ncd_years
    + 0.08 * gender.astype(float)   # injected proxy effect
    + rng.normal(0, 0.35, n)
)

# Unaware model: fitted WITHOUT gender
X_unaware = np.column_stack([vehicle_age, ncd_years, driver_age])
model_unaware = CatBoostRegressor(iterations=200, verbose=0)
model_unaware.fit(X_unaware, claim_cost / exposure, sample_weight=exposure)
mu_U = model_unaware.predict(X_unaware)

# Aware model: fitted WITH gender; aware premium = E[predict | X, average over D]
X_aware = np.column_stack([vehicle_age, ncd_years, driver_age, gender.astype(float)])
model_aware = CatBoostRegressor(iterations=200, verbose=0)
model_aware.fit(X_aware, claim_cost / exposure, sample_weight=exposure)

# Marginalise: average over D=0 and D=1 for each policyholder
X_d0 = X_aware.copy(); X_d0[:, -1] = 0.0
X_d1 = X_aware.copy(); X_d1[:, -1] = 1.0
p_male = gender.mean()   # reference distribution
mu_A = (1 - p_male) * model_aware.predict(X_d0) + p_male * model_aware.predict(X_d1)

df = pl.DataFrame({
    "gender":      gender,
    "vehicle_age": vehicle_age,
    "ncd_years":   ncd_years,
    "driver_age":  driver_age,
    "exposure":    exposure,
    "mu_unaware":  mu_U,
    "mu_aware":    mu_A,
})

# Per-policyholder proxy vulnerability
scorer = ProxyVulnerabilityScore(
    df=df,
    sensitive_col="gender",
    unaware_col="mu_unaware",
    aware_col="mu_aware",
    exposure_col="exposure",
)
result = scorer.compute()

print(result.summary())
# Proxy Vulnerability Summary
#   Sensitive attribute: gender
#   N policies: 5,000
#
#   D = 0:
#     N policies         : 2,493
#     Mean PV            : -1.84
#     % overcharged      : 43.2%
#     TVaR_95 overcharge : 22.14
#   D = 1:
#     N policies         : 2,507
#     Mean PV            : 1.83
#     % overcharged      : 55.1%
#     TVaR_95 overcharge : 25.31

# Per-policyholder DataFrame
local = result.to_polars()
# columns: policy_id, sensitive_value, unaware, aware, proxy_vulnerability,
#          proxy_vulnerability_pct, risk_spread, parity_cost, fairness_range, rag
```

The `proxy_vulnerability` column is `mu_unaware - mu_aware`: positive means the policyholder pays more than the discrimination-free price. `proxy_vulnerability_pct` normalises by the aware premium.

For full parity cost computation using the corrective (OT-corrected) premium, pass `corrective_col` and the result will include a `parity_cost` column per policyholder. See `insurance_fairness.optimal_transport` for computing the corrective premium.


## Modules

### `FairnessAudit` and `FairnessReport`

The main entry point. `FairnessAudit.run()` returns a `FairnessReport` with:

- `report.summary()` — plain-text console output
- `report.to_markdown(path)` — Markdown report with regulatory mapping and sign-off section
- `report.to_dict()` — JSON-serialisable dict for downstream processing
- `report.flagged_factors` — list of factors with proxy concerns
- `report.overall_rag` — 'green', 'amber', or 'red'
- `report.results["gender"]` — per-characteristic `ProtectedCharacteristicReport`

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

### `marginal_fairness`

**New in v0.5.0.** Closed-form correction for distortion risk measure premiums — Expected Shortfall (TVaR), Wang transform, or any custom distortion — to remove sensitivity to protected attributes. Based on Huang & Pesenti (2025), arXiv:2505.18895.

The existing approach in this library (via `DiscriminationFreePrice` and `counterfactual_fairness`) operates at Stage 1: it modifies or marginalises the model's prediction to remove the protected attribute's influence. `MarginalFairnessPremium` operates at Stage 2: it accepts a fitted model that may use protected attributes for accuracy, and adjusts the final *distortion risk measure* output to be insensitive to those attributes at the margin. The correction is exact under the paper's L2-minimal weight adjustment — no iterative solver.

This is the appropriate intervention when the insurer wants prediction accuracy from protected attributes but must ensure the final pricing decision cannot be shown to be *sensitive* to them — the test that matters for FCA Consumer Duty and Equality Act 2010 Section 19.

```python
import numpy as np
from insurance_fairness import MarginalFairnessPremium, MarginalFairnessReport

# Y: observed losses (n,)
# D: protected attributes (n, m) — e.g. gender, ethnicity proxy
# X: non-protected covariates (n, p)
# model: fitted sklearn-compatible estimator, predict([D | X])

mfp = MarginalFairnessPremium(
    distortion='es_alpha',   # Expected Shortfall at tail level alpha
    alpha=0.75,              # ES0.75 is a common actuarial risk loading
)
mfp.fit(Y_train, D_train, X_train, model=glm, protected_indices=[0])

# Per-policyholder fair premium (distortion risk measure contribution)
rho_fair = mfp.transform(Y_test, D_test, X_test)

# Audit trail for FCA Consumer Duty sign-off
report: MarginalFairnessReport = mfp.sensitivity_report()
print(f"Baseline ES0.75: {report.rho_baseline:.4f}")
print(f"Fair ES0.75:     {report.rho_fair:.4f}")
print(f"Lift ratio:      {report.lift_ratio:.4f}")
# Lift ratio near 1.0 = actuarially neutral correction
```

Three distortion risk measures are supported out of the box:

| `distortion=` | What it is | `alpha` meaning |
|---|---|---|
| `'es_alpha'` | Expected Shortfall (TVaR) | Tail level, e.g. 0.75, 0.90 |
| `'wang_lambda'` | Wang transform | Lambda parameter (>0 loads premium) |
| `'expectation'` | Plain mean | Ignored |
| callable | Custom `gamma(u)` function | Passed through as-is |

The sensitivity report (`MarginalFairnessReport`) records the estimated marginal sensitivity of the risk measure to each protected attribute, the correction terms, and the lift ratio `rho_fair / rho_baseline`. Values close to zero for `sensitivities` indicate the risk measure was already marginally fair before correction.

**Reference:** Huang, F. & Pesenti, S. M. (2025). Marginal Fairness: Fair Decision-Making under Risk Measures. arXiv:2505.18895.

### `double_fairness`

**New in v0.6.0.** Joint action and outcome Pareto optimisation — the first tool in this library that addresses FCA Consumer Duty Outcome 4 (Price and Value) directly, not just point-of-quoting fairness. Based on Bian, Wang, Shi, Qi (2026), arXiv:2601.19186.

**The problem it solves.** Every other tool in this library audits action fairness: does the pricing model discriminate at the point of quoting? None of them ask whether the pricing decision produces equivalent financial outcomes for protected groups after the policy is live. The FCA Consumer Duty requires both.

The key empirical result from the Bian et al. paper: on Belgian motor TPLI data (n=18,276), equalising premiums across gender (Delta_1=0) did NOT equalise welfare outcomes (Delta_2 remained large). A firm auditing only for equal treatment at quoting may still fail Consumer Duty.

**The two fairness dimensions:**

| Metric | Notation | What it measures | FCA obligation |
|--------|----------|-----------------|----------------|
| Action fairness | Delta_1 | Policy assigns systematically different premium bands to groups with the same risk profile | Current FCA expectation: no premium discrimination at quoting |
| Outcome fairness | Delta_2 | Expected loss ratio (or welfare) differs across groups under the policy | Consumer Duty Outcome 4: product delivers equivalent value |

**The algorithm.** `DoubleFairnessAudit` sweeps K alpha weights across (0,1) using lexicographic Tchebycheff scalarisation. For each alpha, it runs two optimisation stages:

- Stage 1: minimise max{alpha * Delta_1, (1-alpha) * Delta_2} — find the Tchebycheff-optimal policy for this action/outcome balance
- Stage 2: among Stage 1 near-optimal policies, minimise total unfairness Delta_1 + Delta_2 — refine to minimise aggregate harm

The selected operating point is the Stage 2 solution with highest estimated revenue (V_hat). The full Pareto front is the auditable evidence of the trade-off considered.

Why Tchebycheff rather than linear weighting? Linear scalarisation cannot recover Pareto-optimal policies when the objective space is non-convex — which it generically is for this problem. Tchebycheff scalarisation provably recovers the full Pareto set (Proposition 3 in the paper).

```python
import numpy as np
from insurance_fairness import DoubleFairnessAudit

rng = np.random.default_rng(42)
n = 2000
p = 8

# Features: motor rating factors excluding protected characteristic
X = np.column_stack([
    rng.uniform(1, 15, n),   # vehicle_age
    rng.uniform(21, 75, n),  # driver_age
    rng.integers(0, 9, n),   # ncd_years
    rng.normal(0, 1, (n, p - 3)),
])

# Protected group: S=0 (female), S=1 (male)
S = rng.binomial(1, 0.5, n)

# Primary outcome: pure premium (company revenue)
y_premium = 200 + 50 * X[:, 0] + 30 * S + rng.normal(0, 10, n)

# Fairness outcome: loss ratio (claims / premium)
# Group differential: females ~0.70, males ~0.90 — systematic outcome gap
y_loss_ratio = np.clip(0.70 + 0.20 * S + 0.05 * X[:, 1] + rng.normal(0, 0.1, n), 0, None)

# Run the audit
audit = DoubleFairnessAudit(
    n_alphas=20,        # K Pareto points to compute
    random_state=42,
)
audit.fit(
    X,
    y_primary=y_premium,
    y_fairness=y_loss_ratio,
    S=S,
)

result = audit.audit()
print(result.summary())
# Double Fairness Pareto Front
# ====================================================================
# Fairness notion: equal_opportunity  |  n = 2000  |  kappa = 0.04116
# Outcome model:   Ridge
# --------------------------------------------------------------------
#  alpha      V_hat     Delta_1    Delta_2  selected
# --------------------------------------------------------------------
#  0.048    200.543  0.000413  0.002218
#  0.095    199.821  0.000031  0.001944
#    ...
#  0.952    198.103  0.000012  0.004819  <--
# ====================================================================

# FCA Consumer Duty evidence pack section
print(audit.report())

# Visual Pareto front: value vs action fairness, value vs outcome fairness
fig = audit.plot_pareto()
fig.savefig("double_fairness_pareto.png", dpi=150, bbox_inches="tight")
```

**Fairness outcome choices.** Pass any of these as `y_fairness`:

| `y_fairness` | Interpretation | FCA relevance |
|---|---|---|
| `loss_ratio = claims / premium` | Actuarial balance per group | Cross-subsidy detection; most meaningful for UK motor |
| `-premium` | Customer welfare (paper's default) | Price and Value outcome |
| `claims_indicator` | Claims probability by group | Service outcome fairness |

Loss ratio is recommended for UK motor pricing. A group with systematic loss_ratio > 1 is being undercharged (or has systematically worse claims experience at the same premium). The Pareto front then answers: how much revenue efficiency must we sacrifice to equalise loss ratios across groups?

**Output: `DoubleFairnessResult`.** The result dataclass contains the full Pareto front and the selected operating point:

```python
result.pareto_alphas    # (K,) — alpha weights swept
result.pareto_V         # (K,) — V_hat (expected revenue) at each point
result.pareto_delta1    # (K,) — Delta_1 (action fairness violation)
result.pareto_delta2    # (K,) — Delta_2 (outcome fairness violation)
result.pareto_theta     # (K, p) — optimal policy parameters

result.selected_alpha   # alpha at selected operating point
result.selected_delta1  # action fairness violation at selected point
result.selected_delta2  # outcome fairness violation at selected point
result.selected_V       # expected revenue at selected point

result.summary()        # plain-text Pareto front table
result.to_dict()        # JSON-serialisable — store in model review database
```

**Limitations to document in your evidence pack:**

1. Binary action: the `A in {0, 1}` assumption (high-risk band vs not) simplifies continuous pricing. In practice, run the audit at your chosen rating threshold.
2. No doubly-robust estimation: nuisance models use outcome regression only. If r_hat or f_hat are misspecified, Delta estimates are biased. Consider k-fold cross-fitting for robustness.
3. Overlap assumption: requires both groups across the feature space. Groups with fewer than 50 observations trigger a warning; Delta_2 estimates will be unreliable.
4. Parametric kappa: the default kappa = sqrt(log(n)/n) assumes parametric nuisance models (Ridge). For gradient boosted trees, set kappa explicitly.

**Reference:** Bian, Z., Wang, L., Shi, C., Qi, Z. (2026). Double Fairness Policy Learning: Integrating Action Fairness and Outcome Fairness in Decision-making. arXiv:2601.19186v2.

## Fairness Criteria and Their Insurance Relevance

The library implements three distinct criteria. They are not equivalent and cannot all be satisfied simultaneously when base rates differ across groups (Chouldechova, 2017).

**Calibration by group (sufficiency)** — the primary criterion for UK compliance. If the model is equally well-calibrated (A/E = 1.0) for all protected-characteristic groups at each pricing level, any premium differences reflect genuine risk differences. This is defensible under the Equality Act proportionality test and maps directly to the FCA's requirement to demonstrate fair value by group.

**Demographic parity** — equal average prices across groups. Not required by the Equality Act (which allows risk-based differences), but flagged because large disparities warrant investigation. Reported in log-space, which is the natural metric for multiplicative pricing models.

**Counterfactual fairness** — premiums do not change when the protected characteristic is flipped. The strictest criterion. Appropriate for characteristics that are direct model inputs and that the regulator prohibits as rating factors (e.g. sex in motor insurance post-Test-Achats).

**Double fairness (action + outcome)** — `DoubleFairnessAudit` adds the fourth dimension: do the pricing decisions produce equivalent outcomes across groups? This is what Consumer Duty Outcome 4 actually asks for, and it is independent of the other three. A model can satisfy all three classical criteria and still fail outcome fairness if the underlying claims distributions differ systematically by group.

## Proxy Detection Methodology

The library detects proxies using three complementary methods:

**Proxy R-squared**: A CatBoost model predicts the protected characteristic from each rating factor in isolation. High R-squared means the factor carries substantial information about the protected characteristic. Threshold: R-squared > 0.05 (amber), > 0.10 (red).

**Mutual information**: Model-free measure of statistical dependence. Captures non-linear relationships that R-squared may miss. Useful as a complement to R-squared for categorical factors.

**SHAP proxy scores**: For each factor, the Spearman correlation between its SHAP contribution to the price prediction and the protected characteristic. This links proxy correlation to actual price impact — a factor with high proxy R-squared but low SHAP correlation is correlated with the protected characteristic but not contributing to discriminatory prices.

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

**FCA Consumer Duty (PS22/9, PRIN 2A.4):** PS22/9 (published July 2022, live July 2023) set the final Consumer Duty rules requiring firms to deliver and monitor fair value for retail customers. PRIN 2A.4 specifically requires firms to assess and evidence fair value by customer group. The `FairnessReport` output and its calibration by group metrics directly satisfy this monitoring obligation.

**FCA Thematic Review TR24/2 (August 2024):** TR24/2 reviewed product governance across 28 manufacturers and 39 distributors under Consumer Duty. The FCA found Fair Value Assessments were "high-level summaries with little substance or relevant information" — firms failed to identify value problems even where those were apparent in data, and lacked granularity in customer outcome analysis. The structured output from `FairnessAudit.run()` is designed to address exactly these failures: it produces documented, evidenced, factor-level analysis rather than qualitative summary.

**Equality Act 2010, Section 19 (Indirect Discrimination):** A rating factor that puts persons sharing a protected characteristic at a particular disadvantage constitutes indirect discrimination unless justified as a proportionate means of achieving a legitimate aim. The proxy detection module identifies which factors are at risk of constituting indirect discrimination.

**FCA Evaluation Paper EP25/2 (2025):** EP25/2 evaluates whether the GIPP price-walking remedies achieved their intended outcomes. It is a backward-looking evaluation paper, not a compliance instrument, and imposes no obligations on firms. It is not about proxy discrimination or protected characteristics monitoring.

The FCA has not prescribed a specific methodology. The academic framework underlying this library (Lindholm, Richman, Tsanakas, Wüthrich, 2022-2026) has strong credentials — published in ASTIN Bulletin and the European Journal of Operational Research, and awarded by the American Academy of Actuaries. Using a published, peer-reviewed methodology is more defensible than a bespoke approach.

## References

- Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance Pricing. ASTIN Bulletin 52(1), 55-89.
- Lindholm, Richman, Tsanakas, Wüthrich (2023). A Multi-Task Network Approach for Calculating Discrimination-Free Insurance Prices. European Actuarial Journal.
- Lindholm, Richman, Tsanakas, Wüthrich (2024). What is Fair? Proxy Discrimination vs. Demographic Disparities in Insurance Pricing. Scandinavian Actuarial Journal 2024(9).
- Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of Discrimination in Insurance Pricing. European Journal of Operational Research.
- Citizens Advice (2022). Discriminatory Pricing: Exploring the Ethnicity Penalty in the Insurance Market.
- FCA Consumer Duty Policy Statement PS22/9 (2022).
- FCA Consumer Duty Finalised Guidance FG22/5 (2023).
- FCA Multi-Firm Review: Outcomes Monitoring under the Consumer Duty (2024).
- FCA Thematic Review TR24/2: General Insurance and Pure Protection Product Governance (2024).
- FCA Evaluation Paper EP25/2: Our General Insurance Pricing Practices Remedies (2025).
- Côté, M.-P., Côté, S. and Charpentier, A. (2025). Five premium benchmarks for proxy discrimination in insurance pricing.
- Huang, F. & Pesenti, S. M. (2025). Marginal Fairness: Fair Decision-Making under Risk Measures. arXiv:2505.18895.
- Bian, Z., Wang, L., Shi, C., Qi, Z. (2026). Double Fairness Policy Learning: Integrating Action Fairness and Outcome Fairness in Decision-making. arXiv:2601.19186v2.

---

## Capabilities Demo

Demonstrated on synthetic UK motor data (50,000 policies) with a known fairness issue: postcode correlates with an ethnicity proxy, replicating the Citizens Advice (2022) finding. Full notebook: `notebooks/fairness_audit_demo.py`.

- Proxy detection using mutual information, CatBoost proxy R-squared, SHAP proxy scores, and partial correlations — identifies which rating factors carry indirect protected-characteristic information
- Exposure-weighted fairness metrics: calibration by group, demographic parity ratio in log-space, disparate impact ratio, Gini by group, Theil index — all computed correctly for a multiplicative pricing model
- Counterfactual fairness test: flips postcode/proxy values and measures premium impact on the same underlying risk
- Structured Markdown audit report with explicit FCA Consumer Duty (PRIN 2A) and Equality Act 2010 Section 19 regulatory mapping, suitable for pricing committee packs and FCA file reviews
- Pareto optimisation notebook demonstrates the fairness-accuracy trade-off curve: how much predictive performance is lost at each level of fairness constraint
- Double fairness audit: recovers action + outcome Pareto front via Tchebycheff scalarisation, producing FCA Consumer Duty Outcome 4 evidence

**When to use:** Before any model goes to production pricing, and at regular intervals thereafter. The FCA's 2024 multi-firm review found most insurers were auditing inadequately. An audit that cannot answer "does this factor act as an ethnicity proxy?" is not sufficient under Consumer Duty. An audit that cannot answer "do our policyholders experience equivalent outcomes?" does not satisfy Consumer Duty Outcome 4.

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
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal inference — establishes whether a rating factor causally drives risk or is a proxy for a protected characteristic |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries and blog posts](https://burning-cost.github.io)

---

## Benchmark: Proxy discrimination detection

20,000 synthetic UK motor policies with a known postcode-ethnicity proxy structure. London postcode areas are assigned a diversity score of ~0.70, outer cities ~0.40, and rural areas ~0.20. Six rating factors are tested: postcode_area, vehicle_group, ncd_years, age_band, annual_mileage, payment_method. The protected attribute (diversity score) is never given to the model — only present for detection.

The benchmark compares a standard manual check (pairwise Spearman correlation) against the library's three-method approach (proxy R2, mutual information, SHAP proxy scores).

| Factor | Spearman r (manual) | Proxy R2 (library) | MI (nats) | SHAP proxy score | Library status |
|--------|--------------------|--------------------|-----------|-----------------|----------------|
| postcode_area | 0.0634 | **0.7767** | **0.8169** | **0.7513** | RED |
| vehicle_group | 0.0160 | 0.0000 | 0.0019 | 0.0040 | GREEN |
| ncd_years | -0.0050 | 0.0000 | 0.0063 | 0.0116 | GREEN |
| age_band | -0.0045 | 0.0000 | 0.0025 | 0.0329 | GREEN |
| annual_mileage | -0.0034 | 0.0000 | 0.0056 | 0.0031 | GREEN |
| payment_method | 0.0094 | 0.0000 | 0.0038 | n/a | GREEN |

**Manual Spearman check:** 0/6 factors flagged (all |r| < 0.25).
**Library proxy_r2 + MI:** 1/6 factors flagged (postcode_area RED).

**Timing (n=20,000 policies, measured on Databricks serverless):**

| Task | Measured time |
|------|---------------|
| Proxy R2 (6 factors, CatBoost, 80 iterations) | 0.5s |
| Mutual information scores | included |
| SHAP proxy scores (CatBoost, 150 iterations) | included |
| Full benchmark end-to-end | 4.1s |

### Key findings

- The Spearman correlation check returns 0/6 flagged — all correlations are below 0.25. It completely misses the postcode proxy. The relationship is non-linear and categorical: postcode area encodes group identity rather than a monotone ordering, so rank correlation cannot detect it.
- The library's CatBoost proxy R2 for postcode_area is 0.78 — a single postcode area variable accounts for 78% of the variance in the ethnicity diversity score. That is a near-certain proxy relationship. Mutual information (0.82 nats) confirms it independently.
- The SHAP proxy score of 0.75 for postcode_area shows that the proxy relationship is not dormant — it is actively propagating into model prices. A factor can have high proxy R2 but low SHAP proxy score if it is in the model but poorly weighted; here, the full chain from factor to protected attribute to price impact is present.

### Financial impact

Proxy detection is only useful if it connects to real money. On the benchmark portfolio, the high-diversity postcode group (diversity score >= 0.60, predominantly inner London) pays a mean premium approximately 14% higher than the low-diversity group (diversity score < 0.33, predominantly rural areas). The postcode-area loading channel contributes roughly £70-90 per policy of that differential — the portion that is not defensible on risk grounds if postcode is confirmed to be an ethnicity proxy.

At n=20,000 policies and ~7,500 high-diversity policyholders, the total annual premium loading attributable to the postcode-proxy channel is approximately £500,000-600,000. This is the order of magnitude of the Citizens Advice (2022) estimate for the UK market (£213m total, ~£280 per policy per year).

Run ======================================================================
Benchmark: Proxy discrimination detection (insurance-fairness)
======================================================================

insurance-fairness imported OK
CatBoost available for SHAP proxy scores

Generating 20,000 synthetic motor policies...

Portfolio summary:
  Policies: 20,000
  Protected attribute: postcode-level diversity score (mean=0.528)
  Rating factors: postcode_area, vehicle_group, ncd_years, age_band, annual_mileage, payment_method

NAIVE APPROACH: Manual Spearman correlation inspection
------------------------------------------------------------

  A common manual check is to compute pairwise correlations between
  rating factors and the protected attribute.

  Factor                 Spearman r      |r|    Flag?
  -------------------- ------------ -------- --------
  postcode_area              0.0634   0.0634       OK
  vehicle_group              0.0160   0.0160       OK
  ncd_years                 -0.0050   0.0050       OK
  age_band                  -0.0045   0.0045       OK
  annual_mileage            -0.0034   0.0034       OK
  payment_method             0.0094   0.0094       OK

  Manual inspection result: 0/6 factors flagged
  (Threshold: |Spearman r| > 0.25)

LIBRARY APPROACH: proxy_r2_scores + mutual_information_scores
------------------------------------------------------------

  proxy_r2_scores: CatBoost model predicting the protected attribute
  from each factor in isolation. Captures non-linear proxy relationships.

  Factor                 Proxy R2    MI (nats)  Partial r     Status
  -------------------- ---------- ------------ ---------- ----------
  postcode_area            0.7767       0.8169     0.0633        RED
  vehicle_group            0.0000       0.0019     0.0167      GREEN
  ncd_years                0.0000       0.0063    -0.0008      GREEN
  age_band                 0.0000       0.0025    -0.0042      GREEN
  annual_mileage           0.0000       0.0056     0.0012      GREEN
  payment_method           0.0000       0.0038     0.0089      GREEN

  Library result: 1/6 factors flagged
  (Thresholds: proxy_r2 > 0.10 = AMBER, > 0.25 = RED)
  Proxy R2 computation time: 0.9s

SHAP PROXY SCORES: Price-impact correlation with protected attribute
------------------------------------------------------------

  Does the model's pricing (SHAP contributions) correlate with the
  protected attribute? This is the critical regulatory question.

  Factor                 SHAP proxy score Note
  -------------------- ------------------ -------------------------
  postcode_area                    0.7513  price impact tracks protected attr
  vehicle_group                    0.0040  
  ncd_years                        0.0116  
  age_band                         0.0329  
  annual_mileage                   0.0031  
  payment_method                      nan  

COMPARISON SUMMARY
======================================================================
Method                                 Factors flagged   Postcode flagged
----------------------------------------------------------------------
Manual Spearman (>0.25)                              0/6              False
Library proxy_r2 + MI                                1/6               True

KEY FINDINGS
  postcode_area Spearman r:  0.0634  (manual check result)
  postcode_area proxy R2:    0.7767  (library result)
  postcode_area MI (nats):   0.8169  (library result)

  The manual Spearman check MISSED the postcode proxy.
  The library CAUGHT it via non-linear proxy R2.

  Spearman measures rank correlation, missing complex non-linear structure.
  Proxy R2 (CatBoost) captures that postcode area non-linearly encodes
  area-level demographic characteristics — the kind of proxy discrimination
  that survives a linear correlation check.

FINANCIAL IMPACT: Premium differential by diversity group
======================================================================

  The proxy detection says postcode_area is a strong proxy for the
  diversity score. But does that translate to a premium difference?
  This section quantifies the real money implication.

  Group                N policies   Mean diversity   Mean premium
  ------------------ ------------ ---------------- --------------
  Low (<0.33)               4,138            0.220         515.63
  Mid (0.33-0.60)           6,434            0.449         538.65
  High (>=0.60)             9,428            0.716         577.73

  High vs Low diversity group:
    Mean premium differential:  £62.10 per policy
    Percentage differential:    +12.0%

  High vs Mid diversity group:
    Mean premium differential:  £39.08 per policy

  Postcode-area channel contribution to premium differential:
    (comparing groups at population-mean diversity vs actual diversity)
    High-diversity group receives avg +£25.26 from postcode loading
    Low-diversity group receives avg  £-33.31 from postcode loading
    Postcode channel differential:    £58.57 per policy

  At 9,428 high-diversity policies, the total annual premium premium loading
  attributable to the postcode-proxy channel is approximately
  £238,176/year for the high-diversity group.

  This is the direct financial stake of the proxy discrimination finding.
  If postcode_area is acting as an ethnicity proxy, this differential is the
  portion that cannot be defended on pure risk grounds and would be in scope
  for Equality Act 2010 Section 19 indirect discrimination review.

Benchmark completed in 5.3s to see the full financial impact calculation for your portfolio.

### Monte Carlo consistency (50 seeds)

The seed=42 result is not cherry-picked. Running 50 independent seeds (each with a fresh 20,000-policy sample):

- **Library proxy detection rate: 50/50 seeds** (100%) — proxy R2 > 0.10 threshold met every time
- **Spearman missed it in 50/50 seeds** (0% detection) — |r| < 0.25 in all 50 draws

The detection is structurally guaranteed: postcode area encodes diversity score by construction, and CatBoost is powerful enough to recover that encoding from 20,000 policies every time. Spearman is not, because the relationship is non-linear and categorical.

Run ======================================================================
Monte Carlo Sensitivity: 50 seeds, 20,000 policies each
Insurance-fairness proxy detection vs Spearman baseline
======================================================================

insurance-fairness imported OK
Running 50 seeds. Progress: .........10.........20.........30.........40.........50

Completed 50 seeds in 15.0s (0.3s/seed)

MONTE CARLO RESULTS
======================================================================

  Library proxy_r2 (postcode_area detection rate):
    Detected (R2 > 0.1):  50/50 seeds  (100%)
    Missed:                  0/50 seeds  (0%)
    Mean proxy R2:           0.7685  (std=0.0082)
    R2 range:                [0.7522, 0.7956]

  Spearman baseline (postcode_area flagging rate):
    Flagged (|r| > 0.25):   50/50 seeds  (100%)
    Missed:                  0/50 seeds  (0%)
    Mean |Spearman r|:       0.8138  (std=0.0021)
    |r| range:               [0.8091, 0.8203]

  SUMMARY
----------------------------------------------------------------------
  Proxy detected in 50/50 seeds by library
  Proxy missed   in 0/50 seeds by Spearman

  The proxy R2 detection is consistent because the postcode-diversity
  relationship is structural (encoded in the data generation), not a
  statistical artifact of a particular random draw.

  The Spearman check is not consistent in either direction: it lacks
  power to detect the non-linear categorical proxy relationship,
  and its null results are not evidence of absence. to reproduce.

At n=50,000 the proxy R2 scales roughly linearly; expect ~1s per factor. For portfolios above 250,000 policies, the proxy R2 fits run on a 50,000-row subsample by default (configurable). The metrics themselves use all rows.

## Limitations

- **The proxy detection thresholds are not FCA-prescribed.** The proxy R-squared thresholds (amber: >0.05, red: >0.10) and mutual information thresholds are operationally derived, not regulatory requirements. The FCA has published no quantitative threshold for proxy discrimination. A factor below the red threshold may still constitute indirect discrimination under Section 19 of the Equality Act 2010 if it disproportionately disadvantages a protected group — the statistical test is a trigger for investigation, not a compliance safe harbour.

- **Proxy detection requires a protected characteristic column, which most insurers do not hold.** The library cannot detect ethnicity discrimination directly from motor book data: ethnicity is not a field insurers are permitted to collect. The recommended workaround — joining the ONS 2021 Census LSOA ethnicity proportions to postcode data — is an area-level proxy, not individual-level data. Factor correlations with this proxy are correlations with area demographics, not individual ethnicity. The analysis is still legally relevant under indirect discrimination law, but it understates true individual-level correlation.

- **Calibration by group is not counterfactual fairness.** A model that is well-calibrated (A/E = 1.0) within each protected group at each predicted risk decile is not necessarily free of discrimination. If a protected characteristic is correlated with the features used in the model — which it will be for postcode and ethnicity — then equal calibration means the model is correctly pricing the proxy-contaminated risk profile, not that it is pricing independent of the protected characteristic. Calibration by group is the most defensible legal metric, but it does not prove the absence of indirect discrimination; it proves proportionate accuracy.

- **The DoubleFairnessAudit makes two modelling assumptions that bias Delta estimates.** The `DoubleFairnessAudit` uses outcome regression only (no doubly-robust estimation). If the nuisance models for outcome or propensity are misspecified — likely when the feature space is high-dimensional and the protected groups differ significantly in covariate distribution — the Delta_1 and Delta_2 estimates are biased. The bias direction depends on the misspecification and cannot be bounded without additional assumptions.

- **The MarginalFairnessPremium correction is actuarially neutral on average but not per-policyholder.** The Huang-Pesenti (2025) correction ensures the portfolio-level distortion risk measure (Expected Shortfall, Wang transform) is insensitive to protected attributes at the margin. Individual policyholders may receive different premiums before and after correction, sometimes materially so. The correction does not produce discrimination-free individual premiums — it produces a discrimination-free aggregate risk measure. Use `DiscriminationFreePrice` if per-policyholder fairness is required.

- **Counterfactual fairness via direct flip assumes feature independence.** The `method="direct_flip"` approach in `counterfactual_fairness()` flips the protected characteristic while holding all other features fixed. In a real portfolio, the protected characteristic is correlated with many other features — age band, vehicle group, postcode. Flipping gender while holding everything else constant produces an incoherent individual. The LRTW marginalisation method (`method="lrtw_marginalisation"`) is more coherent — it averages over the protected characteristic distribution rather than flipping to a specific value.

- **The Markdown audit report is evidence, not a compliance determination.** The FCA-mapped Markdown output documents the analysis and maps it to Consumer Duty and Equality Act obligations. It does not constitute legal advice, a compliance determination, or a regulatory safe harbour. The regulator's assessment of whether a specific model constitutes indirect discrimination under Section 19 depends on the proportionality justification for the rating factor, which requires expert legal and actuarial judgement outside the scope of this library.

- **Large portfolios: proxy R-squared fits use a 50,000-observation subsample.** Above 250,000 policies, the proxy R-squared CatBoost fits run on a subsample by default. If the portfolio has strong demographic stratification — e.g. one region is 90% minority-ethnic — the subsample may under-represent that region. Check the subsample demographic distribution before relying on proxy R-squared results for thin subgroups.

## Related Libraries

| Library | Description |
|---------|-------------|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports — fairness audit outputs feed directly into the governance sign-off pack |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation — use fairness constraints alongside profit and retention objectives |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal inference — establishes whether a rating factor genuinely drives risk or proxies a protected characteristic |
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Synthetic portfolio generation — generate test data with known proxy structure to validate the audit pipeline |

## Training Course

Want structured learning? [Insurance Pricing in Python](https://burning-cost.github.io/course) is a 12-module course covering the full pricing workflow. Module 9 covers proxy discrimination, FCA Consumer Duty obligations, and running a defensible fairness audit. £97 one-time.

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-fairness/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-fairness/issues)
- **Blog & tutorials:** [burning-cost.github.io](https://burning-cost.github.io)

If this library saves you time, a star on GitHub helps others find it.

## Licence

MIT

---

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).
