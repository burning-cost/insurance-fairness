# insurance-fairness

FCA Consumer Duty and Equality Act 2010 compliance auditing for UK insurance pricing models — proxy discrimination detection, exposure-weighted bias metrics, and FCA-ready audit reports.

[![PyPI](https://img.shields.io/pypi/v/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Tests](https://github.com/burning-cost/insurance-fairness/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-fairness/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-fairness/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-fairness/blob/main/notebooks/quickstart.ipynb)

**Blog post:** [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/07/your-pricing-model-might-be-discriminating/)

---

Your postcode rating factor probably correlates with ethnicity. The FCA's Consumer Duty (PS22/9) requires you to demonstrate this is not producing indirect discrimination under Section 19 of the Equality Act 2010. The FCA's multi-firm review of Consumer Duty implementation (2024) found most insurers' Fair Value Assessments could not do this. `insurance-fairness` produces the documented, exposure-weighted audit trail your pricing committee can sign off.

This is a compliance audit tool, not a methodology library. It produces evidenced, FCA-mapped analysis that will stand up to an FCA file review. Fairlearn and AIF360 help you satisfy a chosen fairness criterion — they do not answer: "Can I demonstrate to the FCA that this model does not constitute indirect discrimination under Section 19 of the Equality Act?"

---

## Why this library?

| Task | Manual approach | insurance-fairness |
|------|----------------|--------------------|
| Proxy detection | Spearman correlation — misses non-linear relationships; postcode-ethnicity returns \|r\| ≈ 0.10, finds nothing | CatBoost proxy R² + mutual information; returns R² ≈ 0.62, unambiguously RED — 100% detection rate across 50 seeds |
| Exposure-weighted A/E by group | Custom code per model, often not exposure-weighted | `calibration_by_group()` — exposure-weighted, RAG status, 10-decile grid, Equality Act framing |
| Log-space demographic parity | Multiply predictions, take ratios — silently wrong for multiplicative models | `demographic_parity_ratio()` — log-ratio for GLM/GBM multiplicative worlds |
| Financial impact of discrimination | Not quantified | `ProxyVulnerabilityScore` — per-policyholder £ cost of proxy exploitation |
| Action vs outcome fairness (Consumer Duty Outcome 4) | Not possible with standard metrics | `DoubleFairnessAudit` — Pareto front, Tchebycheff scalarisation, FCA evidence pack |
| Audit report for pricing committee | Manual Word document, no regulatory mapping | `report.to_markdown()` — FCA-mapped Markdown with PRIN 2A, FG22/5 cross-references and sign-off table |

---

## Quickstart

```bash
pip install insurance-fairness
```

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness import FairnessAudit

rng = np.random.default_rng(42)
n = 20_000

# Synthetic UK motor book with known proxy structure
postcode_diversity = rng.uniform(0, 1, n)   # ONS LSOA diversity index proxy
gender             = rng.choice(["M", "F"], n)
vehicle_age        = rng.integers(1, 15, n).astype(float)
ncd_years          = rng.integers(0, 9, n).astype(float)

# Postcode diversity drives claims non-linearly — and is correlated with ethnicity
true_rate    = 0.08 + 0.06 * postcode_diversity**2 + 0.01 * vehicle_age - 0.005 * ncd_years
claim_amount = rng.gamma(shape=1.5, scale=true_rate * 300, size=n)
exposure     = rng.uniform(0.5, 1.0, n)

df = pl.DataFrame({
    "claim_amount":        claim_amount,
    "exposure":            exposure,
    "gender":              gender,
    "postcode_diversity":  postcode_diversity,
    "vehicle_age":         vehicle_age,
    "ncd_years":           ncd_years,
})

# Fit a simple pricing model
X = df.select(["postcode_diversity", "vehicle_age", "ncd_years"]).to_numpy()
y = df["claim_amount"].to_numpy()
model = CatBoostRegressor(iterations=100, verbose=0).fit(X, y)
df = df.with_columns(pl.Series("predicted_rate", model.predict(X)))

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_diversity", "vehicle_age", "ncd_years"],
    model_name="Motor Model Q1 2026",
    run_proxy_detection=True,
)
report = audit.run()
report.summary()
report.to_markdown("audit_q1_2026.md")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a fully self-contained example.

---

## Installation

```bash
pip install insurance-fairness
# or
uv add insurance-fairness
```

For the Pareto optimisation subpackage (requires `pymoo`):

```bash
pip install "insurance-fairness[pareto]"
```

**Dependencies:** polars, catboost, scikit-learn, scipy, numpy, jinja2, pyarrow

---

## Features

- **Proxy detection** — identifies which rating factors (postcode, vehicle group, occupation) are acting as protected-characteristic proxies using mutual information, CatBoost proxy R², partial correlation, and SHAP-linked price impact
- **Exposure-weighted bias metrics** — calibration by group (A/E ratio), demographic parity ratio (log-space), disparate impact ratio, equalised odds, Gini by group, Theil index; all weighted by earned car-years, not policy count
- **Counterfactual fairness** — flip protected characteristics and measure premium impact; supports direct flip and LRTW marginalisation
- **Proxy vulnerability scoring** (`ProxyVulnerabilityScore`) — per-policyholder £ difference between unaware premium and discrimination-free benchmark; TVaR overcharge, parity cost, fairness range
- **Indirect discrimination audit** (`IndirectDiscriminationAudit`) — five benchmark premiums from Côté et al. (2025): aware, unaware, unawareness, proxy-free, parity-cost; no causal graph required
- **Multicalibration** (`MulticalibrationAudit`, `IterativeMulticalibrationCorrector`) — audit and iteratively correct pricing models for multicalibration fairness (Denuit, Michaelides & Trufin, 2026)
- **Marginal fairness correction** (`MarginalFairnessPremium`) — closed-form Stage 2 adjustment of Expected Shortfall, Wang transform, or any distortion risk measure; no iterative solver (Huang & Pesenti, 2025)
- **Double fairness** (`DoubleFairnessAudit`) — joint action (Delta_1) and outcome (Delta_2) Pareto optimisation addressing Consumer Duty Outcome 4 (Bian et al., 2026)
- **Discrimination-insensitive reweighting** (`DiscriminationInsensitiveReweighter`) — training-data weights that achieve X ⊥ A without removing the protected attribute; KL divergence minimisation (Miao & Pesenti, 2026)
- **Privatised audit** (`PrivatizedFairnessAudit`) — fairness auditing when protected attributes are estimated from proxies or privatised via local differential privacy (Zhang, Liu & Shi, 2025)
- **Optimal transport** subpackage — discrimination-free pricing via Lindholm marginalisation, causal path decomposition, Wasserstein barycenter correction
- **FCA-mapped audit reports** — Markdown with PRIN 2A, FG22/5, Equality Act s.19 cross-references and a sign-off table; suitable for pricing committee packs and FCA file reviews

---

## Expected performance

On a 20,000-policy synthetic UK motor portfolio with a known postcode-ethnicity proxy structure:

| Metric | Manual Spearman (\|r\| > 0.25) | Library (proxy R² + MI) |
|--------|-------------------------------|-------------------------|
| postcode_area flagged as proxy | No (\|r\| ≈ 0.10) | Yes (proxy R² ≈ 0.62, RED) |
| Detection rate across 50 seeds | 0% | 100% |
| Non-linear proxy detection | No | Yes (CatBoost) |
| Financial impact quantified | No | Yes |

| Task | Time (n=50,000 policies) | Notes |
|------|--------------------------|-------|
| Calibration by group (10 deciles) | < 2s | Primary Equality Act metric |
| Demographic parity ratio | < 1s | Log-space, multiplicative model |
| Proxy R-squared (per factor, CatBoost) | 15–60s | Per factor; subsample for large books |
| Mutual information scores | < 5s | Catches non-linear relationships |
| Full `FairnessAudit.run()` with proxy detection | 2–10 min | Produces FCA-ready Markdown report |

---

## Key modules

### `bias_metrics`

```python
from insurance_fairness import (
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
    gini_by_group,
    theil_index,
)

cal = calibration_by_group(
    df,
    protected_col="ethnicity_group",
    prediction_col="model_freq",
    outcome_col="n_claims",
    exposure_col="exposure",
    n_deciles=10,
)
print(f"Max A/E disparity: {cal.max_disparity:.4f} [{cal.rag}]")

dp = demographic_parity_ratio(df, "gender", "predicted_premium", "exposure")
print(f"Log-ratio: {dp.log_ratio:+.4f} (ratio: {dp.ratio:.4f})")
```

### `proxy_detection`

```python
from insurance_fairness import detect_proxies

result = detect_proxies(
    df,
    protected_col="ethnicity_proxy",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "driver_age_band"],
    run_proxy_r2=True,
    run_mutual_info=True,
    run_partial_corr=True,
)
print(result.flagged_factors)  # ['postcode_district']
print(result.to_polars())      # sorted by proxy R-squared
```

### `double_fairness`

Action fairness (Delta_1) and outcome fairness (Delta_2) can conflict. Minimising premium disparity does not minimise loss ratio disparity. `DoubleFairnessAudit` recovers the full Pareto front:

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(X_train, y_premium, y_loss_ratio, S_gender)
result = audit.audit()
print(result.summary())
print(audit.report())   # FCA Consumer Duty Outcome 4 evidence section
```

---

## Limitations

- **Proxy detection thresholds are not FCA-prescribed.** Proxy R-squared thresholds (amber: >0.05, red: >0.10) are operationally derived. A factor below the red threshold may still constitute indirect discrimination — the statistical test is a trigger for investigation, not a compliance safe harbour.
- **Proxy detection requires a protected characteristic column.** Most insurers do not hold individual ethnicity data. Joining ONS 2021 Census LSOA ethnicity proportions to postcode is an area-level proxy, not individual-level.
- **Calibration by group is not counterfactual fairness.** Equal A/E within each protected group means the model correctly prices the proxy-contaminated risk profile — not that it prices independently of the protected characteristic.
- **The audit report is evidence, not a compliance determination.** It does not constitute legal advice or a regulatory safe harbour.

---

## References

- FCA. (2023). *Consumer Duty: Final rules and guidance* (PS22/9). [fca.org.uk](https://www.fca.org.uk/publications/policy-statements/ps22-9-new-consumer-duty)
- Equality Act 2010, s.19 (indirect discrimination). [legislation.gov.uk](https://www.legislation.gov.uk/ukpga/2010/15)
- Lindholm, M., Richman, R., Tsanakas, A. & Wüthrich, M.V. (2022). "Discrimination-Free Insurance Pricing." *ASTIN Bulletin*, 52(1), 55–89. [doi:10.1017/asb.2021.23](https://doi.org/10.1017/asb.2021.23)
- Côté, O., Côté, M.-P. & Charpentier, A. (2025). "A Scalable Toolbox for Exposing Indirect Discrimination in Insurance Rates." CAS Working Paper.
- Denuit, M., Michaelides, M. & Trufin, J. (2026). "Multicalibration in Insurance Pricing." [arXiv:2603.16317](https://arxiv.org/abs/2603.16317)
- Miao, K.E. & Pesenti, S.M. (2026). "Discrimination-Insensitive Pricing." [arXiv:2603.16720](https://arxiv.org/abs/2603.16720)
- Zhang, Y., Liu, Y. & Shi, P. (2025). "Discrimination-Free Insurance Pricing with Privatized Sensitive Attributes." [arXiv:2504.11775](https://arxiv.org/abs/2504.11775)

---

## Part of the Burning Cost toolkit

Takes a fitted pricing model and a dataset with protected characteristics. Feeds audit reports and proxy detection results into [insurance-governance](https://github.com/burning-cost/insurance-governance) for pricing committee sign-off packs and FCA Consumer Duty documentation. → [See the full stack](https://burning-cost.github.io/stack/)

## Related libraries

| Library | What it does |
|---------|-------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal inference — isolate genuine causal price effects from proxy correlation before auditing for discrimination |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID and Doubly Robust Synthetic Controls — establish whether a rate change caused an outcome shift |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Pricing committee sign-off packs and FCA Consumer Duty documentation |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Post-bind drift detection — flag when protected-group A/E ratios are shifting in the live book |

[All libraries](https://burning-cost.github.io) | [Discussions](https://github.com/burning-cost/insurance-fairness/discussions) | [Issues](https://github.com/burning-cost/insurance-fairness/issues)

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).
