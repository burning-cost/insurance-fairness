# insurance-fairness

> FCA Consumer Duty and Equality Act 2010 compliance auditing for UK insurance pricing models — proxy discrimination detection, exposure-weighted bias metrics, and FCA-ready Markdown audit reports.

[![PyPI](https://img.shields.io/pypi/v/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-fairness)](https://pypi.org/project/insurance-fairness/)
[![Tests](https://github.com/burning-cost/insurance-fairness/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-fairness/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-fairness/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-fairness/blob/main/notebooks/quickstart.ipynb)

Your postcode rating factor probably correlates with ethnicity. The FCA's Consumer Duty (PS22/9) requires you to demonstrate this is not producing indirect discrimination under Section 19 of the Equality Act 2010 — and the FCA's multi-firm review of Consumer Duty implementation (2024) found most insurers' Fair Value Assessments could not do this. insurance-fairness produces the documented, exposure-weighted audit trail your pricing committee can sign off.

**Blog post:** [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/07/your-pricing-model-might-be-discriminating/)

---

Every UK pricing team faces the same problem: postcode encodes protected-characteristic information without anyone ever modelling a protected attribute directly. Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance, totalling £213m per year, driven entirely through proxy factors. Proving — or disproving — this is happening in your book is what proxy detection is for.

The compliance risk is live. The FCA's multi-firm review of Consumer Duty implementation (2024) found most Fair Value Assessments were "high-level summaries with little substance". Six Consumer Duty investigations are now open, two of which directly involve insurers on fair value grounds.

This is a compliance audit tool, not a methodology library. It produces documented, evidenced, FCA-mapped analysis that a pricing committee can sign off and that will stand up to an FCA file review. Fairlearn and AIF360 help you satisfy a chosen fairness criterion — they do not answer the question: "Can I demonstrate to the FCA that this model does not constitute indirect discrimination under Section 19 of the Equality Act?"

---

## Part of the Burning Cost toolkit

Takes a fitted pricing model and a dataset with protected characteristics. Feeds audit reports and proxy detection results into [insurance-governance](https://github.com/burning-cost/insurance-governance) for pricing committee sign-off packs and FCA Consumer Duty documentation. → [See the full stack](https://burning-cost.github.io/stack/)

---

## insurance-fairness vs doing it manually

| Task | Manual approach | insurance-fairness |
|------|----------------|--------------------|
| Proxy detection | Spearman correlation — misses non-linear relationships; postcode-ethnicity returns \|r\| ≈ 0.10, finds nothing | CatBoost proxy R² + mutual information; returns R² ≈ 0.62, unambiguously RED — 100% detection rate across 50 seeds |
| Exposure-weighted A/E by group | Custom code per model, often not exposure-weighted | `calibration_by_group()` — exposure-weighted, RAG status, 10-decile grid, Equality Act framing |
| Log-space demographic parity | Multiply predictions, take ratios — silently wrong for multiplicative models | `demographic_parity_ratio()` — log-ratio for GLM/GBM multiplicative worlds |
| Financial impact of discrimination | Not quantified | `ProxyVulnerabilityScore` — per-policyholder £ cost of proxy exploitation |
| Action vs outcome fairness (Consumer Duty Outcome 4) | Not possible with standard metrics | `DoubleFairnessAudit` — full Pareto front, Tchebycheff scalarisation, FCA evidence pack |
| Audit report for pricing committee | Manual Word document, no regulatory mapping | `report.to_markdown()` — FCA-mapped Markdown with PRIN 2A, FG22/5 cross-references and sign-off table |

---

## Quick start

```python
import polars as pl
from insurance_fairness import FairnessAudit

# df: policy-level DataFrame with claims, exposure, rating factors,
# and a protected characteristic column (or ONS-derived area proxy)
audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    model_name="Motor Model Q4 2024",
    run_proxy_detection=True,
)
report = audit.run()
report.summary()
# Overall RAG: RED
# Proxy detection: postcode_district — proxy R²=0.62 [RED], MI=0.41 [RED]
# Calibration by group (gender): max A/E disparity 0.081 [AMBER]
# Demographic parity log-ratio: +0.082 (ratio 1.085) [AMBER]
report.to_markdown("audit_q4_2024.md")  # FCA-ready Markdown with sign-off table
```

See `examples/quickstart.py` for a fully self-contained example including synthetic data generation and model fitting.

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

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-fairness/discussions). Found it useful? A star helps others find it.

---

## Features

- **Proxy detection** — identifies which rating factors (postcode, vehicle group, occupation) are acting as protected-characteristic proxies using mutual information, CatBoost proxy R², partial correlation, and SHAP-linked price impact
- **Exposure-weighted bias metrics** — calibration by group (A/E ratio), demographic parity ratio (log-space), disparate impact ratio, equalised odds, Gini by group, Theil index; all weight by earned car-years, not policy count
- **Counterfactual fairness** — flip protected characteristics and measure premium impact; supports direct flip and LRTW marginalisation
- **Proxy vulnerability scoring** (`ProxyVulnerabilityScore`) — per-policyholder £ difference between unaware premium and discrimination-free benchmark; TVaR overcharge, parity cost, fairness range
- **Indirect discrimination audit** (`IndirectDiscriminationAudit`) — five benchmark premiums from Côté et al. (2025): aware, unaware, unawareness, proxy-free, parity-cost; no causal graph required
- **Multicalibration** (`MulticalibrationAudit`, `IterativeMulticalibrationCorrector`) — audit and iteratively correct pricing models for multicalibration fairness (Denuit, Michaelides & Trufin, 2026)
- **Marginal fairness correction** (`MarginalFairnessPremium`) — closed-form Stage 2 adjustment of Expected Shortfall, Wang transform, or any distortion risk measure to be marginally fair; no iterative solver (Huang & Pesenti, 2025)
- **Double fairness** (`DoubleFairnessAudit`) — joint action (Delta_1) and outcome (Delta_2) Pareto optimisation via lexicographic Tchebycheff scalarisation; addresses Consumer Duty Outcome 4 directly (Bian et al., 2026)
- **Discrimination-insensitive reweighting** (`DiscriminationInsensitiveReweighter`) — training-data weights that achieve X ⊥ A without removing the protected attribute; KL divergence minimisation (Miao & Pesenti, 2026)
- **Privatised audit** (`PrivatizedFairnessAudit`) — fairness auditing when protected attributes are privatised via local differential privacy or estimated from proxies (Zhang, Liu & Shi, 2025)
- **Optimal transport** (`optimal_transport` subpackage) — discrimination-free pricing via Lindholm marginalisation, causal path decomposition, Wasserstein barycenter correction
- **FCA-mapped audit reports** — Markdown with PRIN 2A, FG22/5, Equality Act s.19 cross-references and a sign-off table; suitable for pricing committee packs and FCA file reviews

---


## Expected performance

On a 20,000-policy synthetic UK motor portfolio with a known postcode-ethnicity proxy structure, replicating the Citizens Advice (2022) finding:

| Metric | Manual Spearman (\|r\| > 0.25) | Library (proxy R² + MI) |
|--------|-------------------------------|-------------------------|
| postcode_area flagged as proxy | No (\|r\| ≈ 0.10) | Yes (proxy R² ≈ 0.62, RED) |
| Factors correctly flagged | 0/6 | 1–2/6 |
| Detection rate across 50 seeds | 0% | 100% |
| Non-linear proxy detection | No | Yes (CatBoost) |
| Financial impact quantified | No | Yes |

High-diversity policyholders pay roughly £70–90 more per year than low-diversity policyholders, driven through the postcode area loading. The manual Spearman check returns |r| ≈ 0.10 and finds nothing. The library returns proxy R² ≈ 0.62 — unambiguously RED — because postcode encodes diversity non-linearly across London vs outer vs rural areas.

| Task | Time (n=50,000 policies) | Notes |
|------|--------------------------|-------|
| Calibration by group (10 deciles) | < 2s | Primary Equality Act metric |
| Demographic parity ratio | < 1s | Log-space, multiplicative model |
| Proxy R-squared (per factor, CatBoost) | 15–60s | Per factor; subsample for large books |
| Mutual information scores | < 5s | Catches non-linear relationships |
| Full `FairnessAudit.run()` with proxy detection | 2–10 min | Produces FCA-ready Markdown report |

[Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py)

---

## Modules

### `FairnessAudit` and `FairnessReport`

The main entry point. `FairnessAudit.run()` returns a `FairnessReport` with:

- `report.summary()` — plain-text console output
- `report.to_markdown(path)` — Markdown report with regulatory mapping and sign-off section
- `report.to_dict()` — JSON-serialisable dict for downstream processing
- `report.flagged_factors` — list of factors with proxy concerns
- `report.overall_rag` — `'green'`, `'amber'`, or `'red'`
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

# Calibration by group (sufficiency) — most defensible under Equality Act s.19
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
```

For models that do not use the protected characteristic directly, use `method="lrtw_marginalisation"`.

### `double_fairness`

**v0.6.0.** Action fairness (Delta_1) and outcome fairness (Delta_2) are not the same obligation and they can conflict. Minimising premium disparity does not minimise loss ratio disparity. On a 20,000-policy synthetic UK motor portfolio, the policy with the most equal premiums produced the most unequal loss ratios — a compliance gap the FCA's multi-firm review of Consumer Duty implementation (2024) identified.

`DoubleFairnessAudit` recovers the full Pareto front via lexicographic Tchebycheff scalarisation and selects the value-maximising Pareto point as the recommended operating policy:

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(
    X_train,       # features excluding protected attribute
    y_premium,     # primary outcome: pure premium
    y_loss_ratio,  # fairness outcome: claims / premium
    S_gender,      # binary protected group indicator
)
result = audit.audit()
print(result.summary())
fig = audit.plot_pareto()
print(audit.report())   # FCA Consumer Duty Outcome 4 evidence section
```

[Run the benchmark on Databricks](https://github.com/burning-cost/insurance-fairness/blob/main/notebooks/benchmark_double_fairness.py)

### `marginal_fairness`

**v0.5.0.** Closed-form Stage 2 correction for distortion risk measure premiums — Expected Shortfall, Wang transform, or any custom distortion — to be marginally fair with respect to protected attributes. No iterative solver (Huang & Pesenti, 2025).

```python
from insurance_fairness import MarginalFairnessPremium

mfp = MarginalFairnessPremium(distortion='es_alpha', alpha=0.75)
mfp.fit(Y_train, D_train, X_train, model=glm, protected_indices=[0])
rho_fair = mfp.transform(Y_test, D_test, X_test)

report = mfp.sensitivity_report()
print(f"Baseline ES0.75: {report.rho_baseline:.4f}")
print(f"Fair ES0.75:     {report.rho_fair:.4f}")
```

### `indirect`

**v0.6.4.** End-to-end partition-based audit of indirect discrimination. Computes five benchmark premiums from Côté et al. (2025): aware, unaware, unawareness, proxy-free, and parity-cost. Proxy vulnerability = mean |h_U(x) - h_A(x)| quantifies how much the unaware model exploits proxies. No causal graph required.

```python
from insurance_fairness import IndirectDiscriminationAudit

audit = IndirectDiscriminationAudit(
    protected_attr="gender",
    proxy_features=["postcode_district", "occupation"],
    exposure_col="exposure",
)
result = audit.fit(X_train, y_train, X_test, y_test)
print(f"Proxy vulnerability: {result.proxy_vulnerability:.2f}")
print(result.segment_report)
```

### `multicalibration`

**v0.3.7+.** Audit and correct pricing models for multicalibration fairness — the model should be well-calibrated not just in aggregate, but within every protected-group × risk-decile cell (Denuit, Michaelides & Trufin, 2026). `IterativeMulticalibrationCorrector` (v0.6.7) applies iterative recalibration across cells.

```python
from insurance_fairness import MulticalibrationAudit, IterativeMulticalibrationCorrector

audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
report = audit.audit(y_true, y_pred, protected, exposure)
corrected = audit.correct(y_pred, protected, report, exposure)
```

### `discrimination_insensitive`

**v0.6.3.** Training-data reweighting that achieves X ⊥ A without removing the protected attribute from the training data. Weights integrate with any sklearn `sample_weight` parameter (Miao & Pesenti, 2026).

```python
from insurance_fairness import DiscriminationInsensitiveReweighter

rw = DiscriminationInsensitiveReweighter(protected_col="gender")
weights = rw.fit_transform(X_train)
model.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)
diag = rw.diagnostics(X_train)
```

### `optimal_transport` subpackage

Discrimination-free pricing via Lindholm marginalisation, causal path decomposition, and Wasserstein barycenter correction:

```python
from insurance_fairness.optimal_transport import (
    CausalGraph,
    DiscriminationFreePrice,
    FCAReport,
)
```

### `diagnostics` subpackage

Proxy discrimination diagnostics with D_proxy scalar, Shapley attribution, and per-policyholder vulnerability scores:

```python
from insurance_fairness.diagnostics import ProxyDiscriminationAudit
```

---

## Limitations

- **The proxy detection thresholds are not FCA-prescribed.** Proxy R-squared thresholds (amber: >0.05, red: >0.10) are operationally derived, not regulatory requirements. A factor below the red threshold may still constitute indirect discrimination under Section 19 — the statistical test is a trigger for investigation, not a compliance safe harbour.
- **Proxy detection requires a protected characteristic column, which most insurers do not hold.** Joining ONS 2021 Census LSOA ethnicity proportions to postcode data is an area-level proxy, not individual-level. Factor correlations with this proxy are correlations with area demographics, not individual ethnicity.
- **Calibration by group is not counterfactual fairness.** Equal calibration (A/E = 1.0) within each protected group means the model is correctly pricing the proxy-contaminated risk profile — not that it is pricing independent of the protected characteristic.
- **The Markdown audit report is evidence, not a compliance determination.** It does not constitute legal advice or a regulatory safe harbour. Whether a specific model constitutes indirect discrimination depends on the proportionality justification for the rating factor, which requires expert legal and actuarial judgement outside the scope of this library.

---

## References

**Regulatory instruments**

- FCA. (2023). *Consumer Duty: Final rules and guidance* (PS22/9). Financial Conduct Authority. [www.fca.org.uk/publications/policy-statements/ps22-9-new-consumer-duty](https://www.fca.org.uk/publications/policy-statements/ps22-9-new-consumer-duty)
- FCA. (2023). *Guidance for firms on the fair treatment of vulnerable customers* (FG21/1). Financial Conduct Authority.
- FCA. (2022). *General insurance pricing practices* (PS21/5). Financial Conduct Authority.
- FCA. (2023). *Algorithmic pricing* (DP23/3). Financial Conduct Authority.
- Equality Act 2010, s.19 (indirect discrimination). UK Parliament. [www.legislation.gov.uk/ukpga/2010/15](https://www.legislation.gov.uk/ukpga/2010/15)

**Fairness metrics and methodology**

- Hardt, M., Price, E. & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." *Advances in Neural Information Processing Systems* 29 (NeurIPS 2016). [arXiv:1610.02413](https://arxiv.org/abs/1610.02413)
- Chouldechova, A. (2017). "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments." *Big Data*, 5(2), 153–163. [doi:10.1089/big.2016.0047](https://doi.org/10.1089/big.2016.0047)
- Feldman, M., Friedler, S.A., Moeller, J., Scheidegger, C. & Venkatasubramanian, S. (2015). "Certifying and Removing Disparate Impact." *KDD 2015*. [arXiv:1412.3756](https://arxiv.org/abs/1412.3756)

**Insurance-specific fairness literature**

- Lindholm, M., Richman, R., Tsanakas, A. & Wüthrich, M.V. (2022). "Discrimination-Free Insurance Pricing." *ASTIN Bulletin*, 52(1), 55–89. [doi:10.1017/asb.2021.23](https://doi.org/10.1017/asb.2021.23)
- Lindholm, M., Richman, R., Tsanakas, A. & Wüthrich, M.V. (2026). "Sensitivity-Based Measures of Discrimination in Insurance Pricing." *European Journal of Operational Research*.
- Denuit, M., Michaelides, M. & Trufin, J. (2026). "Multicalibration in Insurance Pricing." [arXiv:2603.16317](https://arxiv.org/abs/2603.16317)
- Zhang, Y., Liu, Y. & Shi, P. (2025). "Discrimination-Free Insurance Pricing with Privatized Sensitive Attributes." [arXiv:2504.11775](https://arxiv.org/abs/2504.11775)
- Côté, O., Côté, M.-P. & Charpentier, A. (2025). "A Scalable Toolbox for Exposing Indirect Discrimination in Insurance Rates." CAS Working Paper.
- Miao, K.E. & Pesenti, S.M. (2026). "Discrimination-Insensitive Pricing." [arXiv:2603.16720](https://arxiv.org/abs/2603.16720)

---

## Related libraries

| Library | What it does |
|---------|-------------|
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID and Doubly Robust Synthetic Controls — establishes whether a rate change caused an outcome shift, not just correlated with one |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Pricing committee sign-off packs and FCA Consumer Duty documentation |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Post-bind drift detection — flags when protected-group A/E ratios are shifting in the live book |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

---

Part of the [Burning Cost](https://burning-cost.github.io) open-source insurance analytics toolkit. → [See all libraries](https://burning-cost.github.io/stack/)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-fairness/discussions). Found it useful? A [GitHub star](https://github.com/burning-cost/insurance-fairness) helps others find it.

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).

## Related Libraries

| Library | Description |
|---------|-------------|
| [`insurance-governance`](https://github.com/burning-cost/insurance-governance) | Model governance and validation reports — document your fairness audit in a PRA-compliant validation framework |
| [`insurance-causal`](https://github.com/burning-cost/insurance-causal) | DML causal inference — isolate genuine causal price effects from proxy correlation before auditing for discrimination |
| [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) | Monitor fairness metrics post-deployment — PSI and A/E ratios on protected characteristic proxies |
