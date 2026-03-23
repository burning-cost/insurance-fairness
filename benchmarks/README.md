# Benchmarks — insurance-fairness

**Headline:** Manual Spearman correlation misses postcode as an ethnicity proxy in 100% of runs across 50 random seeds; the library's CatBoost proxy R² detects it in 100% of runs, with mean proxy R² ≈ 0.62 vs Spearman |r| ≈ 0.10.

---

## Comparison table

20,000 synthetic UK motor policies. Protected attribute: postcode-level ethnic diversity score (continuous, 0–1). Six rating factors including postcode area, vehicle group, NCD, age band, mileage, payment method.

| Metric | Manual Spearman (|r| > 0.25) | Library proxy_r2 + MI |
|---|---|---|
| Postcode flagged as proxy | No (|r| ≈ 0.10, below threshold) | Yes (proxy R² ≈ 0.62, RED) |
| Factors correctly flagged | 0/6 | 1–2/6 |
| Detection rate (50 seeds) | 0/50 (0%) | 50/50 (100%) |
| Mean postcode proxy R² | N/A | ~0.62 (std ≈ 0.02) |
| Mean postcode Spearman |r| | ~0.10 (std ≈ 0.02) | N/A |
| Captures non-linear proxy | No | Yes (CatBoost) |
| Quantifies financial impact | No | Yes (£/policy differential) |
| Computation time | <1s | ~12–15s (80 CatBoost iterations) |

The Spearman check measures rank correlation. Postcode area encodes diversity non-linearly: Inner London postcodes have high diversity; outer postcodes have lower diversity; rural postcodes the lowest. This is a categorical-to-continuous relationship with a non-monotone within-category structure that Spearman cannot detect at the levels seen.

The library fits a CatBoost model predicting the protected attribute from each factor in isolation. Proxy R² of 0.62 means postcode area explains 62% of variance in the diversity score — a finding that survives regulation scrutiny under Equality Act 2010 Section 19.

The financial impact: high-diversity policyholders pay roughly £70–90 more per year than low-diversity policyholders, driven through the postcode area loading. This is the number a pricing committee needs to decide whether to act.

---

## How to run

### Main benchmark (single seed, full SHAP analysis)

```bash
uv run python benchmarks/benchmark.py
```

### Monte Carlo sensitivity (50 seeds, ~10 minutes)

```bash
uv run python benchmarks/benchmark_sensitivity.py
```

### Databricks

```bash
databricks workspace import-dir benchmarks /Workspace/insurance-fairness/benchmarks
# Open benchmark.py as a notebook, attach to serverless compute, run all cells.
```

Dependencies: `insurance-fairness`, `catboost`, `numpy`, `polars`, `scipy`.
