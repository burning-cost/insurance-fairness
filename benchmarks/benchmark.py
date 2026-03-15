"""
Benchmark: Proxy discrimination detection — library catches what manual inspection misses.

The scenario: a UK motor pricing model uses no gender or age (protected by Equality
Act 2010 / Ageas v FCA). A manual review of direct correlations between rating
factors and the protected attribute looks clean. But the library detects that
postcode area is a strong proxy for ethnicity (via census-derived deprivation
scores), and that the model's SHAP contributions for postcode are correlated
with the proxy attribute.

This demonstrates the difference between:
  - Manual check: "Is [protected attribute] in the model? No. Are any variables
    strongly correlated with [protected attribute]? Looks OK."
  - Library: CatBoost proxy R-squared + mutual information + partial Spearman
    correlation, with SHAP price-impact correlation for the full chain.

Setup:
- Synthetic motor portfolio with 20,000 policies
- Protected attribute: postcode-level ethnicity diversity score (0-1 continuous)
- Rating factors: vehicle_group, ncd_years, age_band, postcode_area, annual_mileage
- Proxy: postcode_area has high information overlap with the ethnicity score
- The model does not USE the ethnicity score but postcode IS in the model

Expected output:
- Manual Spearman correlation: looks benign for most factors
- Library detects postcode as a high-proxy-R-squared factor (red/amber status)
- SHAP proxy score confirms postcode's price impact correlates with the protected attribute

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: Proxy discrimination detection (insurance-fairness)")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_fairness.proxy_detection import (
        proxy_r2_scores,
        mutual_information_scores,
        partial_correlation,
    )
    print("insurance-fairness imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-fairness: {e}")
    print("Install with: pip install insurance-fairness")
    sys.exit(1)

try:
    from catboost import CatBoostRegressor, Pool
    _CATBOOST_OK = True
    print("CatBoost available for SHAP proxy scores")
except ImportError:
    _CATBOOST_OK = False
    print("CatBoost not available — skipping SHAP proxy scores")

import numpy as np
import polars as pl
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Generate synthetic motor portfolio with embedded proxy
# ---------------------------------------------------------------------------

N_POLICIES = 20_000
SEED = 42
rng = np.random.default_rng(SEED)

print(f"\nGenerating {N_POLICIES:,} synthetic motor policies...")
print()

# Postcode areas (UK motor style)
POSTCODE_AREAS = [
    "E1", "E2", "E3", "N1", "N7", "SE1", "SE5", "SW1", "SW9",
    "W1", "W9", "WC1", "EC1", "BR1", "CR0", "DA1", "HA0", "IG1",
    "KT1", "N15", "NW1", "NW10", "RM1", "SM1", "TW1", "UB1",
    "AL1", "B1", "BN1", "BS1", "CB1", "CF1", "CO1", "CV1",
    "DE1", "DH1", "DL1", "DN1", "DY1", "EX1", "GL1", "GU1",
    "HG1", "HP1", "HR1", "HU1", "HX1", "IP1", "L1", "LA1",
]

n_areas = len(POSTCODE_AREAS)
area_idx = rng.integers(0, n_areas, size=N_POLICIES)
postcode_area = np.array([POSTCODE_AREAS[i] for i in area_idx])

# Protected attribute: postcode-level diversity score (proxy for ethnic diversity)
# Inner London postcodes tend to have higher diversity scores than rural postcodes.
# This is a statistical correlation at area level, not a characteristic of individuals.
london_postcodes = {"E1", "E2", "E3", "N1", "N7", "SE1", "SE5", "SW1", "SW9",
                    "W1", "W9", "WC1", "EC1", "BR1", "CR0", "DA1", "HA0", "IG1",
                    "KT1", "N15", "NW1", "NW10", "RM1", "SM1", "TW1", "UB1"}
outer_postcodes = {"AL1", "B1", "BN1", "BS1", "CB1", "CF1", "CO1", "CV1", "DE1",
                   "DH1", "DL1", "DN1", "DY1", "EX1", "GL1", "GU1"}

# Diversity score: London areas ~0.7, outer cities ~0.4, rural ~0.2
# Plus individual-level noise
base_diversity = np.array([
    0.70 if p in london_postcodes else (0.40 if p in outer_postcodes else 0.20)
    for p in postcode_area
])
diversity_score = np.clip(base_diversity + rng.normal(0, 0.08, N_POLICIES), 0, 1)

# Other risk factors
vehicle_group = rng.choice(["A", "B", "C", "D", "E"], N_POLICIES,
                           p=[0.30, 0.28, 0.22, 0.14, 0.06])
ncd_years = rng.integers(0, 10, N_POLICIES)
age = rng.integers(17, 80, N_POLICIES)
age_band = np.where(age < 25, "17-24",
           np.where(age < 35, "25-34",
           np.where(age < 45, "35-44",
           np.where(age < 55, "45-54",
           np.where(age < 65, "55-64", "65+")))))
annual_mileage = rng.lognormal(9.6, 0.5, N_POLICIES)
payment_method = rng.choice(["direct_debit", "annual"], N_POLICIES, p=[0.65, 0.35])

# Technical premium
# Note: postcode IS a rating factor here (area-based risk loading)
area_loading = base_diversity * 0.25  # London areas: higher premiums (theft, congestion)
age_loading = np.where(age < 25, 0.55, np.where(age > 70, 0.20, 0.0))
vg_loading = {"A": 0.0, "B": 0.12, "C": 0.25, "D": 0.40, "E": 0.60}
ncd_discount = -0.10 * np.minimum(ncd_years, 5)
vg_load_arr = np.array([vg_loading[v] for v in vehicle_group])
mileage_loading = 0.10 * np.log(annual_mileage / 15000)

log_premium = (
    6.2
    + area_loading
    + age_loading
    + vg_load_arr
    + ncd_discount
    + mileage_loading
    + rng.normal(0, 0.08, N_POLICIES)
)
technical_premium = np.exp(log_premium)
exposure = np.ones(N_POLICIES)

df = pl.DataFrame({
    "postcode_area": postcode_area,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years.astype(np.int32),
    "age_band": age_band,
    "annual_mileage": annual_mileage,
    "payment_method": payment_method,
    "diversity_score": diversity_score,
    "technical_premium": technical_premium,
    "exposure": exposure,
})

factor_cols = ["postcode_area", "vehicle_group", "ncd_years", "age_band",
               "annual_mileage", "payment_method"]
protected_col = "diversity_score"

print(f"Portfolio summary:")
print(f"  Policies: {len(df):,}")
print(f"  Protected attribute: postcode-level diversity score (mean={diversity_score.mean():.3f})")
print(f"  Rating factors: {', '.join(factor_cols)}")
print()

# ---------------------------------------------------------------------------
# NAIVE manual check: Spearman correlation
# ---------------------------------------------------------------------------

print("NAIVE APPROACH: Manual Spearman correlation inspection")
print("-" * 60)
print()
print("  A common manual check is to compute pairwise correlations between")
print("  rating factors and the protected attribute.")
print()

# Encode categorical factors for correlation
def _encode_simple(col_name: str) -> np.ndarray:
    s = df[col_name]
    if s.dtype in (pl.String, pl.Utf8, pl.Categorical):
        return s.cast(pl.Categorical).to_physical().to_numpy().astype(float)
    return s.to_numpy().astype(float)

prot_arr = df[protected_col].to_numpy()

print(f"  {'Factor':<20} {'Spearman r':>12} {'|r|':>8} {'Flag?':>8}")
print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8}")
naive_flags = []
for col in factor_cols:
    arr = _encode_simple(col)
    r, p = spearmanr(arr, prot_arr)
    flag = abs(r) > 0.25
    naive_flags.append((col, float(r), flag))
    print(f"  {col:<20} {r:>12.4f} {abs(r):>8.4f} {'FLAG' if flag else 'OK':>8}")

n_naive_flagged = sum(f for _, _, f in naive_flags)
print()
print(f"  Manual inspection result: {n_naive_flagged}/{len(factor_cols)} factors flagged")
print(f"  (Threshold: |Spearman r| > 0.25)")
print()

# ---------------------------------------------------------------------------
# Library: proxy_r2_scores + mutual_information_scores + partial_correlation
# ---------------------------------------------------------------------------

print("LIBRARY APPROACH: proxy_r2_scores + mutual_information_scores")
print("-" * 60)
print()
print("  proxy_r2_scores: CatBoost model predicting the protected attribute")
print("  from each factor in isolation. Captures non-linear proxy relationships.")
print()

t0 = time.time()
r2_scores = proxy_r2_scores(
    df=df,
    protected_col=protected_col,
    factor_cols=factor_cols,
    exposure_col="exposure",
    catboost_iterations=80,
    catboost_depth=4,
    is_binary_protected=False,
    random_seed=SEED,
)
r2_time = time.time() - t0

mi_scores = mutual_information_scores(
    df=df,
    protected_col=protected_col,
    factor_cols=factor_cols,
    is_binary_protected=False,
    random_seed=SEED,
)

pc_scores = partial_correlation(
    df=df,
    protected_col=protected_col,
    factor_cols=factor_cols,
    control_cols=["ncd_years", "annual_mileage"],
)

print(f"  {'Factor':<20} {'Proxy R2':>10} {'MI (nats)':>12} {'Partial r':>10} {'Status':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

lib_flags = []
for col in factor_cols:
    r2 = r2_scores.get(col, float("nan"))
    mi = mi_scores.get(col, float("nan"))
    pc = pc_scores.get(col, float("nan"))
    # Library RAG thresholds (DEFAULT_THRESHOLDS): proxy_r2 > 0.10 = amber, > 0.25 = red
    status = "RED" if r2 > 0.25 else ("AMBER" if r2 > 0.10 else "GREEN")
    flagged = status in ("RED", "AMBER")
    lib_flags.append((col, r2, flagged))
    print(f"  {col:<20} {r2:>10.4f} {mi:>12.4f} {pc:>10.4f} {status:>10}")

n_lib_flagged = sum(f for _, _, f in lib_flags)
print()
print(f"  Library result: {n_lib_flagged}/{len(factor_cols)} factors flagged")
print(f"  (Thresholds: proxy_r2 > 0.10 = AMBER, > 0.25 = RED)")
print(f"  Proxy R2 computation time: {r2_time:.1f}s")
print()

# ---------------------------------------------------------------------------
# SHAP proxy scores (if CatBoost available)
# ---------------------------------------------------------------------------

if _CATBOOST_OK:
    print("SHAP PROXY SCORES: Price-impact correlation with protected attribute")
    print("-" * 60)
    print()
    print("  Does the model's pricing (SHAP contributions) correlate with the")
    print("  protected attribute? This is the critical regulatory question.")
    print()

    # Train a simple premium model
    cat_cols_for_model = ["postcode_area", "vehicle_group", "age_band", "payment_method"]
    X_pd = df.select(factor_cols).to_pandas()

    model = CatBoostRegressor(
        iterations=150,
        depth=4,
        learning_rate=0.05,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    pool = Pool(X_pd, df["technical_premium"].to_numpy(),
                cat_features=cat_cols_for_model)
    model.fit(pool)

    # Get SHAP values
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = shap_vals[:, :len(factor_cols)]  # drop bias column

    from insurance_fairness.proxy_detection import shap_proxy_scores
    shap_scores = shap_proxy_scores(
        df=df,
        protected_col=protected_col,
        factor_cols=factor_cols,
        shap_values=shap_vals,
    )

    print(f"  {'Factor':<20} {'SHAP proxy score':>18} {'Note'}")
    print(f"  {'-'*20} {'-'*18} {'-'*25}")
    for col in factor_cols:
        score = shap_scores.get(col, float("nan"))
        note = "price impact tracks protected attr" if score > 0.3 else ""
        print(f"  {col:<20} {score:>18.4f}  {note}")
    print()

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Method':<35} {'Factors flagged':>18} {'Postcode flagged':>18}")
print("-" * 70)
postcode_naive = next((f for n, _, f in naive_flags if n == "postcode_area"), False)
postcode_lib = next((f for n, _, f in lib_flags if n == "postcode_area"), False)
print(f"{'Manual Spearman (>0.25)':<35} {n_naive_flagged:>18}/{len(factor_cols)} {str(postcode_naive):>18}")
print(f"{'Library proxy_r2 + MI':<35} {n_lib_flagged:>18}/{len(factor_cols)} {str(postcode_lib):>18}")
print()

print("KEY FINDINGS")
postcode_r2 = r2_scores.get("postcode_area", float("nan"))
postcode_mi = mi_scores.get("postcode_area", float("nan"))
postcode_spearman = next((r for n, r, _ in naive_flags if n == "postcode_area"), float("nan"))
print(f"  postcode_area Spearman r:  {postcode_spearman:.4f}  (manual check result)")
print(f"  postcode_area proxy R2:    {postcode_r2:.4f}  (library result)")
print(f"  postcode_area MI (nats):   {postcode_mi:.4f}  (library result)")
print()
print(f"  The manual Spearman check {'MISSED' if not postcode_naive else 'caught'} the postcode proxy.")
print(f"  The library {'CAUGHT' if postcode_lib else 'missed'} it via non-linear proxy R2.")
print()
print("  Spearman measures rank correlation, missing complex non-linear structure.")
print("  Proxy R2 (CatBoost) captures that postcode area non-linearly encodes")
print("  area-level demographic characteristics — the kind of proxy discrimination")
print("  that survives a linear correlation check.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
