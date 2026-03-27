"""
Benchmark: ausprivauto0405 — real-data fairness audit with an explicit Gender field.

ausprivauto0405 is 67,856 Australian private motor insurance policies from
2004-05, sourced from the CASdatasets R package (Dutang & Charpentier, 2024).
It contains an explicit Gender column, making it one of the few publicly
available insurance datasets suitable for gender fairness methodology validation.

The benchmark fits a CatBoost claim frequency model on rating factors
(VehValue, VehAge, VehBody, DrivAge — Gender excluded), then runs the
library's full fairness audit suite with Gender as the protected characteristic.
It also runs IndirectDiscriminationAudit to measure proxy vulnerability: how
much of Gender's information leaks back into the model via other features?

DISCLAIMER: ausprivauto0405 is Australian motor data used for methodology
validation only. It is not UK market data and does not represent FCA-regulated
insurance products. Results should not be extrapolated to UK pricing practice
or used as benchmarks for UK Equality Act 2010 compliance.

Data source:
    CASdatasets R package — https://github.com/dutangc/CASdatasets
    Dutang, C. & Charpentier, A. (2024). CASdatasets: Insurance Datasets.
    Dataset: ausprivauto0405 (67,856 policies, 9 columns)

Dependencies beyond the library's own requirements:
    pip install rdata requests

Run:
    python benchmarks/ausprivauto_fairness.py

On Databricks (recommended for CI):
    See notebooks/ausprivauto_fairness_databricks.py
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: ausprivauto0405 — real-data gender fairness audit")
print("=" * 70)
print()
print("DISCLAIMER: ausprivauto0405 is Australian motor data used for")
print("methodology validation. Not UK market data.")
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from insurance_fairness import (
        FairnessAudit,
        equalised_odds,
        detect_proxies,
        MulticalibrationAudit,
        IndirectDiscriminationAudit,
    )
    print("insurance-fairness imported OK")
except ImportError as exc:
    print(f"ERROR: Could not import insurance-fairness: {exc}")
    print("Install with: pip install insurance-fairness")
    sys.exit(1)

try:
    import rdata as rdata_lib
    _RDATA_OK = True
except ImportError:
    _RDATA_OK = False
    print("WARNING: `rdata` package not found.")
    print("  Install with: pip install rdata")
    print("  This is required to load the ausprivauto0405.rda file.")

try:
    import numpy as np
    import polars as pl
    import pandas as pd
    import requests
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError as exc:
    print(f"ERROR: Missing dependency: {exc}")
    sys.exit(1)

SEED = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

RDA_URL = (
    "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
    "ausprivauto0405.rda"
)
CACHE_PATH = "/tmp/ausprivauto0405.rda"


def _download_rda(url: str, cache_path: str) -> bytes:
    """Download the .rda file, using a local cache if available."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    print(f"  Downloading from CASdatasets GitHub...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(resp.content)
    print(f"  Downloaded {len(resp.content):,} bytes → cached at {cache_path}")
    return resp.content


def load_ausprivauto0405() -> pd.DataFrame:
    """
    Load ausprivauto0405 from CASdatasets GitHub.

    Returns a pandas DataFrame with columns:
        Exposure, VehValue, VehAge, VehBody, Gender, DrivAge,
        ClaimOcc, ClaimNb, ClaimAmount

    Requires the `rdata` package: pip install rdata
    """
    if not _RDATA_OK:
        raise ImportError(
            "The `rdata` package is required to load the .rda file. "
            "Install with: pip install rdata"
        )

    raw = _download_rda(RDA_URL, CACHE_PATH)

    parsed = rdata_lib.read_rda(io.BytesIO(raw))
    key = next(iter(parsed))
    obj = parsed[key]

    if isinstance(obj, pd.DataFrame):
        df = obj
    else:
        df = pd.DataFrame(obj)

    # rdata returns categorical columns as pandas Categorical; convert to str
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].astype(str)

    return df


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

print("Loading ausprivauto0405...")
print(f"  Source: {RDA_URL}")
print()

try:
    df_raw = load_ausprivauto0405()
except Exception as exc:
    print(f"ERROR loading dataset: {exc}")
    sys.exit(1)

print(f"  Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
print(f"  Columns: {list(df_raw.columns)}")
print()

# ---- Cleaning ----

df = df_raw.copy()

for col in ["Exposure", "VehValue", "ClaimNb", "ClaimAmount", "ClaimOcc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Exposure", "ClaimOcc"])
df = df[df["Exposure"] > 0].reset_index(drop=True)

# Standardise Gender values (R factors can encode as "1"/"2" or "M"/"F")
if "Gender" not in df.columns:
    print("ERROR: Gender column not found. Check the dataset schema.")
    sys.exit(1)

df["Gender"] = df["Gender"].astype(str).str.strip()
gender_map = {"1": "Male", "2": "Female", "M": "Male", "F": "Female"}
df["Gender"] = df["Gender"].replace(gender_map)

FACTOR_COLS = [c for c in ["VehValue", "VehAge", "VehBody", "DrivAge"] if c in df.columns]

print("Data summary")
print("-" * 60)
print(f"  Policies: {len(df):,}")
print(f"  Rating factors: {FACTOR_COLS}")
print(f"  Claim frequency: {df['ClaimOcc'].sum() / df['Exposure'].sum() * 100:.3f}% "
      f"({int(df['ClaimNb'].sum()):,} claims)")
print()
print(f"  {'Gender':<10} {'Policies':>10} {'Exposure':>12} {'Freq %':>8}")
print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
for g in sorted(df["Gender"].unique()):
    gm = df["Gender"] == g
    freq = df.loc[gm, "ClaimOcc"].sum() / df.loc[gm, "Exposure"].sum() * 100
    print(f"  {g:<10} {int(gm.sum()):>10,} {df.loc[gm, 'Exposure'].sum():>12.1f} {freq:>8.3f}")
print()

# ---------------------------------------------------------------------------
# Fit CatBoost claim frequency model (Gender excluded)
# ---------------------------------------------------------------------------

print("Fitting CatBoost claim frequency model")
print("-" * 60)
print("  Target: ClaimOcc (binary claim indicator)")
print(f"  Features: {FACTOR_COLS}  (Gender withheld)")
print()

X = df[FACTOR_COLS + ["Gender"]].copy()
y = df["ClaimOcc"].values.astype(int)
exposure = df["Exposure"].values

X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
    X, y, exposure,
    test_size=0.25,
    stratify=y,
    random_state=SEED,
)

# Identify categorical features
cat_features = [
    c for c in FACTOR_COLS
    if df[c].dtype == object or str(df[c].dtype).startswith("category")
]

X_train_nogen = X_train.drop(columns=["Gender"])
X_test_nogen = X_test.drop(columns=["Gender"])

pool_train = Pool(X_train_nogen, label=y_train, cat_features=cat_features, weight=exp_train)
pool_test = Pool(X_test_nogen, label=y_test, cat_features=cat_features, weight=exp_test)

model = CatBoostClassifier(
    iterations=250,
    depth=4,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False,
)

t0 = time.time()
model.fit(pool_train, eval_set=pool_test, early_stopping_rounds=20)
fit_time = time.time() - t0

y_pred_test = model.predict_proba(X_test_nogen)[:, 1]
y_pred_all = model.predict_proba(df[FACTOR_COLS])[:, 1]

auc_test = roc_auc_score(y_test, y_pred_test, sample_weight=exp_test)
print(f"  Fit time: {fit_time:.1f}s  |  Test AUC: {auc_test:.4f}")
print()

# ---------------------------------------------------------------------------
# Build audit DataFrame
# ---------------------------------------------------------------------------

df_audit = pl.DataFrame({
    "gender": df["Gender"].tolist(),
    "predicted_freq": y_pred_all.tolist(),
    "claim_occ": df["ClaimOcc"].tolist(),
    "exposure": df["Exposure"].tolist(),
    **{c: df[c].tolist() for c in FACTOR_COLS},
})

# ---------------------------------------------------------------------------
# FairnessAudit
# ---------------------------------------------------------------------------

print("FairnessAudit — proxy detection + calibration + parity metrics")
print("-" * 60)
print()

audit = FairnessAudit(
    model=None,
    data=df_audit,
    protected_cols=["gender"],
    prediction_col="predicted_freq",
    outcome_col="claim_occ",
    exposure_col="exposure",
    factor_cols=FACTOR_COLS,
    model_name="CatBoost claim frequency — ausprivauto0405",
    run_proxy_detection=True,
    run_counterfactual=False,
    proxy_catboost_iterations=80,
    n_bootstrap=0,
)

t0 = time.time()
report = audit.run()
audit_time = time.time() - t0
print(f"  Completed in {audit_time:.1f}s")
print()
report.summary()
print()

# ---------------------------------------------------------------------------
# Detailed metric output
# ---------------------------------------------------------------------------

print("DETAILED METRICS")
print("=" * 70)
print()

gr = report.results.get("gender")

if gr and gr.demographic_parity:
    dp = gr.demographic_parity
    print("Demographic Parity")
    print("-" * 40)
    for g, mean_pred in sorted(dp.group_means.items()):
        print(f"  {g:<10}  mean predicted frequency: {mean_pred:.5f}")
    print(f"  Log-ratio:  {dp.log_ratio:+.4f}")
    print(f"  Ratio:      {dp.ratio:.4f}")
    print(f"  RAG:        {dp.rag.upper()}")
    print()

if gr and gr.calibration:
    cal = gr.calibration
    print("Calibration by Group")
    print("-" * 40)
    print(f"  Max A/E disparity across (decile, gender) cells: {cal.max_disparity:.4f}")
    print(f"  RAG:        {cal.rag.upper()}")
    print()

if gr and gr.disparate_impact:
    di = gr.disparate_impact
    print("Disparate Impact Ratio")
    print("-" * 40)
    print(f"  Ratio:      {di.ratio:.4f}")
    print(f"  RAG:        {di.rag.upper()}")
    print()

if gr and gr.proxy_detection:
    pdr = gr.proxy_detection
    print("Proxy Detection — Gender proxy scores per rating factor")
    print("-" * 40)
    print(f"  {'Factor':<20} {'Proxy R2':>10} {'MI (nats)':>12} {'RAG':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*6}")
    for s in pdr.scores:
        print(f"  {s.factor:<20} {s.proxy_r2:>10.4f} {s.mutual_information:>12.4f} "
              f"{s.rag.upper():>6}")
    print()
    if pdr.flagged_factors:
        print(f"  Flagged as gender proxies: {', '.join(pdr.flagged_factors)}")
        print()
        print("  These factors partially predict Gender from the rating factor alone.")
        print("  A flag means the model indirectly routes Gender information through")
        print("  the factor even when Gender itself is excluded from training.")
    else:
        print("  No rating factors flagged as gender proxies.")
    print()

# ---------------------------------------------------------------------------
# Equalised odds
# ---------------------------------------------------------------------------

print("Equalised Odds")
print("-" * 40)

threshold = float(np.mean(y_pred_all))
df_eo = df_audit.with_columns(
    pl.when(pl.col("predicted_freq") >= threshold)
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("predicted_binary"),
    pl.col("claim_occ").cast(pl.Int32),
)

try:
    eo = equalised_odds(
        df=df_eo,
        protected_col="gender",
        prediction_col="predicted_binary",
        outcome_col="claim_occ",
        exposure_col="exposure",
    )
    print(f"  TPR disparity: {eo.tpr_disparity:.4f}")
    print(f"  FPR disparity: {eo.fpr_disparity:.4f}")
    print(f"  RAG:           {eo.rag.upper()}")
except Exception as exc:
    print(f"  Skipped: {exc}")
print()

# ---------------------------------------------------------------------------
# Multicalibration audit
# ---------------------------------------------------------------------------

print("Multicalibration Audit")
print("-" * 40)
print("  Tests E[Y | mu(X)=p, Gender=g] = p for all bins p and groups g.")
print()

try:
    gender_binary = (df["Gender"] == "Male").astype(int).values
    mc = MulticalibrationAudit(n_bins=8, alpha=0.05)
    mc_report = mc.audit(
        y_true=df["ClaimOcc"].values.astype(float),
        y_pred=y_pred_all,
        protected=gender_binary,
        exposure=df["Exposure"].values,
    )
    print(f"  Is multicalibrated:              {mc_report.is_multicalibrated}")
    print(f"  Overall calibration p-value:     {mc_report.overall_calibration_pvalue:.4f}")
    for g_name, pval in mc_report.group_calibration.items():
        g_label = "Male" if str(g_name) == "1" else "Female"
        status = "PASS" if pval > 0.05 else "FAIL"
        print(f"  {g_label} within-bin p-value:        {pval:.4f}  [{status}]")
    if not mc_report.is_multicalibrated:
        worst_pd = (
            mc_report.worst_cells.to_pandas()
            if hasattr(mc_report.worst_cells, "to_pandas")
            else mc_report.worst_cells
        )
        print("  Worst (bin, gender) cells by |AE - 1|:")
        for _, row in worst_pd.head(5).iterrows():
            g_label = "Male" if str(row.get("group", "")) == "1" else "Female"
            print(f"    Bin {row.get('bin', '?')}, {g_label}: "
                  f"AE={row.get('ae_ratio', float('nan')):.3f}, "
                  f"n={int(row.get('n_obs', 0)):,}")
except Exception as exc:
    print(f"  Skipped: {exc}")
print()

# ---------------------------------------------------------------------------
# Indirect discrimination audit
# ---------------------------------------------------------------------------

print("Indirect Discrimination Audit (Côté et al. 2025)")
print("-" * 40)
print("  Fits aware (with Gender) and unaware (without Gender) models.")
print("  Proxy vulnerability = mean |h_U(x) - h_A(x)|: how much does the")
print("  unaware model differ from the aware model at policyholder level?")
print()

try:
    X_full = df[FACTOR_COLS + ["Gender", "Exposure"]].copy()
    y_freq = df["ClaimOcc"].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_freq,
        test_size=0.25,
        random_state=SEED,
        stratify=y_freq.astype(int),
    )

    indirect = IndirectDiscriminationAudit(
        protected_attr="Gender",
        proxy_features=[],
        model_class=GradientBoostingClassifier,
        model_kwargs={"n_estimators": 100, "max_depth": 3, "random_state": SEED},
        exposure_col="Exposure",
        random_state=SEED,
    )

    t0 = time.time()
    indirect_result = indirect.fit(X_tr, y_tr, X_te, y_te)
    print(f"  Completed in {time.time() - t0:.1f}s")
    print()
    print(f"  Portfolio proxy vulnerability: {indirect_result.proxy_vulnerability:.4f}")
    print()
    print(f"  {'Gender':<10} {'N':>7} {'Mean aware':>12} {'Mean unaware':>14} {'Vuln':>8}")
    print(f"  {'-'*10} {'-'*7} {'-'*12} {'-'*14} {'-'*8}")
    for _, row in indirect_result.summary.iterrows():
        print(f"  {str(row['segment']):<10} {int(row['n']):>7,} "
              f"{row['mean_aware']:>12.5f} {row['mean_unaware']:>14.5f} "
              f"{row['mean_proxy_vulnerability']:>8.5f}")
    print()
    print("  A higher proxy vulnerability score means the unaware model has")
    print("  learned to infer Gender from VehAge, VehBody, DrivAge, or VehValue.")
    print("  It is the audit metric most relevant to indirect discrimination")
    print("  under Equality Act 2010 Section 19.")
except Exception as exc:
    print(f"  Skipped: {exc}")
print()

# ---------------------------------------------------------------------------
# Premium differential by gender
# ---------------------------------------------------------------------------

print("Predicted Frequency Differential by Gender")
print("-" * 40)
print()

group_means = {}
for g in sorted(df["Gender"].unique()):
    gm = df["Gender"] == g
    group_means[g] = float(np.average(y_pred_all[gm], weights=df.loc[gm, "Exposure"].values))
    actual = df.loc[gm, "ClaimOcc"].sum() / df.loc[gm, "Exposure"].sum()
    print(f"  {g:<10}  predicted: {group_means[g]:.5f}  actual: {actual:.5f}")

print()

groups_sorted = sorted(group_means.keys())
if len(groups_sorted) == 2:
    g0, g1 = groups_sorted
    m0, m1 = group_means[g0], group_means[g1]
    ratio = m1 / m0 if m0 > 0 else float("nan")
    pct = (ratio - 1) * 100
    higher = g1 if m1 > m0 else g0
    lower = g0 if m1 > m0 else g1
    print(f"  {g1} / {g0} predicted frequency ratio: {ratio:.4f}")
    print(f"  {abs(pct):.1f}% higher predicted frequency for {higher} than {lower}")
    print()
    print("  Note: the model was fitted without Gender, so this differential")
    print("  represents only the variation in VehAge/DrivAge/VehBody/VehValue")
    print("  that correlates with Gender in this dataset.")

print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("BENCHMARK SUMMARY")
print("=" * 70)
print()
print(f"  Dataset:              ausprivauto0405 (CASdatasets)")
print(f"  Policies:             {len(df):,}")
print(f"  Protected attribute:  Gender (Male / Female)")
print(f"  Model:                CatBoost claim frequency (Gender excluded)")
print(f"  Test AUC:             {auc_test:.4f}")
print()
print(f"  Overall audit RAG:    {report.overall_rag.upper()}")
if gr and gr.demographic_parity:
    dp = gr.demographic_parity
    print(f"  Demographic parity log-ratio:  {dp.log_ratio:+.4f}  [{dp.rag.upper()}]")
if gr and gr.calibration:
    cal = gr.calibration
    print(f"  Max calibration disparity:     {cal.max_disparity:.4f}  [{cal.rag.upper()}]")
if gr and gr.disparate_impact:
    di = gr.disparate_impact
    print(f"  Disparate impact ratio:        {di.ratio:.4f}  [{di.rag.upper()}]")
if gr and gr.proxy_detection:
    flagged = gr.proxy_detection.flagged_factors
    print(f"  Proxy factors flagged:         {flagged or 'none'}")
print()
print("  DISCLAIMER: ausprivauto0405 is Australian motor data used for")
print("  methodology validation. Not UK market data.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
