# Databricks notebook source
# MAGIC %md
# MAGIC # ausprivauto0405 — Real-Data Gender Fairness Benchmark
# MAGIC
# MAGIC Validates `insurance-fairness` against a real published insurance dataset
# MAGIC with an explicit Gender field.
# MAGIC
# MAGIC **ausprivauto0405** — 67,856 Australian private motor insurance policies
# MAGIC (2004-05), from the CASdatasets R package (Dutang & Charpentier, 2024).
# MAGIC Columns: Exposure, VehValue, VehAge, VehBody, Gender, DrivAge, ClaimOcc,
# MAGIC ClaimNb, ClaimAmount.
# MAGIC
# MAGIC This benchmark:
# MAGIC 1. Downloads the dataset from CASdatasets GitHub
# MAGIC 2. Fits a CatBoost claim frequency model (Gender excluded from features)
# MAGIC 3. Runs FairnessAudit: proxy detection, calibration, demographic parity,
# MAGIC    disparate impact
# MAGIC 4. Runs MulticalibrationAudit: E[Y | mu(X), Gender] calibration check
# MAGIC 5. Runs IndirectDiscriminationAudit: proxy vulnerability per Côté et al. (2025)
# MAGIC
# MAGIC **DISCLAIMER:** ausprivauto0405 is Australian motor data used for methodology
# MAGIC validation only. Not UK market data. Results should not be used as benchmarks
# MAGIC for UK Equality Act 2010 compliance.

# COMMAND ----------

# MAGIC %pip install insurance-fairness rdata requests

# COMMAND ----------

import io
import warnings
import time

import numpy as np
import pandas as pd
import polars as pl
import requests
import rdata as rdata_lib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

from insurance_fairness import (
    FairnessAudit,
    equalised_odds,
    MulticalibrationAudit,
    IndirectDiscriminationAudit,
)

warnings.filterwarnings("ignore")
SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

RDA_URL = (
    "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
    "ausprivauto0405.rda"
)

print(f"Downloading ausprivauto0405 from CASdatasets GitHub...")
resp = requests.get(RDA_URL, timeout=60)
resp.raise_for_status()
raw = resp.content
print(f"Downloaded {len(raw):,} bytes")

parsed = rdata_lib.read_rda(io.BytesIO(raw))
key = next(iter(parsed))
df_raw = parsed[key]
if not isinstance(df_raw, pd.DataFrame):
    df_raw = pd.DataFrame(df_raw)

# Normalise categoricals
for col in df_raw.columns:
    if hasattr(df_raw[col], "cat"):
        df_raw[col] = df_raw[col].astype(str)

print(f"Loaded: {len(df_raw):,} rows × {len(df_raw.columns)} columns")
print(f"Columns: {list(df_raw.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Preparation

# COMMAND ----------

df = df_raw.copy()

for col in ["Exposure", "VehValue", "ClaimNb", "ClaimAmount", "ClaimOcc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Exposure", "ClaimOcc"])
df = df[df["Exposure"] > 0].reset_index(drop=True)

# Standardise Gender values
df["Gender"] = df["Gender"].astype(str).str.strip()
gender_map = {"1": "Male", "2": "Female", "M": "Male", "F": "Female"}
df["Gender"] = df["Gender"].replace(gender_map)

FACTOR_COLS = [c for c in ["VehValue", "VehAge", "VehBody", "DrivAge"] if c in df.columns]
print(f"Policies after cleaning: {len(df):,}")
print(f"Rating factors: {FACTOR_COLS}")
print()

print("Claim frequency by gender:")
for g in sorted(df["Gender"].unique()):
    gm = df["Gender"] == g
    freq = df.loc[gm, "ClaimOcc"].sum() / df.loc[gm, "Exposure"].sum()
    print(f"  {g}: {freq:.4f}  (n={int(gm.sum()):,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit CatBoost Claim Frequency Model

# COMMAND ----------

X = df[FACTOR_COLS + ["Gender"]].copy()
y = df["ClaimOcc"].values.astype(int)
exposure = df["Exposure"].values

X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
    X, y, exposure, test_size=0.25, stratify=y, random_state=SEED
)

cat_features = [
    c for c in FACTOR_COLS
    if df[c].dtype == object or str(df[c].dtype).startswith("category")
]

X_train_nogen = X_train.drop(columns=["Gender"])
X_test_nogen = X_test.drop(columns=["Gender"])

pool_train = Pool(X_train_nogen, label=y_train, cat_features=cat_features, weight=exp_train)
pool_test = Pool(X_test_nogen, label=y_test, cat_features=cat_features, weight=exp_test)

model = CatBoostClassifier(
    iterations=300,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=50,
    allow_writing_files=False,
)

t0 = time.time()
model.fit(pool_train, eval_set=pool_test, early_stopping_rounds=30)
fit_time = time.time() - t0

y_pred_all = model.predict_proba(df[FACTOR_COLS])[:, 1]
y_pred_test = model.predict_proba(X_test_nogen)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_test, sample_weight=exp_test)

print(f"Fit time: {fit_time:.1f}s  |  Test AUC: {auc_test:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. FairnessAudit

# COMMAND ----------

df_audit = pl.DataFrame({
    "gender": df["Gender"].tolist(),
    "predicted_freq": y_pred_all.tolist(),
    "claim_occ": df["ClaimOcc"].tolist(),
    "exposure": df["Exposure"].tolist(),
    **{c: df[c].tolist() for c in FACTOR_COLS},
})

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
    proxy_catboost_iterations=100,
    n_bootstrap=0,
)

t0 = time.time()
report = audit.run()
print(f"Audit completed in {time.time() - t0:.1f}s")

report.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metric Detail

# COMMAND ----------

gr = report.results.get("gender")

if gr and gr.demographic_parity:
    dp = gr.demographic_parity
    print("Demographic Parity")
    print(f"  Group means:  {dp.group_means}")
    print(f"  Log-ratio:    {dp.log_ratio:+.4f}  [{dp.rag.upper()}]")
    print(f"  Ratio:        {dp.ratio:.4f}")
    print()

if gr and gr.calibration:
    cal = gr.calibration
    print("Calibration by Group")
    print(f"  Max A/E disparity: {cal.max_disparity:.4f}  [{cal.rag.upper()}]")
    print()

if gr and gr.disparate_impact:
    di = gr.disparate_impact
    print("Disparate Impact Ratio")
    print(f"  Ratio: {di.ratio:.4f}  [{di.rag.upper()}]")
    print()

if gr and gr.proxy_detection:
    pdr = gr.proxy_detection
    print("Proxy Detection")
    print(f"  {'Factor':<20} {'Proxy R2':>10} {'MI':>10} {'RAG':>6}")
    for s in pdr.scores:
        print(f"  {s.factor:<20} {s.proxy_r2:>10.4f} {s.mutual_information:>10.4f} {s.rag.upper():>6}")
    print()
    print(f"  Flagged: {pdr.flagged_factors or 'none'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Equalised Odds

# COMMAND ----------

threshold = float(np.mean(y_pred_all))
df_eo = df_audit.with_columns(
    pl.when(pl.col("predicted_freq") >= threshold).then(pl.lit(1)).otherwise(pl.lit(0)).alias("predicted_binary"),
    pl.col("claim_occ").cast(pl.Int32),
)

eo = equalised_odds(
    df=df_eo,
    protected_col="gender",
    prediction_col="predicted_binary",
    outcome_col="claim_occ",
    exposure_col="exposure",
)

print(f"TPR disparity: {eo.tpr_disparity:.4f}  [{eo.rag.upper()}]")
print(f"FPR disparity: {eo.fpr_disparity:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Multicalibration Audit

# COMMAND ----------

gender_binary = (df["Gender"] == "Male").astype(int).values

mc = MulticalibrationAudit(n_bins=8, alpha=0.05)
mc_report = mc.audit(
    y_true=df["ClaimOcc"].values.astype(float),
    y_pred=y_pred_all,
    protected=gender_binary,
    exposure=df["Exposure"].values,
)

print(f"Is multicalibrated: {mc_report.is_multicalibrated}")
print(f"Overall calibration p-value: {mc_report.overall_calibration_pvalue:.4f}")
for g_name, pval in mc_report.group_calibration.items():
    g_label = "Male" if str(g_name) == "1" else "Female"
    print(f"  {g_label}: p={pval:.4f}  [{'PASS' if pval > 0.05 else 'FAIL'}]")

if not mc_report.is_multicalibrated:
    worst_pd = mc_report.worst_cells
    if hasattr(worst_pd, "to_pandas"):
        worst_pd = worst_pd.to_pandas()
    print("\nWorst cells:")
    print(worst_pd.head(5).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Indirect Discrimination Audit

# COMMAND ----------

X_full = df[FACTOR_COLS + ["Gender", "Exposure"]].copy()
y_freq = df["ClaimOcc"].values.astype(float)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_freq, test_size=0.25, random_state=SEED, stratify=y_freq.astype(int)
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
elapsed = time.time() - t0
print(f"Completed in {elapsed:.1f}s")
print()
print(f"Portfolio proxy vulnerability: {indirect_result.proxy_vulnerability:.5f}")
print()
print("Segment summary:")
print(indirect_result.summary.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary

# COMMAND ----------

print("=" * 70)
print("BENCHMARK SUMMARY: ausprivauto0405 gender fairness audit")
print("=" * 70)
print()
print(f"Dataset:             ausprivauto0405 (67,856 Australian motor policies)")
print(f"Model:               CatBoost claim frequency (Gender excluded)")
print(f"Test AUC:            {auc_test:.4f}")
print()
print(f"FairnessAudit overall RAG:   {report.overall_rag.upper()}")
if gr:
    if gr.demographic_parity:
        dp = gr.demographic_parity
        print(f"Demographic parity log-ratio:  {dp.log_ratio:+.4f}  [{dp.rag.upper()}]")
    if gr.calibration:
        cal = gr.calibration
        print(f"Max calibration disparity:     {cal.max_disparity:.4f}  [{cal.rag.upper()}]")
    if gr.disparate_impact:
        di = gr.disparate_impact
        print(f"Disparate impact ratio:        {di.ratio:.4f}  [{di.rag.upper()}]")
    if gr.proxy_detection:
        print(f"Proxy factors flagged:         {gr.proxy_detection.flagged_factors or 'none'}")
print()
print(f"Multicalibration:    {'PASS' if mc_report.is_multicalibrated else 'FAIL'}")
print(f"Proxy vulnerability: {indirect_result.proxy_vulnerability:.5f}")
print()
print("DISCLAIMER: Australian motor data — methodology validation only.")
print("Not UK market data.")
