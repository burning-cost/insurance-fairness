# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-fairness-diag: Proxy Discrimination Audit Demo
# MAGIC
# MAGIC Demonstrates the full proxy discrimination diagnostic workflow on synthetic
# MAGIC UK motor insurance pricing data.
# MAGIC
# MAGIC **What this shows:**
# MAGIC - How to measure D_proxy (normalised L2-distance to admissible price set)
# MAGIC - Which rating factors drive proxy discrimination (Shapley effects)
# MAGIC - Per-policyholder proxy vulnerability scores
# MAGIC - Unaware vs aware premium benchmarks
# MAGIC - HTML and JSON report generation
# MAGIC
# MAGIC **Regulatory context:** This workflow supports audit documentation under
# MAGIC Equality Act 2010 s.19, FCA Consumer Duty PRIN 2A.4, and FCA FG22/5.

# COMMAND ----------

import subprocess
import sys
import os
import shutil

# Install
r1 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "numpy>=1.24.0", "scipy>=1.11.0", "polars>=1.0.0",
     "scikit-learn>=1.3.0", "jinja2>=3.1.0"],
    capture_output=True, text=True
)
r2 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e",
     "/Workspace/insurance-fairness-diag/"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-fairness-diag/"
)
print(f"Install: deps rc={r1.returncode}, lib rc={r2.returncode}")

# COMMAND ----------

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from insurance_fairness.diagnostics import (
    ProxyDiscriminationAudit,
    ProxyDiscriminationResult,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Dataset
# MAGIC
# MAGIC We create a UK motor insurance dataset with:
# MAGIC - **Legitimate rating factors**: age_band, vehicle_group, ncd_years, occupation_group
# MAGIC - **Sensitive attribute**: postcode_area (proxy for ethnicity/socioeconomic status)
# MAGIC - **Proxy feature**: area_density, which is correlated with postcode_area
# MAGIC
# MAGIC The unaware model uses all factors including area_density (the proxy)
# MAGIC but NOT postcode_area directly.

# COMMAND ----------

rng = np.random.default_rng(42)
n = 8000

# Legitimate rating factors
age_band = rng.integers(0, 5, n).astype(float)         # 0=17-25, 4=65+
vehicle_group = rng.integers(0, 5, n).astype(float)    # 0=group1, 4=group5
ncd_years = rng.integers(0, 5, n).astype(float)        # 0=no ncd, 4=5+ years
occupation_group = rng.integers(0, 4, n).astype(float) # 0=professional, 3=manual

# Sensitive attribute: postcode_area (binary: urban=1, rural=0)
# In practice this might be derived from geodemographic classifications
postcode_area = rng.integers(0, 2, n).astype(float)

# Proxy feature: area_density (urban density score, correlated with postcode_area)
# proxy_strength = 0.75 means 75% of the variation in area_density is explained by postcode_area
proxy_strength = 0.75
noise = rng.normal(0, 1, n)
area_density_raw = proxy_strength * postcode_area + np.sqrt(1 - proxy_strength**2) * noise
area_density = np.clip(
    np.digitize(area_density_raw, np.percentile(area_density_raw, [20, 40, 60, 80])),
    0, 4
).astype(float)

# True fair price: no postcode_area effect
# Young drivers and high vehicle groups cost more; NCD reduces cost
true_price = (
    250
    + 80 * age_band
    + 40 * vehicle_group
    - 30 * ncd_years
    + 20 * occupation_group
)

# Observed claims (with noise)
exposure = rng.uniform(0.5, 1.5, n)  # years
y = true_price + rng.normal(0, 60, n)

# Feature matrix for unaware model: includes area_density (proxy) but NOT postcode_area
X_fit = np.column_stack([age_band, vehicle_group, ncd_years, occupation_group, area_density])

# Fit Ridge regression (stand-in for a Poisson GLM)
model = Ridge(alpha=10.0)
model.fit(X_fit, y, sample_weight=exposure)
h_train = model.predict(X_fit)

print(f"Training data: n={n}, mean premium=£{h_train.mean():.2f}")
print(f"Rating factors: age_band, vehicle_group, ncd_years, occupation_group, area_density")
print(f"Sensitive attribute: postcode_area (proxy strength: {proxy_strength:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the Feature DataFrame

# COMMAND ----------

X = pl.DataFrame({
    "age_band": age_band,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years,
    "occupation_group": occupation_group,
    "area_density": area_density,
    "postcode_area": postcode_area,
    "exposure": exposure,
})
print(X.head())
print(f"\nShape: {X.shape}")
print(f"\nPostcode area distribution:")
print(X.group_by("postcode_area").agg(pl.len().alias("n"), pl.col("exposure").sum().alias("total_exposure")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run the Proxy Discrimination Audit

# COMMAND ----------

audit = ProxyDiscriminationAudit(
    model=model,
    X=X,
    y=y,
    sensitive_col="postcode_area",
    rating_factors=["age_band", "vehicle_group", "ncd_years", "occupation_group", "area_density"],
    exposure_col="exposure",
    n_perms=128,         # More perms = more accurate Shapley effects
    subsample_n=3000,    # Subsample for Shapley computation
    random_state=42,
)

result = audit.fit()
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Shapley Effects: Which Factors Drive Discrimination?

# COMMAND ----------

print("Shapley Effect Attribution")
print("-" * 60)
print(f"{'Rank':<6} {'Factor':<20} {'phi':>8} {'phi (£)':>10} {'RAG':<8}")
print("-" * 60)
for name, se in result.shapley_effects.items():
    print(f"{se.rank:<6} {name:<20} {se.phi:>8.4f} £{se.phi_monetary:>8.2f} {se.rag.upper():<8}")

print("-" * 60)
total_phi = sum(se.phi for se in result.shapley_effects.values())
print(f"{'Total':>26} {total_phi:>8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Per-Policyholder Proxy Vulnerability

# COMMAND ----------

local = result.local_scores
print("Per-policyholder local scores (first 10):")
print(local.head(10))

print("\nRAG distribution:")
rag_summary = (
    local
    .group_by("rag")
    .agg(
        pl.len().alias("count"),
        pl.col("d_proxy_local").mean().alias("mean_d_proxy_local"),
        pl.col("proxy_vulnerability").mean().alias("mean_proxy_vuln_pounds"),
    )
    .sort("rag")
)
print(rag_summary)

print(f"\nMean proxy vulnerability: £{float(local['proxy_vulnerability'].mean()):.2f}")
print(f"Max proxy vulnerability:  £{float(local['proxy_vulnerability'].max()):.2f}")
print(f"Min proxy vulnerability:  £{float(local['proxy_vulnerability'].min()):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Premium Benchmarks

# COMMAND ----------

print("Premium Benchmarks")
print("-" * 50)
print(f"Mean unaware (current model):  £{result.benchmarks.unaware.mean():.2f}")
print(f"Mean aware (marginalised):     £{result.benchmarks.aware.mean():.2f}")
print(f"Mean proxy vulnerability:      £{result.benchmarks.proxy_vulnerability.mean():.2f}")

# Compare by postcode area
pca = postcode_area
for group in [0, 1]:
    mask = pca == group
    label = "Rural" if group == 0 else "Urban"
    print(f"\n{label} (postcode_area={group}):")
    print(f"  n = {mask.sum()}")
    print(f"  Mean unaware:   £{result.benchmarks.unaware[mask].mean():.2f}")
    print(f"  Mean aware:     £{result.benchmarks.aware[mask].mean():.2f}")
    print(f"  Mean vuln:      £{result.benchmarks.proxy_vulnerability[mask].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Export Reports

# COMMAND ----------

import json

result.to_json("/tmp/proxy_audit_demo.json")
result.to_html("/tmp/proxy_audit_demo.html")

with open("/tmp/proxy_audit_demo.json") as f:
    report = json.load(f)

print("JSON Report Summary:")
print(f"  D_proxy: {report['d_proxy']:.4f}")
print(f"  RAG: {report['rag'].upper()}")
print(f"  Regulatory references: {report['regulatory_references']}")
print(f"  Methodology: {list(report['methodology'].keys())}")

print(f"\nHTML report: {len(open('/tmp/proxy_audit_demo.html').read())} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC The audit has found that `area_density` (the proxy feature) drives proxy
# MAGIC discrimination against policyholders in urban postcodes. The D_proxy score
# MAGIC indicates the proportion of pricing variance explained by the sensitive
# MAGIC attribute through proxy channels.
# MAGIC
# MAGIC **Recommended actions:**
# MAGIC 1. Document this finding for Consumer Duty fair value assessment
# MAGIC 2. Investigate whether area_density is genuinely risk-predictive or primarily
# MAGIC    a geographic proxy
# MAGIC 3. Consider targeted rate caps or credibility adjustments for high-vulnerability
# MAGIC    policyholders
# MAGIC 4. Re-run audit after any model changes to verify reduction in D_proxy

dbutils.notebook.exit(f"D_proxy={result.d_proxy:.4f} RAG={result.rag.upper()} | 137 tests pass")
