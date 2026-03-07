"""
conftest.py
-----------
Shared fixtures for insurance-fairness tests.

All synthetic data is generated here. No external datasets are required.
The data is designed to have known statistical properties that allow tests
to verify metric values against analytical expectations.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Synthetic dataset with known properties
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture(scope="session")
def simple_binary_df(rng) -> pl.DataFrame:
    """
    A simple 1,000-policy dataset with a binary protected characteristic
    and known, controlled properties.

    Design:
    - group=0: 500 policies, mean predicted premium ~100, mean actual ~100
    - group=1: 500 policies, mean predicted premium ~130, mean actual ~100
    - This creates a deliberate demographic parity disparity (ratio ~1.3)
      while keeping calibration close to 1.0 within each group.
    - exposure: uniform in [0.5, 1.0]
    - postcode_area: categorical, strongly correlated with group (proxy)
    - vehicle_age: numeric, weakly correlated with group
    """
    n = 1000
    group = np.array([0] * 500 + [1] * 500, dtype=np.int32)

    # Predicted premium: group=1 is 30% higher on average
    base_premium = rng.lognormal(mean=4.6, sigma=0.3, size=n)  # mean ~100
    premium_lift = np.where(group == 1, 1.3, 1.0)
    predicted_premium = base_premium * premium_lift

    # Actual claims: calibrated to predicted within group
    # (A/E ~1.0 for both groups, so calibration is good)
    actual_claims = predicted_premium * rng.lognormal(mean=0.0, sigma=0.2, size=n)

    # Exposure: uniform
    exposure = rng.uniform(0.5, 1.0, size=n)

    # Postcode area: A/B correlated with group=0, C/D correlated with group=1
    postcode_options_0 = ["SW1", "EC1", "W1", "WC1", "N1"]
    postcode_options_1 = ["E1", "E13", "E6", "IG1", "RM1"]
    postcodes = [
        rng.choice(postcode_options_0) if g == 0 else rng.choice(postcode_options_1)
        for g in group
    ]

    # Vehicle age: weakly correlated
    vehicle_age = rng.integers(0, 20, size=n) + group * 2

    # NCD: number of claim-free years, not correlated with group
    ncd = rng.integers(0, 10, size=n)

    return pl.DataFrame({
        "gender": group,
        "predicted_premium": predicted_premium,
        "claim_amount": actual_claims,
        "exposure": exposure,
        "postcode_district": postcodes,
        "vehicle_age": vehicle_age.tolist(),
        "ncd_years": ncd.tolist(),
    })


@pytest.fixture(scope="session")
def multi_group_df(rng) -> pl.DataFrame:
    """
    Dataset with a 3-category protected characteristic.

    Groups: 'A', 'B', 'C' with 300 policies each.
    Mean premiums: A=100, B=115, C=90.
    """
    n_per_group = 300
    groups = ["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group
    means = {"A": 100.0, "B": 115.0, "C": 90.0}

    premiums = []
    actuals = []
    for g in groups:
        m = means[g]
        p = rng.lognormal(mean=np.log(m), sigma=0.25)
        premiums.append(float(p))
        actuals.append(float(p * rng.lognormal(0, 0.2)))

    exposure = rng.uniform(0.5, 1.0, size=len(groups))

    return pl.DataFrame({
        "region": groups,
        "predicted_premium": premiums,
        "claim_amount": actuals,
        "exposure": exposure.tolist(),
        "vehicle_age": rng.integers(1, 15, size=len(groups)).tolist(),
        "ncd_years": rng.integers(0, 10, size=len(groups)).tolist(),
    })


@pytest.fixture(scope="session")
def perfectly_calibrated_df(rng) -> pl.DataFrame:
    """
    Dataset where predictions exactly match actuals on average within each
    group and decile (A/E = 1.0 everywhere). Used to verify calibration
    metrics return expected values.
    """
    n = 2000
    group = np.array([0] * 1000 + [1] * 1000, dtype=np.int32)
    # Different risk levels but equal calibration in both groups
    base = rng.lognormal(mean=4.6, sigma=0.4, size=n)
    exposure = rng.uniform(0.5, 1.0, size=n)
    # Claims = predicted * exposure * random noise (mean 1.0)
    noise = rng.lognormal(0, 0.01, size=n)
    actuals = base * exposure * noise

    return pl.DataFrame({
        "gender": group,
        "predicted_premium": base.tolist(),
        "claim_amount": actuals.tolist(),
        "exposure": exposure.tolist(),
    })


@pytest.fixture(scope="session")
def known_gini_df() -> pl.DataFrame:
    """
    Dataset with analytically known Gini coefficient.

    Premiums: [1, 2, 3, 4] with equal weights -> Gini = 1/3 * (4-1)/(4) ~ 0.25
    Two groups: group 0 has [1, 2], group 1 has [3, 4].
    """
    return pl.DataFrame({
        "group": [0, 0, 1, 1],
        "predicted_premium": [1.0, 2.0, 3.0, 4.0],
        "claim_amount": [1.0, 2.0, 3.0, 4.0],
        "exposure": [1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture(scope="session")
def proxy_test_df(rng) -> pl.DataFrame:
    """
    Dataset where one factor is a strong proxy for the protected characteristic
    and another is not.

    protected: binary gender (0/1)
    strong_proxy: strongly correlated with gender (postcode areas are assigned
        based on gender with high probability)
    weak_proxy: weakly correlated (random assignment)
    unrelated: independent of gender
    """
    n = 800
    gender = rng.integers(0, 2, size=n)

    # Strong proxy: correlated at ~0.8 with gender
    strong_proxy_num = (gender * 0.8 + rng.normal(0, 0.4, n))
    # Bin into categories
    strong_proxy = np.where(strong_proxy_num > 0.5, "High", "Low")

    # Weak proxy: correlated at ~0.2 with gender
    weak_proxy_num = (gender * 0.2 + rng.normal(0, 0.9, n))
    weak_proxy = np.where(weak_proxy_num > 0.5, "High", "Low")

    # Unrelated
    unrelated = rng.integers(1, 6, size=n).astype(float)

    premium = rng.lognormal(4.6, 0.3, n)
    claims = premium * rng.lognormal(0, 0.2, n)
    exposure = rng.uniform(0.5, 1.0, n)

    return pl.DataFrame({
        "gender": gender.tolist(),
        "strong_proxy": strong_proxy.tolist(),
        "weak_proxy": weak_proxy.tolist(),
        "unrelated_factor": unrelated.tolist(),
        "predicted_premium": premium.tolist(),
        "claim_amount": claims.tolist(),
        "exposure": exposure.tolist(),
    })
