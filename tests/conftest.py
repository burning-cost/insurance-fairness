"""
conftest.py
-----------
Shared fixtures for insurance-fairness tests.

All synthetic data is generated here. No external datasets are required.
The data is designed to have known statistical properties that allow tests
to verify metric values against analytical expectations.

Also includes fixtures and helpers for the diagnostics subpackage tests
(originally from insurance-fairness-diag).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Synthetic dataset with known properties (core fairness tests)
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


# ---------------------------------------------------------------------------
# Fixtures and helpers for diagnostics subpackage tests
# (absorbed from insurance-fairness-diag)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n: int = 2000,
    proxy_strength: float = 0.3,
    random_state: int = 42,
) -> tuple[pl.DataFrame, np.ndarray, object]:
    """
    Generate a synthetic insurance pricing dataset with known proxy structure.

    The dataset has:
      - age_band: legitimate rating factor (0..4)
      - vehicle_group: legitimate rating factor (0..4)
      - ncd_years: legitimate rating factor (0..4)
      - postcode_area: SENSITIVE attribute (binary: 0 or 1)
      - proxy_feature: feature that is correlated with postcode_area
        by *proxy_strength* (this creates proxy discrimination)

    The true price is:
      true_price = 200 + 50*age_band + 30*vehicle_group + 20*ncd_years

    The model is fitted WITHOUT postcode_area, but WITH proxy_feature,
    which introduces proxy discrimination.

    Parameters
    ----------
    n:
        Number of policyholders.
    proxy_strength:
        Correlation between proxy_feature and postcode_area (0 = no proxy).
    random_state:
        Random seed.

    Returns
    -------
    (X, h, model) where:
      X is the Polars feature DataFrame (includes all columns)
      h is the fitted premium array
      model is the fitted sklearn Ridge model
    """
    rng_d = np.random.default_rng(random_state)

    # Sensitive attribute: binary postcode_area (0 = North, 1 = South)
    postcode = rng_d.integers(0, 2, size=n).astype(float)

    # Legitimate factors
    age_band = rng_d.integers(0, 5, size=n).astype(float)
    vehicle_group = rng_d.integers(0, 5, size=n).astype(float)
    ncd_years = rng_d.integers(0, 5, size=n).astype(float)

    # Proxy feature: correlated with postcode but not identical
    noise = rng_d.normal(0, 1, size=n)
    proxy_feature = proxy_strength * postcode + np.sqrt(1 - proxy_strength**2) * noise
    # Discretise to 5 bands
    proxy_feature = np.clip(
        np.digitize(proxy_feature, np.percentile(proxy_feature, [20, 40, 60, 80])),
        0, 4
    ).astype(float)

    # True price (no proxy discrimination)
    true_price = 200.0 + 50.0 * age_band + 30.0 * vehicle_group + 20.0 * ncd_years

    # Observed claims with noise
    y = true_price + rng_d.normal(0, 50, size=n)

    # Feature matrix for model: does NOT include postcode_area, but includes proxy_feature
    X_model = np.column_stack([age_band, vehicle_group, ncd_years, proxy_feature])

    # Fit a Ridge model
    model = Ridge(alpha=1.0)
    model.fit(X_model, y)
    h = model.predict(X_model)

    # Build Polars DataFrame with ALL columns (including sensitive)
    X = pl.DataFrame({
        "age_band": age_band,
        "vehicle_group": vehicle_group,
        "ncd_years": ncd_years,
        "proxy_feature": proxy_feature,
        "postcode_area": postcode,
    })

    return X, h, model


def make_zero_proxy_dataset(
    n: int = 1000,
    random_state: int = 42,
) -> tuple[pl.DataFrame, np.ndarray, object]:
    """
    Generate a dataset where the model has NO proxy discrimination.

    The proxy_feature is independent of postcode_area (proxy_strength=0).
    D_proxy should be approximately 0 in this case.
    """
    return make_synthetic_dataset(n=n, proxy_strength=0.0, random_state=random_state)


def make_high_proxy_dataset(
    n: int = 2000,
    random_state: int = 42,
) -> tuple[pl.DataFrame, np.ndarray, object]:
    """
    Generate a dataset with strong proxy discrimination.

    proxy_feature is strongly correlated (0.8) with postcode_area.
    D_proxy should be clearly above 0.
    """
    return make_synthetic_dataset(n=n, proxy_strength=0.8, random_state=random_state)


@pytest.fixture
def synthetic_data():
    """Standard synthetic dataset with moderate proxy discrimination."""
    return make_synthetic_dataset(n=2000, proxy_strength=0.3, random_state=42)


@pytest.fixture
def zero_proxy_data():
    """Synthetic dataset with zero proxy discrimination."""
    return make_zero_proxy_dataset(n=1000, random_state=42)


@pytest.fixture
def high_proxy_data():
    """Synthetic dataset with strong proxy discrimination."""
    return make_high_proxy_dataset(n=2000, proxy_strength=0.8, random_state=42)


@pytest.fixture
def simple_model_and_data():
    """
    Very simple analytical case for testing admissible price computation.

    Two groups (S=0, S=1) with equal sizes. Model predicts:
      - Group 0: h = 100 for all
      - Group 1: h = 200 for all

    h_star = 150 for all (mean across groups).
    D_proxy = sqrt(mean(50^2)) / sqrt(mean(150^2)) = 50/150 = 1/3.
    """
    n = 100
    s = np.array([0] * 50 + [1] * 50, dtype=float)
    h = np.array([100.0] * 50 + [200.0] * 50)
    weights = np.ones(n)

    X = pl.DataFrame({
        "factor_a": np.random.default_rng(0).integers(0, 5, size=n).astype(float),
        "postcode": s,
    })

    return h, s, weights, X
