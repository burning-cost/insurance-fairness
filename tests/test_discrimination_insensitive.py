"""
Tests for DiscriminationInsensitiveReweighter.

The DGP used in the integration test (test_reweighting_reduces_demographic_parity)
is constructed carefully:
  - x0 is a PURE PROXY for A (highly correlated with group membership)
  - x1, x2 are noise features uncorrelated with A
  - y depends ONLY on x1 and x2, NOT on x0 and NOT directly on A

This means:
  - Without reweighting, a model that uses x0 learns a proxy for A.
  - With reweighting (X⊥A), the model can no longer exploit x0 to separate groups.
  - Demographic parity gap should shrink.

We deliberately avoid a DGP where x0 is both proxy AND direct cause of y.
In such a DGP, removing x0's proxy signal would also remove a legitimate
predictive signal, making the test ambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from insurance_fairness import DiscriminationInsensitiveReweighter, ReweighterDiagnostics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def binary_dataset(rng):
    """Binary A (0/1), n=500, 4 features."""
    n = 500
    A = rng.integers(0, 2, size=n)
    # x0 correlates with A; x1-x3 are noise
    x0 = A * 0.8 + rng.standard_normal(n) * 0.5
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = np.column_stack([x0, x1, x2, x3])
    return X, A


@pytest.fixture
def multiclass_dataset(rng):
    """Multi-class A (0/1/2), n=600, 3 features."""
    n = 600
    A = rng.integers(0, 3, size=n)
    x0 = A * 0.6 + rng.standard_normal(n) * 0.6
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = np.column_stack([x0, x1, x2])
    return X, A


@pytest.fixture
def string_label_dataset(rng):
    """String labels for A."""
    n = 400
    A_int = rng.integers(0, 2, size=n)
    A = np.where(A_int == 0, "male", "female")
    x0 = A_int * 0.7 + rng.standard_normal(n) * 0.4
    x1 = rng.standard_normal(n)
    X = np.column_stack([x0, x1])
    return X, A


# ---------------------------------------------------------------------------
# Basic fit / transform
# ---------------------------------------------------------------------------


def test_fit_returns_self(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    result = r.fit(X, A)
    assert result is r


def test_transform_shape(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    w = r.transform(X, A)
    assert w.shape == (len(A),)


def test_weights_all_positive(binary_dataset):
    X, A = binary_dataset
    w = DiscriminationInsensitiveReweighter().fit_transform(X, A)
    assert np.all(w > 0)


def test_weights_sum_to_n(binary_dataset):
    X, A = binary_dataset
    n = len(A)
    w = DiscriminationInsensitiveReweighter().fit_transform(X, A)
    assert abs(w.sum() - n) < 0.5, f"Expected sum~{n}, got {w.sum():.2f}"


def test_fit_transform_matches_fit_then_transform(binary_dataset):
    X, A = binary_dataset
    r1 = DiscriminationInsensitiveReweighter(random_state=7)
    r1.fit(X, A)
    w1 = r1.transform(X, A)

    r2 = DiscriminationInsensitiveReweighter(random_state=7)
    w2 = r2.fit_transform(X, A)

    np.testing.assert_allclose(w1, w2)


# ---------------------------------------------------------------------------
# Propensity model choices
# ---------------------------------------------------------------------------


def test_logistic_propensity(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter(propensity_model="logistic")
    w = r.fit_transform(X, A)
    assert w.shape == (len(A),)
    assert np.all(w > 0)


def test_random_forest_propensity(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter(
        propensity_model="random_forest",
        rf_kwargs={"n_estimators": 20},
    )
    w = r.fit_transform(X, A)
    assert w.shape == (len(A),)
    assert np.all(w > 0)


def test_logistic_vs_rf_give_different_weights(binary_dataset):
    X, A = binary_dataset
    w_lr = DiscriminationInsensitiveReweighter(propensity_model="logistic").fit_transform(X, A)
    w_rf = DiscriminationInsensitiveReweighter(
        propensity_model="random_forest", rf_kwargs={"n_estimators": 20}
    ).fit_transform(X, A)
    # Not identical but both valid
    assert not np.allclose(w_lr, w_rf)
    assert np.all(w_rf > 0)


# ---------------------------------------------------------------------------
# Multi-class A
# ---------------------------------------------------------------------------


def test_multiclass_shape(multiclass_dataset):
    X, A = multiclass_dataset
    w = DiscriminationInsensitiveReweighter().fit_transform(X, A)
    assert w.shape == (len(A),)


def test_multiclass_weights_positive(multiclass_dataset):
    X, A = multiclass_dataset
    w = DiscriminationInsensitiveReweighter().fit_transform(X, A)
    assert np.all(w > 0)


def test_multiclass_weights_sum_to_n(multiclass_dataset):
    X, A = multiclass_dataset
    n = len(A)
    w = DiscriminationInsensitiveReweighter().fit_transform(X, A)
    assert abs(w.sum() - n) < 0.5


# ---------------------------------------------------------------------------
# String labels
# ---------------------------------------------------------------------------


def test_string_labels_fit_transform(string_label_dataset):
    X, A = string_label_dataset
    w = DiscriminationInsensitiveReweighter().fit_transform(X, A)
    assert w.shape == (len(A),)
    assert np.all(w > 0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_returns_dataclass(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    d = r.diagnostics
    assert isinstance(d, ReweighterDiagnostics)


def test_diagnostics_effective_n_in_range(binary_dataset):
    X, A = binary_dataset
    n = len(A)
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    d = r.diagnostics
    # Effective n must be positive and <= n
    assert 0 < d.effective_n <= n


def test_diagnostics_per_group_propensity_keys(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    d = r.diagnostics
    # Binary A has groups 0 and 1
    assert set(d.per_group_propensity.keys()) == {0, 1}


def test_diagnostics_per_group_propensity_values(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    d = r.diagnostics
    for v in d.per_group_propensity.values():
        assert 0.0 < v <= 1.0


def test_diagnostics_propensity_scores_shape(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    d = r.diagnostics
    assert d.propensity_scores.shape == (len(A),)


def test_diagnostics_n_samples(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    d = r.diagnostics
    assert d.n_samples == len(A)


def test_diagnostics_n_groups_binary(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    assert r.diagnostics.n_groups == 2


def test_diagnostics_n_groups_multiclass(multiclass_dataset):
    X, A = multiclass_dataset
    r = DiscriminationInsensitiveReweighter()
    r.fit(X, A)
    assert r.diagnostics.n_groups == 3


def test_diagnostics_effective_n_strong_correlation(rng):
    """When X perfectly separates A, effective_n should be substantially below n."""
    n = 400
    A = rng.integers(0, 2, size=n)
    # x0 is nearly a deterministic function of A
    X = np.column_stack([A * 3.0 + rng.standard_normal(n) * 0.05])
    r = DiscriminationInsensitiveReweighter(propensity_model="logistic")
    r.fit(X, A)
    d = r.diagnostics
    # Effective n should be meaningfully below n due to extreme reweighting
    assert d.effective_n < n * 0.8, (
        f"Expected effective_n < {n * 0.8:.0f}, got {d.effective_n:.1f}"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises():
    r = DiscriminationInsensitiveReweighter()
    with pytest.raises(RuntimeError, match="not fitted"):
        r.transform(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]))


def test_diagnostics_before_fit_raises():
    r = DiscriminationInsensitiveReweighter()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = r.diagnostics


def test_invalid_propensity_model():
    with pytest.raises(ValueError, match="propensity_model must be"):
        DiscriminationInsensitiveReweighter(propensity_model="xgboost")


def test_mismatched_X_A_raises(binary_dataset):
    X, A = binary_dataset
    r = DiscriminationInsensitiveReweighter()
    with pytest.raises(ValueError, match="same number of observations"):
        r.fit(X, A[:-1])


def test_single_class_A_raises(rng):
    n = 100
    X = rng.standard_normal((n, 3))
    A = np.zeros(n, dtype=int)
    r = DiscriminationInsensitiveReweighter()
    with pytest.raises(ValueError, match="at least 2 distinct values"):
        r.fit(X, A)


# ---------------------------------------------------------------------------
# Integration: pure-proxy DGP
# ---------------------------------------------------------------------------


def test_reweighting_reduces_demographic_parity(rng):
    """
    DGP: x0 is a pure proxy for A (does not cause y).
    y depends only on x1 and x2.
    Reweighting should reduce the demographic parity gap in model predictions.

    We fit a linear ridge regression model:
      - Without reweighting: model learns x0 -> effectively learns A -> gap > 0
      - With reweighting: x0 loses predictive power for A in training distribution
        so model predictions are less correlated with A

    We measure demographic parity gap = |mean(pred | A=1) - mean(pred | A=0)|.
    """
    n = 2000
    # Protected attribute: balanced binary
    A = (rng.standard_normal(n) > 0).astype(int)

    # Pure proxy: x0 is highly correlated with A but is NOT in the outcome model
    x0 = A * 2.0 + rng.standard_normal(n) * 0.3

    # x1, x2 are actual predictors — independent of A
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    # y depends only on x1 and x2, NOT on x0, NOT on A directly
    y = 3.0 * x1 + 2.0 * x2 + rng.standard_normal(n) * 0.5

    X = np.column_stack([x0, x1, x2])

    # ---- Baseline: no reweighting ----
    model_base = Ridge(alpha=1.0)
    model_base.fit(X, y)
    pred_base = model_base.predict(X)
    gap_base = abs(pred_base[A == 1].mean() - pred_base[A == 0].mean())

    # ---- Reweighted ----
    r = DiscriminationInsensitiveReweighter(
        propensity_model="logistic", random_state=42
    )
    weights = r.fit_transform(X, A)

    model_fair = Ridge(alpha=1.0)
    model_fair.fit(X, y, sample_weight=weights)
    pred_fair = model_fair.predict(X)
    gap_fair = abs(pred_fair[A == 1].mean() - pred_fair[A == 0].mean())

    # The reweighted model should show a substantially smaller gap
    assert gap_fair < gap_base * 0.5, (
        f"Expected reweighting to reduce demographic parity gap by >50%. "
        f"Baseline gap: {gap_base:.4f}, Fair gap: {gap_fair:.4f}"
    )


# ---------------------------------------------------------------------------
# Pandas interop (if pandas is installed)
# ---------------------------------------------------------------------------


def test_pandas_dataframe_input(binary_dataset):
    pytest.importorskip("pandas")
    import pandas as pd

    X, A = binary_dataset
    X_df = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    A_s = pd.Series(A)

    r = DiscriminationInsensitiveReweighter()
    w = r.fit_transform(X_df, A_s)
    assert w.shape == (len(A),)
    assert np.all(w > 0)


def test_pandas_string_columns(rng):
    """Pandas 2.x string columns (StringDtype) are handled correctly."""
    pytest.importorskip("pandas")
    import pandas as pd

    n = 300
    A_int = rng.integers(0, 2, size=n)
    A = np.where(A_int == 0, "group_a", "group_b")

    x0 = A_int.astype(float) * 0.8 + rng.standard_normal(n) * 0.3
    x1 = rng.standard_normal(n)

    X = pd.DataFrame({"x0": x0, "x1": x1, "region": ["North", "South"] * (n // 2)})
    # Cast region to pandas StringDtype (pandas 2.x behaviour)
    X["region"] = X["region"].astype("string")

    r = DiscriminationInsensitiveReweighter()
    w = r.fit_transform(X, A)
    assert w.shape == (n,)
    assert np.all(w > 0)
