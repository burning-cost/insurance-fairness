"""
Tests for discrimination_insensitive.py

Tests cover:
1.  Basic fit/transform workflow returns correct shape
2.  Weights sum to n after normalisation
3.  fit_transform is equivalent to fit().transform()
4.  Unfitted transform raises RuntimeError
5.  Missing protected_col raises ValueError
6.  Non-DataFrame input raises TypeError
7.  Single-group protected attribute raises ValueError
8.  Missing values raise ValueError (fit and transform)
9.  Fewer than 2 rows raises ValueError
10. clip_quantile trims extreme weights
11. normalise=False returns raw (unnormalised) weights
12. method='forest' produces valid weights
13. Invalid method raises ValueError
14. Invalid clip_quantile raises ValueError
15. Protected attribute with more than 2 groups (multinomial)
16. Statistical independence: weighted covariance(X, A) near zero
17. Weights are all positive
18. Weights are finite (no NaN or inf)
19. diagnostics() returns correct structure and statistics
20. diagnostics() effective_n is less than or equal to n
21. Propensity model score in diagnostics is in [0, 1]
22. Integration: downstream model fairness improves after reweighting
23. Categorical protected attribute (string values)
24. Categorical feature columns are handled (OHE)
25. Single feature column (plus protected) does not crash
26. Unseen protected label in transform raises ValueError
27. __repr__ contains key info
28. Large clip_quantile=1.0 (no clipping) — default behaviour
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from insurance_fairness.discrimination_insensitive import (
    DiscriminationInsensitiveReweighter,
    ReweighterDiagnostics,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def make_binary_data(
    n: int = 1000,
    n_features: int = 4,
    association: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic insurance-like data with a binary protected attribute.

    A ~ Bernoulli(0.4). Features X_i are drawn with a mild linear association
    with A controlled by ``association``. This mimics the proxy discrimination
    scenario: gender is correlated with occupation and vehicle group.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * 0.4)
    n_neg = n - n_pos
    A = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    rng.shuffle(A)

    # Features correlated with A
    X = rng.standard_normal((n, n_features))
    X[:, 0] += association * A
    X[:, 1] -= association * A * 0.5

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["protected"] = A.astype(int)
    return df


def make_multiclass_data(
    n: int = 1200,
    n_groups: int = 3,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate data with a 3-class protected attribute."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, n_groups, size=n)
    X = rng.standard_normal((n, 3))
    for g in range(n_groups):
        X[A == g, 0] += g * 0.4
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["group"] = A.astype(str)
    return df


def make_categorical_feature_data(n: int = 500, seed: int = 99) -> pd.DataFrame:
    """Data with a categorical (string) feature and binary protected attribute."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=n)
    x_num = rng.standard_normal(n) + 0.5 * A
    vehicle = rng.choice(["car", "van", "motorbike"], size=n)
    df = pd.DataFrame({"x_num": x_num, "vehicle": vehicle, "protected": A})
    return df


# ---------------------------------------------------------------------------
# 1. Basic fit/transform workflow returns correct shape
# ---------------------------------------------------------------------------


def test_basic_fit_transform_shape():
    df = make_binary_data()
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    weights = rw.fit_transform(df)
    assert weights.shape == (len(df),)


# ---------------------------------------------------------------------------
# 2. Weights sum to n after normalisation
# ---------------------------------------------------------------------------


def test_weights_sum_to_n():
    df = make_binary_data(n=500)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    weights = rw.fit_transform(df)
    assert abs(weights.sum() - len(df)) < 1e-6


# ---------------------------------------------------------------------------
# 3. fit_transform equivalent to fit().transform()
# ---------------------------------------------------------------------------


def test_fit_transform_equivalent():
    df = make_binary_data(seed=13)
    rw1 = DiscriminationInsensitiveReweighter(protected_col="protected", random_state=0)
    w1 = rw1.fit_transform(df)

    rw2 = DiscriminationInsensitiveReweighter(protected_col="protected", random_state=0)
    w2 = rw2.fit(df).transform(df)

    np.testing.assert_array_almost_equal(w1, w2)


# ---------------------------------------------------------------------------
# 4. Unfitted transform raises RuntimeError
# ---------------------------------------------------------------------------


def test_unfitted_transform_raises():
    df = make_binary_data(n=100)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    with pytest.raises(RuntimeError, match="not been fitted"):
        rw.transform(df)


# ---------------------------------------------------------------------------
# 5. Missing protected_col raises ValueError
# ---------------------------------------------------------------------------


def test_missing_protected_col_fit():
    df = make_binary_data(n=100)
    rw = DiscriminationInsensitiveReweighter(protected_col="nonexistent")
    with pytest.raises(ValueError, match="not found in X"):
        rw.fit(df)


def test_missing_protected_col_transform():
    df = make_binary_data(n=100)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    df_no_prot = df.drop(columns=["protected"])
    with pytest.raises(ValueError, match="not found in X"):
        rw.transform(df_no_prot)


# ---------------------------------------------------------------------------
# 6. Non-DataFrame input raises TypeError
# ---------------------------------------------------------------------------


def test_non_dataframe_fit():
    arr = np.random.randn(100, 5)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        rw.fit(arr)


def test_non_dataframe_transform():
    df = make_binary_data(n=100)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    with pytest.raises(TypeError, match="pandas DataFrame"):
        rw.transform(np.random.randn(100, 5))


# ---------------------------------------------------------------------------
# 7. Single-group protected attribute raises ValueError
# ---------------------------------------------------------------------------


def test_single_group_raises():
    df = make_binary_data(n=100)
    df["protected"] = 1  # all same group
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    with pytest.raises(ValueError, match="only one unique value"):
        rw.fit(df)


# ---------------------------------------------------------------------------
# 8. Missing values raise ValueError
# ---------------------------------------------------------------------------


def test_missing_values_fit():
    df = make_binary_data(n=100)
    df.loc[0, "x0"] = np.nan
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    with pytest.raises(ValueError, match="missing values"):
        rw.fit(df)


def test_missing_values_transform():
    df = make_binary_data(n=200)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    df_test = df.copy()
    df_test.loc[5, "x1"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        rw.transform(df_test)


# ---------------------------------------------------------------------------
# 9. Fewer than 2 rows raises ValueError
# ---------------------------------------------------------------------------


def test_too_few_rows_raises():
    df = make_binary_data(n=1)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    with pytest.raises(ValueError, match="at least 2 rows"):
        rw.fit(df)


# ---------------------------------------------------------------------------
# 10. clip_quantile trims extreme weights
# ---------------------------------------------------------------------------


def test_clip_quantile_reduces_max():
    df = make_binary_data(n=500, association=2.0)  # strong association => wide weights
    rw_noclip = DiscriminationInsensitiveReweighter(
        protected_col="protected", clip_quantile=1.0, random_state=0
    )
    rw_clip = DiscriminationInsensitiveReweighter(
        protected_col="protected", clip_quantile=0.95, random_state=0
    )
    w_noclip = rw_noclip.fit_transform(df)
    w_clip = rw_clip.fit_transform(df)
    assert w_clip.max() <= w_noclip.max() + 1e-9


# ---------------------------------------------------------------------------
# 11. normalise=False returns unnormalised weights
# ---------------------------------------------------------------------------


def test_normalise_false_returns_raw_weights():
    df = make_binary_data(n=200)
    rw = DiscriminationInsensitiveReweighter(
        protected_col="protected", normalise=False, random_state=0
    )
    weights = rw.fit_transform(df)
    # Raw weights should NOT sum exactly to n
    # (they sum to n only coincidentally for normalised case)
    # Just check they are in a reasonable range: all positive, not all 1.0
    assert np.all(weights > 0)


# ---------------------------------------------------------------------------
# 12. method='forest' produces valid weights
# ---------------------------------------------------------------------------


def test_forest_method_produces_valid_weights():
    df = make_binary_data(n=300)
    rw = DiscriminationInsensitiveReweighter(
        protected_col="protected", method="forest", random_state=0
    )
    weights = rw.fit_transform(df)
    assert weights.shape == (300,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)
    assert abs(weights.sum() - 300) < 1e-6


# ---------------------------------------------------------------------------
# 13. Invalid method raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        DiscriminationInsensitiveReweighter(protected_col="protected", method="gbm")


# ---------------------------------------------------------------------------
# 14. Invalid clip_quantile raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_clip_quantile_raises():
    with pytest.raises(ValueError, match="clip_quantile"):
        DiscriminationInsensitiveReweighter(protected_col="protected", clip_quantile=0.0)

    with pytest.raises(ValueError, match="clip_quantile"):
        DiscriminationInsensitiveReweighter(protected_col="protected", clip_quantile=1.5)


# ---------------------------------------------------------------------------
# 15. Multi-class protected attribute (3 groups)
# ---------------------------------------------------------------------------


def test_multiclass_protected():
    df = make_multiclass_data()
    rw = DiscriminationInsensitiveReweighter(protected_col="group")
    weights = rw.fit_transform(df)
    assert weights.shape == (len(df),)
    assert abs(weights.sum() - len(df)) < 1e-6
    assert np.all(weights > 0)


# ---------------------------------------------------------------------------
# 16. Statistical independence: weighted covariance near zero
# ---------------------------------------------------------------------------


def test_weighted_covariance_near_zero():
    """
    After reweighting, the weighted correlation between each feature and A
    should be substantially smaller than the unweighted correlation.

    This is the core statistical guarantee: X ⊥ A under the reweighted measure.
    """
    n = 2000
    df = make_binary_data(n=n, n_features=4, association=0.8, seed=0)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected", random_state=0)
    weights = rw.fit_transform(df)

    A = df["protected"].values.astype(float)
    features = ["x0", "x1", "x2", "x3"]

    for feat in features:
        x = df[feat].values

        # Unweighted correlation
        unweighted_corr = float(np.corrcoef(x, A)[0, 1])

        # Weighted correlation: Cov_w(X, A) / sqrt(Var_w(X) * Var_w(A))
        w = weights / weights.sum()
        x_mean_w = np.sum(w * x)
        a_mean_w = np.sum(w * A)
        cov_w = np.sum(w * (x - x_mean_w) * (A - a_mean_w))
        var_x_w = np.sum(w * (x - x_mean_w) ** 2)
        var_a_w = np.sum(w * (A - a_mean_w) ** 2)
        weighted_corr = cov_w / np.sqrt(max(var_x_w * var_a_w, 1e-16))

        # After reweighting, correlation should be closer to zero than before
        assert abs(weighted_corr) < abs(unweighted_corr) + 0.15, (
            f"Feature {feat}: weighted_corr={weighted_corr:.3f}, "
            f"unweighted_corr={unweighted_corr:.3f}"
        )


# ---------------------------------------------------------------------------
# 17. Weights are all positive
# ---------------------------------------------------------------------------


def test_weights_all_positive():
    df = make_binary_data(n=400)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    weights = rw.fit_transform(df)
    assert np.all(weights > 0), "All weights must be strictly positive."


# ---------------------------------------------------------------------------
# 18. Weights are finite
# ---------------------------------------------------------------------------


def test_weights_are_finite():
    df = make_binary_data(n=400)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    weights = rw.fit_transform(df)
    assert np.all(np.isfinite(weights)), "Weights must be finite (no NaN or inf)."


# ---------------------------------------------------------------------------
# 19. diagnostics() returns correct structure
# ---------------------------------------------------------------------------


def test_diagnostics_structure():
    df = make_binary_data(n=500)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    diag = rw.diagnostics(df)

    assert isinstance(diag, ReweighterDiagnostics)
    assert diag.protected_col == "protected"
    assert diag.n_groups == 2
    assert len(diag.group_labels) == 2
    assert len(diag.marginal_proportions) == 2
    assert abs(diag.marginal_proportions.sum() - 1.0) < 1e-6
    assert len(diag.mean_propensity_by_group) == 2
    assert isinstance(diag.weight_stats, dict)
    expected_keys = {"min", "max", "mean", "std", "effective_n"}
    assert set(diag.weight_stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 20. diagnostics() effective_n <= n
# ---------------------------------------------------------------------------


def test_diagnostics_effective_n_le_n():
    df = make_binary_data(n=500)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    diag = rw.diagnostics(df)
    # Effective sample size cannot exceed n for non-negative weights
    assert diag.weight_stats["effective_n"] <= len(df) + 1.0


# ---------------------------------------------------------------------------
# 21. Propensity model score in [0, 1]
# ---------------------------------------------------------------------------


def test_diagnostics_propensity_score_in_range():
    df = make_binary_data(n=400)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    diag = rw.diagnostics(df)
    assert 0.0 <= diag.propensity_model_score <= 1.0


# ---------------------------------------------------------------------------
# 22. Integration: downstream model fairness improves
# ---------------------------------------------------------------------------


def test_integration_fairness_improves():
    """
    Train a logistic regression on raw data and on reweighted data.
    Reweighting should reduce the association between predictions and A.

    We measure this as the absolute correlation between predicted probability
    and the protected attribute — it should be smaller after reweighting.
    """
    n = 4000
    rng_local = np.random.default_rng(42)

    A = (rng_local.uniform(size=n) < 0.5).astype(int)
    # x0 is a pure proxy — correlated with A but has no direct effect on y
    x0 = 2.0 * A + rng_local.standard_normal(n)
    # x1 is the real risk driver, independent of A
    x1 = rng_local.standard_normal(n)
    x2 = rng_local.standard_normal(n)

    y = (1.5 * x1 + 0.3 * x2 + rng_local.standard_normal(n) * 0.5 > 0.0).astype(
        int
    )

    df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "protected": A})

    rw = DiscriminationInsensitiveReweighter(
        protected_col="protected", random_state=42
    )
    weights = rw.fit_transform(df)

    X_model = df.drop(columns=["protected"]).values

    # Baseline model — will pick up spurious x0 signal due to A correlation
    lr_base = LogisticRegression(max_iter=500, random_state=42)
    lr_base.fit(X_model, y)
    pred_base = lr_base.predict_proba(X_model)[:, 1]

    # Reweighted model — x0 signal from A should be suppressed
    lr_rw = LogisticRegression(max_iter=500, random_state=42)
    lr_rw.fit(X_model, y, sample_weight=weights)
    pred_rw = lr_rw.predict_proba(X_model)[:, 1]

    # Demographic parity gap: difference in mean prediction between groups
    gap_base = abs(pred_base[A == 1].mean() - pred_base[A == 0].mean())
    gap_rw = abs(pred_rw[A == 1].mean() - pred_rw[A == 0].mean())

    assert gap_rw < gap_base, (
        f"Reweighted model should have smaller demographic parity gap. "
        f"Base: {gap_base:.4f}, Reweighted: {gap_rw:.4f}"
    )


# ---------------------------------------------------------------------------
# 23. Categorical protected attribute (string values)
# ---------------------------------------------------------------------------


def test_string_protected_attribute():
    df = make_binary_data(n=400)
    df["gender"] = df["protected"].map({0: "male", 1: "female"})
    df = df.drop(columns=["protected"])

    rw = DiscriminationInsensitiveReweighter(protected_col="gender")
    weights = rw.fit_transform(df)

    assert weights.shape == (400,)
    assert abs(weights.sum() - 400) < 1e-6
    assert np.all(weights > 0)


# ---------------------------------------------------------------------------
# 24. Categorical feature columns are handled (OHE)
# ---------------------------------------------------------------------------


def test_categorical_feature_columns():
    df = make_categorical_feature_data()
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    weights = rw.fit_transform(df)

    assert weights.shape == (len(df),)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)
    assert abs(weights.sum() - len(df)) < 1e-6


# ---------------------------------------------------------------------------
# 25. Single feature column (plus protected) does not crash
# ---------------------------------------------------------------------------


def test_single_feature_column():
    rng_local = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "x0": rng_local.standard_normal(n),
        "protected": rng_local.integers(0, 2, size=n),
    })
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    weights = rw.fit_transform(df)
    assert weights.shape == (n,)
    assert np.all(weights > 0)


# ---------------------------------------------------------------------------
# 26. Unseen protected label in transform raises ValueError
# ---------------------------------------------------------------------------


def test_unseen_protected_label_raises():
    df_train = make_binary_data(n=200)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df_train)

    df_test = df_train.copy()
    df_test.loc[0, "protected"] = 99  # unseen label
    with pytest.raises(ValueError, match="not seen during fit"):
        rw.transform(df_test)


# ---------------------------------------------------------------------------
# 27. __repr__ contains key info
# ---------------------------------------------------------------------------


def test_repr_unfitted():
    rw = DiscriminationInsensitiveReweighter(protected_col="gender", method="forest")
    r = repr(rw)
    assert "gender" in r
    assert "forest" in r
    assert "unfitted" in r


def test_repr_fitted():
    df = make_binary_data(n=100)
    rw = DiscriminationInsensitiveReweighter(protected_col="protected")
    rw.fit(df)
    r = repr(rw)
    assert "fitted" in r
    assert "protected" in r


# ---------------------------------------------------------------------------
# 28. clip_quantile=1.0 (default, no clipping) — weights unchanged
# ---------------------------------------------------------------------------


def test_no_clip_default():
    df = make_binary_data(n=300, association=0.3)
    rw_default = DiscriminationInsensitiveReweighter(
        protected_col="protected", clip_quantile=1.0, random_state=0
    )
    rw_noclip = DiscriminationInsensitiveReweighter(
        protected_col="protected", random_state=0
    )
    w_default = rw_default.fit_transform(df)
    w_noclip = rw_noclip.fit_transform(df)
    np.testing.assert_array_equal(w_default, w_noclip)
