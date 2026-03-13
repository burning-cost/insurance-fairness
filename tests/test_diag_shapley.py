"""
Tests for _shapley.py: Owen 2014 Shapley effects.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from insurance_fairness.diagnostics._shapley import (
    _build_surrogate,
    _characteristic_function,
    compute_shapley_effects,
    fit_surrogate_and_compute_shapley,
)


class TestBuildSurrogate:
    """Tests for _build_surrogate."""

    def test_returns_fitted_rf(self):
        """Should return a fitted RandomForestRegressor."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (200, 4))
        D = rng.normal(0, 1, 200)
        w = np.ones(200)

        rf = _build_surrogate(X, D, w)

        assert isinstance(rf, RandomForestRegressor)
        # Should be able to predict
        preds = rf.predict(X)
        assert preds.shape == (200,)

    def test_predictions_correlated_with_target(self):
        """Surrogate should learn a non-trivial relationship."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, (500, 3))
        D = 2 * X[:, 0] - 3 * X[:, 1] + rng.normal(0, 0.1, 500)
        w = np.ones(500)

        rf = _build_surrogate(X, D, w)
        preds = rf.predict(X)

        # Should have positive correlation with true D
        corr = np.corrcoef(preds, D)[0, 1]
        assert corr > 0.9


class TestCharacteristicFunction:
    """Tests for _characteristic_function."""

    def test_empty_subset_returns_zero(self):
        """v(empty set) = 0 by convention."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (100, 3))
        D = rng.normal(0, 1, 100)
        w = np.ones(100)
        rf = _build_surrogate(X, D, w)

        v = _characteristic_function(rf, X, D, w, [], n_features=3)
        assert v == 0.0

    def test_full_set_geq_empty_set(self):
        """v(full set) >= v(empty set)."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (300, 4))
        D = X[:, 0] * 2 + rng.normal(0, 0.1, 300)
        w = np.ones(300)
        rf = _build_surrogate(X, D, w)

        v_full = _characteristic_function(rf, X, D, w, [0, 1, 2, 3], n_features=4)
        v_empty = _characteristic_function(rf, X, D, w, [], n_features=4)

        assert v_full >= v_empty

    def test_monotone_subset_expansion(self):
        """Adding features should (on average) not decrease v(S)."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (500, 4))
        D = 3 * X[:, 0] + 2 * X[:, 1] + rng.normal(0, 0.05, 500)
        w = np.ones(500)
        rf = _build_surrogate(X, D, w)

        v_1 = _characteristic_function(rf, X, D, w, [0], n_features=4)
        v_12 = _characteristic_function(rf, X, D, w, [0, 1], n_features=4)

        # v({0,1}) >= v({0}) with high probability for a strong D~X relationship
        # Not strictly guaranteed but very likely with this setup
        assert v_12 >= v_1 - 0.01  # Allow tiny tolerance for RF approximation

    def test_returns_non_negative(self):
        """v(S) should be non-negative (it's a variance)."""
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (200, 3))
        D = rng.normal(0, 1, 200)
        w = np.ones(200)
        rf = _build_surrogate(X, D, w)

        for subset in [[], [0], [1], [0, 1], [0, 1, 2]]:
            v = _characteristic_function(rf, X, D, w, subset, n_features=3)
            assert v >= 0.0


class TestComputeShapleyEffects:
    """Tests for compute_shapley_effects."""

    def test_phi_sum_to_one(self):
        """Shapley effects should sum to 1.0 (after normalisation)."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, (500, 4))
        D = 2 * X[:, 0] + rng.normal(0, 0.5, 500)
        w = np.ones(500)
        rf = _build_surrogate(X, D, w)

        factor_names = ["f0", "f1", "f2", "f3"]
        phi = compute_shapley_effects(
            rf, X, D, w, factor_names, n_perms=64, random_state=42
        )

        assert abs(sum(phi.values()) - 1.0) < 0.01

    def test_phi_non_negative(self):
        """All Shapley effects should be non-negative."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 10, (400, 3))
        D = X[:, 0] + rng.normal(0, 0.2, 400)
        w = np.ones(400)
        rf = _build_surrogate(X, D, w)

        phi = compute_shapley_effects(
            rf, X, D, w, ["a", "b", "c"], n_perms=64, random_state=0
        )

        for name, val in phi.items():
            assert val >= 0.0, f"phi[{name}] = {val} < 0"

    def test_dominant_feature_gets_highest_phi(self):
        """
        When D is driven entirely by feature 0, phi[0] should be largest.
        """
        rng = np.random.default_rng(7)
        n = 800
        X = rng.uniform(0, 10, (n, 4))
        # D driven almost entirely by X[:, 0]
        D = 5 * X[:, 0] + 0.01 * rng.normal(0, 1, n)
        w = np.ones(n)
        rf = _build_surrogate(X, D, w)

        phi = compute_shapley_effects(
            rf, X, D, w, ["f0", "f1", "f2", "f3"], n_perms=128, random_state=7
        )

        assert phi["f0"] == max(phi.values()), (
            f"Expected f0 to dominate but got: {phi}"
        )

    def test_zero_variance_D_gives_zero_phi(self):
        """
        When D is constant (no discrimination variance), all phi should be 0.
        """
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 10, (300, 3))
        D = np.zeros(300)  # No discrimination residual
        w = np.ones(300)
        rf = _build_surrogate(X, D, w)

        phi = compute_shapley_effects(
            rf, X, D, w, ["a", "b", "c"], n_perms=32, random_state=0
        )

        for name, val in phi.items():
            assert val == pytest.approx(0.0, abs=0.05), f"phi[{name}] = {val}"

    def test_returns_all_factor_names(self):
        """phi dict should have an entry for every factor."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (200, 5))
        D = rng.normal(0, 1, 200)
        w = np.ones(200)
        rf = _build_surrogate(X, D, w)

        names = ["alpha", "beta", "gamma", "delta", "epsilon"]
        phi = compute_shapley_effects(rf, X, D, w, names, n_perms=32, random_state=0)

        assert set(phi.keys()) == set(names)

    def test_single_feature(self):
        """With a single feature, phi[0] should be 1.0."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 10, (200, 1))
        D = 3 * X[:, 0] + rng.normal(0, 0.1, 200)
        w = np.ones(200)
        rf = _build_surrogate(X, D, w)

        phi = compute_shapley_effects(rf, X, D, w, ["only"], n_perms=16, random_state=0)

        assert phi["only"] == pytest.approx(1.0, abs=1e-10)


class TestFitSurrogateAndComputeShapley:
    """Tests for the convenience wrapper."""

    def test_fits_surrogate_and_returns_phi(self):
        """Should fit surrogate internally and return phi dict."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, (400, 3))
        D = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, 400)
        w = np.ones(400)

        phi, surrogate = fit_surrogate_and_compute_shapley(
            X, D, w, ["x0", "x1", "x2"], n_perms=64, random_state=42
        )

        assert isinstance(phi, dict)
        assert isinstance(surrogate, RandomForestRegressor)
        assert abs(sum(phi.values()) - 1.0) < 0.02

    def test_pre_fitted_surrogate_is_used(self):
        """If surrogate_model is provided, it should be used directly."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (200, 2))
        D = rng.normal(0, 1, 200)
        w = np.ones(200)

        pre_fitted = _build_surrogate(X, D, w)

        phi, returned_surrogate = fit_surrogate_and_compute_shapley(
            X, D, w, ["a", "b"],
            n_perms=16,
            surrogate_model=pre_fitted,
            random_state=0,
        )

        # The returned surrogate should be the same object
        assert returned_surrogate is pre_fitted
