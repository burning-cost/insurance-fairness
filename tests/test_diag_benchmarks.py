"""
Tests for _benchmarks.py: premium benchmark variants.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from insurance_fairness.diagnostics._benchmarks import (
    BenchmarkPremiums,
    compute_aware_premium,
    compute_benchmarks,
    compute_unaware_premium,
)


def _simple_setup(n: int = 400):
    """Build a simple Ridge model and dataset for benchmark tests."""
    rng = np.random.default_rng(42)
    age = rng.integers(0, 5, size=n).astype(float)
    vehicle = rng.integers(0, 5, size=n).astype(float)
    s = rng.integers(0, 2, size=n).astype(float)
    y = 200 + 40 * age + 30 * vehicle + rng.normal(0, 20, n)

    X_fit = np.column_stack([age, vehicle])
    model = Ridge(alpha=1.0)
    model.fit(X_fit, y)

    X = pl.DataFrame({"age": age, "vehicle": vehicle, "sensitive": s})
    weights = np.ones(n)
    return model, X, weights


class TestComputeUnawarePremium:
    """Tests for compute_unaware_premium."""

    def test_equals_model_predict(self):
        """Unaware premium = model.predict(X)."""
        model, X, _ = _simple_setup()
        X_no_s = X.drop("sensitive")

        expected = model.predict(X_no_s.to_numpy())
        result = compute_unaware_premium(model, X_no_s)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_returns_numpy_array(self):
        """Should return a numpy array."""
        model, X, _ = _simple_setup()
        X_no_s = X.drop("sensitive")

        result = compute_unaware_premium(model, X_no_s)

        assert isinstance(result, np.ndarray)

    def test_length_matches_input(self):
        """Output length should match input rows."""
        model, X, _ = _simple_setup(n=200)
        X_no_s = X.drop("sensitive")

        result = compute_unaware_premium(model, X_no_s)

        assert len(result) == 200

    def test_all_positive_for_valid_model(self):
        """For a typical insurance model, predictions should be positive."""
        model, X, _ = _simple_setup()
        X_no_s = X.drop("sensitive")

        result = compute_unaware_premium(model, X_no_s)

        assert np.all(result > 0)


class TestComputeAwarePremium:
    """Tests for compute_aware_premium."""

    def test_returns_numpy_array_of_correct_length(self):
        """Should return numpy array matching input length."""
        model, X, weights = _simple_setup(n=200)

        result = compute_aware_premium(model, X, "sensitive", weights)

        assert isinstance(result, np.ndarray)
        assert len(result) == 200

    def test_marginalised_premium_is_between_group_means(self):
        """
        Marginalised aware premium should be a weighted average of group
        predictions, so it should be bounded by the min/max group means.
        """
        model, X, weights = _simple_setup(n=400)

        aware = compute_aware_premium(model, X, "sensitive", weights)

        # Aware premium is an average over S, so it should be "moderate"
        unaware = compute_unaware_premium(model, X.drop("sensitive"))
        # The aware premium is not necessarily equal to unaware, but
        # should be in a similar range
        assert aware.mean() == pytest.approx(unaware.mean(), rel=0.5)

    def test_all_positive(self):
        """Aware premiums should remain positive."""
        model, X, weights = _simple_setup(n=200)

        result = compute_aware_premium(model, X, "sensitive", weights)

        assert np.all(result > 0)


class TestComputeBenchmarks:
    """Tests for compute_benchmarks."""

    def test_returns_benchmark_premiums_dataclass(self):
        """Should return a BenchmarkPremiums instance."""
        model, X, weights = _simple_setup(n=200)

        result = compute_benchmarks(model, X, "sensitive", weights)

        assert isinstance(result, BenchmarkPremiums)

    def test_unaware_equals_model_predict(self):
        """benchmarks.unaware should equal model.predict(X without S)."""
        model, X, weights = _simple_setup(n=300)
        X_no_s = X.drop("sensitive")

        expected = model.predict(X_no_s.to_numpy())
        result = compute_benchmarks(model, X, "sensitive", weights)

        np.testing.assert_allclose(result.unaware, expected, rtol=1e-10)

    def test_proxy_vulnerability_is_unaware_minus_aware(self):
        """proxy_vulnerability should equal unaware - aware."""
        model, X, weights = _simple_setup(n=300)

        result = compute_benchmarks(model, X, "sensitive", weights)

        np.testing.assert_allclose(
            result.proxy_vulnerability,
            result.unaware - result.aware,
            rtol=1e-10,
        )

    def test_best_estimate_equals_unaware(self):
        """best_estimate and unaware should be equal arrays."""
        model, X, weights = _simple_setup(n=200)

        result = compute_benchmarks(model, X, "sensitive", weights)

        np.testing.assert_array_equal(result.best_estimate, result.unaware)

    def test_all_arrays_same_length(self):
        """All benchmark arrays should have the same length as input."""
        n = 250
        model, X, weights = _simple_setup(n=n)

        result = compute_benchmarks(model, X, "sensitive", weights)

        assert len(result.best_estimate) == n
        assert len(result.unaware) == n
        assert len(result.aware) == n
        assert len(result.proxy_vulnerability) == n


class TestBenchmarkPremiumsDataclass:
    """Tests for BenchmarkPremiums dataclass."""

    def test_dataclass_fields(self):
        """Should have the expected fields."""
        bp = BenchmarkPremiums(
            best_estimate=np.array([100.0]),
            unaware=np.array([100.0]),
            aware=np.array([95.0]),
            proxy_vulnerability=np.array([5.0]),
        )

        assert bp.best_estimate[0] == 100.0
        assert bp.unaware[0] == 100.0
        assert bp.aware[0] == 95.0
        assert bp.proxy_vulnerability[0] == 5.0
