"""
Tests for _utils.py: utilities, RAG thresholds, validation helpers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_fairness.diagnostics._utils import (
    DEFAULT_D_PROXY_THRESHOLDS,
    DEFAULT_PHI_THRESHOLDS,
    bootstrap_ci,
    d_proxy_rag,
    exposure_weighted_mean,
    exposure_weighted_var,
    phi_rag,
    resolve_exposure,
    subsample_indices,
    validate_columns,
    validate_dataframe,
    validate_model,
    validate_rating_factors,
)


class TestDProxyRAG:
    """Tests for d_proxy_rag."""

    def test_green_below_amber_threshold(self):
        assert d_proxy_rag(0.0) == "green"
        assert d_proxy_rag(0.04) == "green"
        assert d_proxy_rag(0.049) == "green"

    def test_amber_between_thresholds(self):
        assert d_proxy_rag(0.05) == "amber"
        assert d_proxy_rag(0.10) == "amber"
        assert d_proxy_rag(0.149) == "amber"

    def test_red_above_red_threshold(self):
        assert d_proxy_rag(0.15) == "red"
        assert d_proxy_rag(0.20) == "red"
        assert d_proxy_rag(0.99) == "red"

    def test_custom_thresholds(self):
        custom = {"amber": 0.02, "red": 0.10}
        assert d_proxy_rag(0.01, custom) == "green"
        assert d_proxy_rag(0.05, custom) == "amber"
        assert d_proxy_rag(0.15, custom) == "red"

    def test_exactly_at_amber_threshold(self):
        assert d_proxy_rag(DEFAULT_D_PROXY_THRESHOLDS["amber"]) == "amber"

    def test_exactly_at_red_threshold(self):
        assert d_proxy_rag(DEFAULT_D_PROXY_THRESHOLDS["red"]) == "red"


class TestPhiRAG:
    """Tests for phi_rag."""

    def test_green_below_amber(self):
        assert phi_rag(0.0) == "green"
        assert phi_rag(0.09) == "green"

    def test_amber_between_thresholds(self):
        assert phi_rag(0.10) == "amber"
        assert phi_rag(0.25) == "amber"

    def test_red_above_red_threshold(self):
        assert phi_rag(0.30) == "red"
        assert phi_rag(0.99) == "red"

    def test_custom_thresholds(self):
        custom = {"amber": 0.05, "red": 0.20}
        assert phi_rag(0.04, custom) == "green"
        assert phi_rag(0.10, custom) == "amber"
        assert phi_rag(0.25, custom) == "red"


class TestValidateModel:
    """Tests for validate_model."""

    def test_valid_model_passes(self):
        class MockModel:
            def predict(self, X):
                return X

        validate_model(MockModel())  # should not raise

    def test_no_predict_raises(self):
        with pytest.raises(TypeError, match="predict"):
            validate_model(object())

    def test_non_callable_predict_raises(self):
        class BadModel:
            predict = "not_callable"

        with pytest.raises(TypeError, match="predict"):
            validate_model(BadModel())


class TestValidateDataframe:
    """Tests for validate_dataframe."""

    def test_polars_dataframe_passes(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        validate_dataframe(df)  # should not raise

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="Polars DataFrame"):
            validate_dataframe(np.array([1, 2, 3]))

    def test_dict_raises(self):
        with pytest.raises(TypeError, match="Polars DataFrame"):
            validate_dataframe({"a": [1, 2, 3]})


class TestValidateColumns:
    """Tests for validate_columns."""

    def test_present_column_passes(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        validate_columns(df, "a", "b")  # should not raise

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="b"):
            validate_columns(df, "a", "b")

    def test_multiple_missing_raises(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Column"):
            validate_columns(df, "b", "c")


class TestValidateRatingFactors:
    """Tests for validate_rating_factors."""

    def test_valid_factors_pass(self):
        X = pl.DataFrame({"age": [1], "vehicle": [2], "s": [0]})
        validate_rating_factors(["age", "vehicle"], X, "s")  # should not raise

    def test_empty_factors_raises(self):
        X = pl.DataFrame({"age": [1], "s": [0]})
        with pytest.raises(ValueError, match="at least one"):
            validate_rating_factors([], X, "s")

    def test_sensitive_in_factors_raises(self):
        X = pl.DataFrame({"age": [1], "s": [0]})
        with pytest.raises(ValueError, match="sensitive_col"):
            validate_rating_factors(["age", "s"], X, "s")

    def test_missing_factor_raises(self):
        X = pl.DataFrame({"age": [1], "s": [0]})
        with pytest.raises(ValueError, match="Column"):
            validate_rating_factors(["age", "missing_col"], X, "s")


class TestResolveExposure:
    """Tests for resolve_exposure."""

    def test_unit_weights_when_no_exposure_col(self):
        X = pl.DataFrame({"a": [1, 2, 3]})
        w = resolve_exposure(X, None, 3)
        np.testing.assert_array_equal(w, np.ones(3))

    def test_reads_exposure_col(self):
        X = pl.DataFrame({"a": [1, 2], "exp": [0.5, 1.5]})
        w = resolve_exposure(X, "exp", 2)
        np.testing.assert_array_equal(w, [0.5, 1.5])

    def test_unit_weights_when_col_not_in_df(self):
        X = pl.DataFrame({"a": [1, 2, 3]})
        w = resolve_exposure(X, "missing_col", 3)
        np.testing.assert_array_equal(w, np.ones(3))

    def test_raises_on_nonpositive_exposure(self):
        X = pl.DataFrame({"a": [1, 2], "exp": [0.5, -1.0]})
        with pytest.raises(ValueError, match="non-positive"):
            resolve_exposure(X, "exp", 2)


class TestExposureWeightedMean:
    """Tests for exposure_weighted_mean."""

    def test_unit_weights_equals_unweighted_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.ones(4)
        result = exposure_weighted_mean(values, weights)
        assert result == pytest.approx(2.5)

    def test_zero_weight_total_returns_nan(self):
        values = np.array([1.0, 2.0])
        weights = np.zeros(2)
        result = exposure_weighted_mean(values, weights)
        assert np.isnan(result)

    def test_weighted_mean_analytical(self):
        values = np.array([0.0, 100.0])
        weights = np.array([3.0, 1.0])  # 75% weight on 0, 25% on 100
        result = exposure_weighted_mean(values, weights)
        assert result == pytest.approx(25.0)


class TestExposureWeightedVar:
    """Tests for exposure_weighted_var."""

    def test_constant_values_zero_variance(self):
        values = np.full(10, 5.0)
        weights = np.ones(10)
        var = exposure_weighted_var(values, weights)
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_two_values_analytical(self):
        values = np.array([0.0, 10.0])
        weights = np.array([1.0, 1.0])
        # mean = 5, var = E[(x-5)^2] = (25+25)/2 = 25
        var = exposure_weighted_var(values, weights)
        assert var == pytest.approx(25.0)


class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_returns_tuple_of_floats(self):
        values = np.random.default_rng(0).uniform(0, 1, 100)
        weights = np.ones(100)
        lo, hi = bootstrap_ci(values, weights, exposure_weighted_mean, n_bootstrap=50)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lower_leq_upper(self):
        values = np.random.default_rng(0).uniform(0, 1, 200)
        weights = np.ones(200)
        lo, hi = bootstrap_ci(values, weights, exposure_weighted_mean, n_bootstrap=100)
        assert lo <= hi

    def test_ci_contains_true_mean(self):
        """For a large sample, the CI should contain the sample mean."""
        rng = np.random.default_rng(42)
        values = rng.normal(50, 10, 1000)
        weights = np.ones(1000)
        true_mean = exposure_weighted_mean(values, weights)
        lo, hi = bootstrap_ci(
            values, weights, exposure_weighted_mean,
            n_bootstrap=500, ci_level=0.95, rng=rng
        )
        assert lo <= true_mean <= hi


class TestSubsampleIndices:
    """Tests for subsample_indices."""

    def test_returns_all_when_n_leq_subsample(self):
        rng = np.random.default_rng(0)
        idx = subsample_indices(100, 200, rng)
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_returns_subsample_n_when_n_large(self):
        rng = np.random.default_rng(0)
        idx = subsample_indices(1000, 200, rng)
        assert len(idx) == 200

    def test_indices_sorted(self):
        rng = np.random.default_rng(0)
        idx = subsample_indices(1000, 200, rng)
        assert np.all(idx[:-1] <= idx[1:])

    def test_indices_within_bounds(self):
        rng = np.random.default_rng(0)
        idx = subsample_indices(1000, 300, rng)
        assert np.all(idx >= 0)
        assert np.all(idx < 1000)

    def test_no_duplicates(self):
        rng = np.random.default_rng(0)
        idx = subsample_indices(1000, 300, rng)
        assert len(idx) == len(np.unique(idx))
