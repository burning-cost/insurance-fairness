"""
Tests for _utils.py (internal helpers).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_fairness._utils import (
    assign_prediction_deciles,
    bootstrap_ci,
    exposure_weighted_mean,
    log_ratio,
    rag_status,
    resolve_exposure,
    to_polars,
    validate_binary,
    validate_columns,
    validate_positive,
)


class TestToPolars:
    def test_passthrough_polars(self):
        df = pl.DataFrame({"a": [1, 2]})
        assert to_polars(df) is df

    def test_converts_pandas(self):
        import pandas as pd

        pdf = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = to_polars(pdf)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected a Polars or pandas DataFrame"):
            to_polars([1, 2, 3])


class TestValidateColumns:
    def test_raises_on_missing_column(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            validate_columns(df, "b")

    def test_passes_on_present_columns(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        validate_columns(df, "a", "b")  # Should not raise


class TestValidatePositive:
    def test_raises_on_zero(self):
        df = pl.DataFrame({"x": [0.0, 1.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_positive(df, "x")

    def test_raises_on_negative(self):
        df = pl.DataFrame({"x": [-1.0, 1.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_positive(df, "x")

    def test_passes_on_positive(self):
        df = pl.DataFrame({"x": [0.1, 1.0, 100.0]})
        validate_positive(df, "x")  # Should not raise


class TestValidateBinary:
    def test_raises_on_non_binary(self):
        df = pl.DataFrame({"x": [0, 1, 2]})
        with pytest.raises(ValueError, match="binary"):
            validate_binary(df, "x")

    def test_passes_on_binary(self):
        df = pl.DataFrame({"x": [0, 1, 0, 1]})
        validate_binary(df, "x")  # Should not raise


class TestResolveExposure:
    def test_returns_series_from_col(self):
        df = pl.DataFrame({"exp": [0.5, 1.0, 0.75]})
        result = resolve_exposure(df, "exp")
        assert list(result) == [0.5, 1.0, 0.75]

    def test_returns_ones_when_col_none(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = resolve_exposure(df, None)
        assert list(result) == [1.0, 1.0, 1.0]

    def test_returns_ones_when_col_missing(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = resolve_exposure(df, "nonexistent")
        assert list(result) == [1.0, 1.0]


class TestExposureWeightedMean:
    def test_equal_weights(self):
        vals = pl.Series([1.0, 2.0, 3.0])
        wts = pl.Series([1.0, 1.0, 1.0])
        assert abs(exposure_weighted_mean(vals, wts) - 2.0) < 1e-10

    def test_unequal_weights(self):
        vals = pl.Series([0.0, 10.0])
        wts = pl.Series([9.0, 1.0])  # 90% weight on 0, 10% on 10
        result = exposure_weighted_mean(vals, wts)
        assert abs(result - 1.0) < 1e-10

    def test_zero_weight_returns_nan(self):
        vals = pl.Series([1.0, 2.0])
        wts = pl.Series([0.0, 0.0])
        result = exposure_weighted_mean(vals, wts)
        assert result != result  # nan != nan


class TestLogRatio:
    def test_equal_values(self):
        assert abs(log_ratio(1.0, 1.0)) < 1e-10

    def test_double(self):
        import math

        assert abs(log_ratio(2.0, 1.0) - math.log(2)) < 1e-10

    def test_zero_denominator(self):
        assert log_ratio(1.0, 0.0) != log_ratio(1.0, 0.0)  # nan


class TestAssignPredictionDeciles:
    def test_correct_number_of_deciles(self):
        import numpy as np

        n = 100
        df = pl.DataFrame({
            "pred": np.linspace(1, 100, n).tolist()
        })
        result = assign_prediction_deciles(df, "pred", n_deciles=10)
        assert "prediction_decile" in result.columns
        unique_deciles = result["prediction_decile"].unique().sort().to_list()
        assert min(unique_deciles) >= 1
        assert max(unique_deciles) <= 10

    def test_single_value(self):
        """All same predictions -> all in decile 1."""
        df = pl.DataFrame({"pred": [100.0] * 20})
        result = assign_prediction_deciles(df, "pred", n_deciles=5)
        assert result["prediction_decile"].unique().to_list() == [1]


class TestBootstrapCI:
    def test_mean_ci_contains_true_mean(self):
        """Bootstrap CI for the mean should contain the true population mean."""
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 1000)
        wts = np.ones(1000)

        def _mean(v, w):
            return float(np.average(v, weights=w))

        lo, hi = bootstrap_ci(vals, wts, _mean, n_bootstrap=200)
        # True mean is 0; CI should contain 0 with high probability
        assert lo < 0.2 and hi > -0.2

    def test_ci_lower_less_than_upper(self):
        rng = np.random.default_rng(0)
        vals = rng.uniform(0, 1, 100)
        wts = np.ones(100)
        lo, hi = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)), n_bootstrap=50)
        assert lo <= hi


class TestRagStatus:
    def test_disparate_impact_green(self):
        assert rag_status("disparate_impact_ratio", 1.0) == "green"

    def test_disparate_impact_amber(self):
        assert rag_status("disparate_impact_ratio", 0.87) == "amber"

    def test_disparate_impact_red(self):
        assert rag_status("disparate_impact_ratio", 0.79) == "red"

    def test_proxy_r2_green(self):
        assert rag_status("proxy_r2", 0.02) == "green"

    def test_proxy_r2_amber(self):
        assert rag_status("proxy_r2", 0.07) == "amber"

    def test_proxy_r2_red(self):
        assert rag_status("proxy_r2", 0.15) == "red"

    def test_unknown_metric(self):
        assert rag_status("nonexistent_metric", 0.5) == "unknown"
