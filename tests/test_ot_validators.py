"""Tests for input validators."""
import numpy as np
import polars as pl
import pytest

from insurance_fairness.optimal_transport._validators import (
    validate_dataframe_aligned,
    validate_epsilon,
    validate_exposure,
    validate_predictions,
    validate_protected_attrs_present,
)


class TestValidateExposure:
    def test_none_returns_ones(self):
        result = validate_exposure(None, 5)
        assert np.all(result == 1.0)
        assert result.shape == (5,)

    def test_valid_array_passes_through(self):
        exp = np.array([0.5, 1.0, 2.0])
        result = validate_exposure(exp, 3)
        assert np.allclose(result, exp)

    def test_wrong_shape_raises(self):
        exp = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="shape"):
            validate_exposure(exp, 5)

    def test_negative_exposure_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_exposure(np.array([-1.0, 1.0, 1.0]), 3)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="all zero"):
            validate_exposure(np.zeros(4), 4)

    def test_zero_values_allowed_if_not_all_zero(self):
        exp = np.array([0.0, 1.0, 2.0])
        result = validate_exposure(exp, 3)
        assert result[0] == 0.0


class TestValidatePredictions:
    def test_valid_predictions_pass(self):
        p = np.array([0.1, 0.2, 0.5])
        result = validate_predictions(p)
        assert np.allclose(result, p)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            validate_predictions(np.array([-0.1, 0.2]))

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            validate_predictions(np.array([0.0, 0.2]))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_predictions(np.array([0.1, float("nan")]))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_predictions(np.array([0.1, float("inf")]))

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            validate_predictions(np.array([[0.1, 0.2]]))


class TestValidateDataframeAligned:
    def test_correct_shape_passes(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        validate_dataframe_aligned(df, "test", 3)  # no exception

    def test_wrong_shape_raises(self):
        # df has 3 rows, expected 5 — message will say "must have 5 rows, got 3"
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="got 3"):
            validate_dataframe_aligned(df, "test", 5)


class TestValidateEpsilon:
    def test_zero_passes(self):
        validate_epsilon(0.0)  # no exception

    def test_one_passes(self):
        validate_epsilon(1.0)

    def test_midpoint_passes(self):
        validate_epsilon(0.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            validate_epsilon(-0.1)

    def test_over_one_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            validate_epsilon(1.1)


class TestValidateProtectedAttrsPresent:
    def test_present_passes(self):
        df = pl.DataFrame({"gender": ["M", "F"], "age": [25, 30]})
        validate_protected_attrs_present(["gender"], df, "D")

    def test_missing_raises(self):
        df = pl.DataFrame({"age": [25, 30]})
        with pytest.raises(ValueError, match="gender"):
            validate_protected_attrs_present(["gender"], df, "D")

    def test_multiple_attrs_one_missing_raises(self):
        df = pl.DataFrame({"gender": ["M"], "age": [25]})
        with pytest.raises(ValueError, match="disability"):
            validate_protected_attrs_present(["gender", "disability"], df, "D")
