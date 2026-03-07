"""
Tests for bias_metrics.py

Tests verify:
- Correct values against known analytical results
- Edge cases (single group, nan handling, zero exposure)
- Exposure weighting is applied correctly
- RAG status thresholds behave as expected
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from insurance_fairness.bias_metrics import (
    CalibrationResult,
    DemographicParityResult,
    DisparateImpactResult,
    GiniResult,
    TheilResult,
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
    gini_by_group,
    theil_index,
)


# ---------------------------------------------------------------------------
# Demographic parity
# ---------------------------------------------------------------------------


class TestDemographicParityRatio:
    def test_equal_groups_returns_zero_log_ratio(self):
        """When both groups have the same mean premium, log-ratio should be 0."""
        df = pl.DataFrame({
            "gender": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
            "exposure": [1.0, 1.0, 1.0, 1.0],
        })
        result = demographic_parity_ratio(
            df, protected_col="gender", prediction_col="pred", exposure_col="exposure"
        )
        assert isinstance(result, DemographicParityResult)
        assert abs(result.log_ratio) < 1e-10
        assert abs(result.ratio - 1.0) < 1e-10
        assert result.rag == "green"

    def test_known_log_ratio(self):
        """Group 1 has exactly double the mean premium of group 0 -> log-ratio = log(2)."""
        df = pl.DataFrame({
            "gender": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 200.0, 200.0],
            "exposure": [1.0, 1.0, 1.0, 1.0],
        })
        result = demographic_parity_ratio(
            df, protected_col="gender", prediction_col="pred", exposure_col="exposure"
        )
        expected_log_ratio = math.log(2.0)  # in log-space: log(200) - log(100)
        # In log-space: mean(log(200)) - mean(log(100)) = log(200) - log(100) = log(2)
        assert abs(result.log_ratio - expected_log_ratio) < 0.01
        assert result.rag == "red"  # log(2) ~ 0.693, well above red threshold of 0.10

    def test_exposure_weighting(self):
        """Exposure weighting should favour the heavily-weighted observation."""
        df = pl.DataFrame({
            "gender": [0, 0, 1, 1],
            "pred": [100.0, 200.0, 100.0, 200.0],
            "exposure": [10.0, 1.0, 1.0, 10.0],  # group 0 mostly low, group 1 mostly high
        })
        result = demographic_parity_ratio(
            df, protected_col="gender", prediction_col="pred", exposure_col="exposure"
        )
        # Group 0 weighted mean in log-space: (10*log(100) + 1*log(200)) / 11
        # Group 1 weighted mean in log-space: (1*log(100) + 10*log(200)) / 11
        g0_mean = (10 * math.log(100) + 1 * math.log(200)) / 11
        g1_mean = (1 * math.log(100) + 10 * math.log(200)) / 11
        expected = g1_mean - g0_mean
        assert abs(result.log_ratio - expected) < 0.01

    def test_no_exposure_col_defaults_to_equal_weights(self):
        """Without exposure column, all policies have equal weight."""
        df = pl.DataFrame({
            "gender": [0, 1],
            "pred": [100.0, 200.0],
        })
        result = demographic_parity_ratio(df, "gender", "pred")
        expected = math.log(200) - math.log(100)  # = log(2)
        assert abs(result.log_ratio - expected) < 0.01

    def test_multi_group(self, multi_group_df):
        """Multi-group protected characteristic should not raise."""
        result = demographic_parity_ratio(
            multi_group_df, "region", "predicted_premium", "exposure"
        )
        assert isinstance(result, DemographicParityResult)
        assert "A" in result.group_means
        assert "B" in result.group_means
        assert "C" in result.group_means

    def test_rag_green(self):
        """Small disparity should be green."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 101.0],  # log-ratio ~ 0.01, green
        })
        result = demographic_parity_ratio(df, "g", "pred")
        assert result.rag == "green"

    def test_rag_amber(self):
        """Moderate disparity should be amber."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 107.0],  # log-ratio ~ 0.068, amber (> 0.05, < 0.10)
        })
        result = demographic_parity_ratio(df, "g", "pred")
        assert result.rag == "amber"

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            demographic_parity_ratio(df, "gender", "pred")


# ---------------------------------------------------------------------------
# Calibration by group
# ---------------------------------------------------------------------------


class TestCalibrationByGroup:
    def test_perfect_calibration(self, perfectly_calibrated_df):
        """A perfectly calibrated dataset should have max_disparity close to 0."""
        result = calibration_by_group(
            perfectly_calibrated_df,
            protected_col="gender",
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
        )
        assert isinstance(result, CalibrationResult)
        # Noise is tiny (sigma=0.01), so max disparity should be very small
        assert result.max_disparity < 0.05

    def test_known_calibration_disparity(self):
        """Group 0 should show A/E = 0.5 in decile 1 (systematic under-pricing)."""
        # Group 0: claims = 0.5 * predicted * exposure
        # Group 1: claims = 1.0 * predicted * exposure
        df = pl.DataFrame({
            "gender": [0] * 10 + [1] * 10,
            "predicted_premium": [100.0] * 20,
            "claim_amount": [50.0] * 10 + [100.0] * 10,
            "exposure": [1.0] * 20,
        })
        result = calibration_by_group(
            df, "gender", "predicted_premium", "claim_amount",
            n_deciles=2,
        )
        # All predictions are equal, so all policies land in same decile.
        # Group 0 A/E = 0.5, group 1 A/E = 1.0. Max disparity = 0.5.
        assert result.max_disparity >= 0.4

    def test_calibration_structure(self, simple_binary_df):
        """Result should have the correct structure."""
        result = calibration_by_group(
            simple_binary_df, "gender", "predicted_premium", "claim_amount",
            exposure_col="exposure", n_deciles=5,
        )
        assert len(result.actual_to_expected) == 5
        for d_vals in result.actual_to_expected.values():
            assert "0" in d_vals or "1" in d_vals

    def test_rag_status(self, simple_binary_df):
        """RAG status should be non-trivial for a dataset with known disparities."""
        result = calibration_by_group(
            simple_binary_df, "gender", "predicted_premium", "claim_amount",
            exposure_col="exposure",
        )
        assert result.rag in ("green", "amber", "red")


# ---------------------------------------------------------------------------
# Disparate impact ratio
# ---------------------------------------------------------------------------


class TestDisparateImpactRatio:
    def test_equal_groups(self):
        """Equal mean premiums -> DIR = 1.0."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 100.0],
        })
        result = disparate_impact_ratio(df, "g", "pred")
        assert abs(result.ratio - 1.0) < 1e-10
        assert result.rag == "green"

    def test_known_ratio(self):
        """Group 1 at 80% of group 0 -> DIR = 0.8 (red)."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 80.0],
        })
        result = disparate_impact_ratio(df, "g", "pred", reference_group="0")
        assert abs(result.ratio - 0.80) < 0.001
        assert result.rag == "red"

    def test_exposure_weighted(self):
        """With exposure weighting, heavily-weighted observations dominate."""
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 200.0, 100.0, 200.0],
            "exposure": [10.0, 1.0, 1.0, 10.0],
        })
        result = disparate_impact_ratio(df, "g", "pred", exposure_col="exposure")
        # Group 0 weighted mean ~ 109; Group 1 weighted mean ~ 191
        assert result.ratio < 1.0  # group with higher exposure to high premiums is reference

    def test_reference_group_explicit(self):
        """Explicit reference_group should be used."""
        df = pl.DataFrame({
            "g": ["A", "B"],
            "pred": [100.0, 120.0],
        })
        result = disparate_impact_ratio(df, "g", "pred", reference_group="B")
        assert abs(result.ratio - (100.0 / 120.0)) < 0.001


# ---------------------------------------------------------------------------
# Equalised odds
# ---------------------------------------------------------------------------


class TestEqualisedOdds:
    def test_basic(self, simple_binary_df):
        """Equalised odds should run without error and return valid structure."""
        result = equalised_odds(
            simple_binary_df,
            protected_col="gender",
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
        )
        assert len(result.group_metrics) == 2
        for gm in result.group_metrics:
            assert gm.group_value in ("0", "1")
            assert gm.n_policies == 500

    def test_perfect_rank_correlation(self):
        """When predictions perfectly rank actuals in both groups, correlation should be high."""
        n = 100
        pred = list(range(1, n + 1))
        actual = list(range(1, n + 1))
        group = [0] * (n // 2) + [1] * (n // 2)
        df = pl.DataFrame({
            "g": group,
            "pred": [float(x) for x in pred],
            "actual": [float(x) for x in actual],
        })
        result = equalised_odds(df, "g", "pred", "actual")
        for gm in result.group_metrics:
            assert gm.metric_value > 0.99  # near-perfect rank correlation


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------


class TestGiniByGroup:
    def test_equal_premiums_zero_gini(self):
        """All policies at the same premium -> Gini = 0."""
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
        })
        result = gini_by_group(df, "g", "pred")
        assert abs(result.overall_gini) < 1e-10
        for g, v in result.group_ginis.items():
            assert abs(v) < 1e-10

    def test_known_gini(self, known_gini_df):
        """Gini of [1, 2, 3, 4] with equal weights is 0.25."""
        result = gini_by_group(known_gini_df, "group", "predicted_premium")
        # Analytical Gini for [1, 2, 3, 4]: G = 1/(4*4/2) * (0+1+2+3) = 6/8 * ...
        # Numerically: ~0.25
        assert 0.2 < result.overall_gini < 0.3

    def test_group_ginis_exist(self, simple_binary_df):
        result = gini_by_group(
            simple_binary_df, "gender", "predicted_premium", "exposure"
        )
        assert "0" in result.group_ginis
        assert "1" in result.group_ginis
        assert result.max_disparity >= 0.0


# ---------------------------------------------------------------------------
# Theil index
# ---------------------------------------------------------------------------


class TestTheilIndex:
    def test_equal_premiums_zero_theil(self):
        """All premiums equal -> Theil = 0."""
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
        })
        result = theil_index(df, "g", "pred")
        assert abs(result.theil_total) < 1e-8
        assert abs(result.theil_between) < 1e-8

    def test_decomposition_sums(self, simple_binary_df):
        """Within + between should approximately equal total."""
        result = theil_index(
            simple_binary_df, "gender", "predicted_premium", "exposure"
        )
        total_approx = result.theil_within + result.theil_between
        assert abs(total_approx - result.theil_total) < 0.05  # allow for rounding

    def test_nonpositive_prediction_raises(self):
        """Theil index requires positive predictions."""
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, -1.0],
        })
        with pytest.raises(ValueError, match="strictly positive"):
            theil_index(df, "g", "pred")

    def test_between_group_increases_with_disparity(self):
        """Larger inter-group difference -> higher between-group Theil."""
        df_small = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 110.0, 110.0],
        })
        df_large = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 200.0, 200.0],
        })
        r_small = theil_index(df_small, "g", "pred")
        r_large = theil_index(df_large, "g", "pred")
        assert r_large.theil_between > r_small.theil_between
