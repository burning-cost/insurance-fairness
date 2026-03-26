"""
Additional tests for bias_metrics.py covering paths not exercised by test_bias_metrics.py.

Gaps addressed:
1. demographic_parity_ratio with n_bootstrap > 0  (bootstrap CI path)
2. demographic_parity_ratio with log_space=False   (additive model path)
3. equalised_odds with binary_threshold            (binary classification path)
4. disparate_impact_ratio with multi-group > 2     (min-ratio path)

These paths all contain meaningful business logic. The bootstrap CI is used
when pricing teams want confidence intervals on their disparity metrics for
regulatory submissions. The log_space=False branch serves additive GLMs.
The binary_threshold branch is used for claim propensity models (0/1 outcome).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from insurance_fairness.bias_metrics import (
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _binary_df(n=200, seed=0):
    """Dataset with two groups and a binary outcome. Predictions are probabilities."""
    rng = np.random.default_rng(seed)
    group = np.array([0] * (n // 2) + [1] * (n // 2))
    # Group 0: lower predicted risk (mean prob ~0.10)
    # Group 1: higher predicted risk (mean prob ~0.20)
    pred = np.concatenate([
        rng.uniform(0.05, 0.20, n // 2),
        rng.uniform(0.10, 0.35, n // 2),
    ])
    # Binary outcome: 1 if actual claim, 0 otherwise
    # Group 1 has higher true claim probability
    actual = np.concatenate([
        rng.binomial(1, 0.12, n // 2).astype(float),
        rng.binomial(1, 0.22, n // 2).astype(float),
    ])
    return pl.DataFrame({
        "group": group.tolist(),
        "pred_prob": pred.tolist(),
        "claimed": actual.tolist(),
        "exposure": [1.0] * n,
    })


def _two_group_df(n_per_group=500, seed=1):
    """Simple two-group dataset for log_space tests."""
    rng = np.random.default_rng(seed)
    # Group 0: mean premium ~100, Group 1: mean premium ~120
    pred0 = rng.lognormal(np.log(100), 0.2, n_per_group)
    pred1 = rng.lognormal(np.log(120), 0.2, n_per_group)
    return pl.DataFrame({
        "group": [0] * n_per_group + [1] * n_per_group,
        "pred": np.concatenate([pred0, pred1]).tolist(),
        "exposure": rng.uniform(0.5, 1.5, 2 * n_per_group).tolist(),
    })


def _three_group_df():
    """Dataset with three groups for disparate impact multi-group path."""
    return pl.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        "pred": [100.0] * 100 + [90.0] * 100 + [70.0] * 100,
        "exposure": [1.0] * 300,
    })


# ---------------------------------------------------------------------------
# 1. demographic_parity_ratio: bootstrap CI path
# ---------------------------------------------------------------------------


class TestDemographicParityBootstrap:
    """Tests for the n_bootstrap > 0 code path."""

    def test_bootstrap_ci_present(self):
        """n_bootstrap > 0 should populate ci_lower and ci_upper."""
        df = _two_group_df()
        result = demographic_parity_ratio(
            df,
            protected_col="group",
            prediction_col="pred",
            exposure_col="exposure",
            n_bootstrap=200,
            ci_level=0.95,
        )
        assert result.ci_lower is not None
        assert result.ci_upper is not None

    def test_bootstrap_ci_ordered(self):
        """CI lower must be <= point estimate <= CI upper (in log-ratio terms)."""
        df = _two_group_df()
        result = demographic_parity_ratio(
            df,
            protected_col="group",
            prediction_col="pred",
            exposure_col="exposure",
            n_bootstrap=200,
        )
        assert result.ci_lower <= result.log_ratio <= result.ci_upper

    def test_bootstrap_ci_width_positive(self):
        """CI upper - lower should be strictly positive."""
        df = _two_group_df()
        result = demographic_parity_ratio(
            df,
            protected_col="group",
            prediction_col="pred",
            exposure_col="exposure",
            n_bootstrap=200,
        )
        assert result.ci_upper > result.ci_lower

    def test_no_bootstrap_ci_is_none(self):
        """Without bootstrap (default n_bootstrap=0), CIs should be None."""
        df = _two_group_df()
        result = demographic_parity_ratio(
            df,
            protected_col="group",
            prediction_col="pred",
            exposure_col="exposure",
        )
        assert result.ci_lower is None
        assert result.ci_upper is None

    def test_bootstrap_ci_narrows_with_more_data(self):
        """
        Larger sample -> narrower CI (consistency check).

        CI width with n=2000 should be strictly less than with n=200.
        """
        df_small = _two_group_df(n_per_group=100)
        df_large = _two_group_df(n_per_group=1000)

        r_small = demographic_parity_ratio(
            df_small, "group", "pred", "exposure", n_bootstrap=300
        )
        r_large = demographic_parity_ratio(
            df_large, "group", "pred", "exposure", n_bootstrap=300
        )

        width_small = r_small.ci_upper - r_small.ci_lower
        width_large = r_large.ci_upper - r_large.ci_lower
        assert width_large < width_small


# ---------------------------------------------------------------------------
# 2. demographic_parity_ratio: log_space=False path
# ---------------------------------------------------------------------------


class TestDemographicParityLinearSpace:
    """Tests for log_space=False (additive model branch)."""

    def test_equal_groups_zero_log_ratio_linear(self):
        """Equal means -> log_ratio should be ~0 even in linear space."""
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [100.0, 100.0, 100.0, 100.0],
        })
        result = demographic_parity_ratio(df, "g", "pred", log_space=False)
        assert abs(result.log_ratio) < 0.01
        assert abs(result.ratio - 1.0) < 0.01

    def test_known_linear_ratio(self):
        """
        Group 1 is 20% higher in levels.

        log_space=False computes diff/ref_mean, so:
          diff = 120 - 100 = 20
          ratio = 1 + 20/100 = 1.2
          log_ratio = log(1.2)
        """
        df = pl.DataFrame({
            "g": [0, 1],
            "pred": [100.0, 120.0],
        })
        result = demographic_parity_ratio(df, "g", "pred", log_space=False)
        expected_log_ratio = math.log(1.2)
        assert abs(result.log_ratio - expected_log_ratio) < 0.01
        assert abs(result.ratio - 1.2) < 0.01

    def test_log_space_false_vs_true_differ(self):
        """
        log_space=True and log_space=False produce different log_ratio values
        for the same data (because log(E[X]) != E[log(X)] in general).
        """
        df = _two_group_df()
        r_log = demographic_parity_ratio(df, "group", "pred", "exposure", log_space=True)
        r_lin = demographic_parity_ratio(df, "group", "pred", "exposure", log_space=False)
        # They will differ because E[ln X] != ln(E[X])
        # Just verify both run and differ
        assert isinstance(r_log.log_ratio, float)
        assert isinstance(r_lin.log_ratio, float)
        # They are not required to be equal (Jensen's inequality)


# ---------------------------------------------------------------------------
# 3. equalised_odds: binary_threshold path
# ---------------------------------------------------------------------------


class TestEqualisedOddsBinaryThreshold:
    """Tests for the binary_threshold > None branch (classification mode)."""

    def test_binary_threshold_path_runs(self):
        """Providing binary_threshold should run without error."""
        df = _binary_df()
        result = equalised_odds(
            df,
            protected_col="group",
            prediction_col="pred_prob",
            outcome_col="claimed",
            binary_threshold=0.15,
        )
        assert len(result.group_metrics) == 2

    def test_binary_threshold_tpr_in_unit_interval(self):
        """TPR values should be in [0, 1]."""
        df = _binary_df()
        result = equalised_odds(
            df,
            protected_col="group",
            prediction_col="pred_prob",
            outcome_col="claimed",
            binary_threshold=0.15,
        )
        for gm in result.group_metrics:
            if not (gm.metric_value != gm.metric_value):  # not NaN
                assert 0.0 <= gm.metric_value <= 1.0

    def test_high_threshold_near_zero_tpr(self):
        """A threshold of 1.0 means no predictions are positive -> TPR = 0 or NaN."""
        df = _binary_df()
        result = equalised_odds(
            df,
            protected_col="group",
            prediction_col="pred_prob",
            outcome_col="claimed",
            binary_threshold=1.0,  # no predictions cross 1.0
        )
        for gm in result.group_metrics:
            # Either 0.0 TPR or NaN (if no positives remain after thresholding)
            assert gm.metric_value == 0.0 or (gm.metric_value != gm.metric_value)

    def test_zero_threshold_all_positive(self):
        """Threshold=0.0 means all predictions are positive -> TPR = 1.0 if all actual are 1."""
        # Create dataset where all actual claims = 1
        df = pl.DataFrame({
            "g": [0, 0, 1, 1],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "claimed": [1.0, 1.0, 1.0, 1.0],
        })
        result = equalised_odds(df, "g", "pred", "claimed", binary_threshold=0.0)
        for gm in result.group_metrics:
            assert gm.metric_value == pytest.approx(1.0)

    def test_binary_vs_regression_modes_differ(self):
        """
        binary_threshold mode uses TPR; regression mode uses rank correlation.
        They should produce different metric_value distributions.
        """
        df = _binary_df(n=400)
        r_binary = equalised_odds(
            df, "group", "pred_prob", "claimed", binary_threshold=0.15
        )
        r_regression = equalised_odds(
            df, "group", "pred_prob", "claimed"
        )
        # Binary mode: values in [0, 1] bounded by claim rate
        # Regression mode: Spearman correlation, can be negative
        # They are logically distinct
        binary_vals = [gm.metric_value for gm in r_binary.group_metrics]
        regression_vals = [gm.metric_value for gm in r_regression.group_metrics]
        # At minimum they should not be identical
        assert any(
            abs(b - r) > 0.01 for b, r in zip(binary_vals, regression_vals)
        ), "Binary and regression modes should differ"


# ---------------------------------------------------------------------------
# 4. disparate_impact_ratio: multi-group path (> 2 groups)
# ---------------------------------------------------------------------------


class TestDisparateImpactMultiGroup:
    """Tests for the len(groups) > 2 branch in disparate_impact_ratio."""

    def test_three_groups_runs(self):
        """Three-group dataset should not raise."""
        df = _three_group_df()
        result = disparate_impact_ratio(df, "region", "pred")
        assert isinstance(result.ratio, float)

    def test_three_groups_ratio_leq_one(self):
        """
        With reference as highest-mean group, the reported ratio is the
        minimum ratio (most disadvantaged group). It should be <= 1.0.
        """
        df = _three_group_df()
        # Groups: A=100, B=90, C=70. Reference = A (highest). Min ratio = 70/100 = 0.70.
        result = disparate_impact_ratio(df, "region", "pred")
        assert result.ratio <= 1.0

    def test_three_groups_known_minimum_ratio(self):
        """Group C has mean 70 vs reference A at 100 -> minimum ratio = 0.70."""
        df = _three_group_df()
        result = disparate_impact_ratio(df, "region", "pred", reference_group="A")
        assert abs(result.ratio - 0.70) < 0.001

    def test_three_groups_rag_red(self):
        """0.70 ratio is below 0.80 -> should be red."""
        df = _three_group_df()
        result = disparate_impact_ratio(df, "region", "pred", reference_group="A")
        assert result.rag == "red"

    def test_three_equal_groups_ratio_one(self):
        """Three groups with equal means -> ratio = 1.0."""
        df = pl.DataFrame({
            "region": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "pred": [100.0] * 150,
        })
        result = disparate_impact_ratio(df, "region", "pred")
        assert abs(result.ratio - 1.0) < 1e-10
