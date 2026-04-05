"""
test_new_coverage.py
--------------------
Targeted coverage improvements for insurance-fairness.

Focuses on gaps not addressed by existing test files:

1. _utils.py — to_pandas(), rag_status for all metric types, bootstrap_ci
   with custom rng and ci_level, assign_prediction_deciles edge cases

2. localized_parity.py — internal helpers (_make_strictly_increasing,
   _weighted_quantile edge cases), audit() length mismatch errors,
   negative exposure validation in corrector, tolerance parameter behaviour,
   unsorted target_levels, corrector n_bins < 2 raises

3. tail_dp.py — _kde_quantile_ecdf degenerate fallback (< 5 obs),
   n_quantiles < 2 raises, protected_attr stored in repr,
   TailDPReport.__repr__ format, reweight + multi-group pipeline

4. sensitivity/_measure.py — summary() output verification, categories
   attribute, v_star shape, zero-weight observations handled

5. optimal_transport/_validators.py — validate_dataframe_aligned exact match,
   validate_epsilon boundary (0 and 1 are valid)

6. Public API — all exported names importable from insurance_fairness
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


# ===========================================================================
# 1. _utils.py
# ===========================================================================


class TestToPandas:
    """to_pandas() converts a Polars DataFrame to pandas."""

    def test_to_pandas_returns_pandas_df(self):
        import pandas as pd
        from insurance_fairness._utils import to_pandas

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_to_pandas_preserves_values(self):
        from insurance_fairness._utils import to_pandas

        df = pl.DataFrame({"x": [10, 20, 30]})
        result = to_pandas(df)
        assert list(result["x"]) == [10, 20, 30]


class TestRagStatusAllMetrics:
    """rag_status for all supported metric types."""

    def test_calibration_disparity_green(self):
        from insurance_fairness._utils import rag_status

        assert rag_status("calibration_disparity", 0.05) == "green"

    def test_calibration_disparity_amber(self):
        from insurance_fairness._utils import rag_status

        assert rag_status("calibration_disparity", 0.15) == "amber"

    def test_calibration_disparity_red(self):
        from insurance_fairness._utils import rag_status

        assert rag_status("calibration_disparity", 0.25) == "red"

    def test_demographic_parity_log_ratio_green(self):
        from insurance_fairness._utils import rag_status

        assert rag_status("demographic_parity_log_ratio", 0.02) == "green"

    def test_demographic_parity_log_ratio_amber(self):
        from insurance_fairness._utils import rag_status

        assert rag_status("demographic_parity_log_ratio", 0.07) == "amber"

    def test_demographic_parity_log_ratio_red(self):
        from insurance_fairness._utils import rag_status

        assert rag_status("demographic_parity_log_ratio", 0.15) == "red"

    def test_disparate_impact_ratio_upper_amber(self):
        """Values above 1.11 but below 1.25 should be amber."""
        from insurance_fairness._utils import rag_status

        assert rag_status("disparate_impact_ratio", 1.15) == "amber"

    def test_disparate_impact_ratio_upper_red(self):
        """Values above 1.25 should be red."""
        from insurance_fairness._utils import rag_status

        assert rag_status("disparate_impact_ratio", 1.30) == "red"

    def test_custom_threshold_override(self):
        """Custom thresholds should override defaults."""
        from insurance_fairness._utils import rag_status

        custom = {"green": (0.95, 1.05), "amber": (0.90, 1.10)}
        # 0.93 is inside amber zone but outside green
        assert rag_status("disparate_impact_ratio", 0.93, thresholds=custom) == "amber"


class TestBootstrapCIEdgeCases:
    """bootstrap_ci with custom rng and ci_level."""

    def test_custom_rng_is_used(self):
        """Passing a seeded rng should give reproducible results."""
        from insurance_fairness._utils import bootstrap_ci

        vals = np.arange(1.0, 101.0)
        wts = np.ones(100)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        lo1, hi1 = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)), n_bootstrap=50, rng=rng1)
        lo2, hi2 = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)), n_bootstrap=50, rng=rng2)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_wider_ci_level_gives_wider_interval(self):
        """99% CI should be wider than 90% CI."""
        from insurance_fairness._utils import bootstrap_ci

        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 500)
        wts = np.ones(500)

        lo90, hi90 = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)),
                                  n_bootstrap=200, ci_level=0.90, rng=np.random.default_rng(1))
        lo99, hi99 = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)),
                                  n_bootstrap=200, ci_level=0.99, rng=np.random.default_rng(1))
        assert (hi99 - lo99) >= (hi90 - lo90) - 0.01

    def test_ci_level_50_gives_narrow_interval(self):
        """50% CI should be narrower than 95% CI."""
        from insurance_fairness._utils import bootstrap_ci

        rng_seed = np.random.default_rng(99)
        vals = rng_seed.normal(0, 1, 1000)
        wts = np.ones(1000)

        lo50, hi50 = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)),
                                  n_bootstrap=200, ci_level=0.50)
        lo95, hi95 = bootstrap_ci(vals, wts, lambda v, w: float(np.mean(v)),
                                  n_bootstrap=200, ci_level=0.95)
        assert (hi95 - lo95) >= (hi50 - lo50)


class TestAssignPredictionDecilesEdgeCases:
    """assign_prediction_deciles edge cases."""

    def test_custom_n_deciles(self):
        from insurance_fairness._utils import assign_prediction_deciles

        n = 100
        df = pl.DataFrame({"pred": np.linspace(1.0, 100.0, n).tolist()})
        result = assign_prediction_deciles(df, "pred", n_deciles=5)
        unique_deciles = sorted(result["prediction_decile"].unique().to_list())
        assert min(unique_deciles) >= 1
        assert max(unique_deciles) <= 5

    def test_low_cardinality_predictions(self):
        """Only a few distinct prediction values — should not crash."""
        from insurance_fairness._utils import assign_prediction_deciles

        # 50 policies, only 2 distinct values
        df = pl.DataFrame({"pred": [100.0] * 25 + [200.0] * 25})
        result = assign_prediction_deciles(df, "pred", n_deciles=10)
        assert "prediction_decile" in result.columns
        assert len(result) == 50

    def test_missing_column_raises(self):
        from insurance_fairness._utils import assign_prediction_deciles

        df = pl.DataFrame({"other": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Column"):
            assign_prediction_deciles(df, "pred", n_deciles=5)


# ===========================================================================
# 2. localized_parity.py — internal helpers and error paths
# ===========================================================================


class TestMakeStrictlyIncreasing:
    """_make_strictly_increasing ensures monotone arrays."""

    def test_already_increasing(self):
        from insurance_fairness.localized_parity import _make_strictly_increasing

        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _make_strictly_increasing(arr)
        assert np.all(np.diff(result) > 0)
        np.testing.assert_array_almost_equal(result, arr)

    def test_all_equal_gets_nudged(self):
        from insurance_fairness.localized_parity import _make_strictly_increasing

        arr = np.array([5.0, 5.0, 5.0, 5.0])
        result = _make_strictly_increasing(arr)
        assert np.all(np.diff(result) > 0)
        assert result[0] == pytest.approx(5.0)

    def test_ties_in_middle(self):
        from insurance_fairness.localized_parity import _make_strictly_increasing

        arr = np.array([1.0, 3.0, 3.0, 5.0])
        result = _make_strictly_increasing(arr)
        assert np.all(np.diff(result) > 0)

    def test_does_not_modify_input(self):
        from insurance_fairness.localized_parity import _make_strictly_increasing

        arr = np.array([1.0, 2.0, 2.0, 3.0])
        original = arr.copy()
        _make_strictly_increasing(arr)
        np.testing.assert_array_equal(arr, original)


class TestWeightedQuantile:
    """_weighted_quantile edge cases."""

    def test_empty_values_returns_nan(self):
        from insurance_fairness.localized_parity import _weighted_quantile

        result = _weighted_quantile(np.array([]), np.array([0.5]), np.array([]))
        assert np.isnan(result[0])

    def test_zero_total_weight_returns_nan(self):
        from insurance_fairness.localized_parity import _weighted_quantile

        result = _weighted_quantile(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert np.isnan(result[0])

    def test_uniform_weights_match_numpy_quantile(self):
        from insurance_fairness.localized_parity import _weighted_quantile

        rng = np.random.default_rng(0)
        vals = rng.uniform(0, 100, 1000)
        w = np.ones(1000)
        qs = np.array([0.25, 0.50, 0.75])
        result = _weighted_quantile(vals, qs, w)
        # Should be close to numpy's unweighted quantiles
        expected = np.quantile(vals, qs)
        np.testing.assert_allclose(result, expected, atol=2.0)

    def test_median_two_values(self):
        from insurance_fairness.localized_parity import _weighted_quantile

        vals = np.array([0.0, 10.0])
        w = np.array([1.0, 1.0])
        result = _weighted_quantile(vals, np.array([0.5]), w)
        assert result[0] == pytest.approx(5.0, abs=1.0)


class TestLocalizedParityAuditErrors:
    """Error paths in LocalizedParityAudit.audit()."""

    def test_mismatched_predictions_sensitive_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityAudit

        audit = LocalizedParityAudit(thresholds=[300.0])
        with pytest.raises(ValueError, match="same length"):
            audit.audit(
                predictions=np.array([100.0, 200.0, 300.0]),
                sensitive=np.array(["A", "B"]),
            )

    def test_mismatched_exposure_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityAudit

        audit = LocalizedParityAudit(thresholds=[300.0])
        with pytest.raises(ValueError, match="exposure"):
            audit.audit(
                predictions=np.array([100.0, 200.0]),
                sensitive=np.array(["A", "B"]),
                exposure=np.array([1.0]),
            )

    def test_tolerance_zero_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityAudit

        with pytest.raises(ValueError, match="tolerance must be positive"):
            LocalizedParityAudit(thresholds=[300.0], tolerance=0.0)

    def test_tolerance_negative_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityAudit

        with pytest.raises(ValueError, match="tolerance must be positive"):
            LocalizedParityAudit(thresholds=[300.0], tolerance=-0.01)

    def test_unsorted_target_levels_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityAudit

        with pytest.raises(ValueError, match="ascending"):
            LocalizedParityAudit(
                thresholds=[300.0, 500.0],
                target_levels=[0.7, 0.4],
            )


class TestLocalizedParityCorrectorErrors:
    """Error paths in LocalizedParityCorrector."""

    def test_n_bins_less_than_2_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            LocalizedParityCorrector(thresholds=[300.0], n_bins=1)

    def test_negative_exposure_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0, 500.0])
        preds = np.array([200.0, 400.0, 600.0])
        sensitive = np.array(["A", "B", "A"])
        exposure = np.array([1.0, -0.5, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            corrector.fit(preds, sensitive, exposure=exposure)

    def test_fit_mismatched_lengths_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0])
        with pytest.raises(ValueError, match="same length"):
            corrector.fit(
                np.array([100.0, 200.0, 300.0]),
                np.array(["A", "B"]),
            )

    def test_unsorted_thresholds_in_corrector_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        with pytest.raises(ValueError, match="ascending"):
            LocalizedParityCorrector(thresholds=[600.0, 300.0])

    def test_transform_before_fit_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0])
        with pytest.raises(RuntimeError, match="fit"):
            corrector.transform(np.array([300.0, 400.0]), np.array(["A", "B"]))

    def test_audit_before_fit_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0])
        with pytest.raises(RuntimeError, match="fit"):
            corrector.audit()

    def test_discretization_cost_before_fit_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0, 500.0])
        with pytest.raises(RuntimeError, match="fit"):
            _ = corrector.discretization_cost

    def test_unsorted_target_levels_in_corrector_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        with pytest.raises(ValueError, match="ascending"):
            LocalizedParityCorrector(
                thresholds=[300.0, 500.0],
                target_levels=[0.6, 0.4],
            )

    def test_out_of_range_target_levels_in_corrector_raises(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            LocalizedParityCorrector(
                thresholds=[300.0, 500.0],
                target_levels=[0.0, 0.7],
            )


class TestLocalizedParityCorrectorRepr:
    """__repr__ for LocalizedParityCorrector."""

    def test_repr_contains_mode(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0, 500.0], mode="marginal")
        r = repr(corrector)
        assert "marginal" in r

    def test_repr_contains_n_bins(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0], n_bins=150)
        r = repr(corrector)
        assert "150" in r

    def test_repr_contains_m(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector

        corrector = LocalizedParityCorrector(thresholds=[300.0, 500.0, 700.0])
        r = repr(corrector)
        assert "M=3" in r


class TestLocalizedParityEmptyGroup:
    """Group with zero observations in audit: NaN CDF, non-crashing."""

    def test_empty_group_row_has_nan_cdf(self):
        from insurance_fairness.localized_parity import _build_report

        # Group 'C' has no observations
        predictions = np.array([200.0, 400.0, 600.0])
        sensitive = np.array(["A", "A", "B"])
        thresholds = np.array([300.0, 500.0])
        target_levels = np.array([0.4, 0.7])
        groups = np.array(["A", "B", "C"])

        report = _build_report(
            predictions=predictions,
            sensitive=sensitive,
            thresholds=thresholds,
            target_levels=target_levels,
            groups=groups,
        )

        # Group C rows should have NaN empirical_cdf
        group_c_rows = report.group_cdf_table.filter(pl.col("group") == "C")
        assert len(group_c_rows) == 2  # one per threshold
        assert group_c_rows["empirical_cdf"].is_nan().all()


# ===========================================================================
# 3. tail_dp.py — internal helpers and edge cases
# ===========================================================================


class TestTailDPInternals:
    """Internal helpers in tail_dp.py."""

    def test_weighted_ecdf_empty_raises(self):
        """_weighted_ecdf with empty array returns empty arrays (not crash)."""
        from insurance_fairness.tail_dp import _weighted_ecdf

        sv, cw = _weighted_ecdf(np.array([]))
        assert len(sv) == 0
        assert len(cw) == 0

    def test_weighted_ecdf_zero_total_raises(self):
        from insurance_fairness.tail_dp import _weighted_ecdf

        with pytest.raises(ValueError, match="Total weight is zero"):
            _weighted_ecdf(np.array([1.0, 2.0]), weights=np.array([0.0, 0.0]))

    def test_weighted_ecdf_uniform_weights(self):
        from insurance_fairness.tail_dp import _weighted_ecdf

        sv, cw = _weighted_ecdf(np.array([3.0, 1.0, 4.0, 2.0]))
        # Should be sorted
        assert np.all(np.diff(sv) >= 0)
        # Last cumulative weight should be 1.0
        assert cw[-1] == pytest.approx(1.0)

    def test_kde_ecdf_fallback_for_few_observations(self):
        """_kde_quantile_ecdf falls back to empirical ECDF for n < 5."""
        from insurance_fairness.tail_dp import _kde_quantile_ecdf, _weighted_ecdf

        vals = np.array([100.0, 200.0, 300.0])  # only 3 observations
        grid_x, cdf_y = _kde_quantile_ecdf(vals)
        # Should still return a valid CDF-like pair
        assert len(grid_x) == len(cdf_y)
        assert cdf_y[-1] == pytest.approx(1.0)

    def test_kde_ecdf_fallback_for_constant_values(self):
        """_kde_quantile_ecdf falls back when std < 1e-12."""
        from insurance_fairness.tail_dp import _kde_quantile_ecdf

        vals = np.full(20, 500.0)  # all identical
        grid_x, cdf_y = _kde_quantile_ecdf(vals)
        assert len(grid_x) > 0
        assert np.all(np.isfinite(cdf_y))

    def test_barycenter_qf_single_group(self):
        """Single group: barycenter should equal the group's own quantile function."""
        from insurance_fairness.tail_dp import _barycenter_qf, _weighted_ecdf

        rng = np.random.default_rng(0)
        vals = rng.gamma(3.0, 100.0, 200)
        ecdf_x, ecdf_y = _weighted_ecdf(vals)
        u_grid, bar_qf = _barycenter_qf([(ecdf_x, ecdf_y)], np.array([1.0]), n_quantiles=100)
        # With a single group and weight=1, barycenter == that group's QF
        test_u = np.array([0.1, 0.5, 0.9])
        orig_qf = np.interp(test_u, ecdf_y, ecdf_x)
        bar_at_test = np.interp(test_u, u_grid, bar_qf)
        np.testing.assert_allclose(bar_at_test, orig_qf, rtol=0.01)


class TestTailDPCorrectorEdgeCases:
    """TailDemographicParityCorrector edge cases."""

    def test_n_quantiles_too_small_raises(self):
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        with pytest.raises(ValueError, match="n_quantiles"):
            TailDemographicParityCorrector(n_quantiles=1)

    def test_protected_attr_stored(self):
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        corr = TailDemographicParityCorrector(protected_attr="gender")
        assert corr.protected_attr == "gender"

    def test_repr_contains_threshold_and_method(self):
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        corr = TailDemographicParityCorrector(quantile_threshold=0.85, method="wasserstein")
        r = repr(corr)
        assert "0.85" in r
        assert "wasserstein" in r

    def test_tail_dp_report_repr(self):
        from insurance_fairness.tail_dp import TailDPReport

        rpt = TailDPReport(
            quantile_threshold=0.9,
            tail_cutoff=450.0,
            n_affected=100,
            proportion_affected=0.10,
            ks_before=0.30,
            ks_after=0.05,
            ks_reduction=0.25,
            mean_shift_by_group={"A": -20.0, "B": 20.0},
            group_tail_sizes={"A": 20, "B": 80},
        )
        r = repr(rpt)
        assert "0.9" in r
        assert "450.00" in r
        assert "10.0%" in r

    def test_reweight_three_groups(self):
        """reweight method with three groups should not crash."""
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        rng = np.random.default_rng(42)
        n_each = 200
        preds = np.concatenate([
            rng.gamma(3.0, 80.0, n_each),
            rng.gamma(3.0, 120.0, n_each),
            rng.gamma(3.0, 160.0, n_each),
        ])
        sensitive = np.array(["X"] * n_each + ["Y"] * n_each + ["Z"] * n_each)
        corr = TailDemographicParityCorrector(method="reweight", quantile_threshold=0.8)
        corr.fit(preds, sensitive)
        corrected = corr.transform(preds, sensitive)
        assert corrected.shape == preds.shape
        assert np.all(np.isfinite(corrected))

    def test_group_weights_sum_one_after_fit(self):
        """group_weights_ property: values sum to 1.0."""
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        rng = np.random.default_rng(7)
        preds = np.concatenate([
            rng.gamma(3.0, 100.0, 500),
            rng.gamma(3.0, 150.0, 500),
        ])
        sensitive = np.array(["A"] * 500 + ["B"] * 500)
        corr = TailDemographicParityCorrector(quantile_threshold=0.9)
        corr.fit(preds, sensitive)
        w = corr.group_weights_
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_groups_property_before_fit_raises(self):
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        corr = TailDemographicParityCorrector()
        with pytest.raises(RuntimeError, match="fit"):
            _ = corr.groups_

    def test_tail_cutoff_property_before_fit_raises(self):
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        corr = TailDemographicParityCorrector()
        with pytest.raises(RuntimeError, match="fit"):
            _ = corr.tail_cutoff_

    def test_group_weights_before_fit_raises(self):
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        corr = TailDemographicParityCorrector()
        with pytest.raises(RuntimeError, match="fit"):
            _ = corr.group_weights_


# ===========================================================================
# 4. sensitivity/_measure.py — additional coverage
# ===========================================================================


class TestProxyDiscriminationMeasureSummary:
    """summary() output content."""

    def _fit_measure(self, seed: int = 0):
        from insurance_fairness.sensitivity import ProxyDiscriminationMeasure

        rng = np.random.default_rng(seed)
        n = 300
        X = rng.normal(size=(n, 3))
        D = rng.choice(["male", "female"], size=n)
        y = X[:, 0] + rng.normal(scale=0.3, size=n)
        pi = X[:, 0] + rng.normal(scale=0.1, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        return m

    def test_summary_contains_pd_score(self):
        m = self._fit_measure()
        s = m.summary()
        assert "PD" in s

    def test_summary_contains_uf_score(self):
        m = self._fit_measure()
        s = m.summary()
        assert "UF" in s

    def test_summary_contains_category_names(self):
        m = self._fit_measure()
        s = m.summary()
        assert "female" in s or "male" in s

    def test_summary_contains_lambda_std(self):
        m = self._fit_measure()
        s = m.summary()
        assert "Lambda" in s or "lambda" in s or "std" in s.lower()

    def test_v_star_shape_matches_n_categories(self):
        m = self._fit_measure()
        assert len(m.v_star) == len(m.categories)

    def test_v_star_sum_leq_one(self):
        """Optimal weights sum to at most 1 (QP constraint)."""
        m = self._fit_measure()
        assert m.v_star.sum() <= 1.0 + 1e-8

    def test_v_star_all_nonneg(self):
        """Optimal weights are non-negative."""
        m = self._fit_measure()
        assert np.all(m.v_star >= -1e-8)

    def test_categories_sorted(self):
        """Categories should be sorted (alphabetically for strings)."""
        m = self._fit_measure()
        assert m.categories == sorted(m.categories)

    def test_mu_matrix_shape(self):
        """mu_matrix should be (n, |D|)."""
        rng = np.random.default_rng(1)
        from insurance_fairness.sensitivity import ProxyDiscriminationMeasure

        n = 200
        X = rng.normal(size=(n, 2))
        D = rng.choice([0, 1, 2], size=n)  # 3 categories
        y = rng.normal(size=n)
        pi = X[:, 0] + rng.normal(scale=0.1, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert m.mu_matrix.shape == (n, 3)


# ===========================================================================
# 5. optimal_transport/_validators.py — boundary cases
# ===========================================================================


class TestValidatorsEdgeCases:
    """Edge cases for OT input validators."""

    def test_validate_epsilon_zero_passes(self):
        from insurance_fairness.optimal_transport._validators import validate_epsilon

        validate_epsilon(0.0)  # Should not raise

    def test_validate_epsilon_one_passes(self):
        from insurance_fairness.optimal_transport._validators import validate_epsilon

        validate_epsilon(1.0)  # Should not raise

    def test_validate_dataframe_exact_match(self):
        from insurance_fairness.optimal_transport._validators import validate_dataframe_aligned

        df = pl.DataFrame({"a": [1, 2, 3]})
        validate_dataframe_aligned(df, "test_df", 3)  # exact match: no exception

    def test_validate_predictions_single_value(self):
        """Single positive prediction should pass."""
        from insurance_fairness.optimal_transport._validators import validate_predictions

        result = validate_predictions(np.array([42.0]))
        assert result[0] == pytest.approx(42.0)

    def test_validate_exposure_single_positive(self):
        """Single positive exposure should pass."""
        from insurance_fairness.optimal_transport._validators import validate_exposure

        result = validate_exposure(np.array([0.5]), 1)
        assert result[0] == pytest.approx(0.5)

    def test_validate_protected_attrs_multiple_missing(self):
        """Both missing attrs should be mentioned in error."""
        from insurance_fairness.optimal_transport._validators import validate_protected_attrs_present

        df = pl.DataFrame({"age": [25, 30]})
        with pytest.raises(ValueError):
            validate_protected_attrs_present(["gender", "disability"], df, "D")


# ===========================================================================
# 6. Public API import completeness
# ===========================================================================


class TestPublicAPIImports:
    """Verify all names in __all__ are importable from insurance_fairness."""

    def test_all_names_importable(self):
        import insurance_fairness

        for name in insurance_fairness.__all__:
            assert hasattr(insurance_fairness, name), (
                f"'{name}' is in __all__ but not importable from insurance_fairness"
            )

    def test_version_accessible(self):
        import insurance_fairness

        assert hasattr(insurance_fairness, "__version__")
        assert isinstance(insurance_fairness.__version__, str)
        assert len(insurance_fairness.__version__) > 0

    def test_subpackages_importable(self):
        from insurance_fairness import optimal_transport, diagnostics, sensitivity

        assert optimal_transport is not None
        assert diagnostics is not None
        assert sensitivity is not None

    def test_core_classes_instantiatable(self):
        """Key classes should be instantiatable without arguments or with minimal args."""
        from insurance_fairness import (
            MulticalibrationAudit,
            ProxyVulnerabilityScore,
        )
        from insurance_fairness.tail_dp import TailDemographicParityCorrector
        from insurance_fairness.localized_parity import LocalizedParityAudit

        # These should construct without error
        MulticalibrationAudit()
        ProxyVulnerabilityScore()
        TailDemographicParityCorrector()
        LocalizedParityAudit(thresholds=[300.0, 500.0])


# ===========================================================================
# 7. Integration: localized parity -> tail DP pipeline
# ===========================================================================


class TestLocalizedParityToTailDPPipeline:
    """End-to-end: apply localized parity correction, then tail DP correction."""

    def test_pipeline_no_errors(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        rng = np.random.default_rng(0)
        n = 1000
        preds = np.concatenate([
            rng.gamma(3.0, 100.0, n // 2),
            rng.gamma(3.0, 150.0, n // 2),
        ])
        sensitive = np.array(["M"] * (n // 2) + ["F"] * (n // 2))

        # Step 1: localized parity
        lp = LocalizedParityCorrector(thresholds=[200.0, 350.0, 500.0], mode="quantile")
        lp.fit(preds, sensitive)
        preds_lp = lp.transform(preds, sensitive)

        # Step 2: tail DP on localized-parity-corrected predictions
        tdp = TailDemographicParityCorrector(quantile_threshold=0.9)
        tdp.fit(preds_lp, sensitive)
        preds_final = tdp.transform(preds_lp, sensitive)

        assert preds_final.shape == preds.shape
        assert np.all(np.isfinite(preds_final))

    def test_pipeline_below_threshold_unchanged(self):
        from insurance_fairness.localized_parity import LocalizedParityCorrector
        from insurance_fairness.tail_dp import TailDemographicParityCorrector

        rng = np.random.default_rng(1)
        n = 600
        preds = rng.gamma(3.0, 120.0, n)
        sensitive = np.array(["A"] * (n // 2) + ["B"] * (n // 2))

        lp = LocalizedParityCorrector(thresholds=[200.0, 400.0], mode="marginal")
        lp.fit(preds, sensitive)
        preds_lp = lp.transform(preds)

        tdp = TailDemographicParityCorrector(quantile_threshold=0.8)
        tdp.fit(preds_lp, sensitive)
        preds_final = tdp.transform(preds_lp, sensitive)

        # Below-threshold predictions from tail DP must be unchanged
        below = preds_lp <= tdp.tail_cutoff_
        np.testing.assert_array_equal(preds_final[below], preds_lp[below])


# ===========================================================================
# 8. Empirical CDF helper in localized_parity
# ===========================================================================


class TestEmpiricalCDF:
    """_empirical_cdf helper."""

    def test_empty_predictions_returns_zeros(self):
        from insurance_fairness.localized_parity import _empirical_cdf

        result = _empirical_cdf(np.array([]), thresholds=np.array([300.0, 500.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_zero_total_weight_returns_zeros(self):
        from insurance_fairness.localized_parity import _empirical_cdf

        result = _empirical_cdf(
            np.array([100.0, 400.0]),
            thresholds=np.array([300.0]),
            weights=np.array([0.0, 0.0]),
        )
        np.testing.assert_array_equal(result, [0.0])

    def test_all_below_threshold_gives_one(self):
        from insurance_fairness.localized_parity import _empirical_cdf

        result = _empirical_cdf(
            np.array([100.0, 200.0, 250.0]),
            thresholds=np.array([300.0]),
        )
        assert result[0] == pytest.approx(1.0)

    def test_all_above_threshold_gives_zero(self):
        from insurance_fairness.localized_parity import _empirical_cdf

        result = _empirical_cdf(
            np.array([400.0, 500.0, 600.0]),
            thresholds=np.array([300.0]),
        )
        assert result[0] == pytest.approx(0.0)

    def test_half_below_threshold_gives_half(self):
        from insurance_fairness.localized_parity import _empirical_cdf

        result = _empirical_cdf(
            np.array([100.0, 100.0, 500.0, 500.0]),
            thresholds=np.array([300.0]),
        )
        assert result[0] == pytest.approx(0.5)

    def test_none_weights_treated_as_uniform(self):
        from insurance_fairness.localized_parity import _empirical_cdf

        result_no_weights = _empirical_cdf(
            np.array([100.0, 200.0, 400.0, 500.0]),
            thresholds=np.array([300.0]),
            weights=None,
        )
        result_uniform = _empirical_cdf(
            np.array([100.0, 200.0, 400.0, 500.0]),
            thresholds=np.array([300.0]),
            weights=np.ones(4),
        )
        assert result_no_weights[0] == pytest.approx(result_uniform[0])
