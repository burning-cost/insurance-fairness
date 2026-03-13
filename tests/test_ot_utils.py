"""Tests for utility functions."""
import numpy as np
import pytest

from insurance_fairness.optimal_transport._utils import (
    apply_ot_correction,
    barycenter_quantile,
    exposure_weighted_ecdf,
    quantile_function,
    wasserstein_distance_1d,
)


class TestExposureWeightedECDF:
    def test_uniform_exposure_matches_empirical_ecdf(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        exposure = np.ones(4)
        x, y = exposure_weighted_ecdf(values, exposure)
        assert np.allclose(y[-1], 1.0)
        assert np.all(np.diff(y) >= 0)

    def test_sorted_output(self):
        values = np.array([3.0, 1.0, 4.0, 2.0])
        exposure = np.ones(4)
        x, y = exposure_weighted_ecdf(values, exposure)
        assert np.all(np.diff(x) >= 0)

    def test_weighted_concentrates_on_high_exposure(self):
        # values [1, 2, 3, 4], high exposure on value 4
        values = np.array([1.0, 2.0, 3.0, 4.0])
        exposure = np.array([0.1, 0.1, 0.1, 10.0])
        x, y = exposure_weighted_ecdf(values, exposure)
        # Most of the CDF mass should be at value 4
        assert y[2] < 0.5  # value 3 < median

    def test_zero_exposure_raises(self):
        with pytest.raises(ValueError, match="zero"):
            exposure_weighted_ecdf(np.array([1.0, 2.0]), np.zeros(2))

    def test_single_value(self):
        x, y = exposure_weighted_ecdf(np.array([5.0]), np.array([1.0]))
        assert x[0] == 5.0
        assert y[0] == 1.0


class TestQuantileFunction:
    def test_basic_interpolation(self):
        # Uniform [0,1]
        ecdf_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        ecdf_y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = quantile_function(ecdf_x, ecdf_y, np.array([0.5]))
        assert np.isclose(result[0], 0.5)

    def test_quantile_monotone(self):
        rng = np.random.default_rng(0)
        vals = rng.exponential(1, 200)
        x, y = exposure_weighted_ecdf(vals, np.ones(200))
        u = np.linspace(0.01, 0.99, 50)
        qf = quantile_function(x, y, u)
        assert np.all(np.diff(qf) >= 0)

    def test_clamp_at_edges(self):
        ecdf_x = np.array([1.0, 2.0, 3.0])
        ecdf_y = np.array([0.3, 0.7, 1.0])
        # u = 0 should return minimum value
        result = quantile_function(ecdf_x, ecdf_y, np.array([0.0]))
        assert result[0] == ecdf_x[0]


class TestBarycenterQuantile:
    def test_identical_distributions_unchanged(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1, 500)
        x, y = exposure_weighted_ecdf(vals, np.ones(500))
        ecdfs = [(x, y), (x, y)]
        weights = np.array([0.5, 0.5])
        u_grid, bar = barycenter_quantile(ecdfs, weights)
        # Barycenter of identical distributions should equal the distribution
        test_u = np.array([0.1, 0.5, 0.9])
        orig = quantile_function(x, y, test_u)
        bar_q = quantile_function(np.linspace(min(x), max(x), len(u_grid)), u_grid, test_u)
        # Check they're close — use direct evaluation
        bar_at_test = np.interp(test_u, u_grid, bar)
        assert np.allclose(orig, bar_at_test, atol=0.1)

    def test_barycenter_between_extremes(self):
        # Two distributions: N(0,1) and N(2,1). Barycenter should be N(1,1).
        rng = np.random.default_rng(99)
        v0 = rng.normal(0, 1, 1000)
        v1 = rng.normal(2, 1, 1000)
        e0 = np.ones(1000)
        e1 = np.ones(1000)
        x0, y0 = exposure_weighted_ecdf(v0, e0)
        x1, y1 = exposure_weighted_ecdf(v1, e1)
        ecdfs = [(x0, y0), (x1, y1)]
        weights = np.array([0.5, 0.5])
        u_grid, bar = barycenter_quantile(ecdfs, weights, n_quantiles=500)
        median_bar = float(np.interp(0.5, u_grid, bar))
        # Median of barycenter should be ~1.0
        assert abs(median_bar - 1.0) < 0.15

    def test_weights_normalised(self):
        rng = np.random.default_rng(7)
        v = rng.normal(0, 1, 200)
        x, y = exposure_weighted_ecdf(v, np.ones(200))
        ecdfs = [(x, y)]
        # Un-normalised weights — should still work
        u_grid, bar = barycenter_quantile(ecdfs, np.array([5.0]))
        assert np.all(np.isfinite(bar))


class TestWassersteinDistance:
    def test_identical_distributions_zero(self):
        rng = np.random.default_rng(5)
        v = rng.normal(0, 1, 300)
        w2 = wasserstein_distance_1d(v, v)
        assert w2 < 0.01

    def test_shifted_distributions(self):
        rng = np.random.default_rng(6)
        v0 = rng.normal(0, 1, 500)
        v1 = rng.normal(1, 1, 500)
        w2 = wasserstein_distance_1d(v0, v1)
        # W2 of N(0,1) vs N(1,1) is 1.0
        assert abs(w2 - 1.0) < 0.1

    def test_exposure_weighted(self):
        rng = np.random.default_rng(8)
        v0 = rng.normal(0, 1, 100)
        v1 = rng.normal(2, 1, 100)
        w0 = np.ones(100)
        w1 = np.ones(100)
        w2 = wasserstein_distance_1d(v0, v1, w0, w1)
        assert w2 > 0


class TestApplyOTCorrection:
    def test_identity_when_same_ecdf_for_all(self):
        rng = np.random.default_rng(10)
        predictions = rng.lognormal(0, 0.5, 200)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        # Use the same ECDF for both groups — correction should be identity
        vals = np.log(predictions)
        x_all, y_all = exposure_weighted_ecdf(vals, np.ones(200))
        # Barycenter of two identical distributions is itself
        u_grid, bar_qf = barycenter_quantile([(x_all, y_all), (x_all, y_all)], np.array([0.5, 0.5]))
        ecdfs = {"A": (x_all, y_all), "B": (x_all, y_all)}
        corrected = apply_ot_correction(predictions, groups, ecdfs, u_grid, bar_qf, log_space=True)
        # Should be approximately unchanged
        assert np.allclose(np.log(corrected), np.log(predictions), atol=0.1)

    def test_corrects_distribution_shape(self):
        rng = np.random.default_rng(11)
        pA = rng.lognormal(0, 0.3, 200)
        pB = rng.lognormal(0.5, 0.3, 200)
        predictions = np.concatenate([pA, pB])
        groups = np.array(["A"] * 200 + ["B"] * 200)
        xA, yA = exposure_weighted_ecdf(np.log(pA), np.ones(200))
        xB, yB = exposure_weighted_ecdf(np.log(pB), np.ones(200))
        ecdfs = {"A": (xA, yA), "B": (xB, yB)}
        wA = 200 / 400
        wB = 200 / 400
        u_grid, bar_qf = barycenter_quantile([(xA, yA), (xB, yB)], np.array([wA, wB]))
        corrected = apply_ot_correction(predictions, groups, ecdfs, u_grid, bar_qf, log_space=True)
        # After correction, mean of A and mean of B should be closer
        mean_before = abs(np.mean(pA) - np.mean(pB))
        mean_after = abs(np.mean(corrected[:200]) - np.mean(corrected[200:]))
        assert mean_after < mean_before
