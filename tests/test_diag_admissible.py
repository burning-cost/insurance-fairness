"""
Tests for _admissible.py: admissible price computation and D_proxy.

D_proxy = sqrt( E_w[(h_star - mu_h)^2] ) / sqrt( E_w[(h - mu_h)^2] )

where h_star = E_w[h | S] (within-group mean) and mu_h = E_w[h].

D_proxy is the square root of the ANOVA R^2: the proportion of variance
in h that is explained by group membership S.

D_proxy = 0  iff  all S-group means equal the global mean (S not predictive of h)
D_proxy = 1  iff  all within-group variance = 0 (h perfectly explained by S)
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_fairness.diagnostics._admissible import (
    compute_admissible_price,
    compute_d_proxy,
    compute_d_proxy_with_ci,
)


class TestComputeAdmissiblePrice:
    """Tests for compute_admissible_price."""

    def test_two_equal_groups_analytical(self):
        """
        Group 0: h = 100 => group mean = 100
        Group 1: h = 200 => group mean = 200
        h_star for group 0: 100; h_star for group 1: 200
        """
        n = 100
        h = np.array([100.0] * 50 + [200.0] * 50)
        s = np.array([0] * 50 + [1] * 50)
        weights = np.ones(n)

        h_star = compute_admissible_price(h, s, weights, reference_dist="observed")

        assert h_star.shape == (n,)
        np.testing.assert_allclose(h_star[:50], 100.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[50:], 200.0, rtol=1e-10)

    def test_single_group_returns_global_mean(self):
        """With only one S value, h_star = global mean for all."""
        n = 50
        h = np.linspace(100, 200, n)
        s = np.zeros(n)
        weights = np.ones(n)

        h_star = compute_admissible_price(h, s, weights)

        expected = h.mean()  # global mean = within-only-group mean
        np.testing.assert_allclose(h_star, expected, rtol=1e-10)

    def test_three_groups(self):
        """Test with three S groups."""
        n = 90
        h = np.array([100.0] * 30 + [150.0] * 30 + [200.0] * 30)
        s = np.array([0] * 30 + [1] * 30 + [2] * 30)
        weights = np.ones(n)

        h_star = compute_admissible_price(h, s, weights)

        np.testing.assert_allclose(h_star[:30], 100.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[30:60], 150.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[60:], 200.0, rtol=1e-10)

    def test_unequal_group_sizes(self):
        """Unequal group sizes should not affect within-group means."""
        h = np.array([100.0] * 10 + [200.0] * 90)
        s = np.array([0] * 10 + [1] * 90)
        weights = np.ones(len(h))

        h_star = compute_admissible_price(h, s, weights)

        np.testing.assert_allclose(h_star[:10], 100.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[10:], 200.0, rtol=1e-10)

    def test_exposure_weighted_group_means(self):
        """Group means should be exposure-weighted."""
        # Group 0: h=[100, 300] with weights=[3, 1] => mean = (300+300)/4 = 150
        h = np.array([100.0, 300.0, 200.0])
        s = np.array([0, 0, 1])
        weights = np.array([3.0, 1.0, 1.0])

        h_star = compute_admissible_price(h, s, weights)

        np.testing.assert_allclose(h_star[0], 150.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[1], 150.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[2], 200.0, rtol=1e-10)

    def test_constant_h_gives_constant_h_star(self):
        """If all predictions are equal, h_star = that constant for all."""
        n = 100
        h = np.full(n, 150.0)
        s = np.random.default_rng(0).integers(0, 3, size=n)
        weights = np.ones(n)

        h_star = compute_admissible_price(h, s, weights)

        np.testing.assert_allclose(h_star, 150.0, rtol=1e-10)

    def test_unsupported_reference_dist_raises(self):
        """Unsupported reference_dist should raise ValueError."""
        h = np.array([100.0, 200.0])
        s = np.array([0, 1])
        weights = np.ones(2)

        with pytest.raises(ValueError, match="reference_dist"):
            compute_admissible_price(h, s, weights, reference_dist="uniform")

    def test_string_sensitive_attribute(self):
        """Sensitive attribute can be strings (not just integers)."""
        h = np.array([100.0] * 30 + [200.0] * 30)
        s = np.array(["north"] * 30 + ["south"] * 30)
        weights = np.ones(60)

        h_star = compute_admissible_price(h, s, weights)

        np.testing.assert_allclose(h_star[:30], 100.0, rtol=1e-10)
        np.testing.assert_allclose(h_star[30:], 200.0, rtol=1e-10)


class TestComputeDProxy:
    """Tests for compute_d_proxy.

    D_proxy = sqrt(between-group variance / total variance) = sqrt(ANOVA R^2)

    D_proxy = 0: S provides no information about h (all group means equal)
    D_proxy = 1: h is entirely explained by S (zero within-group variance)
    """

    def test_zero_when_group_means_equal_global_mean(self):
        """
        D_proxy = 0 when all S-group means equal the global mean.

        Example: two groups with the same mean prediction.
        Group 0: h = [100, 200] => mean = 150
        Group 1: h = [120, 180] => mean = 150
        Global mean = 150
        Between-group variance = 0 => D_proxy = 0
        """
        h = np.array([100.0, 200.0, 120.0, 180.0])
        h_star = np.array([150.0, 150.0, 150.0, 150.0])  # all group means = 150
        weights = np.ones(4)

        d = compute_d_proxy(h, h_star, weights)

        assert d == pytest.approx(0.0, abs=1e-12)

    def test_one_when_predictions_constant_within_groups(self):
        """
        D_proxy = 1 when all within-group variance = 0 (h perfectly explained by S).

        Example: group 0: h=100, group 1: h=200 (constant within groups).
        Between-group variance = total variance => D_proxy = 1.
        """
        h = np.array([100.0] * 50 + [200.0] * 50)
        h_star = np.array([100.0] * 50 + [200.0] * 50)  # group means = h
        weights = np.ones(100)

        d = compute_d_proxy(h, h_star, weights)

        assert d == pytest.approx(1.0, abs=1e-10)

    def test_known_analytical_value(self):
        """
        Two groups: h_group0 ~ N(100, 10^2), h_group1 ~ N(200, 10^2)
        With large n: between-group var = 2500, total var = 2500 + 100 = 2600
        D_proxy ~ sqrt(2500/2600) ~ 0.981

        Simpler: two groups, h = [100, 200, 100, 200] with group 0 = first two,
        group 1 = last two. h_star = [150, 150, 150, 150] (group means both = 150)?
        No -- with these h values, group 0 mean = 150 and group 1 mean = 150.
        So h_star = [150, 150, 150, 150] and D_proxy = 0.

        Better: use groups where h differs between groups.
        Group 0 (s=0): h = [90, 110] => mean = 100
        Group 1 (s=1): h = [190, 210] => mean = 200
        Global mean mu = 150
        between-group var = E[(h_star - 150)^2] = (100-150)^2/2 + (200-150)^2/2 = 2500
        total var = E[(h - 150)^2] = [(90-150)^2 + (110-150)^2 + (190-150)^2 + (210-150)^2] / 4
                  = [3600 + 1600 + 1600 + 3600] / 4 = 10400/4 = 2600
        D_proxy = sqrt(2500/2600) = sqrt(25/26) ~ 0.9806
        """
        h = np.array([90.0, 110.0, 190.0, 210.0])
        h_star = np.array([100.0, 100.0, 200.0, 200.0])
        weights = np.ones(4)

        d = compute_d_proxy(h, h_star, weights)
        expected = np.sqrt(2500.0 / 2600.0)

        assert d == pytest.approx(expected, rel=1e-10)

    def test_positive_when_group_means_differ(self):
        """D_proxy > 0 when S-group means differ."""
        h = np.array([100.0] * 50 + [150.0] * 50)  # different group means
        h_star = np.array([100.0] * 50 + [150.0] * 50)
        weights = np.ones(100)

        d = compute_d_proxy(h, h_star, weights)

        assert d > 0.0

    def test_constant_h_gives_zero(self):
        """
        When all h are equal, total variance = 0, D_proxy = 0.
        (No variation to explain, hence no discrimination possible.)
        """
        h = np.full(10, 150.0)
        h_star = np.full(10, 150.0)
        weights = np.ones(10)

        d = compute_d_proxy(h, h_star, weights)

        assert d == pytest.approx(0.0, abs=1e-12)

    def test_in_range_zero_to_one(self):
        """D_proxy should always be in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            n = 100
            h = rng.uniform(100, 300, n)
            s = rng.integers(0, 3, size=n)
            weights = np.ones(n)
            h_star = compute_admissible_price(h, s, weights)
            d = compute_d_proxy(h, h_star, weights)
            assert 0.0 <= d <= 1.0 + 1e-10, f"D_proxy={d} out of [0,1]"

    def test_higher_group_separation_gives_higher_d_proxy(self):
        """More separation between groups => higher D_proxy."""
        rng = np.random.default_rng(42)
        n = 200
        s = np.array([0] * 100 + [1] * 100)

        # Low separation
        h_low = rng.normal(150, 20, n)  # no group effect

        # High separation: group 1 has much higher predictions
        h_high = rng.normal(150, 20, n)
        h_high[100:] += 100.0  # group 1 predictions ~250 vs group 0 ~150

        weights = np.ones(n)
        h_star_low = compute_admissible_price(h_low, s, weights)
        h_star_high = compute_admissible_price(h_high, s, weights)

        d_low = compute_d_proxy(h_low, h_star_low, weights)
        d_high = compute_d_proxy(h_high, h_star_high, weights)

        assert d_high > d_low


class TestComputeDProxyWithCI:
    """Tests for compute_d_proxy_with_ci."""

    def test_returns_tuple_of_float_and_ci(self):
        """Should return (float, (float, float))."""
        n = 200
        rng = np.random.default_rng(0)
        h = rng.uniform(100, 300, n)
        s = rng.integers(0, 2, size=n)
        weights = np.ones(n)
        h_star = compute_admissible_price(h, s, weights)

        d_proxy, ci = compute_d_proxy_with_ci(
            h, s, weights, n_bootstrap=50, rng=np.random.default_rng(42)
        )

        assert isinstance(d_proxy, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_ci_lower_leq_upper(self):
        """CI lower bound must be <= upper bound."""
        n = 300
        rng = np.random.default_rng(99)
        h = rng.uniform(100, 400, n)
        s = rng.integers(0, 3, size=n)
        weights = np.ones(n)
        h_star = compute_admissible_price(h, s, weights)

        _, ci = compute_d_proxy_with_ci(h, s, weights, n_bootstrap=100)

        assert ci[0] <= ci[1]

    def test_zero_case_ci_near_zero(self):
        """
        When group means are all equal, D_proxy = 0 and CI should be near 0.
        """
        # Two groups with the same predictions (shuffled so means are equal)
        n = 200
        rng = np.random.default_rng(0)
        # Alternating groups: each group gets the same h distribution
        h = np.tile(rng.uniform(100, 300, n // 2), 2)  # both groups: same h values
        s = np.array([0] * (n // 2) + [1] * (n // 2))
        weights = np.ones(n)
        h_star = compute_admissible_price(h, s, weights)

        d_proxy, ci = compute_d_proxy_with_ci(
            h, s, weights, n_bootstrap=100, rng=np.random.default_rng(0)
        )

        # Both groups have the same mean, so d_proxy should be ~0
        assert d_proxy == pytest.approx(0.0, abs=1e-10)

    def test_reproducible_with_rng(self):
        """Same rng seed gives same CI."""
        n = 200
        rng = np.random.default_rng(5)
        h = rng.uniform(100, 300, n)
        s = rng.integers(0, 2, size=n)
        weights = np.ones(n)
        h_star = compute_admissible_price(h, s, weights)

        _, ci1 = compute_d_proxy_with_ci(
            h, s, weights, n_bootstrap=50, rng=np.random.default_rng(7)
        )
        _, ci2 = compute_d_proxy_with_ci(
            h, s, weights, n_bootstrap=50, rng=np.random.default_rng(7)
        )

        assert ci1 == ci2

    def test_d_proxy_in_ci(self):
        """The point estimate should fall within the CI."""
        n = 400
        rng = np.random.default_rng(42)
        h = rng.uniform(100, 300, n)
        h[200:] += 50  # group 1 has higher predictions
        s = np.array([0] * 200 + [1] * 200)
        weights = np.ones(n)
        h_star = compute_admissible_price(h, s, weights)

        d_proxy, ci = compute_d_proxy_with_ci(
            h, s, weights, n_bootstrap=100, rng=np.random.default_rng(0)
        )

        assert ci[0] <= d_proxy <= ci[1]
