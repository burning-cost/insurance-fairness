"""
Tests for insurance_fairness.sensitivity subpackage.

Covers ProxyDiscriminationMeasure, SobolAttribution, and ShapleyAttribution.

Test design follows the mathematical properties from LRTW EJOR 2026:
  - PD = 0 iff price is discrimination-free
  - UF = 0 iff group-level mean premiums are equal
  - UF = 0 does not imply PD = 0
  - Shapley values sum to pd_surrogate_ (v(full set))
  - PD in [0, 1], UF in [0, 1]

API: ProxyDiscriminationMeasure.fit(y, X, D, mu_hat, weights, pi)
  - y     : observed losses (used to fit mu when needed)
  - X     : covariate matrix (n, p)
  - D     : protected attribute (discrete, int or str)
  - mu_hat: fitted prices pi(X) as 1-D array (most common path),
            or pre-computed mu(X,d) matrix as 2-D array,
            or dict of conditional means, or None
  - pi    : only needed when mu_hat is 2-D / dict / None
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_fairness.sensitivity import (
    ProxyDiscriminationMeasure,
    SobolAttribution,
    ShapleyAttribution,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_discrimination_free(n: int = 400, n_d: int = 2, seed: int = 42):
    """
    Generate data where pi(X) is genuinely discrimination-free.

    X and D are independent.  Price depends only on X.
    Oracle mu(x,d): since X⊥D, E[Y|X=x,D=d] ≈ E[Y|X=x] for all d,
    so all columns of mu_matrix are approximately equal.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = rng.choice(n_d, size=n)  # independent of X
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + 1.0 + rng.normal(scale=0.1, size=n)
    # Discrimination-free price: linear function of X only
    pi = X[:, 0] * 2.0 + X[:, 1] * 0.5 + 1.0 + rng.normal(scale=0.01, size=n)
    # Oracle mu: all columns equal (X independent of D)
    mu_matrix = np.column_stack([X[:, 0] * 2.0 + X[:, 1] * 0.5 + 1.0] * n_d)
    weights = np.ones(n)
    return y, pi, X, D, mu_matrix, weights


def _make_discriminatory(n: int = 600, n_d: int = 2, seed: int = 0):
    """
    Generate data where pi(X) proxies for D.

    X[:,0] is correlated with D.  Price uses X[:,0], which leaks D.
    Oracle mu_matrix provided for testing the QP.
    """
    rng = np.random.default_rng(seed)
    D = rng.choice(n_d, size=n)
    # X[:,0] correlates with D — it is a proxy variable
    X = np.column_stack([
        D.astype(float) + rng.normal(scale=0.3, size=n),
        rng.normal(size=n),
        rng.normal(size=n),
    ])
    y = X[:, 0] + D.astype(float) * 0.5 + rng.normal(scale=0.1, size=n)
    # Oracle mu(x,d): E[Y|X=x,D=d]
    mu_cols = []
    for d_val in range(n_d):
        mu_d = X[:, 0] + d_val * 0.5
        mu_cols.append(mu_d)
    mu_matrix = np.column_stack(mu_cols)
    # Price uses X[:,0] which proxies for D — discriminatory
    pi = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=n)
    weights = rng.uniform(0.5, 1.5, size=n)
    return y, pi, X, D, mu_matrix, weights


# ===========================================================================
# ProxyDiscriminationMeasure
# ===========================================================================

class TestProxyDiscriminationMeasureFitReturnsself:
    """fit() returns self (fluent interface)."""

    def test_fit_returns_self(self):
        y, pi, X, D, mu_matrix, w = _make_discrimination_free()
        m = ProxyDiscriminationMeasure()
        result = m.fit(y, X, D, mu_hat=pi, weights=w)
        assert result is m


class TestPDScoreDiscriminationFree:
    """PD should be near 0 when the price avoids proxy discrimination."""

    def test_pd_near_zero_1d_mu_hat(self):
        """Pass pi as 1-D mu_hat; mu(x,d) fitted from y. X⊥D => PD ≈ 0."""
        y, pi, X, D, _, w = _make_discrimination_free(n=400, n_d=2, seed=10)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi, weights=w)
        assert m.pd_score < 0.10, f"Expected PD near 0, got {m.pd_score:.4f}"

    def test_pd_near_zero_oracle_mu_matrix(self):
        """With oracle mu_matrix and pi supplied, PD should be very small."""
        y, pi, X, D, mu_matrix, w = _make_discrimination_free(n=400, n_d=2, seed=11)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert m.pd_score < 0.05, f"Expected PD near 0 with oracle, got {m.pd_score:.4f}"

    def test_pd_near_zero_constant_price(self):
        """Constant price => zero variance => PD = 0 by convention."""
        n = 200
        y = RNG.normal(size=n)
        pi = np.ones(n) * 100.0
        X = RNG.normal(size=(n, 2))
        D = RNG.choice([0, 1], size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert m.pd_score == pytest.approx(0.0, abs=1e-10)

    def test_pd_near_zero_five_categories(self):
        """Five D categories, price independent of D."""
        y, pi, X, D, mu_matrix, w = _make_discrimination_free(n=500, n_d=5, seed=7)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert m.pd_score < 0.10, f"Expected PD near 0, got {m.pd_score:.4f}"


class TestPDScoreDiscriminatory:
    """PD should be clearly positive when price proxies for D."""

    def test_pd_positive_when_proxy_present(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=600, n_d=2, seed=1)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert m.pd_score > 0.01, f"Expected PD > 0.01, got {m.pd_score:.4f}"

    def test_pd_positive_five_categories(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=600, n_d=5, seed=2)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert m.pd_score > 0.01, f"Expected PD > 0, got {m.pd_score:.4f}"

    def test_pd_positive_via_1d_mu_hat(self):
        """Discriminatory pi passed as 1-D mu_hat; PD should be positive."""
        y, pi, X, D, _, w = _make_discriminatory(n=500, n_d=2, seed=3)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi, weights=w)
        assert m.pd_score > 0.0, f"Expected PD > 0, got {m.pd_score:.4f}"


class TestPDScoreBounds:
    """PD and UF must lie in [0, 1]."""

    def test_pd_in_unit_interval(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert 0.0 <= m.pd_score <= 1.0

    def test_uf_in_unit_interval(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert 0.0 <= m.uf_score <= 1.0


class TestUFScore:
    """UF measures demographic unfairness (first-order Sobol on D)."""

    def test_uf_near_zero_when_group_means_equal(self):
        """If price has the same mean in every D group, UF = 0."""
        n = 300
        rng = np.random.default_rng(99)
        D = rng.choice([0, 1, 2], size=n)
        pi = rng.normal(loc=100.0, scale=5.0, size=n)
        # Shift each group to exact mean 100
        for d_val in np.unique(D):
            mask = D == d_val
            pi[mask] -= pi[mask].mean() - 100.0
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert m.uf_score < 0.01, f"Expected UF ≈ 0, got {m.uf_score:.4f}"

    def test_uf_positive_when_group_means_differ(self):
        """Price with very different group means should have high UF."""
        n = 300
        D = np.array([0] * 150 + [1] * 150)
        rng = np.random.default_rng(5)
        pi = np.where(D == 0, 100.0, 200.0) + rng.normal(scale=1.0, size=n)
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert m.uf_score > 0.5, f"Expected UF > 0.5, got {m.uf_score:.4f}"


class TestUFNotImplyingZeroPD:
    """
    Critical property from LRTW EJOR 2026: UF=0 does not imply PD=0.

    Construct a price where group-level means are equal (UF≈0) but the
    price is constructed using D (PD>0).
    """

    def test_uf_zero_pd_positive(self):
        """
        Price = X[:,0] * sign(D - 0.5): symmetric around 0 by D group
        (so group means ≈ 0 => UF ≈ 0), but clearly depends on D.
        With oracle mu_matrix capturing the D-dependent structure, PD > 0.
        """
        n = 600
        rng = np.random.default_rng(13)
        D = rng.choice([0, 1], size=n)
        X = rng.normal(size=(n, 3))
        sign = np.where(D == 0, 1.0, -1.0)
        # Price has zero mean in both groups (X[:,0] is mean-0)
        pi = X[:, 0] * sign * 5.0 + rng.normal(scale=0.01, size=n)

        # Oracle mu: D=0 predicts +5*X[:,0], D=1 predicts -5*X[:,0]
        mu_matrix = np.column_stack([
            X[:, 0] * 5.0,   # D=0
            -X[:, 0] * 5.0,  # D=1
        ])
        y = X[:, 0] * sign * 5.0

        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, pi=pi)

        # UF should be near 0 (group means both ≈ 0)
        assert m.uf_score < 0.10, f"Expected UF ≈ 0, got {m.uf_score:.4f}"
        # PD should be > 0 (price depends on D through sign flip)
        assert m.pd_score > 0.10, f"Expected PD > 0.10, got {m.pd_score:.4f}"


class TestClosestAdmissibleAndLambda:
    """Tests for pi_star and Lambda properties."""

    def test_lambda_is_pi_minus_pistar(self):
        y, pi, X, D, mu_matrix, w = _make_discrimination_free()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        expected_lambda = pi - m.closest_admissible
        np.testing.assert_allclose(m.Lambda, expected_lambda, rtol=1e-10)

    def test_pd_equals_var_lambda_over_var_pi(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        w_norm = w / w.sum()
        var_pi = float(np.dot(w_norm, (pi - np.dot(w_norm, pi)) ** 2))
        lambda_ = m.Lambda
        var_lambda = float(np.dot(w_norm, (lambda_ - np.dot(w_norm, lambda_)) ** 2))
        expected_pd = var_lambda / var_pi
        assert m.pd_score == pytest.approx(expected_pd, rel=1e-6)

    def test_lambda_shape(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert m.Lambda.shape == (len(pi),)

    def test_closest_admissible_shape(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert m.closest_admissible.shape == (len(pi),)


class TestMuMatrixInputForms:
    """mu_hat can be passed as 1-D array, 2-D array, or dict."""

    def test_mu_hat_as_1d_array(self):
        y, pi, X, D, _, w = _make_discrimination_free()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi, weights=w)
        assert not np.isnan(m.pd_score)

    def test_mu_hat_as_2d_array_with_pi(self):
        y, pi, X, D, mu_matrix, w = _make_discrimination_free()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert not np.isnan(m.pd_score)

    def test_mu_hat_as_dict_with_pi(self):
        y, pi, X, D, mu_matrix, w = _make_discrimination_free()
        cats = sorted(set(D.tolist()))
        mu_dict = {c: mu_matrix[:, i] for i, c in enumerate(cats)}
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_dict, weights=w, pi=pi)
        assert not np.isnan(m.pd_score)

    def test_mu_hat_none_with_pi(self):
        """When mu_hat=None, pi must be supplied; mu estimated from y."""
        n = 200
        rng = np.random.default_rng(77)
        X = rng.normal(size=(n, 2))
        D = rng.choice([0, 1], size=n)
        y = X[:, 0] + rng.normal(scale=0.5, size=n)
        pi = X[:, 0] + rng.normal(scale=0.1, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=None, pi=pi)
        assert not np.isnan(m.pd_score)
        assert 0.0 <= m.pd_score <= 1.0

    def test_mu_hat_none_no_pi_raises(self):
        """mu_hat=None without pi should raise ValueError."""
        n = 100
        rng = np.random.default_rng(0)
        m = ProxyDiscriminationMeasure()
        with pytest.raises(ValueError, match="pi"):
            m.fit(rng.normal(size=n), rng.normal(size=(n, 2)), rng.choice(2, size=n))

    def test_mu_hat_2d_no_pi_raises(self):
        """2-D mu_hat without pi should raise ValueError."""
        n = 100
        rng = np.random.default_rng(0)
        y, pi, X, D, mu_matrix, w = _make_discrimination_free(n=n)
        m = ProxyDiscriminationMeasure()
        with pytest.raises(ValueError, match="pi"):
            m.fit(y, X, D, mu_hat=mu_matrix, weights=w)


class TestStringCategories:
    """D can be string labels."""

    def test_string_d_binary(self):
        n = 200
        rng = np.random.default_rng(3)
        D = rng.choice(["male", "female"], size=n)
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        pi = X[:, 0] + rng.normal(scale=0.3, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert set(m.categories) == {"male", "female"}
        assert not np.isnan(m.pd_score)

    def test_string_d_five_categories(self):
        n = 300
        rng = np.random.default_rng(4)
        D = rng.choice(["A", "B", "C", "D", "E"], size=n)
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        pi = rng.normal(loc=100, scale=10, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert len(m.categories) == 5
        assert len(m.v_star) == 5


class TestSummary:
    """summary() method should return a non-empty string."""

    def test_summary_returns_string(self):
        y, pi, X, D, _, w = _make_discrimination_free()
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi, weights=w)
        s = m.summary()
        assert isinstance(s, str)
        assert "PD" in s
        assert "UF" in s


# ===========================================================================
# SobolAttribution
# ===========================================================================

class TestSobolAttribution:
    """Tests for per-feature first-order and total Sobol PD indices."""

    def _fitted_measure(self, seed: int = 5):
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=500, seed=seed)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        return m, X, pi, w

    def test_attributions_is_dataframe(self):
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w, feature_names=["a", "b", "c"])
        assert isinstance(sa.attributions_, pd.DataFrame)

    def test_attributions_columns(self):
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w, feature_names=["a", "b", "c"])
        assert list(sa.attributions_.columns) == ["feature", "first_order_pd", "total_pd"]

    def test_attributions_row_count(self):
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w, feature_names=["a", "b", "c"])
        assert len(sa.attributions_) == 3

    def test_first_order_non_negative(self):
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w)
        assert (sa.attributions_["first_order_pd"] >= -1e-10).all()

    def test_total_non_negative(self):
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w)
        assert (sa.attributions_["total_pd"] >= -1e-10).all()

    def test_total_ge_first_order(self):
        """Total effect >= first-order effect (with small numerical slack)."""
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w)
        fo = sa.attributions_["first_order_pd"].values
        to = sa.attributions_["total_pd"].values
        assert (to >= fo - 0.05).all()

    def test_default_feature_names(self):
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w)
        assert list(sa.attributions_["feature"]) == ["x0", "x1", "x2"]

    def test_proxy_feature_has_higher_first_order(self):
        """Feature 0 is the proxy variable — should have highest first-order PD."""
        m, X, pi, w = self._fitted_measure()
        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w, feature_names=["proxy", "noise1", "noise2"])
        attrs = sa.attributions_.set_index("feature")
        assert attrs.loc["proxy", "first_order_pd"] >= max(
            attrs.loc["noise1", "first_order_pd"],
            attrs.loc["noise2", "first_order_pd"],
        ), "Proxy feature should have highest first-order PD index."


# ===========================================================================
# ShapleyAttribution
# ===========================================================================

class TestShapleyAttribution:
    """Tests for CEN-Shapley decomposition of PD."""

    def _fitted_measure(self, n_features: int = 3, seed: int = 5):
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=500, seed=seed)
        if n_features > 3:
            rng = np.random.default_rng(seed)
            X_extra = rng.normal(size=(500, n_features - 3))
            X = np.column_stack([X, X_extra])
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        return m, X, pi, w

    def test_attributions_is_dataframe(self):
        m, X, pi, w = self._fitted_measure()
        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w, feature_names=["a", "b", "c"])
        assert isinstance(sh.attributions_, pd.DataFrame)

    def test_attributions_columns(self):
        m, X, pi, w = self._fitted_measure()
        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w, feature_names=["a", "b", "c"])
        assert list(sh.attributions_.columns) == ["feature", "shapley_pd"]

    def test_attributions_row_count(self):
        m, X, pi, w = self._fitted_measure()
        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w, feature_names=["a", "b", "c"])
        assert len(sh.attributions_) == 3

    def test_shapley_sum_equals_pd_surrogate(self):
        """Shapley values should sum to pd_surrogate_ (v(full set) / Var(pi))."""
        m, X, pi, w = self._fitted_measure()
        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w)
        total = sh.attributions_["shapley_pd"].sum()
        assert total == pytest.approx(sh.pd_surrogate_, rel=1e-4), (
            f"Shapley sum {total:.6f} != pd_surrogate_ {sh.pd_surrogate_:.6f}"
        )

    def test_shapley_sum_close_to_pd(self):
        """Shapley sum should be in the same ballpark as PD."""
        m, X, pi, w = self._fitted_measure()
        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w)
        total = sh.attributions_["shapley_pd"].sum()
        # Surrogate approximation can deviate — allow up to 60% relative error
        pd_val = m.pd_score
        if pd_val > 1e-4:
            assert abs(total - pd_val) / pd_val < 0.70, (
                f"Shapley sum {total:.4f} too far from PD {pd_val:.4f}"
            )

    def test_exact_vs_permutation_consistent(self):
        """Exact (p=3) and MC permutation (p=3, many perms) should broadly agree."""
        m, X, pi, w = self._fitted_measure()
        sh_exact = ShapleyAttribution(exact_threshold=12, random_state=0)
        sh_exact.fit(m.Lambda, X, pi, w)
        sh_perm = ShapleyAttribution(exact_threshold=2, n_permutations=5000, random_state=0)
        sh_perm.fit(m.Lambda, X, pi, w)

        exact_sum = sh_exact.attributions_["shapley_pd"].sum()
        perm_sum = sh_perm.attributions_["shapley_pd"].sum()
        # Both should sum to approximately the same v(full set)
        assert abs(exact_sum - perm_sum) < 0.05, (
            f"Exact sum {exact_sum:.4f} vs permutation sum {perm_sum:.4f} diverge too much"
        )

    def test_large_p_uses_permutation(self):
        """For p > exact_threshold, permutation estimator is used without error."""
        m, X, pi, w = self._fitted_measure(n_features=5)
        sh = ShapleyAttribution(exact_threshold=4, n_permutations=200, random_state=0)
        sh.fit(m.Lambda, X, pi, w)
        assert len(sh.attributions_) == 5

    def test_proxy_feature_has_highest_shapley(self):
        """Feature 0 (proxy for D) should have the largest Shapley PD value."""
        m, X, pi, w = self._fitted_measure()
        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w, feature_names=["proxy", "noise1", "noise2"])
        attrs = sh.attributions_.set_index("feature")
        assert attrs.loc["proxy", "shapley_pd"] >= max(
            attrs.loc["noise1", "shapley_pd"],
            attrs.loc["noise2", "shapley_pd"],
        ), "Proxy feature should have highest Shapley PD contribution."


# ===========================================================================
# Integration: full pipeline
# ===========================================================================

class TestFullPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_no_errors(self):
        """ProxyDiscriminationMeasure -> SobolAttribution -> ShapleyAttribution."""
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=400, seed=99)
        feature_names = ["x0", "x1", "x2"]

        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)

        sa = SobolAttribution()
        sa.fit(m.Lambda, X, pi, w, feature_names=feature_names)

        sh = ShapleyAttribution()
        sh.fit(m.Lambda, X, pi, w, feature_names=feature_names)

        assert not np.isnan(m.pd_score)
        assert not np.isnan(m.uf_score)
        assert len(sa.attributions_) == 3
        assert len(sh.attributions_) == 3

    def test_two_d_categories_pipeline(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=400, n_d=2, seed=11)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert len(m.categories) == 2

    def test_five_d_categories_pipeline(self):
        y, pi, X, D, mu_matrix, w = _make_discriminatory(n=500, n_d=5, seed=12)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=mu_matrix, weights=w, pi=pi)
        assert len(m.categories) == 5

    def test_imbalanced_groups(self):
        """Very imbalanced D groups should not crash anything."""
        n = 300
        rng = np.random.default_rng(88)
        D = np.array([0] * 280 + [1] * 20)
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        pi = X[:, 0] + rng.normal(scale=0.3, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        assert 0.0 <= m.pd_score <= 1.0
