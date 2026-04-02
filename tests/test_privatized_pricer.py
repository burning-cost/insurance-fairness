"""
Tests for privatized_pricer.py — PrivatizedFairPricer

Covers:
 1. test_fit_returns_self                    — fit() returns self (sklearn contract)
 2. test_predict_shape                       — predict() returns (n,) array
 3. test_predict_nonnegative                 — premiums are non-negative
 4. test_fit_predict_matches_predict         — fit_predict == fit().predict(X_train)
 5. test_group_predictions_shape             — group_predictions() returns (n, K)
 6. test_group_predictions_marginalise_to_fair — group_preds @ p_star == predict()
 7. test_excess_risk_bound_positive          — Theorem 4.3 bound is a positive scalar
 8. test_excess_risk_bound_increases_with_noise — tighter epsilon -> larger bound
 9. test_audit_report_type                   — audit_report() returns PrivatizedAuditResult
10. test_correction_summary_keys             — correction_summary() has expected keys
11. test_minimum_sample_size_positive        — minimum_sample_size() returns positive int
12. test_base_estimator_poisson_glm          — poisson_glm runs without error
13. test_base_estimator_tweedie_glm          — tweedie_glm runs without error
14. test_base_estimator_ridge                — ridge runs without error
15. test_base_estimator_catboost             — catboost runs (skipped if not installed)
16. test_mechanism_laplace_raises            — laplace mechanism raises NotImplementedError
17. test_mechanism_gaussian_raises           — gaussian mechanism raises NotImplementedError
18. test_fairness_equalized_odds_raises      — equalized_odds raises NotImplementedError
19. test_delta_nonzero_raises                — delta != 0 raises NotImplementedError
20. test_predict_before_fit_raises           — RuntimeError before fit
21. test_audit_report_before_fit_raises      — RuntimeError before fit
22. test_group_predictions_before_fit_raises — RuntimeError before fit
23. test_excess_risk_bound_before_fit_raises — RuntimeError before fit
24. test_invalid_mechanism_constructor       — ValueError for unknown mechanism
25. test_invalid_constraint_constructor      — ValueError for unknown constraint
26. test_invalid_base_estimator_constructor  — ValueError for unknown base_estimator
27. test_n_features_in_set_after_fit         — n_features_in_ attribute set correctly
28. test_pi_override_matches_epsilon         — supplying pi directly matches epsilon derivation
29. test_k3_groups_smoke                     — three groups run end-to-end
30. test_uniform_reference_group_gap         — uniform P* equalises group mean premiums
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_fairness.privatized_audit import PrivatizedAuditResult
from insurance_fairness.privatized_pricer import PrivatizedFairPricer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _randomised_response(D: np.ndarray, pi: float, K: int, rng) -> np.ndarray:
    """Apply k-randomised response to true labels D."""
    S = D.copy()
    for i in range(len(D)):
        if rng.random() > pi:
            others = [k for k in range(K) if k != D[i]]
            S[i] = rng.choice(others)
    return S


def _make_data(
    n: int = 800,
    K: int = 2,
    pi: float = 0.88,
    seed: int = 0,
    n_features: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, S) for a synthetic Poisson pricing dataset."""
    rng = np.random.default_rng(seed)
    D = rng.integers(0, K, n)
    X = rng.normal(0, 1, (n, n_features))
    rates = 0.08 + 0.03 * D + 0.02 * X[:, 0]
    y = rng.poisson(np.clip(rates, 0.01, None)).astype(float)
    S = _randomised_response(D, pi, K, rng)
    return X, y, S


def _fitted_pricer(**kwargs) -> tuple[PrivatizedFairPricer, np.ndarray, np.ndarray, np.ndarray]:
    """Return a fitted PrivatizedFairPricer and the data it was trained on."""
    X, y, S = _make_data()
    defaults = dict(epsilon=2.0, n_groups=2, base_estimator="poisson_glm", random_state=0)
    defaults.update(kwargs)
    pricer = PrivatizedFairPricer(**defaults)
    pricer.fit(X, y, S)
    return pricer, X, y, S


# ---------------------------------------------------------------------------
# 1–3: sklearn contract and output shape/sign
# ---------------------------------------------------------------------------


class TestSklearnContract:
    def test_fit_returns_self(self):
        X, y, S = _make_data()
        pricer = PrivatizedFairPricer(epsilon=2.0, n_groups=2)
        result = pricer.fit(X, y, S)
        assert result is pricer

    def test_predict_shape(self):
        pricer, X, y, S = _fitted_pricer()
        preds = pricer.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_nonnegative(self):
        pricer, X, y, S = _fitted_pricer()
        preds = pricer.predict(X)
        assert np.all(preds >= 0), "Fair premiums contain negative values"


# ---------------------------------------------------------------------------
# 4–6: fit_predict and group_predictions
# ---------------------------------------------------------------------------


class TestFitPredictAndGroupPredictions:
    def test_fit_predict_matches_predict(self):
        """fit_predict() should equal fit().predict(X_train)."""
        X, y, S = _make_data(seed=1)
        pricer = PrivatizedFairPricer(epsilon=2.0, n_groups=2, random_state=1)
        fp = pricer.fit_predict(X, y, S)
        p = pricer.predict(X)
        np.testing.assert_allclose(fp, p, rtol=1e-10)

    def test_group_predictions_shape(self):
        pricer, X, y, S = _fitted_pricer()
        gp = pricer.group_predictions(X)
        assert gp.shape == (len(X), pricer.n_groups)

    def test_group_predictions_marginalise_to_fair(self):
        """group_predictions(X) @ p_star_ should equal predict(X)."""
        pricer, X, y, S = _fitted_pricer()
        gp = pricer.group_predictions(X)
        p_star = pricer.audit_.p_star_
        reconstructed = gp @ p_star
        fair = pricer.predict(X)
        np.testing.assert_allclose(reconstructed, fair, rtol=1e-10)


# ---------------------------------------------------------------------------
# 7–8: excess_risk_bound
# ---------------------------------------------------------------------------


class TestExcessRiskBound:
    def test_excess_risk_bound_positive(self):
        pricer, X, y, S = _fitted_pricer()
        bound = pricer.excess_risk_bound()
        assert isinstance(bound, float)
        assert bound > 0.0

    def test_excess_risk_bound_increases_with_noise(self):
        """Smaller epsilon (more noise) should produce a larger bound."""
        X, y, S_low = _make_data(seed=5, pi=0.99)   # high pi ~ large epsilon
        _, _, S_noisy = _make_data(seed=5, pi=0.60)  # low pi ~ small epsilon

        pricer_tight = PrivatizedFairPricer(epsilon=5.0, n_groups=2, random_state=0)
        pricer_tight.fit(X, y, S_low)

        pricer_noisy = PrivatizedFairPricer(epsilon=0.5, n_groups=2, random_state=0)
        pricer_noisy.fit(X, y, S_noisy)

        assert pricer_noisy.excess_risk_bound() > pricer_tight.excess_risk_bound(), (
            "Noisier LDP should produce a larger excess risk bound"
        )


# ---------------------------------------------------------------------------
# 9–11: diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_audit_report_type(self):
        pricer, X, y, S = _fitted_pricer()
        report = pricer.audit_report()
        assert isinstance(report, PrivatizedAuditResult)

    def test_correction_summary_keys(self):
        pricer, X, y, S = _fitted_pricer()
        summary = pricer.correction_summary()
        expected = {"Pi_inv", "T_inv", "pi", "pi_bar", "C1", "negative_weight_frac", "bound_95"}
        assert expected.issubset(summary.keys())

    def test_minimum_sample_size_positive(self):
        pricer, X, y, S = _fitted_pricer()
        n_min = pricer.minimum_sample_size()
        assert isinstance(n_min, int)
        assert n_min > 0


# ---------------------------------------------------------------------------
# 12–15: base_estimator variants
# ---------------------------------------------------------------------------


class TestBaseEstimators:
    """Each base_estimator should run fit() and predict() without error."""

    def _run(self, base_estimator: str) -> None:
        X, y, S = _make_data(seed=10)
        pricer = PrivatizedFairPricer(
            epsilon=2.0, n_groups=2, base_estimator=base_estimator, random_state=0
        )
        pricer.fit(X, y, S)
        preds = pricer.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_base_estimator_poisson_glm(self):
        self._run("poisson_glm")

    def test_base_estimator_tweedie_glm(self):
        self._run("tweedie_glm")

    def test_base_estimator_ridge(self):
        self._run("ridge")

    def test_base_estimator_catboost(self):
        pytest.importorskip("catboost")
        self._run("catboost")


# ---------------------------------------------------------------------------
# 16–19: NotImplementedError stubs
# ---------------------------------------------------------------------------


class TestNotImplementedStubs:
    def _data(self):
        return _make_data(seed=20)

    def test_mechanism_laplace_raises(self):
        X, y, S = self._data()
        pricer = PrivatizedFairPricer(epsilon=2.0, mechanism="laplace")
        with pytest.raises(NotImplementedError, match="laplace"):
            pricer.fit(X, y, S)

    def test_mechanism_gaussian_raises(self):
        X, y, S = self._data()
        pricer = PrivatizedFairPricer(epsilon=2.0, mechanism="gaussian")
        with pytest.raises(NotImplementedError, match="gaussian"):
            pricer.fit(X, y, S)

    def test_fairness_equalized_odds_raises(self):
        X, y, S = self._data()
        pricer = PrivatizedFairPricer(
            epsilon=2.0, fairness_constraint="equalized_odds"
        )
        with pytest.raises(NotImplementedError, match="equalized_odds"):
            pricer.fit(X, y, S)

    def test_delta_nonzero_raises(self):
        X, y, S = self._data()
        pricer = PrivatizedFairPricer(epsilon=2.0, delta=1e-5)
        with pytest.raises(NotImplementedError, match="delta"):
            pricer.fit(X, y, S)


# ---------------------------------------------------------------------------
# 20–23: pre-fit RuntimeErrors
# ---------------------------------------------------------------------------


class TestPreFitErrors:
    def _unfitted(self) -> PrivatizedFairPricer:
        return PrivatizedFairPricer(epsilon=2.0, n_groups=2)

    def _X(self) -> np.ndarray:
        return np.random.default_rng(99).normal(0, 1, (10, 4))

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            self._unfitted().predict(self._X())

    def test_audit_report_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            self._unfitted().audit_report()

    def test_group_predictions_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            self._unfitted().group_predictions(self._X())

    def test_excess_risk_bound_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            self._unfitted().excess_risk_bound()


# ---------------------------------------------------------------------------
# 24–26: constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_mechanism_constructor(self):
        with pytest.raises(ValueError, match="mechanism"):
            PrivatizedFairPricer(epsilon=2.0, mechanism="exponential")

    def test_invalid_constraint_constructor(self):
        with pytest.raises(ValueError, match="fairness_constraint"):
            PrivatizedFairPricer(epsilon=2.0, fairness_constraint="individual")

    def test_invalid_base_estimator_constructor(self):
        with pytest.raises(ValueError, match="base_estimator"):
            PrivatizedFairPricer(epsilon=2.0, base_estimator="xgboost")


# ---------------------------------------------------------------------------
# 27: n_features_in_
# ---------------------------------------------------------------------------


class TestNFeaturesIn:
    def test_n_features_in_set_after_fit(self):
        X, y, S = _make_data(n_features=6)
        pricer = PrivatizedFairPricer(epsilon=2.0, n_groups=2)
        pricer.fit(X, y, S)
        assert pricer.n_features_in_ == 6


# ---------------------------------------------------------------------------
# 28: pi override matches epsilon derivation
# ---------------------------------------------------------------------------


class TestPiOverride:
    def test_pi_override_matches_epsilon(self):
        """Supplying pi directly should produce same predictions as the epsilon route."""
        epsilon = 2.0
        K = 2
        pi = float(np.exp(epsilon) / (K - 1 + np.exp(epsilon)))

        X, y, S = _make_data(seed=30)

        pricer_eps = PrivatizedFairPricer(epsilon=epsilon, n_groups=K, random_state=7)
        pricer_eps.fit(X, y, S)

        pricer_pi = PrivatizedFairPricer(pi=pi, n_groups=K, random_state=7)
        pricer_pi.fit(X, y, S)

        np.testing.assert_allclose(
            pricer_eps.predict(X),
            pricer_pi.predict(X),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# 29: K=3 groups smoke test
# ---------------------------------------------------------------------------


class TestThreeGroupsSmoke:
    def test_k3_groups_smoke(self):
        K = 3
        rng = np.random.default_rng(40)
        n = 900
        D = rng.integers(0, K, n)
        X = rng.normal(0, 1, (n, 4))
        y = rng.poisson(0.05 + 0.02 * D).astype(float)
        S = _randomised_response(D, 0.80, K, rng)

        pricer = PrivatizedFairPricer(
            epsilon=2.0, n_groups=K, base_estimator="poisson_glm", random_state=0
        )
        pricer.fit(X, y, S)

        preds = pricer.predict(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

        gp = pricer.group_predictions(X)
        assert gp.shape == (n, K)

        report = pricer.audit_report()
        assert report.p_star.shape == (K,)
        assert len(report.group_models) == K


# ---------------------------------------------------------------------------
# 30: uniform reference equalises group premiums
# ---------------------------------------------------------------------------


class TestUniformReferenceGap:
    def test_uniform_reference_group_gap(self):
        """
        With uniform P*, the fair premium should not differ systematically by
        privatised group label (gap < 10%).
        """
        rng = np.random.default_rng(50)
        n = 2000
        K = 2
        D = rng.choice([0, 1], n, p=[0.6, 0.4])
        X = rng.normal(0, 1, (n, 4))
        y = rng.poisson(0.08 + 0.04 * D).astype(float)
        S = _randomised_response(D, 0.88, K, rng)

        pricer = PrivatizedFairPricer(
            epsilon=2.0,
            n_groups=K,
            reference_distribution="uniform",
            base_estimator="poisson_glm",
            random_state=0,
        )
        pricer.fit(X, y, S)
        preds = pricer.predict(X)

        mean_g0 = preds[S == 0].mean()
        mean_g1 = preds[S == 1].mean()
        gap = abs(mean_g0 - mean_g1) / max(mean_g0, mean_g1)
        assert gap < 0.10, f"Group premium gap too large with uniform P*: {gap:.4f}"
