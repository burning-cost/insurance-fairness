"""
Tests for privatized_audit.py

Covers:
1. test_binary_known_epsilon        — two groups, known pi, correction matrices + fair premium gap
2. test_anchor_point_pi_recovery    — estimate pi from anchor points, |pi_est - pi_true| < 0.04
3. test_uniform_vs_empirical_reference — uniform reference equalises group premiums
4. test_negative_weight_warning     — small epsilon triggers UserWarning
5. test_poisson_deviance            — fair model does not catastrophically degrade deviance
6. Additional: ValueError for missing pi/epsilon/X_anchor
7. Additional: prediction before fit raises RuntimeError
8. Additional: PrivatizedAuditResult dataclass fields present
9. Additional: custom reference distribution (array)
10. Additional: K=3 groups basic smoke test
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_fairness.privatized_audit import PrivatizedAuditResult, PrivatizedFairnessAudit


# ---------------------------------------------------------------------------
# Shared helpers / data generators
# ---------------------------------------------------------------------------


def _randomised_response(D: np.ndarray, pi: float, K: int, rng) -> np.ndarray:
    """Apply binary/categorical randomised response to true labels D."""
    n = len(D)
    S = D.copy()
    for i in range(n):
        if rng.random() > pi:
            # Incorrect response: choose uniformly among other K-1 groups
            others = [k for k in range(K) if k != D[i]]
            S[i] = rng.choice(others)
    return S


def _make_binary_poisson_data(
    n: int = 3000,
    pi: float = 0.88,
    rate_group0: float = 0.10,
    rate_group1: float = 0.12,  # 20% higher
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Poisson claim data with two groups.

    Returns (X, Y, S, D_true) where S is privatised via randomised response.
    """
    rng = np.random.default_rng(seed)
    K = 2

    # True group assignment: balanced
    D = rng.integers(0, K, size=n)

    # Feature matrix: 4 features
    X = rng.normal(0, 1, size=(n, 4))

    # Poisson rates: group 1 has higher base rate
    rates = np.where(D == 0, rate_group0, rate_group1)
    rates = rates * np.exp(0.3 * X[:, 0])  # add feature effect
    Y = rng.poisson(rates).astype(float)

    # Randomised response
    S = _randomised_response(D, pi, K, rng)

    return X, Y, S, D


# ---------------------------------------------------------------------------
# 1. Known epsilon: correction matrices and fair premium gap
# ---------------------------------------------------------------------------


class TestBinaryKnownEpsilon:
    """Two groups, known pi from epsilon. Verify correction matrices and fairness."""

    def setup_method(self):
        epsilon = 2.0  # pi = exp(2) / (1 + exp(2)) ~ 0.88 for K=2
        K = 2
        exp_e = np.exp(epsilon)
        self.pi_true = exp_e / (K - 1 + exp_e)

        X, Y, S, _ = _make_binary_poisson_data(n=3000, pi=self.pi_true, seed=0)
        self.X = X
        self.Y = Y
        self.S = S

        self.audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )
        self.audit.fit(X, Y, S)

    def test_pi_derived_from_epsilon(self):
        """pi should equal exp(epsilon)/(1 + exp(epsilon)) for K=2."""
        assert abs(self.audit.pi_ - self.pi_true) < 1e-10

    def test_T_inv_shape(self):
        mats = self.audit.correction_matrices()
        assert mats["T_inv"].shape == (2, 2)
        assert mats["Pi_inv"].shape == (2, 2)

    def test_T_inv_diagonal_dominance(self):
        """Diagonal of T_inv should be larger than off-diagonal for pi > 1/K."""
        mats = self.audit.correction_matrices()
        T_inv = mats["T_inv"]
        assert T_inv[0, 0] > T_inv[0, 1]

    def test_C1_positive(self):
        mats = self.audit.correction_matrices()
        assert mats["C1"] > 0

    def test_fair_premium_gap_small(self):
        """Mean fair premium should not differ by more than 5% between groups."""
        fair = self.audit.fair_predictions_
        mean_g0 = fair[self.S == 0].mean()
        mean_g1 = fair[self.S == 1].mean()
        ratio = max(mean_g0, mean_g1) / min(mean_g0, mean_g1)
        assert ratio < 1.05, f"Premium gap too large: ratio={ratio:.4f}"

    def test_predict_fair_premium_matches_training(self):
        """predict_fair_premium on training X should match stored fair_predictions_."""
        preds = self.audit.predict_fair_premium(self.X)
        np.testing.assert_allclose(preds, self.audit.fair_predictions_, rtol=1e-10)

    def test_statistical_bound_positive(self):
        bound = self.audit.statistical_bound(delta=0.05)
        assert bound > 0

    def test_p_corrected_sums_to_one(self):
        assert abs(self.audit.p_corrected_.sum() - 1.0) < 1e-6

    def test_p_star_uniform(self):
        np.testing.assert_allclose(self.audit.p_star_, [0.5, 0.5], atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Anchor-point pi recovery
# ---------------------------------------------------------------------------


class TestAnchorPointPiRecovery:
    """
    Estimate pi from anchor points where D is known with near-certainty.
    True pi = 0.85; verify |pi_estimated - 0.85| < 0.04.

    Design rationale:
    The anchor-point method (Procedure 4.5) requires that there exist covariate
    regions where P(D=j*|X) ≈ 1. In insurance this is analogous to fleet
    vehicles (P(D=commercial) ≈ 1) or postcodes with homogeneous demographics.

    To make the test reliable:
    - Main dataset uses 3 uninformative features (independent of D) so the
      classifier only gets moderate probabilities from the main data.
    - A separate set of 300 "anchor" observations has a 4th feature = D*20
      (very strong group signal) and 0 in that column for regular obs.
    - The X_anchor matrix is passed with all 4 columns (including the strong
      anchor signal). X_pricing has only 3 columns (the actual pricing features).
    - This mirrors the real-world setup: anchor features (fleet flag, postcode
      census data) are distinct from the pricing rating factors.

    At anchor observations, P_hat(S=j*|X*) should approach pi (not 1.0), because
    S is a noised version of D and the classifier learns P(S=j*|D=j*) = pi.
    """

    def test_pi_recovery_within_tolerance(self):
        pi_true = 0.85
        K = 2
        rng = np.random.default_rng(7)
        # Use larger dataset and fewer partitions for reliable C1 estimation.
        # Each partition (n/n_anchor_groups ~ 350 obs) contains ~50 anchor obs,
        # so max predicted prob per partition reliably approaches pi.
        n_main = 3000
        n_anchor = 500
        n = n_main + n_anchor

        # True labels
        D_main = rng.integers(0, K, n_main)
        D_anchor = rng.integers(0, K, n_anchor)
        D_all = np.concatenate([D_main, D_anchor])

        # Main observations: 3 uninformative features + anchor_signal=0
        # (pricing model does not observe group membership)
        X_main_pricing = rng.normal(0, 1, (n_main, 3))
        X_main_anchor_col = np.zeros(n_main)

        # Anchor observations: 3 near-zero features + anchor_signal = D*30
        # (strong group signal, e.g. fleet type flag)
        X_anch_pricing = rng.normal(0, 0.1, (n_anchor, 3))
        X_anch_anchor_col = D_anchor.astype(float) * 30.0

        # X_pricing: what the GLM uses to fit f_k (no anchor column)
        X_pricing = np.vstack([X_main_pricing, X_anch_pricing])

        # X_anchor: what the anchor classifier uses (includes anchor column)
        X_anchor = np.column_stack([
            np.vstack([X_main_pricing, X_anch_pricing]),
            np.concatenate([X_main_anchor_col, X_anch_anchor_col]),
        ])

        # Outcomes
        Y_main = rng.poisson(0.1 + 0.05 * D_main).astype(float)
        Y_anchor = rng.poisson(0.1, n_anchor).astype(float)
        Y_all = np.concatenate([Y_main, Y_anchor])

        # Noised labels (randomised response with true pi)
        S_all = _randomised_response(D_all, pi_true, K, rng)

        # n_anchor_groups=10 gives ~350 obs per partition, ~50 anchors each.
        # Larger partitions ensure each has enough anchor obs for reliable
        # max-probability estimation.
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            reference_distribution="empirical",
            loss="poisson",
            nuisance_backend="sklearn",
            n_anchor_groups=10,
            random_state=99,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            audit.fit(X_pricing, Y_all, S_all, X_anchor=X_anchor)

        assert abs(audit.pi_ - pi_true) < 0.04, (
            f"pi_estimated={audit.pi_:.4f}, pi_true={pi_true}, "
            f"diff={abs(audit.pi_ - pi_true):.4f}"
        )

    def test_pi_known_is_false_for_anchor(self):
        pi_true = 0.85
        K = 2
        rng = np.random.default_rng(11)
        n = 500
        D = rng.integers(0, K, n)
        X = np.column_stack([rng.normal(0, 1, n), D.astype(float) * 5])
        Y = rng.poisson(0.1, n).astype(float)
        S = _randomised_response(D, pi_true, K, rng)

        audit = PrivatizedFairnessAudit(n_groups=2, nuisance_backend="sklearn", random_state=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            audit.fit(X, Y, S, X_anchor=X)
        assert audit.pi_known_ is False

    def test_anchor_quality_returned(self):
        rng = np.random.default_rng(13)
        n = 500
        D = rng.integers(0, 2, n)
        X = np.column_stack([rng.normal(0, 1, n), D.astype(float) * 5])
        Y = rng.poisson(0.1, n).astype(float)
        S = _randomised_response(D, 0.85, 2, rng)

        audit = PrivatizedFairnessAudit(n_groups=2, nuisance_backend="sklearn", random_state=2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            audit.fit(X, Y, S, X_anchor=X)
        result = audit.audit_report()
        assert result.anchor_quality is not None
        assert 0.0 < result.anchor_quality <= 1.0


# ---------------------------------------------------------------------------
# 3. Uniform vs empirical reference distribution
# ---------------------------------------------------------------------------


class TestUniformVsEmpiricalReference:
    """
    Imbalanced groups: 70% group 0, 30% group 1.
    Uniform p_star should equalise group mean premiums more than empirical.
    """

    def setup_method(self):
        rng = np.random.default_rng(21)
        n = 2000
        K = 2

        # Imbalanced groups: 70/30
        D = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        X = rng.normal(0, 1, size=(n, 3))
        # Group 1 has higher claim rate
        Y = rng.poisson(0.08 + 0.04 * D).astype(float)
        S = _randomised_response(D, 0.88, K, rng)

        self.X = X
        self.Y = Y
        self.S = S

    def test_uniform_reference_equalises_premiums(self):
        """With uniform p_star, group mean fair premiums should be close."""
        audit_uniform = PrivatizedFairnessAudit(
            n_groups=2,
            pi=0.88,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )
        audit_uniform.fit(self.X, self.Y, self.S)

        fair = audit_uniform.fair_predictions_
        mean_g0 = fair[self.S == 0].mean()
        mean_g1 = fair[self.S == 1].mean()

        # Uniform reference collapses per-group average toward the same value
        gap_uniform = abs(mean_g0 - mean_g1) / max(mean_g0, mean_g1)
        assert gap_uniform < 0.10, (
            f"Uniform reference gap={gap_uniform:.4f} should be < 10%"
        )

    def test_empirical_reference_preserves_prevalence_weighting(self):
        """With empirical p_star, the distribution reflects noise-corrected marginals."""
        audit_empirical = PrivatizedFairnessAudit(
            n_groups=2,
            pi=0.88,
            reference_distribution="empirical",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )
        audit_empirical.fit(self.X, self.Y, self.S)

        p_star = audit_empirical.p_star_
        # Empirical should reflect 70/30 imbalance (corrected for noise)
        assert p_star[0] > 0.5, "Empirical p_star should weight majority group more"
        assert abs(p_star.sum() - 1.0) < 1e-6

    def test_custom_array_reference(self):
        """Custom array reference distribution should be accepted and used."""
        p_custom = np.array([0.6, 0.4])
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            pi=0.88,
            reference_distribution=p_custom,
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )
        audit.fit(self.X, self.Y, self.S)
        np.testing.assert_allclose(audit.p_star_, p_custom, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Negative weight warning
# ---------------------------------------------------------------------------


class TestNegativeWeightWarning:
    """
    epsilon=0.3 is very noisy (K=2: pi ~ 0.574). Pi_inv off-diagonal entries
    should be negative, triggering a UserWarning.
    """

    def test_warning_raised(self):
        epsilon = 0.3
        K = 2
        rng = np.random.default_rng(31)
        n = 1000

        D = rng.integers(0, K, n)
        X = rng.normal(0, 1, size=(n, 3))
        Y = rng.poisson(0.1, n).astype(float)

        exp_e = np.exp(epsilon)
        pi_noisy = exp_e / (K - 1 + exp_e)
        S = _randomised_response(D, pi_noisy, K, rng)

        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            audit.fit(X, Y, S)

        warning_messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("negative_weight_frac" in msg for msg in warning_messages), (
            f"Expected UserWarning about negative_weight_frac. Got: {warning_messages}"
        )

    def test_negative_weight_frac_positive(self):
        epsilon = 0.3
        K = 2
        rng = np.random.default_rng(33)
        n = 1000

        D = rng.integers(0, K, n)
        X = rng.normal(0, 1, size=(n, 3))
        Y = rng.poisson(0.1, n).astype(float)

        exp_e = np.exp(epsilon)
        pi_noisy = exp_e / (K - 1 + exp_e)
        S = _randomised_response(D, pi_noisy, K, rng)

        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            audit.fit(X, Y, S)

        result = audit.audit_report()
        assert result.negative_weight_frac > 0.0, (
            "Expected negative_weight_frac > 0 for very noisy epsilon=0.3"
        )


# ---------------------------------------------------------------------------
# 5. Poisson deviance: fair model should not catastrophically degrade deviance
# ---------------------------------------------------------------------------


class TestPoissonDeviance:
    """
    Fair model deviance should be within reasonable range of a naive model.
    We don't expect it to beat a group-aware model, but it should not be
    catastrophically worse (less than 3x deviance increase).
    """

    @staticmethod
    def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Poisson deviance: 2*(y*log(y/mu) - y + mu)."""
        y_pred = np.maximum(y_pred, 1e-10)
        # Avoid log(0): replace 0 outcomes with near-zero
        y_safe = np.where(y_true > 0, y_true, 1e-10)
        d = 2.0 * (y_safe * np.log(y_safe / y_pred) - y_true + y_pred)
        return float(d.mean())

    def test_fair_deviance_acceptable(self):
        pi_true = 0.88
        K = 2
        rng = np.random.default_rng(51)
        n = 3000

        D = rng.integers(0, K, n)
        X = rng.normal(0, 1, size=(n, 4))
        Y = rng.poisson(0.08 + 0.04 * D + 0.02 * X[:, 0]).astype(float)
        S = _randomised_response(D, pi_true, K, rng)

        # Fair model
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            pi=pi_true,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )
        audit.fit(X, Y, S)
        fair_preds = audit.fair_predictions_

        # Naive baseline: predict mean
        naive_preds = np.full(n, Y.mean())

        dev_fair = self._poisson_deviance(Y, fair_preds)
        dev_naive = self._poisson_deviance(Y, naive_preds)

        # Fair model should be at least as good as naive (or not 3x worse)
        assert dev_fair < dev_naive * 3.0, (
            f"Fair model deviance {dev_fair:.4f} is worse than 3x naive {dev_naive:.4f}"
        )
        # Also check predictions are non-negative
        assert np.all(fair_preds >= 0), "Fair premiums should be non-negative"


# ---------------------------------------------------------------------------
# 6. ValueError for missing pi/epsilon/X_anchor
# ---------------------------------------------------------------------------


class TestMissingNoiseRateError:
    def test_raises_value_error_no_params(self):
        rng = np.random.default_rng(61)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        Y = rng.poisson(0.1, n).astype(float)
        S = rng.integers(0, 2, n)

        audit = PrivatizedFairnessAudit(n_groups=2, nuisance_backend="sklearn")
        with pytest.raises(ValueError, match="Must supply one of"):
            audit.fit(X, Y, S)


# ---------------------------------------------------------------------------
# 7. RuntimeError before fit
# ---------------------------------------------------------------------------


class TestUnfittedErrors:
    def test_predict_before_fit_raises(self):
        audit = PrivatizedFairnessAudit(n_groups=2, pi=0.8, nuisance_backend="sklearn")
        X = np.random.default_rng(71).normal(0, 1, (10, 2))
        with pytest.raises(RuntimeError, match="not been fitted"):
            audit.predict_fair_premium(X)

    def test_correction_matrices_before_fit_raises(self):
        audit = PrivatizedFairnessAudit(n_groups=2, pi=0.8, nuisance_backend="sklearn")
        with pytest.raises(RuntimeError):
            audit.correction_matrices()

    def test_audit_report_before_fit_raises(self):
        audit = PrivatizedFairnessAudit(n_groups=2, pi=0.8, nuisance_backend="sklearn")
        with pytest.raises(RuntimeError):
            audit.audit_report()


# ---------------------------------------------------------------------------
# 8. PrivatizedAuditResult dataclass
# ---------------------------------------------------------------------------


class TestAuditResultDataclass:
    def setup_method(self):
        rng = np.random.default_rng(81)
        n = 500
        D = rng.integers(0, 2, n)
        X = rng.normal(0, 1, (n, 3))
        Y = rng.poisson(0.1, n).astype(float)
        S = _randomised_response(D, 0.88, 2, rng)

        self.audit = PrivatizedFairnessAudit(
            n_groups=2, pi=0.88, nuisance_backend="sklearn", random_state=0
        )
        self.audit.fit(X, Y, S)

    def test_result_has_all_fields(self):
        result = self.audit.audit_report()
        assert isinstance(result, PrivatizedAuditResult)
        assert hasattr(result, "fair_premium")
        assert hasattr(result, "group_models")
        assert hasattr(result, "pi_estimated")
        assert hasattr(result, "pi_known")
        assert hasattr(result, "p_star")
        assert hasattr(result, "p_corrected")
        assert hasattr(result, "bound_95")
        assert hasattr(result, "anchor_quality")
        assert hasattr(result, "negative_weight_frac")

    def test_group_models_length(self):
        result = self.audit.audit_report()
        assert len(result.group_models) == 2

    def test_p_star_sums_to_one(self):
        result = self.audit.audit_report()
        assert abs(result.p_star.sum() - 1.0) < 1e-6

    def test_anchor_quality_none_for_known_pi(self):
        result = self.audit.audit_report()
        assert result.anchor_quality is None

    def test_pi_known_true_for_supplied_pi(self):
        result = self.audit.audit_report()
        assert result.pi_known is True

    def test_bound_95_positive(self):
        result = self.audit.audit_report()
        assert result.bound_95 > 0


# ---------------------------------------------------------------------------
# 9. K=3 groups smoke test
# ---------------------------------------------------------------------------


class TestThreeGroupsSmokeTest:
    def test_k3_runs_without_error(self):
        K = 3
        pi_true = 0.80
        rng = np.random.default_rng(91)
        n = 1500

        D = rng.integers(0, K, n)
        X = rng.normal(0, 1, (n, 3))
        Y = rng.poisson(0.05 + 0.02 * D).astype(float)
        S = _randomised_response(D, pi_true, K, rng)

        audit = PrivatizedFairnessAudit(
            n_groups=K,
            pi=pi_true,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            random_state=42,
        )
        audit.fit(X, Y, S)

        assert audit.p_star_.shape == (K,)
        assert len(audit.group_models_) == K
        mats = audit.correction_matrices()
        assert mats["T_inv"].shape == (K, K)
        assert mats["Pi_inv"].shape == (K, K)

        preds = audit.predict_fair_premium(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

    def test_k3_correction_matrices_row_sums(self):
        """T_inv rows should approximately sum to 1 (T_inv @ 1 = 1 for stochastic inverse)."""
        # T_inv is the inverse of the transition matrix T.
        # T has row sums = 1, so T_inv @ T = I.
        # This test checks the structure is self-consistent.
        K = 3
        pi_true = 0.80
        rng = np.random.default_rng(92)
        n = 600
        D = rng.integers(0, K, n)
        X = rng.normal(0, 1, (n, 3))
        Y = rng.poisson(0.05, n).astype(float)
        S = _randomised_response(D, pi_true, K, rng)

        audit = PrivatizedFairnessAudit(
            n_groups=K, pi=pi_true, nuisance_backend="sklearn", random_state=0
        )
        audit.fit(X, Y, S)
        mats = audit.correction_matrices()

        # Build T (forward matrix)
        pi = mats["pi"]
        pi_bar = mats["pi_bar"]
        T = np.full((K, K), pi_bar)
        np.fill_diagonal(T, pi)

        # T_inv @ T should be close to identity
        product = mats["T_inv"] @ T
        np.testing.assert_allclose(product, np.eye(K), atol=1e-8)


# ---------------------------------------------------------------------------
# 10. Invalid constructor arguments
# ---------------------------------------------------------------------------


class TestInvalidConstructorArgs:
    def test_invalid_loss(self):
        with pytest.raises(ValueError, match="loss"):
            PrivatizedFairnessAudit(loss="gamma")

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="nuisance_backend"):
            PrivatizedFairnessAudit(nuisance_backend="xgboost")

    def test_invalid_reference_string(self):
        with pytest.raises(ValueError, match="reference_distribution"):
            PrivatizedFairnessAudit(reference_distribution="population")

    def test_custom_reference_wrong_sum(self):
        rng = np.random.default_rng(101)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        Y = rng.poisson(0.1, n).astype(float)
        S = rng.integers(0, 2, n)
        p_bad = np.array([0.4, 0.4])  # sums to 0.8, not 1

        audit = PrivatizedFairnessAudit(
            n_groups=2, pi=0.85, reference_distribution=p_bad, nuisance_backend="sklearn"
        )
        with pytest.raises(ValueError, match="sum to 1"):
            audit.fit(X, Y, S)
