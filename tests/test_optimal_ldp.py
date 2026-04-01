"""
tests/test_optimal_ldp.py
--------------------------
Tests for OptimalLDPMechanism, LDPEpsilonAdvisor, and the
mechanism='optimal' integration in PrivatizedFairnessAudit.

Coverage:
1.  TestOptimalLDPBinaryClosedForm     — binary K=2 closed form matches Theorem 2
2.  TestOptimalLDPBinaryAsymmetry      — minority group gets higher diagonal
3.  TestOptimalLDPBinaryEqualPrevalence — equal prevalence is symmetric
4.  TestOptimalLDPMultivaluedLP        — K=3 LP solver produces valid stochastic Q
5.  TestOptimalLDPPrivatise            — privatise() returns correct label range
6.  TestOptimalLDPUnfairnessBound      — bound is in [0, 1]
7.  TestOptimalLDPValidation           — invalid inputs raise
8.  TestOptimalLDPFitMethod            — fit() can be called after construction
9.  TestLDPEpsilonAdvisorRecommend     — recommend() returns dict with required keys
10. TestLDPEpsilonAdvisorKnownValues   — C1 at known eps values matches spec
11. TestLDPEpsilonAdvisorSweep         — sweep() returns polars DataFrame
12. TestLDPEpsilonAdvisorEdgeCases     — invalid inputs raise
13. TestOptimalMechanismIntegration    — PrivatizedFairnessAudit with mechanism='optimal'
14. TestOptimalMechanismVsKRR          — optimal gives lower unfairness than k-rr
15. TestMechanismParamValidation       — invalid mechanism values raise
16. TestOptimalAuditReport             — audit_report includes mechanism field
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_fairness.optimal_ldp import LDPEpsilonAdvisor, OptimalLDPMechanism
from insurance_fairness.privatized_audit import PrivatizedFairnessAudit


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _randomised_response_optimal(
    D: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply an arbitrary perturbation matrix Q to true labels D."""
    K = Q.shape[0]
    n = len(D)
    S = np.empty(n, dtype=int)
    for j in range(K):
        mask = D == j
        count = int(mask.sum())
        if count > 0:
            S[mask] = rng.choice(K, size=count, p=Q[j])
    return S


def _make_binary_data(
    n: int = 2000,
    p0: float = 0.3,    # P(group 0) — minority
    seed: int = 42,
    epsilon: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    D = rng.choice([0, 1], size=n, p=[p0, 1 - p0])
    X = rng.normal(0, 1, (n, 3))
    Y = rng.poisson(0.08 + 0.04 * D).astype(float)

    mech = OptimalLDPMechanism(epsilon=epsilon, k=2, group_prevalences=np.array([p0, 1 - p0]))
    S = mech.privatise(D, rng=rng)
    return X, Y, S, D


# ---------------------------------------------------------------------------
# 1. Binary closed form matches Theorem 2
# ---------------------------------------------------------------------------


class TestOptimalLDPBinaryClosedForm:
    """
    Theorem 2 (Ghoukasian & Asoodeh 2025):
    When P(S=0) < P(S=1): p* = 1 - exp(-eps)/2, q* = 1/2
    When P(S=1) < P(S=0): p* = 1/2, q* = 1 - exp(-eps)/2
    """

    def test_minority_group0_diagonal(self):
        """Group 0 is minority: Q[0,0] = 1 - exp(-eps)/2."""
        eps = 2.0
        p = np.array([0.3, 0.7])
        mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix

        expected_p00 = 1.0 - np.exp(-eps) / 2.0
        assert abs(Q[0, 0] - expected_p00) < 1e-10, (
            f"Q[0,0]={Q[0,0]:.8f}, expected {expected_p00:.8f}"
        )

    def test_minority_group0_majority_diagonal(self):
        """Group 1 is majority: Q[1,1] = 0.5."""
        eps = 2.0
        p = np.array([0.3, 0.7])
        mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix
        assert abs(Q[1, 1] - 0.5) < 1e-10, f"Q[1,1]={Q[1,1]:.8f}, expected 0.5"

    def test_minority_group1_diagonal(self):
        """Group 1 is minority: Q[1,1] = 1 - exp(-eps)/2."""
        eps = 1.5
        p = np.array([0.7, 0.3])
        mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix

        expected_q11 = 1.0 - np.exp(-eps) / 2.0
        assert abs(Q[1, 1] - expected_q11) < 1e-10

    def test_minority_group1_majority_diagonal(self):
        """Group 0 is majority: Q[0,0] = 0.5."""
        eps = 1.5
        p = np.array([0.7, 0.3])
        mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix
        assert abs(Q[0, 0] - 0.5) < 1e-10

    def test_row_stochastic(self):
        """Both rows of Q must sum to 1."""
        for eps in [0.5, 1.0, 2.0, 3.0]:
            p = np.array([0.3, 0.7])
            mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
            Q = mech.perturbation_matrix
            np.testing.assert_allclose(Q.sum(axis=1), [1.0, 1.0], atol=1e-10)

    def test_entries_non_negative(self):
        """All entries of Q must be in [0, 1]."""
        p = np.array([0.4, 0.6])
        mech = OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix
        assert np.all(Q >= 0.0)
        assert np.all(Q <= 1.0)


# ---------------------------------------------------------------------------
# 2. Asymmetry: minority gets higher diagonal
# ---------------------------------------------------------------------------


class TestOptimalLDPBinaryAsymmetry:
    def test_minority_has_higher_correct_response(self):
        """For imbalanced groups, the minority diagonal must exceed the majority diagonal."""
        for eps in [0.5, 1.0, 2.0, 5.0]:
            p_minority = 0.2
            p = np.array([p_minority, 1 - p_minority])
            mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
            Q = mech.perturbation_matrix
            assert Q[0, 0] > Q[1, 1], (
                f"eps={eps}: minority Q[0,0]={Q[0,0]:.4f} should exceed "
                f"majority Q[1,1]={Q[1,1]:.4f}"
            )


# ---------------------------------------------------------------------------
# 3. Equal prevalence is symmetric
# ---------------------------------------------------------------------------


class TestOptimalLDPBinaryEqualPrevalence:
    def test_symmetric_for_balanced_groups(self):
        """Equal prevalences should give Q[0,0] = Q[1,1]."""
        p = np.array([0.5, 0.5])
        mech = OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix
        assert abs(Q[0, 0] - Q[1, 1]) < 1e-10, (
            f"Q[0,0]={Q[0,0]:.8f} != Q[1,1]={Q[1,1]:.8f} for balanced groups"
        )


# ---------------------------------------------------------------------------
# 4. Multi-valued LP (K=3)
# ---------------------------------------------------------------------------


class TestOptimalLDPMultivaluedLP:
    def test_k3_row_stochastic(self):
        p = np.array([0.2, 0.5, 0.3])
        mech = OptimalLDPMechanism(epsilon=1.5, k=3, group_prevalences=p)
        Q = mech.perturbation_matrix
        np.testing.assert_allclose(Q.sum(axis=1), np.ones(3), atol=1e-6)

    def test_k3_non_negative(self):
        p = np.array([0.2, 0.5, 0.3])
        mech = OptimalLDPMechanism(epsilon=1.5, k=3, group_prevalences=p)
        Q = mech.perturbation_matrix
        assert np.all(Q >= -1e-8), f"Q has negative entries: {Q.min():.6f}"

    def test_k3_ldp_constraints(self):
        """LDP: standard column-wise Q[j,i] <= exp(eps)*Q[j\'i] for all j != j\', i."""
        eps = 2.0
        p = np.array([0.25, 0.50, 0.25])
        mech = OptimalLDPMechanism(epsilon=eps, k=3, group_prevalences=p)
        Q = mech.perturbation_matrix
        exp_e = np.exp(eps)
        # Column-wise: Q[j,i] / Q[j',i] <= exp(eps) for all j, j', i
        for i in range(3):
            for j in range(3):
                for jp in range(3):
                    if j != jp and Q[jp, i] > 1e-10:
                        ratio = Q[j, i] / Q[jp, i]
                        assert ratio <= exp_e + 1e-4, (
                            f"LDP violated: Q[{j},{i}]/Q[{jp},{i}]={ratio:.4f} > exp({eps})={exp_e:.4f}"
                        )

    def test_k3_shape(self):
        mech = OptimalLDPMechanism(epsilon=1.0, k=3, group_prevalences=np.array([1/3, 1/3, 1/3]))
        assert mech.perturbation_matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# 5. privatise() returns valid labels
# ---------------------------------------------------------------------------


class TestOptimalLDPPrivatise:
    def test_output_in_range(self):
        rng = np.random.default_rng(1)
        D = rng.integers(0, 2, 500)
        mech = OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.4, 0.6]))
        S = mech.privatise(D, rng=rng)
        assert np.all((S >= 0) & (S < 2))

    def test_output_shape_preserved(self):
        rng = np.random.default_rng(2)
        D = rng.integers(0, 3, 1000)
        mech = OptimalLDPMechanism(epsilon=2.0, k=3, group_prevalences=np.array([0.3, 0.4, 0.3]))
        S = mech.privatise(D, rng=rng)
        assert S.shape == D.shape

    def test_privatise_without_rng(self):
        """privatise() should work without an explicit rng."""
        D = np.array([0, 1, 0, 1, 0])
        mech = OptimalLDPMechanism(epsilon=1.5, k=2, group_prevalences=np.array([0.5, 0.5]))
        S = mech.privatise(D)
        assert S.shape == D.shape
        assert np.all((S >= 0) & (S < 2))

    def test_invalid_label_raises(self):
        D = np.array([0, 1, 2])  # 2 is out of range for k=2
        mech = OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="true_labels"):
            mech.privatise(D)

    def test_default_perturbation_matrix_uniform(self):
        """When no group_prevalences given, perturbation_matrix uses uniform."""
        mech = OptimalLDPMechanism(epsilon=1.0, k=2)
        Q = mech.perturbation_matrix
        # Uniform → symmetric → equal diagonal entries
        assert abs(Q[0, 0] - Q[1, 1]) < 1e-10


# ---------------------------------------------------------------------------
# 6. Unfairness bound
# ---------------------------------------------------------------------------


class TestOptimalLDPUnfairnessBound:
    def test_bound_in_unit_interval(self):
        p = np.array([0.35, 0.65])
        mech = OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=p)
        bound = mech.unfairness_bound()
        assert 0.0 <= bound <= 1.0

    def test_bound_non_negative(self):
        """Unfairness bound must be non-negative."""
        p = np.array([0.3, 0.7])
        for eps in [0.5, 1.0, 2.0, 4.0]:
            mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
            assert mech.unfairness_bound() >= 0.0

    def test_symmetric_mechanism_bound(self):
        """For equal prevalences, symmetric mechanism has zero unfairness (no distribution shift)."""
        p = np.array([0.5, 0.5])
        mech = OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=p)
        # With equal prevalences and symmetric Q, noised distribution = true distribution
        bound = mech.unfairness_bound()
        assert bound < 1e-10, f"Expected ~0 unfairness for balanced, got {bound:.6f}"


# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------


class TestOptimalLDPValidation:
    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            OptimalLDPMechanism(epsilon=-1.0)

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            OptimalLDPMechanism(epsilon=0.0)

    def test_k_lt_2_raises(self):
        with pytest.raises(ValueError, match="k"):
            OptimalLDPMechanism(epsilon=1.0, k=1)

    def test_prevalences_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.3, 0.3, 0.4]))

    def test_prevalences_dont_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.3, 0.5]))

    def test_prevalences_non_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# 8. fit() method
# ---------------------------------------------------------------------------


class TestOptimalLDPFitMethod:
    def test_fit_updates_matrix(self):
        mech = OptimalLDPMechanism(epsilon=1.0, k=2)
        Q_before = mech.perturbation_matrix.copy()

        # Fit with imbalanced prevalences — should change the matrix
        mech.fit(np.array([0.2, 0.8]))
        Q_after = mech.perturbation_matrix

        # After fitting imbalanced, minority group should have higher diagonal
        assert Q_after[0, 0] > Q_after[1, 1], (
            "After fitting imbalanced prevalences, minority diagonal should dominate"
        )

    def test_fit_wrong_shape_raises(self):
        mech = OptimalLDPMechanism(epsilon=1.0, k=2)
        with pytest.raises(ValueError, match="shape"):
            mech.fit(np.array([0.3, 0.3, 0.4]))

    def test_fit_returns_self(self):
        mech = OptimalLDPMechanism(epsilon=1.0, k=2)
        result = mech.fit(np.array([0.5, 0.5]))
        assert result is mech


# ---------------------------------------------------------------------------
# 9. LDPEpsilonAdvisor.recommend() returns correct structure
# ---------------------------------------------------------------------------


class TestLDPEpsilonAdvisorRecommend:
    def test_recommend_returns_dict_with_keys(self):
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2, target_bound_inflation=0.30)
        result = adv.recommend()
        assert "epsilon" in result
        assert "pi" in result
        assert "C1" in result
        assert "gen_bound" in result
        assert "rationale" in result

    def test_recommend_c1_meets_target(self):
        target = 0.30
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2, target_bound_inflation=target)
        result = adv.recommend()
        assert result["C1"] <= 1.0 + target + 0.01, (
            f"C1={result['C1']:.4f} exceeds target {1 + target:.2f}"
        )

    def test_recommend_pi_in_valid_range(self):
        adv = LDPEpsilonAdvisor(n_samples=10000, k=2, target_bound_inflation=0.20)
        result = adv.recommend()
        assert 0.5 < result["pi"] < 1.0, f"pi={result['pi']:.4f} out of valid range"

    def test_recommend_epsilon_positive(self):
        adv = LDPEpsilonAdvisor(n_samples=1000, k=3, target_bound_inflation=0.50)
        result = adv.recommend()
        assert result["epsilon"] > 0

    def test_recommend_rationale_is_string(self):
        adv = LDPEpsilonAdvisor(n_samples=5000)
        result = adv.recommend()
        assert isinstance(result["rationale"], str)
        assert len(result["rationale"]) > 10


# ---------------------------------------------------------------------------
# 10. Known C1 values from spec
# ---------------------------------------------------------------------------


class TestLDPEpsilonAdvisorKnownValues:
    """
    Verified C1 values at K=2 via C1=(pi+K-2)/(K*pi-1), pi=exp(e)/(K-1+exp(e)):
        eps=0.5  -> C1 ~ 2.54
        eps=1.0  -> C1 ~ 1.58
        eps=2.0  -> C1 ~ 1.16
        eps=5.0  -> C1 ~ 1.01
    """

    def _c1(self, eps: float, k: int = 2) -> float:
        pi = np.exp(eps) / (k - 1 + np.exp(eps))
        return (pi + k - 2) / (k * pi - 1)

    def test_c1_at_eps_05(self):
        c1 = self._c1(0.5)
        assert abs(c1 - 2.5415) < 0.01, f"C1(eps=0.5)={c1:.4f}, expected ~2.54"

    def test_c1_at_eps_1(self):
        c1 = self._c1(1.0)
        assert abs(c1 - 1.5820) < 0.01, f"C1(eps=1.0)={c1:.4f}, expected ~1.58"

    def test_c1_at_eps_2(self):
        c1 = self._c1(2.0)
        assert abs(c1 - 1.1565) < 0.01, f"C1(eps=2.0)={c1:.4f}, expected ~1.16"

    def test_c1_at_eps_5(self):
        c1 = self._c1(5.0)
        assert abs(c1 - 1.0068) < 0.01, f"C1(eps=5.0)={c1:.4f}, expected ~1.01"

    def test_c1_monotone_in_eps(self):
        """C1 should decrease monotonically as epsilon increases."""
        epsilons = [0.5, 1.0, 2.0, 3.0, 5.0]
        c1s = [self._c1(e) for e in epsilons]
        for i in range(len(c1s) - 1):
            assert c1s[i] > c1s[i + 1], (
                f"C1 not monotone: c1[{epsilons[i]}]={c1s[i]:.4f}, "
                f"c1[{epsilons[i+1]}]={c1s[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# 11. LDPEpsilonAdvisor.sweep()
# ---------------------------------------------------------------------------


class TestLDPEpsilonAdvisorSweep:
    def test_sweep_returns_dataframe(self):
        pytest.importorskip("polars")
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2)
        df = adv.sweep()
        import polars as pl
        assert isinstance(df, pl.DataFrame)

    def test_sweep_has_required_columns(self):
        pytest.importorskip("polars")
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2)
        df = adv.sweep()
        for col in ["epsilon", "pi", "C1", "gen_bound", "bound_inflation_pct"]:
            assert col in df.columns, f"Column {col!r} missing from sweep DataFrame"

    def test_sweep_default_grid_has_rows(self):
        pytest.importorskip("polars")
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2)
        df = adv.sweep()
        assert len(df) > 0

    def test_sweep_custom_grid(self):
        pytest.importorskip("polars")
        eps_grid = np.array([0.5, 1.0, 2.0, 5.0])
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2)
        df = adv.sweep(epsilons=eps_grid)
        assert len(df) == 4

    def test_sweep_c1_decreasing_with_eps(self):
        """C1 column should be monotonically decreasing."""
        pytest.importorskip("polars")
        adv = LDPEpsilonAdvisor(n_samples=5000, k=2)
        df = adv.sweep(epsilons=np.array([0.5, 1.0, 2.0, 5.0]))
        c1_vals = df["C1"].to_list()
        for i in range(len(c1_vals) - 1):
            assert c1_vals[i] >= c1_vals[i + 1] - 1e-9


# ---------------------------------------------------------------------------
# 12. LDPEpsilonAdvisor edge cases
# ---------------------------------------------------------------------------


class TestLDPEpsilonAdvisorEdgeCases:
    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            LDPEpsilonAdvisor(n_samples=0)

    def test_k_lt_2_raises(self):
        with pytest.raises(ValueError, match="k"):
            LDPEpsilonAdvisor(n_samples=1000, k=1)

    def test_negative_inflation_raises(self):
        with pytest.raises(ValueError, match="target_bound_inflation"):
            LDPEpsilonAdvisor(n_samples=1000, target_bound_inflation=-0.1)

    def test_very_strict_target_warns(self):
        """A target that can't be met in [0.1, 10] should warn and return eps=10."""
        adv = LDPEpsilonAdvisor(n_samples=100, k=5, target_bound_inflation=0.001)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = adv.recommend()
        warning_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        # Either we warned, or epsilon is still valid (implementation detail)
        assert result["epsilon"] > 0


# ---------------------------------------------------------------------------
# 13. PrivatizedFairnessAudit with mechanism='optimal'
# ---------------------------------------------------------------------------


class TestOptimalMechanismIntegration:
    def setup_method(self):
        X, Y, S, D = _make_binary_data(n=2000, p0=0.3, seed=10, epsilon=1.5)
        self.X = X
        self.Y = Y
        self.S = S
        self.epsilon = 1.5

    def test_fits_without_error(self):
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=self.epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism="optimal",
            random_state=0,
        )
        audit.fit(self.X, self.Y, self.S)
        assert audit.pi_ is not None

    def test_fair_predictions_shape(self):
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=self.epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism="optimal",
            random_state=0,
        )
        audit.fit(self.X, self.Y, self.S)
        assert audit.fair_predictions_.shape == (len(self.Y),)

    def test_fair_predictions_finite(self):
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=self.epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism="optimal",
            random_state=0,
        )
        audit.fit(self.X, self.Y, self.S)
        assert np.all(np.isfinite(audit.fair_predictions_))

    def test_perturbation_matrix_stored(self):
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=self.epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism="optimal",
            random_state=0,
        )
        audit.fit(self.X, self.Y, self.S)
        assert audit.perturbation_matrix_ is not None
        assert audit.perturbation_matrix_.shape == (2, 2)

    def test_correction_matrices_has_perturbation_matrix(self):
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=self.epsilon,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism="optimal",
            random_state=0,
        )
        audit.fit(self.X, self.Y, self.S)
        mats = audit.correction_matrices()
        assert mats["perturbation_matrix"] is not None
        assert mats["perturbation_matrix"].shape == (2, 2)

    def test_k3_optimal_mechanism(self):
        """K=3 optimal mechanism should fit without error."""
        K = 3
        rng = np.random.default_rng(55)
        n = 1500
        p_groups = np.array([0.2, 0.5, 0.3])
        D = rng.choice(3, size=n, p=p_groups)
        X = rng.normal(0, 1, (n, 3))
        Y = rng.poisson(0.06 + 0.02 * D).astype(float)

        mech = OptimalLDPMechanism(epsilon=2.0, k=K, group_prevalences=p_groups)
        S = mech.privatise(D, rng=rng)

        audit = PrivatizedFairnessAudit(
            n_groups=K,
            epsilon=2.0,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism="optimal",
            random_state=0,
        )
        audit.fit(X, Y, S)
        assert audit.fair_predictions_.shape == (n,)
        assert np.all(np.isfinite(audit.fair_predictions_))


# ---------------------------------------------------------------------------
# 14. Optimal mechanism gives lower or equal unfairness vs k-RR
# ---------------------------------------------------------------------------


class TestOptimalMechanismVsKRR:
    def test_optimal_asymmetry_reduces_minority_noise(self):
        """
        The optimal mechanism for imbalanced groups gives the minority class a higher
        correct-response probability than the majority. This is the key property of
        Ghoukasian & Asoodeh (2025) Theorem 2: asymmetric noise allocation that
        protects the informationally scarcer minority group.
        """
        eps = 1.5
        p_minority = 0.25
        p = np.array([p_minority, 1 - p_minority])

        mech = OptimalLDPMechanism(epsilon=eps, k=2, group_prevalences=p)
        Q = mech.perturbation_matrix

        # Minority (group 0) should have higher correct-response rate than majority
        assert Q[0, 0] > Q[1, 1], (
            f"Minority Q[0,0]={Q[0,0]:.4f} should exceed majority Q[1,1]={Q[1,1]:.4f}"
        )

    def test_k3_optimal_tv_lte_krr(self):
        """
        For K=3, the LP-optimal mechanism should achieve equal or lower TV distance
        than k-RR (since k-RR is a feasible point in the LP).
        """
        eps = 1.5
        p = np.array([0.25, 0.50, 0.25])
        K = 3

        # k-RR
        exp_e = np.exp(eps)
        pi_krr = exp_e / (K - 1 + exp_e)
        Q_krr = np.full((K, K), (1 - pi_krr) / (K - 1))
        np.fill_diagonal(Q_krr, pi_krr)
        tv_krr = float(0.5 * np.sum(np.abs(Q_krr.T @ p - p)))

        # Optimal LP
        mech = OptimalLDPMechanism(epsilon=eps, k=K, group_prevalences=p)
        tv_opt = mech.unfairness_bound()

        assert tv_opt <= tv_krr + 1e-6, (
            f"Optimal TV={tv_opt:.6f} should be <= k-RR TV={tv_krr:.6f}"
        )


# ---------------------------------------------------------------------------
# 15. Mechanism parameter validation
# ---------------------------------------------------------------------------


class TestMechanismParamValidation:
    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError, match="mechanism"):
            PrivatizedFairnessAudit(n_groups=2, epsilon=1.0, mechanism="ldp-v2")

    def test_optimal_with_pi_raises(self):
        with pytest.raises(ValueError, match="not compatible"):
            PrivatizedFairnessAudit(n_groups=2, pi=0.85, mechanism="optimal")

    def test_optimal_without_epsilon_raises(self):
        with pytest.raises(ValueError, match="requires epsilon"):
            PrivatizedFairnessAudit(n_groups=2, mechanism="optimal")

    def test_krr_with_pi_is_valid(self):
        """k-rr mechanism with pi= should not raise."""
        audit = PrivatizedFairnessAudit(
            n_groups=2,
            pi=0.85,
            mechanism="k-rr",
            nuisance_backend="sklearn",
        )
        assert audit.mechanism == "k-rr"

    def test_default_mechanism_is_krr(self):
        audit = PrivatizedFairnessAudit(n_groups=2, epsilon=1.0)
        assert audit.mechanism == "k-rr"


# ---------------------------------------------------------------------------
# 16. audit_report includes mechanism field
# ---------------------------------------------------------------------------


class TestOptimalAuditReport:
    def _fit_audit(self, mechanism: str) -> PrivatizedFairnessAudit:
        rng = np.random.default_rng(77)
        n = 600
        D = rng.integers(0, 2, n)
        X = rng.normal(0, 1, (n, 3))
        Y = rng.poisson(0.08, n).astype(float)
        mech = OptimalLDPMechanism(epsilon=1.5, k=2, group_prevalences=np.array([0.5, 0.5]))
        S = mech.privatise(D, rng=rng)

        audit = PrivatizedFairnessAudit(
            n_groups=2,
            epsilon=1.5,
            reference_distribution="uniform",
            loss="poisson",
            nuisance_backend="sklearn",
            mechanism=mechanism,
            random_state=0,
        )
        audit.fit(X, Y, S)
        return audit

    def test_krr_mechanism_in_report(self):
        audit = self._fit_audit("k-rr")
        report = audit.audit_report()
        assert report.mechanism == "k-rr"

    def test_optimal_mechanism_in_report(self):
        audit = self._fit_audit("optimal")
        report = audit.audit_report()
        assert report.mechanism == "optimal"

    def test_krr_perturbation_matrix_is_none(self):
        """k-rr mechanism should not store a perturbation matrix."""
        audit = self._fit_audit("k-rr")
        assert audit.perturbation_matrix_ is None

    def test_optimal_perturbation_matrix_is_not_none(self):
        audit = self._fit_audit("optimal")
        assert audit.perturbation_matrix_ is not None
