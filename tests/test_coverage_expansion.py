"""
test_coverage_expansion.py
--------------------------
Expanded test coverage targeting the newest additions to insurance-fairness
(v0.8.x through v1.2.0). Fills gaps in:

- SequentialOTCorrector (three-attribute, three-group, _compute_w1_unfairness)
- IntersectionalFairnessAudit / DistanceCovFairnessRegulariser (edge cases)
- MultiStateTransitionFairness (single group, discount rate monotonicity)
- OptimalLDPMechanism / LDPEpsilonAdvisor (numerical constraints, bounds)
- LipschitzMetric / LipschitzResult (analytical checks, custom distance)
- PrivatizedFairPricer (empirical ref distribution, custom p_star, repr)
- ProxyDiscriminationMeasure (bad ndim, exposure_weighted=False, pandas D)
- ShapleyAttribution (zero-variance Lambda, single feature)
- LocalizedParityCorrector (three groups, report after held-out transform)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from insurance_fairness.optimal_transport.correction import (
    SequentialOTCorrector,
    WassersteinCorrector,
)
from insurance_fairness.intersectional import (
    IntersectionalFairnessAudit,
    DistanceCovFairnessRegulariser,
)
from insurance_fairness.multi_state import (
    KolmogorovPremiumCalculator,
    MultiStateFairnessReport,
    MultiStateTransitionFairness,
    PoissonTransitionFitter,
    TransitionDataBuilder,
)
from insurance_fairness.optimal_ldp import LDPEpsilonAdvisor, OptimalLDPMechanism
from insurance_fairness.pareto import LipschitzMetric, LipschitzResult
from insurance_fairness.privatized_pricer import PrivatizedFairPricer
from insurance_fairness.sensitivity import (
    ProxyDiscriminationMeasure,
    ShapleyAttribution,
    SobolAttribution,
)
from insurance_fairness.localized_parity import (
    LocalizedParityAudit,
    LocalizedParityCorrector,
    LocalizedParityReport,
)


# ===========================================================================
# Helpers / shared data factories
# ===========================================================================


def _make_seq_three_attr(n: int = 300, seed: int = 0):
    """Three protected attributes, each with two groups."""
    rng = np.random.default_rng(seed)
    gender = ["M"] * n + ["F"] * n
    age_band = (["young"] * (n // 2) + ["old"] * (n // 2)) * 2
    region = (["north"] * (n // 3) + ["south"] * (n // 3) + ["midlands"] * (n // 3)) * 2
    region = region[: 2 * n]  # trim to length
    preds = rng.lognormal(0, 0.4, 2 * n)
    D = pl.DataFrame({"gender": gender, "age_band": age_band, "region": region})
    return preds, D


def _make_three_group_preds(n: int = 300, seed: int = 1):
    """Single protected attribute with three groups."""
    rng = np.random.default_rng(seed)
    groups = ["A"] * n + ["B"] * n + ["C"] * n
    preds = np.concatenate([
        rng.lognormal(0.0, 0.3, n),
        rng.lognormal(0.3, 0.3, n),
        rng.lognormal(0.6, 0.3, n),
    ])
    D = pl.DataFrame({"group": groups})
    return preds, D


def _discrimination_free_data(n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = rng.choice(2, size=n)
    y = X[:, 0] * 2.0 + 1.0 + rng.normal(scale=0.1, size=n)
    pi = X[:, 0] * 2.0 + 1.0 + rng.normal(scale=0.01, size=n)
    mu_matrix = np.column_stack([X[:, 0] * 2.0 + 1.0] * 2)
    w = np.ones(n)
    return y, pi, X, D, mu_matrix, w


# ===========================================================================
# A. SequentialOTCorrector — gaps
# ===========================================================================


class TestSequentialOTCorrectorThreeAttr:
    """Three-attribute sequential correction: verify shape and positivity."""

    def test_three_attr_fit_returns_self(self):
        preds, D = _make_seq_three_attr()
        c = SequentialOTCorrector(["gender", "age_band", "region"])
        assert c.fit(preds, D) is c

    def test_three_attr_transform_shape(self):
        preds, D = _make_seq_three_attr()
        c = SequentialOTCorrector(["gender", "age_band", "region"])
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert result.shape == preds.shape

    def test_three_attr_transform_positive(self):
        preds, D = _make_seq_three_attr()
        c = SequentialOTCorrector(["gender", "age_band", "region"])
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert np.all(result > 0)

    def test_three_attr_four_intermediates(self):
        """K=3 attributes => f_0, f_1, f_2, f_3 = 4 intermediate arrays."""
        preds, D = _make_seq_three_attr()
        c = SequentialOTCorrector(["gender", "age_band", "region"])
        c.fit(preds, D)
        intermediates = c.get_intermediate_predictions()
        assert len(intermediates) == 4

    def test_three_attr_unfairness_reductions_has_all_keys(self):
        preds, D = _make_seq_three_attr()
        c = SequentialOTCorrector(["gender", "age_band", "region"])
        c.fit(preds, D)
        reductions = c.unfairness_reductions_
        assert set(reductions.keys()) == {"gender", "age_band", "region"}

    def test_three_attr_epsilon_list_correct_length(self):
        preds, D = _make_seq_three_attr()
        # Each attribute gets its own blend factor
        c = SequentialOTCorrector(["gender", "age_band", "region"], epsilon=[0.0, 0.5, 1.0])
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert np.all(result > 0)

    def test_three_attr_epsilon_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="epsilon list length"):
            SequentialOTCorrector(["gender", "age_band"], epsilon=[0.0, 0.5, 0.0])


class TestSequentialOTCorrectorThreeGroups:
    """Single attribute with three groups (A, B, C)."""

    def test_three_groups_fit_transform(self):
        preds, D = _make_three_group_preds()
        c = SequentialOTCorrector(["group"], log_space=True)
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert result.shape == preds.shape
        assert np.all(result > 0)

    def test_three_groups_unfairness_reduction(self):
        preds, D = _make_three_group_preds()
        c = SequentialOTCorrector(["group"], log_space=True)
        c.fit(preds, D)
        reductions = c.unfairness_reductions_
        before, after = reductions["group"]
        # Correction should not increase unfairness
        assert after <= before + 1e-6

    def test_three_groups_wasserstein_distance_not_populated(self):
        """W2 distance is only populated for exactly 2 groups."""
        preds, D = _make_three_group_preds()
        c = SequentialOTCorrector(["group"])
        c.fit(preds, D)
        # Three groups: W2 entry should be absent (not computed)
        assert "group" not in c.wasserstein_distances_


class TestSequentialOTStaticMethod:
    """_compute_w1_unfairness static method — analytical checks."""

    def test_equal_group_means_gives_zero(self):
        """All groups at the same mean => W1 unfairness = 0."""
        rng = np.random.default_rng(0)
        working = rng.normal(0.0, 0.5, 300)
        col = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
        # Force all group means to 0
        for g in ["A", "B", "C"]:
            mask = col == g
            working[mask] -= working[mask].mean()

        groups = np.unique(col)
        result = SequentialOTCorrector._compute_w1_unfairness(working, col, groups)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_known_two_group_deviation(self):
        """Two groups with means 1 and -1: overall mean = 0, MAD = 1.0."""
        working = np.array([1.0] * 100 + [-1.0] * 100)
        col = np.array(["A"] * 100 + ["B"] * 100)
        groups = np.array(["A", "B"])
        result = SequentialOTCorrector._compute_w1_unfairness(working, col, groups)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_single_group_gives_zero(self):
        """Single group: deviation from overall mean = 0."""
        working = np.array([1.0, 2.0, 3.0])
        col = np.array(["A", "A", "A"])
        groups = np.array(["A"])
        result = SequentialOTCorrector._compute_w1_unfairness(working, col, groups)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestSequentialOTNQuantiles:
    """n_quantiles parameter propagates correctly."""

    def test_high_n_quantiles_same_result(self):
        """Higher n_quantiles should give essentially the same correction."""
        rng = np.random.default_rng(7)
        preds = rng.lognormal(0, 0.3, 400)
        D = pl.DataFrame({"group": ["A"] * 200 + ["B"] * 200})

        c_default = SequentialOTCorrector(["group"], n_quantiles=1000)
        c_fine = SequentialOTCorrector(["group"], n_quantiles=5000)
        c_default.fit(preds, D)
        c_fine.fit(preds, D)
        r_default = c_default.transform(preds, D)
        r_fine = c_fine.transform(preds, D)
        # Results should be very close but not necessarily identical
        np.testing.assert_allclose(r_default, r_fine, rtol=0.01)


# ===========================================================================
# B. IntersectionalFairnessAudit — additional edge cases
# ===========================================================================


class TestIntersectionalAuditEdgeCases:
    """Edge cases not covered by the main test file."""

    def _make_data(self, n: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        y_hat = rng.lognormal(0, 0.3, n)
        D = pd.DataFrame({
            "gender": rng.choice(["M", "F"], size=n),
            "age_band": rng.choice(["young", "old"], size=n),
        })
        return y_hat, D

    def test_single_obs_per_intersection_no_crash(self):
        """Very small groups should not crash the audit (only warn)."""
        rng = np.random.default_rng(5)
        # 4 policies: one per (gender x age_band) cell
        y_hat = rng.lognormal(0, 0.3, 4)
        D = pd.DataFrame({
            "gender": ["M", "M", "F", "F"],
            "age_band": ["young", "old", "young", "old"],
        })
        audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
        # Should not raise; may emit warnings
        report = audit.audit(y_hat, D)
        assert not np.isnan(report.ccDcov)

    def test_three_protected_attrs(self):
        """Three attrs: ccDcov should be computed without error."""
        rng = np.random.default_rng(3)
        n = 300
        y_hat = rng.lognormal(0, 0.5, n)
        D = pd.DataFrame({
            "gender": rng.choice(["M", "F"], size=n),
            "age_band": rng.choice(["young", "old"], size=n),
            "region": rng.choice(["north", "south"], size=n),
        })
        audit = IntersectionalFairnessAudit(
            protected_attrs=["gender", "age_band", "region"]
        )
        report = audit.audit(y_hat, D)
        assert not np.isnan(report.ccDcov)
        assert report.ccDcov >= 0.0

    def test_ccDcov_non_negative(self):
        """CCdCov (as a measure of dependence squared) should be >= 0."""
        y_hat, D = self._make_data()
        audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
        report = audit.audit(y_hat, D)
        assert report.ccDcov >= 0.0

    def test_eta_non_negative(self):
        """Intersectional residual eta >= 0."""
        y_hat, D = self._make_data()
        audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
        report = audit.audit(y_hat, D)
        assert report.eta >= 0.0

    def test_marginals_plus_eta_equals_ccDcov(self):
        """CCdCov = sum(marginal dCov^2) + eta — key decomposition property."""
        y_hat, D = self._make_data(n=200)
        audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
        report = audit.audit(y_hat, D)
        reconstructed = sum(report.marginal_dcov.values()) + report.eta
        assert reconstructed == pytest.approx(report.ccDcov, rel=1e-5)

    def test_audit_report_subgroup_stats_has_correct_columns(self):
        y_hat, D = self._make_data()
        audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
        report = audit.audit(y_hat, D)
        required_cols = {"subgroup", "n", "mean_prediction"}
        assert required_cols.issubset(set(report.subgroup_statistics.columns))

    def test_regulariser_lambda_zero_returns_zero(self):
        rng = np.random.default_rng(7)
        n = 100
        y_hat = rng.lognormal(0, 0.3, n)
        D = pd.DataFrame({"gender": rng.choice(["M", "F"], size=n)})
        reg = DistanceCovFairnessRegulariser(
            protected_attrs=["gender"], lambda_val=0.0
        )
        penalty = reg.penalty(y_hat, D)
        assert penalty == pytest.approx(0.0, abs=1e-12)

    def test_regulariser_positive_lambda_positive_penalty_dependent_data(self):
        """Strongly dependent data should produce positive penalty."""
        rng = np.random.default_rng(9)
        n = 200
        D_arr = rng.choice(["M", "F"], size=n)
        # Predictions perfectly determined by gender => max dependence
        y_hat = np.where(D_arr == "M", 200.0, 100.0) + rng.normal(scale=0.1, size=n)
        D = pd.DataFrame({"gender": D_arr})
        reg = DistanceCovFairnessRegulariser(
            protected_attrs=["gender"], lambda_val=1.0
        )
        penalty = reg.penalty(y_hat, D)
        assert penalty > 0.0


# ===========================================================================
# C. MultiStateTransitionFairness — additional cases
# ===========================================================================


class TestKolmogorovPremiumDiscountRate:
    """Discount rate effect: higher rate => lower premium (time value of money)."""

    @pytest.fixture
    def intensity_fns(self):
        return {
            "healthy->sick": lambda age: 0.05,
            "sick->healthy": lambda age: 0.25,
            "sick->dead": lambda age: 0.02,
            "healthy->dead": lambda age: 0.005,
        }

    def test_higher_discount_lower_premium(self, intensity_fns):
        cash_flows = {"healthy->sick": 1.0}
        states = ["healthy", "sick", "dead"]

        calc_low = KolmogorovPremiumCalculator(
            states=states, discount_rate=0.01, dt=0.1, max_age=65.0
        )
        calc_high = KolmogorovPremiumCalculator(
            states=states, discount_rate=0.10, dt=0.1, max_age=65.0
        )

        epv_low = calc_low.compute_premium(intensity_fns, cash_flows, entry_age=30.0)
        epv_high = calc_high.compute_premium(intensity_fns, cash_flows, entry_age=30.0)

        assert epv_low > epv_high, (
            f"Higher discount rate should give lower premium: "
            f"low-rate={epv_low:.4f}, high-rate={epv_high:.4f}"
        )

    def test_older_entry_age_lower_remaining_coverage(self, intensity_fns):
        """Starting later = fewer years of coverage => lower EPV."""
        cash_flows = {"healthy->sick": 1.0}
        states = ["healthy", "sick", "dead"]
        calc = KolmogorovPremiumCalculator(
            states=states, discount_rate=0.05, dt=0.1, max_age=70.0
        )
        epv_young = calc.compute_premium(intensity_fns, cash_flows, entry_age=30.0)
        epv_old = calc.compute_premium(intensity_fns, cash_flows, entry_age=60.0)
        assert epv_young > epv_old

    def test_generator_correct_size(self):
        states = ["healthy", "sick", "dead"]
        calc = KolmogorovPremiumCalculator(states=states, discount_rate=0.05)
        intensity_fns = {"healthy->sick": lambda a: 0.05}
        Q = calc._build_generator(intensity_fns, age=40.0)
        assert Q.shape == (3, 3)


class TestMultiStateFairnessReportFields:
    """MultiStateFairnessReport field access and str representation."""

    def test_report_to_dict_has_expected_keys(self):
        report = MultiStateFairnessReport(
            transitions=["healthy->sick"],
            premium_before={"M": 0.25, "F": 0.20},
            premium_after={"M": 0.22, "F": 0.22},
            transition_corrections={"healthy->sick": -0.03},
            n_policies=400,
            protected_attrs=["gender"],
        )
        s = report.summary()
        assert "gender" in s
        assert "0.22" in s or "0.25" in s  # some premium value

    def test_report_empty_transitions_edge_case(self):
        """Report with empty transitions list should not crash summary()."""
        report = MultiStateFairnessReport(
            transitions=[],
            premium_before={"M": 0.20},
            premium_after={"M": 0.20},
            transition_corrections={},
            n_policies=100,
            protected_attrs=["gender"],
        )
        s = report.summary()
        assert isinstance(s, str)


# ===========================================================================
# D. OptimalLDPMechanism — numerical constraint verification
# ===========================================================================


class TestOptimalLDPConstraints:
    """Verify that Q satisfies epsilon-LDP column constraints and row-stochastic."""

    def _check_row_stochastic(self, Q: np.ndarray, atol: float = 1e-8):
        K = Q.shape[0]
        assert Q.shape == (K, K), f"Q should be square, got {Q.shape}"
        assert np.all(Q >= -atol), f"Q has negative entry: {Q.min():.2e}"
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=atol, err_msg="Q rows don't sum to 1")

    def _check_ldp(self, Q: np.ndarray, epsilon: float, atol: float = 1e-6):
        """Check P(Z=i|X=j) / P(Z=i|X=j') <= exp(epsilon) for all i, j, j'."""
        exp_eps = np.exp(epsilon)
        K = Q.shape[0]
        for i in range(K):
            for j in range(K):
                for j_prime in range(K):
                    if j != j_prime and Q[j_prime, i] > atol:
                        ratio = Q[j, i] / Q[j_prime, i]
                        assert ratio <= exp_eps + atol, (
                            f"LDP violated: Q[{j},{i}]/Q[{j_prime},{i}]={ratio:.4f} "
                            f"> exp({epsilon})={exp_eps:.4f}"
                        )

    def test_binary_row_stochastic(self):
        mech = OptimalLDPMechanism(epsilon=2.0, k=2, group_prevalences=np.array([0.3, 0.7]))
        Q = mech.perturbation_matrix
        self._check_row_stochastic(Q)

    def test_binary_satisfies_ldp(self):
        mech = OptimalLDPMechanism(epsilon=1.5, k=2, group_prevalences=np.array([0.4, 0.6]))
        Q = mech.perturbation_matrix
        self._check_ldp(Q, epsilon=1.5)

    def test_three_group_row_stochastic(self):
        mech = OptimalLDPMechanism(
            epsilon=2.0, k=3, group_prevalences=np.array([0.2, 0.5, 0.3])
        )
        Q = mech.perturbation_matrix
        self._check_row_stochastic(Q)

    def test_three_group_satisfies_ldp(self):
        mech = OptimalLDPMechanism(
            epsilon=2.0, k=3, group_prevalences=np.array([0.2, 0.5, 0.3])
        )
        Q = mech.perturbation_matrix
        self._check_ldp(Q, epsilon=2.0)

    def test_large_epsilon_diagonal_dominant(self):
        """Very large epsilon => mechanism approaches deterministic (diagonal ~ 1)."""
        mech = OptimalLDPMechanism(
            epsilon=10.0, k=2, group_prevalences=np.array([0.5, 0.5])
        )
        Q = mech.perturbation_matrix
        # Diagonal entries should be close to 1 for large epsilon
        assert Q[0, 0] > 0.9, f"Q[0,0]={Q[0,0]:.4f} expected > 0.9 for eps=10"
        assert Q[1, 1] > 0.9, f"Q[1,1]={Q[1,1]:.4f} expected > 0.9 for eps=10"

    def test_binary_minority_gets_higher_diagonal(self):
        """Theorem 2: minority group (lower prevalence) gets preferential noise."""
        # Group 0 is minority (prevalence 0.2)
        mech = OptimalLDPMechanism(
            epsilon=1.0, k=2, group_prevalences=np.array([0.2, 0.8])
        )
        Q = mech.perturbation_matrix
        # Minority (group 0) should have higher correct-response probability
        assert Q[0, 0] > Q[1, 1], (
            f"Minority group 0 should have Q[0,0]={Q[0,0]:.4f} > Q[1,1]={Q[1,1]:.4f}"
        )

    def test_unfairness_bound_in_range(self):
        mech = OptimalLDPMechanism(
            epsilon=2.0, k=2, group_prevalences=np.array([0.4, 0.6])
        )
        bound = mech.unfairness_bound()
        assert 0.0 <= bound <= 0.5 + 1e-8

    def test_unfairness_bound_monotone_in_epsilon(self):
        """Higher epsilon => less noise => smaller unfairness bound."""
        p = np.array([0.3, 0.7])
        mech_low = OptimalLDPMechanism(epsilon=0.5, k=2, group_prevalences=p)
        mech_high = OptimalLDPMechanism(epsilon=3.0, k=2, group_prevalences=p)
        assert mech_low.unfairness_bound() >= mech_high.unfairness_bound() - 1e-8

    def test_privatise_output_in_range(self):
        mech = OptimalLDPMechanism(
            epsilon=2.0, k=3, group_prevalences=np.array([0.3, 0.4, 0.3])
        )
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 3, size=200)
        privatised = mech.privatise(labels, rng=rng)
        assert privatised.shape == (200,)
        assert np.all((privatised >= 0) & (privatised < 3))

    def test_privatise_invalid_labels_raises(self):
        mech = OptimalLDPMechanism(
            epsilon=2.0, k=2, group_prevalences=np.array([0.5, 0.5])
        )
        with pytest.raises(ValueError, match="true_labels must be"):
            mech.privatise(np.array([0, 1, 2]))  # group 2 doesn't exist for k=2

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            OptimalLDPMechanism(epsilon=0.0, k=2)

    def test_k_one_raises(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            OptimalLDPMechanism(epsilon=1.0, k=1)

    def test_prevalences_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.3, 0.3]))

    def test_prevalences_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            OptimalLDPMechanism(epsilon=1.0, k=2, group_prevalences=np.array([0.3, 0.3, 0.4]))

    def test_fit_updates_perturbation_matrix(self):
        """fit() with new prevalences should update Q."""
        mech = OptimalLDPMechanism(epsilon=2.0, k=2)
        Q_uniform = mech.perturbation_matrix.copy()

        mech.fit(np.array([0.1, 0.9]))  # very imbalanced
        Q_imbalanced = mech.perturbation_matrix

        assert not np.allclose(Q_uniform, Q_imbalanced), (
            "fit() with imbalanced prevalences should change Q"
        )


class TestLDPEpsilonAdvisorExtended:
    """Additional LDPEpsilonAdvisor numerical checks."""

    def test_recommend_returns_expected_keys(self):
        advisor = LDPEpsilonAdvisor(n_samples=1000, k=2)
        result = advisor.recommend()
        assert "epsilon" in result
        assert "C1" in result
        assert "pi" in result
        assert "gen_bound" in result

    def test_larger_target_inflation_allows_smaller_epsilon(self):
        """A more lenient target inflation allows smaller epsilon (more noise)."""
        advisor_strict = LDPEpsilonAdvisor(n_samples=1000, k=2, target_bound_inflation=0.05)
        advisor_lenient = LDPEpsilonAdvisor(n_samples=1000, k=2, target_bound_inflation=0.50)
        result_strict = advisor_strict.recommend()
        result_lenient = advisor_lenient.recommend()
        # Stricter target => needs larger epsilon (less noise)
        assert result_strict["epsilon"] >= result_lenient["epsilon"] - 0.01

    def test_sweep_returns_polars_dataframe(self):
        advisor = LDPEpsilonAdvisor(n_samples=1000, k=2)
        df = advisor.sweep(epsilons=np.array([0.5, 1.0, 2.0]))
        assert isinstance(df, pl.DataFrame)
        assert "epsilon" in df.columns
        assert "C1" in df.columns
        assert len(df) == 3

    def test_c1_monotone_decreasing_in_epsilon(self):
        """C1 (amplification factor) decreases as epsilon increases."""
        advisor = LDPEpsilonAdvisor(n_samples=1000, k=2)
        epsilons = np.array([0.5, 1.0, 2.0, 3.0])
        df = advisor.sweep(epsilons=epsilons)
        c1_vals = df["C1"].to_numpy()
        # C1 should be decreasing in epsilon (more privacy budget = less amplification)
        assert np.all(np.diff(c1_vals) <= 0.01), (
            f"C1 not monotone decreasing: {c1_vals}"
        )

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            LDPEpsilonAdvisor(n_samples=1000, k=1)

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError):
            LDPEpsilonAdvisor(n_samples=0, k=2)


# ===========================================================================
# E. LipschitzMetric — analytical checks and custom distance
# ===========================================================================


class TestLipschitzMetricAnalytical:
    """Analytical results for LipschitzMetric."""

    def test_constant_predictions_gives_zero(self):
        """f(x) = c for all x => Lipschitz constant = 0."""
        X = np.random.default_rng(0).normal(size=(100, 3))
        predictions = np.full(100, 100.0)
        metric = LipschitzMetric(log_predictions=False, n_pairs=500, random_seed=0)
        result = metric.compute(X, predictions)
        assert result.lipschitz_constant == pytest.approx(0.0, abs=1e-10)

    def test_linear_function_known_lipschitz(self):
        """f(x) = x[0] in 1-D, Euclidean distance => Lipschitz = 1.0."""
        n = 200
        X = np.linspace(0, 10, n).reshape(-1, 1)
        predictions = X[:, 0]  # f(x) = x
        metric = LipschitzMetric(log_predictions=False, n_pairs=n * (n - 1) // 2, random_seed=0)
        result = metric.compute(X, predictions)
        # Sampled max should be close to 1.0 (exact Lipschitz of f(x) = x)
        assert result.lipschitz_constant == pytest.approx(1.0, rel=0.01)

    def test_result_has_all_fields(self):
        X = np.random.default_rng(1).normal(size=(50, 2))
        predictions = np.exp(X[:, 0])
        metric = LipschitzMetric(n_pairs=100, random_seed=1)
        result = metric.compute(X, predictions)
        assert isinstance(result, LipschitzResult)
        assert result.n_pairs_sampled > 0
        assert result.max_ratio == result.lipschitz_constant
        assert result.p95_ratio <= result.max_ratio + 1e-10
        assert result.p50_ratio <= result.p95_ratio + 1e-10

    def test_log_predictions_false_uses_raw_diff(self):
        """In linear space, double all predictions => double the Lipschitz constant."""
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, (100, 2))
        predictions = rng.uniform(1, 10, 100)

        metric = LipschitzMetric(log_predictions=False, n_pairs=500, random_seed=2)
        result1 = metric.compute(X, predictions)
        result2 = metric.compute(X, predictions * 2)

        # Doubling predictions doubles differences => doubles the Lipschitz constant
        assert result2.lipschitz_constant == pytest.approx(
            result1.lipschitz_constant * 2, rel=0.01
        )

    def test_log_predictions_scale_invariant(self):
        """In log-space, scaling predictions by a constant has no effect."""
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (100, 2))
        predictions = rng.uniform(1, 10, 100)

        metric = LipschitzMetric(log_predictions=True, n_pairs=500, random_seed=3)
        result1 = metric.compute(X, predictions)
        result2 = metric.compute(X, predictions * 5)

        # log(k*f(x)) - log(k*f(x')) = log(f(x)) - log(f(x')) => same L
        assert result1.lipschitz_constant == pytest.approx(
            result2.lipschitz_constant, rel=0.01
        )

    def test_custom_distance_function(self):
        """Custom distance function changes the Lipschitz estimate."""
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (100, 2))
        predictions = rng.uniform(1, 10, 100)

        # Manhattan distance
        def manhattan(x1, x2):
            return float(np.sum(np.abs(x1 - x2)))

        metric_euclidean = LipschitzMetric(
            log_predictions=False, n_pairs=500, random_seed=4
        )
        metric_manhattan = LipschitzMetric(
            distance_fn=manhattan, log_predictions=False, n_pairs=500, random_seed=4
        )
        r_euc = metric_euclidean.compute(X, predictions)
        r_man = metric_manhattan.compute(X, predictions)

        # In 2D: Euclidean <= Manhattan, so Lipschitz should be >= with Euclidean
        assert r_euc.lipschitz_constant >= r_man.lipschitz_constant - 1e-6

    def test_too_few_policies_raises(self):
        X = np.array([[1.0, 2.0]])
        predictions = np.array([5.0])
        metric = LipschitzMetric()
        with pytest.raises(ValueError, match="At least 2"):
            metric.compute(X, predictions)

    def test_non_positive_predictions_log_mode_raises(self):
        X = np.random.default_rng(5).normal(size=(50, 2))
        predictions = np.ones(50)
        predictions[10] = -1.0  # one negative value
        metric = LipschitzMetric(log_predictions=True)
        with pytest.raises(ValueError, match="strictly positive"):
            metric.compute(X, predictions)

    def test_log_result_has_flag_set(self):
        X = np.random.default_rng(6).normal(size=(50, 2))
        predictions = np.exp(X[:, 0])
        metric = LipschitzMetric(log_predictions=True, n_pairs=100, random_seed=6)
        result = metric.compute(X, predictions)
        assert result.log_predictions is True


class TestLipschitzFourObjectiveIntegration:
    """LipschitzMetric wired into FairnessProblem (four-objective mode)."""

    def _make_problem(self):
        from insurance_fairness.pareto import FairnessProblem

        class _ConstModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        rng = np.random.default_rng(10)
        n = 80
        X = pl.DataFrame({
            "gender": (["M"] * 40 + ["F"] * 40),
            "age": rng.integers(20, 70, n).tolist(),
            "ncd": rng.uniform(0, 5, n).tolist(),
        })
        y = rng.uniform(50, 200, n)
        exposure = np.ones(n)
        model = _ConstModel()
        return FairnessProblem(
            models={"base": model},
            X=X,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "ncd"],
            lipschitz_log_predictions=False,
        )

    def test_four_objectives_active(self):
        prob = self._make_problem()
        assert prob.n_obj == 4

    def test_evaluate_returns_four_values(self):
        prob = self._make_problem()
        result = prob.evaluate(np.array([1.0]))
        assert result.shape == (4,)

    def test_fourth_objective_finite(self):
        prob = self._make_problem()
        result = prob.evaluate(np.array([1.0]))
        assert np.isfinite(result[3])


# ===========================================================================
# F. PrivatizedFairPricer — additional coverage
# ===========================================================================


def _make_pricer_data(n: int = 600, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    D = rng.integers(0, K, n)
    X = rng.normal(0, 1, (n, 4))
    y = rng.poisson(0.07 + 0.03 * D).astype(float)
    # Apply K-RR with pi ~ 0.88 for K=2
    pi = np.exp(2.0) / (K - 1 + np.exp(2.0))
    S = D.copy()
    for i in range(n):
        if rng.random() > pi:
            others = [k for k in range(K) if k != D[i]]
            S[i] = rng.choice(others)
    return X, y, S


class TestPrivatizedFairPricerReferenceDistributions:
    """Test empirical and custom reference distributions."""

    def test_empirical_reference_runs(self):
        X, y, S = _make_pricer_data()
        pricer = PrivatizedFairPricer(
            epsilon=2.0, n_groups=2, reference_distribution="empirical"
        )
        pricer.fit(X, y, S)
        preds = pricer.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_custom_pstar_array(self):
        """Passing a numpy array as reference_distribution."""
        X, y, S = _make_pricer_data()
        p_star = np.array([0.3, 0.7])  # custom non-uniform
        pricer = PrivatizedFairPricer(
            epsilon=2.0, n_groups=2, reference_distribution=p_star
        )
        pricer.fit(X, y, S)
        preds = pricer.predict(X)
        assert preds.shape == (len(X),)

    def test_repr_contains_class_name(self):
        pricer = PrivatizedFairPricer(epsilon=2.0, n_groups=2)
        assert "PrivatizedFairPricer" in repr(pricer) or "PrivatizedFairPricer" in str(pricer)

    def test_correction_summary_pi_in_unit_interval(self):
        X, y, S = _make_pricer_data()
        pricer = PrivatizedFairPricer(epsilon=2.0, n_groups=2)
        pricer.fit(X, y, S)
        summary = pricer.correction_summary()
        assert 0.0 < summary["pi"] <= 1.0

    def test_minimum_sample_size_grows_with_groups(self):
        """More groups => larger minimum sample size needed."""
        X2, y2, S2 = _make_pricer_data(K=2)
        X3, y3, S3 = _make_pricer_data(n=900, K=3, seed=1)

        p2 = PrivatizedFairPricer(epsilon=2.0, n_groups=2)
        p2.fit(X2, y2, S2)

        p3 = PrivatizedFairPricer(epsilon=2.0, n_groups=3)
        p3.fit(X3, y3, S3)

        assert p3.minimum_sample_size() >= p2.minimum_sample_size()


# ===========================================================================
# G. ProxyDiscriminationMeasure — additional coverage
# ===========================================================================


class TestProxyDiscriminationEdgeCases:
    """Edge cases not in the main sensitivity test file."""

    def test_mu_hat_3d_raises(self):
        """mu_hat with ndim=3 should raise ValueError."""
        n = 100
        rng = np.random.default_rng(0)
        y = rng.normal(size=n)
        X = rng.normal(size=(n, 2))
        D = rng.choice(2, size=n)
        mu_bad = rng.normal(size=(n, 2, 2))  # 3D
        m = ProxyDiscriminationMeasure()
        with pytest.raises(ValueError, match="ndim"):
            m.fit(y, X, D, mu_hat=mu_bad)

    def test_exposure_weighted_false(self):
        """exposure_weighted=False should still compute a valid PD."""
        rng = np.random.default_rng(1)
        n = 200
        X = rng.normal(size=(n, 2))
        D = rng.choice(2, size=n)
        y = X[:, 0] + rng.normal(scale=0.3, size=n)
        pi = X[:, 0] + rng.normal(scale=0.1, size=n)
        m = ProxyDiscriminationMeasure(exposure_weighted=False)
        m.fit(y, X, D, mu_hat=pi)
        assert 0.0 <= m.pd_score <= 1.0
        assert not np.isnan(m.pd_score)

    def test_pandas_dataframe_d_raises_or_converts(self):
        """D as a pandas Series (1D) should work after np.asarray conversion."""
        rng = np.random.default_rng(2)
        n = 150
        X = rng.normal(size=(n, 2))
        D_series = pd.Series(rng.choice(["A", "B"], size=n))
        y = rng.normal(size=n)
        pi = rng.normal(loc=100, scale=10, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D_series.to_numpy(), mu_hat=pi)
        assert not np.isnan(m.pd_score)

    def test_all_same_group_single_category(self):
        """D with only one category: trivially D-free."""
        rng = np.random.default_rng(3)
        n = 100
        X = rng.normal(size=(n, 2))
        D = np.zeros(n, dtype=int)  # all same group
        y = rng.normal(size=n)
        pi = X[:, 0] + rng.normal(scale=0.1, size=n)
        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)
        # Single category: UF must be 0 (no between-group variance)
        assert m.uf_score == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# H. ShapleyAttribution / SobolAttribution — edge cases
# ===========================================================================


class TestShapleyAttributionEdgeCases:

    def test_zero_variance_lambda_gives_zero_shapley(self):
        """Lambda = constant => Var(Lambda) = 0 => all Shapley values = 0."""
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 3))
        pi = rng.normal(loc=100, scale=10, size=n)
        Lambda = np.zeros(n)  # constant residual

        sh = ShapleyAttribution()
        sh.fit(Lambda, X, pi)
        # With zero-variance Lambda, all attributions should be zero
        np.testing.assert_allclose(
            sh.attributions_["shapley_pd"].values, 0.0, atol=1e-10
        )

    def test_single_feature_exact_shapley(self):
        """For p=1, the exact Shapley value equals the full value function."""
        rng = np.random.default_rng(1)
        n = 200
        X = rng.normal(size=(n, 1))
        pi = rng.normal(loc=100, scale=5, size=n)
        # Lambda correlated with X[:,0]
        Lambda = X[:, 0] * 3.0 + rng.normal(scale=0.5, size=n)

        sh = ShapleyAttribution(exact_threshold=12)  # exact for p=1
        sh.fit(Lambda, X, pi, feature_names=["x0"])

        assert len(sh.attributions_) == 1
        assert sh.attributions_["feature"].iloc[0] == "x0"
        # Single feature gets all the value
        assert sh.attributions_["shapley_pd"].iloc[0] == pytest.approx(
            sh.pd_surrogate_, rel=1e-4
        )

    def test_shapley_non_negative_for_orthogonal_features(self):
        """Orthogonal features: each Shapley value should be non-negative."""
        rng = np.random.default_rng(2)
        n = 400
        # Three independent features
        X = rng.normal(size=(n, 3))
        pi = rng.uniform(50, 150, n)
        Lambda = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)

        sh = ShapleyAttribution(random_state=0)
        sh.fit(Lambda, X, pi, feature_names=["signal", "noise1", "noise2"])
        shapley_vals = sh.attributions_["shapley_pd"].values
        # Allow small negative due to Monte Carlo noise
        assert np.all(shapley_vals >= -0.02), (
            f"Negative Shapley values: {shapley_vals}"
        )


class TestSobolAttributionEdgeCases:

    def test_zero_variance_lambda_gives_zero_indices(self):
        """Lambda = constant => all Sobol indices = 0."""
        rng = np.random.default_rng(5)
        n = 200
        X = rng.normal(size=(n, 3))
        pi = rng.normal(loc=100, scale=10, size=n)
        Lambda = np.zeros(n)

        sa = SobolAttribution()
        sa.fit(Lambda, X, pi)
        fo = sa.attributions_["first_order_pd"].values
        to = sa.attributions_["total_pd"].values
        np.testing.assert_allclose(fo, 0.0, atol=1e-10)
        np.testing.assert_allclose(to, 0.0, atol=1e-10)

    def test_single_feature_first_order_dominates(self):
        """For a single relevant feature, first-order index should dominate."""
        rng = np.random.default_rng(6)
        n = 300
        X = rng.normal(size=(n, 3))
        pi = rng.normal(loc=100, scale=10, size=n)
        Lambda = X[:, 0] * 5.0 + rng.normal(scale=0.1, size=n)

        sa = SobolAttribution(random_state=0)
        sa.fit(Lambda, X, pi, feature_names=["x0", "noise1", "noise2"])
        attrs = sa.attributions_.set_index("feature")

        # x0 should have the highest first-order index
        assert attrs.loc["x0", "first_order_pd"] > max(
            attrs.loc["noise1", "first_order_pd"],
            attrs.loc["noise2", "first_order_pd"],
        )


# ===========================================================================
# I. LocalizedParityCorrector — three groups and held-out audit
# ===========================================================================


class TestLocalizedParityThreeGroups:
    """Localized parity with three groups."""

    @pytest.fixture
    def three_group_data(self):
        rng = np.random.default_rng(10)
        n = 600
        g0 = rng.gamma(3, 100, n // 3)
        g1 = rng.gamma(3, 100, n // 3) + 100.0
        g2 = rng.gamma(3, 100, n // 3) + 200.0
        preds = np.concatenate([g0, g1, g2])
        sensitive = np.array(["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n // 3))
        return preds, sensitive

    def test_three_groups_audit_no_error(self, three_group_data):
        preds, sensitive = three_group_data
        audit = LocalizedParityAudit(thresholds=[200.0, 400.0, 600.0])
        report = audit.audit(preds, sensitive)
        assert isinstance(report, LocalizedParityReport)
        assert report.max_disparity >= 0.0

    def test_three_groups_corrector_reduces_disparity(self, three_group_data):
        preds, sensitive = three_group_data
        thresholds = [200.0, 400.0, 600.0]

        audit = LocalizedParityAudit(thresholds=thresholds)
        pre_report = audit.audit(preds, sensitive)

        corrector = LocalizedParityCorrector(thresholds=thresholds, mode="quantile")
        corrector.fit(preds, sensitive)
        post_report = corrector.audit()

        assert post_report.max_disparity < pre_report.max_disparity

    def test_three_groups_corrector_transform_shape(self, three_group_data):
        preds, sensitive = three_group_data
        corrector = LocalizedParityCorrector(thresholds=[300.0, 500.0], mode="quantile")
        corrector.fit(preds, sensitive)
        corrected = corrector.transform(preds, sensitive)
        assert corrected.shape == preds.shape

    def test_audit_predictions_on_held_out_data(self, three_group_data):
        """audit_predictions() should accept held-out data not used in fit()."""
        preds, sensitive = three_group_data
        n = len(preds)
        train_mask = np.arange(n) < n * 2 // 3
        test_mask = ~train_mask

        corrector = LocalizedParityCorrector(
            thresholds=[200.0, 400.0, 600.0], mode="quantile"
        )
        corrector.fit(preds[train_mask], sensitive[train_mask])

        corrected_test = corrector.transform(preds[test_mask], sensitive[test_mask])
        report = corrector.audit_predictions(corrected_test, sensitive[test_mask])
        assert isinstance(report, LocalizedParityReport)
        assert report.max_disparity >= 0.0


class TestLocalizedParityReportAttributes:
    def test_report_has_group_cdf_table_with_correct_columns(self):
        rng = np.random.default_rng(20)
        preds = rng.gamma(3, 100, 500)
        sensitive = np.array(["A"] * 250 + ["B"] * 250)
        audit = LocalizedParityAudit(thresholds=[300.0, 500.0])
        report = audit.audit(preds, sensitive)
        assert isinstance(report.group_cdf_table, pl.DataFrame)
        required_cols = {"group", "threshold", "empirical_cdf", "target_cdf", "deviation"}
        assert required_cols.issubset(set(report.group_cdf_table.columns))

    def test_report_has_correct_number_of_rows(self):
        rng = np.random.default_rng(21)
        preds = rng.gamma(3, 100, 500)
        sensitive = np.array(["A"] * 250 + ["B"] * 250)
        thresholds = [300.0, 500.0, 700.0]
        audit = LocalizedParityAudit(thresholds=thresholds)
        report = audit.audit(preds, sensitive)
        # 2 groups x 3 thresholds = 6 rows
        assert len(report.group_cdf_table) == 6

    def test_discretization_cost_matches_formula(self):
        rng = np.random.default_rng(22)
        preds = rng.gamma(3, 100, 500)
        sensitive = np.array(["A"] * 250 + ["B"] * 250)
        M = 5
        thresholds = list(np.linspace(200, 700, M))
        audit = LocalizedParityAudit(thresholds=thresholds)
        report = audit.audit(preds, sensitive)
        # discretization_cost = 1/M
        assert report.discretization_cost == pytest.approx(1.0 / M, abs=1e-10)


# ===========================================================================
# J. Integration: sensitivity -> LipschitzMetric pipeline
# ===========================================================================


class TestSensitivityLipschitzIntegration:
    """Verify ProxyDiscriminationMeasure + LipschitzMetric can run together."""

    def test_pd_then_lipschitz_no_error(self):
        rng = np.random.default_rng(99)
        n = 300
        X = rng.normal(size=(n, 3))
        D = rng.choice(2, size=n)
        y = X[:, 0] + rng.normal(scale=0.3, size=n)
        pi = X[:, 0] + D * 0.5 + rng.normal(scale=0.1, size=n)

        m = ProxyDiscriminationMeasure()
        m.fit(y, X, D, mu_hat=pi)

        metric = LipschitzMetric(log_predictions=False, n_pairs=200, random_seed=99)
        result = metric.compute(X, pi - m.closest_admissible + pi.mean())
        assert isinstance(result.lipschitz_constant, float)
        assert result.lipschitz_constant >= 0.0


# ===========================================================================
# K. SequentialOTCorrector: epsilon validation at construction
# ===========================================================================


class TestSequentialOTEpsilonValidation:
    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError):
            SequentialOTCorrector(["group"], epsilon=-0.1)

    def test_epsilon_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            SequentialOTCorrector(["group"], epsilon=1.5)

    def test_epsilon_list_with_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SequentialOTCorrector(["gender", "age"], epsilon=[0.5, -0.1])

    def test_epsilon_zero_is_valid(self):
        """Epsilon = 0 means full correction — should construct without error."""
        c = SequentialOTCorrector(["group"], epsilon=0.0)
        assert c._epsilons == [0.0]

    def test_epsilon_one_is_valid(self):
        """Epsilon = 1 means no correction — should construct without error."""
        c = SequentialOTCorrector(["group"], epsilon=1.0)
        assert c._epsilons == [1.0]
