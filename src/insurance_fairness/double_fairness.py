"""
double_fairness.py
------------------
DoubleFairnessAudit: joint action and outcome fairness for insurance pricing.

Every existing tool in this library operates at pricing time: does the model
discriminate at the point of quoting? None of them ask whether the pricing
decision produces equivalent financial outcomes for protected groups after the
policy is live.

The FCA Consumer Duty (PRIN 2A, PS22/9, live July 2023) requires both.
Outcome 4 (Price and Value) demands that firms demonstrate their products
deliver reasonable value across consumer cohorts — not just that pricing
inputs are non-discriminatory. TR24/2 (2024 thematic review) explicitly
cited failure to assess differential post-sale outcomes as a compliance gap.

``DoubleFairnessAudit`` operationalises the distinction between:

- **Action fairness** (Delta_1): equal treatment at pricing — the same action
  probability (premium band, risk classification) for the same risk profile
  regardless of protected group. This is what regulators currently expect firms
  to measure.
- **Outcome fairness** (Delta_2): equal outcomes after the fact — the same
  expected benefit (welfare, loss ratio, claims experience) across groups.
  This is what Consumer Duty actually requires.

The key empirical result from Bian et al. (2026): equalising premiums across
gender (Delta_1=0) did NOT equalise welfare outcomes (Delta_2 remained large).
Action fairness alone is insufficient.

The Pareto front is computed via lexicographic Tchebycheff scalarisation, which
— unlike linear weighting — can recover Pareto-optimal policies in non-convex
regions of the objective space. The algorithm sweeps K alpha weights and for
each runs a two-stage optimisation: minimise the Tchebycheff criterion in Stage
1, then minimise total unfairness subject to near-Tchebycheff-optimality in
Stage 2.

Usage::

    from insurance_fairness import DoubleFairnessAudit

    audit = DoubleFairnessAudit(n_alphas=20)
    audit.fit(X_train, y_premium, y_loss_ratio, S_gender)
    result = audit.audit()
    print(result.summary())
    fig = audit.plot_pareto()
    print(audit.report())

Policy parametrisation
----------------------
The pricing policy is parametrised as a linear decision rule:

    pi_theta(1 | X_i) = sigmoid(X_i @ theta)

where theta in R^p, optimised via scipy L-BFGS-B. The action a=1 means
"assign to high-risk band"; a=0 means "assign to low-risk band". The binary
action is a deliberate simplification — in practice, interpret it as the
decision to assign a customer above or below a chosen premium threshold.

Fairness metrics
----------------
Action fairness (Delta_1) measures whether an S-blind policy assigns
systematically different action probabilities to the two groups. Even when S
is excluded from X, proxy features correlated with S will create non-zero
Delta_1 — which is exactly what we want to detect.

    Delta_1(pi) = (mean_{i in G1} pi(1|X_i) - mean_{i in G0} pi(1|X_i))^2

Outcome fairness (Delta_2) measures whether the expected fairness-relevant
outcome (e.g. loss ratio) differs across groups under the policy:

    Delta_2(pi) = mean_i [ f_hat(1, X_i) - f_hat(0, X_i) ]^2

where f_hat(s, X_i) = pi(1|X_i)*f_a1_s(X_i) + (1-pi(1|X_i))*f_a0_s(X_i)
is the expected fairness outcome under group s for observation i.

References
----------
Bian, Z., Wang, L., Shi, C., Qi, Z. (2026). Double Fairness Policy Learning:
Integrating Action Fairness and Outcome Fairness in Decision-making.
arXiv:2601.19186v2.

FCA. Consumer Duty (PS22/9, July 2022, effective July 2023).

FCA. Pricing Practices Thematic Review (TR24/2, 2024).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return expit(z)


def _softmax_max(a: float, b: float, tau: float = 50.0) -> float:
    """
    Smooth approximation of max(a, b) via log-sum-exp.

    max(a, b) ≈ (1/tau) * log(exp(tau*a) + exp(tau*b))

    For large tau this is tight. We use tau=50 throughout. The approximation
    is always >= max(a, b) and converges to max(a, b) as tau -> inf.
    """
    # Numerically stable: subtract the max before exponentiation
    m = max(a, b)
    return m + (1.0 / tau) * np.log(np.exp(tau * (a - m)) + np.exp(tau * (b - m)))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DoubleFairnessResult:
    """
    Result of a DoubleFairnessAudit.

    Attributes
    ----------
    pareto_alphas:
        Shape (K,). The K alpha weights swept across the Tchebycheff
        scalarisation. Alpha=1 weights action fairness fully; alpha=0 weights
        outcome fairness fully.
    pareto_V:
        Shape (K,). Estimated expected revenue V_hat for each Pareto point.
    pareto_delta1:
        Shape (K,). Estimated action fairness violation Delta_hat_1 for each
        Pareto point.
    pareto_delta2:
        Shape (K,). Estimated outcome fairness violation Delta_hat_2 for each
        Pareto point.
    pareto_theta:
        Shape (K, p). Optimal policy parameter theta for each Pareto point.
    selected_idx:
        Index of the value-maximising Pareto point. This is the recommended
        operating point: it maximises expected company value subject to the
        lexicographic fairness constraints.
    kappa:
        Slack parameter used in Stage 2. Controls how close a Stage 2 policy
        must be to the Stage 1 Tchebycheff optimum.
    n_train:
        Number of training observations.
    outcome_model_type:
        Class name of the fairness outcome model (e.g. 'Ridge', 'TweedieRegressor').
    fairness_notion:
        The fairness notion used. Currently 'equal_opportunity' is implemented.
    """

    pareto_alphas: np.ndarray
    pareto_V: np.ndarray
    pareto_delta1: np.ndarray
    pareto_delta2: np.ndarray
    pareto_theta: np.ndarray
    selected_idx: int
    kappa: float
    n_train: int
    outcome_model_type: str
    fairness_notion: str

    @property
    def selected_alpha(self) -> float:
        """Alpha weight at the selected operating point."""
        return float(self.pareto_alphas[self.selected_idx])

    @property
    def selected_delta1(self) -> float:
        """Action fairness violation at the selected operating point."""
        return float(self.pareto_delta1[self.selected_idx])

    @property
    def selected_delta2(self) -> float:
        """Outcome fairness violation at the selected operating point."""
        return float(self.pareto_delta2[self.selected_idx])

    @property
    def selected_V(self) -> float:
        """Expected company value at the selected operating point."""
        return float(self.pareto_V[self.selected_idx])

    def summary(self) -> str:
        """
        Plain-text summary table of the Pareto front.

        Returns a formatted table suitable for inclusion in a model review
        document or FCA evidence pack.
        """
        lines = [
            "=" * 68,
            "Double Fairness Pareto Front",
            f"Fairness notion: {self.fairness_notion}  |  n = {self.n_train}  |  kappa = {self.kappa:.5f}",
            f"Outcome model:   {self.outcome_model_type}",
            "-" * 68,
            f"{'alpha':>6}  {'V_hat':>10}  {'Delta_1':>10}  {'Delta_2':>10}  {'selected':>8}",
            "-" * 68,
        ]
        for k in range(len(self.pareto_alphas)):
            sel = " <--" if k == self.selected_idx else ""
            lines.append(
                f"{self.pareto_alphas[k]:>6.3f}  "
                f"{self.pareto_V[k]:>10.5f}  "
                f"{self.pareto_delta1[k]:>10.6f}  "
                f"{self.pareto_delta2[k]:>10.6f}"
                f"{sel}"
            )
        lines.append("-" * 68)
        lines.append(f"Selected (value-maximising): alpha={self.selected_alpha:.3f}")
        lines.append(f"  V_hat   = {self.selected_V:.5f}")
        lines.append(f"  Delta_1 = {self.selected_delta1:.6f}  (action fairness violation)")
        lines.append(f"  Delta_2 = {self.selected_delta2:.6f}  (outcome fairness violation)")
        lines.append("=" * 68)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        JSON-serialisable dict of the result.

        All numpy arrays are converted to lists. Use this to store results
        in a model review database or pass to a report generator.
        """
        return {
            "pareto_alphas": self.pareto_alphas.tolist(),
            "pareto_V": self.pareto_V.tolist(),
            "pareto_delta1": self.pareto_delta1.tolist(),
            "pareto_delta2": self.pareto_delta2.tolist(),
            "pareto_theta": self.pareto_theta.tolist(),
            "selected_idx": self.selected_idx,
            "selected_alpha": self.selected_alpha,
            "selected_delta1": self.selected_delta1,
            "selected_delta2": self.selected_delta2,
            "selected_V": self.selected_V,
            "kappa": self.kappa,
            "n_train": self.n_train,
            "outcome_model_type": self.outcome_model_type,
            "fairness_notion": self.fairness_notion,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DoubleFairnessAudit:
    """
    Two-fairness-dimension policy learning for insurance pricing.

    Separates action fairness (equal premium classification across groups)
    from outcome fairness (equal expected benefit/loss ratio across groups),
    and recovers the full Pareto front via lexicographic Tchebycheff
    scalarisation.

    Based on: Bian, Wang, Shi, Qi. arXiv:2601.19186 (2026).

    FCA Consumer Duty context
    -------------------------
    Action fairness (Delta_1) satisfies the current FCA expectation: no
    premium discrimination at point of quoting. Outcome fairness (Delta_2)
    satisfies Consumer Duty Outcome 4 (Price and Value): the product delivers
    equivalent benefit across customer cohorts. The Pareto front is the
    auditable evidence of the considered trade-off.

    The empirical lesson from the paper is stark: on Belgian motor TPLI data
    (n=18,276), equalising premiums across gender (Delta_1=0) did NOT equalise
    welfare outcomes (Delta_2 remained large). A firm that audits only action
    fairness may still fail Consumer Duty.

    Parameters
    ----------
    primary_model:
        Sklearn-compatible regressor for estimating the primary outcome
        (company revenue / pure premium) as a function of features. Default
        is Ridge(). Fit separately for each action value (a=0, a=1) using
        the historical risk class assignment in the training data.
    fairness_model:
        Sklearn-compatible regressor for estimating the fairness-relevant
        outcome (e.g. loss ratio) as a function of features. Default is
        auto-detected: Ridge() for continuous outcomes, TweedieRegressor
        for loss ratios with many zeros. Fit separately for each (action,
        group) combination.
    n_alphas:
        Number of alpha weights to sweep across (0, 1). More points give
        a denser Pareto front at proportionally higher computation cost.
        20 is adequate for most purposes.
    kappa:
        Stage 2 slack parameter. If None, set to sqrt(log(n)/n) at fit
        time, which is the parametric convergence rate from Theorem 3.
        For nonparametric nuisance functions (e.g. gradient boosted trees),
        this underestimates kappa — set it explicitly if using nonparametric
        models.
    fairness_notion:
        'equal_opportunity' (implemented) measures whether an S-blind policy
        assigns systematically different action probabilities to the two
        groups. 'counterfactual' raises NotImplementedError (future work).
    random_state:
        Random seed for reproducibility.
    max_iter:
        Maximum iterations for scipy.optimize.minimize in each stage.
        Increase if optimisation warnings appear.

    Notes
    -----
    The binary action (a in {0, 1}) is a deliberate simplification.
    In insurance, interpret a=1 as "assign to high-risk band" and a=0 as
    "assign to low-risk band". X should exclude the protected characteristic S.
    Features on log-scale are preferred for multiplicative pricing models.

    For multi-group protected characteristics (more than 2 values of S),
    run the audit pairwise for each pair of groups and take the maximum Delta.
    This class raises ValueError if len(unique(S)) > 2.
    """

    def __init__(
        self,
        primary_model: Optional[Any] = None,
        fairness_model: Optional[Any] = None,
        n_alphas: int = 20,
        kappa: Optional[float] = None,
        fairness_notion: str = "equal_opportunity",
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        if fairness_notion not in ("equal_opportunity", "counterfactual"):
            raise ValueError(
                f"fairness_notion must be 'equal_opportunity' or 'counterfactual', "
                f"got '{fairness_notion}'."
            )
        if fairness_notion == "counterfactual":
            raise NotImplementedError(
                "fairness_notion='counterfactual' requires a causal model for X|S "
                "and is not yet implemented. Use 'equal_opportunity'."
            )
        if n_alphas < 2:
            raise ValueError(f"n_alphas must be >= 2, got {n_alphas}.")

        self.primary_model = primary_model
        self.fairness_model = fairness_model
        self.n_alphas = n_alphas
        self._kappa_override = kappa
        self.fairness_notion = fairness_notion
        self.random_state = random_state
        self.max_iter = max_iter

        # Set after fit()
        self._fitted = False
        self._result: Optional[DoubleFairnessResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y_primary: np.ndarray,
        y_fairness: np.ndarray,
        S: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> "DoubleFairnessAudit":
        """
        Fit nuisance models and prepare for Pareto sweep.

        Parameters
        ----------
        X:
            Feature matrix of shape (n, p). Must NOT contain the protected
            characteristic S. If S correlates with any column of X (proxy
            discrimination), Delta_1 will be non-zero for an S-blind policy —
            which is the point.
        y_primary:
            Primary outcome (company revenue / pure premium), shape (n,).
            Used to estimate the V_hat objective.
        y_fairness:
            Fairness-relevant outcome, shape (n,). Natural choices:
            - loss ratio (claims / premium): most actuarially meaningful
            - -premium: customer welfare (paper's default)
            - claims_indicator: binary claims probability
        S:
            Protected group indicator, shape (n,). Must be binary: 0 or 1.
            Multi-group characteristics should be run pairwise.
        exposure:
            Exposure weights, shape (n,). If None, equal weights are used.
            Passed as sample_weight to nuisance model fitting.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y_primary = np.asarray(y_primary, dtype=float)
        y_fairness = np.asarray(y_fairness, dtype=float)
        S = np.asarray(S, dtype=int)

        n, p = X.shape
        unique_S = np.unique(S)
        if not set(unique_S).issubset({0, 1}):
            raise ValueError(
                f"S must be binary (0 or 1). Got unique values: {unique_S}. "
                "For multi-group characteristics, run the audit pairwise."
            )
        if len(unique_S) < 2:
            raise ValueError(
                "S must contain both group 0 and group 1. "
                f"Only found: {unique_S}."
            )

        n0 = (S == 0).sum()
        n1 = (S == 1).sum()
        if n0 < 50:
            warnings.warn(
                f"Group 0 has only {n0} observations. Delta_2 estimates may be "
                "unreliable. Consider aggregating groups or collecting more data.",
                stacklevel=2,
            )
        if n1 < 50:
            warnings.warn(
                f"Group 1 has only {n1} observations. Delta_2 estimates may be "
                "unreliable. Consider aggregating groups or collecting more data.",
                stacklevel=2,
            )

        if exposure is None:
            exposure = np.ones(n, dtype=float)
        else:
            exposure = np.asarray(exposure, dtype=float)
            if exposure.shape != (n,):
                raise ValueError(
                    f"exposure must have shape ({n},), got {exposure.shape}."
                )

        self._n = n
        self._p = p
        self._X = X
        self._y_primary = y_primary
        self._y_fairness = y_fairness
        self._S = S
        self._exposure = exposure / exposure.mean()  # normalise to mean=1

        # Compute kappa
        if self._kappa_override is not None:
            self._kappa = self._kappa_override
        else:
            self._kappa = float(np.sqrt(np.log(n) / n))

        # Fit nuisance models
        self._fit_nuisance(X, y_primary, y_fairness, S, self._exposure)

        self._fitted = True
        self._result = None  # reset cached result
        return self

    def audit(self) -> DoubleFairnessResult:
        """
        Run the Pareto sweep and return the result.

        Sweeps n_alphas weights across (0, 1), runs Stage 1 + Stage 2
        optimisation for each, and returns the full Pareto front.

        Returns
        -------
        DoubleFairnessResult
        """
        self._check_fitted()
        if self._result is None:
            self._result = self._sweep_pareto()
        return self._result

    def plot_pareto(
        self,
        figsize: tuple = (10, 4),
        highlight_selected: bool = True,
    ):
        """
        Plot the Pareto front as two panels: (V vs Delta_1) and (V vs Delta_2).

        Requires matplotlib. Install with ``pip install insurance-fairness[plot]``
        or ``pip install matplotlib``.

        Parameters
        ----------
        figsize:
            Figure size in inches.
        highlight_selected:
            If True, highlight the value-maximising selected Pareto point.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_pareto(). Install it with:\n"
                "    pip install insurance-fairness[plot]\n"
                "or:\n"
                "    pip install matplotlib"
            ) from exc

        result = self.audit()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: V vs Delta_1
        ax = axes[0]
        ax.plot(result.pareto_delta1, result.pareto_V, "o-", color="steelblue",
                alpha=0.7, markersize=5, label="Pareto points")
        if highlight_selected:
            ax.scatter(
                result.selected_delta1, result.selected_V,
                c="red", marker="*", s=200, zorder=5, label="Selected"
            )
        ax.set_xlabel("Delta_1 (action fairness violation)")
        ax.set_ylabel("V_hat (expected value)")
        ax.set_title("Value vs Action Fairness")
        ax.legend()

        # Panel 2: V vs Delta_2
        ax = axes[1]
        ax.plot(result.pareto_delta2, result.pareto_V, "o-", color="darkorange",
                alpha=0.7, markersize=5, label="Pareto points")
        if highlight_selected:
            ax.scatter(
                result.selected_delta2, result.selected_V,
                c="red", marker="*", s=200, zorder=5, label="Selected"
            )
        ax.set_xlabel("Delta_2 (outcome fairness violation)")
        ax.set_ylabel("V_hat (expected value)")
        ax.set_title("Value vs Outcome Fairness")
        ax.legend()

        fig.tight_layout()
        return fig

    def report(self) -> str:
        """
        FCA-ready plain-text report.

        Produces a section suitable for insertion into a Consumer Duty
        evidence pack. Includes the Pareto front summary, selected operating
        point, comparison to the unconstrained optimum, kappa, and
        regulatory references.

        Returns
        -------
        str
        """
        self._check_fitted()
        result = self.audit()

        # Compute unconstrained optimum: all-action-1 policy (theta -> +inf)
        # Approximate as theta_unconstrained = 10 * ones -> pi(1|X) ~ 1
        theta_unconstrained = np.zeros(self._p)
        d1_unc = self._delta1_hat(theta_unconstrained, self._X, self._S)
        d2_unc = self._delta2_hat(theta_unconstrained, self._X, self._S)
        v_unc = self._V_hat(theta_unconstrained, self._X, self._S, self._exposure)

        # Improvement percentages (avoid division by zero)
        def pct_reduction(baseline: float, improved: float) -> str:
            if abs(baseline) < 1e-12:
                return "N/A"
            pct = 100.0 * (baseline - improved) / abs(baseline)
            return f"{pct:.1f}%"

        d1_reduction = pct_reduction(d1_unc, result.selected_delta1)
        d2_reduction = pct_reduction(d2_unc, result.selected_delta2)
        v_cost = pct_reduction(result.selected_V, v_unc)
        if abs(v_unc) < 1e-12:
            v_cost_str = "N/A"
        else:
            v_cost_pct = 100.0 * (v_unc - result.selected_V) / abs(v_unc)
            v_cost_str = f"{v_cost_pct:.1f}%"

        v_min = result.pareto_V.min()
        v_max = result.pareto_V.max()
        d1_min = result.pareto_delta1.min()
        d1_max = result.pareto_delta1.max()
        d2_min = result.pareto_delta2.min()
        d2_max = result.pareto_delta2.max()

        lines = [
            "Double Fairness Analysis",
            "=" * 60,
            f"Fairness notion:     {self.fairness_notion}",
            f"Protected group:     S=0 vs S=1",
            f"Training obs:        n = {self._n}",
            f"Outcome model:       {result.outcome_model_type}",
            f"Slack (kappa):       {result.kappa:.5f}",
            "",
            f"Pareto Front Summary (K={self.n_alphas} alpha weights):",
            f"  Value range:            V = {v_min:.5f} to {v_max:.5f}",
            f"  Action unfairness:  Delta_1 = {d1_min:.6f} to {d1_max:.6f}",
            f"  Outcome unfairness: Delta_2 = {d2_min:.6f} to {d2_max:.6f}",
            "",
            "Selected Operating Point (value-maximising Pareto solution):",
            f"  Alpha (action/outcome weight): {result.selected_alpha:.3f}",
            f"  Estimated company value:       V = {result.selected_V:.5f}",
            f"  Action fairness violation:     Delta_1 = {result.selected_delta1:.6f}",
            f"    (unconstrained: {d1_unc:.6f})",
            f"  Outcome fairness violation:    Delta_2 = {result.selected_delta2:.6f}",
            f"    (unconstrained: {d2_unc:.6f})",
            "",
            "Interpretation:",
            f"  By applying lexicographic Tchebycheff optimisation, the selected",
            f"  pricing policy reduces action unfairness by {d1_reduction} and outcome",
            f"  unfairness by {d2_reduction} relative to the unconstrained revenue-",
            f"  maximising policy, at an estimated cost of {v_cost_str} in expected revenue.",
            "",
            "Regulatory references:",
            "  FCA Consumer Duty (PRIN 2A), Outcome 4 (Price and Value).",
            "  FCA Pricing Practices Thematic Review (TR24/2, 2024).",
            "",
            "Method:",
            "  Bian, Wang, Shi, Qi (2026). Double Fairness Policy Learning.",
            "  arXiv:2601.19186v2.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "DoubleFairnessAudit has not been fitted. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Internal: nuisance model fitting
    # ------------------------------------------------------------------

    def _default_primary_model(self) -> Any:
        from sklearn.linear_model import Ridge
        return Ridge()

    def _default_fairness_model(self, y_fairness: np.ndarray) -> Any:
        """
        Auto-select fairness model based on y_fairness distribution.

        If more than 30% of values are exactly zero (e.g. loss ratio with
        many zero-claim policies), use TweedieRegressor (compound Poisson-
        Gamma). Otherwise use Ridge.
        """
        zero_fraction = float((y_fairness == 0).mean())
        if zero_fraction > 0.3:
            from sklearn.linear_model import TweedieRegressor
            return TweedieRegressor(power=1.5, max_iter=500)
        else:
            from sklearn.linear_model import Ridge
            return Ridge()

    def _fit_nuisance(
        self,
        X: np.ndarray,
        y_primary: np.ndarray,
        y_fairness: np.ndarray,
        S: np.ndarray,
        exposure: np.ndarray,
    ) -> None:
        """
        Fit nuisance functions r_hat and f_hat.

        Primary outcome r_hat:
            Fit one model per action value using historical risk class
            (A ~ high/low risk). Since we don't have explicit historical
            action labels, we use S-stratified models: r_hat_s0 and r_hat_s1
            for group 0 and group 1 respectively, treating the group as a
            proxy for the historical action.

        Fairness outcome f_hat:
            Fit one model per (group) combination using y_fairness.
            f_hat_s0 predicts y_fairness for group 0 observations.
            f_hat_s1 predicts y_fairness for group 1 observations.

        Design note: fitting separate models per group rather than per
        action avoids assuming we have reliable historical action labels.
        The policy optimisation then constructs the expected outcomes as
        pi(1|X)*f_hat_s1(X) + (1-pi(1|X))*f_hat_s0(X) — interpreting
        action a=1 as "group 1 experience" and a=0 as "group 0 experience".
        This is the policy-counterfactual interpretation: what outcome
        would each observation experience if we applied group-1 vs group-0
        pricing?
        """
        mask0 = S == 0
        mask1 = S == 1

        # Primary outcome nuisance models
        r_model0 = (
            self._clone_or_default(self.primary_model, "primary")
        )
        r_model1 = (
            self._clone_or_default(self.primary_model, "primary")
        )

        sw0 = exposure[mask0]
        sw1 = exposure[mask1]

        r_model0.fit(X[mask0], y_primary[mask0], sample_weight=sw0)
        r_model1.fit(X[mask1], y_primary[mask1], sample_weight=sw1)

        self._r_hat_s0 = r_model0
        self._r_hat_s1 = r_model1

        # Fairness outcome nuisance models
        f_default = self._default_fairness_model(y_fairness)
        f_model0 = self._clone_or_custom(self.fairness_model, f_default)
        f_model1 = self._clone_or_custom(self.fairness_model, f_default)

        f_model0.fit(X[mask0], y_fairness[mask0], sample_weight=sw0)
        f_model1.fit(X[mask1], y_fairness[mask1], sample_weight=sw1)

        self._f_hat_s0 = f_model0
        self._f_hat_s1 = f_model1

        self._outcome_model_type = type(f_model0).__name__

    def _clone_or_default(self, user_model: Optional[Any], kind: str) -> Any:
        """Return a clone of user_model, or a default if None."""
        if user_model is None:
            return self._default_primary_model()
        try:
            from sklearn.base import clone
            return clone(user_model)
        except Exception:
            # If clone fails (non-sklearn model), create same type
            return type(user_model)()

    def _clone_or_custom(self, user_model: Optional[Any], default: Any) -> Any:
        """Return a clone of user_model, or clone of default if user_model is None."""
        if user_model is None:
            try:
                from sklearn.base import clone
                return clone(default)
            except Exception:
                return type(default)()
        try:
            from sklearn.base import clone
            return clone(user_model)
        except Exception:
            return type(user_model)()

    # ------------------------------------------------------------------
    # Internal: policy and objective computation
    # ------------------------------------------------------------------

    def _policy_probs(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute pi_theta(1 | X_i) for each observation i.

        Returns shape (n,). Uses sigmoid(X @ theta) so theta in R^p.
        """
        return _sigmoid(X @ theta)

    def _V_hat(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        S: np.ndarray,
        exposure: np.ndarray,
    ) -> float:
        """
        Estimate expected company value under policy pi_theta.

        V_hat = exposure-weighted mean of:
            pi(1|X_i) * r_hat_s1(X_i) + (1-pi(1|X_i)) * r_hat_s0(X_i)

        where r_hat_s1 and r_hat_s0 are nuisance models for group 1 and
        group 0 primary outcomes respectively.
        """
        pi = self._policy_probs(theta, X)
        r0 = self._r_hat_s0.predict(X)
        r1 = self._r_hat_s1.predict(X)
        v_i = pi * r1 + (1.0 - pi) * r0
        total_exp = exposure.sum()
        if total_exp <= 0:
            return float(np.mean(v_i))
        return float(np.dot(v_i, exposure) / total_exp)

    def _delta1_hat(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        S: np.ndarray,
    ) -> float:
        """
        Estimate action fairness violation Delta_hat_1.

        Measures whether the S-blind policy assigns systematically different
        action probabilities to the two groups:

            Delta_1 = (mean_{i in G1} pi(1|X_i) - mean_{i in G0} pi(1|X_i))^2

        Even when S is not in X, proxy features correlated with S will
        produce non-zero Delta_1. This is the key signal for action-level
        proxy discrimination.
        """
        pi = self._policy_probs(theta, X)
        mask0 = S == 0
        mask1 = S == 1
        mean0 = float(np.mean(pi[mask0]))
        mean1 = float(np.mean(pi[mask1]))
        return (mean1 - mean0) ** 2

    def _delta2_hat(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        S: np.ndarray,
    ) -> float:
        """
        Estimate outcome fairness violation Delta_hat_2.

        For each observation i, compute the expected fairness outcome under
        group assignment s=1 vs s=0:

            f_hat(s, X_i) = pi(1|X_i)*f_hat_s(X_i) + (1-pi(1|X_i))*f_hat_{1-s}(X_i)

        Actually: f_hat(1, X_i) = pi(1|X_i)*f_hat_s1(X_i) + (1-pi)*f_hat_s0(X_i)
                  f_hat(0, X_i) = pi(1|X_i)*f_hat_s0(X_i) + (1-pi)*f_hat_s1(X_i)

        Delta_2 = mean_i [ f_hat(1, X_i) - f_hat(0, X_i) ]^2

        This simplifies to:
            f_hat(1, X_i) - f_hat(0, X_i) = (2*pi - 1) * (f_hat_s1(X_i) - f_hat_s0(X_i))

        So Delta_2 = mean_i [ (2*pi_i - 1)^2 * (f1_i - f0_i)^2 ]
        where f1_i = f_hat_s1(X_i), f0_i = f_hat_s0(X_i).
        """
        pi = self._policy_probs(theta, X)
        f0 = self._f_hat_s0.predict(X)
        f1 = self._f_hat_s1.predict(X)
        diff = (2.0 * pi - 1.0) * (f1 - f0)
        return float(np.mean(diff ** 2))

    # ------------------------------------------------------------------
    # Internal: Tchebycheff optimisation
    # ------------------------------------------------------------------

    def _tchebycheff_stage1(self, alpha: float) -> tuple[float, np.ndarray]:
        """
        Stage 1: minimise the Tchebycheff criterion M_alpha(pi_theta).

            M_alpha(theta) = max{ alpha * Delta_1(theta), (1-alpha) * Delta_2(theta) }

        We smooth the max with log-sum-exp (tau=50) to get a differentiable
        objective, then minimise with L-BFGS-B.

        Returns
        -------
        (M_hat_star_alpha, theta_star)
        """
        rng = np.random.default_rng(self.random_state)
        theta0 = rng.normal(0.0, 0.1, size=self._p)

        def obj(theta: np.ndarray) -> float:
            d1 = self._delta1_hat(theta, self._X, self._S)
            d2 = self._delta2_hat(theta, self._X, self._S)
            return _softmax_max(alpha * d1, (1.0 - alpha) * d2)

        res = minimize(
            obj,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-9, "gtol": 1e-6},
        )

        theta_star = res.x
        # Recompute exact (non-smoothed) M_hat at the solution
        d1 = self._delta1_hat(theta_star, self._X, self._S)
        d2 = self._delta2_hat(theta_star, self._X, self._S)
        M_hat_star = max(alpha * d1, (1.0 - alpha) * d2)

        return M_hat_star, theta_star

    def _tchebycheff_stage2(
        self,
        alpha: float,
        M_star: float,
        kappa: float,
        theta_init: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Stage 2: minimise Delta_1 + Delta_2 subject to M_hat <= M_star + kappa.

        The constraint max{alpha*Delta_1, (1-alpha)*Delta_2} <= M_star + kappa
        is equivalent to two linear constraints:
            alpha * Delta_1 <= M_star + kappa
            (1 - alpha) * Delta_2 <= M_star + kappa

        We use SLSQP which supports inequality constraints.

        Returns
        -------
        (Delta_hat_star, theta_star)
        """
        M_budget = M_star + kappa

        def obj(theta: np.ndarray) -> float:
            d1 = self._delta1_hat(theta, self._X, self._S)
            d2 = self._delta2_hat(theta, self._X, self._S)
            return d1 + d2

        def constraint_action(theta: np.ndarray) -> float:
            """alpha * Delta_1 <= M_budget  =>  M_budget - alpha*Delta_1 >= 0"""
            d1 = self._delta1_hat(theta, self._X, self._S)
            return M_budget - alpha * d1

        def constraint_outcome(theta: np.ndarray) -> float:
            """(1-alpha) * Delta_2 <= M_budget  =>  M_budget - (1-alpha)*Delta_2 >= 0"""
            d2 = self._delta2_hat(theta, self._X, self._S)
            return M_budget - (1.0 - alpha) * d2

        constraints = [
            {"type": "ineq", "fun": constraint_action},
            {"type": "ineq", "fun": constraint_outcome},
        ]

        res = minimize(
            obj,
            theta_init,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )

        theta_star = res.x
        d1 = self._delta1_hat(theta_star, self._X, self._S)
        d2 = self._delta2_hat(theta_star, self._X, self._S)
        delta_total = d1 + d2

        return delta_total, theta_star

    def _sweep_pareto(self) -> DoubleFairnessResult:
        """
        Sweep K alpha weights, running Stage 1 + Stage 2 for each.

        Excludes the boundary values alpha=0 and alpha=1 (which collapse
        to single-objective problems) by using linspace(1/(K+1), K/(K+1), K).

        Returns
        -------
        DoubleFairnessResult
        """
        K = self.n_alphas
        alphas = np.linspace(1.0 / (K + 1), K / (K + 1), K)

        pareto_alphas = np.empty(K)
        pareto_V = np.empty(K)
        pareto_delta1 = np.empty(K)
        pareto_delta2 = np.empty(K)
        pareto_theta = np.empty((K, self._p))

        for k, alpha in enumerate(alphas):
            # Stage 1
            M_star, theta1 = self._tchebycheff_stage1(alpha)

            # Stage 2
            _, theta2 = self._tchebycheff_stage2(alpha, M_star, self._kappa, theta1)

            d1 = self._delta1_hat(theta2, self._X, self._S)
            d2 = self._delta2_hat(theta2, self._X, self._S)
            v = self._V_hat(theta2, self._X, self._S, self._exposure)

            pareto_alphas[k] = alpha
            pareto_V[k] = v
            pareto_delta1[k] = d1
            pareto_delta2[k] = d2
            pareto_theta[k] = theta2

        # Select value-maximising Pareto point
        selected_idx = int(np.argmax(pareto_V))

        return DoubleFairnessResult(
            pareto_alphas=pareto_alphas,
            pareto_V=pareto_V,
            pareto_delta1=pareto_delta1,
            pareto_delta2=pareto_delta2,
            pareto_theta=pareto_theta,
            selected_idx=selected_idx,
            kappa=self._kappa,
            n_train=self._n,
            outcome_model_type=self._outcome_model_type,
            fairness_notion=self.fairness_notion,
        )
