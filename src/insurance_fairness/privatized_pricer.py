"""
privatized_pricer.py
---------------------
PrivatizedFairPricer: sklearn-compatible wrapper for discrimination-free pricing
with privatised sensitive attributes.

This module wraps :class:`~insurance_fairness.privatized_audit.PrivatizedFairnessAudit`
in a standard fit/predict interface so it can drop into an sklearn pipeline or
Databricks MLflow model registry without modification.

The distinction between the two classes is deliberate. ``PrivatizedFairnessAudit``
is an audit tool: it returns correction matrices, noise diagnostics, and a
structured audit report suitable for regulatory evidence packs. This class is a
pricing tool: fit it, call predict, slot it into a pipeline. The underlying
maths are identical.

Mechanism stubs
---------------
Only the k-randomised response (k-RR) mechanism is implemented, matching the
MPTP-LDP protocol in Zhang, Liu & Shi (2025). Laplace and Gaussian mechanisms
apply to continuous sensitive attributes — not the categorical group membership
that k-RR handles — and are left as NotImplementedError stubs.

Fairness constraint stubs
--------------------------
Only ``'demographic_parity'`` is implemented (the Lindholm 2022 marginalisation
with fixed P*). ``'equalized_odds'`` would require constrained optimisation
beyond the paper's scope and is a NotImplementedError stub.

References
----------
Zhang, Z., Liu, J. & Shi, Y. (2025). Discrimination-Free Insurance Pricing
with Privatized Sensitive Attributes. arXiv:2504.11775.

Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.
"""

from __future__ import annotations

import numpy as np

from insurance_fairness.privatized_audit import PrivatizedAuditResult, PrivatizedFairnessAudit

# ---------------------------------------------------------------------------
# Mapping from public base_estimator names to internal (backend, loss) pairs
# ---------------------------------------------------------------------------

_BASE_ESTIMATOR_MAP: dict[str, tuple[str, str]] = {
    "poisson_glm": ("sklearn", "poisson"),
    "tweedie_glm": ("sklearn", "gaussian"),  # Ridge is the closest sklearn GLM for Tweedie
    "ridge": ("sklearn", "gaussian"),
    "catboost": ("catboost", "poisson"),
}

_VALID_MECHANISMS = ("k-rr", "laplace", "gaussian")
_VALID_CONSTRAINTS = ("demographic_parity", "equalized_odds")


class PrivatizedFairPricer:
    """
    Discrimination-free pricer with privatised sensitive attributes.

    Sklearn-compatible fit/predict wrapper around
    :class:`~insurance_fairness.privatized_audit.PrivatizedFairnessAudit`.
    Implements the MPTP-LDP protocol (Zhang, Liu & Shi, 2025): group-specific
    models are trained on LDP-corrected reweighted samples, and the fair premium
    marginalises over a fixed reference distribution P* to satisfy
    discrimination-free pricing (Lindholm et al., 2022).

    Parameters
    ----------
    epsilon :
        LDP privacy budget. Correct-response probability is derived as
        ``pi = exp(epsilon) / (K - 1 + exp(epsilon))``. Mutually exclusive
        with ``pi``.
    delta :
        Reserved for (epsilon, delta)-DP. Only ``delta=0.0`` (pure epsilon-LDP)
        is currently implemented. Non-zero values will raise NotImplementedError
        when fit() is called.
    mechanism :
        LDP mechanism. Only ``'k-rr'`` (k-randomised response) is implemented.
        ``'laplace'`` and ``'gaussian'`` raise NotImplementedError — those
        mechanisms apply to continuous attributes; k-RR handles categorical
        group membership.
    n_groups :
        Number of protected groups K. Must match the distinct values in S
        passed to fit().
    fairness_constraint :
        Fairness criterion. Only ``'demographic_parity'`` is implemented,
        which uses the Lindholm marginalisation with fixed P*. ``'equalized_odds'``
        raises NotImplementedError.
    reference_distribution :
        Reference distribution P* used when computing the fair premium.
        ``'uniform'`` sets P*(D=k) = 1/K; ``'empirical'`` uses the
        noise-corrected marginal; a numpy array of shape (K,) may be supplied.
        ``'uniform'`` is recommended for UK gender-neutrality requirements.
    base_estimator :
        Group-specific model family. Options:

        - ``'poisson_glm'``: sklearn PoissonRegressor (default, claim frequency)
        - ``'tweedie_glm'``: sklearn Ridge regression (pure premium proxy)
        - ``'ridge'``: sklearn Ridge regression
        - ``'catboost'``: CatBoost gradient boosting (requires catboost package)
    pi :
        Correct-response probability, overrides epsilon. Use when the LDP
        parameter is known from the data collection protocol rather than
        the privacy budget.
    n_anchor_groups :
        Partition count for Procedure 4.5 anchor-point pi estimation. Only
        relevant when neither pi nor epsilon is supplied and X_anchor is
        passed to fit().
    random_state :
        Random seed for reproducibility.

    Attributes
    ----------
    audit_ :
        The fitted :class:`~insurance_fairness.privatized_audit.PrivatizedFairnessAudit`
        instance. Exposes all internal diagnostics.
    n_features_in_ :
        Number of features seen during fit.

    Examples
    --------
    Known LDP budget (typical production use)::

        from insurance_fairness import PrivatizedFairPricer

        pricer = PrivatizedFairPricer(
            epsilon=2.0,
            n_groups=2,
            reference_distribution='uniform',
            base_estimator='poisson_glm',
        )
        pricer.fit(X_train, y_train, S_train)
        premiums = pricer.predict(X_test)
        print(pricer.excess_risk_bound())

    References
    ----------
    Zhang, Z., Liu, J. & Shi, Y. (2025). arXiv:2504.11775.
    """

    def __init__(
        self,
        epsilon: float | None = None,
        delta: float = 0.0,
        mechanism: str = "k-rr",
        n_groups: int = 2,
        fairness_constraint: str = "demographic_parity",
        reference_distribution: str | np.ndarray = "uniform",
        base_estimator: str = "poisson_glm",
        pi: float | None = None,
        n_anchor_groups: int = 20,
        random_state: int | None = None,
    ) -> None:
        if mechanism not in _VALID_MECHANISMS:
            raise ValueError(
                f"mechanism must be one of {_VALID_MECHANISMS}, got {mechanism!r}."
            )
        if fairness_constraint not in _VALID_CONSTRAINTS:
            raise ValueError(
                f"fairness_constraint must be one of {_VALID_CONSTRAINTS}, "
                f"got {fairness_constraint!r}."
            )
        if base_estimator not in _BASE_ESTIMATOR_MAP:
            raise ValueError(
                f"base_estimator must be one of {list(_BASE_ESTIMATOR_MAP)}, "
                f"got {base_estimator!r}."
            )

        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self.n_groups = n_groups
        self.fairness_constraint = fairness_constraint
        self.reference_distribution = reference_distribution
        self.base_estimator = base_estimator
        self.pi = pi
        self.n_anchor_groups = n_anchor_groups
        self.random_state = random_state

        # Fitted attributes
        self.audit_: PrivatizedFairnessAudit | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # sklearn-compatible fit/predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        S: np.ndarray,
        exposure: np.ndarray | None = None,
        X_anchor: np.ndarray | None = None,
    ) -> "PrivatizedFairPricer":
        """
        Fit K group-specific models and compute the discrimination-free premium.

        Parameters
        ----------
        X :
            Non-sensitive feature matrix. Shape (n, p).
        y :
            Outcome vector (claim frequency, pure premium, etc.). Shape (n,).
        S :
            Privatised sensitive attribute. Integer labels in {0, ..., K-1}.
            Shape (n,).
        exposure :
            Exposure weights (policy years, vehicle months, etc.). Shape (n,).
            If None, all observations are weighted equally.
        X_anchor :
            Anchor-point features for Procedure 4.5 pi estimation. Required
            when neither ``pi`` nor ``epsilon`` is set. Shape (n, q) — may
            be the full X or a subset of features that correlate with group
            membership (e.g. occupation flags, regional dummies).

        Returns
        -------
        self
        """
        # Mechanism and constraint checks — raise before any computation
        if self.mechanism != "k-rr":
            raise NotImplementedError(
                f"mechanism={self.mechanism!r} is not implemented. "
                "Only 'k-rr' (k-randomised response) is currently supported. "
                "Laplace and Gaussian mechanisms apply to continuous sensitive "
                "attributes; this library addresses categorical group membership "
                "via k-RR as per Zhang, Liu & Shi (2025)."
            )
        if self.fairness_constraint != "demographic_parity":
            raise NotImplementedError(
                f"fairness_constraint={self.fairness_constraint!r} is not implemented. "
                "Only 'demographic_parity' (Lindholm marginalisation with fixed P*) "
                "is currently supported. 'equalized_odds' requires constrained "
                "optimisation not covered by the Zhang/Liu/Shi protocol."
            )
        if self.delta != 0.0:
            raise NotImplementedError(
                f"delta={self.delta} is not zero. (epsilon, delta)-DP with the Gaussian "
                "mechanism is not implemented. Only pure epsilon-LDP (delta=0.0) with "
                "k-randomised response is currently supported."
            )

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else 1

        backend, loss = _BASE_ESTIMATOR_MAP[self.base_estimator]

        self.audit_ = PrivatizedFairnessAudit(
            n_groups=self.n_groups,
            epsilon=self.epsilon,
            pi=self.pi,
            reference_distribution=self.reference_distribution,
            loss=loss,
            n_anchor_groups=self.n_anchor_groups,
            nuisance_backend=backend,
            random_state=self.random_state,
        )
        self.audit_.fit(X, y, S, X_anchor=X_anchor, exposure=exposure)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the discrimination-free premium for new observations.

        Returns h*(X) = sum_k f_k(X) * P*(D=k).

        Parameters
        ----------
        X :
            Feature matrix. Shape (n_new, p). Must have the same number of
            columns as the X passed to fit().

        Returns
        -------
        np.ndarray
            Fair premium predictions. Shape (n_new,). Non-negative.
        """
        self._check_fitted()
        return self.audit_.predict_fair_premium(np.asarray(X, dtype=float))

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        S: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit on training data and return the in-sample fair premiums.

        Convenience method equivalent to ``fit(X, y, S).predict(X)``, but
        returns the stored training predictions directly (no second pass).

        Parameters
        ----------
        X :
            Feature matrix. Shape (n, p).
        y :
            Outcome vector. Shape (n,).
        S :
            Privatised sensitive attribute. Shape (n,).
        exposure :
            Exposure weights. Shape (n,). Optional.

        Returns
        -------
        np.ndarray
            In-sample fair premiums. Shape (n,).
        """
        self.fit(X, y, S, exposure=exposure)
        return self.audit_.fair_predictions_.copy()

    def group_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Return group-specific model predictions before marginalisation.

        Each column k contains f_k(X) — the prediction from the model trained
        on reweighted samples for group k. The fair premium is
        ``group_predictions(X) @ p_star``.

        Parameters
        ----------
        X :
            Feature matrix. Shape (n, p).

        Returns
        -------
        np.ndarray
            Shape (n, K). Column k is f_k(X).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        preds = np.stack(
            [m.predict(X) for m in self.audit_.group_models_], axis=1
        )
        return preds

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def excess_risk_bound(self, delta: float = 0.05, vc_dim: int | None = None) -> float:
        """
        Theorem 4.3 excess risk bound (Zhang et al., 2025).

        Upper bound on the excess empirical risk of the LDP-corrected model
        relative to the oracle model trained on true (unnoised) attributes.
        Scales O(C1 * K^2 / sqrt(n)); heavier noise or more groups demands
        more data for the same guarantee quality.

        Parameters
        ----------
        delta :
            Failure probability. Default 0.05 (95% confidence).
        vc_dim :
            VC dimension of the function class. If None, defaults to a
            conservative heuristic of 10 (equivalent to a linear model in
            ~9 features).

        Returns
        -------
        float
            Upper bound on excess empirical risk.
        """
        self._check_fitted()
        return self.audit_.statistical_bound(delta=delta, vc_dim=vc_dim)

    def audit_report(self) -> PrivatizedAuditResult:
        """
        Structured audit result from the underlying PrivatizedFairnessAudit.

        Returns a :class:`~insurance_fairness.privatized_audit.PrivatizedAuditResult`
        dataclass with fair premiums, group models, noise diagnostics, and the
        Theorem 4.3 bound. Intended for regulatory evidence packs.

        Returns
        -------
        PrivatizedAuditResult
        """
        self._check_fitted()
        return self.audit_.audit_report()

    def correction_summary(self) -> dict:
        """
        LDP correction matrices and noise parameters.

        Returns
        -------
        dict with keys:

        ``Pi_inv`` :
            (K, K) reweighting matrix (Lemma 4.2).
        ``T_inv`` :
            (K, K) noise-inversion matrix.
        ``pi`` :
            Correct-response probability.
        ``pi_bar`` :
            Incorrect-response probability per non-target group.
        ``C1`` :
            Noise amplification factor (K*pi - 1)^{-1} * (pi + K - 2).
        ``negative_weight_frac`` :
            Fraction of (observation, group) pairs with negative reweighting
            before clipping. Values above 0.05 indicate the LDP noise is too
            heavy for reliable correction.
        ``bound_95`` :
            Theorem 4.3 generalisation bound at delta=0.05.
        """
        self._check_fitted()
        mats = self.audit_.correction_matrices()
        mats["negative_weight_frac"] = self.audit_.negative_weight_frac_
        mats["bound_95"] = self.audit_.statistical_bound(delta=0.05)
        return mats

    def minimum_sample_size(
        self, delta: float = 0.05, target_bound: float = 0.05
    ) -> int:
        """
        Minimum sample size for the Theorem 4.3 bound to meet ``target_bound``.

        Useful at design time: given a planned epsilon and number of groups,
        how many policyholders are needed to guarantee excess risk below the
        target?

        Parameters
        ----------
        delta :
            Failure probability. Default 0.05.
        target_bound :
            Desired maximum excess risk. Default 0.05.

        Returns
        -------
        int
            Recommended minimum n.
        """
        self._check_fitted()
        return self.audit_.minimum_n_recommended(delta=delta, target_bound=target_bound)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if fit() has not been called."""
        if self.audit_ is None or self.audit_.pi_ is None:
            raise RuntimeError(
                "PrivatizedFairPricer has not been fitted. Call fit() first."
            )
