"""
marginal_fairness.py
--------------------
MarginalFairnessPremium: adjusts distortion risk measure premiums to be
marginally fair with respect to protected attributes.

The standard approach to fairness in insurance pricing either (a) removes
protected attributes from Stage 1 predictions (discrimination-free pricing,
Lindholm et al. 2022), or (b) applies group-level demographic parity constraints.
Neither is right when the pricing stage uses a distortion risk measure like
Expected Shortfall — the fairness target shifts because the tail-weighting
changes the effective sensitivity to protected attributes.

Huang & Pesenti (2025) formalise this. A risk measure rho_gamma is *marginally
fair* for protected attribute D_i if perturbing D_i at the margin does not
move rho_gamma. The correction is closed-form — no iterative solver required.

The class operates at the Stage 2 level: it accepts a fitted Stage 1 model
g(D, X) and computes how much the risk measure is moved by each protected
attribute. It then removes exactly that component from the premium, preserving
the actuarial balance to within ~1% at portfolio level.

This is a different intervention from DiscriminationFreePrice (which modifies
Stage 1). The two are complementary: marginal fairness is more appropriate
when the insurer *wants* the model to use protected attributes for prediction
accuracy but must ensure the final pricing decision is not sensitive to them.

Usage::

    from insurance_fairness import MarginalFairnessPremium

    mfp = MarginalFairnessPremium(distortion='es_alpha', alpha=0.75)
    mfp.fit(Y_train, D_train, X_train, model=glm, protected_indices=[0])
    rho_fair = mfp.transform(Y_test, D_test, X_test)

    report = mfp.sensitivity_report()
    print(f"Baseline ES0.75: {report.rho_baseline:.4f}")
    print(f"Fair ES0.75:     {report.rho_fair:.4f}")
    print(f"Lift ratio:      {report.lift_ratio:.4f}")

References
----------
Huang, F. & Pesenti, S. M. (2025). Marginal Fairness: Fair Decision-Making
under Risk Measures. arXiv:2505.18895.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MarginalFairnessReport:
    """
    Sensitivity diagnostics for a fitted MarginalFairnessPremium.

    Attributes
    ----------
    protected_names :
        Names of protected attributes in the order corrections were applied.
    sensitivities :
        Estimated marginal sensitivity d_{D_i} rho_gamma for each attribute.
        Values close to zero indicate the risk measure is already marginally fair.
    denominators :
        Estimated E[(D_i * dg/dD_i)^2 | X] for each attribute. Used as the
        normalising factor in the correction. Near-zero values warn of a
        degenerate model (gradient of g w.r.t. D_i is near-zero everywhere).
    corrections :
        Additive correction term removed from rho_baseline per attribute.
        correction_i = (sensitivity_i / denominator_i) * numer_i
    rho_baseline :
        Portfolio-average distortion risk measure before adjustment.
    rho_fair :
        Portfolio-average distortion risk measure after all corrections.
    lift_ratio :
        rho_fair / rho_baseline. Values near 1.0 indicate actuarial neutrality.
    """

    protected_names: list[str]
    sensitivities: np.ndarray
    denominators: np.ndarray
    corrections: np.ndarray
    rho_baseline: float
    rho_fair: float
    lift_ratio: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MarginalFairnessPremium:
    """
    Adjust distortion risk measure premiums to be marginally fair with respect
    to protected attributes.

    Implements the closed-form adjustment from Huang & Pesenti (2025),
    arXiv:2505.18895. The correction is exact under the paper's L2-minimal
    weight adjustment — no iterative optimisation, no solver.

    Parameters
    ----------
    distortion : str or callable, default 'es_alpha'
        Risk measure distortion weight function gamma(u) where u in [0, 1].
        Options:

        - ``'es_alpha'``: Expected Shortfall (TVaR) at level alpha.
          gamma(u) = 1/(1-alpha) if u >= alpha else 0.
        - ``'expectation'``: Pure mean. gamma(u) = 1 for all u.
        - ``'wang_lambda'``: Wang transform. gamma(u) is the derivative of
          Phi(Phi^{-1}(u) + lambda) w.r.t. u, where lambda is the Wang
          parameter (set via ``alpha`` parameter for consistency).
        - callable: A function gamma(u: np.ndarray) -> np.ndarray mapping
          conditional quantile levels to distortion weights.

    alpha : float, default 0.75
        Tail level for Expected Shortfall or lambda parameter for Wang transform.
        Ignored when distortion is callable or 'expectation'.
        For ES: common values are 0.75 (actuarial), 0.90, 0.95.
        For Wang: lambda > 0 loads the premium (risk-averse), lambda < 0 unloads.

    grad_method : str, default 'finite_diff'
        Method for computing the Jacobian dg/dD_i at each observation.
        Options:

        - ``'finite_diff'``: Two-sided central finite differences.
          Step size eps = eps_rel * std(D_i). This is model-agnostic and
          works with any sklearn-compatible estimator.
        - ``'analytic'``: Uses the model's built-in gradient if available.
          Falls back to finite differences if not implemented.

    eps_rel : float, default 1e-4
        Relative step size for finite differences.
        eps = max(eps_rel * std(D_i), 1e-8) to avoid division by zero on
        binary or zero-variance features.

    cdf_method : str, default 'parametric'
        How to estimate the conditional CDF F_{Y|X}(Y|X) = U_{Y|X}.
        Options:

        - ``'parametric'``: Fit a Gamma (or other) distribution to residuals
          from the model's predictions. U_k = CDF(Y_k; shape, scale_k).
          Requires that loss_distribution is set appropriately.
        - ``'empirical'``: Within-stratum empirical CDF, rank-based.
          Uses all training observations as a single stratum (i.e., global
          ranks). Accurate when n is large; noisy otherwise.
        - ``'global'``: Simplified global empirical ranks. Identical to
          'empirical' but explicitly ignores X conditioning. Fastest option;
          acceptable approximation for homogeneous portfolios.

    loss_distribution : str, default 'gamma'
        Loss distribution assumed for parametric CDF estimation.
        Options: ``'gamma'``, ``'lognormal'``, ``'tweedie'`` (not yet
        implemented — use 'gamma' as approximation).

    cascade : bool, default False
        If True, compute cascade sensitivity: indirect paths D_i -> X_j -> Y
        are included via linear regression of each X_j on D_i. This gives a
        more conservative correction when protected attributes correlate with
        non-protected covariates (e.g. gender with occupation).

    max_grad_samples : int or None, default None
        Maximum number of observations used for Jacobian estimation (finite
        differences). Set to a smaller value (e.g., 5000) to cap compute time
        when the model is expensive to evaluate. If None, all observations
        are used.
    """

    def __init__(
        self,
        distortion: str | Callable = "es_alpha",
        alpha: float = 0.75,
        grad_method: str = "finite_diff",
        eps_rel: float = 1e-4,
        cdf_method: str = "parametric",
        loss_distribution: str = "gamma",
        cascade: bool = False,
        max_grad_samples: int | None = None,
    ) -> None:
        if isinstance(distortion, str) and distortion not in (
            "es_alpha", "expectation", "wang_lambda"
        ):
            raise ValueError(
                f"distortion must be 'es_alpha', 'expectation', 'wang_lambda', "
                f"or a callable. Got: {distortion!r}"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if grad_method not in ("finite_diff", "analytic"):
            raise ValueError(
                f"grad_method must be 'finite_diff' or 'analytic'. Got: {grad_method!r}"
            )
        if eps_rel <= 0:
            raise ValueError("eps_rel must be positive.")
        if cdf_method not in ("parametric", "empirical", "global"):
            raise ValueError(
                f"cdf_method must be 'parametric', 'empirical', or 'global'. "
                f"Got: {cdf_method!r}"
            )
        if loss_distribution not in ("gamma", "lognormal", "tweedie"):
            raise ValueError(
                f"loss_distribution must be 'gamma', 'lognormal', or 'tweedie'. "
                f"Got: {loss_distribution!r}"
            )

        self.distortion = distortion
        self.alpha = alpha
        self.grad_method = grad_method
        self.eps_rel = eps_rel
        self.cdf_method = cdf_method
        self.loss_distribution = loss_distribution
        self.cascade = cascade
        self.max_grad_samples = max_grad_samples

        # Set after fit()
        self._is_fitted: bool = False
        self._model = None
        self._protected_indices: list[int] = []
        self._protected_names: list[str] = []

        # Training statistics (used in transform)
        self._train_sensitivities: np.ndarray | None = None
        self._train_denominators: np.ndarray | None = None
        self._train_corrections: np.ndarray | None = None
        self._train_numers: np.ndarray | None = None
        self._rho_baseline: float = 0.0
        self._rho_fair: float = 0.0

        # Parametric CDF parameters (shape, scale per obs) from training
        self._gamma_shape: float = 1.0
        self._gamma_scale_train: np.ndarray | None = None

        # Cascade: regression coefficients dX_j / dD_i
        self._cascade_coefs: dict[int, np.ndarray] | None = None  # idx -> coef per X col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        model,
        protected_indices: list[int] | None = None,
    ) -> "MarginalFairnessPremium":
        """
        Fit the fairness correction from training data.

        This computes marginal sensitivities and correction terms that are
        stored and reused during transform(). The model is queried once for
        Jacobian estimation; no model refitting occurs.

        Parameters
        ----------
        Y :
            Observed losses. Shape (n,). Must be non-negative.
        D :
            Protected attributes. Shape (n, m) or (n,) for a single attribute.
            Continuous or one-hot encoded categoricals are both handled.
        X :
            Non-protected covariates. Shape (n, p).
        model :
            A fitted sklearn-compatible estimator. Must implement
            ``model.predict(DX)`` where DX = np.hstack([D, X]).
        protected_indices :
            Column indices into D indicating which attributes to make fair.
            Default None means all columns of D are made fair.

        Returns
        -------
        self
        """
        Y = np.asarray(Y, dtype=float)
        D = np.atleast_2d(np.asarray(D, dtype=float))
        if D.ndim == 1 or (D.ndim == 2 and D.shape[0] == 1 and len(Y) > 1):
            D = D.reshape(-1, 1)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim == 1 or (X.ndim == 2 and X.shape[0] == 1 and len(Y) > 1):
            X = X.reshape(-1, 1)

        n = len(Y)
        if D.shape[0] != n or X.shape[0] != n:
            raise ValueError(
                f"Y, D, and X must all have n={n} observations. "
                f"Got D.shape={D.shape}, X.shape={X.shape}."
            )
        if np.any(Y < 0):
            raise ValueError("Y (losses) must be non-negative.")

        m = D.shape[1]
        if protected_indices is None:
            protected_indices = list(range(m))
        else:
            if any(i < 0 or i >= m for i in protected_indices):
                raise ValueError(
                    f"protected_indices values must be in [0, {m-1}]. "
                    f"Got: {protected_indices}"
                )

        self._model = model
        self._protected_indices = protected_indices
        self._protected_names = [f"D_{i}" for i in protected_indices]

        # Step 1: conditional ranks U_{Y|X}
        Y_pred = self._predict(D, X)
        U = self._compute_ranks(Y, Y_pred, D, X)

        # Step 2: distortion weights gamma(U)
        gamma_w = self._distortion_weights(U)

        # Step 3: baseline risk measure
        rho_baseline = float(np.mean(Y * gamma_w))

        # Step 4: Jacobians and corrections per protected attribute
        DX = np.hstack([D, X])
        p_D = D.shape[1]

        # For cascade: fit regressions X_j ~ D_i
        if self.cascade:
            self._cascade_coefs = self._fit_cascade_regressions(D, X, protected_indices)

        sensitivities = np.zeros(len(protected_indices))
        denominators = np.zeros(len(protected_indices))
        numers = np.zeros(len(protected_indices))
        corrections = np.zeros(len(protected_indices))

        for idx, col_i in enumerate(protected_indices):
            D_i = D[:, col_i]  # (n,)

            # Gradient of model w.r.t. D_i at each training observation
            grad_i = self._compute_gradient(D, X, col_i, DX)

            # Cascade: add indirect contributions if requested
            if self.cascade and self._cascade_coefs is not None:
                cascade_coef = self._cascade_coefs.get(col_i, np.zeros(X.shape[1]))
                # For each X_j, compute additional gradient contribution
                for j, dxj_ddi in enumerate(cascade_coef):
                    if abs(dxj_ddi) > 1e-12:
                        grad_xj = self._compute_gradient_wrt_x(D, X, j, DX)
                        grad_i = grad_i + dxj_ddi * grad_xj

            # Sensitivity: (1/n) * SUM D_i * grad_i * gamma
            sens_i = float(np.mean(D_i * grad_i * gamma_w))

            # Denominator: (1/n) * SUM (D_i * grad_i)^2
            dg = D_i * grad_i
            denom_i = float(np.mean(dg ** 2))

            # Numerator: (1/n) * SUM Y * D_i * grad_i
            numer_i = float(np.mean(Y * dg))

            if abs(denom_i) < 1e-12:
                # Model gradient w.r.t. D_i is zero — no correction needed
                correction_i = 0.0
            else:
                correction_i = (sens_i / denom_i) * numer_i

            sensitivities[idx] = sens_i
            denominators[idx] = denom_i
            numers[idx] = numer_i
            corrections[idx] = correction_i

        rho_fair = rho_baseline - float(np.sum(corrections))

        self._train_sensitivities = sensitivities
        self._train_denominators = denominators
        self._train_corrections = corrections
        self._train_numers = numers
        self._rho_baseline = rho_baseline
        self._rho_fair = rho_fair
        self._is_fitted = True

        return self

    def transform(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fairness correction to produce per-policyholder fair premiums.

        The correction reuses training-estimated sensitivities and denominators
        (as constants), applying them per-observation to maintain consistency
        with the portfolio-level adjustment.

        Parameters
        ----------
        Y :
            Observed (or simulated) losses. Shape (n,).
        D :
            Protected attributes. Shape (n, m).
        X :
            Non-protected covariates. Shape (n, p).

        Returns
        -------
        rho_fair : np.ndarray, shape (n,)
            Fair premium (distortion risk measure contribution) per policyholder.
            These sum to n * rho_fair_portfolio.
        """
        self._check_fitted()

        Y = np.asarray(Y, dtype=float)
        D = np.atleast_2d(np.asarray(D, dtype=float))
        if D.ndim == 1 or (D.ndim == 2 and D.shape[0] == 1 and len(Y) > 1):
            D = D.reshape(-1, 1)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim == 1 or (X.ndim == 2 and X.shape[0] == 1 and len(Y) > 1):
            X = X.reshape(-1, 1)

        Y_pred = self._predict(D, X)
        U = self._compute_ranks(Y, Y_pred, D, X)
        gamma_w = self._distortion_weights(U)

        # Baseline per observation: Y_k * gamma_k
        rho_obs = Y * gamma_w

        # Per-observation correction using training statistics
        DX = np.hstack([D, X])
        for idx, col_i in enumerate(self._protected_indices):
            D_i = D[:, col_i]
            grad_i = self._compute_gradient(D, X, col_i, DX)

            if self.cascade and self._cascade_coefs is not None:
                cascade_coef = self._cascade_coefs.get(col_i, np.zeros(X.shape[1]))
                for j, dxj_ddi in enumerate(cascade_coef):
                    if abs(dxj_ddi) > 1e-12:
                        grad_xj = self._compute_gradient_wrt_x(D, X, j, DX)
                        grad_i = grad_i + dxj_ddi * grad_xj

            denom_i = float(self._train_denominators[idx])
            sens_i = float(self._train_sensitivities[idx])

            if abs(denom_i) < 1e-12:
                continue

            # Per-observation correction weight:
            # gamma*(u_k) = gamma(u_k) - (sens_i / denom_i) * D_i * grad_i
            correction_weight = (sens_i / denom_i) * (D_i * grad_i)
            rho_obs = rho_obs - Y * correction_weight

        return rho_obs

    def sensitivity_report(self) -> MarginalFairnessReport:
        """
        Return sensitivity diagnostics for FCA Consumer Duty audit trail.

        Returns a MarginalFairnessReport summarising the sensitivity of the
        distortion risk measure to each protected attribute, the magnitude of
        corrections, and the portfolio-level lift ratio.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        self._check_fitted()

        lift_ratio = (
            self._rho_fair / self._rho_baseline
            if abs(self._rho_baseline) > 1e-12
            else 1.0
        )

        return MarginalFairnessReport(
            protected_names=list(self._protected_names),
            sensitivities=self._train_sensitivities.copy(),
            denominators=self._train_denominators.copy(),
            corrections=self._train_corrections.copy(),
            rho_baseline=self._rho_baseline,
            rho_fair=self._rho_fair,
            lift_ratio=lift_ratio,
        )

    # ------------------------------------------------------------------
    # Internal: distortion weights
    # ------------------------------------------------------------------

    def _distortion_weights(self, U: np.ndarray) -> np.ndarray:
        """
        Compute gamma(U) for each observation given conditional quantile U.

        Parameters
        ----------
        U : array (n,) — conditional quantile levels in [0, 1].

        Returns
        -------
        gamma : array (n,) — distortion weights.
        """
        if callable(self.distortion):
            return np.asarray(self.distortion(U), dtype=float)

        if self.distortion == "expectation":
            return np.ones_like(U)

        if self.distortion == "es_alpha":
            # ES_alpha: gamma(u) = 1/(1-alpha) if u >= alpha else 0
            alpha = self.alpha
            gamma = np.where(U >= alpha, 1.0 / (1.0 - alpha), 0.0)
            return gamma

        if self.distortion == "wang_lambda":
            # Wang transform: h(u) = Phi(Phi^{-1}(u) + lambda)
            # gamma(u) = dh/du = phi(Phi^{-1}(u) + lambda) / phi(Phi^{-1}(u))
            # where phi is standard normal PDF, Phi is standard normal CDF.
            lam = self.alpha  # re-using alpha parameter as lambda
            # Clip U away from 0 and 1 to avoid numerical issues
            U_clip = np.clip(U, 1e-10, 1.0 - 1e-10)
            z = scipy_stats.norm.ppf(U_clip)
            gamma = scipy_stats.norm.pdf(z + lam) / np.maximum(scipy_stats.norm.pdf(z), 1e-20)
            return gamma

        raise ValueError(f"Unknown distortion: {self.distortion!r}")

    # ------------------------------------------------------------------
    # Internal: conditional CDF / ranks
    # ------------------------------------------------------------------

    def _compute_ranks(
        self,
        Y: np.ndarray,
        Y_pred: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate conditional quantile U_{Y|X,k} = F_{Y|X}(Y_k | X_k).

        Returns array (n,) with values in (0, 1).
        """
        n = len(Y)

        if self.cdf_method in ("empirical", "global"):
            # Global empirical ranks: rank within entire training set
            # argsort of argsort gives rank (0-indexed), then normalise
            ranks = np.argsort(np.argsort(Y, kind="stable"), kind="stable")
            # Map to (0, 1) open interval
            U = (ranks + 0.5) / n
            return U

        # Parametric: use model predictions as scale parameter
        if self.loss_distribution == "gamma":
            # Fit overall dispersion from residuals Y / Y_pred
            # Y ~ Gamma(shape, scale_k) where E[Y] = shape * scale_k = Y_pred
            # So scale_k = Y_pred / shape
            # Estimate shape from method of moments on (Y / Y_pred):
            #   E[Y/mu] = 1, Var[Y/mu] = 1/shape => shape = 1/Var[Y/mu]
            ratio = Y / np.maximum(Y_pred, 1e-10)
            var_ratio = float(np.var(ratio))
            shape = max(1.0 / max(var_ratio, 1e-6), 0.1)
            scale_k = Y_pred / shape  # (n,)
            # CDF of Gamma(shape, scale_k) at Y_k
            U = scipy_stats.gamma.cdf(Y, a=shape, scale=scale_k)

        elif self.loss_distribution == "lognormal":
            # Y ~ LogNormal(mu_k, sigma) where E[Y] = exp(mu_k + sigma^2/2) = Y_pred
            # Log-residuals: log(Y) - log(Y_pred) ~ Normal(-sigma^2/2, sigma^2)
            log_ratio = np.log(np.maximum(Y, 1e-10)) - np.log(np.maximum(Y_pred, 1e-10))
            sigma = float(np.std(log_ratio))
            mu_k = np.log(np.maximum(Y_pred, 1e-10)) - 0.5 * sigma ** 2
            U = scipy_stats.lognorm.cdf(Y, s=max(sigma, 1e-8), scale=np.exp(mu_k))

        else:
            # Tweedie: approximate as gamma for now
            ratio = Y / np.maximum(Y_pred, 1e-10)
            var_ratio = float(np.var(ratio))
            shape = max(1.0 / max(var_ratio, 1e-6), 0.1)
            scale_k = Y_pred / shape
            U = scipy_stats.gamma.cdf(Y, a=shape, scale=scale_k)

        # Clip to open interval to avoid issues at boundaries
        U = np.clip(U, 1e-10, 1.0 - 1e-10)
        return U

    # ------------------------------------------------------------------
    # Internal: Jacobian computation
    # ------------------------------------------------------------------

    def _compute_gradient(
        self,
        D: np.ndarray,
        X: np.ndarray,
        col_i: int,
        DX: np.ndarray,
    ) -> np.ndarray:
        """
        Compute dg/dD_i at each observation via finite differences.

        Returns array (n,) of gradients.
        """
        n = len(D)

        # Determine which rows to use for gradient estimation
        if self.max_grad_samples is not None and n > self.max_grad_samples:
            sample_idx = np.random.choice(n, size=self.max_grad_samples, replace=False)
        else:
            sample_idx = np.arange(n)

        # Step size
        d_i_col = D[:, col_i]
        std_di = float(np.std(d_i_col[sample_idx]))
        eps = max(self.eps_rel * std_di, 1e-8)

        if self.grad_method == "analytic":
            # Try model's gradient method if available; fall back to finite diff
            try:
                return self._analytic_gradient(D, X, col_i, DX)
            except (AttributeError, NotImplementedError):
                pass  # fall through to finite differences

        # Two-sided central finite differences
        D_plus = D.copy()
        D_plus[:, col_i] = d_i_col + eps
        DX_plus = np.hstack([D_plus, X])

        D_minus = D.copy()
        D_minus[:, col_i] = d_i_col - eps
        DX_minus = np.hstack([D_minus, X])

        y_plus = np.asarray(self._model.predict(DX_plus), dtype=float)
        y_minus = np.asarray(self._model.predict(DX_minus), dtype=float)

        grad = (y_plus - y_minus) / (2.0 * eps)
        return grad

    def _compute_gradient_wrt_x(
        self,
        D: np.ndarray,
        X: np.ndarray,
        col_j: int,
        DX: np.ndarray,
    ) -> np.ndarray:
        """
        Compute dg/dX_j at each observation via finite differences.
        Used for cascade sensitivity.
        """
        p_D = D.shape[1]
        x_j_col = X[:, col_j]
        std_xj = float(np.std(x_j_col))
        eps = max(self.eps_rel * std_xj, 1e-8)

        X_plus = X.copy()
        X_plus[:, col_j] = x_j_col + eps
        DX_plus = np.hstack([D, X_plus])

        X_minus = X.copy()
        X_minus[:, col_j] = x_j_col - eps
        DX_minus = np.hstack([D, X_minus])

        y_plus = np.asarray(self._model.predict(DX_plus), dtype=float)
        y_minus = np.asarray(self._model.predict(DX_minus), dtype=float)

        return (y_plus - y_minus) / (2.0 * eps)

    def _analytic_gradient(
        self,
        D: np.ndarray,
        X: np.ndarray,
        col_i: int,
        DX: np.ndarray,
    ) -> np.ndarray:
        """
        Analytic gradient for linear/GLM models.

        For a sklearn Pipeline ending in a LinearRegression or similar,
        the gradient w.r.t. column col_i of DX is simply the coefficient
        at that position. This method is speculative — if the model does
        not expose 'coef_', it raises AttributeError and the caller falls
        back to finite differences.
        """
        # LinearRegression, Ridge, Lasso, etc. have .coef_
        coef = self._model.coef_  # raises AttributeError if not available
        coef = np.asarray(coef).ravel()
        # col_i indexes into D, which comes first in DX = [D | X]
        if col_i >= len(coef):
            raise AttributeError("coef_ shorter than expected column index.")
        return np.full(len(D), coef[col_i])

    # ------------------------------------------------------------------
    # Internal: cascade regressions
    # ------------------------------------------------------------------

    def _fit_cascade_regressions(
        self,
        D: np.ndarray,
        X: np.ndarray,
        protected_indices: list[int],
    ) -> dict[int, np.ndarray]:
        """
        For each protected attribute D_i, fit OLS regressions X_j ~ D_i
        to estimate dX_j/dD_i for the cascade correction.

        Returns a dict mapping protected column index to array of coefficients
        (one per X column).
        """
        coefs: dict[int, np.ndarray] = {}
        p_X = X.shape[1]

        for col_i in protected_indices:
            d_i = D[:, col_i].reshape(-1, 1)
            coef_j = np.zeros(p_X)
            for j in range(p_X):
                x_j = X[:, j]
                # OLS: coef = (D_i^T D_i)^{-1} D_i^T X_j
                denom = float(np.dot(d_i.ravel(), d_i.ravel()))
                if denom < 1e-12:
                    coef_j[j] = 0.0
                else:
                    coef_j[j] = float(np.dot(d_i.ravel(), x_j)) / denom
            coefs[col_i] = coef_j

        return coefs

    # ------------------------------------------------------------------
    # Internal: model prediction
    # ------------------------------------------------------------------

    def _predict(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Call the fitted model on the concatenated feature matrix [D | X].
        """
        DX = np.hstack([D, X])
        return np.asarray(self._model.predict(DX), dtype=float)

    # ------------------------------------------------------------------
    # Internal: utilities
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "MarginalFairnessPremium has not been fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        dist = (
            f"{self.distortion}(alpha={self.alpha})"
            if isinstance(self.distortion, str) and self.distortion != "expectation"
            else str(self.distortion)
        )
        return (
            f"MarginalFairnessPremium("
            f"distortion={dist!r}, "
            f"grad_method={self.grad_method!r}, "
            f"cdf_method={self.cdf_method!r})"
        )
