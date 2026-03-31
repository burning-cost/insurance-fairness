"""
ProxyDiscriminationMeasure: LRTW EJOR 2026 Definition 4 (Equation 7).

Implements the sensitivity-based proxy discrimination (PD) metric and its
decomposition, following:

  Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
  Discrimination in Insurance Pricing. European Journal of Operational Research.
  DOI: 10.1016/j.ejor.2026.01.021.

The key quantity is the constrained regression of fitted prices pi(X) onto
the best-estimate conditional means mu(X, d) = E[Y | X=x, D=d]:

  PD(pi) = min_{c in R, v in V} E[(pi(X) - c - sum_d mu(X,d)*v_d)^2] / Var(pi(X))

where V = {v in [0,1]^|D| : sum v_d <= 1}.

PD = 0 iff the price avoids proxy discrimination (tight characterisation).
UF = Var(E[pi|D]) / Var(pi) measures demographic unfairness (first-order
Sobol on D) but UF=0 does NOT imply PD=0.

Terminology note
----------------
In the LRTW paper:
  Y    = observed loss (response)
  X    = non-protected covariates
  D    = protected attribute
  pi   = the pricing model being audited (fitted prices)
  mu(x,d) = E[Y|X=x, D=d] — best-estimate conditional mean

The fit() signature follows the task spec:
  y       -> observed losses (used to estimate mu if mu_hat not provided)
  X       -> covariate matrix
  D       -> protected attribute
  mu_hat  -> fitted prices pi(X) being audited (the model output)

If mu_hat is an ndarray of shape (n, |D|), it is interpreted as the
pre-computed mu(X, d) matrix (oracle conditional means).  In this case
the user must also supply the fitted prices via the ``pi`` parameter.

The most common usage pattern:
  1. You have a fitted GLM/GBM giving prices pi(X).  Pass as mu_hat.
     The class estimates mu(x,d) from y and X using gradient boosting.
  2. You also have oracle mu(x,d) estimates.  Pass mu_hat as an (n, |D|)
     array and supply pi separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    """Exposure-weighted variance of x."""
    w_norm = w / w.sum()
    mu = float(np.dot(w_norm, x))
    return float(np.dot(w_norm, (x - mu) ** 2))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Exposure-weighted mean of x."""
    w_norm = w / w.sum()
    return float(np.dot(w_norm, x))


def _encode_d(D: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Encode a discrete protected attribute as a zero-based integer index.

    Returns (D_encoded, categories) where categories[i] is the original
    label for encoded value i.
    """
    categories = sorted(set(D.tolist()), key=lambda v: (str(type(v)), v))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    D_encoded = np.array([cat_to_idx[v] for v in D], dtype=int)
    return D_encoded, categories


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@dataclass
class ProxyDiscriminationMeasure:
    """
    Sensitivity-based proxy discrimination measure (LRTW EJOR 2026, Def 4).

    Computes PD (proxy discrimination), UF (demographic unfairness), the
    discrimination residual Lambda, and the closest admissible price pi_star.

    Parameters
    ----------
    mu_estimator:
        Sklearn-compatible regressor used to estimate mu(x, d) = E[Y|X=x, D=d]
        when a pre-computed mu matrix is not supplied.  Must support
        ``fit(X, y, sample_weight=w)`` and ``predict(X)``.  Defaults to
        ``GradientBoostingRegressor(n_estimators=200, max_depth=3)``.
    exposure_weighted:
        If True (default) all expectations are exposure-weighted.

    Attributes
    ----------
    pd_score:
        Scalar in [0, 1].  PD = Var(Lambda) / Var(pi).
    uf_score:
        Scalar in [0, 1].  UF = Var(E[pi|D]) / Var(pi).
    v_star:
        Optimal weight vector v* in V, shape (|D|,).
    c_star:
        Optimal intercept c*.
    Lambda:
        Discrimination residual pi(X) - c* - sum_d mu(X,d)*v*_d, shape (n,).
    closest_admissible:
        Closest admissible price c* + sum_d mu(X,d)*v*_d, shape (n,).
    categories:
        List of original D category labels in the order used by v_star.
    mu_matrix:
        Array of shape (n, |D|) where column d is the estimate of
        mu(X, d) = E[Y|X=x, D=d] for each observation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> X = rng.normal(size=(n, 3))
    >>> D = rng.choice([0, 1], size=n)
    >>> # Observed losses
    >>> y = X[:, 0] + rng.normal(scale=0.3, size=n)
    >>> # Fitted prices — discrimination-free (depends only on X, not D)
    >>> pi = X[:, 0] + X[:, 1] * 0.2
    >>> m = ProxyDiscriminationMeasure()
    >>> m.fit(y, X, D, mu_hat=pi)
    >>> assert m.pd_score < 0.05
    """

    mu_estimator: object = field(default=None)
    exposure_weighted: bool = True

    # Fitted attributes
    pd_score: float = field(init=False, default=float("nan"))
    uf_score: float = field(init=False, default=float("nan"))
    v_star: np.ndarray = field(init=False, default=None)
    c_star: float = field(init=False, default=float("nan"))
    Lambda: np.ndarray = field(init=False, default=None)
    closest_admissible: np.ndarray = field(init=False, default=None)
    categories: list = field(init=False, default=None)
    mu_matrix: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        if self.mu_estimator is None:
            self.mu_estimator = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, random_state=42
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        D: np.ndarray,
        mu_hat: np.ndarray | dict | None = None,
        weights: np.ndarray | None = None,
        pi: np.ndarray | None = None,
    ) -> "ProxyDiscriminationMeasure":
        """
        Fit the proxy discrimination measure.

        Parameters
        ----------
        y:
            Observed losses (response), shape (n,).  Used to estimate
            mu(x,d) = E[Y|X=x, D=d] when mu_hat is a 1-D price vector or None.
        X:
            Non-protected covariate matrix, shape (n, p).  Must be numeric.
        D:
            Protected attribute, shape (n,).  Discrete; supports int or str
            categories.
        mu_hat:
            Fitted prices pi(X) being audited, or pre-computed best-estimate
            conditional means.  Three accepted forms:

            *   ``np.ndarray`` of shape (n,) — these are the fitted prices
                pi(X).  The conditional means mu(x,d) are estimated from y.
            *   ``np.ndarray`` of shape (n, |D|) — pre-computed mu matrix.
                Column d is E[Y|X, D=d_i] for the d-th category.  The fitted
                prices pi(X) must be supplied via the ``pi`` parameter.
            *   ``dict`` mapping each D category label to a length-n array —
                pre-computed mu matrix in dict form.  Requires ``pi`` too.
            *   None — fitted prices must then be supplied via ``pi``, and
                mu(x,d) is estimated from y via gradient boosting.

        weights:
            Exposure weights, shape (n,).  Defaults to uniform.
        pi:
            Fitted prices pi(X), shape (n,).  Required when mu_hat is an
            (n, |D|) matrix or dict (i.e. when pre-computed mu is supplied).
            Ignored when mu_hat is a 1-D array (fitted prices).

        Returns
        -------
        self
        """
        y_arr = np.asarray(y, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        D_raw = np.asarray(D)
        n = len(y_arr)

        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum() * n  # normalise so sum = n

        D_enc, self.categories = _encode_d(D_raw)
        n_d = len(self.categories)

        # ------------------------------------------------------------------
        # 1. Resolve pi (fitted prices being audited) and mu_matrix
        # ------------------------------------------------------------------
        if mu_hat is None:
            # No pre-computed values: pi must be supplied separately
            if pi is None:
                raise ValueError(
                    "When mu_hat=None, supply the fitted prices via the ``pi`` "
                    "parameter.  mu(x,d) will be estimated from y."
                )
            pi_arr = np.asarray(pi, dtype=float)
            self.mu_matrix = self._estimate_mu_matrix(y_arr, X_arr, D_enc, n_d, w)

        elif isinstance(mu_hat, dict):
            # Dict form: pre-computed mu matrix; need pi separately
            if pi is None:
                raise ValueError(
                    "When mu_hat is a dict of conditional means, supply the "
                    "fitted prices via the ``pi`` parameter."
                )
            pi_arr = np.asarray(pi, dtype=float)
            self.mu_matrix = np.column_stack(
                [np.asarray(mu_hat[c], dtype=float) for c in self.categories]
            )

        else:
            mu_arr = np.asarray(mu_hat, dtype=float)
            if mu_arr.ndim == 1:
                # 1-D: mu_hat IS the fitted prices pi(X)
                pi_arr = mu_arr
                self.mu_matrix = self._estimate_mu_matrix(y_arr, X_arr, D_enc, n_d, w)
            elif mu_arr.ndim == 2:
                # 2-D: mu_hat is the pre-computed mu(x,d) matrix
                if pi is None:
                    raise ValueError(
                        "When mu_hat is a 2-D matrix of conditional means, "
                        "supply the fitted prices via the ``pi`` parameter."
                    )
                pi_arr = np.asarray(pi, dtype=float)
                self.mu_matrix = mu_arr
            else:
                raise ValueError(
                    f"mu_hat must be 1-D (fitted prices) or 2-D (mu matrix), "
                    f"got ndim={mu_arr.ndim}."
                )

        # ------------------------------------------------------------------
        # 2. Solve constrained QP for v* and c*
        # ------------------------------------------------------------------
        self.v_star, self.c_star = self._solve_qp(pi_arr, self.mu_matrix, w)

        # ------------------------------------------------------------------
        # 3. Compute discrimination residual Lambda
        # ------------------------------------------------------------------
        pi_star = self.c_star + self.mu_matrix @ self.v_star
        self.Lambda = pi_arr - pi_star
        self.closest_admissible = pi_star

        # ------------------------------------------------------------------
        # 4. PD and UF scalars
        # ------------------------------------------------------------------
        var_pi = _weighted_var(pi_arr, w)
        var_lambda = _weighted_var(self.Lambda, w)
        self.pd_score = float(var_lambda / var_pi) if var_pi > 0 else 0.0
        self.uf_score = self._compute_uf(pi_arr, D_enc, w)

        # Store pi for downstream use
        self._pi = pi_arr

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_mu_matrix(
        self,
        y: np.ndarray,
        X: np.ndarray,
        D_enc: np.ndarray,
        n_d: int,
        w: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate mu(x,d) = E[Y|X=x, D=d] via gradient boosting.

        Augments X with D as a numeric column, fits one model on y, then
        predicts at each D value for all observations.

        Returns
        -------
        mu_matrix : shape (n, n_d)
        """
        n = len(y)
        # Augment X with D encoded as float
        X_aug = np.column_stack([X, D_enc.astype(float)])
        self.mu_estimator.fit(X_aug, y, sample_weight=w)

        mu_cols = []
        for d in range(n_d):
            X_d = np.column_stack([X, np.full(n, d, dtype=float)])
            mu_cols.append(self.mu_estimator.predict(X_d))
        return np.column_stack(mu_cols)

    @staticmethod
    def _solve_qp(
        pi: np.ndarray,
        mu_matrix: np.ndarray,
        w: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Solve the constrained quadratic programme (LRTW 2026, Eq. 7).

        min_{c in R, v in [0,1]^|D|, sum v <= 1}
            E_w[ (pi(X) - c - sum_d mu(X,d)*v_d)^2 ]

        The intercept c* is analytically eliminated (it equals
        E_w[pi] - sum_d E_w[mu_d]*v_d), leaving a QP in v only.

        Parameters
        ----------
        pi:
            Fitted prices, shape (n,).
        mu_matrix:
            Conditional mean matrix, shape (n, n_d).
        w:
            Exposure weights (positive, not required to sum to 1).

        Returns
        -------
        (v_star, c_star)
        """
        w_norm = w / w.sum()
        n_d = mu_matrix.shape[1]

        # Demean: work with centred variables to eliminate c analytically
        pi_c = pi - np.dot(w_norm, pi)
        mu_c = mu_matrix - (w_norm @ mu_matrix)  # shape (n, n_d)

        # Objective: E_w[(pi_c - mu_c @ v)^2]
        # = v^T (mu_c^T W mu_c) v - 2 (pi_c^T W mu_c) v + const
        # where W = diag(w_norm)
        Q = (mu_c * w_norm[:, None]).T @ mu_c   # (n_d, n_d)
        q = -2.0 * (pi_c * w_norm) @ mu_c       # (n_d,)

        # Regularise Q for numerical stability
        Q += 1e-10 * np.eye(n_d)

        def objective(v):
            return float(v @ Q @ v + q @ v)

        def gradient(v):
            return 2.0 * Q @ v + q

        # Constraints: sum(v) <= 1
        constraints = [{"type": "ineq", "fun": lambda v: 1.0 - v.sum()}]
        bounds = [(0.0, 1.0)] * n_d
        v0 = np.zeros(n_d)

        result = minimize(
            objective,
            v0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        v_star = np.clip(result.x, 0.0, 1.0)

        # Recover c* analytically
        mu_mean = w_norm @ mu_matrix  # (n_d,)
        c_star = float(np.dot(w_norm, pi)) - float(np.dot(mu_mean, v_star))

        return v_star, c_star

    @staticmethod
    def _compute_uf(pi: np.ndarray, D_enc: np.ndarray, w: np.ndarray) -> float:
        """
        Compute UF = Var(E[pi|D]) / Var(pi).

        This is the first-order Sobol index of pi with respect to D.
        For discrete D: E[pi|D=d] is the weighted group mean.
        """
        w_norm = w / w.sum()
        var_pi = float(np.dot(w_norm, (pi - np.dot(w_norm, pi)) ** 2))
        if var_pi <= 0:
            return 0.0

        mu_global = float(np.dot(w_norm, pi))
        between_var = 0.0
        for d_val in np.unique(D_enc):
            mask = D_enc == d_val
            w_d = w_norm[mask].sum()
            if w_d <= 0:
                continue
            mu_d = float(np.dot(w_norm[mask], pi[mask]) / w_d)
            between_var += w_d * (mu_d - mu_global) ** 2

        return float(between_var / var_pi)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a plain-text summary of PD and UF scores."""
        lines = [
            "Sensitivity-Based Proxy Discrimination (LRTW EJOR 2026)",
            f"  PD score (proxy discrimination) : {self.pd_score:.6f}",
            f"  UF score (demographic unfairness): {self.uf_score:.6f}",
            f"  c* (optimal intercept)           : {self.c_star:.4f}",
            "  v* (optimal weights) by category:",
        ]
        for cat, v in zip(self.categories, self.v_star):
            lines.append(f"    {cat}: {v:.6f}")
        lines.append(
            f"  Lambda std dev (residual spread)  : {float(np.std(self.Lambda)):.4f}"
        )
        return "\n".join(lines)
