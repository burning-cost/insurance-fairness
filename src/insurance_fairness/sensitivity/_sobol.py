"""
Sobol attribution of proxy discrimination to individual rating factors.

Implements LRTW EJOR 2026 Definition 5 (Equations 10-11):

  PD_S  = Var(E[Lambda | X_S]) / Var(pi)   (first-order)
  PDtilde_S = 1 - Var(E[Lambda | X_{-S}]) / Var(pi)  (total effect)

where Lambda = pi(X) - pi_star(X) is the discrimination residual and
X_S denotes the sub-vector of X indexed by S.

We estimate E[Lambda | X_S] using a RandomForestRegressor surrogate fitted
on the full feature set, then use the marginalisation trick: replace the
features *not* in S with out-of-bag predictions that effectively marginalise
them out. In practice we use mean imputation of excluded features
(set to their weighted mean) and re-predict — this is the Interventional
SHAP / conditional mean approach. For correlated features this is an
approximation; the exact approach would require sampling from the conditional
distribution of X_{-S} | X_S.

The first-order and total indices are computed per feature (S = {j}).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    w_norm = w / w.sum()
    mu = float(np.dot(w_norm, x))
    return float(np.dot(w_norm, (x - mu) ** 2))


def _conditional_var(
    target: np.ndarray,
    X_subset: np.ndarray,
    w: np.ndarray,
    rf: RandomForestRegressor,
    X_full: np.ndarray,
    feature_means: np.ndarray,
    subset_indices: list[int],
    all_indices: list[int],
) -> float:
    """
    Estimate Var(E[target | X_S]) using surrogate conditional expectations.

    Uses the surrogate RF (fitted on all features) to predict with non-S
    features replaced by their weighted mean, capturing only the X_S signal.
    """
    n = len(target)
    # Build an input matrix with non-S features fixed at their weighted mean
    X_input = np.copy(X_full)
    complement = [j for j in all_indices if j not in subset_indices]
    X_input[:, complement] = feature_means[complement]
    cond_mean = rf.predict(X_input)

    return _weighted_var(cond_mean, w)


@dataclass
class SobolAttribution:
    """
    Per-feature first-order and total Sobol-style attribution of PD.

    Parameters
    ----------
    n_estimators:
        Number of trees in the surrogate RandomForestRegressor.
    max_depth:
        Maximum tree depth for the surrogate.
    random_state:
        Random seed.

    Attributes
    ----------
    attributions_:
        DataFrame with columns ``feature``, ``first_order_pd``, ``total_pd``.
        Both are normalised by Var(pi).

    Examples
    --------
    >>> attr = SobolAttribution()
    >>> attr.fit(Lambda, X, pi, weights, feature_names=["age", "vehicle", "ncd"])
    >>> print(attr.attributions_)
    """

    n_estimators: int = 100
    max_depth: int = 6
    random_state: int = 42

    attributions_: pd.DataFrame = field(init=False, default=None)

    def fit(
        self,
        Lambda: np.ndarray,
        X: np.ndarray,
        pi: np.ndarray,
        weights: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "SobolAttribution":
        """
        Compute first-order and total Sobol PD indices per feature.

        Parameters
        ----------
        Lambda:
            Discrimination residual from ProxyDiscriminationMeasure, shape (n,).
        X:
            Non-protected covariate matrix, shape (n, p).  Numeric.
        pi:
            Fitted prices, shape (n,).  Used only to compute Var(pi).
        weights:
            Exposure weights, shape (n,).  Defaults to uniform.
        feature_names:
            Names for the p features.  Defaults to ['x0', 'x1', ...].

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        Lambda = np.asarray(Lambda, dtype=float)
        pi = np.asarray(pi, dtype=float)
        n, p = X.shape

        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum() * n

        if feature_names is None:
            feature_names = [f"x{j}" for j in range(p)]
        feature_names = list(feature_names)

        var_pi = _weighted_var(pi, w)
        if var_pi <= 0:
            self.attributions_ = pd.DataFrame(
                {"feature": feature_names,
                 "first_order_pd": np.zeros(p),
                 "total_pd": np.zeros(p)}
            )
            return self

        # Fit surrogate RF on Lambda ~ f(X)
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        rf.fit(X, Lambda, sample_weight=w)

        all_indices = list(range(p))
        feature_means = np.array(
            [float(np.dot(w / w.sum(), X[:, j])) for j in range(p)]
        )

        first_order = np.zeros(p)
        total_order = np.zeros(p)

        for j in range(p):
            # First-order: E[Lambda | X_j] — only feature j active
            first_order[j] = _conditional_var(
                Lambda, X[:, [j]], w, rf, X, feature_means,
                [j], all_indices
            ) / var_pi

            # Total: 1 - E[Lambda | X_{-j}] — all features except j
            complement = [i for i in all_indices if i != j]
            if len(complement) == 0:
                total_order[j] = first_order[j]
            else:
                var_complement = _conditional_var(
                    Lambda, X[:, complement], w, rf, X, feature_means,
                    complement, all_indices
                )
                total_order[j] = 1.0 - var_complement / var_pi

        # Clip to [0, 1] to handle numerical noise
        first_order = np.clip(first_order, 0.0, 1.0)
        total_order = np.clip(total_order, 0.0, 1.0)

        self.attributions_ = pd.DataFrame({
            "feature": feature_names,
            "first_order_pd": first_order,
            "total_pd": total_order,
        })

        return self
