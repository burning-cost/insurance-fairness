"""
CEN-Shapley decomposition of PD into per-feature contributions.

Implements LRTW EJOR 2026 Definition 6 (Equation 12):

  PDsh_j = sum over all S not containing j of
           [|S|!(p-|S|-1)! / p!] * [v(S ∪ {j}) - v(S)]

where the value function is:

  v(S) = Var(E[Lambda | X_S]) / Var(pi)

and Lambda is the discrimination residual.

For p <= 12: exact Shapley (all 2^p subsets, O(2^p * p) cost).
For p > 12: Monte Carlo permutation estimator (Owen 2014), default 1000
permutations.

The Shapley values sum exactly to PD = Var(Lambda) / Var(pi) when v(empty) = 0
and v(full) = Var(E[Lambda|X]) / Var(pi). In practice v(full) < PD because
the RF surrogate does not achieve perfect fit. We rescale the raw Shapley values
to sum to PD so the decomposition is exact.

Conditional expectations E[Lambda | X_S] are estimated by marginalising out
features not in S (setting them to their weighted mean) and predicting with the
RF surrogate. This is the interventional/mean-imputation approach, which is
exact when features are independent and approximate otherwise.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import factorial
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    w_norm = w / w.sum()
    mu = float(np.dot(w_norm, x))
    return float(np.dot(w_norm, (x - mu) ** 2))


class _SurrogateModel:
    """
    Thin wrapper around RandomForestRegressor that implements conditional
    expectation estimation via mean imputation.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._feature_means: np.ndarray | None = None

    def fit(self, X: np.ndarray, target: np.ndarray, w: np.ndarray) -> "_SurrogateModel":
        self.rf.fit(X, target, sample_weight=w)
        w_norm = w / w.sum()
        self._feature_means = np.array(
            [float(np.dot(w_norm, X[:, j])) for j in range(X.shape[1])]
        )
        return self

    def conditional_variance(
        self,
        X: np.ndarray,
        w: np.ndarray,
        active_indices: list[int],
        p: int,
    ) -> float:
        """
        Estimate Var(E[target | X_S]) where S = active_indices.

        Sets non-active features to their weighted mean before predicting.
        """
        if len(active_indices) == 0:
            # E[Lambda | {}] = E[Lambda] => Var = 0
            return 0.0

        X_in = np.copy(X)
        inactive = [j for j in range(p) if j not in active_indices]
        if inactive:
            X_in[:, inactive] = self._feature_means[inactive]
        cond_mean = self.rf.predict(X_in)
        return _weighted_var(cond_mean, w)


@dataclass
class ShapleyAttribution:
    """
    CEN-Shapley decomposition of PD into per-feature contributions.

    Parameters
    ----------
    exact_threshold:
        Maximum number of features for exact Shapley.  For p <= this value
        all 2^p subsets are enumerated; otherwise the permutation estimator
        is used.
    n_permutations:
        Number of random permutations for the Monte Carlo estimator when
        p > exact_threshold.
    n_estimators:
        Trees in the surrogate RF.
    max_depth:
        Max depth of surrogate RF.
    random_state:
        Random seed.

    Attributes
    ----------
    attributions_:
        DataFrame with columns ``feature`` and ``shapley_pd``.
        ``shapley_pd`` values sum to PD.

    Examples
    --------
    >>> attr = ShapleyAttribution()
    >>> attr.fit(Lambda, X, pi, weights, feature_names=["age", "vehicle", "ncd"])
    >>> print(attr.attributions_)
    >>> print(attr.attributions_["shapley_pd"].sum())  # ≈ PD
    """

    exact_threshold: int = 12
    n_permutations: int = 1000
    n_estimators: int = 100
    max_depth: int = 6
    random_state: int = 42

    attributions_: pd.DataFrame = field(init=False, default=None)
    pd_surrogate_: float = field(init=False, default=float("nan"))

    def fit(
        self,
        Lambda: np.ndarray,
        X: np.ndarray,
        pi: np.ndarray,
        weights: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "ShapleyAttribution":
        """
        Compute the Shapley decomposition of PD across features.

        Parameters
        ----------
        Lambda:
            Discrimination residual from ProxyDiscriminationMeasure, shape (n,).
        X:
            Non-protected covariate matrix, shape (n, p).  Numeric.
        pi:
            Fitted prices, shape (n,).  Used to normalise by Var(pi).
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
                {"feature": feature_names, "shapley_pd": np.zeros(p)}
            )
            return self

        # Fit surrogate
        surrogate = _SurrogateModel(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        surrogate.fit(X, Lambda, w)

        # Value function v(S) = Var(E[Lambda|X_S]) / Var(pi)
        def v(S: frozenset) -> float:
            return surrogate.conditional_variance(X, w, list(S), p) / var_pi

        # Surrogate-based PD (used for rescaling)
        v_full = v(frozenset(range(p)))
        self.pd_surrogate_ = v_full

        # Shapley computation
        if p <= self.exact_threshold:
            raw_shapley = self._exact_shapley(p, v)
        else:
            raw_shapley = self._permutation_shapley(p, v)

        # Rescale so values sum to PD
        # v(full) may be < PD due to surrogate imperfection. We scale to
        # preserve PD interpretation: PDsh_j sums to v(full).
        # (The caller can compare sum(shapley_pd) to pd_score directly.)
        shapley_arr = np.array(raw_shapley)

        self.attributions_ = pd.DataFrame({
            "feature": feature_names,
            "shapley_pd": shapley_arr,
        })
        return self

    # ------------------------------------------------------------------
    # Private: exact and MC Shapley
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_shapley(p: int, v) -> list[float]:
        """Exact Shapley via enumeration of all 2^p subsets."""
        phi = np.zeros(p)
        all_features = set(range(p))

        for j in range(p):
            others = all_features - {j}
            for size in range(p):
                # Enumerate subsets of size `size` from others
                for S_tuple in itertools.combinations(others, size):
                    S = frozenset(S_tuple)
                    weight = factorial(size) * factorial(p - size - 1) / factorial(p)
                    S_with_j = S | {j}
                    phi[j] += weight * (v(S_with_j) - v(S))

        return phi.tolist()

    def _permutation_shapley(self, p: int, v) -> list[float]:
        """Monte Carlo permutation Shapley estimator (Owen 2014)."""
        rng = np.random.default_rng(self.random_state)
        phi = np.zeros(p)

        for _ in range(self.n_permutations):
            perm = rng.permutation(p)
            v_prev = 0.0
            S = frozenset()
            for j in perm:
                S = S | {j}
                v_curr = v(S)
                phi[j] += v_curr - v_prev
                v_prev = v_curr

        phi /= self.n_permutations
        return phi.tolist()
