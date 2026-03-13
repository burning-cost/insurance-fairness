"""
Owen 2014 permutation Shapley effects for proxy discrimination attribution.

Implements the random permutation estimator from:

  Owen, A.B. (2014). Sobol' indices and Shapley value.
  SIAM/ASA Journal on Uncertainty Quantification 2(1), 245-251.

Applied to proxy discrimination measurement following:

  Biessy, G. (2024). Revisiting the Discrimination-Free Principle Through
  Shapley Values. ASTIN Bulletin (in press).

  Lindholm, Richman, Tsanakas, Wüthrich (2024). What is Fair? Proxy
  Discrimination vs. Demographic Disparities in Insurance Pricing.
  Scandinavian Actuarial Journal.

The characteristic function used is the variance of the conditional
expectation of the discrimination residual:

  v(S) = Var_w[ E_w[D | X_S] ]

where D = h - h_star is the discrimination residual and X_S is the
feature subset. This decomposes total discrimination variance across factors.

We estimate the conditional expectation E[D | X_S] using a surrogate model
fitted on D ~ f(X). The surrogate is a RandomForestRegressor by default.

For each random permutation of features, the marginal contribution of factor
j at position k is:

  delta_j(pi) = v(pi_1, ..., pi_k) - v(pi_1, ..., pi_{k-1})

Shapley effect phi_j = mean over all permutations of delta_j(pi).

SALib does NOT implement this variant (Owen 2014 Shapley effects). The SALib
shapley_effects function implements the Janon/Gamboa variance decomposition
for sensitivity analysis, which is different. We implement Owen 2014 natively.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor


def _build_surrogate(
    X_sub: np.ndarray,
    D: np.ndarray,
    weights: np.ndarray,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Fit a RandomForestRegressor surrogate for D = h - h_star on X.

    The surrogate allows fast evaluation of E[D | X_S] for arbitrary
    feature subsets S without re-scoring the original model.

    Parameters
    ----------
    X_sub:
        Feature matrix (n_samples, n_features) -- numeric only, post-encoding.
    D:
        Discrimination residual (n_samples,).
    weights:
        Exposure weights for sample_weight in RF fitting.
    random_state:
        Random seed.

    Returns
    -------
    Fitted RandomForestRegressor.
    """
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_sub, D, sample_weight=weights)
    return rf


def _characteristic_function(
    surrogate: RandomForestRegressor,
    X_sub: np.ndarray,
    D: np.ndarray,
    weights: np.ndarray,
    feature_subset: Sequence[int],
    n_features: int,
) -> float:
    """
    Estimate v(S) = Var_w[E_w[D | X_S]].

    We estimate E[D | X_S] by predicting D using only the features in
    *feature_subset*, with all other features marginalised by replacing
    them with their exposure-weighted mean (mean imputation approximates
    marginalisation for uncorrelated features; for correlated features
    this is a known approximation).

    Parameters
    ----------
    surrogate:
        Fitted surrogate model (fitted on ALL features).
    X_sub:
        Feature matrix (n_samples, n_features).
    D:
        Discrimination residual (n_samples,).
    weights:
        Exposure weights.
    feature_subset:
        Indices of features in subset S.
    n_features:
        Total number of features.

    Returns
    -------
    Estimated v(S) = Var_w[E[D | X_S]].
    """
    if len(feature_subset) == 0:
        # v(empty set) = 0 by convention (no information)
        return 0.0

    # Build X_S: replace non-S features with their weighted mean
    X_masked = X_sub.copy()
    total_weight = weights.sum()
    for j in range(n_features):
        if j not in feature_subset:
            col_mean = float((X_sub[:, j] * weights).sum() / total_weight)
            X_masked[:, j] = col_mean

    # E[D | X_S] approximated by surrogate prediction on X_masked
    E_D_given_XS = surrogate.predict(X_masked)

    # v(S) = Var_w[E[D | X_S]]
    w_mean = float((E_D_given_XS * weights).sum() / total_weight)
    var_s = float(((E_D_given_XS - w_mean) ** 2 * weights).sum() / total_weight)
    return var_s


def compute_shapley_effects(
    surrogate: RandomForestRegressor,
    X_sub: np.ndarray,
    D: np.ndarray,
    weights: np.ndarray,
    factor_names: list[str],
    n_perms: int = 256,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compute Owen 2014 permutation Shapley effects.

    For p features and n_perms random permutations, estimates the Shapley
    effect phi_j for each feature j. The phi_j values sum to 1.0 (after
    normalisation by total variance v(all features)).

    Parameters
    ----------
    surrogate:
        Fitted surrogate model for the discrimination residual.
    X_sub:
        Feature matrix (n_samples, n_features), matching surrogate training.
    D:
        Discrimination residual (n_samples,).
    weights:
        Exposure weights.
    factor_names:
        Names of features (length = n_features).
    n_perms:
        Number of random permutations to draw.
    random_state:
        Random seed.

    Returns
    -------
    Dict mapping factor name -> normalised Shapley effect in [0, 1].
    Sum of values is 1.0 (or 0.0 if total variance is 0).
    """
    rng = np.random.default_rng(random_state)
    n_features = len(factor_names)
    feature_indices = list(range(n_features))

    # Accumulate marginal contributions per feature
    phi_raw = np.zeros(n_features)
    perm_count = np.zeros(n_features, dtype=int)

    for _ in range(n_perms):
        perm = rng.permutation(n_features).tolist()
        v_prev = 0.0
        subset: list[int] = []

        for k, j in enumerate(perm):
            subset_with_j = subset + [j]
            v_with_j = _characteristic_function(
                surrogate, X_sub, D, weights, subset_with_j, n_features
            )
            phi_raw[j] += v_with_j - v_prev
            perm_count[j] += 1
            v_prev = v_with_j
            subset = subset_with_j

    # Average over permutations
    phi_avg = np.where(perm_count > 0, phi_raw / perm_count, 0.0)

    # Normalise: total variance is v(all features)
    total_var = _characteristic_function(
        surrogate, X_sub, D, weights, feature_indices, n_features
    )

    if total_var <= 0:
        # No discriminatory variance -- all phi = 0
        normalised = np.zeros(n_features)
    else:
        # Normalise so that sum = 1.0
        normalised = phi_avg / total_var
        # Clip negatives (numerical noise from permutation estimator)
        normalised = np.clip(normalised, 0.0, None)
        phi_sum = normalised.sum()
        if phi_sum > 0:
            normalised = normalised / phi_sum

    return {name: float(normalised[j]) for j, name in enumerate(factor_names)}


def fit_surrogate_and_compute_shapley(
    X: np.ndarray,
    D: np.ndarray,
    weights: np.ndarray,
    factor_names: list[str],
    n_perms: int = 256,
    surrogate_model: RandomForestRegressor | None = None,
    random_state: int = 42,
) -> tuple[dict[str, float], RandomForestRegressor]:
    """
    Fit surrogate and compute Shapley effects in one step.

    Parameters
    ----------
    X:
        Feature matrix (numeric, post-encoding).
    D:
        Discrimination residual h - h_star.
    weights:
        Exposure weights.
    factor_names:
        Feature names (length = X.shape[1]).
    n_perms:
        Number of permutations.
    surrogate_model:
        Pre-fitted surrogate. If None, fits a RandomForestRegressor.
    random_state:
        Random seed.

    Returns
    -------
    (phi_dict, surrogate)
    """
    if surrogate_model is None:
        surrogate = _build_surrogate(X, D, weights, random_state=random_state)
    else:
        surrogate = surrogate_model

    phi = compute_shapley_effects(
        surrogate=surrogate,
        X_sub=X,
        D=D,
        weights=weights,
        factor_names=factor_names,
        n_perms=n_perms,
        random_state=random_state,
    )

    return phi, surrogate
