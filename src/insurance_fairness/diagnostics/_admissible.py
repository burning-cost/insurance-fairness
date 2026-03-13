"""
Admissible price set and proxy discrimination scalar (D_proxy).

Implements the L2-distance from the fitted price to the admissible
(discrimination-free) price set, following:

  Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
  Pricing. ASTIN Bulletin 52(1), 55-89.

  Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
  Proxy Discrimination. European Journal of Operational Research (SSRN 4897265).

The admissible price h_star is computed as the within-S-group mean prediction.
This represents the price that a model would give if it could only "see" the
S-group-level information, removing all within-group discrimination while
preserving between-group differences.

However, D_proxy is defined as the between-group dispersion of h, normalised
by the total spread of h. It measures how strongly the model's predictions
co-vary with the sensitive attribute S.

  D_proxy = sqrt( E_w[ (E[h|S] - E[h])^2 ] ) / sqrt( E_w[ (h - E[h])^2 ] )

This is 0 when h is independent of S (no proxy discrimination), and increases
as h becomes more correlated with S.

The admissible price h_star_i = E[h | S = s_i] (the conditional expectation
of h given the sensitive group). The deviation h_i - h_star_i is the
within-group residual (legitimate variation unexplained by S).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import (
    bootstrap_ci,
    d_proxy_rag,
    exposure_weighted_mean,
)


def compute_admissible_price(
    h: np.ndarray,
    s: np.ndarray,
    weights: np.ndarray,
    reference_dist: str = "observed",
) -> np.ndarray:
    """
    Compute the admissible (discrimination-free) price h_star.

    h_star_i = E_w[h | S = s_i] -- the exposure-weighted mean prediction
    within each sensitive group. This is the price the model assigns on
    average to policyholders in the same S group as policyholder i.

    For an unaware model, variation in h within S groups comes from legitimate
    factors only. The between-group variation in h (i.e., h_star varying across
    S values) is the discriminatory component measured by D_proxy.

    Parameters
    ----------
    h:
        Fitted prices (model predictions), shape (n,).
    s:
        Sensitive attribute values (any dtype -- used as group keys), shape (n,).
    weights:
        Exposure weights, shape (n,).
    reference_dist:
        Currently only 'observed' is supported.

    Returns
    -------
    h_star:
        Admissible prices: within-S-group exposure-weighted mean, shape (n,).
    """
    if reference_dist != "observed":
        raise ValueError(
            f"reference_dist='{reference_dist}' is not supported. Use 'observed'."
        )

    unique_s = np.unique(s)

    # Compute E[h | S=sv] for each unique S value
    group_means: dict[Any, float] = {}
    for sv in unique_s:
        mask = s == sv
        group_means[sv] = exposure_weighted_mean(h[mask], weights[mask])

    # h_star_i = group mean for policyholder i's S group
    h_star = np.array([float(group_means[sv]) for sv in s])
    return h_star


def compute_d_proxy(
    h: np.ndarray,
    h_star: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute the normalised L2 proxy discrimination scalar D_proxy.

    D_proxy measures the between-group dispersion of h relative to the
    total variation in h:

      D_proxy = sqrt( E_w[ (h_star - mu_h)^2 ] ) / sqrt( E_w[ (h - mu_h)^2 ] )

    where:
      h_star_i = E_w[h | S = s_i]  (within-group mean = between-group component)
      mu_h = E_w[h]                (global mean)

    This equals 0 when all group means are equal (h is independent of S),
    and approaches 1 when within-group variation is negligible compared to
    between-group variation.

    When h_star is computed as the within-group mean of h (per
    compute_admissible_price), the numerator is the between-group variance
    of h, and the denominator is the total variance of h. D_proxy is thus
    the square root of the R-squared of regressing h on S (ANOVA R^2).

    Parameters
    ----------
    h:
        Fitted prices.
    h_star:
        Admissible prices (within-S-group means).
    weights:
        Exposure weights.

    Returns
    -------
    Scalar in [0, 1].
    """
    mu_h = exposure_weighted_mean(h, weights)

    # Between-group variance: E[(h_star - mu_h)^2]
    between_var = exposure_weighted_mean((h_star - mu_h) ** 2, weights)

    # Total variance: E[(h - mu_h)^2]
    total_var = exposure_weighted_mean((h - mu_h) ** 2, weights)

    if total_var <= 0:
        # All predictions identical -- no discrimination possible
        return 0.0

    return float(np.sqrt(between_var / total_var))


def compute_d_proxy_with_ci(
    h: np.ndarray,
    h_star: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[float, float]]:
    """
    Compute D_proxy and a bootstrap confidence interval.

    The bootstrap resamples policyholders (with replacement) and recomputes
    D_proxy on each resample. This captures sampling uncertainty in the
    between-group dispersion estimate.

    Parameters
    ----------
    h:
        Fitted prices.
    h_star:
        Admissible prices (within-S-group means).
    weights:
        Exposure weights.
    n_bootstrap:
        Number of bootstrap replicates.
    ci_level:
        Coverage level for the CI.
    rng:
        Random Generator for reproducibility.

    Returns
    -------
    (d_proxy, (ci_lower, ci_upper))
    """
    if rng is None:
        rng = np.random.default_rng(42)

    d_proxy = compute_d_proxy(h, h_star, weights)

    n = len(h)
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[i] = compute_d_proxy(h[idx], h_star[idx], weights[idx])

    alpha = (1.0 - ci_level) / 2.0
    ci = (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha)))
    return d_proxy, ci
