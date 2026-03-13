"""Utility functions for exposure-weighted optimal transport and ECDF operations."""
from __future__ import annotations

import numpy as np

# numpy.trapz was renamed to numpy.trapezoid in NumPy 2.0
try:
    from numpy import trapezoid as _trapezoid
except ImportError:
    from numpy import trapz as _trapezoid  # type: ignore[attr-defined]


def exposure_weighted_ecdf(
    values: np.ndarray, exposure: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exposure-weighted empirical CDF.

    Returns (sorted_values, cumulative_weights) suitable for quantile interpolation.
    Handles duplicate values by taking the maximum cumulative weight at each step,
    consistent with the right-continuous convention for CDFs.
    """
    values = np.asarray(values, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    sorted_exp = exposure[order]
    total = sorted_exp.sum()
    if total == 0:
        raise ValueError("Total exposure is zero; cannot compute ECDF")
    cum_exp = np.cumsum(sorted_exp) / total
    return sorted_vals, cum_exp


def quantile_function(
    ecdf_x: np.ndarray, ecdf_y: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Evaluate the quantile function (inverse CDF) at probability levels u.

    Uses linear interpolation between ECDF points. Values of u outside [ecdf_y[0], 1]
    are clamped to the range of ecdf_x.
    """
    u = np.asarray(u, dtype=float)
    return np.interp(u, ecdf_y, ecdf_x)


def barycenter_quantile(
    ecdfs: list[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    n_quantiles: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Wasserstein barycenter quantile function as a weighted average.

    Returns (u_grid, bar_qf) where bar_qf[i] = sum_d w_d * Q_d(u_grid[i]).
    """
    u_grid = np.linspace(0.0, 1.0, n_quantiles)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    qf_matrix = np.vstack([quantile_function(x, y, u_grid) for x, y in ecdfs])
    bar_qf = weights @ qf_matrix
    return u_grid, bar_qf


def apply_ot_map(
    value: float,
    ecdf_x: np.ndarray,
    ecdf_y: np.ndarray,
    u_grid: np.ndarray,
    bar_qf: np.ndarray,
) -> float:
    """Apply the OT map T*(x) = Q_bar(F_s(x)) for a single scalar value."""
    u = float(np.interp(value, ecdf_x, ecdf_y))
    return float(np.interp(u, u_grid, bar_qf))


def apply_ot_correction(
    predictions: np.ndarray,
    group_labels: np.ndarray,
    ecdfs: dict[object, tuple[np.ndarray, np.ndarray]],
    u_grid: np.ndarray,
    bar_qf: np.ndarray,
    log_space: bool = True,
) -> np.ndarray:
    """Apply T*(x) = Q_bar(F_s(x)) for each observation.

    When log_space=True, the map operates on log(predictions) and the
    result is exponentiated, preserving multiplicativity.
    """
    predictions = np.asarray(predictions, dtype=float)
    result = predictions.copy()
    if log_space:
        vals = np.log(predictions)
    else:
        vals = predictions

    for group, (ex, ey) in ecdfs.items():
        mask = group_labels == group
        if not mask.any():
            continue
        group_vals = vals[mask]
        corrected = np.array(
            [apply_ot_map(v, ex, ey, u_grid, bar_qf) for v in group_vals]
        )
        result[mask] = np.exp(corrected) if log_space else corrected

    return result


def wasserstein_distance_1d(
    x: np.ndarray,
    y: np.ndarray,
    wx: np.ndarray | None = None,
    wy: np.ndarray | None = None,
) -> float:
    """Compute W2 distance between two 1-D distributions.

    Uses the closed-form identity: W2^2 = integral (Q_x(u) - Q_y(u))^2 du.
    """
    n_grid = 1000
    u = np.linspace(0, 1, n_grid)
    if wx is None:
        wx = np.ones(len(x))
    if wy is None:
        wy = np.ones(len(y))
    ex, ey_x = exposure_weighted_ecdf(x, wx)
    fy, ey_y = exposure_weighted_ecdf(y, wy)
    qx = quantile_function(ex, ey_x, u)
    qy = quantile_function(fy, ey_y, u)
    return float(np.sqrt(_trapezoid((qx - qy) ** 2, u)))
