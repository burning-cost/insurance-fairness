"""
Internal utilities for insurance-fairness-diag.

Exposure weighting, input validation, RAG thresholds, and common helpers
used across modules. Not part of the public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# RAG thresholds
# ---------------------------------------------------------------------------

# D_proxy RAG thresholds (normalised L2-distance, 0..1).
# These are not prescribed by the FCA. They reflect the authors' judgement
# about materiality in a Consumer Duty context.
#   Green  < 0.05 : minimal proxy leakage; document and monitor
#   Amber  0.05–0.15 : material leakage; investigate and explain
#   Red    > 0.15 : significant leakage; consider remediation
DEFAULT_D_PROXY_THRESHOLDS: dict[str, float] = {
    "amber": 0.05,
    "red": 0.15,
}

# Shapley effect RAG thresholds (individual factor phi, 0..1).
# A single factor driving more than 30% of discrimination is red.
DEFAULT_PHI_THRESHOLDS: dict[str, float] = {
    "amber": 0.10,
    "red": 0.30,
}


def d_proxy_rag(
    value: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """
    Return 'green', 'amber', or 'red' for a D_proxy value.

    Parameters
    ----------
    value:
        D_proxy scalar in [0, 1].
    thresholds:
        Override default thresholds. Must have keys 'amber' and 'red'.
    """
    t = thresholds or DEFAULT_D_PROXY_THRESHOLDS
    if value >= t["red"]:
        return "red"
    if value >= t["amber"]:
        return "amber"
    return "green"


def phi_rag(
    value: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Return 'green', 'amber', or 'red' for a Shapley effect phi value."""
    t = thresholds or DEFAULT_PHI_THRESHOLDS
    if value >= t["red"]:
        return "red"
    if value >= t["amber"]:
        return "amber"
    return "green"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_model(model: Any) -> None:
    """Raise TypeError if *model* does not have a callable predict method."""
    if not hasattr(model, "predict") or not callable(model.predict):
        raise TypeError(
            f"model must have a callable predict method. Got {type(model).__name__}."
        )


def validate_dataframe(X: pl.DataFrame, name: str = "X") -> None:
    """Raise TypeError if *X* is not a Polars DataFrame."""
    if not isinstance(X, pl.DataFrame):
        raise TypeError(
            f"{name} must be a Polars DataFrame. Got {type(X).__name__}. "
            "Convert with pl.from_pandas(df) if you have a pandas DataFrame."
        )


def validate_columns(df: pl.DataFrame, *cols: str) -> None:
    """Raise ValueError if any column is absent from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {df.columns}"
        )


def validate_positive_array(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if *arr* contains non-positive values."""
    if np.any(arr <= 0):
        raise ValueError(
            f"{name} must contain strictly positive values "
            f"(min found: {arr.min():.6f})."
        )


def validate_rating_factors(
    rating_factors: list[str],
    X: pl.DataFrame,
    sensitive_col: str,
) -> None:
    """
    Validate rating_factors list.

    Raises ValueError if:
    - rating_factors is empty
    - sensitive_col appears in rating_factors (it should not)
    - any factor is missing from X
    """
    if not rating_factors:
        raise ValueError("rating_factors must contain at least one factor.")
    if sensitive_col in rating_factors:
        raise ValueError(
            f"sensitive_col '{sensitive_col}' must not appear in rating_factors. "
            "rating_factors should be the legitimate pricing variables only."
        )
    validate_columns(X, *rating_factors)


# ---------------------------------------------------------------------------
# Exposure helpers
# ---------------------------------------------------------------------------


def resolve_exposure(
    X: pl.DataFrame,
    exposure_col: str | None,
    n: int,
) -> np.ndarray:
    """
    Return a numpy array of exposure weights.

    If *exposure_col* is None or absent, returns unit weights.
    """
    if exposure_col is not None and exposure_col in X.columns:
        w = X[exposure_col].to_numpy().astype(float)
        if np.any(w <= 0):
            raise ValueError(
                f"exposure_col '{exposure_col}' contains non-positive values."
            )
        return w
    return np.ones(n, dtype=float)


def exposure_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the exposure-weighted mean of *values*."""
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float((values * weights).sum() / total)


def exposure_weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the exposure-weighted variance of *values*."""
    mean = exposure_weighted_mean(values, weights)
    return float(exposure_weighted_mean((values - mean) ** 2, weights))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    weights: np.ndarray,
    stat_fn,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a weighted statistic.

    Parameters
    ----------
    values:
        Array of values to resample.
    weights:
        Corresponding exposure weights.
    stat_fn:
        Callable(values, weights) -> float.
    n_bootstrap:
        Number of bootstrap replicates.
    ci_level:
        Coverage level, e.g. 0.95 for 95% CI.
    rng:
        Optional numpy random Generator for reproducibility.

    Returns
    -------
    (lower, upper) bounds of the confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(values)
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(values[idx], weights[idx])

    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def subsample_indices(
    n: int,
    subsample_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return indices for a random subsample of size min(subsample_n, n).

    Returns all indices (sorted) if n <= subsample_n.
    """
    if n <= subsample_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=subsample_n, replace=False))
