"""
Internal utilities for insurance-fairness.

Handles Polars/pandas bridging, input validation, and common helpers used
across modules. Not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Polars / pandas bridge
# ---------------------------------------------------------------------------


def to_polars(data: "pl.DataFrame | pd.DataFrame") -> pl.DataFrame:
    """Convert a pandas DataFrame to Polars if necessary."""
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass
    if isinstance(data, pl.DataFrame):
        return data
    raise TypeError(
        f"Expected a Polars or pandas DataFrame, got {type(data).__name__}."
    )


def to_pandas(data: pl.DataFrame) -> "pd.DataFrame":
    """Convert a Polars DataFrame to pandas. Used where upstream deps require it."""
    return data.to_pandas()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_columns(df: pl.DataFrame, *cols: str) -> None:
    """Raise ValueError if any column is absent from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {df.columns}"
        )


def validate_positive(df: pl.DataFrame, col: str) -> None:
    """Raise ValueError if *col* contains non-positive values."""
    if df[col].min() <= 0:
        raise ValueError(
            f"Column '{col}' must contain strictly positive values "
            f"(min found: {df[col].min()})."
        )


def validate_binary(df: pl.DataFrame, col: str) -> None:
    """Raise ValueError if *col* contains values other than 0 and 1."""
    unique_vals = set(df[col].unique().to_list())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Column '{col}' must be binary (0/1). Found values: {unique_vals}."
        )


# ---------------------------------------------------------------------------
# Exposure helpers
# ---------------------------------------------------------------------------


def resolve_exposure(
    df: pl.DataFrame, exposure_col: str | None
) -> pl.Series:
    """
    Return the exposure series.

    If *exposure_col* is None or absent, return a series of ones with the
    same length as *df* (each policy counted equally).
    """
    if exposure_col is not None and exposure_col in df.columns:
        return df[exposure_col]
    return pl.Series("exposure", [1.0] * len(df))


def exposure_weighted_mean(values: pl.Series, exposure: pl.Series) -> float:
    """Return the exposure-weighted mean of *values*."""
    total_exposure = exposure.sum()
    if total_exposure == 0:
        return float("nan")
    weighted = (values * exposure).sum()
    return float(weighted / total_exposure)


# ---------------------------------------------------------------------------
# Decile helpers
# ---------------------------------------------------------------------------


def assign_prediction_deciles(
    df: pl.DataFrame,
    prediction_col: str,
    exposure_col: str | None = None,
    n_deciles: int = 10,
) -> pl.DataFrame:
    """
    Add a 'prediction_decile' column to *df*.

    Decile boundaries are based on the prediction values, not exposure-weighted
    quantiles. This matches common actuarial practice (order by predicted
    frequency or pure premium, split into bands).
    """
    validate_columns(df, prediction_col)
    quantile_cuts = [i / n_deciles for i in range(1, n_deciles)]
    breaks = df[prediction_col].quantile(quantile_cuts, interpolation="nearest")

    # Use cut with explicit breakpoints
    pred = df[prediction_col].to_numpy()
    breaks_arr = np.array(breaks)
    deciles = np.digitize(pred, breaks_arr) + 1  # 1-indexed
    deciles = np.clip(deciles, 1, n_deciles)

    return df.with_columns(pl.Series("prediction_decile", deciles.astype(np.int32)))


# ---------------------------------------------------------------------------
# Log-space helpers
# ---------------------------------------------------------------------------


def log_ratio(a: float, b: float) -> float:
    """
    Return log(a/b).

    Safe against division by zero - returns nan if b is 0.
    """
    if b == 0:
        return float("nan")
    return float(np.log(a / b))


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
    Return a bootstrap confidence interval for a weighted statistic.

    Parameters
    ----------
    values:
        Array of values.
    weights:
        Array of weights (exposures).
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
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(values[idx], weights[idx]))

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(stats, alpha))
    upper = float(np.quantile(stats, 1.0 - alpha))
    return lower, upper


# ---------------------------------------------------------------------------
# RAG status thresholds
# ---------------------------------------------------------------------------

# Default thresholds for traffic-light reporting.
# These are not prescribed by the FCA. Adjust to your firm's risk appetite.
# "Red" means the metric warrants immediate investigation and potential remediation.
# "Amber" means the metric should be documented and monitored.
# "Green" means no material concern identified.

DEFAULT_THRESHOLDS = {
    # Disparate impact ratio: values outside [0.8, 1.25] are flagged.
    # The US EEOC 4/5ths rule uses 0.8; the upper bound is the reciprocal.
    # These are indicative only for UK use.
    "disparate_impact_ratio": {"amber": (0.85, 1.18), "red": (0.80, 1.25)},
    # Proxy R-squared: above 0.1 warrants investigation.
    "proxy_r2": {"amber": 0.05, "red": 0.10},
    # Calibration disparity: ratio of actual/expected by group.
    # Greater than 10% deviation is amber; 20% is red.
    "calibration_disparity": {"amber": 0.10, "red": 0.20},
    # Demographic parity log-ratio (in log-space, so 0.1 ~ 10.5% price difference).
    "demographic_parity_log_ratio": {"amber": 0.05, "red": 0.10},
}


def rag_status(
    metric_name: str, value: float, thresholds: dict | None = None
) -> str:
    """
    Return 'green', 'amber', or 'red' for a metric value.

    Parameters
    ----------
    metric_name:
        One of the keys in DEFAULT_THRESHOLDS.
    value:
        Metric value to evaluate. For ratio metrics, pass the absolute value.
    thresholds:
        Override default thresholds. Must have the same structure as
        DEFAULT_THRESHOLDS[metric_name].
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS.get(metric_name, {})
    if not thresholds:
        return "unknown"

    t = thresholds

    # Ratio-type metrics with (lower, upper) tuple thresholds
    if metric_name in ("disparate_impact_ratio",):
        amber_lo, amber_hi = t["amber"]
        red_lo, red_hi = t["red"]
        if value < red_lo or value > red_hi:
            return "red"
        if value < amber_lo or value > amber_hi:
            return "amber"
        return "green"

    # Scalar thresholds (higher is worse)
    if metric_name in ("proxy_r2", "calibration_disparity", "demographic_parity_log_ratio"):
        red_val = t["red"]
        amber_val = t["amber"]
        if value >= red_val:
            return "red"
        if value >= amber_val:
            return "amber"
        return "green"

    return "unknown"
