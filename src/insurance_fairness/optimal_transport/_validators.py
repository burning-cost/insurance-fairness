"""Input validation for insurance-fairness-ot."""
from __future__ import annotations

import numpy as np
import polars as pl


def validate_exposure(exposure: np.ndarray | None, n: int) -> np.ndarray:
    """Return exposure array of shape (n,), defaulting to ones if None."""
    if exposure is None:
        return np.ones(n, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if exposure.shape != (n,):
        raise ValueError(f"exposure must have shape ({n},), got {exposure.shape}")
    if np.any(exposure < 0):
        raise ValueError("exposure must be non-negative")
    if np.all(exposure == 0):
        raise ValueError("exposure is all zero")
    return exposure


def validate_predictions(predictions: np.ndarray) -> np.ndarray:
    """Validate model output array."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 1:
        raise ValueError(f"predictions must be 1-D, got shape {predictions.shape}")
    if np.any(predictions <= 0):
        raise ValueError("predictions must be strictly positive (premiums cannot be zero or negative)")
    if np.any(~np.isfinite(predictions)):
        raise ValueError("predictions contain NaN or Inf")
    return predictions


def validate_dataframe_aligned(df: pl.DataFrame, name: str, n: int) -> None:
    """Check a Polars DataFrame has the expected number of rows."""
    if df.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows, got {df.shape[0]}")


def validate_epsilon(epsilon: float) -> None:
    """Check epsilon is in [0, 1]."""
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")


def validate_protected_attrs_present(
    protected_attrs: list[str], df: pl.DataFrame, df_name: str
) -> None:
    """Check all protected attribute columns exist in the DataFrame."""
    missing = [a for a in protected_attrs if a not in df.columns]
    if missing:
        raise ValueError(
            f"Protected attribute(s) {missing} not found in {df_name} columns {df.columns}"
        )
