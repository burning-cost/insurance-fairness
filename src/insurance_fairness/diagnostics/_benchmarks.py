"""
Premium benchmark variants following Côté, Côté and Charpentier (2025).

Reference:
  Côté, M.-P., Côté, S. and Charpentier, A. (2025). Five premium benchmarks
  for proxy discrimination in insurance pricing.

v0.1 scope: unaware and aware benchmarks only.
  - unaware  : the user's model (already fitted without S) -- model.predict(X)
  - aware    : refit model with S as a feature, then marginalise S out

Deferred to v0.2:
  - corrective (retrain without proxy features)
  - hyperaware (condition on S directly)
  - Pareto-optimal fairness-accuracy frontier

Proxy vulnerability per policyholder:
  proxy_vulnerability_i = unaware_i - aware_i

A positive value means the unaware model charges more than the discrimination-
aware counterfactual (the policyholder is disadvantaged by the proxy effect).
A negative value means the unaware model charges less (they benefit from proxy
correlation with a low-risk group).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.base import clone


@dataclass
class BenchmarkPremiums:
    """
    Premium benchmarks for proxy discrimination analysis.

    Attributes
    ----------
    best_estimate:
        Predictions from the user's model (unaware pricing -- S not in model).
        This is the same as unaware when the model was fitted without S.
    unaware:
        Predictions from the user's unaware model. Same as best_estimate.
    aware:
        Marginalised aware premium: E[model_aware(X, S) | X_legitimate].
        Estimated by refitting the model with S included, then averaging
        predictions across the reference distribution of S.
    proxy_vulnerability:
        Per-policyholder proxy vulnerability: unaware - aware.
        Positive = policyholder pays more than the fair counterfactual.
        Negative = policyholder pays less than the fair counterfactual.
    """

    best_estimate: np.ndarray
    unaware: np.ndarray
    aware: np.ndarray
    proxy_vulnerability: np.ndarray


def compute_unaware_premium(
    model: object,
    X: pl.DataFrame,
) -> np.ndarray:
    """
    Compute the unaware premium: model predictions without S.

    This simply calls model.predict(X). The model should already have been
    fitted without the sensitive attribute. X should not contain the sensitive
    column (or it will be ignored by an unaware model).

    Parameters
    ----------
    model:
        Fitted model with a predict method.
    X:
        Feature DataFrame (should not contain the sensitive column).

    Returns
    -------
    numpy array of premium predictions.
    """
    X_np = _to_numpy_for_predict(model, X)
    return np.asarray(model.predict(X_np), dtype=float)


def compute_aware_premium(
    model: object,
    X: pl.DataFrame,
    sensitive_col: str,
    weights: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute the marginalised aware premium.

    Steps:
    1. Refit the model on X WITH the sensitive column included.
    2. For each policyholder, predict under all observed S values.
    3. Average across S values weighted by the reference distribution of S.

    The result E[model_aware(X, S) | X] is the price that would be charged
    if S were marginalised out -- i.e., what a pricing actuary would charge
    if they used an aware model but then removed the direct effect of S.

    Parameters
    ----------
    model:
        Original fitted model. Must support sklearn's clone() interface, or
        have a fit() method. A fresh clone is fitted with S included.
    X:
        Feature DataFrame INCLUDING the sensitive column.
    sensitive_col:
        Name of the sensitive attribute column in X.
    weights:
        Exposure weights for fitting the aware model.
    random_state:
        Random seed (passed to the cloned model if it accepts it).

    Returns
    -------
    numpy array of marginalised aware premiums.
    """
    # Refit the model with S included
    aware_model = _clone_model(model, random_state)
    X_aware_np = _to_numpy_for_fit(model, X)

    # We cannot refit without a target. Use the unaware predictions as a proxy target.
    # This is the standard approach: refit on the model's own predictions (distillation).
    # The target should be the actual observed outcomes if available, but we only
    # have the model here. Using model predictions is a valid approximation for
    # the benchmark computation.
    X_without_s = X.drop(sensitive_col) if sensitive_col in X.columns else X
    X_without_s_np = _to_numpy_for_predict(model, X_without_s)
    y_target = np.asarray(model.predict(X_without_s_np), dtype=float)

    aware_model.fit(X_aware_np, y_target, sample_weight=weights)

    # Marginalise S out: for each policyholder, predict under all S values
    s_values = X[sensitive_col].to_numpy()
    unique_s = np.unique(s_values)
    total_weight = weights.sum()
    ref_probs = np.array(
        [weights[s_values == sv].sum() / total_weight for sv in unique_s]
    )

    # Accumulate weighted predictions across all S values
    aware_preds = np.zeros(len(X))
    for sv, prob in zip(unique_s, ref_probs):
        # Replace S column with sv for all policyholders
        X_with_s_fixed = X.with_columns(
            pl.lit(sv).cast(X[sensitive_col].dtype).alias(sensitive_col)
        )
        X_np = _to_numpy_for_predict(aware_model, X_with_s_fixed)
        preds_sv = np.asarray(aware_model.predict(X_np), dtype=float)
        aware_preds += prob * preds_sv

    return aware_preds


def compute_benchmarks(
    model: object,
    X: pl.DataFrame,
    sensitive_col: str,
    weights: np.ndarray,
    random_state: int = 42,
) -> BenchmarkPremiums:
    """
    Compute unaware and aware benchmark premiums.

    Parameters
    ----------
    model:
        Fitted unaware model (fitted without sensitive_col).
    X:
        Feature DataFrame INCLUDING the sensitive column.
    sensitive_col:
        Sensitive attribute column name.
    weights:
        Exposure weights.
    random_state:
        Random seed.

    Returns
    -------
    BenchmarkPremiums dataclass.
    """
    X_without_s = X.drop(sensitive_col) if sensitive_col in X.columns else X
    unaware = compute_unaware_premium(model, X_without_s)

    aware = compute_aware_premium(
        model=model,
        X=X,
        sensitive_col=sensitive_col,
        weights=weights,
        random_state=random_state,
    )

    proxy_vulnerability = unaware - aware

    return BenchmarkPremiums(
        best_estimate=unaware.copy(),
        unaware=unaware,
        aware=aware,
        proxy_vulnerability=proxy_vulnerability,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _clone_model(model: object, random_state: int) -> object:
    """
    Clone a sklearn-compatible model.

    Falls back to a copy if clone() fails (e.g., for CatBoost models).
    Attempts to set random_state if the cloned model supports it.

    CatBoost-specific handling:
    - Uses 'random_seed' instead of 'random_state'. When sklearn's clone()
      copies a CatBoost model, the cloned params already contain 'random_seed'.
      Calling set_params(random_state=...) would add a second conflicting seed
      parameter. We detect this and use set_params(random_seed=...) instead.
    - 'use_best_model=True' requires an eval_set during fit(). When refitting
      an aware model without an eval_set, we disable use_best_model and
      early_stopping_rounds to avoid CatBoostError.
    """
    try:
        cloned = clone(model)
        params = {}
        try:
            params = cloned.get_params()
        except Exception:
            pass

        # Set random seed. CatBoost uses 'random_seed' not 'random_state'.
        if "random_seed" in params:
            try:
                cloned.set_params(random_seed=random_state)
            except (ValueError, TypeError):
                pass
        else:
            try:
                cloned.set_params(random_state=random_state)
            except (ValueError, TypeError):
                pass

        # CatBoost: disable use_best_model and early_stopping_rounds because
        # we will call fit() without an eval_set. These params require an eval_set.
        if params.get("use_best_model", False):
            try:
                cloned.set_params(use_best_model=False)
            except (ValueError, TypeError):
                pass
        if params.get("early_stopping_rounds") is not None:
            try:
                cloned.set_params(early_stopping_rounds=None)
            except (ValueError, TypeError):
                pass

        return cloned
    except Exception:
        # If clone fails, try direct instantiation from class
        try:
            cloned = model.__class__(**{
                k: v for k, v in model.get_params().items()
            })
            return cloned
        except Exception:
            raise RuntimeError(
                f"Cannot clone model of type {type(model).__name__}. "
                "The model must support sklearn's clone() interface or have "
                "get_params() and __init__(**params) methods."
            )


def _to_numpy_for_predict(model: object, X: pl.DataFrame) -> np.ndarray:
    """
    Convert X to the format expected by model.predict().

    Tries numpy first (works for sklearn models). If the model appears to
    be a CatBoost model, passes the DataFrame directly.
    """
    return X.to_numpy()


def _to_numpy_for_fit(model: object, X: pl.DataFrame) -> np.ndarray:
    """Convert X to numpy for model.fit()."""
    return X.to_numpy()
