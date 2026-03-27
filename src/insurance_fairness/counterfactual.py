"""
counterfactual.py
-----------------
Counterfactual fairness testing for insurance pricing models.

Counterfactual fairness asks: would the predicted premium change if the
policyholder's protected characteristic were different, holding all other
factors constant?

For a model that does not directly use protected characteristics as inputs,
this is equivalent to asking whether any input variable carries information
about the protected characteristic that influences the price. The
Lindholm-Richman-Tsanakas-Wüthrich (LRTW) discrimination-free price is the
natural counterfactual-fair price.

Two approaches are implemented:

1. Direct flip: flip the protected characteristic value and re-predict. Only
   applicable when the protected characteristic is a direct model input (e.g.
   age, sex where these are used explicitly).

2. LRTW marginalisation: average the model's predictions over the conditional
   distribution of the protected characteristic given non-protected factors.
   This is the theoretically correct approach for proxy discrimination and is
   applicable whether or not the model uses S directly.

Regulatory context
------------------
Counterfactual testing directly addresses FCA expectations that firms satisfy
themselves their pricing does not result in systematically worse outcomes for
groups sharing protected characteristics (Consumer Duty PRIN 2A, FCA Consumer
Duty Finalised Guidance FG22/5).

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from insurance_fairness._utils import (
    resolve_exposure,
    validate_columns,
)


@dataclass
class CounterfactualResult:
    """
    Result of a counterfactual fairness test.

    Attributes
    ----------
    protected_col:
        The protected characteristic tested.
    original_mean_premium:
        Exposure-weighted mean of original predictions.
    counterfactual_mean_premium:
        Exposure-weighted mean of counterfactual predictions.
    premium_impact_ratio:
        counterfactual_mean / original_mean. A ratio of 1.0 means flipping
        the protected characteristic has no average price impact.
    premium_impact_log:
        log(premium_impact_ratio). Zero means no impact.
    policy_level_impacts:
        Polars Series of per-policy counterfactual impacts (ratio of
        counterfactual to original prediction). Allows identification of
        the policies most affected.
    n_policies:
        Number of policies in the test set.
    method:
        'direct_flip' or 'lrtw_marginalisation'.
    """

    protected_col: str
    original_mean_premium: float
    counterfactual_mean_premium: float
    premium_impact_ratio: float
    premium_impact_log: float
    policy_level_impacts: pl.Series
    n_policies: int
    method: str

    def summary(self) -> str:
        """Return a plain-text summary of the counterfactual result."""
        lines = [
            f"Counterfactual fairness test: {self.protected_col}",
            f"Method: {self.method}",
            f"Policies tested: {self.n_policies:,}",
            f"Original mean premium: {self.original_mean_premium:.4f}",
            f"Counterfactual mean premium: {self.counterfactual_mean_premium:.4f}",
            f"Premium impact ratio: {self.premium_impact_ratio:.4f}",
            f"Premium impact (log): {self.premium_impact_log:+.4f}",
            f"  ({(self.premium_impact_ratio - 1) * 100:+.1f}%)",
            "",
            "Policy-level impact distribution (ratio of cf/original):",
            f"  5th pct:  {self.policy_level_impacts.quantile(0.05):.4f}",
            f"  25th pct: {self.policy_level_impacts.quantile(0.25):.4f}",
            f"  Median:   {self.policy_level_impacts.quantile(0.50):.4f}",
            f"  75th pct: {self.policy_level_impacts.quantile(0.75):.4f}",
            f"  95th pct: {self.policy_level_impacts.quantile(0.95):.4f}",
        ]
        return "\n".join(lines)


def counterfactual_fairness(
    model,
    df: pl.DataFrame,
    protected_col: str,
    feature_cols: Sequence[str],
    prediction_col: str | None = None,
    exposure_col: str | None = None,
    flip_values: dict[Any, Any] | None = None,
    method: str = "direct_flip",
    n_monte_carlo: int = 100,
    random_seed: int = 42,
) -> CounterfactualResult:
    """
    Test counterfactual fairness by measuring premium impact of flipping the
    protected characteristic.

    Parameters
    ----------
    model:
        Fitted CatBoost model (CatBoostRegressor or CatBoostClassifier).
        Must accept the same feature columns as the training data.
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column to flip.
    feature_cols:
        All feature columns the model uses for prediction. Must include
        *protected_col* if method='direct_flip'.
    prediction_col:
        Name of the existing prediction column in df. If None, predictions
        are computed from *model*.
    exposure_col:
        Exposure column.
    flip_values:
        Mapping of current value to counterfactual value. E.g.
        {'M': 'F', 'F': 'M'} for a gender flip.
        For binary columns (0/1): {0: 1, 1: 0}.
        If None, a binary flip (0 <-> 1) is assumed.
    method:
        'direct_flip': flip the protected characteristic and re-predict.
        'lrtw_marginalisation': average predictions over the marginal
        distribution of the protected characteristic (LRTW correction).
        This method is appropriate when the protected characteristic is not
        a direct model input but may correlate with inputs.
    n_monte_carlo:
        Number of Monte Carlo samples for 'lrtw_marginalisation'. Larger
        values give more stable estimates.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    CounterfactualResult
    """
    from catboost import Pool  # noqa: PLC0415

    validate_columns(df, protected_col, *feature_cols)
    exposure = resolve_exposure(df, exposure_col)
    exp_arr = exposure.to_numpy()

    # Compute original predictions
    if prediction_col is not None and prediction_col in df.columns:
        orig_preds = df[prediction_col].to_numpy().astype(float)
    else:
        X_pd = df.select(feature_cols).to_pandas()
        cat_cols = [
            c for c in feature_cols
            if df[c].dtype in (pl.String, pl.String, pl.Categorical)
        ]
        pool = Pool(X_pd, cat_features=cat_cols)
        orig_preds = model.predict(pool).astype(float)

    orig_mean = float(np.average(orig_preds, weights=exp_arr))

    if method == "direct_flip":
        cf_preds = _direct_flip(
            model=model,
            df=df,
            protected_col=protected_col,
            feature_cols=feature_cols,
            flip_values=flip_values,
        )
    elif method == "lrtw_marginalisation":
        cf_preds = _lrtw_marginalise(
            model=model,
            df=df,
            protected_col=protected_col,
            feature_cols=feature_cols,
            n_monte_carlo=n_monte_carlo,
            random_seed=random_seed,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'direct_flip' or "
            "'lrtw_marginalisation'."
        )

    cf_mean = float(np.average(cf_preds, weights=exp_arr))
    impact_ratio = cf_mean / orig_mean if orig_mean > 0 else float("nan")
    impact_log = float(np.log(impact_ratio)) if impact_ratio > 0 else float("nan")

    policy_impacts = pl.Series(
        "cf_impact_ratio",
        np.where(orig_preds > 0, cf_preds / orig_preds, float("nan")).tolist(),
    )

    return CounterfactualResult(
        protected_col=protected_col,
        original_mean_premium=orig_mean,
        counterfactual_mean_premium=cf_mean,
        premium_impact_ratio=impact_ratio,
        premium_impact_log=impact_log,
        policy_level_impacts=policy_impacts,
        n_policies=len(df),
        method=method,
    )


def _direct_flip(
    model,
    df: pl.DataFrame,
    protected_col: str,
    feature_cols: Sequence[str],
    flip_values: dict[Any, Any] | None = None,
) -> np.ndarray:
    """
    Flip the protected characteristic and return new model predictions.
    """
    from catboost import Pool  # noqa: PLC0415

    if protected_col not in feature_cols:
        raise ValueError(
            f"'{protected_col}' must be in feature_cols for direct_flip method. "
            "Use method='lrtw_marginalisation' for models that do not use the "
            "protected characteristic as a direct input."
        )

    # Build flip mapping if not provided
    if flip_values is None:
        unique_vals = df[protected_col].unique().to_list()
        if set(unique_vals).issubset({0, 1, True, False}):
            flip_values = {0: 1, 1: 0, True: False, False: True}
        elif len(unique_vals) == 2:
            flip_values = {unique_vals[0]: unique_vals[1], unique_vals[1]: unique_vals[0]}
        else:
            raise ValueError(
                "Cannot infer flip_values for a non-binary protected characteristic. "
                "Provide flip_values explicitly."
            )

    # Apply the flip
    flipped_series = df[protected_col].map_elements(
        lambda v: flip_values.get(v, v), return_dtype=df[protected_col].dtype
    )
    df_flipped = df.with_columns(flipped_series.alias(protected_col))

    cat_cols = [
        c for c in feature_cols
        if df[c].dtype in (pl.String, pl.String, pl.Categorical)
    ]
    X_pd = df_flipped.select(feature_cols).to_pandas()
    pool = Pool(X_pd, cat_features=cat_cols)
    return model.predict(pool).astype(float)


def _lrtw_marginalise(
    model,
    df: pl.DataFrame,
    protected_col: str,
    feature_cols: Sequence[str],
    n_monte_carlo: int = 100,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Compute discrimination-free prices by averaging model predictions over
    the marginal distribution of the protected characteristic.

    For each policy i, the discrimination-free price is:

        p_DF(x_i) = (1/K) * sum_k f(x_i, s_k)

    where s_k are drawn from the empirical marginal distribution of S (or,
    if S is not a direct model input, from the empirical distribution of all
    values of S in the dataset). The idea: if the protected characteristic
    had no influence on price, the model should produce the same prediction
    regardless of which value of S we assign.

    When the model does not use S as a direct input, all K samples produce
    the same prediction, so the result equals the original prediction. In
    that case, use proxy R-squared and SHAP proxy scores to detect indirect
    discrimination instead.

    This implementation handles the case where S is a direct model input.
    """
    from catboost import Pool  # noqa: PLC0415

    rng = np.random.default_rng(random_seed)
    n = len(df)

    # Empirical distribution of the protected characteristic
    s_vals = df[protected_col].to_numpy()

    if protected_col not in feature_cols:
        # Model does not use S directly - predictions won't change.
        warnings.warn(
            f"Protected characteristic '{protected_col}' is not in the model features. "
            "LRTW marginalisation has no effect — predictions are returned unchanged. "
            "Consider using proxy detection to check for indirect discrimination.",
            UserWarning,
            stacklevel=2,
        )
        cat_cols = [
            c for c in feature_cols
            if df[c].dtype in (pl.String, pl.String, pl.Categorical)
        ]
        X_pd = df.select(feature_cols).to_pandas()
        pool = Pool(X_pd, cat_features=cat_cols)
        return model.predict(pool).astype(float)

    cat_cols = [
        c for c in feature_cols
        if df[c].dtype in (pl.String, pl.String, pl.Categorical)
    ]

    accumulated = np.zeros(n, dtype=float)

    for _ in range(n_monte_carlo):
        # Sample a value of S for each policy from the marginal distribution
        sampled_s = rng.choice(s_vals, size=n, replace=True)

        # Build a modified DataFrame
        df_modified = df.with_columns(
            pl.Series(protected_col, sampled_s)
        )
        X_pd = df_modified.select(feature_cols).to_pandas()
        pool = Pool(X_pd, cat_features=cat_cols)
        preds = model.predict(pool).astype(float)
        accumulated += preds

    return accumulated / n_monte_carlo
