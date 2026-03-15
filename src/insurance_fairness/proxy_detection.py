"""
proxy_detection.py
------------------
Detect which rating factors act as proxies for protected characteristics.

A rating factor X_j is a proxy for protected attribute S if knowing X_j
provides information about S that is useful for predicting S. Proxy detection
is the first step in a discrimination audit: it identifies which factors
warrant deeper analysis via attribution and calibration checks.

Three complementary approaches are provided:

1. Mutual information: model-free, captures non-linear dependencies.
2. Partial correlation: linear association after controlling for other factors.
3. SHAP-based proxy scores: how much does each feature's SHAP contribution
   correlate with the protected characteristic?
4. Proxy R-squared: CatBoost-predicted R-squared of S from X_j.

Regulatory context
------------------
FCA TR24/2 (2024) and Consumer Duty (PRIN 2A) require firms to satisfy
themselves that rating factors do not result in systematically worse outcomes
for groups defined by protected characteristics. Mutual information and proxy
R-squared are the primary diagnostics; SHAP attribution links proxy correlation
to actual price impact.

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
Discrimination in Insurance Pricing. European Journal of Operational Research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from insurance_fairness._utils import (
    DEFAULT_THRESHOLDS,
    rag_status,
    resolve_exposure,
    validate_columns,
)


def _encode_series(series: pl.Series) -> np.ndarray:
    """Encode a Polars Series to a float numpy array, label-encoding strings."""
    if series.dtype in (pl.Utf8, pl.String, pl.Categorical):
        return series.cast(pl.Categorical).to_physical().to_numpy().astype(float)
    return series.to_numpy().astype(float)


@dataclass
class ProxyScore:
    """Proxy score for a single rating factor / protected characteristic pair."""

    factor: str
    protected_col: str
    proxy_r2: float | None
    mutual_information: float | None
    partial_correlation: float | None
    shap_proxy_score: float | None
    rag: str


@dataclass
class ProxyDetectionResult:
    """Ranked table of proxy scores for all rating factors."""

    protected_col: str
    scores: list[ProxyScore]

    def to_polars(self) -> pl.DataFrame:
        """Return proxy scores as a Polars DataFrame, sorted by proxy R-squared."""
        rows = [
            {
                "factor": s.factor,
                "protected_col": s.protected_col,
                "proxy_r2": s.proxy_r2,
                "mutual_information": s.mutual_information,
                "partial_correlation": s.partial_correlation,
                "shap_proxy_score": s.shap_proxy_score,
                "rag": s.rag,
            }
            for s in self.scores
        ]
        df = pl.DataFrame(rows)
        if "proxy_r2" in df.columns:
            df = df.sort("proxy_r2", descending=True, nulls_last=True)
        return df

    @property
    def flagged_factors(self) -> list[str]:
        """Factors with amber or red proxy status."""
        return [s.factor for s in self.scores if s.rag in ("amber", "red")]


# ---------------------------------------------------------------------------
# Proxy R-squared via CatBoost
# ---------------------------------------------------------------------------


def proxy_r2_scores(
    df: pl.DataFrame,
    protected_col: str,
    factor_cols: Sequence[str],
    exposure_col: str | None = None,
    catboost_iterations: int = 100,
    catboost_depth: int = 4,
    random_seed: int = 42,
    is_binary_protected: bool | None = None,
) -> dict[str, float]:
    """
    Compute CatBoost-based proxy R-squared for each rating factor.

    For each factor X_j, fits a CatBoost model predicting the protected
    characteristic S from X_j alone, and returns the R-squared on a held-out validation set. For binary
    S, R-squared is computed between predicted probabilities and true
    binary labels. For continuous S, standard R-squared is used. Both
    are clamped to [0, 1].

    A high value indicates X_j is a strong proxy for S: knowing X_j tells
    you a lot about the policyholder's protected characteristic. This does not
    by itself prove discrimination - the factor may also legitimately predict
    risk. Use calibration and attribution metrics for the full picture.

    The threshold for flagging is R-squared > 0.10 (configurable in
    DEFAULT_THRESHOLDS['proxy_r2']). This is not an FCA-prescribed threshold.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column. Can be binary (0/1), categorical,
        or continuous (e.g. ONS ethnicity proportion at LSOA level).
    factor_cols:
        Rating factor columns to test as potential proxies.
    exposure_col:
        Exposure column. Used as sample weights in the CatBoost fit.
    catboost_iterations:
        CatBoost training iterations (keep low for diagnostic speed).
    catboost_depth:
        CatBoost tree depth.
    random_seed:
        Random seed for reproducibility.
    is_binary_protected:
        If True, use classification and return AUC. If False, use regression
        and return R-squared. If None, inferred from the column dtype.

    Returns
    -------
    dict mapping factor name to proxy R-squared (R^2 between predicted
    probability/value and true S, on a held-out validation set).
    """
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # noqa: PLC0415
    from sklearn.metrics import r2_score, roc_auc_score  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    validate_columns(df, protected_col, *factor_cols)
    exposure = resolve_exposure(df, exposure_col)

    s_series = df[protected_col]
    if is_binary_protected is None:
        is_binary_protected = s_series.dtype in (pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64) and set(s_series.unique().to_list()).issubset({0, 1, True, False})

    s_arr = _encode_series(s_series)
    exp_arr = exposure.to_numpy().astype(float)

    results: dict[str, float] = {}

    for col in factor_cols:
        x_arr = df[col].to_numpy()

        # Identify categorical: string or low-cardinality integer
        col_dtype = df[col].dtype
        is_cat = col_dtype == pl.Utf8 or col_dtype == pl.String or col_dtype == pl.Categorical

        # Handle string categories for CatBoost
        if is_cat:
            x_arr = x_arr.astype(str)

        # Train/validation split (stratified for classification)
        try:
            x_train, x_val, s_train, s_val, w_train, w_val = train_test_split(
                x_arr.reshape(-1, 1),
                s_arr,
                exp_arr,
                test_size=0.2,
                random_state=random_seed,
                stratify=s_arr.astype(int) if is_binary_protected else None,
            )
        except Exception:
            # Fallback: unstratified split
            x_train, x_val, s_train, s_val, w_train, w_val = train_test_split(
                x_arr.reshape(-1, 1),
                s_arr,
                exp_arr,
                test_size=0.2,
                random_state=random_seed,
            )

        cat_features = [0] if is_cat else []

        if is_binary_protected:
            model = CatBoostClassifier(
                iterations=catboost_iterations,
                depth=catboost_depth,
                random_seed=random_seed,
                verbose=0,
                allow_writing_files=False,
            )
            train_pool = Pool(x_train, s_train.astype(int), weight=w_train, cat_features=cat_features)
            val_pool = Pool(x_val, s_val.astype(int), weight=w_val, cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20)
            probs = model.predict_proba(x_val)[:, 1]
            try:
                # Use R-squared of predicted probabilities vs true labels.
                # This is consistent with the R-squared used for continuous S,
                # and makes the proxy_r2 field and its thresholds meaningful for
                # both binary and continuous protected characteristics.
                score = float(r2_score(s_val.astype(int), probs, sample_weight=w_val))
                score = max(0.0, score)  # Clamp negative R-squared to 0
            except Exception:
                score = float("nan")
        else:
            model = CatBoostRegressor(
                iterations=catboost_iterations,
                depth=catboost_depth,
                random_seed=random_seed,
                verbose=0,
                allow_writing_files=False,
                loss_function="RMSE",
            )
            train_pool = Pool(x_train, s_train, weight=w_train, cat_features=cat_features)
            val_pool = Pool(x_val, s_val, weight=w_val, cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20)
            preds = model.predict(x_val)
            try:
                score = float(r2_score(s_val, preds, sample_weight=w_val))
                score = max(0.0, score)  # Clamp negative R-squared to 0
            except Exception:
                score = float("nan")

        results[col] = score

    return results


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------


def mutual_information_scores(
    df: pl.DataFrame,
    protected_col: str,
    factor_cols: Sequence[str],
    exposure_col: str | None = None,
    is_binary_protected: bool | None = None,
    random_seed: int = 42,
) -> dict[str, float]:
    """
    Compute mutual information between each rating factor and the protected
    characteristic.

    Mutual information is model-free and captures non-linear dependencies.
    It is measured in nats (natural log base); higher values indicate stronger
    association. Values are not bounded above, so interpret relative to each
    other rather than against a fixed threshold.

    This is complementary to proxy R-squared: factors with high MI but low
    proxy R-squared may have a non-monotone relationship with S.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    factor_cols:
        Rating factor columns to test.
    exposure_col:
        Exposure column. Used as sample weights.
    is_binary_protected:
        If True, use classification MI. If None, inferred from column dtype.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    dict mapping factor name to mutual information (nats).
    """
    validate_columns(df, protected_col, *factor_cols)
    exposure = resolve_exposure(df, exposure_col)

    s_series = df[protected_col]
    if is_binary_protected is None:
        is_binary_protected = s_series.dtype in (pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64) and set(s_series.unique().to_list()).issubset({0, 1, True, False})

    s_arr = _encode_series(s_series)
    exp_arr = exposure.to_numpy().astype(float)

    X_matrix = []
    cat_col_indices = []

    for i, col in enumerate(factor_cols):
        col_dtype = df[col].dtype
        is_cat = col_dtype in (pl.Utf8, pl.String, pl.Categorical)
        if is_cat:
            # Encode strings as integer codes
            codes = df[col].cast(pl.Categorical).to_physical().to_numpy().astype(float)
            cat_col_indices.append(i)
            X_matrix.append(codes)
        else:
            X_matrix.append(df[col].to_numpy().astype(float))

    X_arr = np.column_stack(X_matrix) if X_matrix else np.empty((len(df), 0))

    if is_binary_protected:
        mi_vals = mutual_info_classif(
            X_arr,
            s_arr.astype(int),
            discrete_features=[i in cat_col_indices for i in range(len(factor_cols))],
            n_neighbors=3,
            random_state=random_seed,
        )
    else:
        mi_vals = mutual_info_regression(
            X_arr,
            s_arr,
            discrete_features=[i in cat_col_indices for i in range(len(factor_cols))],
            n_neighbors=3,
            random_state=random_seed,
        )

    return {col: float(mi_vals[i]) for i, col in enumerate(factor_cols)}


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------


def partial_correlation(
    df: pl.DataFrame,
    protected_col: str,
    factor_cols: Sequence[str],
    control_cols: Sequence[str] | None = None,
    exposure_col: str | None = None,
) -> dict[str, float]:
    """
    Compute partial Spearman correlation between each factor and the protected
    characteristic, controlling for other specified variables.

    For factor X_j and protected attribute S, the partial correlation controls
    for the variables in *control_cols*. This asks: "After accounting for the
    other variables, how much does X_j alone tell us about S?"

    A high partial correlation for postcode, for example, after controlling
    for vehicle type and age, indicates postcode carries ethnicity-related
    information beyond what the other factors explain.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    factor_cols:
        Factors to test.
    control_cols:
        Variables to control for. If None, no control is applied (raw
        Spearman correlation is returned).
    exposure_col:
        Exposure column (not used in partial correlation calculation, but
        kept for API consistency).

    Returns
    -------
    dict mapping factor name to partial Spearman correlation with S.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    validate_columns(df, protected_col, *factor_cols)
    if control_cols:
        validate_columns(df, *control_cols)

    def _encode(series: pl.Series) -> np.ndarray:
        if series.dtype in (pl.Utf8, pl.String, pl.Categorical):
            return series.cast(pl.Categorical).to_physical().to_numpy().astype(float)
        return series.to_numpy().astype(float)

    def _residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return OLS residuals of y on X (with intercept)."""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            residuals = y - X_with_intercept @ coeffs
        except np.linalg.LinAlgError:
            residuals = y
        return residuals

    s_arr = _encode(df[protected_col])

    if control_cols:
        control_matrix = np.column_stack([_encode(df[c]) for c in control_cols])
        s_residual = _residualise(s_arr, control_matrix)
    else:
        s_residual = s_arr

    results: dict[str, float] = {}
    for col in factor_cols:
        x_arr = _encode(df[col])
        if control_cols:
            x_residual = _residualise(x_arr, control_matrix)
        else:
            x_residual = x_arr
        corr, _ = spearmanr(x_residual, s_residual)
        results[col] = float(corr)

    return results


# ---------------------------------------------------------------------------
# SHAP-based proxy scores
# ---------------------------------------------------------------------------


def shap_proxy_scores(
    df: pl.DataFrame,
    protected_col: str,
    factor_cols: Sequence[str],
    shap_values: np.ndarray | None = None,
    model=None,
    prediction_col: str | None = None,
    exposure_col: str | None = None,
) -> dict[str, float]:
    """
    Compute SHAP-based proxy scores: the correlation between each feature's
    SHAP contribution to the price and the protected characteristic.

    A high correlation for feature X_j means: the contribution of X_j to the
    model's price prediction is systematically related to the protected
    characteristic. This is a stronger signal than proxy R-squared alone: it
    links feature correlation with S to actual price impact.

    Formally, for feature j:

        shap_proxy_score_j = |Spearman(SHAP_j, S)|

    Values near 0 indicate the feature's price impact is unrelated to the
    protected characteristic. Values near 1 indicate the feature's price
    contribution tracks the protected characteristic closely.

    This is a simplified approximation to the LRTW sensitivity measure.
    The full LRTW measure requires computing the discrimination-free price;
    this approach uses only SHAP values and is faster.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    factor_cols:
        Rating factors corresponding to columns in *shap_values*.
    shap_values:
        SHAP value matrix, shape (n_policies, n_features). If None, *model*
        must be provided and SHAP values will be computed via CatBoost.
    model:
        Fitted CatBoost model. Used to compute SHAP values if *shap_values*
        is None.
    prediction_col:
        Not used directly; kept for API consistency.
    exposure_col:
        Exposure column (not used in SHAP correlation calculation).

    Returns
    -------
    dict mapping factor name to SHAP proxy score (0 to 1).
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    validate_columns(df, protected_col, *factor_cols)

    if shap_values is None:
        if model is None:
            raise ValueError(
                "Either shap_values or model must be provided."
            )
        from catboost import Pool  # noqa: PLC0415

        # Identify categorical columns
        cat_cols = [
            c for c in factor_cols
            if df[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
        ]
        X_pd = df.select(factor_cols).to_pandas()
        pool = Pool(X_pd, cat_features=cat_cols)
        shap_values = model.get_feature_importance(pool, type="ShapValues")
        # CatBoost ShapValues returns (n, n_features + 1); drop the bias column
        shap_values = shap_values[:, : len(factor_cols)]

    if shap_values.shape[1] != len(factor_cols):
        raise ValueError(
            f"shap_values has {shap_values.shape[1]} columns but "
            f"{len(factor_cols)} factor_cols were provided."
        )

    s_arr = _encode_series(df[protected_col])
    results: dict[str, float] = {}

    for i, col in enumerate(factor_cols):
        shap_col = shap_values[:, i]
        corr, _ = spearmanr(shap_col, s_arr)
        results[col] = float(abs(corr))

    return results


# ---------------------------------------------------------------------------
# Combined proxy detection report
# ---------------------------------------------------------------------------


def detect_proxies(
    df: pl.DataFrame,
    protected_col: str,
    factor_cols: Sequence[str],
    exposure_col: str | None = None,
    model=None,
    run_proxy_r2: bool = True,
    run_mutual_info: bool = True,
    run_partial_corr: bool = True,
    run_shap: bool = False,
    catboost_iterations: int = 100,
    is_binary_protected: bool | None = None,
) -> ProxyDetectionResult:
    """
    Run all proxy detection methods and return a combined ranked report.

    This is the main entry point for proxy detection. Run all methods by
    default (except SHAP, which requires a fitted model).

    The RAG status is based on proxy R-squared (if computed) or mutual
    information as a fallback.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    factor_cols:
        Rating factors to test as proxies.
    exposure_col:
        Exposure column.
    model:
        Fitted CatBoost model (required only if run_shap=True).
    run_proxy_r2:
        Whether to compute CatBoost proxy R-squared.
    run_mutual_info:
        Whether to compute mutual information.
    run_partial_corr:
        Whether to compute partial Spearman correlations.
    run_shap:
        Whether to compute SHAP proxy scores. Requires *model*.
    catboost_iterations:
        CatBoost iterations for proxy R-squared computation.
    is_binary_protected:
        If None, inferred from column dtype.

    Returns
    -------
    ProxyDetectionResult with all scores and RAG statuses.
    """
    r2_scores: dict[str, float] = {}
    mi_scores: dict[str, float] = {}
    pc_scores: dict[str, float] = {}
    sp_scores: dict[str, float] = {}

    if run_proxy_r2:
        r2_scores = proxy_r2_scores(
            df,
            protected_col,
            factor_cols,
            exposure_col=exposure_col,
            catboost_iterations=catboost_iterations,
            is_binary_protected=is_binary_protected,
        )

    if run_mutual_info:
        mi_scores = mutual_information_scores(
            df,
            protected_col,
            factor_cols,
            exposure_col=exposure_col,
            is_binary_protected=is_binary_protected,
        )

    if run_partial_corr:
        pc_scores = partial_correlation(
            df,
            protected_col,
            factor_cols,
            exposure_col=exposure_col,
        )

    if run_shap and model is not None:
        sp_scores = shap_proxy_scores(
            df,
            protected_col,
            factor_cols,
            model=model,
            exposure_col=exposure_col,
        )

    scores: list[ProxyScore] = []
    for col in factor_cols:
        r2 = r2_scores.get(col)
        mi = mi_scores.get(col)
        pc = pc_scores.get(col)
        sp = sp_scores.get(col)

        # Use proxy R-squared for RAG if available, else mutual information
        if r2 is not None:
            rag = rag_status("proxy_r2", r2)
        else:
            rag = "unknown"

        scores.append(
            ProxyScore(
                factor=col,
                protected_col=protected_col,
                proxy_r2=r2,
                mutual_information=mi,
                partial_correlation=pc,
                shap_proxy_score=sp,
                rag=rag,
            )
        )

    # Sort by proxy R-squared descending
    scores.sort(
        key=lambda s: s.proxy_r2 if s.proxy_r2 is not None else -1.0,
        reverse=True,
    )

    return ProxyDetectionResult(protected_col=protected_col, scores=scores)
