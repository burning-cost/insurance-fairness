"""
bias_metrics.py
---------------
Fairness metrics for insurance pricing models.

All metrics support exposure weighting. Where a metric is defined in log-space
(for multiplicative models with a log link), this is the default. Pass
``log_space=False`` for additive models.

The metrics implemented here are:

- Demographic parity ratio (exposure-weighted, log-space)
- Equalised odds (exposure-weighted true/false positive rate parity)
- Calibration by group (sufficiency)
- Disparate impact ratio (exposure-weighted)
- Gini coefficient by group
- Theil index of premium inequality

Regulatory context
------------------
FCA Consumer Duty (PRIN 2A.4) requires firms to monitor whether products
provide fair value for different groups of customers, including groups defined
by protected characteristics (Equality Act 2010, Section 19 - Indirect
Discrimination). Calibration by group (sufficiency) is the most defensible
metric under the Equality Act: a model that is equally well-calibrated for all
groups does not systematically over-charge any group.

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from insurance_fairness._utils import (
    DEFAULT_THRESHOLDS,
    assign_prediction_deciles,
    bootstrap_ci,
    exposure_weighted_mean,
    log_ratio,
    rag_status,
    resolve_exposure,
    validate_columns,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupMetric:
    """A metric value for a single protected-characteristic group."""

    group_value: str
    metric_value: float
    n_policies: int
    total_exposure: float
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class DemographicParityResult:
    """
    Result of an exposure-weighted demographic parity check.

    log_ratio is log(mean_price_group1 / mean_price_group0), where groups are
    defined by a binary protected characteristic. Zero indicates no disparity.
    For continuous characteristics, see group_ratios.
    """

    protected_col: str
    log_ratio: float
    ratio: float
    group_means: dict[str, float]
    group_exposures: dict[str, float]
    rag: str
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class CalibrationResult:
    """
    Calibration (sufficiency) by group and prediction decile.

    actual_to_expected[decile][group] gives the A/E ratio for that cell.
    A perfectly calibrated model has all ratios equal to 1.0.
    max_disparity is the maximum absolute deviation from 1.0 across all cells.
    """

    protected_col: str
    actual_to_expected: dict[int, dict[str, float]]
    max_disparity: float
    group_counts: dict[str, dict[int, int]]
    rag: str


@dataclass
class DisparateImpactResult:
    """
    Disparate impact ratio (DIR): ratio of adverse outcome rates between groups.

    For pricing, DIR = mean_price_disadvantaged / mean_price_advantaged.
    Values below 0.80 are typically considered adverse (US EEOC 4/5ths rule);
    for UK use, interpret in context rather than applying this threshold
    mechanically.
    """

    protected_col: str
    ratio: float
    group_means: dict[str, float]
    group_exposures: dict[str, float]
    rag: str


@dataclass
class EqualisedOddsResult:
    """
    Equalised odds check for frequency models.

    Compares true positive rate (TPR) and false positive rate (FPR) across
    protected-characteristic groups. For regression pricing, adapted to
    compare within-group rank correlation of predictions and actuals.
    """

    protected_col: str
    group_metrics: list[GroupMetric]
    max_tpr_disparity: float
    rag: str


@dataclass
class GiniResult:
    """Gini coefficient of premium distribution, computed by group."""

    protected_col: str
    group_ginis: dict[str, float]
    overall_gini: float
    max_disparity: float


@dataclass
class TheilResult:
    """
    Theil T index of premium inequality, decomposed within- and between-group.

    T = T_within + T_between. A high T_between relative to T_total indicates
    that inequality is driven by systematic between-group differences rather
    than within-group variation.
    """

    protected_col: str
    theil_total: float
    theil_within: float
    theil_between: float
    group_contributions: dict[str, float]


# ---------------------------------------------------------------------------
# Demographic parity
# ---------------------------------------------------------------------------


def demographic_parity_ratio(
    df: pl.DataFrame,
    protected_col: str,
    prediction_col: str,
    exposure_col: str | None = None,
    log_space: bool = True,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
) -> DemographicParityResult:
    """
    Compute the exposure-weighted demographic parity ratio across groups.

    For a binary protected characteristic, this is:

        ratio = E[price * exposure | S=1] / E[price * exposure | S=0]

    In log-space (default for multiplicative pricing models):

        log_ratio = log(weighted_mean_S1) - log(weighted_mean_S0)

    Zero log_ratio (ratio = 1.0) indicates no parity difference.

    This metric does not control for risk differences between groups. A high
    ratio may reflect genuine risk differences rather than discrimination. Use
    :func:`calibration_by_group` to check whether price differences are
    justified by actual loss experience.

    Regulatory note: FCA Consumer Duty monitoring requires firms to check
    for differential outcomes by protected-characteristic group. Demographic
    parity disparity is a first-pass flag, not a conclusive discrimination test.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Name of the protected characteristic column (any dtype, including
        string categories).
    prediction_col:
        Name of the model prediction column (predicted premium or loss rate).
    exposure_col:
        Name of the exposure column. If None, all policies weighted equally.
    log_space:
        If True (default), compute metrics in log-space (appropriate for
        multiplicative models with a log link). If False, compute differences
        in levels (appropriate for additive models).
    n_bootstrap:
        Number of bootstrap replicates for confidence intervals. Set to 0 to
        skip (faster, no CIs).
    ci_level:
        Confidence interval coverage level (default 0.95).

    Returns
    -------
    DemographicParityResult
    """
    validate_columns(df, protected_col, prediction_col)
    exposure = resolve_exposure(df, exposure_col)

    groups = df[protected_col].unique().sort().to_list()
    group_means: dict[str, float] = {}
    group_exposures: dict[str, float] = {}

    for g in groups:
        mask = df[protected_col] == g
        g_pred = df.filter(mask)[prediction_col]
        g_exp = exposure.filter(mask)

        if log_space:
            g_vals = g_pred.log()
        else:
            g_vals = g_pred

        group_means[str(g)] = exposure_weighted_mean(g_vals, g_exp)
        group_exposures[str(g)] = float(g_exp.sum())

    group_keys = [str(g) for g in groups]

    if len(group_keys) == 2:
        ref, comp = group_keys[0], group_keys[1]
        if log_space:
            lr = group_means[comp] - group_means[ref]
            ratio = float(np.exp(lr))
        else:
            diff = group_means[comp] - group_means[ref]
            ref_mean = group_means[ref]
            lr = float(np.log(1 + diff / ref_mean)) if ref_mean != 0 else float("nan")
            ratio = (1 + diff / ref_mean) if ref_mean != 0 else float("nan")
    else:
        # For multi-group, compare each group to the exposure-weighted overall mean
        total_exp = sum(group_exposures.values())
        overall = sum(
            group_means[g] * group_exposures[g] / total_exp for g in group_keys
        )
        lr = max(abs(group_means[g] - overall) for g in group_keys)
        ratio = float(np.exp(lr)) if log_space else float(np.exp(lr))

    rag = rag_status("demographic_parity_log_ratio", abs(lr))

    ci_lower: float | None = None
    ci_upper: float | None = None
    if n_bootstrap > 0 and len(group_keys) == 2:
        ref, comp = group_keys[0], group_keys[1]
        ref_mask = (df[protected_col] == groups[0]).to_numpy()
        comp_mask = (df[protected_col] == groups[1]).to_numpy()

        pred_arr = df[prediction_col].to_numpy()
        exp_arr = exposure.to_numpy()

        if log_space:
            pred_arr = np.log(pred_arr)

        def _stat(vals: np.ndarray, wts: np.ndarray) -> float:
            m_ref = float(np.average(vals[ref_mask], weights=wts[ref_mask]))
            m_comp = float(np.average(vals[comp_mask], weights=wts[comp_mask]))
            return m_comp - m_ref

        ci_lower, ci_upper = bootstrap_ci(
            pred_arr, exp_arr, _stat, n_bootstrap=n_bootstrap, ci_level=ci_level
        )

    return DemographicParityResult(
        protected_col=protected_col,
        log_ratio=float(lr),
        ratio=ratio,
        group_means=group_means,
        group_exposures=group_exposures,
        rag=rag,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


# ---------------------------------------------------------------------------
# Calibration by group (sufficiency)
# ---------------------------------------------------------------------------


def calibration_by_group(
    df: pl.DataFrame,
    protected_col: str,
    prediction_col: str,
    outcome_col: str,
    exposure_col: str | None = None,
    n_deciles: int = 10,
) -> CalibrationResult:
    """
    Compute calibration (actual-to-expected ratio) by group and pricing decile.

    For each combination of (protected characteristic group, prediction decile),
    compute:

        A/E = sum(actual_claims) / sum(predicted_claims * exposure)

    A well-calibrated model has A/E = 1.0 for all groups at all deciles. Large
    deviations indicate the model is systematically over- or under-pricing
    a particular group within a pricing band.

    This is the 'sufficiency' criterion in the fairness literature (also called
    calibration fairness): the price is equally informative for each group.
    It is the most defensible criterion under the Equality Act 2010 Section 19:
    if the model is equally calibrated for all groups, any price differences
    reflect genuine risk differences, not proxy discrimination.

    Regulatory note: FCA Consumer Duty multi-firm review (2024) found that
    most firms lacked adequate monitoring of differential outcomes by
    demographic group. This metric directly satisfies that monitoring
    requirement.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    prediction_col:
        Model prediction column (expected loss rate or pure premium).
    outcome_col:
        Actual outcome column (claim amount or claim count).
    exposure_col:
        Exposure column. If None, all policies counted equally.
    n_deciles:
        Number of prediction deciles (default 10).

    Returns
    -------
    CalibrationResult
    """
    validate_columns(df, protected_col, prediction_col, outcome_col)
    exposure = resolve_exposure(df, exposure_col)
    df = df.with_columns(exposure.alias("_exposure"))
    df = assign_prediction_deciles(df, prediction_col, n_deciles=n_deciles)

    groups = df[protected_col].unique().sort().to_list()
    group_strs = [str(g) for g in groups]
    deciles = list(range(1, n_deciles + 1))

    ae: dict[int, dict[str, float]] = {}
    group_counts: dict[str, dict[int, int]] = {g: {} for g in group_strs}

    for d in deciles:
        ae[d] = {}
        for g, g_str in zip(groups, group_strs):
            cell = df.filter(
                (df["prediction_decile"] == d) & (df[protected_col] == g)
            )
            n = len(cell)
            group_counts[g_str][d] = n
            if n == 0:
                ae[d][g_str] = float("nan")
                continue

            actual = float(cell[outcome_col].sum())
            predicted = float((cell[prediction_col] * cell["_exposure"]).sum())
            ae[d][g_str] = actual / predicted if predicted > 0 else float("nan")

    # Maximum absolute disparity from 1.0 across all non-nan cells
    all_vals = [v for d_vals in ae.values() for v in d_vals.values() if not np.isnan(v)]
    max_disp = max(abs(v - 1.0) for v in all_vals) if all_vals else 0.0

    rag = rag_status("calibration_disparity", max_disp)

    return CalibrationResult(
        protected_col=protected_col,
        actual_to_expected=ae,
        max_disparity=max_disp,
        group_counts=group_counts,
        rag=rag,
    )


# ---------------------------------------------------------------------------
# Disparate impact ratio
# ---------------------------------------------------------------------------


def disparate_impact_ratio(
    df: pl.DataFrame,
    protected_col: str,
    prediction_col: str,
    exposure_col: str | None = None,
    reference_group: str | None = None,
) -> DisparateImpactResult:
    """
    Compute the exposure-weighted disparate impact ratio (DIR).

    DIR = (exposure-weighted mean price for disadvantaged group) /
          (exposure-weighted mean price for reference group)

    A ratio of 1.0 indicates equal average prices. The US EEOC 4/5ths rule
    flags values below 0.80 as adverse; this is a US regulatory concept and
    should not be applied mechanically in the UK. Use as a directional
    diagnostic alongside calibration and proxy correlation.

    For binary protected characteristics, the reference group is the group with
    the lower mean price (to ensure DIR <= 1.0 in the adverse direction). For
    multi-group characteristics, all groups are compared to the reference.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    prediction_col:
        Model prediction column.
    exposure_col:
        Exposure column. If None, all policies weighted equally.
    reference_group:
        Value of the reference (advantaged) group. If None, the group with
        the lowest exposure-weighted mean price is used as reference.

    Returns
    -------
    DisparateImpactResult
    """
    validate_columns(df, protected_col, prediction_col)
    exposure = resolve_exposure(df, exposure_col)

    groups = df[protected_col].unique().sort().to_list()
    group_means: dict[str, float] = {}
    group_exposures: dict[str, float] = {}

    for g in groups:
        mask = df[protected_col] == g
        g_pred = df.filter(mask)[prediction_col]
        g_exp = exposure.filter(mask)
        group_means[str(g)] = exposure_weighted_mean(g_pred, g_exp)
        group_exposures[str(g)] = float(g_exp.sum())

    if reference_group is None:
        # Use the group with the highest mean price as the reference
        # (DIR is then <= 1.0 for all other groups, making adverse cases < 1.0)
        reference_group = max(group_means, key=lambda k: group_means[k])

    ref_mean = group_means[reference_group]
    if len(groups) == 2:
        other = next(k for k in group_means if k != reference_group)
        ratio = group_means[other] / ref_mean if ref_mean > 0 else float("nan")
    else:
        # For multi-group, report the minimum ratio (most disadvantaged group)
        ratio = min(
            group_means[k] / ref_mean
            for k in group_means
            if k != reference_group and ref_mean > 0
        )

    rag = rag_status("disparate_impact_ratio", ratio)

    return DisparateImpactResult(
        protected_col=protected_col,
        ratio=ratio,
        group_means=group_means,
        group_exposures=group_exposures,
        rag=rag,
    )


# ---------------------------------------------------------------------------
# Equalised odds
# ---------------------------------------------------------------------------


def equalised_odds(
    df: pl.DataFrame,
    protected_col: str,
    prediction_col: str,
    outcome_col: str,
    exposure_col: str | None = None,
    binary_threshold: float | None = None,
) -> EqualisedOddsResult:
    """
    Compute equalised odds across protected-characteristic groups.

    For binary classification: compares true positive rate (TPR) and false
    positive rate (FPR) across groups. Both must be equal for equalised odds.

    For continuous regression (the typical insurance case): adapts to compare
    the Spearman rank correlation between predictions and actuals within each
    group. Groups where the model ranks risk less accurately may be
    systematically mis-priced.

    Note: strict equalised odds (equal TPR and FPR) conflicts with calibration
    when base rates differ across groups. The Chouldechova impossibility theorem
    shows these criteria cannot all hold simultaneously when group base rates
    differ. Calibration by group is the more appropriate criterion for
    insurance pricing.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    prediction_col:
        Prediction column.
    outcome_col:
        Actual outcome column. For binary classification, pass binary 0/1.
        For regression, pass claim amounts or counts.
    exposure_col:
        Exposure column.
    binary_threshold:
        If provided, predictions above this threshold are treated as positive.
        If None, the metric uses rank correlation (appropriate for regression).

    Returns
    -------
    EqualisedOddsResult
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    validate_columns(df, protected_col, prediction_col, outcome_col)
    exposure = resolve_exposure(df, exposure_col)

    groups = df[protected_col].unique().sort().to_list()
    group_metrics: list[GroupMetric] = []

    for g in groups:
        mask = df[protected_col] == g
        g_df = df.filter(mask)
        g_exp = exposure.filter(mask)
        n = len(g_df)

        pred = g_df[prediction_col].to_numpy()
        actual = g_df[outcome_col].to_numpy()

        if binary_threshold is not None:
            pred_bin = (pred >= binary_threshold).astype(int)
            actual_bin = actual.astype(int)
            tpr = float(np.mean(pred_bin[actual_bin == 1])) if actual_bin.sum() > 0 else float("nan")
            metric_val = tpr
        else:
            # Rank correlation as a continuous analogue
            if len(pred) < 3:
                metric_val = float("nan")
            else:
                corr, _ = spearmanr(pred, actual)
                metric_val = float(corr)

        group_metrics.append(
            GroupMetric(
                group_value=str(g),
                metric_value=metric_val,
                n_policies=n,
                total_exposure=float(g_exp.sum()),
            )
        )

    valid_vals = [m.metric_value for m in group_metrics if not np.isnan(m.metric_value)]
    max_disp = max(valid_vals) - min(valid_vals) if len(valid_vals) >= 2 else 0.0

    rag = rag_status("calibration_disparity", max_disp)

    return EqualisedOddsResult(
        protected_col=protected_col,
        group_metrics=group_metrics,
        max_tpr_disparity=max_disp,
        rag=rag,
    )


# ---------------------------------------------------------------------------
# Gini coefficient by group
# ---------------------------------------------------------------------------


def gini_by_group(
    df: pl.DataFrame,
    protected_col: str,
    prediction_col: str,
    exposure_col: str | None = None,
) -> GiniResult:
    """
    Compute the Gini coefficient of the premium distribution within each group.

    Identical within-group Gini coefficients suggest the model ranks risk
    similarly across groups. A group with a lower Gini than others has less
    premium spread - the model is less discriminating within that group, which
    may indicate it is not capturing risk heterogeneity within the group.

    The Gini coefficient is computed exposure-weighted: policies are sorted by
    predicted premium, and cumulative exposure is used in place of count.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    prediction_col:
        Prediction column (predicted premium or loss rate).
    exposure_col:
        Exposure column.

    Returns
    -------
    GiniResult
    """
    validate_columns(df, protected_col, prediction_col)
    exposure = resolve_exposure(df, exposure_col)

    def _weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
        sorted_idx = np.argsort(values)
        sorted_vals = values[sorted_idx]
        sorted_wts = weights[sorted_idx]
        cum_wts = np.cumsum(sorted_wts)
        total_wt = cum_wts[-1]
        cum_vals = np.cumsum(sorted_vals * sorted_wts)
        total_val = cum_vals[-1]
        if total_val == 0:
            return 0.0
        # Trapezoidal approximation of the area under Lorenz curve.
        # Prepend (0, 0) to the Lorenz curve so the integration starts from the origin.
        lorenz_x = np.concatenate([[0.0], cum_wts / total_wt])
        lorenz_y = np.concatenate([[0.0], cum_vals / total_val])
        area_under = float(np.trapz(lorenz_y, lorenz_x))
        return 1.0 - 2.0 * area_under

    groups = df[protected_col].unique().sort().to_list()
    group_ginis: dict[str, float] = {}

    all_pred = df[prediction_col].to_numpy()
    all_exp = exposure.to_numpy()
    overall_gini = _weighted_gini(all_pred, all_exp)

    for g in groups:
        mask = (df[protected_col] == g).to_numpy()
        g_pred = all_pred[mask]
        g_exp = all_exp[mask]
        group_ginis[str(g)] = _weighted_gini(g_pred, g_exp)

    valid_ginis = [v for v in group_ginis.values() if not np.isnan(v)]
    max_disp = max(valid_ginis) - min(valid_ginis) if len(valid_ginis) >= 2 else 0.0

    return GiniResult(
        protected_col=protected_col,
        group_ginis=group_ginis,
        overall_gini=overall_gini,
        max_disparity=max_disp,
    )


# ---------------------------------------------------------------------------
# Theil index
# ---------------------------------------------------------------------------


def theil_index(
    df: pl.DataFrame,
    protected_col: str,
    prediction_col: str,
    exposure_col: str | None = None,
) -> TheilResult:
    """
    Compute the Theil T index of premium inequality, decomposed by group.

    The Theil index decomposes as:

        T_total = T_between + T_within

    T_between captures inequality between groups (systematic inter-group
    premium differences). T_within captures inequality within groups (risk
    heterogeneity within each group).

    A high T_between relative to T_total is a flag for potential proxy
    discrimination: the model assigns premiums that differ systematically
    between protected-characteristic groups beyond what within-group risk
    heterogeneity would predict.

    Parameters
    ----------
    df:
        Policy-level Polars DataFrame.
    protected_col:
        Protected characteristic column.
    prediction_col:
        Prediction column (predicted premium or loss rate).
    exposure_col:
        Exposure column.

    Returns
    -------
    TheilResult
    """
    validate_columns(df, protected_col, prediction_col)
    exposure = resolve_exposure(df, exposure_col)

    pred = df[prediction_col].to_numpy()
    exp = exposure.to_numpy()

    # Check all predictions are strictly positive
    if np.any(pred <= 0):
        raise ValueError(
            "Theil index requires strictly positive predictions. "
            f"Minimum prediction found: {pred.min():.6f}."
        )

    # Exposure-weighted overall mean
    overall_mean = float(np.average(pred, weights=exp))

    # Overall Theil T: exposure-weighted
    ratios = pred / overall_mean
    # Guard against log(0) - replace zeros with a tiny positive number
    ratios = np.where(ratios <= 0, 1e-10, ratios)
    theil_total = float(np.average(ratios * np.log(ratios), weights=exp))

    groups = df[protected_col].unique().sort().to_list()
    group_contributions: dict[str, float] = {}

    theil_within = 0.0
    theil_between = 0.0

    total_exp = float(exp.sum())

    for g in groups:
        mask = (df[protected_col] == g).to_numpy()
        g_pred = pred[mask]
        g_exp = exp[mask]
        g_total_exp = float(g_exp.sum())
        g_share = g_total_exp / total_exp

        g_mean = float(np.average(g_pred, weights=g_exp))
        if g_mean <= 0:
            group_contributions[str(g)] = float("nan")
            continue

        # Within-group Theil contribution
        g_ratios = g_pred / g_mean
        g_ratios = np.where(g_ratios <= 0, 1e-10, g_ratios)
        g_theil = float(np.average(g_ratios * np.log(g_ratios), weights=g_exp))
        theil_within += g_share * g_theil

        # Between-group contribution: (g_mean / overall_mean) * log(g_mean / overall_mean)
        inter_ratio = g_mean / overall_mean
        theil_between += g_share * inter_ratio * np.log(inter_ratio)

        group_contributions[str(g)] = float(g_share * g_theil)

    return TheilResult(
        protected_col=protected_col,
        theil_total=theil_total,
        theil_within=float(theil_within),
        theil_between=float(theil_between),
        group_contributions=group_contributions,
    )
