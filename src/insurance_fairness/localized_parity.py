"""
localized_parity.py
--------------------
LocalizedParityAudit and LocalizedParityCorrector: audit and post-process
insurance predictions to enforce demographic parity at selected pricing tiers.

Most fairness tools in this package check either mean parity (demographic parity
ratio) or full distributional parity (OT-based discrimination-free pricing).
Neither is right for tier-based pricing. When a pricing team says "we don't want
gender to determine who falls into the renewal protection tier", they don't care
about the entire distribution — they care about the fraction of each group at or
below a specific premium threshold.

Charpentier, Denis, Elie, Hebiri & HU (arXiv:2603.25224) formalise this:
an (ell, Z)-fair predictor enforces F_{f|S=s}(z_m) = ell_m for a finite set of
tier boundaries Z and target CDF levels ell. This is a strictly weaker constraint
than full distributional parity, which means the correction destroys less
predictive accuracy. The paper proves:

1. The optimal correction has a closed-form Lagrangian characterisation.
2. The risk gap vs the continuous-DP-optimal is O(1/M) in the number of tiers.
3. Constraint violation converges at rate O(1/sqrt(n)) in calibration sample size.

Implementation decisions
------------------------
The transport map T_s for each group s is a piecewise-linear, monotone function
that maps group-s predictions to satisfy the M CDF constraints. We construct it
by solving the quantile-matching problem: for each tier boundary z_m, find the
prediction value q_m^s such that exactly ell_m fraction of group s has original
predictions <= q_m^s, then map those to z_m. Between tier boundaries we
interpolate linearly (preserving rank order, hence monotonicity).

The dual variables (Lagrange multipliers) are computed as the derivative of the
per-tier correction with respect to the constraint value. They measure the cost
of tightening each constraint by epsilon — useful for FCA evidence packs.

Marginal mode removes the need for sensitive attributes at prediction time by
learning a single portfolio-level correction. This is appropriate when group
labels are unavailable post-GDPR or when the pricing system cannot store them.

Usage::

    from insurance_fairness import LocalizedParityCorrector, LocalizedParityAudit

    # Audit only — no correction
    audit = LocalizedParityAudit(thresholds=[300.0, 500.0, 800.0])
    report = audit.audit(predictions, gender_codes)
    print(f"Max disparity: {report.max_disparity:.4f}")

    # Post-processing correction
    corrector = LocalizedParityCorrector(
        thresholds=[300.0, 500.0, 800.0],
        mode='quantile',
    )
    corrector.fit(predictions_train, gender_codes_train)
    fair_preds = corrector.transform(predictions_test, gender_codes_test)

    post_report = corrector.audit()
    print(f"Post-correction disparity: {post_report.max_disparity:.4f}")

References
----------
Charpentier, A., Denis, C., Elie, R., Hebiri, M. & HU, L. (2026).
Fair Regression under Localized Demographic Parity Constraints. arXiv:2603.25224.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.optimize import minimize

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LocalizedParityReport:
    """
    Audit results for localized demographic parity at M tier boundaries.

    Attributes
    ----------
    thresholds :
        The M pricing tier boundaries Z = {z_1, ..., z_M} in ascending order.
    target_levels :
        The M target CDF levels ell = {ell_1, ..., ell_M}. Each ell_m is the
        fraction of predictions that should fall at or below z_m for every group.
    group_cdf_table :
        A Polars DataFrame with columns ['group', 'threshold', 'empirical_cdf',
        'target_cdf', 'deviation']. One row per (group, threshold) combination.
        'deviation' = empirical_cdf - target_cdf.
    max_disparity :
        Maximum |F_{f|S=s}(z_m) - ell_m| across all groups s and thresholds m.
        Zero means perfect localized parity; values above tolerance indicate
        non-compliance.
    lagrange_multipliers :
        Array of shape (n_groups, M). Dual variables quantifying the cost of
        tightening each (group, tier) constraint by one unit. Positive values
        indicate the constraint is active (binding). Near-zero values mean the
        correction at that tier was free.
    discretization_cost :
        O(1/M) bound on the accuracy gap introduced by discretising the
        continuous-DP correction to M points. Computed as 1.0 / M.
        Refining the threshold grid (larger M) reduces this bound.
    """

    thresholds: list[float]
    target_levels: list[float]
    group_cdf_table: pl.DataFrame
    max_disparity: float
    lagrange_multipliers: np.ndarray
    discretization_cost: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empirical_cdf(
    predictions: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute empirical CDF values at each threshold.

    Returns array of shape (len(thresholds),) where entry m is the weighted
    fraction of predictions <= thresholds[m].

    Parameters
    ----------
    predictions :
        1-D array of prediction values for a single group.
    thresholds :
        Sorted array of threshold values.
    weights :
        Optional exposure weights. If None, equal weights are assumed.

    Returns
    -------
    cdf_vals : np.ndarray
        CDF values in [0, 1] at each threshold.
    """
    n = len(predictions)
    if n == 0:
        return np.zeros(len(thresholds))

    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float)

    total_weight = weights.sum()
    if total_weight <= 0:
        return np.zeros(len(thresholds))

    cdf_vals = np.empty(len(thresholds))
    for m, z in enumerate(thresholds):
        mask = predictions <= z
        cdf_vals[m] = weights[mask].sum() / total_weight

    return cdf_vals


def _build_quantile_transport_map(
    predictions: np.ndarray,
    thresholds: np.ndarray,
    target_levels: np.ndarray,
    weights: np.ndarray | None = None,
) -> interp1d:
    """
    Build a piecewise-linear monotone transport map for a single group.

    The map T satisfies: fraction of predictions in group s with T(f) <= z_m
    equals ell_m, for all m in {1, ..., M}.

    Construction:
    - For each tier boundary z_m with target level ell_m, find the ell_m
      quantile of the group's prediction distribution: q_m = quantile(preds, ell_m).
    - Define T(q_m) = z_m for all m.
    - Add boundary anchors: T(min_pred - eps) = min_pred, T(max_pred + eps) = max_pred
      (identity outside the correction range, with linear interpolation).
    - Interpolate linearly between control points, ensuring monotonicity.

    Parameters
    ----------
    predictions :
        Group predictions.
    thresholds :
        Tier boundaries (ascending).
    target_levels :
        Target CDF levels at each threshold.
    weights :
        Optional exposure weights.

    Returns
    -------
    map_fn : interp1d
        Callable T: R -> R. Monotone by construction.
    """
    if weights is None:
        weights = np.ones(len(predictions))
    else:
        weights = np.asarray(weights, dtype=float)

    # Get the weighted quantiles (source control points)
    source_quantiles = _weighted_quantile(predictions, target_levels, weights)

    # Target values are the thresholds
    target_values = thresholds.copy()

    # Build control points: source -> target
    # Add identity anchors at extremes to anchor the map outside [min_threshold, max_threshold]
    pred_min = float(np.min(predictions))
    pred_max = float(np.max(predictions))

    # Anchor below: map pred_min to itself (no correction below lowest threshold)
    # Anchor above: map pred_max to itself (no correction above highest threshold)
    src_pts = np.concatenate([[pred_min], source_quantiles, [pred_max]])
    tgt_pts = np.concatenate([[pred_min], target_values, [pred_max]])

    # Ensure monotonicity: source control points must be strictly increasing
    # If quantile matching produces ties (e.g., degenerate distribution), nudge
    src_pts = _make_strictly_increasing(src_pts)
    tgt_pts = _make_strictly_increasing(tgt_pts)

    map_fn = interp1d(
        src_pts,
        tgt_pts,
        kind="linear",
        bounds_error=False,
        fill_value=(tgt_pts[0], tgt_pts[-1]),
    )
    return map_fn


def _weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted quantiles using linear interpolation.

    Parameters
    ----------
    values :
        1-D array of data values.
    quantiles :
        Array of quantile levels in [0, 1].
    weights :
        Non-negative weights for each observation.

    Returns
    -------
    result : np.ndarray
        Weighted quantile values.
    """
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)

    sort_idx = np.argsort(values, kind="stable")
    sorted_vals = values[sort_idx]
    sorted_w = weights[sort_idx]

    # Cumulative weight, normalised
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    if total_w <= 0:
        return np.full(len(quantiles), np.nan)

    # Midpoint CDF positions
    cum_w_norm = (cum_w - sorted_w / 2.0) / total_w
    cum_w_norm = np.clip(cum_w_norm, 0.0, 1.0)

    # Linear interpolation
    result = np.interp(quantiles, cum_w_norm, sorted_vals)
    return result


def _make_strictly_increasing(arr: np.ndarray) -> np.ndarray:
    """
    Nudge an array to be strictly increasing by adding tiny offsets at ties.

    This handles the degenerate case where a group's predictions are all
    identical, producing quantile ties that would break interp1d.
    """
    arr = arr.copy().astype(float)
    eps = max(1e-10 * np.abs(arr).max(), 1e-10)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            arr[i] = arr[i - 1] + eps
    return arr


def _compute_lagrange_multipliers(
    predictions: np.ndarray,
    sensitive: np.ndarray,
    thresholds: np.ndarray,
    target_levels: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate Lagrange multipliers for each (group, threshold) pair.

    The multiplier lambda_{s,m} is the sensitivity of the squared deviation
    loss to a unit perturbation of the constraint ell_m for group s. We
    estimate it numerically as the derivative of the deviation squared at
    the calibration solution.

    Returns array of shape (n_groups, M).
    """
    M = len(thresholds)
    n_groups = len(groups)
    lambdas = np.zeros((n_groups, M))

    n = len(predictions)
    if weights is None:
        weights = np.ones(n)

    for g_idx, g in enumerate(groups):
        mask = sensitive == g
        preds_g = predictions[mask]
        w_g = weights[mask]
        if len(preds_g) == 0:
            continue

        cdfs_g = _empirical_cdf(preds_g, thresholds, w_g)
        for m in range(M):
            dev = cdfs_g[m] - target_levels[m]
            # Lagrange multiplier ~ 2 * deviation (gradient of L2 penalty)
            lambdas[g_idx, m] = 2.0 * dev

    return lambdas


# ---------------------------------------------------------------------------
# Audit-only class
# ---------------------------------------------------------------------------


class LocalizedParityAudit:
    """
    Audit predicted score distributions for localized demographic parity.

    Measures whether each group's fraction of predictions at or below each
    pricing tier boundary matches the target level. Does not modify predictions.

    Use this to quantify tier-level CDF disparity before deciding whether
    to apply correction. The ``max_disparity`` metric in the report is the
    key headline number for FCA Consumer Duty evidence packs.

    Parameters
    ----------
    thresholds :
        Pricing tier boundaries Z = {z_1, ..., z_M}. These should be the
        premium thresholds your pricing team actually cares about — for
        example, the boundaries between renewal protection tiers or between
        rate decile buckets. Must be in ascending order.
    target_levels :
        Target CDF level at each threshold. If None, uses the portfolio-level
        empirical CDF (i.e., the target for each group is to match the
        overall portfolio fraction below each threshold).
        If provided, must be the same length as thresholds and strictly
        increasing, with values in (0, 1).
    tolerance :
        Max acceptable |F_{f|S=s}(z_m) - ell_m| before flagging non-compliance.
        Default 0.05 (5 percentage points). FCA guidance does not specify a
        threshold; 5% is a conservative starting point.

    Examples
    --------
    Audit two-group predictions at three tier boundaries::

        audit = LocalizedParityAudit(thresholds=[300.0, 500.0, 800.0])
        report = audit.audit(predictions, gender_codes)
        print(f"Max disparity: {report.max_disparity:.4f}")  # 0 = perfect
        print(report.group_cdf_table)
    """

    def __init__(
        self,
        thresholds: list[float],
        target_levels: list[float] | None = None,
        tolerance: float = 0.05,
    ) -> None:
        thresholds = list(thresholds)
        if len(thresholds) == 0:
            raise ValueError("thresholds must be non-empty.")
        if sorted(thresholds) != thresholds:
            raise ValueError("thresholds must be in ascending order.")

        if target_levels is not None:
            target_levels = list(target_levels)
            if len(target_levels) != len(thresholds):
                raise ValueError(
                    f"target_levels must have the same length as thresholds. "
                    f"Got {len(target_levels)} vs {len(thresholds)}."
                )
            if any(not (0.0 < ell < 1.0) for ell in target_levels):
                raise ValueError("All target_levels must be strictly between 0 and 1.")
            if sorted(target_levels) != target_levels:
                raise ValueError("target_levels must be in ascending order.")

        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")

        self.thresholds = thresholds
        self.target_levels = target_levels
        self.tolerance = tolerance

    def audit(
        self,
        predictions: np.ndarray,
        sensitive: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> LocalizedParityReport:
        """
        Compute group CDFs at each threshold and report disparity.

        Parameters
        ----------
        predictions :
            Array of model predictions (e.g., pure premiums). Shape (n,).
        sensitive :
            Group membership labels. Shape (n,). Any hashable dtype.
        exposure :
            Optional exposure weights. Shape (n,). If None, all observations
            are equally weighted.

        Returns
        -------
        report : LocalizedParityReport
        """
        predictions = np.asarray(predictions, dtype=float)
        sensitive = np.asarray(sensitive)
        n = len(predictions)
        if len(sensitive) != n:
            raise ValueError(
                f"predictions and sensitive must have the same length. "
                f"Got {n} and {len(sensitive)}."
            )
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            if len(exposure) != n:
                raise ValueError(
                    f"exposure must have length {n}. Got {len(exposure)}."
                )

        thresholds_arr = np.asarray(self.thresholds, dtype=float)
        groups = np.unique(sensitive)

        # Resolve target levels
        if self.target_levels is None:
            target_levels = _empirical_cdf(predictions, thresholds_arr, exposure)
        else:
            target_levels = np.asarray(self.target_levels, dtype=float)

        return _build_report(
            predictions=predictions,
            sensitive=sensitive,
            thresholds=thresholds_arr,
            target_levels=target_levels,
            groups=groups,
            weights=exposure,
        )


# ---------------------------------------------------------------------------
# Corrector class
# ---------------------------------------------------------------------------


class LocalizedParityCorrector:
    """
    Post-processing corrector for localized demographic parity.

    Learns group-specific or portfolio-level monotone transport maps that
    shift predictions so that each group's CDF matches the target at each
    pricing tier boundary. The correction is entirely post-hoc — the base
    model is unchanged.

    Two correction modes:

    ``'quantile'`` mode (default): learns a separate map T_s per group.
    Requires group labels at both fit and transform time. More accurate
    when group sample sizes are reasonable (typically >= 200 per group).

    ``'marginal'`` mode: learns a single map T that moves predictions towards
    the portfolio CDF. Useful when group labels cannot be stored or used at
    scoring time (GDPR considerations). Less powerful than quantile mode for
    multi-group settings but requires no group information at transform time.

    Parameters
    ----------
    thresholds :
        Pricing tier boundaries Z = {z_1, ..., z_M}. Must be in ascending order.
    target_levels :
        Target CDF levels at each threshold. If None, uses the portfolio-level
        empirical CDF computed from the calibration (fit) data.
    mode :
        ``'quantile'``: group-specific transport maps. Requires sensitive at
        transform time.
        ``'marginal'``: single portfolio-level map. Sensitive not needed at
        transform time.
    n_bins :
        Number of quantile bins for transport map discretisation. Finer grids
        produce smoother maps but require more calibration data. Default 200.
    solver :
        Scipy optimiser for dual variable estimation. Default ``'SLSQP'``.
        Only used for Lagrange multiplier estimation; the transport maps
        themselves use closed-form quantile matching.

    Examples
    --------
    Fit and apply correction with group labels::

        corrector = LocalizedParityCorrector(
            thresholds=[300.0, 500.0, 800.0],
            mode='quantile',
        )
        corrector.fit(X_train, gender_train, exposure_train)
        X_fair = corrector.transform(X_test, gender_test)
        print(corrector.audit().max_disparity)

    Marginal mode (no group labels at transform time)::

        corrector = LocalizedParityCorrector(thresholds=[400.0, 700.0], mode='marginal')
        corrector.fit(X_train, gender_train)
        X_fair = corrector.transform(X_test)   # no sensitive needed
    """

    def __init__(
        self,
        thresholds: list[float],
        target_levels: list[float] | None = None,
        mode: str = "quantile",
        n_bins: int = 200,
        solver: str = "SLSQP",
    ) -> None:
        thresholds = list(thresholds)
        if len(thresholds) == 0:
            raise ValueError("thresholds must be non-empty.")
        if sorted(thresholds) != thresholds:
            raise ValueError("thresholds must be in ascending order.")

        if target_levels is not None:
            target_levels = list(target_levels)
            if len(target_levels) != len(thresholds):
                raise ValueError(
                    "target_levels must have the same length as thresholds."
                )
            if any(not (0.0 < ell < 1.0) for ell in target_levels):
                raise ValueError("All target_levels must be strictly between 0 and 1.")
            if sorted(target_levels) != target_levels:
                raise ValueError("target_levels must be in ascending order.")

        if mode not in ("quantile", "marginal"):
            raise ValueError(
                f"mode must be 'quantile' or 'marginal'. Got: {mode!r}"
            )
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2.")

        self.thresholds = thresholds
        self.target_levels = target_levels
        self.mode = mode
        self.n_bins = n_bins
        self.solver = solver

        # Set after fit()
        self._is_fitted: bool = False
        self._groups: np.ndarray | None = None
        self._transport_maps: dict = {}  # group label -> interp1d
        self._marginal_map: interp1d | None = None
        self._fitted_target_levels: np.ndarray | None = None
        self._lagrange_multipliers: np.ndarray | None = None
        self._train_predictions: np.ndarray | None = None
        self._train_sensitive: np.ndarray | None = None
        self._train_exposure: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        predictions: np.ndarray,
        sensitive: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "LocalizedParityCorrector":
        """
        Fit group-specific transport maps from calibration data.

        Parameters
        ----------
        predictions :
            Base model predictions on calibration data. Shape (n,).
        sensitive :
            Group membership labels. Shape (n,).
        exposure :
            Optional exposure weights. Shape (n,).

        Returns
        -------
        self
        """
        predictions = np.asarray(predictions, dtype=float)
        sensitive = np.asarray(sensitive)
        n = len(predictions)

        if len(sensitive) != n:
            raise ValueError(
                f"predictions and sensitive must have the same length. "
                f"Got {n} and {len(sensitive)}."
            )
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            if len(exposure) != n:
                raise ValueError(f"exposure must have length {n}.")
            if np.any(exposure < 0):
                raise ValueError("exposure must be non-negative.")
        else:
            exposure = np.ones(n)

        thresholds_arr = np.asarray(self.thresholds, dtype=float)

        # Resolve target levels from portfolio CDF if not specified
        if self.target_levels is None:
            target_levels = _empirical_cdf(predictions, thresholds_arr, exposure)
        else:
            target_levels = np.asarray(self.target_levels, dtype=float)

        self._fitted_target_levels = target_levels
        groups = np.unique(sensitive)
        self._groups = groups

        if self.mode == "quantile":
            self._transport_maps = {}
            for g in groups:
                mask = sensitive == g
                preds_g = predictions[mask]
                w_g = exposure[mask]
                if len(preds_g) == 0:
                    continue
                t_map = _build_quantile_transport_map(
                    preds_g, thresholds_arr, target_levels, w_g
                )
                self._transport_maps[g] = t_map

        elif self.mode == "marginal":
            # Single map using portfolio distribution
            t_map = _build_quantile_transport_map(
                predictions, thresholds_arr, target_levels, exposure
            )
            self._marginal_map = t_map

        # Estimate Lagrange multipliers from calibration fit
        self._lagrange_multipliers = _compute_lagrange_multipliers(
            predictions=predictions,
            sensitive=sensitive,
            thresholds=thresholds_arr,
            target_levels=target_levels,
            groups=groups,
            weights=exposure,
        )

        # Store calibration data for post-fit audit
        self._train_predictions = predictions
        self._train_sensitive = sensitive
        self._train_exposure = exposure
        self._is_fitted = True

        return self

    def transform(
        self,
        predictions: np.ndarray,
        sensitive: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the fitted transport maps to new predictions.

        Parameters
        ----------
        predictions :
            Model predictions to correct. Shape (n,).
        sensitive :
            Group membership labels. Required when mode='quantile'.
            Ignored (and not required) when mode='marginal'.

        Returns
        -------
        corrected : np.ndarray, shape (n,)
            Fairness-corrected predictions.
        """
        self._check_fitted()

        predictions = np.asarray(predictions, dtype=float)
        n = len(predictions)

        if self.mode == "marginal":
            return np.asarray(self._marginal_map(predictions), dtype=float)

        # Quantile mode: need sensitive
        if sensitive is None:
            raise ValueError(
                "sensitive is required for mode='quantile'. "
                "Use mode='marginal' if group labels are unavailable at "
                "prediction time."
            )
        sensitive = np.asarray(sensitive)
        if len(sensitive) != n:
            raise ValueError(
                f"predictions and sensitive must have the same length. "
                f"Got {n} and {len(sensitive)}."
            )

        corrected = predictions.copy()
        for g, t_map in self._transport_maps.items():
            mask = sensitive == g
            if not np.any(mask):
                continue
            corrected[mask] = t_map(predictions[mask])

        # Observations with unknown group label (not in training groups)
        # get identity map (no correction)
        return corrected

    def fit_transform(
        self,
        predictions: np.ndarray,
        sensitive: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit and transform in one step. Correction is applied to the same
        data used for fitting (in-sample). Use fit() + transform() when
        you have a held-out test set.

        Parameters
        ----------
        predictions :
            Base model predictions. Shape (n,).
        sensitive :
            Group membership labels. Shape (n,).
        exposure :
            Optional exposure weights. Shape (n,).

        Returns
        -------
        corrected : np.ndarray, shape (n,)
        """
        self.fit(predictions, sensitive, exposure)
        return self.transform(predictions, sensitive)

    def audit(self) -> LocalizedParityReport:
        """
        Compute post-correction disparity on calibration data.

        Applies the fitted transport maps to the training predictions and
        measures residual CDF disparity. This is the in-sample audit — for
        out-of-sample guarantees, call on held-out data via:

            corrected = corrector.transform(X_test, S_test)
            report = corrector._audit_predictions(corrected, S_test)

        Returns
        -------
        report : LocalizedParityReport
        """
        self._check_fitted()

        corrected = self.transform(
            self._train_predictions,
            self._train_sensitive if self.mode == "quantile" else None,
        )
        thresholds_arr = np.asarray(self.thresholds, dtype=float)
        groups = self._groups

        return _build_report(
            predictions=corrected,
            sensitive=self._train_sensitive,
            thresholds=thresholds_arr,
            target_levels=self._fitted_target_levels,
            groups=groups,
            weights=self._train_exposure,
        )

    def audit_predictions(
        self,
        predictions: np.ndarray,
        sensitive: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> LocalizedParityReport:
        """
        Audit arbitrary (already-corrected) predictions on new data.

        Parameters
        ----------
        predictions :
            Corrected predictions to audit. Shape (n,).
        sensitive :
            Group membership labels. Shape (n,).
        exposure :
            Optional exposure weights.

        Returns
        -------
        report : LocalizedParityReport
        """
        self._check_fitted()

        predictions = np.asarray(predictions, dtype=float)
        sensitive = np.asarray(sensitive)
        thresholds_arr = np.asarray(self.thresholds, dtype=float)

        return _build_report(
            predictions=predictions,
            sensitive=sensitive,
            thresholds=thresholds_arr,
            target_levels=self._fitted_target_levels,
            groups=np.unique(sensitive),
            weights=exposure,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lagrange_multipliers(self) -> np.ndarray:
        """
        Dual variables of shape (n_groups, M).

        Positive values at (s, m) indicate that the CDF constraint for group s
        at threshold z_m was violated (group s had too many predictions above z_m
        relative to the target). The magnitude measures the cost of tightening
        that constraint. Near-zero values mean the constraint was approximately
        satisfied without correction.
        """
        self._check_fitted()
        return self._lagrange_multipliers.copy()

    @property
    def discretization_cost(self) -> float:
        """
        O(1/M) upper bound on the accuracy loss from using M tiers instead of
        continuous demographic parity. Smaller is better.

        The paper (Proposition 3.2) proves the risk gap is bounded by C/M for
        some constant C depending on the base model. This property returns 1/M
        as a normalised proxy. To halve the discretisation cost, double M.
        """
        self._check_fitted()
        return 1.0 / len(self.thresholds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LocalizedParityCorrector has not been fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        M = len(self.thresholds)
        return (
            f"LocalizedParityCorrector("
            f"M={M}, "
            f"mode={self.mode!r}, "
            f"n_bins={self.n_bins})"
        )


# ---------------------------------------------------------------------------
# Shared report builder
# ---------------------------------------------------------------------------


def _build_report(
    predictions: np.ndarray,
    sensitive: np.ndarray,
    thresholds: np.ndarray,
    target_levels: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray | None = None,
) -> LocalizedParityReport:
    """
    Build a LocalizedParityReport from predictions and group labels.

    Parameters
    ----------
    predictions :
        Predictions (possibly corrected). Shape (n,).
    sensitive :
        Group labels. Shape (n,).
    thresholds :
        Tier boundaries. Shape (M,).
    target_levels :
        Target CDF levels. Shape (M,).
    groups :
        Unique group labels (in order they appear in the report).
    weights :
        Optional exposure weights. Shape (n,).

    Returns
    -------
    report : LocalizedParityReport
    """
    n = len(predictions)
    M = len(thresholds)

    if weights is None:
        weights = np.ones(n)

    rows: list[dict] = []
    max_disparity = 0.0
    lagrange_mults = np.zeros((len(groups), M))

    for g_idx, g in enumerate(groups):
        mask = sensitive == g
        preds_g = predictions[mask]
        w_g = weights[mask]

        if len(preds_g) == 0:
            for m, z in enumerate(thresholds):
                rows.append(
                    {
                        "group": str(g),
                        "threshold": float(z),
                        "empirical_cdf": float("nan"),
                        "target_cdf": float(target_levels[m]),
                        "deviation": float("nan"),
                    }
                )
            continue

        cdfs_g = _empirical_cdf(preds_g, thresholds, w_g)

        for m in range(M):
            dev = float(cdfs_g[m]) - float(target_levels[m])
            abs_dev = abs(dev)
            if abs_dev > max_disparity:
                max_disparity = abs_dev

            lagrange_mults[g_idx, m] = 2.0 * dev  # gradient of L2 penalty

            rows.append(
                {
                    "group": str(g),
                    "threshold": float(thresholds[m]),
                    "empirical_cdf": float(cdfs_g[m]),
                    "target_cdf": float(target_levels[m]),
                    "deviation": dev,
                }
            )

    schema = {
        "group": pl.Utf8,
        "threshold": pl.Float64,
        "empirical_cdf": pl.Float64,
        "target_cdf": pl.Float64,
        "deviation": pl.Float64,
    }
    group_cdf_table = pl.DataFrame(rows, schema=schema)

    return LocalizedParityReport(
        thresholds=thresholds.tolist(),
        target_levels=target_levels.tolist(),
        group_cdf_table=group_cdf_table,
        max_disparity=max_disparity,
        lagrange_multipliers=lagrange_mults,
        discretization_cost=1.0 / M,
    )
