"""
multicalibration.py
-------------------
MulticalibrationAudit: calibration-aware fairness auditing for insurance pricing.

Multicalibration unifies actuarial calibration with group fairness. A model is
multicalibrated if its predictions are unbiased conditional on both (a) the
predicted value and (b) any protected group membership:

    E[Y | mu(X) = p, A = a] = p   for all bins p and groups a

This extends autocalibration (E[Y | mu(X) = p] = p) from the overall portfolio
to every protected subgroup. When a model fails this check in a specific bin for
a specific group, the group is being systematically overcharged or undercharged
at that premium level — which is both actuarially unsound and potentially
discriminatory.

The implementation follows Denuit, Michaelides & Trufin (2026), arXiv:2603.16317,
adapted for UK insurance pricing conventions (exposure weighting, GLM A/E ratios,
minimum credibility thresholds).

Correction mechanism
--------------------
When a cell (bin, group) fails the calibration test, we apply a credibility-
weighted A/E correction. Full credibility at ``min_credible`` observations; below
that, blend towards the portfolio-level ratio (1.0):

    corrected = predicted * (z * AE + (1 - z) * 1.0)
    where z = min(n / min_credible, 1.0)

This is a conservative approach: small cells get little correction even if the
A/E ratio is large, which prevents overfitting on noise.

References
----------
Denuit, M., Michaelides, M. & Trufin, J. (2026). Multicalibration in Insurance
Pricing. arXiv:2603.16317.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import polars as pl
from scipy import stats


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MulticalibrationReport:
    """
    Results of a multicalibration audit.

    Attributes
    ----------
    overall_calibration_pvalue :
        p-value for the overall autocalibration test (chi-squared across all
        bins, ignoring group membership).
    group_calibration :
        Dict mapping group name (string) to p-value for that group's
        within-bin calibration test.
    bin_group_table :
        Polars DataFrame with one row per (bin, group) cell. Columns:
        bin, group, n_obs, exposure, observed, expected, ae_ratio, pvalue,
        significant, small_cell.
    worst_cells :
        Top 10 cells sorted by |AE - 1|, descending.
    is_multicalibrated :
        True if every cell with sufficient observations passes at alpha.
    alpha :
        Significance level used.
    n_bins :
        Number of quantile bins used.
    """

    overall_calibration_pvalue: float
    group_calibration: dict[str, float]
    bin_group_table: pl.DataFrame
    worst_cells: pl.DataFrame
    is_multicalibrated: bool
    alpha: float
    n_bins: int


# ---------------------------------------------------------------------------
# MulticalibrationAudit
# ---------------------------------------------------------------------------


class MulticalibrationAudit:
    """
    Audit and correct pricing models for multicalibration fairness.

    Multicalibration ensures premium predictions are calibrated not just
    overall, but within every protected subgroup:

        E[Y | mu(X) = p, A = a] = p   for all bins p and groups a

    This unifies actuarial accuracy (autocalibration) with group fairness.

    The audit bins predictions into quantile bands, then for each (bin, group)
    cell computes the observed-to-expected (A/E) ratio and tests whether it
    is statistically distinguishable from 1.0.

    Reference: Denuit, Michaelides & Trufin (2026) arXiv:2603.16317

    Parameters
    ----------
    n_bins :
        Number of quantile bins for calibration assessment. Each bin contains
        roughly equal exposure (or equal observations if no exposure is given).
        Default 10.
    alpha :
        Significance level for calibration tests. Cells with p < alpha are
        flagged as miscalibrated. Default 0.05.
    min_bin_size :
        Minimum observations per bin-group cell. Cells smaller than this are
        flagged as ``small_cell=True`` and excluded from the ``is_multicalibrated``
        verdict, though they are still included in the table. Default 30.
    min_credible :
        Observation count for full credibility in the correction step.
        Below this, bias corrections are blended towards zero (no change).
        Default 1000.
    """

    def __init__(
        self,
        n_bins: int = 10,
        alpha: float = 0.05,
        min_bin_size: int = 30,
        min_credible: int = 1000,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if min_bin_size < 1:
            raise ValueError("min_bin_size must be at least 1.")
        if min_credible < 1:
            raise ValueError("min_credible must be at least 1.")

        self.n_bins = n_bins
        self.alpha = alpha
        self.min_bin_size = min_bin_size
        self.min_credible = min_credible

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> MulticalibrationReport:
        """
        Audit a set of predictions for multicalibration.

        Parameters
        ----------
        y_true :
            Observed outcomes (claims, losses, etc.). Shape (n,).
        y_pred :
            Model predictions (pure premiums, expected losses). Shape (n,).
            Must be non-negative.
        protected :
            Protected group labels. Shape (n,). Any hashable type — strings,
            ints, booleans all work. The audit is run for every unique value.
        exposure :
            Exposure weights. Shape (n,). If None, all observations are
            weighted equally (i.e., exposure = 1.0 for all).

        Returns
        -------
        MulticalibrationReport
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        protected = np.asarray(protected)

        n = len(y_true)
        if len(y_pred) != n or len(protected) != n:
            raise ValueError("y_true, y_pred, and protected must have the same length.")
        if np.any(y_pred < 0):
            raise ValueError("y_pred must be non-negative.")

        if exposure is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(exposure, dtype=float)
            if len(w) != n:
                raise ValueError("exposure must have the same length as y_true.")
            if np.any(w < 0):
                raise ValueError("exposure must be non-negative.")

        # Assign quantile bins based on y_pred (exposure-weighted quantiles)
        bin_ids = self._assign_bins(y_pred, w)

        groups = np.unique(protected)

        # Build the (bin, group) table
        rows: list[dict[str, Any]] = []

        for b in range(self.n_bins):
            mask_bin = bin_ids == b
            for g in groups:
                mask_cell = mask_bin & (protected == g)
                n_obs = int(mask_cell.sum())
                exp_cell = float(w[mask_cell].sum())
                obs_cell = float((y_true[mask_cell] * w[mask_cell]).sum())
                pred_cell = float((y_pred[mask_cell] * w[mask_cell]).sum())

                small_cell = n_obs < self.min_bin_size

                if pred_cell <= 0 or n_obs == 0:
                    ae_ratio = float("nan")
                    pvalue = float("nan")
                    significant = False
                else:
                    ae_ratio = obs_cell / pred_cell
                    # z-test: (observed - expected) / sqrt(expected)
                    # Under Poisson-like assumptions, variance of sum(Y*w) ~ sum(mu*w)
                    # This is a one-sample z-test on the AE ratio.
                    z_stat = (obs_cell - pred_cell) / np.sqrt(max(pred_cell, 1e-12))
                    pvalue = float(2 * stats.norm.sf(abs(z_stat)))
                    significant = (not small_cell) and (pvalue < self.alpha)

                rows.append({
                    "bin": b,
                    "group": str(g),
                    "n_obs": n_obs,
                    "exposure": round(exp_cell, 6),
                    "observed": round(obs_cell, 6),
                    "expected": round(pred_cell, 6),
                    "ae_ratio": round(ae_ratio, 6) if not np.isnan(ae_ratio) else None,
                    "pvalue": round(pvalue, 6) if not np.isnan(pvalue) else None,
                    "significant": significant,
                    "small_cell": small_cell,
                })

        table = pl.DataFrame(rows, schema={
            "bin": pl.Int32,
            "group": pl.String,
            "n_obs": pl.Int32,
            "exposure": pl.Float64,
            "observed": pl.Float64,
            "expected": pl.Float64,
            "ae_ratio": pl.Float64,
            "pvalue": pl.Float64,
            "significant": pl.Boolean,
            "small_cell": pl.Boolean,
        })

        # Overall autocalibration: aggregate across all bins (ignoring group)
        overall_pvalue = self._overall_calibration_pvalue(y_true, y_pred, w, bin_ids)

        # Per-group calibration: for each group, aggregate across bins
        group_calibration: dict[str, float] = {}
        for g in groups:
            mask_g = protected == g
            gp = self._overall_calibration_pvalue(
                y_true[mask_g], y_pred[mask_g], w[mask_g],
                bin_ids[mask_g],
            )
            group_calibration[str(g)] = gp

        # is_multicalibrated: no significant cells (among non-small cells)
        testable = table.filter(pl.col("small_cell").not_())
        is_mc = not testable.filter(pl.col("significant")).height > 0

        # worst_cells: top 10 by |AE - 1|
        worst = (
            table
            .filter(pl.col("ae_ratio").is_not_null())
            .with_columns(
                (pl.col("ae_ratio") - 1.0).abs().alias("abs_deviation")
            )
            .sort("abs_deviation", descending=True)
            .head(10)
            .drop("abs_deviation")
        )

        return MulticalibrationReport(
            overall_calibration_pvalue=overall_pvalue,
            group_calibration=group_calibration,
            bin_group_table=table,
            worst_cells=worst,
            is_multicalibrated=is_mc,
            alpha=self.alpha,
            n_bins=self.n_bins,
        )

    def correct(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray,
        report: MulticalibrationReport,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply credibility-weighted bias correction to predictions.

        For each (bin, group) cell that was flagged as significant in the
        audit report, the prediction is scaled by a credibility-blended A/E
        factor:

            corrected = predicted * (z * AE + (1 - z) * 1.0)
            z = min(n_obs / min_credible, 1.0)

        Cells that passed the calibration test are left unchanged. Cells
        smaller than ``min_bin_size`` are also left unchanged.

        Parameters
        ----------
        y_pred :
            Model predictions to correct. Shape (n,).
        protected :
            Protected group labels (same encoding as passed to ``audit``).
        report :
            The MulticalibrationReport returned by ``audit``.
        exposure :
            Optional exposure weights (used only to recompute bins consistently).

        Returns
        -------
        np.ndarray
            Corrected predictions, same shape as y_pred.
        """
        y_pred = np.asarray(y_pred, dtype=float)
        protected = np.asarray(protected)
        n = len(y_pred)

        if exposure is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(exposure, dtype=float)

        bin_ids = self._assign_bins(y_pred, w)

        # Build lookup from (bin, group) -> correction factor
        correction_lookup: dict[tuple[int, str], float] = {}
        sig_rows = report.bin_group_table.filter(pl.col("significant"))
        for row in sig_rows.iter_rows(named=True):
            b = int(row["bin"])
            g = str(row["group"])
            ae = row["ae_ratio"]
            n_obs = int(row["n_obs"])
            if ae is None or np.isnan(ae):
                continue
            z = min(n_obs / self.min_credible, 1.0)
            factor = z * ae + (1.0 - z) * 1.0
            correction_lookup[(b, g)] = factor

        corrected = y_pred.copy()
        for (b, g), factor in correction_lookup.items():
            mask = (bin_ids == b) & (np.array([str(x) for x in protected]) == g)
            corrected[mask] *= factor

        return corrected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_bins(self, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Assign each observation to a quantile bin based on y_pred.

        Uses exposure-weighted quantile boundaries. Returns an integer array
        of bin indices in [0, n_bins).
        """
        n = len(y_pred)
        # Sort by predicted value
        order = np.argsort(y_pred, kind="stable")
        sorted_w = w[order]

        # Compute cumulative exposure fraction
        cumw = np.cumsum(sorted_w)
        total_w = cumw[-1]
        if total_w <= 0:
            return np.zeros(n, dtype=int)

        frac = cumw / total_w
        # Assign bins by cumulative fraction thresholds
        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_ids_sorted = np.searchsorted(bin_boundaries[1:], frac, side="left")
        bin_ids_sorted = np.clip(bin_ids_sorted, 0, self.n_bins - 1)

        # Reverse the sorting
        bin_ids = np.empty(n, dtype=int)
        bin_ids[order] = bin_ids_sorted
        return bin_ids

    def _overall_calibration_pvalue(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        w: np.ndarray,
        bin_ids: np.ndarray,
    ) -> float:
        """
        Compute a chi-squared p-value for calibration across bins.

        For each bin b: chi^2 contribution = (O_b - E_b)^2 / E_b
        Total statistic is approximately chi-squared with n_bins - 1 df.
        """
        unique_bins = np.unique(bin_ids)
        chi2_stat = 0.0
        df = 0

        for b in unique_bins:
            mask = bin_ids == b
            obs = float((y_true[mask] * w[mask]).sum())
            exp = float((y_pred[mask] * w[mask]).sum())
            if exp <= 0:
                continue
            chi2_stat += (obs - exp) ** 2 / exp
            df += 1

        if df <= 1:
            return 1.0

        pvalue = float(stats.chi2.sf(chi2_stat, df=df - 1))
        return pvalue
