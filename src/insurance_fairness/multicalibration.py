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
weighted A/E correction. The paper's formulation (eq. 4.3) shrinks toward the
bin-level pooled A/E ratio b_hat_k (ignoring group stratification), NOT toward
1.0. This preserves marginal autocalibration even when group-level corrections
are partially shrunk:

    b_tilde_kl = z_kl * b_hat_kl + (1 - z_kl) * b_hat_k
    where z_kl = min(n_kl / c, 1.0)

    corrected = predicted * b_tilde_kl

Full credibility at c observations; below that, blend towards b_hat_k (the
pooled bin mean). b_hat_k is stored in MulticalibrationReport.bin_level_ae so
downstream users can inspect it.

For iterative correction converging to full multicalibration, see
:class:`IterativeMulticalibrationCorrector`.

For a one-shot, non-iterative isotonic regression approach, see
:class:`IsotonicMulticalibrationCorrector`.

For iterative bias correction with a continuous sensitive feature using
local GLM (Algorithm C), see :class:`LocalGLMMulticalibrationCorrector`.

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
from sklearn.isotonic import IsotonicRegression


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
    bin_level_ae :
        Dict mapping bin index (int) to the pooled A/E ratio for that bin,
        aggregated across all groups (b_hat_k in Denuit et al. notation).
        Used as the credibility shrinkage target in correct().
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
    bin_level_ae: dict[int, float]
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

    The correction step follows Denuit et al. (2026) eq. 4.3: credibility
    blending uses the bin-level pooled A/E ratio (b_hat_k) as the shrinkage
    target, not 1.0. This preserves marginal autocalibration for small groups.

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
        Below this, bias corrections are blended towards the bin-level pooled
        A/E ratio (b_hat_k). Default 1000.
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

        # Compute bin-level pooled A/E ratios (b_hat_k): aggregate across all groups
        bin_level_ae: dict[int, float] = {}
        for b in range(self.n_bins):
            mask_bin = bin_ids == b
            obs_bin = float((y_true[mask_bin] * w[mask_bin]).sum())
            pred_bin = float((y_pred[mask_bin] * w[mask_bin]).sum())
            if pred_bin > 0:
                bin_level_ae[b] = obs_bin / pred_bin
            else:
                bin_level_ae[b] = 1.0  # fallback for empty bins

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
            bin_level_ae=bin_level_ae,
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
        factor following Denuit et al. (2026) eq. 4.3:

            b_tilde_kl = z_kl * b_hat_kl + (1 - z_kl) * b_hat_k
            z_kl = min(n_obs / min_credible, 1.0)
            corrected = predicted * b_tilde_kl

        The shrinkage target is b_hat_k (the bin-level pooled A/E ratio,
        ignoring group), NOT 1.0. This preserves marginal autocalibration
        even when group corrections are shrunk due to low credibility.

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
        # Shrinkage target is b_hat_k (bin-level pooled AE), not 1.0
        correction_lookup: dict[tuple[int, str], float] = {}
        sig_rows = report.bin_group_table.filter(pl.col("significant"))
        for row in sig_rows.iter_rows(named=True):
            b = int(row["bin"])
            g = str(row["group"])
            ae = row["ae_ratio"]
            n_obs = int(row["n_obs"])
            if ae is None or np.isnan(ae):
                continue
            # b_hat_k: pooled bin-level AE from the report (fallback to 1.0)
            b_hat_k = report.bin_level_ae.get(b, 1.0)
            z = min(n_obs / self.min_credible, 1.0)
            # Denuit et al. (2026) eq. 4.3: blend toward bin mean, not 1.0
            factor = z * ae + (1.0 - z) * b_hat_k
            correction_lookup[(b, g)] = factor

        corrected = y_pred.copy()
        protected_str = np.array([str(x) for x in protected])
        for (b, g), factor in correction_lookup.items():
            mask = (bin_ids == b) & (protected_str == g)
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


# ---------------------------------------------------------------------------
# IterativeMulticalibrationCorrector
# ---------------------------------------------------------------------------


class IterativeMulticalibrationCorrector:
    """
    Iterative multicalibration corrector following Denuit et al. (2026).

    Applies repeated additive corrections on the response scale until the
    maximum scaled correction across all (bin, group) cells drops below a
    convergence threshold. This is the iterative version of the one-shot
    credibility correction in MulticalibrationAudit.correct().

    The algorithm at each iteration j:

        1. Compute b_hat_kl^(j) = O_kl / E_kl^(j)  (cell-level A/E)
        2. Compute b_hat_k^(j) = O_k / E_k^(j)      (bin-level pooled A/E)
        3. Credibility blend: b_tilde_kl = z_kl * b_hat_kl + (1 - z_kl) * b_hat_k
           where z_kl = min(n_kl / c, 1.0)
        4. Additive update: pi^(j+1) = pi^(j) + eta * (b_tilde_kl - 1) * pi^(j)
        5. Stop if max_{k,l} eta * |b_tilde_kl - 1| < delta  (Section 7.1.2)

    Stopping criterion (Section 7.1.2)
    ------------------------------------
    The paper's stopping criterion is scale-invariant:

        max_{k,l} |eta * b_tilde_kl| / pi_bar_kl <= delta

    where pi_bar_kl is the exposure-weighted mean prediction in cell (k,l).
    Since the correction applied to pi is eta * (b_tilde_kl - 1) * pi, dividing
    by pi_bar_kl gives eta * |b_tilde_kl - 1|. This normalisation means the
    threshold is in relative terms regardless of the premium level in each cell.

    The practical effect: the criterion uses the BLENDED factor b_tilde_kl
    (not the raw A/E ratio b_hat_kl) and applies the learning rate eta, making
    small corrections cheap even in high-premium cells where absolute deviations
    are large.

    The "additive on response scale" phrasing from the paper means we update
    the predicted premium value, not a log-scale correction. The update
    formula pi += eta * (factor - 1) * pi is equivalent to a multiplicative
    step pi *= 1 + eta * (factor - 1), which reduces to full multiplicative
    correction (pi *= factor) when eta=1.

    fit() learns the correction surface on training data; transform() applies
    it to new observations by bin+group lookup.

    Parameters
    ----------
    n_bins :
        Number of quantile bins. Default 10.
    eta :
        Learning rate for the additive update step. Must be in (0, 1].
        Smaller values are more conservative. Default 0.2.
    delta :
        Convergence threshold: stop when max eta * |b_tilde_kl - 1| < delta
        across all non-empty cells. Default 0.01 (1% scaled correction).
    c :
        Credibility threshold: full credibility at c exposure units.
        z_kl = min(n_kl / c, 1.0). Default 100.0.
    max_iter :
        Maximum number of iterations. Default 50.
    min_bin_size :
        Minimum observations per cell to include in convergence check.
        Default 10.
    """

    def __init__(
        self,
        n_bins: int = 10,
        eta: float = 0.2,
        delta: float = 0.01,
        c: float = 100.0,
        max_iter: int = 50,
        min_bin_size: int = 10,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1].")
        if delta <= 0.0:
            raise ValueError("delta must be positive.")
        if c <= 0.0:
            raise ValueError("c must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        self.n_bins = n_bins
        self.eta = eta
        self.delta = delta
        self.c = c
        self.max_iter = max_iter
        self.min_bin_size = min_bin_size

        # Fitted state
        self._bin_edges: np.ndarray | None = None
        self._correction_table: dict[tuple[int, str], float] | None = None
        self._convergence_history: list[dict[str, Any]] = []
        self._final_cell_biases: dict[tuple[int, str], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "IterativeMulticalibrationCorrector":
        """
        Learn the correction surface via iterative multicalibration.

        Parameters
        ----------
        y_true :
            Observed outcomes. Shape (n,).
        y_pred :
            Model predictions (initial premiums). Shape (n,).
        protected :
            Protected group labels. Shape (n,).
        exposure :
            Exposure weights. If None, defaults to 1.0 for all. Shape (n,).

        Returns
        -------
        self
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

        protected_str = np.array([str(x) for x in protected])
        groups = np.unique(protected_str)

        # Compute bin edges from initial predictions (fixed throughout iterations)
        self._bin_edges = self._compute_bin_edges(y_pred, w)

        # Working copy of predictions that gets updated each iteration
        pi = y_pred.copy()
        bin_ids = self._assign_bins_from_edges(pi, self._bin_edges)

        self._convergence_history = []
        self._final_cell_biases = {}

        # Cumulative multiplicative correction per observation (product across iters)
        # We track this so transform() can apply the learned correction to new data
        # via bin+group lookup. We store the final per-(bin,group) factor.
        # After fit, _correction_table maps (bin, group) -> cumulative factor.
        cumulative_factor: dict[tuple[int, str], float] = {
            (b, g): 1.0
            for b in range(self.n_bins)
            for g in groups
        }

        converged = False
        for iteration in range(self.max_iter):
            # Compute bin-level pooled A/E ratios
            bin_ae: dict[int, float] = {}
            for b in range(self.n_bins):
                mask_bin = bin_ids == b
                obs_bin = float((y_true[mask_bin] * w[mask_bin]).sum())
                pred_bin = float((pi[mask_bin] * w[mask_bin]).sum())
                bin_ae[b] = (obs_bin / pred_bin) if pred_bin > 0 else 1.0

            # Compute cell-level A/E ratios and apply credibility blending
            cell_factors: dict[tuple[int, str], float] = {}
            # Stopping metric: max_{k,l} eta * |b_tilde_kl - 1| (Section 7.1.2).
            # This is the scale-invariant criterion from the paper: the correction
            # applied to pi is eta * (b_tilde_kl - 1) * pi; dividing by pi_bar_kl
            # gives eta * |b_tilde_kl - 1|, which is dimensionless and independent
            # of the premium level in each cell.
            max_scaled_correction = 0.0
            iter_cell_biases: dict[str, float] = {}

            for b in range(self.n_bins):
                mask_bin = bin_ids == b
                b_hat_k = bin_ae[b]

                for g in groups:
                    mask_cell = mask_bin & (protected_str == g)
                    n_obs = int(mask_cell.sum())
                    if n_obs == 0:
                        cell_factors[(b, g)] = 1.0
                        continue

                    obs_cell = float((y_true[mask_cell] * w[mask_cell]).sum())
                    pred_cell = float((pi[mask_cell] * w[mask_cell]).sum())

                    if pred_cell <= 0:
                        cell_factors[(b, g)] = 1.0
                        continue

                    b_hat_kl = obs_cell / pred_cell
                    # Credibility weight: use exposure sum as n_kl
                    n_kl = float(w[mask_cell].sum())
                    z_kl = min(n_kl / self.c, 1.0)
                    # Denuit et al. (2026): blend toward bin mean, not 1.0
                    b_tilde_kl = z_kl * b_hat_kl + (1.0 - z_kl) * b_hat_k
                    cell_factors[(b, g)] = b_tilde_kl

                    # Track convergence metric (large cells only).
                    # Paper Section 7.1.2: criterion is eta * |b_tilde_kl - 1|.
                    # Using b_tilde_kl (blended) not b_hat_kl (raw AE) ensures
                    # the stopping criterion is consistent with what we actually
                    # apply as a correction — partially-shrunk cells converge
                    # faster than fully-credible cells with the same raw bias.
                    if n_obs >= self.min_bin_size:
                        scaled_correction = self.eta * abs(b_tilde_kl - 1.0)
                        max_scaled_correction = max(max_scaled_correction, scaled_correction)
                        iter_cell_biases[f"bin{b}_group{g}"] = float(b_tilde_kl - 1.0)

            self._convergence_history.append({
                "iteration": iteration + 1,
                "max_relative_bias": float(max_scaled_correction),
            })

            # Apply additive update: pi += eta * (b_tilde_kl - 1) * pi
            for b in range(self.n_bins):
                mask_bin = bin_ids == b
                for g in groups:
                    mask_cell = mask_bin & (protected_str == g)
                    if not mask_cell.any():
                        continue
                    factor = cell_factors.get((b, g), 1.0)
                    # Multiplicative equivalent of additive update on response scale
                    step_factor = 1.0 + self.eta * (factor - 1.0)
                    pi[mask_cell] *= step_factor
                    cumulative_factor[(b, g)] *= step_factor

            # Check convergence using the paper's scale-invariant criterion
            if max_scaled_correction < self.delta:
                converged = True
                # Store final cell biases for convergence_report()
                self._final_cell_biases = dict(iter_cell_biases)
                break

        if not converged:
            # Store biases from last iteration
            self._final_cell_biases = dict(iter_cell_biases)

        self._correction_table = cumulative_factor
        self._groups = list(groups)
        return self

    def transform(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the learned correction surface to new predictions.

        Each observation is assigned to a (bin, group) cell and the
        cumulative correction factor learned during fit() is applied.
        Observations in unseen groups or bins receive no correction.

        Parameters
        ----------
        y_pred :
            Model predictions to correct. Shape (n,).
        protected :
            Protected group labels. Shape (n,).
        exposure :
            Exposure weights for bin assignment. If None, defaults to 1.0.

        Returns
        -------
        np.ndarray
            Corrected predictions, same shape as y_pred.
        """
        if self._correction_table is None or self._bin_edges is None:
            raise RuntimeError("Call fit() before transform().")

        y_pred = np.asarray(y_pred, dtype=float)
        protected_str = np.array([str(x) for x in protected])
        n = len(y_pred)

        if exposure is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(exposure, dtype=float)

        bin_ids = self._assign_bins_from_edges(y_pred, w)
        corrected = y_pred.copy()

        for (b, g), factor in self._correction_table.items():
            mask = (bin_ids == b) & (protected_str == g)
            corrected[mask] *= factor

        return corrected

    def convergence_report(self) -> dict[str, Any]:
        """
        Return a summary of the iterative correction process.

        Returns
        -------
        dict with keys:
            iterations : int
                Number of iterations run.
            converged : bool
                True if convergence criterion was met before max_iter.
            max_relative_bias_per_iteration : list[float]
                Maximum eta * |b_tilde_kl - 1| across credible cells at each
                iteration (the paper's scale-invariant stopping criterion).
            final_cell_biases : dict[str, float]
                Blended bias (b_tilde_kl - 1) for each (bin, group) cell at
                the final iteration. Keys formatted as "binK_groupG".
        """
        if not self._convergence_history:
            raise RuntimeError("Call fit() before convergence_report().")

        n_iters = len(self._convergence_history)
        final_bias = self._convergence_history[-1]["max_relative_bias"]
        converged = final_bias < self.delta

        return {
            "iterations": n_iters,
            "converged": converged,
            "max_relative_bias_per_iteration": [
                h["max_relative_bias"] for h in self._convergence_history
            ],
            "final_cell_biases": dict(self._final_cell_biases),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_bin_edges(self, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute exposure-weighted quantile bin edges from y_pred.

        Returns array of length n_bins + 1 with boundary values.
        """
        order = np.argsort(y_pred, kind="stable")
        sorted_pred = y_pred[order]
        sorted_w = w[order]

        cumw = np.cumsum(sorted_w)
        total_w = cumw[-1]
        if total_w <= 0:
            return np.linspace(y_pred.min(), y_pred.max() + 1e-10, self.n_bins + 1)

        frac = cumw / total_w
        quantile_positions = np.linspace(0.0, 1.0, self.n_bins + 1)

        edges = np.empty(self.n_bins + 1)
        edges[0] = sorted_pred[0] - 1e-10
        edges[-1] = sorted_pred[-1] + 1e-10

        for i, q in enumerate(quantile_positions[1:-1], start=1):
            idx = np.searchsorted(frac, q, side="left")
            idx = min(idx, len(sorted_pred) - 1)
            edges[i] = sorted_pred[idx]

        return edges

    def _assign_bins_from_edges(
        self, y_pred: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """
        Assign observations to bins using stored bin edges.

        Uses the edges computed at fit() time so transform() places new
        observations in the same bins as the training data.
        """
        if self._bin_edges is None:
            raise RuntimeError("Bin edges not computed. Call fit() first.")
        bin_ids = np.searchsorted(self._bin_edges[1:], y_pred, side="left")
        bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)
        return bin_ids


# ---------------------------------------------------------------------------
# IsotonicMulticalibrationCorrector
# ---------------------------------------------------------------------------


class IsotonicMulticalibrationCorrector:
    """
    One-shot isotonic multicalibration corrector (Algorithm A, Section 7.1.1).

    Implements the non-iterative groupwise isotonic regression approach from
    Denuit, Michaelides & Trufin (2026), arXiv:2603.16317.

    For each group l, the corrected premium is:

        pi_mbc(X) = m_hat_l(pi(X))

    where m_hat_l is an isotonic regression of Y on pi(X), fitted on the
    observations in group l weighted by exposure.

    Theoretical guarantee (Proposition 6.2): achieves

        E[Y | pi_mbc(X) = p, S = l] = p   for all p and categorical S = l

    i.e., full multicalibration within each group simultaneously.

    When to use vs. IterativeMulticalibrationCorrector
    ---------------------------------------------------
    Use IsotonicMulticalibrationCorrector when:
    - Group sizes are large and roughly even (all groups >= ~5% of portfolio).
    - You need a one-shot solution with no hyperparameter tuning (no eta, delta).
    - The miscalibration pattern within a group is non-monotonic — isotonic
      regression is more flexible than a single multiplicative correction.
    - You want the theoretical guarantee from Proposition 6.2 directly.

    Use IterativeMulticalibrationCorrector when:
    - One or more groups are small (< ~5% of portfolio). The isotonic fit for
      a small group will be noisy and may overfit.
    - You want to control the size of each correction step (eta parameter).
    - The correction surface is expected to be smooth across prediction levels.

    Note on out-of-range predictions at transform() time
    -----------------------------------------------------
    Isotonic regression is defined on the training range [min(pi), max(pi)]
    for each group. Predictions outside this range are clipped to the nearest
    training boundary before lookup. This is the standard interpolation
    convention and avoids extrapolating a monotone step function outside its
    support.

    Parameters
    ----------
    increasing :
        Direction of monotonicity constraint for isotonic regression.
        'auto' lets sklearn determine the direction from the data;
        True forces increasing (higher predictions -> higher corrections);
        False forces decreasing. Default 'auto'.
    out_of_bounds :
        How sklearn's IsotonicRegression handles predictions outside the
        training range. 'clip' (default) clips to the nearest boundary;
        'nan' returns NaN (not recommended for production use).

    References
    ----------
    Denuit, M., Michaelides, M. & Trufin, J. (2026). Multicalibration in
    Insurance Pricing. arXiv:2603.16317, Section 7.1.1 and Proposition 6.2.
    """

    def __init__(
        self,
        increasing: bool | str = "auto",
        out_of_bounds: str = "clip",
    ) -> None:
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

        # Fitted state: one IsotonicRegression per group
        self._group_models: dict[str, IsotonicRegression] | None = None
        self._groups_seen: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "IsotonicMulticalibrationCorrector":
        """
        Fit an isotonic regression of Y on predictions within each group.

        Parameters
        ----------
        y :
            Observed outcomes (claims, losses). Shape (n,).
        predictions :
            Model predictions (pure premiums, expected losses). Shape (n,).
            Must be non-negative.
        sensitive_features :
            Group labels. Shape (n,). Any hashable type.
        exposure :
            Exposure weights. Shape (n,). If None, defaults to 1.0 for all.
            Passed as ``sample_weight`` to IsotonicRegression.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        sensitive_features = np.asarray(sensitive_features)
        n = len(y)

        if len(predictions) != n or len(sensitive_features) != n:
            raise ValueError(
                "y, predictions, and sensitive_features must have the same length."
            )
        if np.any(predictions < 0):
            raise ValueError("predictions must be non-negative.")

        if exposure is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(exposure, dtype=float)
            if len(w) != n:
                raise ValueError("exposure must have the same length as y.")
            if np.any(w < 0):
                raise ValueError("exposure must be non-negative.")

        sensitive_str = np.array([str(x) for x in sensitive_features])
        groups = np.unique(sensitive_str)

        self._group_models = {}
        for g in groups:
            mask = sensitive_str == g
            pi_g = predictions[mask]
            y_g = y[mask]
            w_g = w[mask]

            if len(pi_g) == 0:
                continue

            iso = IsotonicRegression(
                increasing=self.increasing,
                out_of_bounds=self.out_of_bounds,
            )
            iso.fit(pi_g, y_g, sample_weight=w_g)
            self._group_models[g] = iso

        self._groups_seen = list(groups)
        return self

    def transform(
        self,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """
        Apply group-wise isotonic correction to predictions.

        Each observation is corrected using the isotonic regression fitted
        for its group. Observations in groups not seen during fit() are
        passed through unchanged (no correction applied).

        Out-of-range predictions are handled according to the ``out_of_bounds``
        parameter set at construction (default: clip to training range).

        Parameters
        ----------
        predictions :
            Model predictions to correct. Shape (n,).
        sensitive_features :
            Group labels. Shape (n,).

        Returns
        -------
        np.ndarray
            Corrected predictions, same shape as predictions.
        """
        if self._group_models is None or self._groups_seen is None:
            raise RuntimeError("Call fit() before transform().")

        predictions = np.asarray(predictions, dtype=float)
        sensitive_str = np.array([str(x) for x in sensitive_features])
        corrected = predictions.copy()

        for g, iso in self._group_models.items():
            mask = sensitive_str == g
            if not mask.any():
                continue
            corrected[mask] = iso.predict(predictions[mask])

        return corrected

    def fit_transform(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        sensitive_features: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit and immediately transform training predictions.

        Convenience method equivalent to fit(...).transform(...) on the
        same data. Note that this applies the correction to the training
        data itself — the result is the in-sample isotonic fit.

        Parameters
        ----------
        y :
            Observed outcomes. Shape (n,).
        predictions :
            Model predictions. Shape (n,).
        sensitive_features :
            Group labels. Shape (n,).
        exposure :
            Exposure weights. If None, defaults to 1.0 for all.

        Returns
        -------
        np.ndarray
            Corrected predictions, same shape as predictions.
        """
        return self.fit(y, predictions, sensitive_features, exposure).transform(
            predictions, sensitive_features
        )



# ---------------------------------------------------------------------------
# LocalGLMMulticalibrationCorrector — Algorithm C (Section 7.2.2)
# ---------------------------------------------------------------------------


class LocalGLMMulticalibrationCorrector:
    """
    Iterative multicalibration corrector via local GLM for continuous S (Algorithm C).

    Implements Section 7.2.2 of Denuit, Michaelides & Trufin (2026),
    arXiv:2603.16317. This is the continuous-sensitive-feature analogue of
    :class:`IterativeMulticalibrationCorrector` (Algorithm B).

    When S is truly continuous — age, income, vehicle age, distance to coast —
    binning it into a few categories loses information and introduces arbitrary
    boundary effects. Algorithm C replaces the bin+group lookup table with a
    smooth bias surface b(p, s) estimated via local regression over the joint
    space of prediction p and sensitive feature s.

    The iterative loop
    ------------------
    At each iteration j:

    1. Compute residuals r_i = Y_i - pi_i^(j) for each observation.
    2. Estimate marginal bias via 1D local regression:
       b_hat^(j)(p) = smooth(r, on=pi)   [Nadaraya-Watson with Gaussian kernel]
    3. Estimate bivariate bias via 2D local regression:
       b_hat^(j)(p, s) = smooth(r, on=(pi, s))
    4. Group deviation: delta^(j)(p, s) = b_hat^(j)(p, s) - b_hat^(j)(p)
    5. Credibility-weight the deviation:
       delta_tilde^(j)(p, s) = z(p, s) * delta^(j)(p, s)
    6. Center to preserve marginal autocalibration:
       b_tilde^(j)(p, s) = b_hat^(j)(p) + delta_tilde^(j)(p, s)
                            - smooth(delta_tilde^(j), on=pi)
    7. Update: pi_i^(j+1) = pi_i^(j) + eta * b_tilde^(j)(pi_i, s_i)
    8. Convergence check on a fixed evaluation grid.

    Centering (step 6) is critical: without it, each iteration shifts the
    marginal distribution, and the algorithm diverges from autocalibration.
    The centering term (smooth(delta_tilde^(j), on=pi)) is the exposure-
    weighted expected group deviation at each prediction level — subtracting
    it ensures the marginal bias correction is unchanged.

    Credibility weights
    -------------------
    Unlike Algorithm B where z_kl is recomputed each iteration as bins shift,
    here z(pi_i, s_i) is computed ONCE before the loop using k-nearest-
    neighbour density estimation:

        w_loc(i) = sum of exposure in k nearest neighbours in (pi_norm, s_norm) space
        z_i = w_loc(i) / (w_loc(i) + c)

    This prevents feedback between correction magnitude and shrinkage, which
    would otherwise create instability when the smooth surface is updated.

    Local regression implementation
    --------------------------------
    Python has no direct equivalent of R's locfit. We use Nadaraya-Watson
    kernel regression via statsmodels KernelReg, which estimates:

        E[Y | X = x] via  sum_i K((x - X_i)/h) Y_i / sum_i K((x - X_i)/h)

    with Gaussian kernel. The bandwidth is selected by leave-one-out cross-
    validation when ``bandwidth='cv'`` (default) or can be set manually.

    For 1D estimation (step 2), we regress residuals on predictions alone.
    For 2D estimation (step 3), we regress residuals on (predictions, s),
    treating both as continuous variables.

    Convergence
    -----------
    Convergence is assessed on a fixed grid of (p, s) evaluation points:

        max_{(p,s) in grid} eta * |b_tilde^(j)(p, s)| / p_bar <= delta

    where p_bar is the exposure-weighted mean prediction. The grid is
    n_eval_grid x n_eval_grid points spanning the training range of (pi, s).

    When to use
    -----------
    - S is genuinely continuous and should not be discretised.
    - Portfolio is large enough for kernel bandwidth selection to be stable
      (rule of thumb: n >= 500 after group filtering).
    - The miscalibration pattern is smooth in (p, s) — not driven by sharp
      regime changes at specific ages or income levels.

    Use :class:`IterativeMulticalibrationCorrector` (Algorithm B) when:
    - S has few distinct values (< ~20). Discretise it and use Algorithm B.
    - Portfolio is small. Kernel regression on small samples is unstable.
    - Computation time matters: local regression is O(n^2) per iteration.

    Parameters
    ----------
    eta :
        Learning rate for additive update. Must be in (0, 1]. Default 0.2.
    delta :
        Convergence threshold: stop when max scaled correction < delta.
        Default 0.01 (1% relative correction).
    c :
        Credibility tuning parameter. Full credibility when local exposure
        sum equals c. Default 100.0.
    max_iter :
        Maximum number of iterations. Default 50.
    bandwidth :
        Kernel bandwidth for local regression. ``'cv'`` selects bandwidth via
        leave-one-out cross-validation (slow but principled). A positive
        float uses that fixed bandwidth for both 1D and 2D regressions.
        Default ``'cv'``.
    n_neighbours :
        Number of nearest neighbours to use for local exposure density
        estimation (used to compute credibility weights z_i). Default 50.
    n_eval_grid :
        Number of evaluation grid points per axis for convergence check.
        Total grid size is n_eval_grid^2. Default 10.
    normalize_features :
        Whether to z-score normalise predictions and s before kernel
        regression. Strongly recommended when predictions and s have
        very different scales (e.g., prediction in [0.05, 0.8] and
        s = age in [17, 90]). Default True.

    References
    ----------
    Denuit, M., Michaelides, M. & Trufin, J. (2026). arXiv:2603.16317,
    Section 7.2.2 (Algorithm C).
    """

    def __init__(
        self,
        eta: float = 0.2,
        delta: float = 0.01,
        c: float = 100.0,
        max_iter: int = 50,
        bandwidth: float | str = "cv",
        n_neighbours: int = 50,
        n_eval_grid: int = 10,
        normalize_features: bool = True,
    ) -> None:
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1].")
        if delta <= 0.0:
            raise ValueError("delta must be positive.")
        if c <= 0.0:
            raise ValueError("c must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if bandwidth != "cv" and not (isinstance(bandwidth, (int, float)) and bandwidth > 0):
            raise ValueError("bandwidth must be 'cv' or a positive float.")
        if n_neighbours < 1:
            raise ValueError("n_neighbours must be at least 1.")
        if n_eval_grid < 2:
            raise ValueError("n_eval_grid must be at least 2.")

        self.eta = eta
        self.delta = delta
        self.c = c
        self.max_iter = max_iter
        self.bandwidth = bandwidth
        self.n_neighbours = n_neighbours
        self.n_eval_grid = n_eval_grid
        self.normalize_features = normalize_features

        # Fitted state
        self._pi_train: np.ndarray | None = None
        self._s_train: np.ndarray | None = None
        self._obs_correction_factor: np.ndarray | None = None
        self._credibility_weights: np.ndarray | None = None
        self._convergence_history: list[dict[str, Any]] = []
        self._converged: bool = False
        # Normalisation parameters (set during fit if normalize_features=True)
        self._pi_mean: float = 0.0
        self._pi_std: float = 1.0
        self._s_mean: float = 0.0
        self._s_std: float = 1.0
        # Grid for convergence check
        self._eval_grid_pi: np.ndarray | None = None
        self._eval_grid_s: np.ndarray | None = None
        # Final corrected predictions (fit-time, for fit_transform)
        self._corrected_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        sensitive_continuous: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "LocalGLMMulticalibrationCorrector":
        """
        Learn the bias correction surface via iterative local GLM.

        Parameters
        ----------
        y :
            Observed outcomes (claims, losses). Shape (n,).
        predictions :
            Initial model predictions. Shape (n,). Must be non-negative.
        sensitive_continuous :
            Continuous sensitive feature (age, income, etc.). Shape (n,).
        exposure :
            Exposure weights. Shape (n,). If None, all set to 1.0.

        Returns
        -------
        self
        """
        try:
            from statsmodels.nonparametric.kernel_regression import KernelReg
        except ImportError as exc:
            raise ImportError(
                "LocalGLMMulticalibrationCorrector requires statsmodels >= 0.14. "
                "Install with: pip install statsmodels"
            ) from exc
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError(
                "LocalGLMMulticalibrationCorrector requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from exc

        y = np.asarray(y, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        sensitive_continuous = np.asarray(sensitive_continuous, dtype=float)
        n = len(y)

        if len(predictions) != n or len(sensitive_continuous) != n:
            raise ValueError(
                "y, predictions, and sensitive_continuous must have the same length."
            )
        if np.any(predictions < 0):
            raise ValueError("predictions must be non-negative.")
        if np.any(~np.isfinite(sensitive_continuous)):
            raise ValueError("sensitive_continuous must be finite (no NaN/inf).")

        if exposure is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(exposure, dtype=float)
            if len(w) != n:
                raise ValueError("exposure must have the same length as y.")
            if np.any(w < 0):
                raise ValueError("exposure must be non-negative.")

        # Normalise features for kernel regression stability.
        # Predictions and s can have wildly different scales (premium 0.05-0.8,
        # age 17-90), which would cause bandwidth selection to favour one axis.
        if self.normalize_features:
            self._pi_mean = float(np.average(predictions, weights=w))
            self._pi_std = float(
                np.sqrt(np.average((predictions - self._pi_mean) ** 2, weights=w))
            )
            self._s_mean = float(np.average(sensitive_continuous, weights=w))
            self._s_std = float(
                np.sqrt(np.average((sensitive_continuous - self._s_mean) ** 2, weights=w))
            )
            if self._pi_std < 1e-10:
                self._pi_std = 1.0
            if self._s_std < 1e-10:
                self._s_std = 1.0
            pi_norm = (predictions - self._pi_mean) / self._pi_std
            s_norm = (sensitive_continuous - self._s_mean) / self._s_std
        else:
            pi_norm = predictions.copy()
            s_norm = sensitive_continuous.copy()

        # Pre-compute credibility weights z_i using kNN density in (pi_norm, s_norm) space.
        # z_i = w_loc(i) / (w_loc(i) + c), where w_loc(i) = sum of exposure of
        # k nearest neighbours. Computed ONCE before the loop — this is the key
        # algorithmic difference from Algorithm B (where z_kl is recomputed per iter).
        features_2d = np.column_stack([pi_norm, s_norm])
        k = min(self.n_neighbours, n - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1)  # +1 because point is own neighbour
        nn_model.fit(features_2d)
        nn_indices = nn_model.kneighbors(features_2d, return_distance=False)
        # Exclude self (index 0 in sorted distance order when using kneighbors)
        local_exposure = np.array([w[nn_indices[i, 1:]].sum() for i in range(n)])
        z = local_exposure / (local_exposure + self.c)
        self._credibility_weights = z

        # Build fixed evaluation grid for convergence check.
        # Using the training range of (pi_norm, s_norm) ensures grid stays in-distribution.
        pi_grid_vals = np.linspace(float(pi_norm.min()), float(pi_norm.max()), self.n_eval_grid)
        s_grid_vals = np.linspace(float(s_norm.min()), float(s_norm.max()), self.n_eval_grid)
        pi_grid_2d, s_grid_2d = np.meshgrid(pi_grid_vals, s_grid_vals)
        self._eval_grid_pi = pi_grid_2d.ravel()
        self._eval_grid_s = s_grid_2d.ravel()

        # Determine bandwidth strategy
        bw_flag: str | None = "cv_ls" if self.bandwidth == "cv" else None
        bw_fixed: float | None = None if self.bandwidth == "cv" else float(self.bandwidth)

        # Mean prediction (original scale) for scale-invariant convergence criterion
        p_bar = float(np.average(predictions, weights=w))
        if p_bar <= 0:
            p_bar = 1.0

        # Working copy of predictions
        pi = predictions.copy()
        pi_norm_work = pi_norm.copy()

        self._convergence_history = []
        self._converged = False

        for iteration in range(self.max_iter):
            residuals = y - pi  # r_i = Y_i - pi_i^(j)

            # Step 2: 1D local regression of residuals on pi_norm (marginal bias estimate)
            bhat_1d = self._kernel_reg_1d(
                y=residuals, x=pi_norm_work, w=w,
                bw_flag=bw_flag, bw_fixed=bw_fixed, KernelReg=KernelReg,
            )

            # Step 3: 2D local regression of residuals on (pi_norm, s_norm)
            bhat_2d = self._kernel_reg_2d(
                y=residuals, x1=pi_norm_work, x2=s_norm, w=w,
                bw_flag=bw_flag, bw_fixed=bw_fixed, KernelReg=KernelReg,
            )

            # Step 4: group deviation — excess bias beyond what pi alone predicts
            delta = bhat_2d - bhat_1d

            # Step 5: credibility-weight the group deviation
            delta_tilde = z * delta

            # Step 6: centering step — critical for preserving marginal autocalibration.
            # Without centering, each iteration shifts the marginal distribution.
            # We subtract E[delta_tilde | pi], the expected group deviation at each
            # prediction level, which ensures the marginal correction is unaffected.
            centering = self._kernel_reg_1d(
                y=delta_tilde, x=pi_norm_work, w=w,
                bw_flag=bw_flag, bw_fixed=bw_fixed, KernelReg=KernelReg,
            )
            b_tilde = bhat_1d + delta_tilde - centering

            # Convergence check on fixed grid before applying update.
            # Evaluate b_tilde surface at grid points via 2D kernel regression.
            b_tilde_grid = self._kernel_reg_2d_predict(
                y=b_tilde, x1=pi_norm_work, x2=s_norm, w=w,
                eval_x1=self._eval_grid_pi, eval_x2=self._eval_grid_s,
                bw_flag=bw_flag, bw_fixed=bw_fixed, KernelReg=KernelReg,
            )
            # Reconstruct grid pi on original scale for denominator
            if self.normalize_features:
                pi_grid_orig = self._eval_grid_pi * self._pi_std + self._pi_mean
            else:
                pi_grid_orig = self._eval_grid_pi.copy()
            pi_grid_orig = np.where(pi_grid_orig > 1e-10, pi_grid_orig, p_bar)
            # Scale-invariant criterion: eta * |b_tilde(p,s)| / p  (analogous to Algorithm B)
            scaled_corrections = self.eta * np.abs(b_tilde_grid) / pi_grid_orig
            max_scaled = float(np.nanmax(scaled_corrections)) if len(scaled_corrections) > 0 else 0.0

            self._convergence_history.append({
                "iteration": iteration + 1,
                "max_scaled_correction": max_scaled,
            })

            # Step 7: additive update on prediction scale
            pi = pi + self.eta * b_tilde
            pi = np.clip(pi, 0.0, None)  # predictions must stay non-negative

            # Update normalised working copy (pi has changed)
            if self.normalize_features:
                pi_norm_work = (pi - self._pi_mean) / self._pi_std
            else:
                pi_norm_work = pi.copy()

            # Step 8: check convergence
            if max_scaled < self.delta:
                self._converged = True
                break

        # Store fitted state needed for transform()
        self._pi_train = pi_norm.copy()       # original normalised pi for kNN
        self._s_train = s_norm.copy()
        self._corrected_train = pi.copy()
        # Per-obs cumulative correction factor: corrected_pi / original_pi
        safe_pred = np.where(predictions > 1e-10, predictions, 1e-10)
        self._obs_correction_factor = pi / safe_pred
        self._w_train = w.copy()
        return self

    def transform(
        self,
        predictions: np.ndarray,
        sensitive_continuous: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the learned bias correction surface to new predictions.

        The correction surface learned during fit() is generalised to new
        observations via k-NN weighted interpolation: each new observation
        is assigned an inverse-distance-weighted average of the correction
        factors of its nearest training neighbours in normalised (pi, s) space.

        Parameters
        ----------
        predictions :
            Model predictions to correct. Shape (n,).
        sensitive_continuous :
            Continuous sensitive feature values. Shape (n,).
        exposure :
            Unused; present for API consistency with other correctors.

        Returns
        -------
        np.ndarray
            Corrected predictions, same shape as predictions.
        """
        if self._obs_correction_factor is None or self._pi_train is None:
            raise RuntimeError("Call fit() before transform().")

        predictions = np.asarray(predictions, dtype=float)
        sensitive_continuous = np.asarray(sensitive_continuous, dtype=float)
        n = len(predictions)

        if len(sensitive_continuous) != n:
            raise ValueError(
                "predictions and sensitive_continuous must have the same length."
            )

        # Normalise using fit-time parameters
        if self.normalize_features:
            pi_norm_new = (predictions - self._pi_mean) / self._pi_std
            s_norm_new = (sensitive_continuous - self._s_mean) / self._s_std
        else:
            pi_norm_new = predictions.copy()
            s_norm_new = sensitive_continuous.copy()

        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError("scikit-learn required for transform().") from exc

        train_features = np.column_stack([self._pi_train, self._s_train])
        new_features = np.column_stack([pi_norm_new, s_norm_new])

        k = min(self.n_neighbours, len(self._pi_train))
        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(train_features)
        distances, indices = nn_model.kneighbors(new_features)

        # Inverse-distance-weighted average of training correction factors.
        # When a new obs coincides exactly with a training obs, distance is 0,
        # so we add eps to avoid division by zero.
        eps = 1e-10
        inv_d = 1.0 / (distances + eps)
        weights_nn = inv_d / inv_d.sum(axis=1, keepdims=True)  # (n_new, k)
        neighbor_factors = self._obs_correction_factor[indices]  # (n_new, k)
        correction_factors = (weights_nn * neighbor_factors).sum(axis=1)  # (n_new,)

        corrected = predictions * correction_factors
        return np.clip(corrected, 0.0, None)

    def fit_transform(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        sensitive_continuous: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit and return corrected predictions on the training data.

        Equivalent to ``fit(...).transform(...)`` on the same data.

        Parameters
        ----------
        y :
            Observed outcomes. Shape (n,).
        predictions :
            Model predictions. Shape (n,).
        sensitive_continuous :
            Continuous sensitive feature. Shape (n,).
        exposure :
            Exposure weights. If None, all set to 1.0.

        Returns
        -------
        np.ndarray
            Corrected predictions on training data, same shape as predictions.
        """
        self.fit(y, predictions, sensitive_continuous, exposure)
        assert self._corrected_train is not None
        return self._corrected_train.copy()

    def convergence_report(self) -> dict[str, Any]:
        """
        Return convergence diagnostics from the iterative correction loop.

        Returns
        -------
        dict with keys:
            iterations : int
                Number of iterations run before stopping.
            converged : bool
                True if convergence criterion was met before max_iter.
            max_scaled_correction_per_iteration : list[float]
                Maximum eta * |b_tilde(p,s)| / p at each iteration.
                Evaluated on fixed grid of (p, s) values.
            credibility_weights_summary : dict
                Summary statistics (min, mean, median, max) of z_i values
                computed before the loop.
        """
        if not self._convergence_history:
            raise RuntimeError("Call fit() before convergence_report().")

        z = self._credibility_weights
        z_summary: dict[str, float] = {}
        if z is not None and len(z) > 0:
            z_summary = {
                "min": float(z.min()),
                "mean": float(z.mean()),
                "median": float(np.median(z)),
                "max": float(z.max()),
            }

        return {
            "iterations": len(self._convergence_history),
            "converged": self._converged,
            "max_scaled_correction_per_iteration": [
                h["max_scaled_correction"] for h in self._convergence_history
            ],
            "credibility_weights_summary": z_summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kernel_reg_1d(
        self,
        y: np.ndarray,
        x: np.ndarray,
        w: np.ndarray,
        bw_flag: str | None,
        bw_fixed: float | None,
        KernelReg: type,
    ) -> np.ndarray:
        """
        Nadaraya-Watson kernel regression of y on x (1D), returning fit at training points.
        """
        try:
            if bw_fixed is not None:
                kr = KernelReg(
                    endog=y, exog=x, var_type="c", reg_type="lc",
                    bw=[bw_fixed], defaults=None,
                )
            else:
                kr = KernelReg(
                    endog=y, exog=x, var_type="c", reg_type="lc", bw=bw_flag,
                )
            fitted, _ = kr.fit()
            return np.asarray(fitted, dtype=float)
        except Exception:
            # Fallback to global mean if kernel regression fails (e.g., degenerate data)
            return np.full(len(y), float(np.average(y, weights=w)))

    def _kernel_reg_2d(
        self,
        y: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
        w: np.ndarray,
        bw_flag: str | None,
        bw_fixed: float | None,
        KernelReg: type,
    ) -> np.ndarray:
        """
        Nadaraya-Watson kernel regression of y on (x1, x2), returning fit at training points.
        """
        X = np.column_stack([x1, x2])
        try:
            if bw_fixed is not None:
                kr = KernelReg(
                    endog=y, exog=X, var_type="cc", reg_type="lc",
                    bw=[bw_fixed, bw_fixed], defaults=None,
                )
            else:
                kr = KernelReg(
                    endog=y, exog=X, var_type="cc", reg_type="lc", bw=bw_flag,
                )
            fitted, _ = kr.fit()
            return np.asarray(fitted, dtype=float)
        except Exception:
            return np.full(len(y), float(np.average(y, weights=w)))

    def _kernel_reg_2d_predict(
        self,
        y: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
        w: np.ndarray,
        eval_x1: np.ndarray,
        eval_x2: np.ndarray,
        bw_flag: str | None,
        bw_fixed: float | None,
        KernelReg: type,
    ) -> np.ndarray:
        """
        Nadaraya-Watson 2D regression predicting at eval_x1, eval_x2 grid points.
        """
        X = np.column_stack([x1, x2])
        eval_X = np.column_stack([eval_x1, eval_x2])
        try:
            if bw_fixed is not None:
                kr = KernelReg(
                    endog=y, exog=X, var_type="cc", reg_type="lc",
                    bw=[bw_fixed, bw_fixed], defaults=None,
                )
            else:
                kr = KernelReg(
                    endog=y, exog=X, var_type="cc", reg_type="lc", bw=bw_flag,
                )
            predicted, _ = kr.fit(eval_X)
            return np.asarray(predicted, dtype=float)
        except Exception:
            return np.full(len(eval_x1), float(np.average(y, weights=w)))


# ---------------------------------------------------------------------------
# ProxySufficiencyReport and proxy_sufficiency_test
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BinSufficiencyResult:
    """
    Per-bin result from the proxy sufficiency test.

    Attributes
    ----------
    bin_index :
        Integer bin index (0-based).
    f_statistic :
        Weighted one-way ANOVA F-statistic for this bin. NaN if the bin
        cannot be tested (fewer than 2 non-empty groups, or no variance).
    p_value :
        p-value for the F-test. NaN if untestable.
    group_means :
        Dict mapping group label (str) to the exposure-weighted mean of Y
        within this bin for that group.
    group_exposures :
        Dict mapping group label to total exposure in this bin-group cell.
    testable :
        True if this bin had at least 2 groups with positive exposure and
        a non-degenerate within-group variance estimate.
    """

    bin_index: int
    f_statistic: float
    p_value: float
    group_means: dict[str, float]
    group_exposures: dict[str, float]
    testable: bool


@dataclasses.dataclass
class ProxySufficiencyReport:
    """
    Result of a proxy sufficiency test (Proposition 6.5, Denuit et al. 2026).

    A model satisfies proxy sufficiency for the excluded characteristic S if
    E[Y | pi(X), S=s] = E[Y | pi(X)] for all s — i.e., the model's predictions
    already capture all risk variation associated with S. This is the necessary
    and sufficient condition for multicalibration to be achievable without S.

    Attributes
    ----------
    sufficient :
        True if we fail to reject conditional mean independence at ``alpha``.
        Equivalently, True means the model captures all risk variation from S
        through allowed proxies.
    overall_p_value :
        Combined p-value from Fisher's method across all testable bins.
    overall_statistic :
        Fisher's combined chi-squared statistic (sum of -2 * log(p_i) across
        testable bins).
    bin_results :
        Per-bin F-test results. See :class:`BinSufficiencyResult`.
    interpretation :
        One-sentence plain-English summary.
    alpha :
        Significance level used.
    n_bins :
        Number of prediction bins used.
    n_testable_bins :
        Number of bins that were testable (had >= 2 non-empty groups).
    failing_bins :
        List of bin indices where the F-test p-value < alpha.
    """

    sufficient: bool
    overall_p_value: float
    overall_statistic: float
    bin_results: list[BinSufficiencyResult]
    interpretation: str
    alpha: float
    n_bins: int
    n_testable_bins: int
    failing_bins: list[int]


def proxy_sufficiency_test(
    y: np.ndarray,
    predictions: np.ndarray,
    sensitive_features: np.ndarray,
    exposure: np.ndarray | None = None,
    n_bins: int = 10,
    alpha: float = 0.05,
    sensitive_name: str | None = None,
) -> ProxySufficiencyReport:
    """
    Test whether a model satisfies proxy sufficiency for an excluded characteristic.

    Implements the conditional mean independence (CMI) check from Proposition 6.5
    of Denuit, Michaelides & Trufin (2026), arXiv:2603.16317.

    A model satisfies proxy sufficiency for excluded characteristic S if:

        E[Y | pi(X), S=s] = E[Y | pi(X)]   for all s

    This is the condition under which multicalibration is achievable without S
    in the model. When it fails, the model's predictions do not fully capture
    the risk variation associated with S, and residual proxy discrimination
    remains in the premium.

    Method
    ------
    1. Bin predictions into ``n_bins`` quantile bins (exposure-weighted).
    2. Within each bin, run a weighted one-way ANOVA (F-test) comparing
       exposure-weighted group means E[Y | bin, S=s] across groups s.
    3. Combine bin-level p-values using Fisher's method to obtain an overall
       test statistic and p-value.

    The weighted ANOVA F-statistic for each bin is:

        F = (SS_between / df_between) / (SS_within / df_within)

    where the sums of squares are exposure-weighted, and degrees of freedom
    are (n_groups - 1) and (sum of n_obs_per_group - n_groups) respectively.
    Bins with fewer than 2 non-empty groups, or with zero within-group
    variance, are skipped and excluded from the Fisher combination.

    Parameters
    ----------
    y :
        Observed outcomes (claims, losses). Shape (n,).
    predictions :
        Model predictions pi(X). Shape (n,). Must be non-negative.
    sensitive_features :
        Labels for the excluded characteristic S. Shape (n,). Any hashable type.
    exposure :
        Observation weights (years of exposure, policy counts, etc.). Shape (n,).
        If None, all observations are weighted equally.
    n_bins :
        Number of quantile bins for the prediction axis. More bins give finer
        resolution but reduce within-bin sample sizes. Default 10.
    alpha :
        Significance level. The test is rejected (sufficient=False) if the
        overall Fisher p-value < alpha. Default 0.05.
    sensitive_name :
        Name of the sensitive characteristic (used in the interpretation string).
        If None, defaults to "the excluded characteristic".

    Returns
    -------
    ProxySufficiencyReport

    References
    ----------
    Denuit, M., Michaelides, M. & Trufin, J. (2026). Multicalibration in
    Insurance Pricing. arXiv:2603.16317, Proposition 6.5.
    """
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    sensitive_features = np.asarray(sensitive_features)
    n = len(y)

    if len(predictions) != n or len(sensitive_features) != n:
        raise ValueError(
            "y, predictions, and sensitive_features must have the same length."
        )
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    if exposure is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(exposure, dtype=float)
        if len(w) != n:
            raise ValueError("exposure must have the same length as y.")
        if np.any(w < 0):
            raise ValueError("exposure must be non-negative.")

    sensitive_str = np.array([str(s) for s in sensitive_features])
    groups = np.unique(sensitive_str)
    s_name = sensitive_name if sensitive_name is not None else "the excluded characteristic"

    # --- Edge case: single group ---
    if len(groups) == 1:
        # With only one group, conditional mean independence holds trivially
        empty_result = BinSufficiencyResult(
            bin_index=0,
            f_statistic=float("nan"),
            p_value=float("nan"),
            group_means={groups[0]: float(np.average(y, weights=w)) if w.sum() > 0 else float("nan")},
            group_exposures={groups[0]: float(w.sum())},
            testable=False,
        )
        return ProxySufficiencyReport(
            sufficient=True,
            overall_p_value=1.0,
            overall_statistic=0.0,
            bin_results=[empty_result],
            interpretation=(
                f"The test is trivially satisfied: only one group was present in "
                f"sensitive_features, so conditional mean independence holds by definition."
            ),
            alpha=alpha,
            n_bins=n_bins,
            n_testable_bins=0,
            failing_bins=[],
        )

    # --- Assign quantile bins (exposure-weighted) ---
    bin_ids = _assign_bins_sufficiency(predictions, w, n_bins)

    # --- Per-bin F-tests ---
    bin_results: list[BinSufficiencyResult] = []
    fisher_log_p_sum = 0.0
    n_testable = 0
    failing_bins: list[int] = []

    unique_bins = np.unique(bin_ids)

    for b in range(n_bins):
        mask_bin = bin_ids == b
        if not mask_bin.any():
            # Empty bin — create a placeholder but mark untestable
            bin_results.append(BinSufficiencyResult(
                bin_index=b,
                f_statistic=float("nan"),
                p_value=float("nan"),
                group_means={},
                group_exposures={},
                testable=False,
            ))
            continue

        # Compute exposure-weighted mean per group within this bin
        group_means: dict[str, float] = {}
        group_exposures: dict[str, float] = {}
        group_obs: dict[str, np.ndarray] = {}
        group_weights: dict[str, np.ndarray] = {}

        for g in groups:
            mask_cell = mask_bin & (sensitive_str == g)
            if not mask_cell.any():
                continue
            w_g = w[mask_cell]
            y_g = y[mask_cell]
            total_exp = float(w_g.sum())
            if total_exp <= 0:
                continue
            mu_g = float(np.dot(y_g, w_g) / total_exp)
            group_means[g] = mu_g
            group_exposures[g] = total_exp
            group_obs[g] = y_g
            group_weights[g] = w_g

        non_empty_groups = list(group_means.keys())
        n_groups_bin = len(non_empty_groups)

        if n_groups_bin < 2:
            bin_results.append(BinSufficiencyResult(
                bin_index=b,
                f_statistic=float("nan"),
                p_value=float("nan"),
                group_means=group_means,
                group_exposures=group_exposures,
                testable=False,
            ))
            continue

        # Exposure-weighted grand mean in this bin
        total_bin_exp = sum(group_exposures[g] for g in non_empty_groups)
        grand_mean = sum(
            group_means[g] * group_exposures[g] for g in non_empty_groups
        ) / total_bin_exp

        # Weighted SS_between (between-group sum of squares)
        ss_between = sum(
            group_exposures[g] * (group_means[g] - grand_mean) ** 2
            for g in non_empty_groups
        )
        df_between = n_groups_bin - 1

        # Weighted SS_within (within-group sum of squares)
        # For each observation i in group g: w_i * (y_i - mu_g)^2
        ss_within = 0.0
        df_within_total = 0
        for g in non_empty_groups:
            y_g = group_obs[g]
            w_g = group_weights[g]
            mu_g = group_means[g]
            ss_within += float(np.dot(w_g, (y_g - mu_g) ** 2))
            # Effective df contribution: n_obs_g - 1 (standard ANOVA approach)
            df_within_total += len(y_g) - 1

        if ss_within <= 0 or df_within_total <= 0:
            # Degenerate: within-group variance is zero (e.g., all Y identical)
            # If ss_between > 0 and ss_within == 0, the F-stat would be infinite
            # but this typically indicates a data problem. We set p=0 in this case.
            if ss_between > 1e-12:
                f_stat = float("inf")
                p_val = 0.0
            else:
                # Both between and within are zero: perfectly uniform, CMI holds
                f_stat = 0.0
                p_val = 1.0
            testable = True
        else:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within_total
            if ms_within <= 0:
                f_stat = float("inf") if ms_between > 0 else 0.0
                p_val = 0.0 if ms_between > 0 else 1.0
            else:
                f_stat = ms_between / ms_within
                p_val = float(stats.f.sf(f_stat, dfn=df_between, dfd=df_within_total))
            testable = True

        bin_results.append(BinSufficiencyResult(
            bin_index=b,
            f_statistic=float(f_stat) if not np.isinf(f_stat) else float("inf"),
            p_value=p_val,
            group_means=group_means,
            group_exposures=group_exposures,
            testable=testable,
        ))

        if testable:
            n_testable += 1
            # Fisher's method: accumulate -2 * log(p)
            # Clamp p to [1e-300, 1.0] to avoid log(0)
            p_clamped = max(p_val, 1e-300)
            fisher_log_p_sum += -2.0 * np.log(p_clamped)
            if p_val < alpha:
                failing_bins.append(b)

    # --- Fisher's combined test ---
    if n_testable == 0:
        overall_stat = 0.0
        overall_pvalue = 1.0
    else:
        overall_stat = float(fisher_log_p_sum)
        # Fisher statistic ~ chi-squared with 2 * n_testable df
        overall_pvalue = float(stats.chi2.sf(overall_stat, df=2 * n_testable))

    sufficient = overall_pvalue >= alpha

    # --- Build interpretation ---
    if sufficient:
        interpretation = (
            f"The model captures the risk variation associated with {s_name}: "
            f"no statistically significant differences in expected claims between "
            f"groups were detected across prediction bins "
            f"(Fisher p={overall_pvalue:.3f}, alpha={alpha})."
        )
    else:
        failing_str = ", ".join(str(b) for b in failing_bins)
        interpretation = (
            f"The model does not fully capture the risk variation associated with "
            f"{s_name}: bins [{failing_str}] show statistically significant "
            f"differences in expected claims between groups, indicating residual "
            f"proxy discrimination "
            f"(Fisher p={overall_pvalue:.4f}, alpha={alpha})."
        )

    return ProxySufficiencyReport(
        sufficient=sufficient,
        overall_p_value=overall_pvalue,
        overall_statistic=overall_stat,
        bin_results=bin_results,
        interpretation=interpretation,
        alpha=alpha,
        n_bins=n_bins,
        n_testable_bins=n_testable,
        failing_bins=failing_bins,
    )


def _assign_bins_sufficiency(
    predictions: np.ndarray,
    w: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Assign observations to quantile bins by exposure-weighted cumulative fraction.

    Returns integer array of bin indices in [0, n_bins).
    """
    n = len(predictions)
    if n == 0:
        return np.array([], dtype=int)

    order = np.argsort(predictions, kind="stable")
    sorted_w = w[order]

    cumw = np.cumsum(sorted_w)
    total_w = cumw[-1]
    if total_w <= 0:
        return np.zeros(n, dtype=int)

    frac = cumw / total_w
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids_sorted = np.searchsorted(bin_boundaries[1:], frac, side="left")
    bin_ids_sorted = np.clip(bin_ids_sorted, 0, n_bins - 1)

    bin_ids = np.empty(n, dtype=int)
    bin_ids[order] = bin_ids_sorted
    return bin_ids
