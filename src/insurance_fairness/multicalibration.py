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
    maximum relative bias across all (bin, group) cells drops below a
    convergence threshold. This is the iterative version of the one-shot
    credibility correction in MulticalibrationAudit.correct().

    The algorithm at each iteration j:

        1. Compute b_hat_kl^(j) = O_kl / E_kl^(j)  (cell-level A/E)
        2. Compute b_hat_k^(j) = O_k / E_k^(j)      (bin-level pooled A/E)
        3. Credibility blend: b_tilde_kl = z_kl * b_hat_kl + (1 - z_kl) * b_hat_k
           where z_kl = min(n_kl / c, 1.0)
        4. Additive update: pi^(j+1) = pi^(j) + eta * (b_tilde_kl - 1) * pi^(j)
        5. Stop if max |b_hat_kl - 1| < delta

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
        Convergence threshold: stop when max |b_hat_kl - 1| < delta across
        all non-empty cells. Default 0.01 (1% relative bias).
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
            max_rel_bias = 0.0
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

                    # Track relative bias for convergence check (large cells only)
                    if n_obs >= self.min_bin_size:
                        rel_bias = abs(b_hat_kl - 1.0)
                        max_rel_bias = max(max_rel_bias, rel_bias)
                        iter_cell_biases[f"bin{b}_group{g}"] = float(b_hat_kl - 1.0)

            self._convergence_history.append({
                "iteration": iteration + 1,
                "max_relative_bias": float(max_rel_bias),
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

            # Check convergence
            if max_rel_bias < self.delta:
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
                Maximum |b_hat_kl - 1| across credible cells at each iteration.
            final_cell_biases : dict[str, float]
                Relative bias (b_hat_kl - 1) for each (bin, group) cell at
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
