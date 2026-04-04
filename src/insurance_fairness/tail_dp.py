"""
tail_dp.py
----------
TailDemographicParityCorrector: enforce demographic parity only over the tail
of the predicted premium distribution.

Background
----------
Standard demographic parity (DP) constraints equalise the *entire* prediction
distribution across protected groups. This is wasteful and accuracy-damaging.
Fairness harms concentrate in the tails: a firm that charges Group A materially
more than Group B at the 95th percentile of its book is causing concrete harm to
high-premium customers — the exact population the FCA Consumer Duty is designed
to protect. The below-median population sees almost no harm differential.

Le, Denis & Hebiri (2025, arXiv:2604.02017) formalise this observation. They
propose enforcing DP only over a targeted tail region, defined by a quantile
threshold q, using optimal transport to equalise the conditional distributions

    F(y | S=s, y > Q_q),  s in groups

leaving predictions below Q_q unchanged. The Wasserstein barycenter (weighted
average quantile function) is the natural target: it minimises the mean squared
displacement to a distribution that satisfies DP in the tail.

This is directly applicable to UK insurance under FCA Consumer Duty because:

- Consumer Duty Outcome 4 (Price and Value) requires that products deliver
  fair value to *all* customer segments, not just on average.
- FCA EP25/2 highlights proxy discrimination risk at the high-premium end of
  motor and home portfolios, where vulnerable customer characteristics
  (deprivation, disability proxies) cluster.
- Correction only in the tail means minimal disturbance to the bulk of pricing
  relativities — pricing teams can defend the correction to reserving and
  underwriting.

Implementation decisions
------------------------
Two methods are supported:

'wasserstein' — optimal transport via empirical quantile matching (sort and
match). For each protected group, the tail predictions are mapped to the
Wasserstein barycenter of all groups' tail distributions. This is exact (in
the empirical sense) on training data and uses linear interpolation for new
data. Computationally O(n log n) per group due to sorting.

'reweight' — density ratio reweighting. For each tail observation, we compute
the weight w(y, s) = p_bar(y) / p_s(y) where p_bar is a kernel density
estimate of the pooled tail distribution and p_s is the KDE for group s. We
do NOT use this to modify predictions directly; instead we use the reweighted
distribution to construct a quantile function, then apply the same OT map as
'wasserstein'. The difference is that 'reweight' uses smoothed KDE-based
quantile functions rather than empirical ones, which is preferable when tail
groups are small (< ~100 observations).

Boundary handling: observations below the threshold are left exactly unchanged.
The OT map is applied only to tail observations. At transform time, the
threshold is determined from the fitted data (not re-computed), so the
correction is stable across train/test splits.

Multi-group: the barycenter is the weighted average quantile function across
all groups, using portfolio proportions as weights (same convention as
WassersteinCorrector). Three or more groups are fully supported.

References
----------
Le, C., Denis, C. & Hebiri, M. (2025). Demographic Parity Tails for
Regression. arXiv:2604.02017.

Le Gouic, T. & Loubes, J.-M. (2017). Existence and consistency of Wasserstein
barycenters. Probability Theory and Related Fields.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _weighted_ecdf(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a weighted empirical CDF.

    Returns (sorted_values, cumulative_weights) suitable for np.interp.
    Uses right-continuous convention: cum_w[i] = P(X <= sorted_vals[i]).

    Parameters
    ----------
    values :
        1-D float array.
    weights :
        Non-negative weights; defaults to equal weights.
    """
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    sv = values[order]
    sw = weights[order]
    total = sw.sum()
    if total <= 0.0:
        raise ValueError("Total weight is zero; cannot compute ECDF.")
    cum_w = np.cumsum(sw) / total
    return sv, cum_w


def _barycenter_qf(
    ecdfs: list[tuple[np.ndarray, np.ndarray]],
    group_weights: np.ndarray,
    n_quantiles: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Wasserstein barycenter quantile function.

    The 1-D Wasserstein barycenter of distributions {mu_s} with weights
    {w_s} has quantile function Q_bar(u) = sum_s w_s * Q_s(u), where Q_s
    is the quantile function of mu_s (Le Gouic & Loubes, 2017).

    Parameters
    ----------
    ecdfs :
        List of (ecdf_x, ecdf_y) pairs from _weighted_ecdf, one per group.
    group_weights :
        Positive weights for each group, summing to 1.
    n_quantiles :
        Resolution of the quantile grid.

    Returns
    -------
    u_grid : np.ndarray, shape (n_quantiles,)
    bar_qf : np.ndarray, shape (n_quantiles,)
    """
    u_grid = np.linspace(0.0, 1.0, n_quantiles)
    w = np.asarray(group_weights, dtype=float)
    w = w / w.sum()
    bar_qf = np.zeros(n_quantiles)
    for (ecdf_x, ecdf_y), wi in zip(ecdfs, w):
        qf_i = np.interp(u_grid, ecdf_y, ecdf_x)
        bar_qf += wi * qf_i
    return u_grid, bar_qf


def _kde_quantile_ecdf(
    values: np.ndarray,
    n_quantiles: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth ECDF using Gaussian KDE.

    Uses scipy.stats.gaussian_kde to estimate a density, then integrates to
    get a smooth CDF evaluated on a grid. Returns (grid_x, cdf_y) in the same
    format as _weighted_ecdf, suitable for _barycenter_qf.

    Falls back to empirical ECDF when the group has fewer than 5 observations
    or when the bandwidth estimate degenerates (all values identical).

    Parameters
    ----------
    values :
        1-D float array of tail observations for a single group.
    n_quantiles :
        Grid resolution.
    """
    from scipy.stats import gaussian_kde

    n = len(values)
    if n < 5 or np.std(values) < 1e-12:
        # Degenerate: fall back to empirical ECDF
        return _weighted_ecdf(values)

    kde = gaussian_kde(values)
    lo = float(np.min(values))
    hi = float(np.max(values))
    # Extend slightly beyond data range to avoid boundary effects
    spread = hi - lo
    lo -= 0.1 * spread
    hi += 0.1 * spread

    grid = np.linspace(lo, hi, n_quantiles)
    density = kde(grid)
    # Integrate density to get CDF using trapezoidal rule
    try:
        from numpy import trapezoid as _trap
    except ImportError:
        from numpy import trapz as _trap  # type: ignore[attr-defined]

    cum = np.array([float(_trap(density[:i + 1], grid[:i + 1])) for i in range(n_quantiles)])
    cum = np.clip(cum / cum[-1], 0.0, 1.0)
    return grid, cum


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TailDPReport:
    """Diagnostic report from TailDemographicParityCorrector.

    Attributes
    ----------
    quantile_threshold :
        The threshold parameter q used to define the tail (e.g. 0.9 = top 10%).
    tail_cutoff :
        The actual prediction value at the quantile threshold, fitted from
        training data.
    n_affected :
        Number of observations in the tail (above tail_cutoff) in the fitted
        data.
    proportion_affected :
        Fraction of observations in the tail.
    ks_before :
        Maximum pairwise KS statistic across all group pairs, measured on tail
        predictions *before* correction. A value near 0 means distributions
        are already similar.
    ks_after :
        Maximum pairwise KS statistic after correction.
    ks_reduction :
        Absolute reduction in max pairwise KS statistic (ks_before - ks_after).
        Positive means correction was effective.
    mean_shift_by_group :
        Dict mapping each group label to the mean shift applied to that group's
        tail predictions (mean_corrected - mean_original). Negative means the
        group's tail was shifted down; positive means up.
    group_tail_sizes :
        Dict mapping each group label to the number of tail observations in
        the fitted data.
    """

    quantile_threshold: float
    tail_cutoff: float
    n_affected: int
    proportion_affected: float
    ks_before: float
    ks_after: float
    ks_reduction: float
    mean_shift_by_group: dict[str, float]
    group_tail_sizes: dict[str, int]

    def __repr__(self) -> str:
        return (
            f"TailDPReport("
            f"q={self.quantile_threshold}, "
            f"cutoff={self.tail_cutoff:.2f}, "
            f"affected={self.proportion_affected:.1%}, "
            f"KS: {self.ks_before:.4f} → {self.ks_after:.4f})"
        )


# ---------------------------------------------------------------------------
# Main corrector
# ---------------------------------------------------------------------------


class TailDemographicParityCorrector:
    """Post-processing corrector enforcing demographic parity in the prediction tail.

    Applies an optimal transport correction only to predictions above a quantile
    threshold, leaving the bulk of the pricing distribution unchanged. This
    implements the tail-DP approach of Le, Denis & Hebiri (arXiv:2604.02017).

    The key insight: in insurance pricing, the harm from proxy discrimination
    concentrates at the high end of the premium distribution. Customers charged
    premiums in the top 10% of the book face the highest rates, and group
    disparities there constitute material harm under FCA Consumer Duty. Correcting
    only this tail minimises the accuracy cost relative to full demographic parity.

    Parameters
    ----------
    quantile_threshold :
        The probability level defining the tail, e.g. 0.9 means the top 10% of
        predictions are subject to correction. Must be in (0, 1). Typical values:
        0.8 (aggressive, corrects top 20%), 0.9 (standard), 0.95 (conservative).
    method :
        'wasserstein' (default) — empirical OT via sort-and-match. Exact on
        training data. Recommended when each tail group has >= 50 observations.
        'reweight' — kernel density estimate-based smooth quantile matching.
        Preferable for small tail groups or heavily skewed distributions.
    protected_attr :
        Optional label for the protected attribute (used in report keys only).
        Does not change any computation.
    n_quantiles :
        Resolution of the barycenter quantile function grid. Default 500.
        Increase to 2000 for very large portfolios.
    """

    def __init__(
        self,
        quantile_threshold: float = 0.9,
        method: Literal["wasserstein", "reweight"] = "wasserstein",
        protected_attr: str | None = None,
        n_quantiles: int = 500,
    ) -> None:
        if not (0.0 < quantile_threshold < 1.0):
            raise ValueError(
                f"quantile_threshold must be in (0, 1). Got: {quantile_threshold!r}"
            )
        if method not in ("wasserstein", "reweight"):
            raise ValueError(
                f"method must be 'wasserstein' or 'reweight'. Got: {method!r}"
            )
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2. Got: {n_quantiles!r}")

        self.quantile_threshold = quantile_threshold
        self.method = method
        self.protected_attr = protected_attr
        self.n_quantiles = n_quantiles

        # Set after fit()
        self._is_fitted: bool = False
        self._tail_cutoff: float | None = None
        self._groups: np.ndarray | None = None
        self._group_weights: np.ndarray | None = None
        # Per-group tail ECDF: {group -> (ecdf_x, ecdf_y)}
        self._tail_ecdfs: dict = {}
        # Barycenter quantile function: (u_grid, bar_qf)
        self._bar_qf: tuple[np.ndarray, np.ndarray] | None = None
        # Calibration data for report()
        self._fit_y_pred: np.ndarray | None = None
        self._fit_sensitive: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> "TailDemographicParityCorrector":
        """Learn group-conditional tail distributions and the correction mapping.

        Computes the quantile threshold on the overall (pooled) distribution,
        identifies tail observations for each group, and builds the Wasserstein
        barycenter quantile function of the tail distributions.

        Parameters
        ----------
        y_pred :
            1-D array of model predictions (e.g. pure premiums). Must be finite.
        sensitive :
            1-D array of group membership labels. Shape (n,). Any hashable dtype.

        Returns
        -------
        self
        """
        y_pred = np.asarray(y_pred, dtype=float)
        sensitive = np.asarray(sensitive)
        n = len(y_pred)

        if len(sensitive) != n:
            raise ValueError(
                f"y_pred and sensitive must have the same length. "
                f"Got {n} and {len(sensitive)}."
            )
        if n == 0:
            raise ValueError("y_pred must not be empty.")
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("y_pred contains non-finite values (NaN or Inf).")

        # Fit the tail cutoff on the pooled distribution
        self._tail_cutoff = float(np.quantile(y_pred, self.quantile_threshold))

        groups = np.unique(sensitive)
        self._groups = groups

        # Portfolio weights: fraction of each group in the *tail*
        tail_mask = y_pred > self._tail_cutoff
        n_tail = tail_mask.sum()
        group_tail_counts = np.array([
            int((sensitive[tail_mask] == g).sum()) for g in groups
        ], dtype=float)
        total_tail = group_tail_counts.sum()

        if total_tail == 0:
            # No observations in the tail: nothing to fit
            self._group_weights = np.ones(len(groups)) / len(groups)
            self._bar_qf = (np.linspace(0, 1, self.n_quantiles),
                            np.full(self.n_quantiles, self._tail_cutoff))
            self._is_fitted = True
            self._fit_y_pred = y_pred
            self._fit_sensitive = sensitive
            return self

        self._group_weights = group_tail_counts / total_tail

        # Compute per-group tail ECDF (or KDE-based smooth ECDF)
        self._tail_ecdfs = {}
        for g, w in zip(groups, self._group_weights):
            mask = (sensitive == g) & tail_mask
            tail_vals = y_pred[mask]

            if len(tail_vals) == 0:
                # Group has no tail observations: use a degenerate point mass
                # at the cutoff so it contributes a flat quantile function
                ecdf_x = np.array([self._tail_cutoff, self._tail_cutoff + 1e-8])
                ecdf_y = np.array([0.5, 1.0])
                self._tail_ecdfs[g] = (ecdf_x, ecdf_y)
                continue

            if self.method == "wasserstein":
                ecdf_x, ecdf_y = _weighted_ecdf(tail_vals)
            else:  # 'reweight'
                ecdf_x, ecdf_y = _kde_quantile_ecdf(tail_vals, self.n_quantiles)

            self._tail_ecdfs[g] = (ecdf_x, ecdf_y)

        # Build barycenter quantile function
        ecdf_list = [self._tail_ecdfs[g] for g in groups]
        self._bar_qf = _barycenter_qf(ecdf_list, self._group_weights, self.n_quantiles)

        # Store for report()
        self._fit_y_pred = y_pred
        self._fit_sensitive = sensitive
        self._is_fitted = True
        return self

    def transform(
        self,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> np.ndarray:
        """Apply tail OT correction to new predictions.

        Predictions at or below the fitted tail cutoff are returned unchanged.
        Predictions above the cutoff are mapped through T_s = Q_bar ∘ F_s,
        where F_s is the calibration ECDF for group s and Q_bar is the
        Wasserstein barycenter quantile function.

        Parameters
        ----------
        y_pred :
            Predictions to correct. Shape (n,).
        sensitive :
            Group membership labels. Shape (n,). Groups not seen at fit time
            receive the identity map (no correction).

        Returns
        -------
        corrected : np.ndarray, shape (n,)
        """
        self._check_fitted()

        y_pred = np.asarray(y_pred, dtype=float)
        sensitive = np.asarray(sensitive)
        n = len(y_pred)

        if len(sensitive) != n:
            raise ValueError(
                f"y_pred and sensitive must have the same length. "
                f"Got {n} and {len(sensitive)}."
            )

        corrected = y_pred.copy()
        tail_mask = y_pred > self._tail_cutoff
        u_grid, bar_qf = self._bar_qf

        for g, (ecdf_x, ecdf_y) in self._tail_ecdfs.items():
            group_tail = (sensitive == g) & tail_mask
            if not group_tail.any():
                continue
            tail_vals = y_pred[group_tail]
            # Map through F_s to get probability rank in the calibration tail
            u_vals = np.interp(tail_vals, ecdf_x, ecdf_y)
            # Map through Q_bar to get the barycenter quantile
            corrected[group_tail] = np.interp(u_vals, u_grid, bar_qf)

        return corrected

    def fit_transform(
        self,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> np.ndarray:
        """Fit and apply the tail correction in one step.

        Equivalent to ``fit(y_pred, sensitive).transform(y_pred, sensitive)``.
        Use fit() + transform() when you have a held-out test set.

        Parameters
        ----------
        y_pred :
            Predictions. Shape (n,).
        sensitive :
            Group membership labels. Shape (n,).

        Returns
        -------
        corrected : np.ndarray, shape (n,)
        """
        self.fit(y_pred, sensitive)
        return self.transform(y_pred, sensitive)

    def report(self) -> TailDPReport:
        """Produce a diagnostic report from the fitted calibration data.

        Computes pairwise KS statistics before and after correction, mean shift
        per group, and basic coverage statistics.

        Returns
        -------
        report : TailDPReport
        """
        self._check_fitted()

        y_pred = self._fit_y_pred
        sensitive = self._fit_sensitive
        groups = self._groups

        tail_mask = y_pred > self._tail_cutoff
        n_affected = int(tail_mask.sum())
        proportion_affected = float(n_affected / len(y_pred))

        corrected = self.transform(y_pred, sensitive)

        # Pairwise KS statistics across all group pairs, before and after
        ks_before = 0.0
        ks_after = 0.0
        for i, g0 in enumerate(groups):
            for j, g1 in enumerate(groups):
                if j <= i:
                    continue
                m0 = (sensitive == g0) & tail_mask
                m1 = (sensitive == g1) & tail_mask
                if m0.sum() == 0 or m1.sum() == 0:
                    continue
                stat_before, _ = ks_2samp(y_pred[m0], y_pred[m1])
                stat_after, _ = ks_2samp(corrected[m0], corrected[m1])
                ks_before = max(ks_before, float(stat_before))
                ks_after = max(ks_after, float(stat_after))

        # Mean shift per group (tail only)
        mean_shift: dict[str, float] = {}
        for g in groups:
            m = (sensitive == g) & tail_mask
            if m.sum() == 0:
                mean_shift[str(g)] = 0.0
            else:
                mean_shift[str(g)] = float(
                    np.mean(corrected[m]) - np.mean(y_pred[m])
                )

        group_tail_sizes = {
            str(g): int(((sensitive == g) & tail_mask).sum())
            for g in groups
        }

        return TailDPReport(
            quantile_threshold=self.quantile_threshold,
            tail_cutoff=float(self._tail_cutoff),
            n_affected=n_affected,
            proportion_affected=proportion_affected,
            ks_before=ks_before,
            ks_after=ks_after,
            ks_reduction=ks_before - ks_after,
            mean_shift_by_group=mean_shift,
            group_tail_sizes=group_tail_sizes,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tail_cutoff_(self) -> float:
        """The prediction value at the fitted quantile threshold."""
        self._check_fitted()
        return float(self._tail_cutoff)

    @property
    def groups_(self) -> np.ndarray:
        """Unique group labels seen at fit time."""
        self._check_fitted()
        return self._groups.copy()

    @property
    def group_weights_(self) -> dict:
        """Portfolio proportions in the tail for each group."""
        self._check_fitted()
        return {str(g): float(w) for g, w in zip(self._groups, self._group_weights)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "TailDemographicParityCorrector has not been fitted. "
                "Call fit() first."
            )

    def __repr__(self) -> str:
        return (
            f"TailDemographicParityCorrector("
            f"quantile_threshold={self.quantile_threshold}, "
            f"method={self.method!r})"
        )
