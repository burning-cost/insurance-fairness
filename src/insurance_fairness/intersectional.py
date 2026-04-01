"""
intersectional.py
-----------------
Intersectional fairness audit and training-time regularisation via distance
covariance (CCdCov).

The fairness gerrymandering problem: a model that satisfies gender parity and
age parity separately is not guaranteed to price young women fairly relative to
both young men and elderly women. The intersectional subgroup — the joint
distribution of protected attributes — can exhibit systematic disparities even
when marginal constraints are satisfied.

This module implements the Concatenated Distance Covariance (CCdCov) approach
from Lee, Antonio, Avanzi, Marchi & Zhou (2025, arXiv:2509.08163):

    CCdCov(ŷ, (s_1,...,s_d)) = Σ_k d̃Cov²(ŷ, s_k) + η(ŷ, s)

The η term captures the intersectional residual — dependence between predictions
and the *joint* protected attribute distribution that is not explained by
individual marginal associations. CCdCov = 0 iff ŷ ⊥ (s_1,...,s_d) jointly.

**Relationship to other modules in this library**:

- :class:`DiscriminationInsensitiveReweighter` handles a single protected
  attribute. This module handles the joint distribution of multiple attributes.
- :class:`LindholmCorrector` post-hoc marginalises over protected attributes.
  CCdCov regularisation modifies model training directly. Use both together:
  CCdCov training followed by Lindholm/multicalibration audit.
- :class:`MulticalibrationAudit` can audit intersectional cells via composite
  group labels, but requires attribute discretisation and a combinatorial grid.
  CCdCov needs no discretisation of continuous attributes.

**Computational complexity note**: distance covariance on a concatenated matrix
S of shape (n, d) uses the dcor library's O(n log n) method for 1D projections
but falls back to O(n²) for multivariate inputs. For n > 50,000 observations,
consider subsampling before calling these classes. A warning is issued
automatically above that threshold.

Usage::

    from insurance_fairness.intersectional import (
        IntersectionalFairnessAudit,
        DistanceCovFairnessRegulariser,
        IntersectionalAuditReport,
        LambdaCalibrationResult,
    )

    # Audit existing predictions
    audit = IntersectionalFairnessAudit(
        protected_attrs=["gender", "age_band"],
        continuous_attrs=["age_band"],    # normalise to [0,1]
    )
    report = audit.audit(y_hat, df_protected)
    print(report.summary())

    # Training-time regulariser
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"],
        continuous_attrs=["age_band"],
        method="ccDcov",
        lambda_val=0.5,
    )
    penalty = reg.penalty(y_hat, df_protected)
    # total_loss = deviance_loss + penalty

References
----------
Lee, H.M., Antonio, K., Avanzi, B., Marchi, L., Zhou, R. (2025). Machine
Learning with Multitype Protected Attributes: Intersectional Fairness through
Regularisation. arXiv:2509.08163.

Székely, G.J., Rizzo, M.L., Bakirov, N.K. (2007). Measuring and Testing
Independence by Correlation of Distances. Annals of Statistics 35(6), 2769–2794.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

# dcor is an optional dependency. Importing at the top level lets the
# ImportError surface immediately when intersectional methods are called,
# rather than silently at class construction time.
_DCOR_AVAILABLE = False
try:
    import dcor as _dcor  # noqa: F401
    _DCOR_AVAILABLE = True
except ImportError:
    pass

_LARGE_N_THRESHOLD = 50_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_dcor() -> Any:
    """Return the dcor module, raising ImportError with a helpful message."""
    if not _DCOR_AVAILABLE:
        raise ImportError(
            "The 'dcor' library is required for intersectional fairness computation. "
            "Install it with:\n\n"
            "    pip install insurance-fairness[intersectional]\n\n"
            "or directly:\n\n"
            "    pip install 'dcor>=0.6'"
        )
    import dcor
    return dcor


def _encode_attributes(
    D: pd.DataFrame,
    protected_attrs: list[str],
    continuous_attrs: list[str],
    fitted_encoders: dict[str, np.ndarray] | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Encode protected attributes into a numeric matrix S of shape (n, d).

    Encoding rules:
    - Categorical (not in continuous_attrs): ordinal integer coding based on
      sorted unique values. NOT one-hot — Euclidean distance on integers is the
      standard for distance covariance on categorical data.
    - Continuous (in continuous_attrs): normalise to [0, 1] using min/max
      scaling to prevent scale dominance in the joint distance matrix.

    Parameters
    ----------
    D :
        DataFrame containing protected attribute columns.
    protected_attrs :
        Ordered list of column names to include.
    continuous_attrs :
        Subset that should be treated as continuous (normalised, not ordinal).
    fitted_encoders :
        Dict mapping column name to either (categories array for ordinal) or
        (min, max array for continuous). Used in transform mode.
    fit :
        If True, learn encoders from D. If False, use fitted_encoders.

    Returns
    -------
    S : np.ndarray, shape (n, d)
        Encoded attribute matrix.
    encoders : dict[str, np.ndarray]
        Fitted encoders for transform-mode use.
    """
    n = len(D)
    cols: list[np.ndarray] = []
    encoders: dict[str, np.ndarray] = fitted_encoders.copy() if fitted_encoders else {}

    for col in protected_attrs:
        vals = D[col].values

        if col in continuous_attrs:
            if fit:
                col_min = float(np.nanmin(vals.astype(float)))
                col_max = float(np.nanmax(vals.astype(float)))
                encoders[col] = np.array([col_min, col_max])
            col_min, col_max = float(encoders[col][0]), float(encoders[col][1])
            rng = col_max - col_min
            if rng < 1e-12:
                normalised = np.zeros(n, dtype=np.float64)
            else:
                normalised = (vals.astype(float) - col_min) / rng
            cols.append(normalised)
        else:
            # Ordinal encoding: assign integers 0, 1, 2, ... in sorted order
            if fit:
                categories = np.array(sorted(set(D[col].dropna().values)))
                encoders[col] = categories
            else:
                categories = encoders[col]

            cat_to_int: dict = {c: i for i, c in enumerate(categories)}
            # Handle unseen values by mapping to max + 1
            n_cats = len(categories)
            encoded = np.array(
                [cat_to_int.get(v, n_cats) for v in vals],
                dtype=np.float64,
            )
            cols.append(encoded)

    if not cols:
        return np.zeros((n, 0), dtype=np.float64), encoders

    S = np.column_stack(cols)
    return S, encoders


def _ccDcov(y_hat: np.ndarray, S: np.ndarray) -> float:
    """
    Compute CCdCov²(ŷ, S) where S is the concatenated attribute matrix.

    This equals d̃Cov²(ŷ, S) when ŷ and S are treated as a single
    (1+d)-dimensional random vector. The decomposition into marginal +
    eta terms is available via _decompose_ccDcov().

    Parameters
    ----------
    y_hat : np.ndarray, shape (n,)
    S : np.ndarray, shape (n, d)
    """
    dcor = _require_dcor()
    return float(dcor.u_distance_covariance_sqr(
        y_hat.reshape(-1, 1).astype(np.float64),
        S.astype(np.float64),
    ))


def _marginal_dcov_sum(y_hat: np.ndarray, S: np.ndarray) -> float:
    """
    Compute Σ_k d̃Cov²(ŷ, s_k) — sum of individual attribute penalties.
    This is the naive multi-attribute penalty (independent auditing).
    """
    dcor = _require_dcor()
    y = y_hat.reshape(-1, 1).astype(np.float64)
    d = S.shape[1]
    total = 0.0
    for k in range(d):
        total += float(dcor.u_distance_covariance_sqr(y, S[:, k : k + 1].astype(np.float64)))
    return total


def _marginal_dcov_per_attr(
    y_hat: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """
    Return d̃Cov²(ŷ, s_k) for each k. Shape: (d,).
    """
    dcor = _require_dcor()
    y = y_hat.reshape(-1, 1).astype(np.float64)
    d = S.shape[1]
    return np.array([
        float(dcor.u_distance_covariance_sqr(y, S[:, k : k + 1].astype(np.float64)))
        for k in range(d)
    ])


def _jdCov(y_hat: np.ndarray, S: np.ndarray) -> float:
    """
    Compute JdCov²(ŷ, s_1,...,s_d).

    JdCov extends dCov to multiple attributes by adding:
    - Σ_k d̃Cov²(ŷ, s_k)  [marginal terms]
    - Σ_{k<l} d̃Cov²(s_k, s_l)  [pairwise attribute terms]

    The pairwise attribute terms introduce instability when attributes are
    themselves correlated — this is JdCov's known limitation. Use CCdCov for
    production work.
    """
    dcor = _require_dcor()
    d = S.shape[1]
    S_f = S.astype(np.float64)
    y = y_hat.reshape(-1, 1).astype(np.float64)

    total = _marginal_dcov_sum(y_hat, S_f)
    for k in range(d):
        for l in range(k + 1, d):
            total += float(dcor.u_distance_covariance_sqr(
                S_f[:, k : k + 1], S_f[:, l : l + 1]
            ))
    return total


def _eta_intersectional(y_hat: np.ndarray, S: np.ndarray) -> float:
    """
    Compute the intersectional residual η(ŷ, s).

    η = CCdCov²(ŷ, S) - Σ_k d̃Cov²(ŷ, s_k)

    η > 0 when predictions depend on the joint attribute distribution beyond
    what marginal associations alone explain. A model that prices young women
    differently from both young men and elderly women will have η > 0 even if
    dCov²(ŷ, gender) ≈ 0 and dCov²(ŷ, age) ≈ 0.

    η can be negative. This happens when the CCdCov estimator is smaller than
    the sum of marginal estimators — a finite-sample artefact of the unbiased
    estimator on small n or when attributes are strongly collinear.
    """
    return _ccDcov(y_hat, S) - _marginal_dcov_sum(y_hat, S)


def _js_divergence(
    y_hat: np.ndarray,
    group_labels: np.ndarray,
    exposure: np.ndarray | None = None,
    n_bins: int = 50,
) -> float:
    """
    Compute D_JS across intersectional subgroups.

    D_JS = Σ_s π_s D_KL( P(Ŷ | S=s) || P(Ŷ) )

    where π_s is the (exposure-weighted) proportion of subgroup s.

    Parameters
    ----------
    y_hat :
        Predicted values, shape (n,).
    group_labels :
        String labels identifying which subgroup each observation belongs to.
        Typically the concatenation of all protected attribute values.
    exposure :
        Optional exposure weights, shape (n,). If None, uniform.
    n_bins :
        Number of histogram bins for approximating distributions.
    """
    from scipy.stats import entropy  # noqa: PLC0415

    if exposure is None:
        exposure = np.ones(len(y_hat), dtype=np.float64)
    exposure = exposure.astype(np.float64)

    y_min, y_max = float(y_hat.min()), float(y_hat.max())
    if y_max - y_min < 1e-12:
        # All predictions identical — perfectly fair by this metric
        return 0.0

    bins = np.linspace(y_min, y_max, n_bins + 1)
    eps = 1e-10

    # Marginal distribution (exposure-weighted)
    marginal_hist, _ = np.histogram(y_hat, bins=bins, weights=exposure)
    marginal_hist = marginal_hist.astype(np.float64) + eps
    marginal_hist /= marginal_hist.sum()

    unique_groups = np.unique(group_labels)
    total_exposure = exposure.sum()
    if total_exposure < 1e-12:
        return 0.0

    js_div = 0.0
    for g in unique_groups:
        mask = group_labels == g
        if mask.sum() == 0:
            continue
        pi_g = float(exposure[mask].sum() / total_exposure)
        group_hist, _ = np.histogram(y_hat[mask], bins=bins, weights=exposure[mask])
        group_hist = group_hist.astype(np.float64) + eps
        group_hist /= group_hist.sum()
        kl = float(entropy(group_hist, marginal_hist))
        js_div += pi_g * kl

    return js_div


def _make_group_labels(D: pd.DataFrame, protected_attrs: list[str]) -> np.ndarray:
    """
    Build string group labels from the Cartesian product of attribute values.
    Returns np.ndarray of str, shape (n,).
    """
    if len(protected_attrs) == 1:
        return D[protected_attrs[0]].astype(str).values
    parts = [D[c].astype(str) for c in protected_attrs]
    labels = parts[0]
    for p in parts[1:]:
        labels = labels + "\u00d7" + p  # × separator
    return labels.values


def _warn_large_n(n: int) -> None:
    if n > _LARGE_N_THRESHOLD:
        warnings.warn(
            f"Dataset has {n:,} observations. Distance covariance on the "
            f"concatenated attribute matrix S is O(n²) and may be slow or "
            f"memory-intensive for n > {_LARGE_N_THRESHOLD:,}. Consider "
            "subsampling before calling audit() or penalty().",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IntersectionalAuditReport:
    """
    Results from :class:`IntersectionalFairnessAudit`.

    Attributes
    ----------
    ccDcov :
        CCdCov²(ŷ, S) — the primary intersectional fairness metric.
        Zero iff predictions are jointly independent of all protected attributes.
    marginal_dcov :
        Dict mapping each protected attribute name to d̃Cov²(ŷ, s_k).
        Sum of these is the naive multi-attribute penalty.
    eta :
        Intersectional residual η = ccDcov - Σ_k marginal_dcov[k].
        Positive values indicate the model exploits intersectional structure
        beyond what marginal associations explain.
    jdCov :
        JdCov²(ŷ, S) for comparison. Includes pairwise d̃Cov²(s_k, s_l)
        terms — can be inflated when protected attributes are correlated.
    js_divergence_overall :
        D_JS across all intersectional subgroups. Lower = fairer.
    js_divergence_by_pair :
        D_JS computed on each pair of protected attributes (two attributes
        at a time). Useful for identifying which attribute pair drives
        intersectional disparity.
    subgroup_statistics :
        DataFrame with columns: subgroup, n, mean_prediction, std_prediction,
        min_prediction, max_prediction, exposure_share. Sorted by mean_prediction.
    n_observations :
        Number of observations in the audit.
    n_subgroups :
        Number of distinct intersectional subgroups.
    protected_attrs :
        List of protected attribute names included in the audit.
    """

    ccDcov: float
    marginal_dcov: dict[str, float]
    eta: float
    jdCov: float
    js_divergence_overall: float
    js_divergence_by_pair: dict[tuple[str, str], float]
    subgroup_statistics: pd.DataFrame
    n_observations: int
    n_subgroups: int
    protected_attrs: list[str]

    def summary(self) -> str:
        """
        Return a concise text summary of the audit results.
        """
        lines = [
            "IntersectionalFairnessAudit",
            "=" * 40,
            f"  n = {self.n_observations:,}  |  {self.n_subgroups} intersectional subgroups",
            f"  Protected attributes: {', '.join(self.protected_attrs)}",
            "",
            "Distance covariance metrics (lower = fairer, 0 = independent):",
            f"  CCdCov²(ŷ, S)       = {self.ccDcov:.6f}",
            f"  Σ marginal dCov²    = {sum(self.marginal_dcov.values()):.6f}",
            f"  η (intersect. resid)= {self.eta:.6f}",
            f"  JdCov²(ŷ, S)        = {self.jdCov:.6f}",
            "",
            "Marginal d̃Cov² per attribute:",
        ]
        for attr, val in self.marginal_dcov.items():
            lines.append(f"    {attr}: {val:.6f}")

        lines += [
            "",
            "Jensen-Shannon divergence (lower = fairer):",
            f"  D_JS overall = {self.js_divergence_overall:.6f}",
        ]
        if self.js_divergence_by_pair:
            lines.append("  D_JS by attribute pair:")
            for (a, b), val in self.js_divergence_by_pair.items():
                lines.append(f"    ({a}, {b}): {val:.6f}")

        lines += [
            "",
            "Subgroup mean predictions (top 5 and bottom 5 by mean):",
        ]
        df = self.subgroup_statistics.sort_values("mean_prediction")
        n_show = min(5, len(df))
        if len(df) > 10:
            display = pd.concat([df.head(n_show), df.tail(n_show)])
        else:
            display = df
        for _, row in display.iterrows():
            lines.append(
                f"    {row['subgroup']:<40s}  "
                f"n={int(row['n']):5d}  "
                f"mean={row['mean_prediction']:.4f}"
            )

        return "\n".join(lines)

    def to_markdown(self, path: str | None = None) -> str:
        """
        Render the audit report as Markdown.

        Parameters
        ----------
        path :
            If provided, write the Markdown to this file path in addition to
            returning the string.
        """
        lines = [
            "# Intersectional Fairness Audit Report",
            "",
            f"**Protected attributes:** {', '.join(self.protected_attrs)}  ",
            f"**Observations:** {self.n_observations:,}  ",
            f"**Intersectional subgroups:** {self.n_subgroups}",
            "",
            "## Distance Covariance Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| CCdCov²(ŷ, S) | {self.ccDcov:.6f} |",
            f"| Σ marginal d̃Cov² | {sum(self.marginal_dcov.values()):.6f} |",
            f"| η (intersectional residual) | {self.eta:.6f} |",
            f"| JdCov²(ŷ, S) | {self.jdCov:.6f} |",
            "",
            "### Marginal d̃Cov² per Attribute",
            "",
            "| Attribute | d̃Cov²(ŷ, sₖ) |",
            "| --- | --- |",
        ]
        for attr, val in self.marginal_dcov.items():
            lines.append(f"| {attr} | {val:.6f} |")

        lines += [
            "",
            "## Jensen-Shannon Divergence",
            "",
            f"**D_JS overall:** {self.js_divergence_overall:.6f}",
            "",
        ]
        if self.js_divergence_by_pair:
            lines.append("| Attribute pair | D_JS |")
            lines.append("| --- | --- |")
            for (a, b), val in self.js_divergence_by_pair.items():
                lines.append(f"| ({a}, {b}) | {val:.6f} |")
            lines.append("")

        lines += [
            "## Subgroup Statistics",
            "",
        ]
        lines.append(
            self.subgroup_statistics.sort_values("mean_prediction")
            .to_markdown(index=False)
        )

        md = "\n".join(lines)
        if path is not None:
            with open(path, "w") as f:
                f.write(md)
        return md


@dataclasses.dataclass
class LambdaCalibrationResult:
    """
    Results from a lambda sweep for :class:`DistanceCovFairnessRegulariser`.

    Stores (lambda, D_JS, validation_loss) triples from a grid sweep and
    identifies the Pareto-efficient operating points.

    Attributes
    ----------
    lambda_values :
        Grid of lambda values tested.
    js_divergence :
        D_JS at each lambda.
    validation_loss :
        Validation loss (e.g., Poisson deviance) at each lambda.
    selected_lambda :
        The lambda selected by the caller. Set via :meth:`select_lambda`.
    method :
        Regularisation method used ('ccDcov', 'jdCov', or 'sum_dcov').
    pareto_indices :
        Indices of the Pareto-efficient points (minimising both loss and D_JS).
    """

    lambda_values: list[float]
    js_divergence: list[float]
    validation_loss: list[float]
    selected_lambda: float
    method: str
    pareto_indices: np.ndarray

    def select_lambda(self, lambda_val: float) -> "LambdaCalibrationResult":
        """
        Record the chosen lambda and return self for chaining.

        The choice of lambda is fully exogenous — it must reflect an explicit
        regulatory or business decision about where to sit on the Pareto front.
        This is intentional: automated selection would obscure the fairness-
        accuracy trade-off from regulators and internal governance.
        """
        return dataclasses.replace(self, selected_lambda=lambda_val)

    def summary(self) -> str:
        lines = [
            f"LambdaCalibrationResult ({self.method})",
            "=" * 40,
            f"  Lambda range: [{min(self.lambda_values):.4g}, {max(self.lambda_values):.4g}]",
            f"  Grid points: {len(self.lambda_values)}",
            f"  Pareto-efficient points: {len(self.pareto_indices)}",
            f"  Selected lambda: {self.selected_lambda:.4g}",
            "",
            "  lambda        D_JS          val_loss",
            "  " + "-" * 44,
        ]
        for lv, djs, vl in zip(
            self.lambda_values, self.js_divergence, self.validation_loss
        ):
            marker = " *" if lv == self.selected_lambda else "  "
            lines.append(f"{marker} {lv:<12.4g}  {djs:<12.6f}  {vl:.6f}")
        lines.append("  (* = selected lambda)")
        return "\n".join(lines)

    def plot(
        self,
        figsize: tuple[float, float] = (10, 4),
        title: str = "Lambda calibration — fairness-accuracy Pareto",
    ) -> Any:
        """
        Plot D_JS and validation loss vs lambda, and the Pareto front.

        Requires the optional ``plot`` extra (matplotlib).

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install insurance-fairness[plot]"
            ) from exc

        lambdas = np.array(self.lambda_values)
        djs = np.array(self.js_divergence)
        loss = np.array(self.validation_loss)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: D_JS vs lambda
        ax1.plot(lambdas, djs, "o-", color="steelblue", linewidth=1.5, markersize=4)
        if self.selected_lambda > 0:
            idx = np.argmin(np.abs(lambdas - self.selected_lambda))
            ax1.axvline(lambdas[idx], color="red", linestyle="--", alpha=0.7,
                        label=f"selected λ={self.selected_lambda:.3g}")
            ax1.legend(fontsize=8)
        ax1.set_xlabel("λ")
        ax1.set_ylabel("D_JS")
        ax1.set_title("Fairness vs regularisation strength")
        ax1.set_xscale("log" if lambdas.min() > 0 else "linear")

        # Panel 2: validation loss vs lambda
        ax2.plot(lambdas, loss, "o-", color="darkorange", linewidth=1.5, markersize=4)
        if self.selected_lambda > 0:
            idx = np.argmin(np.abs(lambdas - self.selected_lambda))
            ax2.axvline(lambdas[idx], color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("λ")
        ax2.set_ylabel("Validation loss")
        ax2.set_title("Accuracy vs regularisation strength")
        ax2.set_xscale("log" if lambdas.min() > 0 else "linear")

        # Panel 3: Pareto front (loss vs D_JS)
        ax3.scatter(djs, loss, color="grey", alpha=0.5, s=20, label="all points")
        if len(self.pareto_indices) > 0:
            pidx = self.pareto_indices
            ax3.scatter(djs[pidx], loss[pidx], color="steelblue", s=50, zorder=5,
                        label="Pareto front")
            ax3.plot(
                djs[pidx][np.argsort(djs[pidx])],
                loss[pidx][np.argsort(djs[pidx])],
                "-", color="steelblue", linewidth=1.2,
            )
        if self.selected_lambda > 0:
            idx = np.argmin(np.abs(lambdas - self.selected_lambda))
            ax3.scatter([djs[idx]], [loss[idx]], color="red", s=80, zorder=6,
                        label=f"selected λ={self.selected_lambda:.3g}")
        ax3.set_xlabel("D_JS (lower = fairer)")
        ax3.set_ylabel("Validation loss (lower = more accurate)")
        ax3.set_title("Fairness-accuracy Pareto front")
        ax3.legend(fontsize=8)

        fig.suptitle(title, y=1.01)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Main classes
# ---------------------------------------------------------------------------


class IntersectionalFairnessAudit:
    """
    Audit intersectional fairness in existing model predictions.

    This class does not modify the model — it produces a structured diagnostic
    report quantifying how much the predictions depend on the joint distribution
    of protected attributes. Use :class:`DistanceCovFairnessRegulariser` if you
    want to penalise this dependence during training.

    Parameters
    ----------
    protected_attrs :
        List of column names identifying protected attributes. Must appear in
        the DataFrame passed to :meth:`audit`.
    continuous_attrs :
        Subset of protected_attrs to treat as continuous. These will be
        normalised to [0, 1] before distance computation. Attributes not in
        this list are treated as categorical and ordinal-encoded.
    exposure_col :
        Column name for exposure weights (e.g., years in force). Used when
        computing exposure-weighted D_JS. If None, uniform weighting.
    n_bins :
        Number of histogram bins for D_JS computation. Default 50.

    Notes
    -----
    Requires the optional ``dcor`` library:
    ``pip install insurance-fairness[intersectional]``

    For n > 50,000 observations, a warning is issued. Consider subsampling
    or using a mini-batch approximation for large portfolios.
    """

    def __init__(
        self,
        protected_attrs: list[str],
        continuous_attrs: list[str] | None = None,
        exposure_col: str | None = None,
        n_bins: int = 50,
    ) -> None:
        if not protected_attrs:
            raise ValueError("protected_attrs must be a non-empty list.")
        self.protected_attrs = list(protected_attrs)
        self.continuous_attrs = list(continuous_attrs) if continuous_attrs else []
        self.exposure_col = exposure_col
        self.n_bins = n_bins

        # Set after audit()
        self._encoders: dict[str, np.ndarray] = {}
        self._is_audited: bool = False

    def audit(self, y_hat: np.ndarray, D: pd.DataFrame) -> IntersectionalAuditReport:
        """
        Compute the full intersectional fairness audit.

        Parameters
        ----------
        y_hat :
            Model predictions, shape (n,). For insurance claim frequency models
            these are predicted claim rates; for classification models they are
            predicted probabilities.
        D :
            DataFrame containing the protected attribute columns. Must include
            all columns named in ``protected_attrs``. May also include
            ``exposure_col`` if specified.

        Returns
        -------
        IntersectionalAuditReport

        Raises
        ------
        ImportError
            If dcor is not installed.
        ValueError
            If required columns are missing, or y_hat length mismatches D.
        """
        _require_dcor()  # fail early with helpful message

        y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
        n = len(y_hat)

        if len(D) != n:
            raise ValueError(
                f"y_hat has {n} observations but D has {len(D)} rows."
            )

        # Validate columns
        missing = [c for c in self.protected_attrs if c not in D.columns]
        if missing:
            raise ValueError(
                f"Columns not found in D: {missing}. "
                f"Available: {list(D.columns)}"
            )
        if self.exposure_col and self.exposure_col not in D.columns:
            raise ValueError(
                f"exposure_col {self.exposure_col!r} not found in D."
            )

        _warn_large_n(n)

        # Encode attributes
        S, self._encoders = _encode_attributes(
            D, self.protected_attrs, self.continuous_attrs, fit=True
        )

        # Core distance covariance metrics
        cc = _ccDcov(y_hat, S)
        marginals_arr = _marginal_dcov_per_attr(y_hat, S)
        marginal_dcov = {
            attr: float(marginals_arr[k])
            for k, attr in enumerate(self.protected_attrs)
        }
        eta = cc - float(marginals_arr.sum())
        jd = _jdCov(y_hat, S)

        # Exposure
        exposure: np.ndarray | None = None
        if self.exposure_col:
            exposure = D[self.exposure_col].values.astype(np.float64)

        # Group labels for JS divergence
        group_labels = _make_group_labels(D, self.protected_attrs)

        js_overall = _js_divergence(y_hat, group_labels, exposure, self.n_bins)

        # Pairwise JS divergences (all pairs of protected attrs)
        js_by_pair: dict[tuple[str, str], float] = {}
        d = len(self.protected_attrs)
        for k in range(d):
            for l in range(k + 1, d):
                pair_attrs = [self.protected_attrs[k], self.protected_attrs[l]]
                pair_labels = _make_group_labels(D, pair_attrs)
                js_by_pair[(self.protected_attrs[k], self.protected_attrs[l])] = (
                    _js_divergence(y_hat, pair_labels, exposure, self.n_bins)
                )

        # Subgroup statistics
        subgroup_stats = self._subgroup_statistics(y_hat, group_labels, exposure)

        self._is_audited = True

        return IntersectionalAuditReport(
            ccDcov=cc,
            marginal_dcov=marginal_dcov,
            eta=eta,
            jdCov=jd,
            js_divergence_overall=js_overall,
            js_divergence_by_pair=js_by_pair,
            subgroup_statistics=subgroup_stats,
            n_observations=n,
            n_subgroups=int(len(np.unique(group_labels))),
            protected_attrs=list(self.protected_attrs),
        )

    def _subgroup_statistics(
        self,
        y_hat: np.ndarray,
        group_labels: np.ndarray,
        exposure: np.ndarray | None,
    ) -> pd.DataFrame:
        """Build per-subgroup summary statistics."""
        total_exposure = exposure.sum() if exposure is not None else float(len(y_hat))
        rows = []
        for g in np.unique(group_labels):
            mask = group_labels == g
            y_g = y_hat[mask]
            n_g = int(mask.sum())
            exp_g = (
                float(exposure[mask].sum()) if exposure is not None else float(n_g)
            )
            rows.append({
                "subgroup": str(g),
                "n": n_g,
                "mean_prediction": float(y_g.mean()),
                "std_prediction": float(y_g.std()),
                "min_prediction": float(y_g.min()),
                "max_prediction": float(y_g.max()),
                "exposure_share": exp_g / total_exposure if total_exposure > 0 else 0.0,
            })
        df = pd.DataFrame(rows).sort_values("mean_prediction").reset_index(drop=True)
        return df

    def __repr__(self) -> str:
        status = "audited" if self._is_audited else "not yet run"
        return (
            f"IntersectionalFairnessAudit("
            f"protected_attrs={self.protected_attrs!r}, "
            f"continuous_attrs={self.continuous_attrs!r}, "
            f"{status})"
        )


class DistanceCovFairnessRegulariser:
    """
    Training-time penalty for intersectional fairness via distance covariance.

    Returns a scalar penalty term that can be added to any predictive loss::

        total_loss = deviance_loss + regulariser.penalty(y_hat, D)

    The regulariser supports three methods:

    - ``'ccDcov'`` (recommended): Penalises the joint dependence of predictions
      on the concatenated protected attribute vector. CCdCov = 0 iff ŷ ⊥
      (s_1,...,s_d) jointly. This is the correct target for intersectional
      demographic parity and the method from Lee et al. (2025).
    - ``'sum_dcov'``: Independent sum Σ_k d̃Cov²(ŷ, s_k). Penalises marginal
      dependence on each attribute separately. A model can have sum_dcov = 0
      while still exploiting intersectional structure.
    - ``'jdCov'``: JdCov includes pairwise d̃Cov²(s_k, s_l) terms. More
      sensitive than sum_dcov but can be inflated when protected attributes
      are correlated, introducing instability.

    Parameters
    ----------
    protected_attrs :
        List of column names identifying protected attributes.
    method :
        Regularisation method. One of 'ccDcov', 'sum_dcov', 'jdCov'.
        Default 'ccDcov'.
    lambda_val :
        Regularisation strength. The penalty returned by :meth:`penalty` is
        already scaled by lambda_val. Select via :meth:`calibrate_lambda`.
    continuous_attrs :
        Subset of protected_attrs to treat as continuous (normalised to [0,1]).
        Attributes not in this list are ordinal-encoded.
    n_bins :
        Number of histogram bins for D_JS computation in calibration.

    Notes
    -----
    Gradient-based training: The penalty is computed from y_hat (predictions),
    not model weights. For neural networks, autodiff will propagate gradients
    through the distance covariance computation automatically if y_hat is a
    differentiable tensor. For CatBoost or sklearn models where this is not
    available, use the penalty as a post-training diagnostic or use the audit
    class to assess the trained model.

    Requires the optional ``dcor`` library:
    ``pip install insurance-fairness[intersectional]``
    """

    _VALID_METHODS = ("ccDcov", "sum_dcov", "jdCov")

    def __init__(
        self,
        protected_attrs: list[str],
        method: Literal["ccDcov", "sum_dcov", "jdCov"] = "ccDcov",
        lambda_val: float = 1.0,
        continuous_attrs: list[str] | None = None,
        n_bins: int = 50,
    ) -> None:
        if not protected_attrs:
            raise ValueError("protected_attrs must be a non-empty list.")
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}. Got: {method!r}"
            )
        if lambda_val < 0:
            raise ValueError(f"lambda_val must be non-negative. Got: {lambda_val}")

        self.protected_attrs = list(protected_attrs)
        self.method = method
        self.lambda_val = lambda_val
        self.continuous_attrs = list(continuous_attrs) if continuous_attrs else []
        self.n_bins = n_bins

        self._encoders: dict[str, np.ndarray] = {}
        self._encoders_fitted: bool = False

    def fit(self, D: pd.DataFrame) -> "DistanceCovFairnessRegulariser":
        """
        Fit attribute encoders on the training data.

        Must be called before :meth:`penalty` if you want consistent encoding
        between training and evaluation. For single-use calls, the encoders
        are fitted automatically on the first call to :meth:`penalty`.

        Parameters
        ----------
        D :
            DataFrame containing protected attribute columns.

        Returns
        -------
        self
        """
        _, self._encoders = _encode_attributes(
            D, self.protected_attrs, self.continuous_attrs, fit=True
        )
        self._encoders_fitted = True
        return self

    def penalty(self, y_hat: np.ndarray, D: pd.DataFrame) -> float:
        """
        Compute lambda * ψ(ŷ, S) — the scaled fairness penalty.

        Parameters
        ----------
        y_hat :
            Predictions, shape (n,).
        D :
            DataFrame with protected attribute columns, shape (n, ...).

        Returns
        -------
        float
            The penalty term. Add to your predictive loss during training.
        """
        _require_dcor()
        y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
        n = len(y_hat)

        if len(D) != n:
            raise ValueError(
                f"y_hat has {n} observations but D has {len(D)} rows."
            )

        _warn_large_n(n)

        fit = not self._encoders_fitted
        S, self._encoders = _encode_attributes(
            D, self.protected_attrs, self.continuous_attrs,
            fitted_encoders=self._encoders if not fit else None,
            fit=fit,
        )
        if not fit:
            pass
        else:
            self._encoders_fitted = True

        raw = self._raw_penalty(y_hat, S)
        return self.lambda_val * raw

    def _raw_penalty(self, y_hat: np.ndarray, S: np.ndarray) -> float:
        """Compute unscaled penalty (before lambda multiplication)."""
        if self.method == "ccDcov":
            return _ccDcov(y_hat, S)
        if self.method == "sum_dcov":
            return _marginal_dcov_sum(y_hat, S)
        if self.method == "jdCov":
            return _jdCov(y_hat, S)
        raise ValueError(f"Unknown method: {self.method!r}")

    def js_divergence(
        self,
        y_hat: np.ndarray,
        D: pd.DataFrame,
        exposure: np.ndarray | None = None,
    ) -> float:
        """
        Compute D_JS across all intersectional subgroups.

        This is the primary fairness diagnostic for model comparison and
        lambda selection. It does not require dcor.

        Parameters
        ----------
        y_hat :
            Predictions, shape (n,).
        D :
            DataFrame with protected attribute columns.
        exposure :
            Optional exposure weights, shape (n,).

        Returns
        -------
        float
            Exposure-weighted average KL divergence between each subgroup's
            prediction distribution and the marginal distribution.
        """
        y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
        group_labels = _make_group_labels(D, self.protected_attrs)
        return _js_divergence(y_hat, group_labels, exposure, self.n_bins)

    def calibrate_lambda(
        self,
        model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        X_train: np.ndarray,
        D_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray,
        D_val: pd.DataFrame,
        y_val: np.ndarray,
        lambda_grid: list[float] | None = None,
        loss: Literal["poisson", "mse", "binary_crossentropy"] = "poisson",
        exposure_val: np.ndarray | None = None,
    ) -> LambdaCalibrationResult:
        """
        Sweep a lambda grid and record the fairness-accuracy Pareto.

        The caller provides a ``model_fn`` that trains a model given training
        data and sample weights, and returns predictions on the validation set.
        This design is intentionally framework-agnostic.

        Parameters
        ----------
        model_fn :
            Callable with signature
            ``model_fn(X_train, y_train, D_train, lambda_val) -> y_val_pred``.
            It should train the model with this regulariser at ``lambda_val``
            and return predictions on the validation set.
        X_train, X_val :
            Feature matrices, shape (n_train, p) and (n_val, p).
        D_train, D_val :
            DataFrames with protected attribute columns.
        y_train, y_val :
            Target arrays.
        lambda_grid :
            List of lambda values to evaluate. If None, a log-spaced grid of
            10 values from 1e-4 to 10 is used.
        loss :
            Validation loss function for measuring accuracy.
        exposure_val :
            Optional exposure weights for D_JS computation on the validation set.

        Returns
        -------
        LambdaCalibrationResult
            Contains all (lambda, D_JS, loss) triples and the Pareto indices.
            Call :meth:`LambdaCalibrationResult.select_lambda` to record your
            chosen lambda.
        """
        if lambda_grid is None:
            lambda_grid = list(np.logspace(-4, 1, 10))

        lambda_values = []
        js_divergences = []
        val_losses = []

        for lv in lambda_grid:
            reg_copy = DistanceCovFairnessRegulariser(
                protected_attrs=self.protected_attrs,
                method=self.method,
                lambda_val=lv,
                continuous_attrs=self.continuous_attrs,
                n_bins=self.n_bins,
            )
            y_pred_val = model_fn(X_train, y_train, D_train, lv)
            y_pred_val = np.asarray(y_pred_val, dtype=np.float64).ravel()

            vl = _compute_loss(y_val, y_pred_val, loss)
            djs = reg_copy.js_divergence(y_pred_val, D_val, exposure_val)

            lambda_values.append(float(lv))
            js_divergences.append(float(djs))
            val_losses.append(float(vl))

        pareto_idx = _pareto_indices(
            np.array(js_divergences), np.array(val_losses)
        )

        return LambdaCalibrationResult(
            lambda_values=lambda_values,
            js_divergence=js_divergences,
            validation_loss=val_losses,
            selected_lambda=0.0,
            method=self.method,
            pareto_indices=pareto_idx,
        )

    def __repr__(self) -> str:
        fitted = "fitted" if self._encoders_fitted else "unfitted"
        return (
            f"DistanceCovFairnessRegulariser("
            f"protected_attrs={self.protected_attrs!r}, "
            f"method={self.method!r}, "
            f"lambda_val={self.lambda_val}, "
            f"{fitted})"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: str,
) -> float:
    """Compute a scalar validation loss."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if loss == "poisson":
        eps = 1e-10
        y_pred = np.maximum(y_pred, eps)
        # Normalised Poisson deviance
        return float(2.0 * np.mean(y_pred - y_true - y_true * np.log(y_pred / np.maximum(y_true, eps))))

    if loss == "mse":
        return float(np.mean((y_true - y_pred) ** 2))

    if loss == "binary_crossentropy":
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    raise ValueError(f"Unknown loss: {loss!r}. Expected 'poisson', 'mse', or 'binary_crossentropy'.")


def _pareto_indices(objectives_1: np.ndarray, objectives_2: np.ndarray) -> np.ndarray:
    """
    Return indices of Pareto-efficient points (minimising both objectives).

    A point is Pareto-efficient if no other point is weakly better on both
    objectives and strictly better on at least one.
    """
    n = len(objectives_1)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = (
            (objectives_1 <= objectives_1[i]) &
            (objectives_2 <= objectives_2[i]) &
            ((objectives_1 < objectives_1[i]) | (objectives_2 < objectives_2[i]))
        )
        dominated[i] = False
        is_pareto[dominated] = False
    return np.where(is_pareto)[0]
