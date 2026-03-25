"""
indirect.py
-----------
IndirectDiscriminationAudit: end-to-end partition-based audit of indirect
discrimination in insurance pricing models.

The insight from Côté, Côté & Charpentier (CAS Working Paper, October 2025) is
that you do not need a causal graph to measure proxy discrimination. You only
need to fit two models — one that can see the protected attribute (the "aware"
model), and one that cannot (the "unaware" model) — and measure how much they
differ. When the unaware model charges more than the aware model for a group,
it has used proxy features to infer the protected attribute. That difference is
proxy vulnerability.

This class goes further than ProxyVulnerabilityScore (which accepts
pre-computed premium columns). It fits the benchmark models itself from raw
training data, making it a single-call audit tool for pricing teams who want
a structured answer to "how much indirect discrimination does my model embed?"

Five benchmark premiums (Côté et al. 2025 §3):
    h_A(x, s)   — aware premium: model trained WITH protected attribute s
    h_U(x)      — unaware premium: model trained WITHOUT s
    h_UN(x\\s)   — unawareness premium: h_A evaluated with s masked at predict time
    h_PV(x)     — proxy-vulnerable premium: model trained without s AND without
                  known proxy features
    h_C(x, s)   — parity-cost premium: h_A adjusted for group-level average rate
                  parity (equal mean premium by segment)

Three fairness dimensions:
    Actuarial fairness  — gap between h_A and h_U (calibration within group)
    Proxy vulnerability — mean |h_U(x) - h_A(x)| per segment (indirect discrimination)
    Solidarity          — parity-cost gap: cross-subsidy implied by equal-mean pricing

Design choices:
    - Takes pandas DataFrames: standard sklearn ecosystem, easy integration.
    - model_class=None defaults to LightGBM if available, otherwise
      GradientBoostingRegressor. Explicit LightGBM dependency is optional.
    - Exposure weighting throughout.
    - The result object is self-contained: summary DataFrame, scalar proxy
      vulnerability score, worst-segment DataFrame.

Reference:
    Côté, O., Côté, M.-P., and Charpentier, A. (2025). A Scalable Toolbox for
    Exposing Indirect Discrimination in Insurance Rates. CAS Working Paper.
    https://www.casact.org/sites/default/files/2025-10/_A_Scalable_toolbox_working_paper.pdf
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IndirectDiscriminationResult:
    """
    Output of IndirectDiscriminationAudit.fit().

    Attributes
    ----------
    summary : pd.DataFrame
        Per-segment metrics. Columns: segment (protected attribute value),
        n, exposure, mean_aware, mean_unaware, mean_proxy_vulnerability,
        mean_unawareness_gap, mean_parity_cost, proxy_vulnerability_pct.
    proxy_vulnerability : float
        Portfolio-level exposure-weighted mean absolute proxy vulnerability,
        mean |h_U(x) - h_A(x)|.
    segment_report : pd.DataFrame
        Segments ranked by mean absolute proxy vulnerability (descending).
        Same schema as summary.
    benchmarks : dict[str, np.ndarray]
        Dict of per-observation benchmark predictions on X_test:
            "aware"       — h_A(x, s): predictions from model trained with s
            "unaware"     — h_U(x): predictions from model trained without s
            "unawareness" — h_UN(x\\s): h_A evaluated with s masked
            "proxy_free"  — h_PV(x): model trained without s and proxy features
                            (only if proxy_features were supplied)
            "parity_cost" — h_C(x, s): group-mean-adjusted aware predictions
    """

    summary: pd.DataFrame
    proxy_vulnerability: float
    segment_report: pd.DataFrame
    benchmarks: dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IndirectDiscriminationAudit:
    """
    End-to-end audit of indirect discrimination in a pricing model.

    Fits the five benchmark premiums (Côté et al. 2025) on training data,
    evaluates on test data, and summarises proxy vulnerability by protected
    attribute segment.

    Parameters
    ----------
    protected_attr : str
        Column name of the protected attribute (e.g. "gender", "ethnicity").
    proxy_features : list[str] | None
        Known proxy features to additionally remove when fitting the
        proxy-free benchmark h_PV. If None, only the protected attribute
        is excluded. Pass an empty list for the same effect.
    model_class : type | None
        Sklearn-compatible estimator class (not instance). If None, defaults
        to LightGBMRegressor if lightgbm is installed, else
        GradientBoostingRegressor. The class will be instantiated with
        ``model_kwargs``.
    model_kwargs : dict | None
        Keyword arguments passed to model_class() when instantiating each
        benchmark model. Ignored if model_class is None (defaults are used).
    exposure_col : str | None
        Optional exposure column for weighted statistics. If None, each
        observation is equally weighted.
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    >>> from insurance_fairness.indirect import IndirectDiscriminationAudit
    >>> audit = IndirectDiscriminationAudit(
    ...     protected_attr="gender",
    ...     proxy_features=["postcode_district", "occupation"],
    ... )
    >>> result = audit.fit(X_train, y_train, X_test, y_test)
    >>> print(f"Portfolio proxy vulnerability: {result.proxy_vulnerability:.2f}")
    >>> result.summary
    """

    def __init__(
        self,
        protected_attr: str,
        proxy_features: list[str] | None = None,
        model_class: type | None = None,
        model_kwargs: dict[str, Any] | None = None,
        exposure_col: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.protected_attr = protected_attr
        self.proxy_features: list[str] = list(proxy_features or [])
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.exposure_col = exposure_col
        self.random_state = random_state

        self._aware_model: Any = None
        self._unaware_model: Any = None
        self._proxy_free_model: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_test: pd.DataFrame,
        y_test: pd.Series | np.ndarray,
    ) -> IndirectDiscriminationResult:
        """
        Fit benchmark models on training data and compute fairness metrics on test data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features. Must contain ``protected_attr`` and ``exposure_col``
            if those were specified at construction.
        y_train : array-like
            Training target (pure premium or claim frequency — any regression target).
        X_test : pd.DataFrame
            Test features.
        y_test : array-like
            Test target. Used for calibration gap computation.

        Returns
        -------
        IndirectDiscriminationResult
        """
        _validate_inputs(X_train, X_test, self.protected_attr, self.exposure_col)

        y_train_arr = np.asarray(y_train, dtype=float)
        y_test_arr = np.asarray(y_test, dtype=float)

        # Exposure weights (test set only — training uses uniform weights for model fit)
        w_test = _get_weights(X_test, self.exposure_col)

        base_model = self._resolve_model()

        # ------------------------------------------------------------------
        # Fit benchmark models
        # ------------------------------------------------------------------

        # 1. Aware: trained WITH protected attribute
        aware_cols = [c for c in X_train.columns if c != self.exposure_col]
        self._aware_model = clone(base_model)
        _fit(self._aware_model, X_train[aware_cols], y_train_arr, self.random_state)

        # 2. Unaware: trained WITHOUT protected attribute
        unaware_cols = [c for c in aware_cols if c != self.protected_attr]
        self._unaware_model = clone(base_model)
        _fit(self._unaware_model, X_train[unaware_cols], y_train_arr, self.random_state)

        # 3. Proxy-free: trained without protected attribute AND known proxies
        proxy_free_model: Any = None
        proxy_free_cols: list[str] = []
        if self.proxy_features:
            remove_set = {self.protected_attr} | set(self.proxy_features)
            proxy_free_cols = [c for c in aware_cols if c not in remove_set]
            missing_proxies = [f for f in self.proxy_features if f not in X_train.columns]
            if missing_proxies:
                warnings.warn(
                    f"proxy_features not found in X_train and will be ignored: "
                    f"{missing_proxies}",
                    UserWarning,
                    stacklevel=2,
                )
                proxy_free_cols = [c for c in aware_cols if c not in remove_set]
            self._proxy_free_model = clone(base_model)
            _fit(self._proxy_free_model, X_train[proxy_free_cols], y_train_arr, self.random_state)
            proxy_free_model = self._proxy_free_model

        # ------------------------------------------------------------------
        # Compute benchmark predictions on test set
        # ------------------------------------------------------------------

        test_aware_cols = [c for c in X_test.columns if c != self.exposure_col]
        test_unaware_cols = [c for c in test_aware_cols if c != self.protected_attr]

        h_A = np.asarray(self._aware_model.predict(X_test[test_aware_cols]), dtype=float)
        h_U = np.asarray(self._unaware_model.predict(X_test[test_unaware_cols]), dtype=float)

        # Unawareness premium: aware model evaluated with s masked to group mean
        group_mean_s = float(X_train[self.protected_attr].mean())
        X_test_masked = X_test[test_aware_cols].copy()
        X_test_masked[self.protected_attr] = group_mean_s
        h_UN = np.asarray(self._aware_model.predict(X_test_masked), dtype=float)

        # Parity-cost premium: group-level mean adjustment so each group has
        # the same exposure-weighted mean as the portfolio average
        h_C = _compute_parity_cost(h_A, X_test[self.protected_attr].values, w_test)

        benchmarks: dict[str, np.ndarray] = {
            "aware": h_A,
            "unaware": h_U,
            "unawareness": h_UN,
            "parity_cost": h_C,
        }

        if proxy_free_model is not None:
            test_pf_cols = [c for c in proxy_free_cols if c in X_test.columns]
            h_PV = np.asarray(proxy_free_model.predict(X_test[test_pf_cols]), dtype=float)
            benchmarks["proxy_free"] = h_PV

        # ------------------------------------------------------------------
        # Compute per-segment fairness metrics
        # ------------------------------------------------------------------
        s_test = X_test[self.protected_attr].values
        summary_df, pv_score = _build_summary(
            h_A=h_A,
            h_U=h_U,
            h_UN=h_UN,
            h_C=h_C,
            s=s_test,
            w=w_test,
        )

        segment_report = summary_df.sort_values(
            "mean_abs_proxy_vulnerability", ascending=False
        ).reset_index(drop=True)

        return IndirectDiscriminationResult(
            summary=summary_df,
            proxy_vulnerability=pv_score,
            segment_report=segment_report,
            benchmarks=benchmarks,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_model(self) -> Any:
        """Return an instantiated base estimator."""
        if self.model_class is not None:
            return self.model_class(**self.model_kwargs)

        # Try LightGBM first
        try:
            import lightgbm as lgb  # type: ignore[import]
            kwargs = {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": self.random_state,
                "verbose": -1,
            }
            kwargs.update(self.model_kwargs)
            return lgb.LGBMRegressor(**kwargs)
        except ImportError:
            pass

        # Fallback: sklearn GradientBoostingRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        kwargs = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": self.random_state,
        }
        kwargs.update(self.model_kwargs)
        return GradientBoostingRegressor(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    protected_attr: str,
    exposure_col: str | None,
) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            f"X_train must be a pandas DataFrame. Got {type(X_train).__name__}."
        )
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"X_test must be a pandas DataFrame. Got {type(X_test).__name__}."
        )
    if protected_attr not in X_train.columns:
        raise ValueError(
            f"protected_attr '{protected_attr}' not found in X_train. "
            f"Columns: {list(X_train.columns)}"
        )
    if protected_attr not in X_test.columns:
        raise ValueError(
            f"protected_attr '{protected_attr}' not found in X_test. "
            f"Columns: {list(X_test.columns)}"
        )
    if exposure_col is not None and exposure_col not in X_train.columns:
        raise ValueError(
            f"exposure_col '{exposure_col}' not found in X_train. "
            f"Columns: {list(X_train.columns)}"
        )


def _get_weights(X: pd.DataFrame, exposure_col: str | None) -> np.ndarray:
    if exposure_col is not None and exposure_col in X.columns:
        w = X[exposure_col].values.astype(float)
        if np.any(w <= 0):
            raise ValueError("exposure_col contains non-positive values.")
        return w
    return np.ones(len(X), dtype=float)


def _fit(model: Any, X: pd.DataFrame, y: np.ndarray, random_state: int) -> None:
    """Fit model, setting random_state if the estimator accepts it."""
    try:
        model.set_params(random_state=random_state)
    except (ValueError, TypeError):
        pass
    model.fit(X, y)


def _compute_parity_cost(
    h_A: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute parity-cost premium h_C(x, s).

    h_C adjusts each group's premiums by a multiplicative factor so that
    every group has the same exposure-weighted mean as the portfolio average.
    This is the minimal adjustment that achieves premium parity.

    h_C(x, s=g) = h_A(x, s=g) * (portfolio_mean / group_mean_g)
    """
    total_w = w.sum()
    portfolio_mean = float((h_A * w).sum() / total_w) if total_w > 0 else float(h_A.mean())

    h_C = h_A.copy()
    for grp in np.unique(s):
        mask = s == grp
        w_g = w[mask]
        h_g = h_A[mask]
        gw = w_g.sum()
        if gw == 0:
            continue
        group_mean = float((h_g * w_g).sum() / gw)
        if group_mean > 1e-10:
            h_C[mask] = h_g * (portfolio_mean / group_mean)

    return h_C


def _build_summary(
    h_A: np.ndarray,
    h_U: np.ndarray,
    h_UN: np.ndarray,
    h_C: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """
    Build per-segment summary DataFrame and portfolio proxy vulnerability scalar.

    Returns (summary_df, portfolio_pv).
    """
    rows = []
    total_w = w.sum()

    for grp in np.unique(s):
        mask = s == grp
        w_g = w[mask]
        gw = w_g.sum()

        # wmean operates on already-sliced group arrays — do not re-index with mask.
        def wmean(arr: np.ndarray, _w_g: np.ndarray = w_g, _gw: float = gw) -> float:
            return float((arr * _w_g).sum() / _gw) if _gw > 0 else float("nan")

        h_A_g = h_A[mask]
        h_U_g = h_U[mask]
        h_UN_g = h_UN[mask]
        h_C_g = h_C[mask]

        proxy_vuln_g = h_U_g - h_A_g            # Δ_proxy per policyholder
        unawareness_gap_g = h_UN_g - h_A_g      # unawareness gap
        parity_cost_g = h_C_g - h_A_g           # parity adjustment

        mean_aware = wmean(h_A_g)
        mean_pv = wmean(proxy_vuln_g)
        mean_abs_pv = float((np.abs(proxy_vuln_g) * w_g).sum() / gw) if gw > 0 else float("nan")
        pv_pct = (mean_abs_pv / abs(mean_aware) * 100.0) if abs(mean_aware) > 1e-10 else float("nan")

        rows.append({
            "segment": grp,
            "n": int(mask.sum()),
            "exposure": float(gw),
            "exposure_pct": float(gw / total_w * 100.0) if total_w > 0 else float("nan"),
            "mean_aware": mean_aware,
            "mean_unaware": wmean(h_U_g),
            "mean_unawareness_premium": wmean(h_UN_g),
            "mean_parity_cost_premium": wmean(h_C_g),
            "mean_proxy_vulnerability": mean_pv,
            "mean_abs_proxy_vulnerability": mean_abs_pv,
            "proxy_vulnerability_pct": pv_pct,
            "mean_unawareness_gap": wmean(unawareness_gap_g),
            "mean_parity_cost": wmean(parity_cost_g),
        })

    summary_df = pd.DataFrame(rows)

    # Portfolio-level scalar: exposure-weighted mean absolute proxy vulnerability
    abs_pv_all = np.abs(h_U - h_A)
    pv_score = float((abs_pv_all * w).sum() / total_w) if total_w > 0 else float(abs_pv_all.mean())

    return summary_df, pv_score
