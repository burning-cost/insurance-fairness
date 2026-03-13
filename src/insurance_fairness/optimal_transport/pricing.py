"""Main orchestrator for discrimination-free insurance pricing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import polars as pl

from .causal import CausalGraph, PathDecomposition, PathDecomposer
from .correction import LindholmCorrector, WassersteinCorrector, _concat_xd
from ._validators import validate_exposure


@dataclass
class PricingResult:
    """Output from DiscriminationFreePrice.transform().

    fair_premium: discrimination-free premium for each observation, shape (n,).
    best_estimate: original model output before correction.
    bias_correction_factor: portfolio-level multiplier applied (close to 1.0).
    decomposition: path attribution breakdown if the graph supports it.
    freq_fair: fair frequency component (frequency/severity models only).
    sev_fair: fair severity component (frequency/severity models only).
    method: which correction was applied.
    protected_attrs: list of protected attributes corrected.
    metadata: portfolio weights, W2 distances, and other diagnostics.
    """

    fair_premium: np.ndarray
    best_estimate: np.ndarray
    bias_correction_factor: float
    decomposition: PathDecomposition | None
    freq_fair: np.ndarray | None
    sev_fair: np.ndarray | None
    method: str
    protected_attrs: list[str]
    metadata: dict = field(default_factory=dict)


class DiscriminationFreePrice:
    """Orchestrates causal decomposition and discrimination-free correction.

    Either combined_model_fn or both frequency_model_fn and severity_model_fn
    must be supplied. When using frequency/severity, each component is
    corrected independently and the fair premium is their product.

    The graph is required — it determines which variables are protected,
    proxy, and justified, and drives the path decomposition.

    Usage::

        graph = (CausalGraph()
            .add_protected("gender")
            .add_justified_mediator("claims_history", parents=["gender"])
            .add_proxy("annual_mileage", parents=["gender"])
            .add_outcome("claim_freq"))
        ...

        dfp = DiscriminationFreePrice(
            graph=graph,
            combined_model_fn=my_model,
            correction="lindholm",
        )
        result = dfp.fit_transform(X_train, D_train, exposure=exposure_train)
    """

    def __init__(
        self,
        graph: CausalGraph,
        correction: Literal["lindholm", "wasserstein", "lindholm+wasserstein"] = "lindholm",
        frequency_model_fn: Callable | None = None,
        severity_model_fn: Callable | None = None,
        combined_model_fn: Callable | None = None,
        bias_correction: Literal["proportional", "uniform", "kl"] = "proportional",
        log_space: bool = True,
        epsilon: float = 0.0,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.correction = correction
        self.bias_correction = bias_correction
        self.log_space = log_space
        self.epsilon = epsilon

        # Resolve model functions
        if combined_model_fn is not None:
            self._freq_fn = None
            self._sev_fn = None
            self._combined_fn = combined_model_fn
        elif frequency_model_fn is not None and severity_model_fn is not None:
            self._freq_fn = frequency_model_fn
            self._sev_fn = severity_model_fn
            self._combined_fn = None
        else:
            raise ValueError(
                "Provide either combined_model_fn or both frequency_model_fn and severity_model_fn"
            )

        protected_attrs = graph.get_protected_nodes()
        self._lindholm: LindholmCorrector | None = None
        self._wasserstein: WassersteinCorrector | None = None
        self._lindholm_freq: LindholmCorrector | None = None
        self._lindholm_sev: LindholmCorrector | None = None

        if correction in ("lindholm", "lindholm+wasserstein"):
            if self._combined_fn is not None:
                self._lindholm = LindholmCorrector(
                    protected_attrs,
                    bias_correction=bias_correction,
                    log_space=log_space,
                )
            else:
                self._lindholm_freq = LindholmCorrector(
                    protected_attrs,
                    bias_correction=bias_correction,
                    log_space=log_space,
                )
                self._lindholm_sev = LindholmCorrector(
                    protected_attrs,
                    bias_correction=bias_correction,
                    log_space=log_space,
                )

        if correction in ("wasserstein", "lindholm+wasserstein"):
            self._wasserstein = WassersteinCorrector(
                protected_attrs,
                epsilon=epsilon,
                log_space=log_space,
            )

        self._is_fitted = False

    def fit(
        self,
        X_calib: pl.DataFrame,
        D_calib: pl.DataFrame,
        exposure: np.ndarray | None = None,
        y_freq: np.ndarray | None = None,
        y_sev: np.ndarray | None = None,
        y_combined: np.ndarray | None = None,
    ) -> "DiscriminationFreePrice":
        """Fit correctors on calibration data."""
        n = X_calib.shape[0]
        exposure = validate_exposure(exposure, n)

        XD_calib = _concat_xd(X_calib, D_calib)

        if self._combined_fn is not None:
            if self._lindholm is not None:
                self._lindholm.fit(
                    self._combined_fn,
                    X_calib,
                    D_calib,
                    exposure=exposure,
                    y_obs=y_combined,
                )
            if self._wasserstein is not None:
                preds = self._combined_fn(XD_calib)
                self._wasserstein.fit(preds, D_calib, exposure=exposure)
        else:
            # Frequency
            if self._lindholm_freq is not None:
                self._lindholm_freq.fit(
                    self._freq_fn,
                    X_calib,
                    D_calib,
                    exposure=exposure,
                    y_obs=y_freq,
                )
            # Severity
            if self._lindholm_sev is not None:
                self._lindholm_sev.fit(
                    self._sev_fn,
                    X_calib,
                    D_calib,
                    exposure=exposure,
                    y_obs=y_sev,
                )
            if self._wasserstein is not None:
                freq_preds = self._freq_fn(XD_calib)
                sev_preds = self._sev_fn(XD_calib)
                combined = freq_preds * sev_preds
                self._wasserstein.fit(combined, D_calib, exposure=exposure)

        self._is_fitted = True
        return self

    def transform(
        self,
        X: pl.DataFrame,
        D: pl.DataFrame,
        exposure: np.ndarray | None = None,
    ) -> PricingResult:
        """Compute discrimination-free premiums."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        n = X.shape[0]
        exposure = validate_exposure(exposure, n)
        XD = _concat_xd(X, D)

        protected_attrs = self.graph.get_protected_nodes()
        freq_fair = None
        sev_fair = None

        if self._combined_fn is not None:
            best_est = self._combined_fn(XD)
            if self._lindholm is not None:
                fair = self._lindholm.transform(self._combined_fn, X, D)
                bcf = self._lindholm.bias_correction_factor_
            else:
                fair = best_est.copy()
                bcf = 1.0

            if self._wasserstein is not None:
                fair = self._wasserstein.transform(fair, D)

        else:
            freq_best = self._freq_fn(XD)
            sev_best = self._sev_fn(XD)
            best_est = freq_best * sev_best

            if self._lindholm_freq is not None:
                freq_fair = self._lindholm_freq.transform(self._freq_fn, X, D)
            else:
                freq_fair = freq_best.copy()

            if self._lindholm_sev is not None:
                sev_fair = self._lindholm_sev.transform(self._sev_fn, X, D)
            else:
                sev_fair = sev_best.copy()

            fair = freq_fair * sev_fair

            if self._wasserstein is not None:
                fair = self._wasserstein.transform(fair, D)

            bcf = float(np.average(fair, weights=exposure) / np.average(best_est, weights=exposure))
            if self._lindholm_freq is not None:
                bcf = self._lindholm_freq.bias_correction_factor_

        # Path decomposition (optional — requires graph edges to outcome)
        decomposition = None
        outcome_nodes = self.graph.get_outcome_nodes()
        if outcome_nodes and (self._combined_fn is not None):
            model_fn = self._combined_fn
            d_values = {
                attr: D[attr].unique().to_list()
                for attr in protected_attrs
                if attr in D.columns
            }
            if d_values:
                try:
                    decomposer = PathDecomposer(self.graph, model_fn)
                    decomposition = decomposer.decompose(XD, d_values)
                except Exception:
                    # Path decomposition is best-effort; don't fail main pipeline
                    pass

        # Build metadata
        metadata: dict = {}
        if self._lindholm is not None and self._lindholm._is_fitted:
            metadata["portfolio_weights"] = self._lindholm.portfolio_weights_
        elif self._lindholm_freq is not None and self._lindholm_freq._is_fitted:
            metadata["portfolio_weights"] = self._lindholm_freq.portfolio_weights_
        if self._wasserstein is not None and self._wasserstein._is_fitted:
            metadata["wasserstein_distances"] = self._wasserstein.wasserstein_distances_

        return PricingResult(
            fair_premium=fair,
            best_estimate=best_est,
            bias_correction_factor=bcf,
            decomposition=decomposition,
            freq_fair=freq_fair,
            sev_fair=sev_fair,
            method=self.correction,
            protected_attrs=protected_attrs,
            metadata=metadata,
        )

    def fit_transform(
        self,
        X: pl.DataFrame,
        D: pl.DataFrame,
        exposure: np.ndarray | None = None,
        **y_kwargs,
    ) -> PricingResult:
        """Fit and transform in a single call."""
        self.fit(X, D, exposure=exposure, **y_kwargs)
        return self.transform(X, D, exposure=exposure)
