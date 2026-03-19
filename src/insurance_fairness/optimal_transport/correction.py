"""Lindholm marginalisation and Wasserstein barycenter correction."""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import polars as pl

from ._utils import (
    apply_ot_correction,
    barycenter_quantile,
    exposure_weighted_ecdf,
    wasserstein_distance_1d,
)
from ._validators import (
    validate_dataframe_aligned,
    validate_epsilon,
    validate_exposure,
    validate_predictions,
    validate_protected_attrs_present,
)


def _concat_xd(X: pl.DataFrame, D: pl.DataFrame) -> pl.DataFrame:
    """Horizontal-concat X and D, dropping any X columns that already appear in D.

    Polars raises DuplicateError if the same column name appears twice in a
    horizontal concat. In the typical usage pattern the model is trained on a
    DataFrame that includes the protected attribute both in X (for feature
    engineering) and in D. We always want D's version to win, so we drop the
    overlap from X before concatenating.
    """
    overlap = [c for c in X.columns if c in D.columns]
    if overlap:
        X = X.drop(overlap)
    return pl.concat([X, D], how="horizontal")


class LindholmCorrector:
    """Discrimination-free price via Lindholm (2022) marginalisation.

    Implements h*(x) = sum_d mu_hat(x, d) * omega_d, where omega_d = P(D=d)
    are the portfolio proportions of the protected attribute.

    The ``log_space`` parameter controls how model outputs are interpreted:
    if True, the model returns predictions on the log scale (e.g. a GLM
    linear predictor) and this class will exponentiate them before arithmetic
    averaging. If False (default), model outputs are already on the natural
    scale (e.g. claim frequencies or pure premiums).

    This is the primary correction for UK insurance fairness compliance.
    It achieves conditional fairness (equal price for equal risk), not
    demographic parity, which is the correct standard under the Equality Act
    and FCA Consumer Duty.

    The model must have been trained with D included as a feature, so that
    mu_hat(x, d) is well-defined for all d in the D domain.
    """

    def __init__(
        self,
        protected_attrs: list[str],
        bias_correction: Literal["proportional", "uniform", "kl"] = "proportional",
        log_space: bool = False,
        d_values: dict[str, list] | None = None,
    ) -> None:
        self.protected_attrs = protected_attrs
        self.bias_correction = bias_correction
        self.log_space = log_space
        self.d_values = d_values or {}
        self._portfolio_weights: dict[str, dict] = {}
        self._bias_correction_factor: float = 1.0
        self._is_fitted = False

    def fit(
        self,
        model_fn: Callable[[pl.DataFrame], np.ndarray],
        X_calib: pl.DataFrame,
        D_calib: pl.DataFrame,
        exposure: np.ndarray | None = None,
        y_obs: np.ndarray | None = None,
    ) -> "LindholmCorrector":
        """Learn portfolio proportions and compute bias correction factor.

        model_fn: callable taking a full DataFrame (X + D columns) and returning
                  a 1-D array of predictions.
        X_calib: non-protected feature columns (may also contain protected cols).
        D_calib: protected attribute columns.
        exposure: per-policy exposure weights. Defaults to ones.
        y_obs: observed losses; required for bias_correction='kl'.
        """
        n = X_calib.shape[0]
        validate_dataframe_aligned(D_calib, "D_calib", n)
        exposure = validate_exposure(exposure, n)
        validate_protected_attrs_present(self.protected_attrs, D_calib, "D_calib")

        # Learn portfolio proportions for each protected attribute
        for attr in self.protected_attrs:
            col = D_calib[attr]
            d_vals = self.d_values.get(attr) or col.unique().to_list()
            weights: dict = {}
            total_exp = exposure.sum()
            for d in d_vals:
                mask = col == d
                mask_arr = mask.to_numpy()
                weights[d] = float(exposure[mask_arr].sum() / total_exp)
            self._portfolio_weights[attr] = weights

        # Compute bias correction factor on calibration data
        XD_calib = _concat_xd(X_calib, D_calib)
        mu_hat = model_fn(XD_calib)
        h_star = self._marginalise(model_fn, X_calib, D_calib)

        if self.bias_correction == "proportional":
            mean_mu = float(np.average(mu_hat, weights=exposure))
            mean_h = float(np.average(h_star, weights=exposure))
            self._bias_correction_factor = mean_mu / mean_h if mean_h > 0 else 1.0

        elif self.bias_correction == "uniform":
            mean_mu = float(np.average(mu_hat, weights=exposure))
            mean_h = float(np.average(h_star, weights=exposure))
            # Stored as additive shift; apply in log-space if log_space=True
            self._bias_correction_additive = mean_mu - mean_h
            self._bias_correction_factor = mean_mu / mean_h if mean_h > 0 else 1.0

        elif self.bias_correction == "kl":
            if y_obs is None:
                raise ValueError("y_obs is required for KL-optimal bias correction")
            self._bias_correction_factor = self._fit_kl_correction(
                model_fn, X_calib, D_calib, exposure, y_obs
            )
        else:
            raise ValueError(f"Unknown bias_correction: {self.bias_correction!r}")

        self._is_fitted = True
        return self

    def _fit_kl_correction(
        self,
        model_fn: Callable[[pl.DataFrame], np.ndarray],
        X_calib: pl.DataFrame,
        D_calib: pl.DataFrame,
        exposure: np.ndarray,
        y_obs: np.ndarray,
    ) -> float:
        """Fit KL-optimal bias correction: P*(d) ∝ exp(beta * zeta(d)).

        Finds beta such that E*[zeta(D)] = mean(y_obs).
        Returns the ratio E[Y] / E[h*(X, beta=beta_opt)] as the scalar correction.
        """
        from scipy.optimize import brentq

        attr = self.protected_attrs[0]
        weights = self._portfolio_weights[attr]
        d_vals = list(weights.keys())

        # zeta(d) = E[mu_hat(X, d)] under empirical X distribution
        zeta: dict = {}
        for d in d_vals:
            D_fixed = D_calib.with_columns(pl.lit(d).alias(attr))
            XD = _concat_xd(X_calib, D_fixed)
            mu_d = model_fn(XD)
            zeta[d] = float(np.average(mu_d, weights=exposure))

        target = float(np.average(y_obs, weights=exposure))

        def objective(beta: float) -> float:
            log_weights = {d: beta * zeta[d] for d in d_vals}
            max_lw = max(log_weights.values())
            unnorm = {d: np.exp(log_weights[d] - max_lw) for d in d_vals}
            total = sum(unnorm.values())
            p_star = {d: unnorm[d] / total for d in d_vals}
            e_star_zeta = sum(p_star[d] * zeta[d] for d in d_vals)
            return e_star_zeta - target

        try:
            beta_opt = brentq(objective, -10.0, 10.0, xtol=1e-6)
        except ValueError:
            # brentq fails if objective doesn't change sign — fall back to proportional
            h_star = self._marginalise(model_fn, X_calib, D_calib)
            mean_h = float(np.average(h_star, weights=exposure))
            return target / mean_h if mean_h > 0 else 1.0

        # Apply KL-optimal weights and compute the scalar correction
        log_weights = {d: beta_opt * zeta[d] for d in d_vals}
        max_lw = max(log_weights.values())
        unnorm = {d: np.exp(log_weights[d] - max_lw) for d in d_vals}
        total = sum(unnorm.values())
        self._kl_portfolio_weights = {attr: {d: unnorm[d] / total for d in d_vals}}

        h_star_kl = self._marginalise(model_fn, X_calib, D_calib, use_kl_weights=True)
        mean_h_kl = float(np.average(h_star_kl, weights=exposure))
        return target / mean_h_kl if mean_h_kl > 0 else 1.0

    def _marginalise(
        self,
        model_fn: Callable[[pl.DataFrame], np.ndarray],
        X: pl.DataFrame,
        D: pl.DataFrame,
        use_kl_weights: bool = False,
    ) -> np.ndarray:
        """Compute h*(x_i) = sum_d mu_hat(x_i, d) * omega_d for all i.

        This is an arithmetic weighted average over the protected attribute
        domain, as required by Lindholm (2022). The ``log_space`` flag only
        controls whether model outputs are exponentiated before averaging
        (i.e. when the model returns log-scale predictions).
        """
        n = X.shape[0]
        h = np.zeros(n)

        for attr in self.protected_attrs:
            if use_kl_weights and hasattr(self, "_kl_portfolio_weights"):
                weights = self._kl_portfolio_weights.get(attr, self._portfolio_weights[attr])
            else:
                weights = self._portfolio_weights[attr]

            for d_val, omega in weights.items():
                if omega == 0.0:
                    continue
                # Replace protected attribute with d_val for all observations
                D_fixed = D.clone().with_columns(pl.lit(d_val).alias(attr))
                XD = _concat_xd(X, D_fixed)
                mu_d = model_fn(XD)
                # Always arithmetic averaging (Lindholm 2022 eq. 3.1).
                # If model outputs are on log-scale, exponentiate first.
                if self.log_space:
                    mu_d = np.exp(mu_d)
                h += omega * mu_d

        return h

    def transform(
        self,
        model_fn: Callable[[pl.DataFrame], np.ndarray],
        X: pl.DataFrame,
        D: pl.DataFrame,
    ) -> np.ndarray:
        """Return discrimination-free predictions h*(x_i), bias-corrected.

        Shape: (n,) — same scale as model output.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")
        validate_protected_attrs_present(self.protected_attrs, D, "D")
        h_star = self._marginalise(model_fn, X, D)
        return h_star * self._bias_correction_factor

    def get_relativities(
        self,
        model_fn: Callable[[pl.DataFrame], np.ndarray],
        X: pl.DataFrame,
        D: pl.DataFrame,
        base_profile: dict,
    ) -> np.ndarray:
        """Return multiplicative relativities versus a base profile.

        base_profile: dict of column name -> value, e.g. {"age_band": "30-39",
        "vehicle_group": 1}.

        In log-space: relativity_i = exp(eta_fair_i - eta_fair_base).
        Compatible with GLM parameter tables.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_relativities()")
        fair_preds = self.transform(model_fn, X, D)

        # Build a single-row DataFrame for the base profile
        base_row = {k: [v] for k, v in base_profile.items()}
        base_X = pl.DataFrame({k: v for k, v in base_row.items() if k not in D.columns})
        base_D = pl.DataFrame({k: v for k, v in base_row.items() if k in D.columns})
        if base_X.is_empty() and not base_D.is_empty():
            base_X = X[:1].clone()
        if base_D.is_empty():
            base_D = D[:1].clone()
        base_fair = self.transform(model_fn, base_X, base_D)
        base_val = float(base_fair[0])
        return fair_preds / base_val

    @property
    def portfolio_weights_(self) -> dict[str, dict]:
        """Fitted portfolio proportions per protected attribute and value."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        return dict(self._portfolio_weights)

    @property
    def bias_correction_factor_(self) -> float:
        """Scalar bias correction applied to h*(X). Should be close to 1.0."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        return self._bias_correction_factor


class WassersteinCorrector:
    """OT barycenter correction for multi-attribute simultaneous fairness.

    Implements m*(x_i) = Q_bar(F_{d_i}(mu_hat(x_i))) where Q_bar is the
    Wasserstein barycenter quantile function across all protected groups.

    This is the *secondary* correction. It achieves demographic parity, not
    conditional fairness. Use Lindholm as the primary correction and this
    only when simultaneous multi-attribute correction is required.
    """

    def __init__(
        self,
        protected_attrs: list[str],
        epsilon: float = 0.0,
        n_quantiles: int = 1000,
        log_space: bool = True,
        exposure_weighted: bool = True,
        method: Literal["sequential"] = "sequential",
    ) -> None:
        validate_epsilon(epsilon)
        self.protected_attrs = protected_attrs
        self.epsilon = epsilon
        self.n_quantiles = n_quantiles
        self.log_space = log_space
        self.exposure_weighted = exposure_weighted
        self.method = method
        self._ecdfs: dict[str, dict] = {}  # attr -> {group -> (ecdf_x, ecdf_y)}
        self._bar_qfs: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # attr -> (u_grid, bar_qf)
        self._group_weights: dict[str, dict] = {}
        self._w2_distances: dict[str, float] = {}
        self._is_fitted = False

    def fit(
        self,
        predictions: np.ndarray,
        D_calib: pl.DataFrame,
        exposure: np.ndarray | None = None,
    ) -> "WassersteinCorrector":
        """Compute per-group ECDFs and barycenter quantile function.

        predictions: mu_hat(x_i) from the base model, shape (n,).
        D_calib: protected attribute columns, shape (n, k).
        exposure: per-policy weights.
        """
        predictions = validate_predictions(predictions)
        n = len(predictions)
        exposure = validate_exposure(exposure, n)
        validate_protected_attrs_present(self.protected_attrs, D_calib, "D_calib")

        if self.log_space:
            preds_for_ecdf = np.log(predictions)
        else:
            preds_for_ecdf = predictions

        for attr in self.protected_attrs:
            col = D_calib[attr].to_numpy()
            groups = np.unique(col)
            ecdfs_attr: dict = {}
            weights_attr: dict = {}

            total_exp = exposure.sum()
            for g in groups:
                mask = col == g
                group_preds = preds_for_ecdf[mask]
                group_exp = exposure[mask] if self.exposure_weighted else np.ones(mask.sum())
                ecdf_x, ecdf_y = exposure_weighted_ecdf(group_preds, group_exp)
                ecdfs_attr[g] = (ecdf_x, ecdf_y)
                weights_attr[g] = float(exposure[mask].sum() / total_exp)

            self._ecdfs[attr] = ecdfs_attr
            self._group_weights[attr] = weights_attr

            # Barycenter quantile function
            ecdf_list = [ecdfs_attr[g] for g in groups]
            w_arr = np.array([weights_attr[g] for g in groups])
            u_grid, bar_qf = barycenter_quantile(ecdf_list, w_arr, self.n_quantiles)
            self._bar_qfs[attr] = (u_grid, bar_qf)

            # W2 distances between pairs of groups (for reporting)
            if len(groups) == 2:
                g0, g1 = groups[0], groups[1]
                mask0 = col == g0
                mask1 = col == g1
                w2 = wasserstein_distance_1d(
                    preds_for_ecdf[mask0],
                    preds_for_ecdf[mask1],
                    exposure[mask0],
                    exposure[mask1],
                )
                self._w2_distances[attr] = w2

        self._is_fitted = True
        return self

    def transform(
        self,
        predictions: np.ndarray,
        D_test: pl.DataFrame,
    ) -> np.ndarray:
        """Apply OT correction: m*(x_i) = Q_bar(F_{d_i}(predictions_i)).

        Sequential method: applies one attribute at a time. Multi-marginal
        is not yet supported and raises NotImplementedError.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")
        predictions = validate_predictions(predictions)
        validate_protected_attrs_present(self.protected_attrs, D_test, "D_test")

        if self.method == "multimarginal":
            raise NotImplementedError(
                "Multi-marginal Wasserstein barycenter is not yet implemented. "
                "Use method='sequential' (default)."
            )

        if self.log_space:
            working = np.log(predictions)
        else:
            working = predictions.copy()

        for attr in self.protected_attrs:
            col = D_test[attr].to_numpy()
            u_grid, bar_qf = self._bar_qfs[attr]
            ecdfs_attr = self._ecdfs[attr]

            corrected = working.copy()
            for g, (ecdf_x, ecdf_y) in ecdfs_attr.items():
                mask = col == g
                if not mask.any():
                    continue
                # F_s(x): map observed value to probability
                u_vals = np.interp(working[mask], ecdf_x, ecdf_y)
                # Q_bar(u): map probability to barycenter quantile
                corrected[mask] = np.interp(u_vals, u_grid, bar_qf)
            working = corrected

        if self.log_space:
            fair_preds = np.exp(working)
        else:
            fair_preds = working

        # Blend with epsilon: 0 = fully corrected, 1 = uncorrected
        if self.epsilon > 0:
            fair_preds = (1.0 - self.epsilon) * fair_preds + self.epsilon * predictions

        return fair_preds

    @property
    def wasserstein_distances_(self) -> dict[str, float]:
        """W2 distance between group distributions prior to correction."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        return dict(self._w2_distances)
