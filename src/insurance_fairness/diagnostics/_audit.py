"""
ProxyDiscriminationAudit: the main entry point for proxy discrimination diagnostics.

Orchestrates the full audit pipeline:
  1. Validate inputs
  2. Compute fitted prices and admissible prices
  3. Compute D_proxy with bootstrap CI
  4. Fit surrogate and compute Shapley effects
  5. Compute benchmark premiums (unaware + aware)
  6. Compute per-policyholder local scores
  7. Package results into ProxyDiscriminationResult

Usage::

    import polars as pl
    from insurance_fairness.diagnostics import ProxyDiscriminationAudit

    audit = ProxyDiscriminationAudit(
        model=my_glm,
        X=df,
        y=df["claim_cost"],
        sensitive_col="postcode_area",
        rating_factors=["age_band", "vehicle_group", "ncd_years"],
    )
    result = audit.fit()
    print(f"D_proxy = {result.d_proxy:.4f} ({result.rag})")
    result.to_html("audit_report.html")
    result.to_json("audit_report.json")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from ._admissible import compute_admissible_price, compute_d_proxy_with_ci
from ._benchmarks import BenchmarkPremiums, compute_benchmarks
from ._local import compute_local_scores
from ._shapley import fit_surrogate_and_compute_shapley
from ._utils import (
    d_proxy_rag,
    exposure_weighted_mean,
    phi_rag,
    resolve_exposure,
    subsample_indices,
    validate_columns,
    validate_dataframe,
    validate_model,
    validate_rating_factors,
)


@dataclass
class ShapleyEffect:
    """
    Shapley effect for a single rating factor.

    Attributes
    ----------
    factor:
        Rating factor name.
    phi:
        Normalised Shapley effect in [0, 1]. Sum across all factors = 1.
    phi_monetary:
        phi * d_proxy_monetary. Monetary attribution in premium units.
    rank:
        Rank by phi (1 = most discriminatory factor).
    rag:
        RAG status: 'green' (phi < 0.10), 'amber' (0.10-0.30), 'red' (>= 0.30).
    """

    factor: str
    phi: float
    phi_monetary: float
    rank: int
    rag: str


@dataclass
class ProxyDiscriminationResult:
    """
    Full results from a proxy discrimination audit.

    Attributes
    ----------
    d_proxy:
        Normalised L2-distance from fitted price to admissible price set.
        Scalar in [0, 1]. 0 = no proxy discrimination; 1 = maximal.
    d_proxy_ci:
        Bootstrap 95% confidence interval for d_proxy: (lower, upper).
    d_proxy_monetary:
        d_proxy * exposure-weighted mean premium. Monetary interpretation
        of the discrimination level (in the same units as predictions).
    shapley_effects:
        Dict mapping rating factor name -> ShapleyEffect.
        Ordered by rank (descending phi).
    local_scores:
        Per-policyholder Polars DataFrame with columns:
        policy_id, h, h_star, d_proxy_local, d_proxy_absolute,
        proxy_vulnerability, rag.
    benchmarks:
        BenchmarkPremiums with unaware, aware, proxy_vulnerability arrays.
    rag:
        Overall RAG status based on d_proxy: 'green', 'amber', or 'red'.
    sensitive_col:
        Name of the sensitive attribute used in the audit.
    n_perms:
        Number of permutations used in the Shapley estimator.
    """

    d_proxy: float
    d_proxy_ci: tuple[float, float]
    d_proxy_monetary: float
    shapley_effects: dict[str, ShapleyEffect]
    local_scores: pl.DataFrame
    benchmarks: BenchmarkPremiums
    rag: str
    sensitive_col: str
    n_perms: int

    def to_html(self, path: str | Path) -> None:
        """Write HTML audit report to *path*."""
        from ._report import generate_html_report
        html = generate_html_report(self)
        Path(path).write_text(html, encoding="utf-8")

    def to_json(self, path: str | Path) -> None:
        """Write JSON audit report to *path*."""
        from ._report import generate_json_report
        js = generate_json_report(self)
        Path(path).write_text(js, encoding="utf-8")

    def summary(self) -> str:
        """Return a plain-text summary of the audit results."""
        lines = [
            f"Proxy Discrimination Audit",
            f"  Sensitive attribute : {self.sensitive_col}",
            f"  D_proxy             : {self.d_proxy:.4f} (95% CI: [{self.d_proxy_ci[0]:.4f}, {self.d_proxy_ci[1]:.4f}])",
            f"  D_proxy monetary    : £{self.d_proxy_monetary:.2f}",
            f"  RAG status          : {self.rag.upper()}",
            f"",
            f"  Top discriminatory factors:",
        ]
        for name, se in self.shapley_effects.items():
            lines.append(
                f"    {se.rank}. {name}: phi={se.phi:.4f} (£{se.phi_monetary:.2f}) [{se.rag}]"
            )
        return "\n".join(lines)


class ProxyDiscriminationAudit:
    """
    Proxy discrimination diagnostic audit for insurance pricing models.

    Given a fitted model, a training dataset, and a sensitive attribute,
    measures how much proxy discrimination exists and which rating factors
    drive it. Implements:

    - D_proxy: normalised L2-distance to admissible price set (LRTW 2026)
    - Shapley effects: Owen (2014) permutation estimator via surrogate model
    - Per-policyholder local scores and proxy vulnerability
    - Unaware vs aware premium benchmarks (Côté et al. 2025)

    Parameters
    ----------
    model:
        Fitted pricing model. Must have a predict(X) method returning
        premium predictions. Should have been fitted WITHOUT the sensitive
        attribute (unaware model). sklearn-compatible models work directly;
        pass X as a Polars DataFrame and the model will receive a numpy array.
    X:
        Polars DataFrame containing all rating factors and the sensitive
        attribute column. Must NOT contain the target/outcome column.
    y:
        Observed outcomes (claim costs or frequencies). Polars Series or
        numpy array. Used for context only in v0.1 (not yet used in fitting).
    sensitive_col:
        Name of the protected characteristic column in X.
        Examples: 'gender', 'postcode_area', 'ethnicity_proxy'.
    rating_factors:
        List of legitimate rating factor column names (not including
        sensitive_col). These are the factors for which Shapley effects
        will be computed.
    exposure_col:
        Name of the exposure column in X, if any. Used for exposure-weighted
        statistics. If None, unit exposure is assumed.
    reference_dist:
        Reference distribution for S when computing h_star. Default 'observed'
        uses the empirical distribution of S in the dataset.
    n_perms:
        Number of random permutations for the Owen Shapley estimator.
        Higher values give more accurate Shapley effects but take longer.
        Default 256 gives reasonable accuracy for up to ~20 factors.
    surrogate_model:
        Pre-fitted sklearn model to use as surrogate instead of fitting a
        RandomForestRegressor. Must have been fitted on the discrimination
        residual D = h - h_star. Advanced use only.
    subsample_n:
        Maximum number of policyholders to use for Shapley computation.
        If the dataset is larger, a random subsample of this size is used.
        Default 10,000.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Any,
        X: pl.DataFrame,
        y: np.ndarray | pl.Series,
        sensitive_col: str,
        rating_factors: list[str],
        exposure_col: str | None = None,
        reference_dist: str = "observed",
        n_perms: int = 256,
        surrogate_model: RandomForestRegressor | None = None,
        subsample_n: int = 10_000,
        random_state: int = 42,
    ) -> None:
        validate_model(model)
        validate_dataframe(X, "X")
        validate_columns(X, sensitive_col)
        validate_rating_factors(rating_factors, X, sensitive_col)

        self.model = model
        self.X = X
        self.y = np.asarray(y) if isinstance(y, pl.Series) else np.asarray(y)
        self.sensitive_col = sensitive_col
        self.rating_factors = rating_factors
        self.exposure_col = exposure_col
        self.reference_dist = reference_dist
        self.n_perms = n_perms
        self.surrogate_model = surrogate_model
        self.subsample_n = subsample_n
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def fit(self) -> ProxyDiscriminationResult:
        """
        Run the full proxy discrimination audit.

        Returns
        -------
        ProxyDiscriminationResult with all computed metrics.
        """
        n = len(self.X)
        weights = resolve_exposure(self.X, self.exposure_col, n)

        # 1. Compute fitted prices h from the model using only rating_factors.
        # We pass only the rating_factors to avoid passing the sensitive_col or
        # the exposure column to the model. This assumes the model was trained on
        # rating_factors only (the standard unaware model setup).
        X_model = self.X.select(self.rating_factors)
        h = np.asarray(self.model.predict(X_model.to_numpy()), dtype=float)

        # 2. Compute admissible prices h_star
        s = self.X[self.sensitive_col].to_numpy()
        h_star = compute_admissible_price(h, s, weights, self.reference_dist)

        # 3. Compute D_proxy with bootstrap CI
        d_proxy, d_proxy_ci = compute_d_proxy_with_ci(
            h=h,
            h_star=h_star,
            weights=weights,
            n_bootstrap=200,
            ci_level=0.95,
            rng=np.random.default_rng(self.random_state),
        )

        # D_proxy monetary: d_proxy * mean premium
        mean_premium = exposure_weighted_mean(h, weights)
        d_proxy_monetary = d_proxy * mean_premium

        # 4. Compute Shapley effects (on a subsample for tractability)
        sub_idx = subsample_indices(n, self.subsample_n, self._rng)
        X_sub = self.X.select(self.rating_factors)[sub_idx].to_numpy().astype(float)
        # Discriminatory residual: between-group component = h_star - mu_h
        # This is the component of h that co-varies with S.
        # Shapley decomposes which features drive this between-group variation.
        mu_h = exposure_weighted_mean(h, weights)
        D_sub = (h_star - mu_h)[sub_idx]
        w_sub = weights[sub_idx]

        phi_dict, _ = fit_surrogate_and_compute_shapley(
            X=X_sub,
            D=D_sub,
            weights=w_sub,
            factor_names=self.rating_factors,
            n_perms=self.n_perms,
            surrogate_model=self.surrogate_model,
            random_state=self.random_state,
        )

        # Build ShapleyEffect dataclasses, ranked by phi
        sorted_factors = sorted(phi_dict.items(), key=lambda kv: kv[1], reverse=True)
        shapley_effects: dict[str, ShapleyEffect] = {}
        for rank, (name, phi) in enumerate(sorted_factors, start=1):
            shapley_effects[name] = ShapleyEffect(
                factor=name,
                phi=phi,
                phi_monetary=phi * d_proxy_monetary,
                rank=rank,
                rag=phi_rag(phi),
            )

        # 5. Compute benchmark premiums
        benchmarks = compute_benchmarks(
            model=self.model,
            X=self.X,
            sensitive_col=self.sensitive_col,
            weights=weights,
            random_state=self.random_state,
        )

        # 6. Compute per-policyholder local scores
        local_scores = compute_local_scores(
            h=h,
            h_star=h_star,
            benchmarks=benchmarks,
            policy_ids=np.arange(n),
        )

        # 7. Overall RAG
        rag = d_proxy_rag(d_proxy)

        return ProxyDiscriminationResult(
            d_proxy=d_proxy,
            d_proxy_ci=d_proxy_ci,
            d_proxy_monetary=d_proxy_monetary,
            shapley_effects=shapley_effects,
            local_scores=local_scores,
            benchmarks=benchmarks,
            rag=rag,
            sensitive_col=self.sensitive_col,
            n_perms=self.n_perms,
        )
