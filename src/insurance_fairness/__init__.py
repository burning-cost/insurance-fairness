"""
insurance-fairness
==================

Proxy discrimination auditing for UK insurance pricing models.

Implements fairness diagnostics and audit reporting aligned with:
- FCA Consumer Duty (PRIN 2A, live July 2023)
- FCA TR24/2 (Pricing Practices Thematic Review, 2024)
- Equality Act 2010, Section 19 (Indirect Discrimination)
- ICOBS pricing practices rules

The primary entry point is :class:`FairnessAudit`, which runs a full audit
of a CatBoost pricing model and returns a structured :class:`FairnessReport`.

For proxy detection without a full audit, use :func:`detect_proxies` directly::

    from insurance_fairness import detect_proxies, ProxyDetectionResult

    result = detect_proxies(
        df=df,
        protected_col="gender",
        factor_cols=["age_band", "vehicle_group", "occupation"],
    )
    result.summary()

For multi-objective Pareto optimisation of the fairness-accuracy trade-off,
see :class:`~insurance_fairness.pareto.NSGA2FairnessOptimiser` and
:class:`~insurance_fairness.pareto.ParetoResult` in the ``pareto`` module.
These require the optional ``pymoo`` dependency
(``pip install insurance-fairness[pareto]``).

v0.3.0 adds two subpackages:

**insurance_fairness.optimal_transport** — discrimination-free pricing via
Lindholm marginalisation, causal path decomposition, and Wasserstein barycenter
correction::

    from insurance_fairness.optimal_transport import (
        CausalGraph,
        DiscriminationFreePrice,
        FCAReport,
    )

**insurance_fairness.diagnostics** — proxy discrimination diagnostics with
D_proxy scalar, Shapley attribution, and per-policyholder vulnerability scores::

    from insurance_fairness.diagnostics import ProxyDiscriminationAudit

v0.3.7 adds :class:`MulticalibrationAudit` — audit and correct pricing models
for multicalibration fairness (Denuit, Michaelides & Trufin, 2026)::

    from insurance_fairness import MulticalibrationAudit

    audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
    report = audit.audit(y_true, y_pred, protected, exposure)
    corrected = audit.correct(y_pred, protected, report, exposure)

v0.3.8 adds :class:`PrivatizedFairnessAudit` — discrimination-free pricing
when protected attributes are privatised via local differential privacy or
estimated from proxies (Zhang, Liu & Shi, 2025)::

    from insurance_fairness import PrivatizedFairnessAudit

    audit = PrivatizedFairnessAudit(
        n_groups=2,
        epsilon=2.0,                          # LDP budget (pi ~ 0.88)
        reference_distribution="uniform",     # equal group weighting
        loss="poisson",
    )
    audit.fit(X, Y, S)                        # S = privatised attribute
    fair_premium = audit.predict_fair_premium(X_new)
    report = audit.audit_report()

v0.4.0 adds per-policyholder proxy vulnerability metrics from Côté et al. (2025):

**insurance_fairness.proxy_vulnerability** — ProxyVulnerabilityScore,
risk spread, parity cost, fairness range, and post-pricing implied propensity::

    from insurance_fairness import (
        ProxyVulnerabilityScore,
        ProxyVulnerabilityResult,
        PremiumSpectrum,
        compute_post_pricing_metrics,
        partition_by_proxy_vulnerability,
    )

v0.5.0 adds :class:`MarginalFairnessPremium` — Stage 2 fairness correction for
distortion risk measure premiums (Huang & Pesenti, 2025, arXiv:2505.18895).
Adjusts Expected Shortfall, Wang transform, or custom distortion risk measures
to be marginally fair with respect to protected attributes. Closed-form
correction with no iterative solver required::

    from insurance_fairness import MarginalFairnessPremium, MarginalFairnessReport

    mfp = MarginalFairnessPremium(distortion='es_alpha', alpha=0.75)
    mfp.fit(Y_train, D_train, X_train, model=glm, protected_indices=[0])
    rho_fair = mfp.transform(Y_test, D_test, X_test)

    report = mfp.sensitivity_report()
    print(f"Baseline ES0.75: {report.rho_baseline:.4f}")
    print(f"Fair ES0.75:     {report.rho_fair:.4f}")

v0.6.0 adds :class:`DoubleFairnessAudit` — joint action and outcome Pareto
optimisation (Bian, Wang, Shi, Qi, 2026, arXiv:2601.19186).

The key distinction this unlocks: action fairness (Delta_1) measures equal
treatment at pricing time; outcome fairness (Delta_2) measures whether the
product delivers equivalent value to each group after the policy is live.
The FCA's Consumer Duty Outcome 4 (Price and Value) requires the latter. The
empirical result from the paper: equalising premiums across gender (Delta_1=0)
does NOT equalise loss ratios (Delta_2 remains large). A firm auditing only
action fairness may still fail Consumer Duty.

``DoubleFairnessAudit`` recovers the full Pareto front via lexicographic
Tchebycheff scalarisation and selects the value-maximising Pareto point as
the recommended operating policy::

    from insurance_fairness import DoubleFairnessAudit

    audit = DoubleFairnessAudit(n_alphas=20)
    audit.fit(
        X_train,        # features excluding protected attribute
        y_premium,      # primary outcome: pure premium
        y_loss_ratio,   # fairness outcome: claims / premium
        S_gender,       # binary protected group indicator
    )
    result = audit.audit()
    print(result.summary())
    fig = audit.plot_pareto()
    print(audit.report())   # FCA evidence pack section

v0.6.3 adds :class:`DiscriminationInsensitiveReweighter` — training-data
reweighting that achieves X ⊥ A without removing the protected attribute.
Implements the KL divergence minimisation approach from Miao & Pesenti (2026,
arXiv:2603.16720). Weights integrate with any sklearn ``sample_weight``
parameter::

    from insurance_fairness import DiscriminationInsensitiveReweighter

    rw = DiscriminationInsensitiveReweighter(protected_col="gender")
    weights = rw.fit_transform(X_train)

    model.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)
    diag = rw.diagnostics(X_train)

Quick start::

    import polars as pl
    from insurance_fairness import FairnessAudit

    audit = FairnessAudit(
        model=catboost_model,
        data=df,
        protected_cols=["gender"],
        prediction_col="predicted_premium",
        outcome_col="claim_amount",
        exposure_col="exposure",
    )
    report = audit.run()
    report.summary()
    report.to_markdown("audit_report.md")

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.

Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
Discrimination in Insurance Pricing. European Journal of Operational Research.

Bellamy et al. (2024). Multi-Objective Fairness Optimisation for Insurance
Pricing Models. arXiv:2512.24747.

Denuit, Michaelides & Trufin (2026). Multicalibration in Insurance Pricing.
arXiv:2603.16317.

Zhang, Liu & Shi (2025). Discrimination-Free Insurance Pricing with Privatized
Sensitive Attributes. arXiv:2504.11775.

Côté, O., Côté, M.-P., and Charpentier, A. (2025). A Scalable Toolbox for
Exposing Indirect Discrimination in Insurance Rates. CAS Working Paper.

Huang, F. & Pesenti, S. M. (2025). Marginal Fairness: Fair Decision-Making
under Risk Measures. arXiv:2505.18895.

Bian, Z., Wang, L., Shi, C., Qi, Z. (2026). Double Fairness Policy Learning:
Integrating Action Fairness and Outcome Fairness in Decision-making.
arXiv:2601.19186v2.

Miao, K. E. & Pesenti, S. M. (2026). Discrimination-Insensitive Pricing.
arXiv:2603.16720.
"""

from insurance_fairness.audit import FairnessAudit, FairnessReport
from insurance_fairness.bias_metrics import (
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    equalised_odds,
    gini_by_group,
    theil_index,
)
from insurance_fairness.counterfactual import counterfactual_fairness
from insurance_fairness.discrimination_insensitive import (
    DiscriminationInsensitiveReweighter,
    ReweighterDiagnostics,
)
from insurance_fairness.double_fairness import DoubleFairnessAudit, DoubleFairnessResult
from insurance_fairness.marginal_fairness import MarginalFairnessPremium, MarginalFairnessReport
from insurance_fairness.multicalibration import MulticalibrationAudit, MulticalibrationReport
from insurance_fairness.privatized_audit import PrivatizedFairnessAudit, PrivatizedAuditResult
from insurance_fairness.proxy_detection import (
    detect_proxies,
    mutual_information_scores,
    partial_correlation,
    proxy_r2_scores,
    ProxyDetectionResult,
    shap_proxy_scores,
)
from insurance_fairness.proxy_vulnerability import (
    ProxyVulnerabilityScore,
    ProxyVulnerabilityResult,
    PremiumSpectrum,
    compute_post_pricing_metrics,
    partition_by_proxy_vulnerability,
)
from insurance_fairness.report import generate_markdown_report

# Subpackages: import for side-effects / discoverability
from insurance_fairness import optimal_transport  # noqa: F401
from insurance_fairness import diagnostics  # noqa: F401

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-fairness")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
__all__ = [
    # Core audit
    "FairnessAudit",
    "FairnessReport",
    # Bias metrics
    "calibration_by_group",
    "demographic_parity_ratio",
    "disparate_impact_ratio",
    "equalised_odds",
    "gini_by_group",
    "theil_index",
    # Counterfactual
    "counterfactual_fairness",
    # Discrimination-insensitive reweighting (v0.6.3)
    "DiscriminationInsensitiveReweighter",
    "ReweighterDiagnostics",
    # Double fairness (v0.6.0)
    "DoubleFairnessAudit",
    "DoubleFairnessResult",
    # Marginal fairness (v0.5.0)
    "MarginalFairnessPremium",
    "MarginalFairnessReport",
    # Multicalibration
    "MulticalibrationAudit",
    "MulticalibrationReport",
    # Privatized audit (LDP)
    "PrivatizedFairnessAudit",
    "PrivatizedAuditResult",
    # Proxy detection
    "detect_proxies",
    "ProxyDetectionResult",
    "mutual_information_scores",
    "partial_correlation",
    "proxy_r2_scores",
    "shap_proxy_scores",
    # Proxy vulnerability (v0.4.0)
    "ProxyVulnerabilityScore",
    "ProxyVulnerabilityResult",
    "PremiumSpectrum",
    "compute_post_pricing_metrics",
    "partition_by_proxy_vulnerability",
    # Reporting
    "generate_markdown_report",
    # Subpackages (import from subpackage directly)
    "optimal_transport",
    "diagnostics",
]
