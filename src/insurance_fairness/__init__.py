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
from insurance_fairness.multicalibration import MulticalibrationAudit, MulticalibrationReport
from insurance_fairness.proxy_detection import (
    detect_proxies,
    mutual_information_scores,
    partial_correlation,
    proxy_r2_scores,
    ProxyDetectionResult,
    shap_proxy_scores,
)
from insurance_fairness.report import generate_markdown_report

# Subpackages: import for side-effects / discoverability
from insurance_fairness import optimal_transport  # noqa: F401
from insurance_fairness import diagnostics  # noqa: F401

__version__ = "0.3.7"
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
    # Multicalibration
    "MulticalibrationAudit",
    "MulticalibrationReport",
    # Proxy detection
    "detect_proxies",
    "ProxyDetectionResult",
    "mutual_information_scores",
    "partial_correlation",
    "proxy_r2_scores",
    "shap_proxy_scores",
    # Reporting
    "generate_markdown_report",
    # Subpackages (import from subpackage directly)
    "optimal_transport",
    "diagnostics",
]
