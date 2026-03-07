"""
insurance-fairness
==================

Proxy discrimination auditing for UK insurance pricing models.

Implements fairness diagnostics and audit reporting aligned with:
- FCA Consumer Duty (PRIN 2A, live July 2023)
- FCA Evaluation Paper EP25/2 (2025)
- Equality Act 2010, Section 19 (Indirect Discrimination)
- ICOBS pricing practices rules

The primary entry point is :class:`FairnessAudit`, which runs a full audit
of a CatBoost pricing model and returns a structured :class:`FairnessReport`.

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
from insurance_fairness.proxy_detection import (
    mutual_information_scores,
    partial_correlation,
    proxy_r2_scores,
    shap_proxy_scores,
)
from insurance_fairness.report import generate_markdown_report

__version__ = "0.1.0"
__all__ = [
    "FairnessAudit",
    "FairnessReport",
    "calibration_by_group",
    "demographic_parity_ratio",
    "disparate_impact_ratio",
    "equalised_odds",
    "gini_by_group",
    "theil_index",
    "counterfactual_fairness",
    "mutual_information_scores",
    "partial_correlation",
    "proxy_r2_scores",
    "shap_proxy_scores",
    "generate_markdown_report",
]
