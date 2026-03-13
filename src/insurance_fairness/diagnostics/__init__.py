"""
insurance_fairness.diagnostics
================================

Proxy discrimination diagnostics for UK personal lines insurance pricing.

Measures how much proxy discrimination exists in a fitted pricing model and
which rating factors drive it, implementing:

  - D_proxy: normalised L2-distance from the fitted price to the admissible
    (discrimination-free) price set (LRTW EJOR 2026, SSRN 4897265)
  - Shapley effects: Owen (2014) permutation estimator via surrogate model,
    decomposing discrimination across rating factors
  - Per-policyholder proxy vulnerability scores (Côté et al. 2025)
  - Unaware vs aware premium benchmarks

Regulatory alignment:
  - Equality Act 2010 s.19 (Indirect Discrimination)
  - FCA Consumer Duty PRIN 2A.4 (Fair Value)
  - FCA FG22/5 paras 8.8-8.12 (Pricing Practices Guidance)

Quick start::

    import polars as pl
    from insurance_fairness.diagnostics import ProxyDiscriminationAudit

    audit = ProxyDiscriminationAudit(
        model=my_glm,
        X=df,
        y=df["claim_cost"],
        sensitive_col="postcode_area",
        rating_factors=["age_band", "vehicle_group", "ncd_years"],
        exposure_col="exposure",
    )
    result = audit.fit()
    print(result.summary())
    result.to_html("proxy_discrimination_audit.html")
    result.to_json("proxy_discrimination_audit.json")

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.

Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
Proxy Discrimination in Insurance Pricing. European Journal of Operational
Research. SSRN 4897265.

Owen, A.B. (2014). Sobol' indices and Shapley value. SIAM/ASA Journal on
Uncertainty Quantification 2(1), 245-251.

Côté, M.-P., Côté, S. and Charpentier, A. (2025). Five premium benchmarks
for proxy discrimination in insurance pricing.

Biessy, G. (2024). Revisiting the Discrimination-Free Principle Through
Shapley Values. ASTIN Bulletin.
"""

from ._audit import (
    ProxyDiscriminationAudit,
    ProxyDiscriminationResult,
    ShapleyEffect,
)
from ._admissible import (
    compute_admissible_price,
    compute_d_proxy,
    compute_d_proxy_with_ci,
)
from ._benchmarks import (
    BenchmarkPremiums,
    compute_benchmarks,
    compute_unaware_premium,
    compute_aware_premium,
)
from ._local import compute_local_scores
from ._utils import (
    d_proxy_rag,
    phi_rag,
    DEFAULT_D_PROXY_THRESHOLDS,
    DEFAULT_PHI_THRESHOLDS,
)

__all__ = [
    # Main API
    "ProxyDiscriminationAudit",
    "ProxyDiscriminationResult",
    "ShapleyEffect",
    # Admissible price computation
    "compute_admissible_price",
    "compute_d_proxy",
    "compute_d_proxy_with_ci",
    # Benchmarks
    "BenchmarkPremiums",
    "compute_benchmarks",
    "compute_unaware_premium",
    "compute_aware_premium",
    # Local scores
    "compute_local_scores",
    # Utilities
    "d_proxy_rag",
    "phi_rag",
    "DEFAULT_D_PROXY_THRESHOLDS",
    "DEFAULT_PHI_THRESHOLDS",
]
