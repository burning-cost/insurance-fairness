"""
insurance_fairness.sensitivity
================================

Sensitivity-based proxy discrimination measures for UK insurance pricing.

Implements the full LRTW EJOR 2026 methodology:

  Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
  Discrimination in Insurance Pricing. European Journal of Operational Research.
  DOI: 10.1016/j.ejor.2026.01.021. Open access: https://openaccess.city.ac.uk/id/eprint/36642/

Three key quantities from the paper:

**PD (proxy discrimination)** — ``ProxyDiscriminationMeasure.pd_score``

  PD(pi) = min_{c, v in V} E[(pi(X) - c - sum_d mu(X,d)*v_d)^2] / Var(pi(X))

  PD = 0 iff the price avoids proxy discrimination. This is a tight
  characterisation: unlike UF=0, PD=0 does not just mean group-level mean
  equality; it means the price cannot be expressed as a weighted combination of
  the best-estimate conditional means mu(X, d). The weights v are constrained to
  the admissible set V = {v in [0,1]^|D| : sum v_d <= 1}.

**UF (demographic unfairness)** — ``ProxyDiscriminationMeasure.uf_score``

  UF(pi) = Var(E[pi(X)|D]) / Var(pi(X))

  The first-order Sobol index of pi with respect to D. UF = 0 iff group-level
  mean premiums are equal (demographic parity). UF = 0 does NOT imply PD = 0.

**Sobol attribution** — ``SobolAttribution``

  Per-feature first-order and total Sobol indices for the discrimination
  residual Lambda. Identifies which rating factors are the primary drivers
  of proxy discrimination.

**Shapley attribution** — ``ShapleyAttribution``

  CEN-Shapley decomposition of PD into per-feature contributions. Shapley values
  sum to PD (after surrogate correction). For p <= 12 features: exact enumeration.
  For p > 12: Monte Carlo permutation estimator.

Quick start::

    import numpy as np
    from insurance_fairness.sensitivity import (
        ProxyDiscriminationMeasure,
        SobolAttribution,
        ShapleyAttribution,
    )

    # Fit the measure (mu_hat=1-D price array is the most common usage)
    m = ProxyDiscriminationMeasure()
    m.fit(y=claims, X=X, D=gender, mu_hat=fitted_prices, weights=exposure)
    print(m.summary())
    # PD = 0.03 (3% of price variance is proxy-discriminatory)
    # UF = 0.01 (1% of variance explained by gender group means)

    # Sobol attribution
    sa = SobolAttribution()
    sa.fit(m.Lambda, X, pi, exposure, feature_names=["age_band", "vehicle", "ncd"])
    print(sa.attributions_)

    # Shapley attribution
    sh = ShapleyAttribution()
    sh.fit(m.Lambda, X, pi, exposure, feature_names=["age_band", "vehicle", "ncd"])
    print(sh.attributions_)
    print(f"Shapley PD sum = {sh.attributions_['shapley_pd'].sum():.4f}")

FCA context
-----------
Under the FCA Consumer Duty (PRIN 2A) and the Equality Act 2010 s.19, pricing
teams must be able to demonstrate that their models do not indirectly
discriminate on protected characteristics. PD provides a single number that
answers 'is this model proxy-discriminatory?' with a tight null hypothesis
(PD=0 is exactly the admissible set condition from LRTW 2022). UF provides the
demographic unfairness picture. Sobol and Shapley attribution tell you *which*
rating factors are doing the discriminatory work.

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.

Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
Discrimination in Insurance Pricing. European Journal of Operational Research.
DOI: 10.1016/j.ejor.2026.01.021.

Owen, A.B. (2014). Sobol' indices and Shapley value. SIAM/ASA Journal on
Uncertainty Quantification 2(1), 245-261.
"""

from ._measure import ProxyDiscriminationMeasure
from ._sobol import SobolAttribution
from ._shapley import ShapleyAttribution

__all__ = [
    "ProxyDiscriminationMeasure",
    "SobolAttribution",
    "ShapleyAttribution",
]
