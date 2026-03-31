"""
insurance_fairness.optimal_transport
=====================================

Discrimination-free insurance pricing via Lindholm marginalisation,
causal path decomposition, and Wasserstein barycenter correction.

Implements the Lindholm (2022) framework for producing premiums that satisfy
the discrimination-free principle: prices based only on justified rating factors,
not protected characteristics. Extends with Côté (2025) causal path decomposition
for separating direct and indirect discrimination, and Wasserstein barycenter
correction for continuous protected characteristics.

Quickstart::

    from insurance_fairness.optimal_transport import (
        CausalGraph,
        DiscriminationFreePrice,
        FairnessReport,
        FCAReport,
    )

    graph = (CausalGraph()
        .add_protected("gender")
        .add_justified_mediator("claims_history", parents=["gender"])
        .add_proxy("annual_mileage", parents=["gender"])
        .add_outcome("claim_freq"))

    dfp = DiscriminationFreePrice(graph=graph, combined_model_fn=my_model)
    result = dfp.fit_transform(X_train, D_train, exposure=exposure_train)

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.

Côté (2025). Causal path decomposition for proxy discrimination in insurance.
"""

from .causal import CausalGraph, PathDecomposer, PathDecomposition
from .correction import LindholmCorrector, SequentialOTCorrector, WassersteinCorrector
from .pricing import DiscriminationFreePrice, PricingResult
from .report import FairnessReport, FCAReport

__all__ = [
    "CausalGraph",
    "PathDecomposer",
    "PathDecomposition",
    "LindholmCorrector",
    "WassersteinCorrector",
    "SequentialOTCorrector",
    "DiscriminationFreePrice",
    "PricingResult",
    "FairnessReport",
    "FCAReport",
]
