"""
Per-policyholder proxy vulnerability scores.

Provides local (individual-level) discrimination diagnostics:
  - d_proxy_local_i = |h_i - h_star_i| / h_i
  - proxy_vulnerability_monetary_i = unaware_i - aware_i (from benchmarks)

These give pricing teams the ability to identify which policyholders are
most affected by proxy discrimination in their current book.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._benchmarks import BenchmarkPremiums
from ._utils import d_proxy_rag


def compute_local_scores(
    h: np.ndarray,
    h_star: np.ndarray,
    benchmarks: BenchmarkPremiums,
    d_proxy_thresholds: dict[str, float] | None = None,
    policy_ids: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Compute per-policyholder proxy discrimination scores.

    Parameters
    ----------
    h:
        Fitted prices (model predictions).
    h_star:
        Admissible prices.
    benchmarks:
        BenchmarkPremiums from _benchmarks.compute_benchmarks().
    d_proxy_thresholds:
        RAG thresholds for local d_proxy scores. Defaults to module defaults.
    policy_ids:
        Optional array of policy identifiers. If None, uses 0-based index.

    Returns
    -------
    Polars DataFrame with columns:
        policy_id           : policy identifier
        h                   : fitted premium
        h_star              : admissible premium
        d_proxy_local       : |h - h_star| / h (relative deviation)
        d_proxy_absolute    : |h - h_star| in premium units (pounds)
        proxy_vulnerability : unaware - aware premium (monetary, pounds)
        rag                 : RAG status based on d_proxy_local
    """
    n = len(h)

    if policy_ids is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(policy_ids)

    # Local d_proxy: |h - h_star| / h
    # Clip to avoid division by very small h values creating extreme ratios
    h_safe = np.where(h > 0, h, np.finfo(float).eps)
    d_proxy_local = np.abs(h - h_star) / h_safe
    d_proxy_abs = np.abs(h - h_star)

    rag_labels = np.array(
        [d_proxy_rag(float(v), d_proxy_thresholds) for v in d_proxy_local]
    )

    return pl.DataFrame({
        "policy_id": ids,
        "h": h.astype(float),
        "h_star": h_star.astype(float),
        "d_proxy_local": d_proxy_local,
        "d_proxy_absolute": d_proxy_abs,
        "proxy_vulnerability": benchmarks.proxy_vulnerability.astype(float),
        "rag": rag_labels,
    })
