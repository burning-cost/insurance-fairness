"""
Tests for _local.py: per-policyholder proxy vulnerability scores.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_fairness.diagnostics._benchmarks import BenchmarkPremiums
from insurance_fairness.diagnostics._local import compute_local_scores


def _make_benchmarks(n: int, proxy_vuln: np.ndarray | None = None) -> BenchmarkPremiums:
    """Create a minimal BenchmarkPremiums for testing."""
    rng = np.random.default_rng(0)
    unaware = rng.uniform(100, 300, n)
    if proxy_vuln is None:
        proxy_vuln = rng.normal(0, 20, n)
    aware = unaware - proxy_vuln
    return BenchmarkPremiums(
        best_estimate=unaware.copy(),
        unaware=unaware,
        aware=aware,
        proxy_vulnerability=proxy_vuln,
    )


class TestComputeLocalScores:
    """Tests for compute_local_scores."""

    def test_returns_polars_dataframe(self):
        """Should return a Polars DataFrame."""
        n = 100
        h = np.random.default_rng(0).uniform(100, 300, n)
        h_star = h * 0.95
        benchmarks = _make_benchmarks(n)

        result = compute_local_scores(h, h_star, benchmarks)

        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self):
        """Should have the required columns."""
        n = 50
        h = np.ones(n) * 200.0
        h_star = np.ones(n) * 190.0
        benchmarks = _make_benchmarks(n)

        result = compute_local_scores(h, h_star, benchmarks)

        expected_cols = {
            "policy_id", "h", "h_star", "d_proxy_local",
            "d_proxy_absolute", "proxy_vulnerability", "rag"
        }
        assert expected_cols.issubset(set(result.columns))

    def test_length_matches_input(self):
        """Output rows should match input length."""
        n = 200
        h = np.random.default_rng(1).uniform(100, 300, n)
        h_star = h * 0.9
        benchmarks = _make_benchmarks(n)

        result = compute_local_scores(h, h_star, benchmarks)

        assert len(result) == n

    def test_d_proxy_local_zero_when_h_equals_h_star(self):
        """d_proxy_local = 0 when h == h_star."""
        n = 100
        h = np.random.default_rng(0).uniform(100, 300, n)
        h_star = h.copy()
        benchmarks = _make_benchmarks(n, proxy_vuln=np.zeros(n))

        result = compute_local_scores(h, h_star, benchmarks)

        np.testing.assert_allclose(
            result["d_proxy_local"].to_numpy(), 0.0, atol=1e-12
        )

    def test_d_proxy_local_formula(self):
        """d_proxy_local_i = |h_i - h_star_i| / h_i."""
        h = np.array([200.0, 300.0, 100.0])
        h_star = np.array([180.0, 300.0, 110.0])
        benchmarks = _make_benchmarks(3, proxy_vuln=np.zeros(3))

        result = compute_local_scores(h, h_star, benchmarks)

        expected = np.abs(h - h_star) / h
        np.testing.assert_allclose(
            result["d_proxy_local"].to_numpy(), expected, rtol=1e-10
        )

    def test_d_proxy_absolute_formula(self):
        """d_proxy_absolute_i = |h_i - h_star_i|."""
        h = np.array([200.0, 300.0, 100.0])
        h_star = np.array([180.0, 300.0, 110.0])
        benchmarks = _make_benchmarks(3, proxy_vuln=np.zeros(3))

        result = compute_local_scores(h, h_star, benchmarks)

        expected = np.abs(h - h_star)
        np.testing.assert_allclose(
            result["d_proxy_absolute"].to_numpy(), expected, rtol=1e-10
        )

    def test_proxy_vulnerability_matches_benchmarks(self):
        """proxy_vulnerability in output should match benchmarks.proxy_vulnerability."""
        n = 100
        h = np.random.default_rng(0).uniform(100, 300, n)
        h_star = h * 0.92
        vuln = np.random.default_rng(1).normal(0, 15, n)
        benchmarks = _make_benchmarks(n, proxy_vuln=vuln)

        result = compute_local_scores(h, h_star, benchmarks)

        np.testing.assert_allclose(
            result["proxy_vulnerability"].to_numpy(), vuln, rtol=1e-10
        )

    def test_rag_labels_are_valid(self):
        """RAG labels should be one of green/amber/red."""
        n = 200
        h = np.random.default_rng(2).uniform(100, 400, n)
        h_star = np.random.default_rng(3).uniform(80, 380, n)
        benchmarks = _make_benchmarks(n)

        result = compute_local_scores(h, h_star, benchmarks)

        rag_vals = set(result["rag"].to_list())
        assert rag_vals.issubset({"green", "amber", "red"})

    def test_custom_policy_ids(self):
        """Should use provided policy IDs."""
        n = 5
        h = np.array([100.0, 200.0, 150.0, 250.0, 175.0])
        h_star = h * 0.95
        benchmarks = _make_benchmarks(n)
        ids = np.array([101, 102, 103, 104, 105])

        result = compute_local_scores(h, h_star, benchmarks, policy_ids=ids)

        np.testing.assert_array_equal(result["policy_id"].to_numpy(), ids)

    def test_default_policy_ids_are_zero_indexed(self):
        """Default policy IDs should be 0, 1, 2, ..."""
        n = 10
        h = np.random.default_rng(0).uniform(100, 300, n)
        h_star = h * 0.9
        benchmarks = _make_benchmarks(n)

        result = compute_local_scores(h, h_star, benchmarks)

        expected_ids = np.arange(n)
        np.testing.assert_array_equal(result["policy_id"].to_numpy(), expected_ids)

    def test_green_when_d_proxy_local_small(self):
        """Policies with d_proxy_local < 0.05 should be green."""
        h = np.array([100.0])
        h_star = np.array([101.0])  # 1% deviation => local d_proxy = 0.01
        benchmarks = _make_benchmarks(1, proxy_vuln=np.zeros(1))

        result = compute_local_scores(h, h_star, benchmarks)

        assert result["rag"][0] == "green"

    def test_red_when_d_proxy_local_large(self):
        """Policies with d_proxy_local > 0.15 should be red."""
        h = np.array([100.0])
        h_star = np.array([120.0])  # 20% deviation => local d_proxy = 0.20
        benchmarks = _make_benchmarks(1, proxy_vuln=np.zeros(1))

        result = compute_local_scores(h, h_star, benchmarks)

        assert result["rag"][0] == "red"
