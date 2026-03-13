"""
Tests for _report.py: HTML and JSON report generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from insurance_fairness.diagnostics._audit import ShapleyEffect, ProxyDiscriminationResult
from insurance_fairness.diagnostics._benchmarks import BenchmarkPremiums
from insurance_fairness.diagnostics._report import generate_html_report, generate_json_report


def _make_mock_result() -> ProxyDiscriminationResult:
    """Create a minimal ProxyDiscriminationResult for report tests."""
    n = 100
    rng = np.random.default_rng(0)
    h = rng.uniform(100, 300, n)
    h_star = h * 0.95
    vuln = rng.normal(0, 15, n)

    benchmarks = BenchmarkPremiums(
        best_estimate=h.copy(),
        unaware=h,
        aware=h - vuln,
        proxy_vulnerability=vuln,
    )

    local_scores = pl.DataFrame({
        "policy_id": np.arange(n),
        "h": h,
        "h_star": h_star,
        "d_proxy_local": np.abs(h - h_star) / h,
        "d_proxy_absolute": np.abs(h - h_star),
        "proxy_vulnerability": vuln,
        "rag": ["green"] * 60 + ["amber"] * 30 + ["red"] * 10,
    })

    shapley_effects = {
        "age_band": ShapleyEffect("age_band", 0.4, 80.0, 1, "red"),
        "vehicle_group": ShapleyEffect("vehicle_group", 0.35, 70.0, 2, "red"),
        "ncd_years": ShapleyEffect("ncd_years", 0.15, 30.0, 3, "amber"),
        "proxy_feature": ShapleyEffect("proxy_feature", 0.10, 20.0, 4, "amber"),
    }

    return ProxyDiscriminationResult(
        d_proxy=0.08,
        d_proxy_ci=(0.06, 0.10),
        d_proxy_monetary=200.0 * 0.08,
        shapley_effects=shapley_effects,
        local_scores=local_scores,
        benchmarks=benchmarks,
        rag="amber",
        sensitive_col="postcode_area",
        n_perms=256,
    )


class TestGenerateHtmlReport:
    """Tests for generate_html_report."""

    def test_returns_string(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        assert isinstance(html, str)

    def test_contains_html_tags(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        assert "<html" in html.lower()
        assert "</html>" in html.lower()

    def test_contains_d_proxy_value(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        assert "0.0800" in html or "0.08" in html

    def test_contains_rag_status(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        # AMBER should appear as text (uppercased in template)
        assert "AMBER" in html or "amber" in html.lower()

    def test_contains_sensitive_col(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        assert "postcode_area" in html

    def test_contains_regulatory_references(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        assert "Equality Act" in html
        assert "FCA" in html

    def test_contains_shapley_factor_names(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        for factor in ["age_band", "vehicle_group", "ncd_years"]:
            assert factor in html

    def test_non_empty_output(self):
        result = _make_mock_result()
        html = generate_html_report(result)
        assert len(html) > 1000  # Should be a substantial document


class TestGenerateJsonReport:
    """Tests for generate_json_report."""

    def test_returns_valid_json(self):
        result = _make_mock_result()
        js = generate_json_report(result)
        data = json.loads(js)  # Should not raise
        assert isinstance(data, dict)

    def test_contains_d_proxy(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert "d_proxy" in data
        assert data["d_proxy"] == pytest.approx(0.08)

    def test_contains_ci(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert "d_proxy_ci_lower" in data
        assert "d_proxy_ci_upper" in data
        assert data["d_proxy_ci_lower"] == pytest.approx(0.06)
        assert data["d_proxy_ci_upper"] == pytest.approx(0.10)

    def test_contains_rag(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert data["rag"] == "amber"

    def test_contains_sensitive_col(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert data["sensitive_col"] == "postcode_area"

    def test_contains_shapley_effects(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert "shapley_effects" in data
        assert "age_band" in data["shapley_effects"]
        assert data["shapley_effects"]["age_band"]["phi"] == pytest.approx(0.4)

    def test_contains_benchmarks(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert "benchmarks" in data
        assert "mean_unaware" in data["benchmarks"]
        assert "mean_aware" in data["benchmarks"]

    def test_contains_regulatory_references(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert "regulatory_references" in data
        refs = data["regulatory_references"]
        assert any("Equality Act" in r for r in refs)

    def test_contains_methodology(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert "methodology" in data
        assert "n_perms" in data["methodology"]
        assert data["methodology"]["n_perms"] == 256

    def test_n_policies_correct(self):
        result = _make_mock_result()
        data = json.loads(generate_json_report(result))
        assert data["n_policies"] == 100
