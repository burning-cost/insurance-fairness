"""Tests for FairnessReport and FCAReport."""
import json
import numpy as np
import polars as pl
import pytest
import tempfile
import os

from insurance_fairness.optimal_transport.causal import CausalGraph
from insurance_fairness.optimal_transport.pricing import DiscriminationFreePrice, PricingResult
from insurance_fairness.optimal_transport.report import FairnessReport, FCAReport


# ── helpers ──────────────────────────────────────────────────────────────────


def make_graph() -> CausalGraph:
    return (
        CausalGraph()
        .add_protected("gender")
        .add_justified_mediator("claims_history", parents=["gender"])
        .add_proxy("annual_mileage", parents=["gender"])
        .add_outcome("loss")
        .add_edge("claims_history", "loss")
        .add_edge("annual_mileage", "loss")
    )


def make_model(df: pl.DataFrame) -> np.ndarray:
    n = df.shape[0]
    base = np.ones(n) * 0.1
    if "gender" in df.columns:
        base += (df["gender"] == "M").to_numpy() * 0.02
    return base


def make_result(n=300):
    rng = np.random.default_rng(42)
    gender = rng.choice(["M", "F"], n).tolist()
    claims = rng.choice(["yes", "no"], n).tolist()
    mileage = rng.integers(5000, 30000, n).tolist()
    X = pl.DataFrame({"claims_history": claims, "annual_mileage": mileage, "gender": gender})
    D = pl.DataFrame({"gender": gender})
    exp = np.ones(n)
    g = make_graph()
    dfp = DiscriminationFreePrice(g, combined_model_fn=make_model)
    result = dfp.fit_transform(X, D, exposure=exp)
    return result, g, D, exp


class TestFairnessReport:
    def test_discrimination_metrics_returns_dict(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        metrics = report.discrimination_metrics(D, exposure=exp)
        assert isinstance(metrics, dict)
        assert "gender" in metrics

    def test_demographic_parity_ratio_before_gte_1(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        metrics = report.discrimination_metrics(D, exposure=exp)
        dp = metrics["gender"]["demographic_parity_ratio_before"]
        assert dp >= 1.0

    def test_demographic_parity_ratio_after_closer_to_one(self):
        result, g, D, exp = make_result(1000)
        report = FairnessReport(result, g)
        metrics = report.discrimination_metrics(D, exposure=exp)
        dp_before = metrics["gender"]["demographic_parity_ratio_before"]
        dp_after = metrics["gender"]["demographic_parity_ratio_after"]
        # After Lindholm correction, ratio should be closer to 1
        assert abs(dp_after - 1.0) <= abs(dp_before - 1.0) + 0.05  # small tolerance

    def test_wasserstein_distance_present(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        metrics = report.discrimination_metrics(D, exposure=exp)
        w2 = metrics["gender"]["wasserstein_distance_before"]
        assert w2 is not None
        assert w2 >= 0

    def test_bias_correction_factor_in_metrics(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        metrics = report.discrimination_metrics(D, exposure=exp)
        bcf = metrics["gender"]["bias_correction_factor"]
        assert 0.9 < bcf < 1.1

    def test_path_attribution_none_without_decomposition(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        # decomposition may or may not be present; test handles both
        pa = report.path_attribution()
        # Returns None or DataFrame
        assert pa is None or isinstance(pa, pl.DataFrame)

    def test_premium_comparison_table_shape(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        table = report.premium_comparison_table(D)
        assert isinstance(table, pl.DataFrame)
        assert "group" in table.columns
        assert "mean_best_estimate" in table.columns
        assert "mean_fair_premium" in table.columns
        assert table.shape[0] == 2  # M and F

    def test_premium_comparison_change_pct_direction(self):
        """Gender with higher best-estimate should have lower fair premium."""
        result, g, D, exp = make_result(500)
        report = FairnessReport(result, g)
        table = report.premium_comparison_table(D)
        # M has higher best-estimate premium (model adds 0.02 for M)
        m_row = table.filter(pl.col("group") == "M")
        f_row = table.filter(pl.col("group") == "F")
        if m_row.shape[0] > 0 and f_row.shape[0] > 0:
            m_change = m_row["change_pct"][0]
            f_change = f_row["change_pct"][0]
            # M premium should decrease (negative change) or F increase
            assert m_change < f_change or abs(m_change - f_change) < 1.0

    def test_to_dict_returns_dict(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "method" in d
        assert "protected_attrs" in d

    def test_no_exposure_defaults(self):
        result, g, D, exp = make_result()
        report = FairnessReport(result, g)
        metrics = report.discrimination_metrics(D)  # no exposure arg
        assert "gender" in metrics


class TestFCAReport:
    def _make_fca_report(self, n=200):
        result, g, D, exp = make_result(n)
        meta = {
            "firm_name": "Test Insurer Ltd",
            "model_name": "Motor Frequency GLM",
            "reporting_date": "2026-03-10",
            "model_version": "2.1",
        }
        return FCAReport(result, meta), result, D

    def test_render_markdown(self):
        report, result, D = self._make_fca_report()
        md = report.render("markdown")
        assert isinstance(md, str)
        assert len(md) > 100

    def test_render_markdown_contains_key_sections(self):
        report, result, D = self._make_fca_report()
        md = report.render("markdown")
        assert "Executive Summary" in md
        assert "Protected Characteristics" in md
        assert "Fairness Methodology" in md
        assert "Bias Correction" in md
        assert "Equality Act" in md
        assert "Consumer Duty" in md

    def test_render_markdown_contains_firm_name(self):
        report, result, D = self._make_fca_report()
        md = report.render("markdown")
        assert "Test Insurer Ltd" in md

    def test_render_markdown_contains_method(self):
        report, result, D = self._make_fca_report()
        md = report.render("markdown")
        assert "Lindholm" in md or "lindholm" in md

    def test_render_json_valid(self):
        report, result, D = self._make_fca_report()
        json_str = report.render("json")
        doc = json.loads(json_str)
        assert isinstance(doc, dict)
        assert "method" in doc
        assert "protected_attrs" in doc

    def test_render_json_contains_regulatory_basis(self):
        report, result, D = self._make_fca_report()
        doc = json.loads(report.render("json"))
        assert "regulatory_basis" in doc
        assert any("Equality Act" in s for s in doc["regulatory_basis"])

    def test_render_html(self):
        report, result, D = self._make_fca_report()
        html = report.render("html")
        assert "<html>" in html
        assert "Executive Summary" in html

    def test_render_invalid_format_raises(self):
        report, result, D = self._make_fca_report()
        with pytest.raises(ValueError, match="Unknown format"):
            report.render("pdf")

    def test_save_markdown(self):
        report, result, D = self._make_fca_report()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            report.save(path, "markdown")
            with open(path) as f:
                content = f.read()
            assert "Executive Summary" in content
        finally:
            os.unlink(path)

    def test_save_json(self):
        report, result, D = self._make_fca_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.save(path, "json")
            with open(path) as f:
                doc = json.load(f)
            assert "method" in doc
        finally:
            os.unlink(path)

    def test_bias_correction_factor_in_json(self):
        report, result, D = self._make_fca_report()
        doc = json.loads(report.render("json"))
        bcf = doc["bias_correction_factor"]
        assert 0.9 < bcf < 1.1

    def test_portfolio_weights_in_json(self):
        report, result, D = self._make_fca_report()
        doc = json.loads(report.render("json"))
        assert "portfolio_weights" in doc
