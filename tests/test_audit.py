"""
Tests for audit.py (FairnessAudit and FairnessReport).

These tests use the synthetic fixtures from conftest.py and verify:
- FairnessAudit runs end-to-end without a model (proxy detection disabled)
- FairnessReport has the correct structure
- to_dict() and to_markdown() produce valid output
- RAG status logic is correct
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from insurance_fairness.audit import FairnessAudit, FairnessReport, ProtectedCharacteristicReport


class TestFairnessAudit:
    def test_runs_without_model(self, simple_binary_df):
        """FairnessAudit should run all non-proxy metrics without a model."""
        audit = FairnessAudit(
            model=None,
            data=simple_binary_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            factor_cols=["vehicle_age", "ncd_years"],
            run_proxy_detection=False,
            run_counterfactual=False,
        )
        report = audit.run()
        assert isinstance(report, FairnessReport)
        assert report.n_policies == len(simple_binary_df)
        assert "gender" in report.results

    def test_report_structure(self, simple_binary_df):
        """FairnessReport should contain all expected fields."""
        audit = FairnessAudit(
            model=None,
            data=simple_binary_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()

        # Check top-level structure
        assert isinstance(report.audit_date, str)
        assert len(report.audit_date) == 10  # ISO date: YYYY-MM-DD
        assert report.protected_cols == ["gender"]
        assert report.total_exposure > 0
        assert report.overall_rag in ("green", "amber", "red")

        # Check per-characteristic results
        gender_result = report.results["gender"]
        assert isinstance(gender_result, ProtectedCharacteristicReport)
        assert gender_result.demographic_parity is not None
        assert gender_result.calibration is not None
        assert gender_result.disparate_impact is not None
        assert gender_result.gini is not None

    def test_multiple_protected_cols(self, simple_binary_df):
        """Should handle multiple protected characteristics."""
        # Temporarily add a second protected col
        df = simple_binary_df.with_columns(
            (pl.col("vehicle_age") > 10).cast(pl.Int32).alias("age_group")
        )
        audit = FairnessAudit(
            model=None,
            data=df,
            protected_cols=["gender", "age_group"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert "gender" in report.results
        assert "age_group" in report.results

    def test_missing_required_column_raises(self):
        """Should raise ValueError if a required column is missing."""
        df = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            FairnessAudit(
                model=None,
                data=df,
                protected_cols=["gender"],
                prediction_col="pred",
                outcome_col="actual",
            )

    def test_auto_factor_cols(self, simple_binary_df):
        """When factor_cols is None, should auto-detect from non-protected, non-metric cols."""
        audit = FairnessAudit(
            model=None,
            data=simple_binary_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            factor_cols=None,
            run_proxy_detection=False,
        )
        # Should have included vehicle_age and ncd_years, excluded gender/predicted/claims/exposure
        assert "vehicle_age" in audit.factor_cols
        assert "ncd_years" in audit.factor_cols
        assert "gender" not in audit.factor_cols
        assert "predicted_premium" not in audit.factor_cols

    def test_pandas_input_accepted(self, simple_binary_df):
        """FairnessAudit should accept pandas DataFrames and convert internally."""
        pandas_df = simple_binary_df.to_pandas()
        audit = FairnessAudit(
            model=None,
            data=pandas_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert isinstance(report, FairnessReport)

    def test_known_disparity_detected(self, simple_binary_df):
        """simple_binary_df has a ~30% premium disparity; this should be flagged."""
        audit = FairnessAudit(
            model=None,
            data=simple_binary_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        dp = report.results["gender"].demographic_parity
        # log(1.3) ~ 0.262, well above red threshold of 0.10
        assert dp.rag == "red"
        assert report.overall_rag == "red"

    def test_no_exposure_col(self, simple_binary_df):
        """Audit should run without an exposure column."""
        audit = FairnessAudit(
            model=None,
            data=simple_binary_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col=None,
            run_proxy_detection=False,
        )
        report = audit.run()
        assert isinstance(report, FairnessReport)

    def test_multi_group_audit(self, multi_group_df):
        """Should handle a 3-category protected characteristic."""
        audit = FairnessAudit(
            model=None,
            data=multi_group_df,
            protected_cols=["region"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        report = audit.run()
        assert "region" in report.results
        dp = report.results["region"].demographic_parity
        assert "A" in dp.group_means
        assert "B" in dp.group_means
        assert "C" in dp.group_means


class TestFairnessReport:
    @pytest.fixture
    def sample_report(self, simple_binary_df):
        audit = FairnessAudit(
            model=None,
            data=simple_binary_df,
            protected_cols=["gender"],
            prediction_col="predicted_premium",
            outcome_col="claim_amount",
            exposure_col="exposure",
            run_proxy_detection=False,
        )
        return audit.run()

    def test_summary_prints_without_error(self, sample_report, capsys):
        """summary() should print without raising."""
        sample_report.summary()
        captured = capsys.readouterr()
        assert "Fairness Audit" in captured.out
        assert "gender" in captured.out

    def test_to_dict_is_json_serialisable(self, sample_report):
        """to_dict() output should be JSON-serialisable."""
        d = sample_report.to_dict()
        json_str = json.dumps(d)  # Should not raise
        parsed = json.loads(json_str)
        assert parsed["protected_cols"] == ["gender"]

    def test_to_dict_structure(self, sample_report):
        """to_dict() should have expected keys."""
        d = sample_report.to_dict()
        assert "model_name" in d
        assert "audit_date" in d
        assert "overall_rag" in d
        assert "results" in d
        assert "gender" in d["results"]
        assert "demographic_parity" in d["results"]["gender"]
        assert "calibration" in d["results"]["gender"]

    def test_to_markdown_creates_file(self, sample_report, tmp_path):
        """to_markdown() should write a file."""
        path = str(tmp_path / "audit.md")
        sample_report.to_markdown(path)
        with open(path, "r") as f:
            content = f.read()
        assert "Fairness Audit Report" in content
        assert "gender" in content
        assert "Regulatory Compliance Framework" in content

    def test_markdown_contains_rag_symbols(self, sample_report, tmp_path):
        """Markdown report should include RAG status symbols."""
        path = str(tmp_path / "audit.md")
        sample_report.to_markdown(path)
        with open(path, "r") as f:
            content = f.read()
        # At least one RAG symbol should appear
        assert any(sym in content for sym in ["[GREEN]", "[AMBER]", "[RED]"])

    def test_markdown_contains_sign_off_section(self, sample_report, tmp_path):
        """Markdown report should have a sign-off section."""
        path = str(tmp_path / "audit.md")
        sample_report.to_markdown(path)
        with open(path, "r") as f:
            content = f.read()
        assert "Sign-off" in content
        assert "Pricing Actuary" in content
