"""
Tests for _audit.py: ProxyDiscriminationAudit and ProxyDiscriminationResult.

These are integration-style tests that exercise the full pipeline.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from insurance_fairness.diagnostics import ProxyDiscriminationAudit, ProxyDiscriminationResult
from insurance_fairness.diagnostics._audit import ShapleyEffect

from conftest import make_synthetic_dataset, make_zero_proxy_dataset


class TestProxyDiscriminationAuditInit:
    """Tests for ProxyDiscriminationAudit.__init__ validation."""

    def _make_valid_args(self):
        rng = np.random.default_rng(0)
        n = 200
        age = rng.integers(0, 5, n).astype(float)
        vehicle = rng.integers(0, 5, n).astype(float)
        s = rng.integers(0, 2, n).astype(float)
        y = 200 + 40 * age + rng.normal(0, 20, n)
        X_fit = np.column_stack([age, vehicle])
        model = Ridge(alpha=1.0)
        model.fit(X_fit, y)
        X = pl.DataFrame({"age": age, "vehicle": vehicle, "sensitive": s})
        return model, X, y, "sensitive", ["age", "vehicle"]

    def test_valid_inputs_do_not_raise(self):
        model, X, y, s_col, factors = self._make_valid_args()
        audit = ProxyDiscriminationAudit(model, X, y, s_col, factors)
        assert audit is not None

    def test_raises_on_non_model(self):
        _, X, y, s_col, factors = self._make_valid_args()
        with pytest.raises(TypeError, match="predict"):
            ProxyDiscriminationAudit("not_a_model", X, y, s_col, factors)

    def test_raises_on_non_dataframe(self):
        model, _, y, s_col, factors = self._make_valid_args()
        with pytest.raises(TypeError, match="Polars DataFrame"):
            ProxyDiscriminationAudit(model, np.zeros((10, 3)), y, s_col, factors)

    def test_raises_when_sensitive_col_missing(self):
        model, X, y, _, factors = self._make_valid_args()
        with pytest.raises(ValueError, match="Column"):
            ProxyDiscriminationAudit(model, X, y, "missing_col", factors)

    def test_raises_when_rating_factors_empty(self):
        model, X, y, s_col, _ = self._make_valid_args()
        with pytest.raises(ValueError, match="at least one"):
            ProxyDiscriminationAudit(model, X, y, s_col, [])

    def test_raises_when_sensitive_in_rating_factors(self):
        model, X, y, s_col, factors = self._make_valid_args()
        with pytest.raises(ValueError, match="sensitive_col"):
            ProxyDiscriminationAudit(model, X, y, s_col, factors + [s_col])


class TestProxyDiscriminationAuditFit:
    """Integration tests for ProxyDiscriminationAudit.fit()."""

    def test_fit_returns_result(self):
        X, h, model = make_synthetic_dataset(n=500, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model,
            X=X,
            y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16,
            subsample_n=200,
        )
        result = audit.fit()
        assert isinstance(result, ProxyDiscriminationResult)

    def test_d_proxy_in_range(self):
        X, h, model = make_synthetic_dataset(n=500, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=200,
        )
        result = audit.fit()
        assert 0.0 <= result.d_proxy <= 1.0

    def test_d_proxy_ci_valid(self):
        X, h, model = make_synthetic_dataset(n=500, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=200,
        )
        result = audit.fit()
        lo, hi = result.d_proxy_ci
        assert lo <= result.d_proxy <= hi

    def test_shapley_effects_sum_to_one(self):
        X, h, model = make_synthetic_dataset(n=600, proxy_strength=0.5)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=32, subsample_n=300,
        )
        result = audit.fit()
        phi_sum = sum(se.phi for se in result.shapley_effects.values())
        assert phi_sum == pytest.approx(1.0, abs=0.02)

    def test_shapley_effects_all_non_negative(self):
        X, h, model = make_synthetic_dataset(n=500, proxy_strength=0.4)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=32, subsample_n=200,
        )
        result = audit.fit()
        for name, se in result.shapley_effects.items():
            assert se.phi >= 0.0, f"phi[{name}] = {se.phi} < 0"

    def test_shapley_effects_have_correct_structure(self):
        X, h, model = make_synthetic_dataset(n=300, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=150,
        )
        result = audit.fit()

        for name, se in result.shapley_effects.items():
            assert isinstance(se, ShapleyEffect)
            assert se.factor == name
            assert isinstance(se.phi, float)
            assert isinstance(se.phi_monetary, float)
            assert isinstance(se.rank, int)
            assert se.rag in ("green", "amber", "red")

    def test_rag_is_valid(self):
        X, h, model = make_synthetic_dataset(n=400, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=200,
        )
        result = audit.fit()
        assert result.rag in ("green", "amber", "red")

    def test_local_scores_is_polars_dataframe(self):
        X, h, model = make_synthetic_dataset(n=400, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=200,
        )
        result = audit.fit()
        assert isinstance(result.local_scores, pl.DataFrame)
        assert len(result.local_scores) == len(X)

    def test_d_proxy_monetary_positive(self):
        X, h, model = make_synthetic_dataset(n=400, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=200,
        )
        result = audit.fit()
        # d_proxy_monetary = d_proxy * mean_premium; should be positive
        assert result.d_proxy_monetary >= 0.0

    def test_sensitive_col_stored_on_result(self):
        X, h, model = make_synthetic_dataset(n=300, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=150,
        )
        result = audit.fit()
        assert result.sensitive_col == "postcode_area"


class TestProxyDiscriminationWithHighProxy:
    """Test that D_proxy is elevated when proxy discrimination is strong."""

    def test_high_proxy_gives_higher_d_proxy(self):
        """
        With high proxy_strength, D_proxy should exceed the zero-proxy case.
        This verifies the measure is sensitive to discrimination.
        """
        X_high, h_high, model_high = make_synthetic_dataset(
            n=800, proxy_strength=0.8, random_state=10
        )
        X_zero, h_zero, model_zero = make_zero_proxy_dataset(n=800, random_state=10)

        audit_high = ProxyDiscriminationAudit(
            model=model_high, X=X_high, y=h_high,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=32, subsample_n=400, random_state=42,
        )
        audit_zero = ProxyDiscriminationAudit(
            model=model_zero, X=X_zero, y=h_zero,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=32, subsample_n=400, random_state=42,
        )

        result_high = audit_high.fit()
        result_zero = audit_zero.fit()

        assert result_high.d_proxy > result_zero.d_proxy, (
            f"High proxy D_proxy ({result_high.d_proxy:.4f}) should exceed "
            f"zero proxy D_proxy ({result_zero.d_proxy:.4f})"
        )


class TestProxyDiscriminationResultMethods:
    """Tests for ProxyDiscriminationResult helper methods."""

    def _get_result(self):
        X, h, model = make_synthetic_dataset(n=400, proxy_strength=0.3)
        audit = ProxyDiscriminationAudit(
            model=model, X=X, y=h,
            sensitive_col="postcode_area",
            rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
            n_perms=16, subsample_n=200,
        )
        return audit.fit()

    def test_summary_returns_string(self):
        result = self._get_result()
        s = result.summary()
        assert isinstance(s, str)
        assert "D_proxy" in s
        assert "postcode_area" in s

    def test_to_html_creates_file(self, tmp_path):
        result = self._get_result()
        path = tmp_path / "report.html"
        result.to_html(path)
        assert path.exists()
        content = path.read_text()
        assert "<html" in content.lower()
        assert "D_proxy" in content or "proxy" in content.lower()

    def test_to_json_creates_valid_json(self, tmp_path):
        import json
        result = self._get_result()
        path = tmp_path / "report.json"
        result.to_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "d_proxy" in data
        assert "shapley_effects" in data
        assert "rag" in data
