"""
Tests for proxy_detection.py

Tests verify:
- Mutual information correctly ranks strong vs weak proxies
- Partial correlation correctly identifies proxy relationships
- SHAP proxy scores require a model (error handling)
- detect_proxies integrates correctly
- Proxy R-squared is not tested here (requires running CatBoost - run on Databricks)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_fairness.proxy_detection import (
    ProxyDetectionResult,
    ProxyScore,
    detect_proxies,
    mutual_information_scores,
    partial_correlation,
    shap_proxy_scores,
)


class TestMutualInformation:
    def test_strong_proxy_higher_mi(self, proxy_test_df):
        """Strong proxy should have higher MI with gender than weak proxy."""
        scores = mutual_information_scores(
            proxy_test_df,
            protected_col="gender",
            factor_cols=["strong_proxy", "weak_proxy", "unrelated_factor"],
            is_binary_protected=True,
        )
        assert "strong_proxy" in scores
        assert "weak_proxy" in scores
        assert "unrelated_factor" in scores
        # Strong proxy should have highest MI
        assert scores["strong_proxy"] > scores["weak_proxy"]

    def test_unrelated_factor_low_mi(self, proxy_test_df):
        """Unrelated factor should have low mutual information with gender."""
        scores = mutual_information_scores(
            proxy_test_df,
            protected_col="gender",
            factor_cols=["unrelated_factor"],
            is_binary_protected=True,
        )
        # MI should be close to 0 for an unrelated factor
        assert scores["unrelated_factor"] < 0.1

    def test_returns_all_requested_cols(self, proxy_test_df):
        """Should return a score for every factor col."""
        factor_cols = ["strong_proxy", "weak_proxy", "unrelated_factor"]
        scores = mutual_information_scores(
            proxy_test_df, "gender", factor_cols, is_binary_protected=True
        )
        assert set(scores.keys()) == set(factor_cols)

    def test_continuous_protected(self):
        """Should work with a continuous protected characteristic."""
        rng = np.random.default_rng(0)
        n = 500
        s = rng.uniform(0, 1, n)
        x_corr = s + rng.normal(0, 0.1, n)
        x_uncorr = rng.normal(0, 1, n)
        df = pl.DataFrame({
            "s": s.tolist(),
            "x_correlated": x_corr.tolist(),
            "x_unrelated": x_uncorr.tolist(),
        })
        scores = mutual_information_scores(
            df, "s", ["x_correlated", "x_unrelated"], is_binary_protected=False
        )
        assert scores["x_correlated"] > scores["x_unrelated"]

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            mutual_information_scores(df, "gender", ["a"])


class TestPartialCorrelation:
    def test_correlation_without_controls(self, proxy_test_df):
        """Without control variables, should return Spearman correlation."""
        scores = partial_correlation(
            proxy_test_df,
            protected_col="gender",
            factor_cols=["strong_proxy", "weak_proxy", "unrelated_factor"],
        )
        assert abs(scores["strong_proxy"]) > abs(scores["weak_proxy"])
        assert abs(scores["strong_proxy"]) > 0.3  # strong correlation expected

    def test_with_controls(self, proxy_test_df):
        """Partial correlation with controls should typically reduce the correlation."""
        raw = partial_correlation(
            proxy_test_df, "gender", ["strong_proxy"]
        )
        controlled = partial_correlation(
            proxy_test_df, "gender", ["strong_proxy"],
            control_cols=["unrelated_factor"]
        )
        # Controlling for an unrelated variable should not dramatically change the result
        assert abs(controlled["strong_proxy"]) > 0.2

    def test_returns_float(self, proxy_test_df):
        """All values should be floats."""
        scores = partial_correlation(proxy_test_df, "gender", ["weak_proxy", "unrelated_factor"])
        for k, v in scores.items():
            assert isinstance(v, float)
            assert -1.0 <= v <= 1.0

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1, 2], "gender": [0, 1]})
        with pytest.raises(ValueError, match="not found"):
            partial_correlation(df, "gender", ["nonexistent"])


class TestShapProxyScores:
    def test_requires_model_or_shap_values(self, proxy_test_df):
        """Should raise if neither model nor shap_values provided."""
        with pytest.raises(ValueError, match="shap_values or model"):
            shap_proxy_scores(
                proxy_test_df,
                protected_col="gender",
                factor_cols=["strong_proxy"],
                shap_values=None,
                model=None,
            )

    def test_with_precomputed_shap_values(self):
        """Should accept pre-computed SHAP values."""
        n = 100
        rng = np.random.default_rng(1)
        gender = rng.integers(0, 2, n)
        # SHAP values for two features: feature 0 strongly correlated with gender
        shap_f0 = gender.astype(float) * 10 + rng.normal(0, 0.5, n)
        shap_f1 = rng.normal(0, 1, n)
        shap_matrix = np.column_stack([shap_f0, shap_f1])

        df = pl.DataFrame({
            "gender": gender.tolist(),
            "feature_0": rng.uniform(0, 1, n).tolist(),
            "feature_1": rng.uniform(0, 1, n).tolist(),
        })
        scores = shap_proxy_scores(
            df,
            protected_col="gender",
            factor_cols=["feature_0", "feature_1"],
            shap_values=shap_matrix,
        )
        assert scores["feature_0"] > scores["feature_1"]
        assert 0.0 <= scores["feature_0"] <= 1.0

    def test_shap_dimension_mismatch_raises(self):
        """Mismatch between shap_values columns and factor_cols should raise."""
        df = pl.DataFrame({
            "gender": [0, 1],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        })
        shap_vals = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 cols but 1 factor
        with pytest.raises(ValueError, match="shap_values has"):
            shap_proxy_scores(df, "gender", ["f1"], shap_values=shap_vals)


class TestDetectProxies:
    def test_returns_correct_type(self, proxy_test_df):
        """detect_proxies should return ProxyDetectionResult."""
        result = detect_proxies(
            proxy_test_df,
            protected_col="gender",
            factor_cols=["strong_proxy", "weak_proxy", "unrelated_factor"],
            run_proxy_r2=False,  # skip CatBoost (not available in local test env)
            run_mutual_info=True,
            run_partial_corr=True,
            run_shap=False,
            is_binary_protected=True,
        )
        assert isinstance(result, ProxyDetectionResult)
        assert result.protected_col == "gender"
        assert len(result.scores) == 3

    def test_scores_are_proxy_score_objects(self, proxy_test_df):
        """All score entries should be ProxyScore dataclass instances."""
        result = detect_proxies(
            proxy_test_df, "gender",
            ["strong_proxy", "weak_proxy"],
            run_proxy_r2=False,
            run_mutual_info=True,
            run_partial_corr=False,
            run_shap=False,
            is_binary_protected=True,
        )
        for s in result.scores:
            assert isinstance(s, ProxyScore)
            assert s.protected_col == "gender"

    def test_to_polars_method(self, proxy_test_df):
        """to_polars() should return a Polars DataFrame."""
        result = detect_proxies(
            proxy_test_df, "gender",
            ["strong_proxy", "unrelated_factor"],
            run_proxy_r2=False,
            run_mutual_info=True,
            run_partial_corr=False,
            run_shap=False,
            is_binary_protected=True,
        )
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "factor" in df.columns
        assert "mutual_information" in df.columns

    def test_flagged_factors_list(self):
        """flagged_factors should list amber and red factors."""
        # Construct a result with known statuses
        result = ProxyDetectionResult(
            protected_col="gender",
            scores=[
                ProxyScore("f1", "gender", 0.15, 0.5, None, None, "red"),
                ProxyScore("f2", "gender", 0.06, 0.2, None, None, "amber"),
                ProxyScore("f3", "gender", 0.01, 0.05, None, None, "green"),
            ],
        )
        assert "f1" in result.flagged_factors
        assert "f2" in result.flagged_factors
        assert "f3" not in result.flagged_factors
