"""
Tests for counterfactual.py

Most tests use pre-computed predictions rather than a live CatBoost model,
since CatBoost fitting must run on Databricks. The LRTW marginalisation
and direct flip logic are tested with a mock model.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from insurance_fairness.counterfactual import (
    CounterfactualResult,
    counterfactual_fairness,
    _direct_flip,
    _lrtw_marginalise,
)


class MockCatBoostModel:
    """
    Minimal mock of a CatBoost model for testing.

    Predictions = value of the 'pred_factor' column * gender_modifier.
    If gender=0, modifier=1.0. If gender=1, modifier=1.2.
    """

    def predict(self, pool_or_df):
        import pandas as pd  # noqa: PLC0415

        if hasattr(pool_or_df, "get_features"):
            # CatBoost Pool
            X = pool_or_df.get_features()
            if hasattr(X, "values"):
                X = X.values
        else:
            X = pool_or_df

        if isinstance(X, np.ndarray):
            # Assume columns: [gender, pred_factor]
            gender = X[:, 0].astype(float)
            pred_factor = X[:, 1].astype(float)
        else:
            # pandas DataFrame
            gender = X.iloc[:, 0].values.astype(float)
            pred_factor = X.iloc[:, 1].values.astype(float)

        modifier = np.where(gender == 1, 1.2, 1.0)
        return pred_factor * modifier


class TestCounterfactualResult:
    def test_summary_runs_without_error(self):
        result = CounterfactualResult(
            protected_col="gender",
            original_mean_premium=100.0,
            counterfactual_mean_premium=108.0,
            premium_impact_ratio=1.08,
            premium_impact_log=math.log(1.08),
            policy_level_impacts=pl.Series([1.0, 1.1, 0.9]),
            n_policies=3,
            method="direct_flip",
        )
        summary = result.summary()
        assert "gender" in summary
        assert "8.0%" in summary or "+8.0%" in summary


class TestDirectFlip:
    def test_flip_binary(self):
        """Flipping gender 0->1 should increase predictions via MockModel."""
        df = pl.DataFrame({
            "gender": [0, 0, 0],
            "pred_factor": [100.0, 150.0, 200.0],
        })
        from catboost import Pool

        feature_cols = ["gender", "pred_factor"]
        model = MockCatBoostModel()

        preds_original = model.predict(
            Pool(df.to_pandas(), cat_features=[])
        )
        preds_flipped = _direct_flip(
            model=model,
            df=df,
            protected_col="gender",
            feature_cols=feature_cols,
            flip_values={0: 1, 1: 0},
        )
        # After flip, gender=1, so modifier=1.2. Predictions should be 1.2x original.
        np.testing.assert_allclose(preds_flipped, preds_original * 1.2, rtol=0.01)

    def test_flip_not_in_feature_cols_raises(self):
        """direct_flip requires protected_col in feature_cols."""
        df = pl.DataFrame({
            "gender": [0, 1],
            "other": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="must be in feature_cols"):
            _direct_flip(
                model=MockCatBoostModel(),
                df=df,
                protected_col="gender",
                feature_cols=["other"],
                flip_values={0: 1, 1: 0},
            )

    def test_flip_non_binary_without_mapping_raises(self):
        """Non-binary protected characteristic without flip_values should raise."""
        df = pl.DataFrame({
            "gender": ["A", "B", "C"],
            "other": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="Cannot infer flip_values"):
            _direct_flip(
                model=MockCatBoostModel(),
                df=df,
                protected_col="gender",
                feature_cols=["gender", "other"],
                flip_values=None,
            )


class TestLRTWMarginalise:
    def test_not_in_feature_cols_returns_original(self):
        """If protected col not in feature_cols, return original predictions."""

        class SingleColModel:
            """Mock model using only pred_factor (column index 0)."""
            def predict(self, pool_or_df):
                import pandas as pd  # noqa
                if hasattr(pool_or_df, 'get_features'):
                    X = pool_or_df.get_features()
                else:
                    X = pool_or_df
                if hasattr(X, 'values'):
                    X = X.values
                return X[:, 0].astype(float)  # return pred_factor directly

        df = pl.DataFrame({
            "gender": [0, 1, 0],
            "pred_factor": [100.0, 150.0, 200.0],
        })
        preds = _lrtw_marginalise(
            model=SingleColModel(),
            df=df,
            protected_col="gender",
            feature_cols=["pred_factor"],  # gender not in features
            n_monte_carlo=5,
        )
        # Model only sees pred_factor, which is unchanged; predictions are pred_factor values
        np.testing.assert_allclose(preds, [100.0, 150.0, 200.0], rtol=0.01)

    def test_averaging_reduces_gender_effect(self):
        """
        When gender is averaged over its marginal distribution (50/50),
        the mean prediction should be between the two group means.
        """
        n = 100
        df = pl.DataFrame({
            "gender": [0] * 50 + [1] * 50,
            "pred_factor": [100.0] * 100,
        })
        preds = _lrtw_marginalise(
            model=MockCatBoostModel(),
            df=df,
            protected_col="gender",
            feature_cols=["gender", "pred_factor"],
            n_monte_carlo=200,
            random_seed=42,
        )
        # With 50/50 gender split, expected mean modifier = 0.5*1.0 + 0.5*1.2 = 1.1
        # So mean prediction ~ 110
        assert 105 < np.mean(preds) < 115


class TestCounterfactualFairness:
    def test_direct_flip_with_mock(self):
        """Integration test: counterfactual_fairness with direct_flip and mock model."""
        df = pl.DataFrame({
            "gender": [0, 0, 1, 1],
            "pred_factor": [100.0, 100.0, 100.0, 100.0],
            "exposure": [1.0, 1.0, 1.0, 1.0],
        })
        result = counterfactual_fairness(
            model=MockCatBoostModel(),
            df=df,
            protected_col="gender",
            feature_cols=["gender", "pred_factor"],
            exposure_col="exposure",
            flip_values={0: 1, 1: 0},
            method="direct_flip",
        )
        assert isinstance(result, CounterfactualResult)
        assert result.method == "direct_flip"
        assert result.n_policies == 4
        # gender=0 group: original=100, cf=120. gender=1 group: original=120, cf=100.
        # Original mean: (100+100+120+120)/4 = 110
        # CF mean: (120+120+100+100)/4 = 110
        assert abs(result.premium_impact_ratio - 1.0) < 0.01

    def test_invalid_method_raises(self):
        df = pl.DataFrame({"gender": [0, 1], "f": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Unknown method"):
            counterfactual_fairness(
                model=MockCatBoostModel(),
                df=df,
                protected_col="gender",
                feature_cols=["gender", "f"],
                method="unknown_method",
            )

    def test_uses_existing_prediction_col(self):
        """If prediction_col is in df, use it instead of re-predicting."""
        df = pl.DataFrame({
            "gender": [0, 1],
            "pred_factor": [100.0, 100.0],
            "predicted": [100.0, 120.0],  # existing predictions
            "exposure": [1.0, 1.0],
        })
        result = counterfactual_fairness(
            model=MockCatBoostModel(),
            df=df,
            protected_col="gender",
            feature_cols=["gender", "pred_factor"],
            prediction_col="predicted",
            exposure_col="exposure",
            flip_values={0: 1, 1: 0},
            method="direct_flip",
        )
        # Original mean: (100 + 120) / 2 = 110
        assert abs(result.original_mean_premium - 110.0) < 1.0
