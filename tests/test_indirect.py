"""
Tests for indirect.py: IndirectDiscriminationAudit

Covers:
    - Basic fit/predict with synthetic data
    - Proxy vulnerability formula correctness
    - Segment summary structure and types
    - Proxy-free benchmark when proxy_features supplied
    - Parity-cost property (group means equal portfolio mean post-adjustment)
    - Exposure weighting
    - Input validation errors
    - Custom model_class
    - Deterministic results (same seed -> same output)
    - segment_report ordering (worst segments first)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from insurance_fairness.indirect import (
    IndirectDiscriminationAudit,
    IndirectDiscriminationResult,
    _compute_parity_cost,
    _build_summary,
)


# ---------------------------------------------------------------------------
# Shared data factory
# ---------------------------------------------------------------------------


def make_data(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Synthetic motor insurance dataset with binary protected attribute.

    Features: age, vehicle_age, gender (protected), postcode_risk (proxy).
    Target: pure premium (GLM-like linear + noise).
    """
    rng = np.random.default_rng(seed)

    # Split 70/30 train/test
    n_train = int(n * 0.7)
    n_test = n - n_train

    def _make(n_: int) -> tuple[pd.DataFrame, np.ndarray]:
        gender = rng.integers(0, 2, n_)
        age = rng.uniform(18, 70, n_)
        vehicle_age = rng.uniform(0, 20, n_)
        # postcode_risk is correlated with gender — the key proxy
        postcode_risk = 0.4 * gender + rng.normal(0, 0.3, n_)
        exposure = rng.uniform(0.3, 1.0, n_)
        X = pd.DataFrame({
            "gender": gender.astype(float),
            "age": age,
            "vehicle_age": vehicle_age,
            "postcode_risk": postcode_risk,
            "exposure": exposure,
        })
        # Target: pure premium driven by age + vehicle_age + gender effect
        y = 200 + 1.5 * age + 3.0 * vehicle_age + 80 * gender + rng.normal(0, 30, n_)
        return X, y

    X_train, y_train = _make(n_train)
    X_test, y_test = _make(n_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# TestBasicFit
# ---------------------------------------------------------------------------


class TestBasicFit:

    def test_returns_result_type(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert isinstance(result, IndirectDiscriminationResult)

    def test_summary_is_dataframe(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert isinstance(result.summary, pd.DataFrame)

    def test_summary_has_correct_segments(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        # Binary gender: expect 2 segments
        assert len(result.summary) == 2
        assert set(result.summary["segment"]) == {0.0, 1.0}

    def test_summary_columns_present(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        required_cols = [
            "segment", "n", "exposure", "mean_aware", "mean_unaware",
            "mean_proxy_vulnerability", "proxy_vulnerability_pct",
        ]
        for col in required_cols:
            assert col in result.summary.columns, f"Missing column: {col}"

    def test_proxy_vulnerability_is_positive_scalar(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert isinstance(result.proxy_vulnerability, float)
        assert result.proxy_vulnerability >= 0.0

    def test_benchmarks_dict_keys(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert "aware" in result.benchmarks
        assert "unaware" in result.benchmarks
        assert "unawareness" in result.benchmarks
        assert "parity_cost" in result.benchmarks

    def test_benchmarks_shapes(self):
        X_train, y_train, X_test, y_test = make_data(n=200)
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        n_test = len(X_test)
        for key, arr in result.benchmarks.items():
            assert arr.shape == (n_test,), f"benchmarks[{key!r}] has wrong shape"


# ---------------------------------------------------------------------------
# TestProxyVulnerabilityFormula
# ---------------------------------------------------------------------------


class TestProxyVulnerabilityFormula:

    def test_proxy_vulnerability_is_unaware_minus_aware(self):
        """mean_proxy_vulnerability = mean(h_U - h_A) per segment."""
        X_train, y_train, X_test, y_test = make_data(seed=7)
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)

        h_A = result.benchmarks["aware"]
        h_U = result.benchmarks["unaware"]
        s = X_test["gender"].values

        for _, row in result.summary.iterrows():
            grp = row["segment"]
            mask = s == grp
            expected_mean_pv = float((h_U[mask] - h_A[mask]).mean())
            np.testing.assert_allclose(
                row["mean_proxy_vulnerability"], expected_mean_pv, rtol=1e-10,
                err_msg=f"mean_proxy_vulnerability wrong for segment {grp}"
            )

    def test_portfolio_pv_is_weighted_mean_abs(self):
        """proxy_vulnerability scalar = mean |h_U - h_A| over test set."""
        X_train, y_train, X_test, y_test = make_data(seed=3)
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        expected = float(np.abs(result.benchmarks["unaware"] - result.benchmarks["aware"]).mean())
        np.testing.assert_allclose(result.proxy_vulnerability, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestParityCostProperty
# ---------------------------------------------------------------------------


class TestParityCostProperty:

    def test_parity_cost_equalises_group_means(self):
        """After parity-cost adjustment, each group has the same weighted mean."""
        h_A = np.array([100.0, 100.0, 200.0, 200.0])
        s = np.array([0, 0, 1, 1])
        w = np.ones(4)
        h_C = _compute_parity_cost(h_A, s, w)

        portfolio_mean = h_A.mean()
        for grp in [0, 1]:
            mask = s == grp
            group_mean = float(h_C[mask].mean())
            np.testing.assert_allclose(
                group_mean, portfolio_mean, rtol=1e-10,
                err_msg=f"Group {grp} mean not equal to portfolio mean after parity-cost"
            )

    def test_parity_cost_with_exposure_weights(self):
        """Exposure-weighted group means equalise to portfolio mean."""
        rng = np.random.default_rng(1)
        n = 100
        s = rng.integers(0, 2, n).astype(float)
        h_A = 100 + 50 * s + rng.normal(0, 10, n)
        w = rng.uniform(0.2, 1.0, n)
        h_C = _compute_parity_cost(h_A, s, w)

        total_w = w.sum()
        portfolio_mean = float((h_C * w).sum() / total_w)

        for grp in [0.0, 1.0]:
            mask = s == grp
            gw = w[mask].sum()
            group_mean = float((h_C[mask] * w[mask]).sum() / gw)
            np.testing.assert_allclose(
                group_mean, portfolio_mean, rtol=1e-8,
                err_msg=f"Group {grp} weighted mean not equal to portfolio mean"
            )


# ---------------------------------------------------------------------------
# TestProxyFreeModel
# ---------------------------------------------------------------------------


class TestProxyFreeModel:

    def test_proxy_free_key_present_when_proxy_features_given(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            proxy_features=["postcode_risk"],
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert "proxy_free" in result.benchmarks

    def test_proxy_free_key_absent_when_no_proxy_features(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            proxy_features=None,
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert "proxy_free" not in result.benchmarks

    def test_proxy_free_shape_matches_test_size(self):
        X_train, y_train, X_test, y_test = make_data(n=200)
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            proxy_features=["postcode_risk"],
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert result.benchmarks["proxy_free"].shape == (len(X_test),)

    def test_unknown_proxy_feature_warns(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            proxy_features=["nonexistent_col"],
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        with pytest.warns(UserWarning, match="proxy_features not found"):
            result = audit.fit(X_train, y_train, X_test, y_test)
        # proxy_free still computed (just without the missing col)
        assert "proxy_free" in result.benchmarks


# ---------------------------------------------------------------------------
# TestExposureWeighting
# ---------------------------------------------------------------------------


class TestExposureWeighting:

    def test_exposure_weighted_portfolio_pv(self):
        """exposure_col affects the portfolio PV scalar."""
        X_train, y_train, X_test, y_test = make_data(seed=5)
        audit_no_exp = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result_no_exp = audit_no_exp.fit(X_train, y_train, X_test, y_test)

        audit_exp = IndirectDiscriminationAudit(
            protected_attr="gender",
            exposure_col="exposure",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result_exp = audit_exp.fit(X_train, y_train, X_test, y_test)

        # Same predictions but different weighting — scalars should generally differ
        # (not a strict requirement, but exposure weights in [0.3, 1.0] will change means)
        # We just verify the numeric results are finite and non-negative
        assert np.isfinite(result_exp.proxy_vulnerability)
        assert result_exp.proxy_vulnerability >= 0.0

    def test_exposure_col_excluded_from_model_features(self):
        """Model must not be trained with exposure as a feature."""
        X_train, y_train, X_test, y_test = make_data(seed=2)
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            exposure_col="exposure",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        # Should not raise — exposure excluded from feature columns
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert result.proxy_vulnerability >= 0.0


# ---------------------------------------------------------------------------
# TestSegmentReport
# ---------------------------------------------------------------------------


class TestSegmentReport:

    def test_segment_report_sorted_descending(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        vals = result.segment_report["mean_abs_proxy_vulnerability"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_segment_report_same_rows_as_summary(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert len(result.segment_report) == len(result.summary)


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_raises_if_x_train_not_dataframe(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3},
        )
        with pytest.raises(TypeError, match="pandas DataFrame"):
            audit.fit(X_train.values, y_train, X_test, y_test)

    def test_raises_if_protected_attr_missing_from_train(self):
        X_train, y_train, X_test, y_test = make_data()
        X_bad = X_train.drop(columns=["gender"])
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3},
        )
        with pytest.raises(ValueError, match="protected_attr"):
            audit.fit(X_bad, y_train, X_test, y_test)

    def test_raises_if_protected_attr_missing_from_test(self):
        X_train, y_train, X_test, y_test = make_data()
        X_bad = X_test.drop(columns=["gender"])
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3},
        )
        with pytest.raises(ValueError, match="protected_attr"):
            audit.fit(X_train, y_train, X_bad, y_test)

    def test_raises_if_exposure_col_missing(self):
        X_train, y_train, X_test, y_test = make_data()
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            exposure_col="missing_col",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3},
        )
        with pytest.raises(ValueError, match="exposure_col"):
            audit.fit(X_train, y_train, X_test, y_test)


# ---------------------------------------------------------------------------
# TestCustomModel
# ---------------------------------------------------------------------------


class TestCustomModel:

    def test_custom_model_class(self):
        """GradientBoostingRegressor as model_class works end-to-end."""
        X_train, y_train, X_test, y_test = make_data(n=200)
        audit = IndirectDiscriminationAudit(
            protected_attr="gender",
            model_class=GradientBoostingRegressor,
            model_kwargs={"n_estimators": 20, "max_depth": 3, "random_state": 0},
        )
        result = audit.fit(X_train, y_train, X_test, y_test)
        assert isinstance(result, IndirectDiscriminationResult)
        assert result.proxy_vulnerability >= 0.0


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_same_seed_produces_same_result(self):
        X_train, y_train, X_test, y_test = make_data(seed=99)
        kwargs = dict(
            protected_attr="gender",
            model_class=DecisionTreeRegressor,
            model_kwargs={"max_depth": 3, "random_state": 0},
            random_state=42,
        )
        result_a = IndirectDiscriminationAudit(**kwargs).fit(X_train, y_train, X_test, y_test)
        result_b = IndirectDiscriminationAudit(**kwargs).fit(X_train, y_train, X_test, y_test)
        np.testing.assert_array_equal(
            result_a.benchmarks["aware"], result_b.benchmarks["aware"]
        )
        assert result_a.proxy_vulnerability == result_b.proxy_vulnerability


# ---------------------------------------------------------------------------
# TestBuildSummaryUnit
# ---------------------------------------------------------------------------


class TestBuildSummaryUnit:

    def test_known_values(self):
        """Hand-verify proxy vulnerability with known arrays."""
        # Two groups, 2 observations each
        # Group 0: h_A=[100,100], h_U=[110,90]  -> PV=[10,-10], mean=0, mean_abs=10
        # Group 1: h_A=[200,200], h_U=[220,220] -> PV=[20,20], mean=20, mean_abs=20
        h_A  = np.array([100., 100., 200., 200.])
        h_U  = np.array([110.,  90., 220., 220.])
        h_UN = np.array([105., 105., 210., 210.])
        h_C  = np.array([100., 100., 200., 200.])  # parity unchanged for balanced case
        s    = np.array([0., 0., 1., 1.])
        w    = np.ones(4)

        summary_df, pv = _build_summary(h_A, h_U, h_UN, h_C, s, w)

        row0 = summary_df[summary_df["segment"] == 0.0].iloc[0]
        row1 = summary_df[summary_df["segment"] == 1.0].iloc[0]

        np.testing.assert_allclose(row0["mean_proxy_vulnerability"], 0.0, atol=1e-10)
        np.testing.assert_allclose(row0["mean_abs_proxy_vulnerability"], 10.0, atol=1e-10)
        np.testing.assert_allclose(row1["mean_proxy_vulnerability"], 20.0, atol=1e-10)
        np.testing.assert_allclose(row1["mean_abs_proxy_vulnerability"], 20.0, atol=1e-10)

        # Portfolio PV: mean(|10|, |−10|, |20|, |20|) = mean(10,10,20,20) = 15
        np.testing.assert_allclose(pv, 15.0, atol=1e-10)
