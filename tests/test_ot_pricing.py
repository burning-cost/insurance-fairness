"""Tests for DiscriminationFreePrice and PricingResult."""
import numpy as np
import polars as pl
import pytest

from insurance_fairness.optimal_transport.causal import CausalGraph
from insurance_fairness.optimal_transport.pricing import DiscriminationFreePrice, PricingResult


# ── fixtures ──────────────────────────────────────────────────────────────────


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


def make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["M", "F"], n).tolist()
    claims = rng.choice(["yes", "no"], n, p=[0.3, 0.7]).tolist()
    mileage = rng.integers(5000, 30000, n).tolist()
    X = pl.DataFrame({
        "claims_history": claims,
        "annual_mileage": mileage,
        "gender": gender,
    })
    D = pl.DataFrame({"gender": gender})
    exposure = rng.uniform(0.5, 1.0, n)
    return X, D, exposure


def make_model(df: pl.DataFrame) -> np.ndarray:
    n = df.shape[0]
    base = np.ones(n) * 0.1
    if "gender" in df.columns:
        base += (df["gender"] == "M").to_numpy() * 0.02
    if "claims_history" in df.columns:
        base += (df["claims_history"] == "yes").to_numpy() * 0.05
    return base


def make_freq_model(df: pl.DataFrame) -> np.ndarray:
    return make_model(df) * 0.6


def make_sev_model(df: pl.DataFrame) -> np.ndarray:
    return make_model(df) * 0.4 + 0.05


# ── DiscriminationFreePrice construction ──────────────────────────────────────


class TestDiscriminationFreePriceInit:
    def test_combined_model_accepted(self):
        g = make_graph()
        dfp = DiscriminationFreePrice(g, combined_model_fn=make_model)
        assert dfp is not None

    def test_freq_sev_models_accepted(self):
        g = make_graph()
        dfp = DiscriminationFreePrice(
            g,
            frequency_model_fn=make_freq_model,
            severity_model_fn=make_sev_model,
        )
        assert dfp is not None

    def test_no_model_raises(self):
        g = make_graph()
        with pytest.raises(ValueError, match="combined_model_fn"):
            DiscriminationFreePrice(g)

    def test_only_freq_without_sev_raises(self):
        g = make_graph()
        with pytest.raises(ValueError):
            DiscriminationFreePrice(g, frequency_model_fn=make_freq_model)

    def test_invalid_graph_raises(self):
        g = CausalGraph().add_protected("gender")  # no outcome
        with pytest.raises(ValueError):
            DiscriminationFreePrice(g, combined_model_fn=make_model)


# ── fit() ─────────────────────────────────────────────────────────────────────


class TestDiscriminationFreePriceFit:
    def test_fit_returns_self(self):
        g = make_graph()
        X, D, exp = make_data()
        dfp = DiscriminationFreePrice(g, combined_model_fn=make_model)
        result = dfp.fit(X, D, exposure=exp)
        assert result is dfp

    def test_fit_sets_is_fitted(self):
        g = make_graph()
        X, D, exp = make_data()
        dfp = DiscriminationFreePrice(g, combined_model_fn=make_model)
        dfp.fit(X, D)
        assert dfp._is_fitted

    def test_fit_with_freq_sev(self):
        g = make_graph()
        X, D, exp = make_data()
        dfp = DiscriminationFreePrice(
            g,
            frequency_model_fn=make_freq_model,
            severity_model_fn=make_sev_model,
        )
        dfp.fit(X, D, exposure=exp)
        assert dfp._is_fitted

    def test_transform_before_fit_raises(self):
        g = make_graph()
        X, D, exp = make_data(100)
        dfp = DiscriminationFreePrice(g, combined_model_fn=make_model)
        with pytest.raises(RuntimeError, match="fit()"):
            dfp.transform(X, D)


# ── transform() ───────────────────────────────────────────────────────────────


class TestDiscriminationFreePriceTransform:
    def _fitted_dfp(self, correction="lindholm", n=200):
        g = make_graph()
        X, D, exp = make_data(n)
        dfp = DiscriminationFreePrice(
            g,
            combined_model_fn=make_model,
            correction=correction,
        )
        dfp.fit(X, D, exposure=exp)
        return dfp, X, D, exp

    def test_returns_pricing_result(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D, exposure=exp)
        assert isinstance(result, PricingResult)

    def test_fair_premium_shape(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D, exposure=exp)
        assert result.fair_premium.shape == (X.shape[0],)

    def test_fair_premium_positive(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D, exposure=exp)
        assert np.all(result.fair_premium > 0)

    def test_best_estimate_shape(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D, exposure=exp)
        assert result.best_estimate.shape == (X.shape[0],)

    def test_bias_correction_factor_near_one(self):
        dfp, X, D, exp = self._fitted_dfp(n=500)
        result = dfp.transform(X, D, exposure=exp)
        assert abs(result.bias_correction_factor - 1.0) < 0.1

    def test_method_attribute(self):
        dfp, X, D, exp = self._fitted_dfp(correction="lindholm")
        result = dfp.transform(X, D, exposure=exp)
        assert result.method == "lindholm"

    def test_protected_attrs_attribute(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D, exposure=exp)
        assert "gender" in result.protected_attrs

    def test_lindholm_reduces_group_disparity(self):
        """After Lindholm correction, gender gap in mean premium should reduce."""
        g = make_graph()
        X, D, exp = make_data(1000)
        dfp = DiscriminationFreePrice(g, combined_model_fn=make_model, correction="lindholm")
        dfp.fit(X, D, exposure=exp)
        result = dfp.transform(X, D, exposure=exp)

        gender = D["gender"].to_numpy()
        mean_before_M = np.mean(result.best_estimate[gender == "M"])
        mean_before_F = np.mean(result.best_estimate[gender == "F"])
        mean_after_M = np.mean(result.fair_premium[gender == "M"])
        mean_after_F = np.mean(result.fair_premium[gender == "F"])

        gap_before = abs(mean_before_M - mean_before_F)
        gap_after = abs(mean_after_M - mean_after_F)
        assert gap_after < gap_before

    def test_wasserstein_correction(self):
        dfp, X, D, exp = self._fitted_dfp(correction="wasserstein")
        result = dfp.transform(X, D, exposure=exp)
        assert result.fair_premium.shape == (X.shape[0],)
        assert np.all(result.fair_premium > 0)

    def test_lindholm_plus_wasserstein(self):
        dfp, X, D, exp = self._fitted_dfp(correction="lindholm+wasserstein")
        result = dfp.transform(X, D, exposure=exp)
        assert result.fair_premium.shape == (X.shape[0],)
        assert np.all(result.fair_premium > 0)

    def test_freq_sev_result_has_components(self):
        g = make_graph()
        X, D, exp = make_data(200)
        dfp = DiscriminationFreePrice(
            g,
            frequency_model_fn=make_freq_model,
            severity_model_fn=make_sev_model,
        )
        dfp.fit(X, D, exposure=exp)
        result = dfp.transform(X, D, exposure=exp)
        assert result.freq_fair is not None
        assert result.sev_fair is not None
        assert result.freq_fair.shape == (200,)
        assert result.sev_fair.shape == (200,)

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        g = make_graph()
        X, D, exp = make_data(100)

        dfp1 = DiscriminationFreePrice(g, combined_model_fn=make_model)
        r1 = dfp1.fit_transform(X, D, exposure=exp)

        dfp2 = DiscriminationFreePrice(g, combined_model_fn=make_model)
        dfp2.fit(X, D, exposure=exp)
        r2 = dfp2.transform(X, D, exposure=exp)

        assert np.allclose(r1.fair_premium, r2.fair_premium)

    def test_metadata_contains_portfolio_weights(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D, exposure=exp)
        assert "portfolio_weights" in result.metadata
        assert "gender" in result.metadata["portfolio_weights"]

    def test_no_exposure_defaults_to_ones(self):
        dfp, X, D, exp = self._fitted_dfp()
        result = dfp.transform(X, D)
        assert result.fair_premium.shape == (X.shape[0],)
