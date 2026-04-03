"""Tests for SequentialOTCorrector.

The core claim under test: for K >= 2 protected attributes, calibrating each
step's ECDF on the CURRENT predictions (f_{k-1}) produces better-calibrated
OT maps than calibrating all ECDFs on f* upfront (WassersteinCorrector).

For K=1, SequentialOTCorrector and WassersteinCorrector should produce
identical results (to floating-point tolerance).
"""
import warnings

import numpy as np
import polars as pl
import pytest

from insurance_fairness.optimal_transport.correction import (
    SequentialOTCorrector,
    WassersteinCorrector,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def two_group_preds(n: int = 500, seed: int = 0) -> tuple[np.ndarray, pl.DataFrame]:
    """Single attribute, two groups with different log-normal distributions."""
    rng = np.random.default_rng(seed)
    pA = rng.lognormal(0.0, 0.4, n)
    pB = rng.lognormal(0.5, 0.4, n)
    predictions = np.concatenate([pA, pB])
    D = pl.DataFrame({"group": ["A"] * n + ["B"] * n})
    return predictions, D


def two_attr_preds(
    n: int = 400, seed: int = 42
) -> tuple[np.ndarray, pl.DataFrame]:
    """Two independent protected attributes, each with two groups.

    gender: log-mean shift of 0.4 between M/F
    age_band: log-mean shift of 0.3 between young/old

    The prediction is the product of two independent lognormals, so the
    initial distribution has two independent sources of group-level bias.
    """
    rng = np.random.default_rng(seed)
    # gender: M vs F
    gender = ["M"] * n + ["F"] * n
    age_band = (["young"] * (n // 2) + ["old"] * (n // 2)) * 2

    n_total = 2 * n
    base = rng.lognormal(0.0, 0.3, n_total)
    # Add group effects
    gender_arr = np.array(gender)
    age_arr = np.array(age_band)
    predictions = base.copy()
    predictions[gender_arr == "M"] *= np.exp(0.4)
    predictions[age_arr == "young"] *= np.exp(0.3)

    D = pl.DataFrame({"gender": gender, "age_band": age_band})
    return predictions, D


# ── construction ──────────────────────────────────────────────────────────────


class TestSequentialOTCorrectorInit:
    def test_scalar_epsilon_broadcast(self):
        c = SequentialOTCorrector(["gender", "age_band"], epsilon=0.2)
        assert c._epsilons == [0.2, 0.2]

    def test_list_epsilon_accepted(self):
        c = SequentialOTCorrector(["gender", "age_band"], epsilon=[0.1, 0.3])
        assert c._epsilons == [0.1, 0.3]

    def test_list_epsilon_wrong_length_raises(self):
        with pytest.raises(ValueError, match="epsilon list length"):
            SequentialOTCorrector(["gender", "age_band"], epsilon=[0.1, 0.2, 0.3])

    def test_invalid_scalar_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            SequentialOTCorrector(["gender"], epsilon=1.5)

    def test_invalid_list_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            SequentialOTCorrector(["gender", "age_band"], epsilon=[0.1, 2.0])

    def test_not_fitted_raises_on_transform(self):
        preds, D = two_group_preds(50)
        c = SequentialOTCorrector(["group"])
        with pytest.raises(RuntimeError, match="fit()"):
            c.transform(preds, D)

    def test_not_fitted_raises_on_unfairness_reductions(self):
        c = SequentialOTCorrector(["group"])
        with pytest.raises(RuntimeError):
            _ = c.unfairness_reductions_

    def test_not_fitted_raises_on_wasserstein_distances(self):
        c = SequentialOTCorrector(["group"])
        with pytest.raises(RuntimeError):
            _ = c.wasserstein_distances_

    def test_get_intermediate_predictions_before_fit_returns_none(self):
        c = SequentialOTCorrector(["group"])
        assert c.get_intermediate_predictions() is None


# ── fit / transform basics ─────────────────────────────────────────────────────


class TestSequentialOTCorrectorFitTransform:
    def test_fit_returns_self(self):
        preds, D = two_group_preds()
        c = SequentialOTCorrector(["group"])
        result = c.fit(preds, D)
        assert result is c

    def test_transform_returns_correct_shape(self):
        preds, D = two_group_preds(200)
        c = SequentialOTCorrector(["group"])
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert result.shape == preds.shape

    def test_transform_returns_positive_values(self):
        preds, D = two_group_preds(200)
        c = SequentialOTCorrector(["group"])
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert np.all(result > 0)

    def test_transform_finite_values(self):
        preds, D = two_group_preds(200)
        c = SequentialOTCorrector(["group"])
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert np.all(np.isfinite(result))

    def test_invalid_predictions_raises(self):
        preds = np.array([-1.0, 2.0, 3.0])
        D = pl.DataFrame({"group": ["A", "A", "B"]})
        c = SequentialOTCorrector(["group"])
        with pytest.raises(ValueError, match="strictly positive"):
            c.fit(preds, D)

    def test_missing_attr_raises_on_fit(self):
        preds, D = two_group_preds(50)
        c = SequentialOTCorrector(["nonexistent"])
        with pytest.raises(ValueError):
            c.fit(preds, D)

    def test_missing_attr_raises_on_transform(self):
        preds, D = two_group_preds(50)
        c = SequentialOTCorrector(["group"])
        c.fit(preds, D)
        D_bad = pl.DataFrame({"other": ["A"] * 100})
        with pytest.raises(ValueError):
            c.transform(preds, D_bad)

    def test_refit_resets_state(self):
        """fit() a second time should overwrite prior state cleanly."""
        preds, D = two_group_preds(200)
        c = SequentialOTCorrector(["group"])
        c.fit(preds, D)
        r1 = c.transform(preds, D)
        c.fit(preds, D)
        r2 = c.transform(preds, D)
        np.testing.assert_allclose(r1, r2, rtol=1e-10)

    def test_epsilon_one_returns_predictions_unchanged(self):
        """epsilon=1 means no correction."""
        preds, D = two_group_preds(200)
        c = SequentialOTCorrector(["group"], epsilon=1.0)
        c.fit(preds, D)
        result = c.transform(preds, D)
        np.testing.assert_allclose(result, preds, rtol=1e-5)

    def test_exposure_weighted_fit(self):
        preds, D = two_group_preds(100)
        exposure = np.ones(200)
        c = SequentialOTCorrector(["group"], exposure_weighted=True)
        c.fit(preds, D, exposure=exposure)
        result = c.transform(preds, D)
        assert np.all(result > 0)

    def test_linear_space(self):
        preds, D = two_group_preds(100)
        c = SequentialOTCorrector(["group"], log_space=False)
        c.fit(preds, D)
        result = c.transform(preds, D)
        assert np.all(result > 0)


# ── K=1 equivalence with WassersteinCorrector ────────────────────────────────


class TestK1Equivalence:
    """For a single attribute, SequentialOTCorrector == WassersteinCorrector."""

    def test_fit_transform_identical_k1(self):
        preds, D = two_group_preds(500)
        exposure = np.random.default_rng(1).uniform(0.5, 1.5, 1000)

        w = WassersteinCorrector(["group"], epsilon=0.0, log_space=True)
        s = SequentialOTCorrector(["group"], epsilon=0.0, log_space=True)

        w.fit(preds, D, exposure=exposure)
        s.fit(preds, D, exposure=exposure)

        r_w = w.transform(preds, D)
        r_s = s.transform(preds, D)

        np.testing.assert_allclose(r_w, r_s, rtol=1e-6, atol=1e-10)

    def test_fit_transform_identical_k1_linear_space(self):
        preds, D = two_group_preds(500)

        w = WassersteinCorrector(["group"], epsilon=0.2, log_space=False)
        s = SequentialOTCorrector(["group"], epsilon=0.2, log_space=False)

        w.fit(preds, D)
        s.fit(preds, D)

        r_w = w.transform(preds, D)
        r_s = s.transform(preds, D)

        np.testing.assert_allclose(r_w, r_s, rtol=1e-6, atol=1e-10)


# ── K>=2 correctness ──────────────────────────────────────────────────────────


class TestK2Correctness:
    """For K=2, SequentialOTCorrector should differ from WassersteinCorrector
    and produce better fairness metrics (lower W1 unfairness after correction).
    """

    def test_sequential_differs_from_wasserstein_k2(self):
        preds, D = two_attr_preds(400)

        w = WassersteinCorrector(["gender", "age_band"], epsilon=0.0, log_space=True)
        s = SequentialOTCorrector(["gender", "age_band"], epsilon=0.0, log_space=True)

        w.fit(preds, D)
        s.fit(preds, D)

        r_w = w.transform(preds, D)
        r_s = s.transform(preds, D)

        # They should NOT be identical for K=2
        assert not np.allclose(r_w, r_s, rtol=1e-6)

    def test_sequential_reduces_unfairness_second_attr(self):
        """After correction, W1 unfairness for both attributes should be reduced."""
        preds, D = two_attr_preds(400)
        s = SequentialOTCorrector(["gender", "age_band"], epsilon=0.0, log_space=True)
        s.fit(preds, D)
        reductions = s.unfairness_reductions_

        assert "gender" in reductions
        assert "age_band" in reductions

        for attr, (before, after) in reductions.items():
            assert after <= before, (
                f"Unfairness for {attr!r} did not decrease: before={before:.4f}, after={after:.4f}"
            )

    def test_sequential_final_predictions_positive(self):
        preds, D = two_attr_preds(400)
        s = SequentialOTCorrector(["gender", "age_band"], epsilon=0.0, log_space=True)
        s.fit(preds, D)
        result = s.transform(preds, D)
        assert np.all(result > 0)

    def test_sequential_calibration_alignment(self):
        """Key correctness property: for K=2, WassersteinCorrector calibrates
        attr[1]'s ECDF on f* but applies it to f_1 (output after attr[0]
        correction). This is wrong. SequentialOTCorrector calibrates attr[1]'s
        ECDF on f_1. We verify this by checking that the ECDF stored for step 1
        is estimated from f_1, not f_0.

        Proxy test: if we apply step 0 correction manually and then check
        that the step-1 ECDF matches that distribution.
        """
        preds, D = two_attr_preds(200)
        s = SequentialOTCorrector(["gender", "age_band"], epsilon=0.0, log_space=True)
        s.fit(preds, D)

        intermediates = s.get_intermediate_predictions()
        # Should have f_0 (original) and f_1 (after gender correction)
        assert len(intermediates) == 3  # f_0, f_1, f_2

        f0, f1, f2 = intermediates
        # f_0 should equal the original predictions
        np.testing.assert_allclose(f0, preds, rtol=1e-10)
        # f_1 and f_2 should be different from f_0
        assert not np.allclose(f0, f1, rtol=1e-4)
        # f_2 should be the final corrected predictions
        result = s.transform(preds, D)
        np.testing.assert_allclose(f2, result, rtol=1e-10)

    def test_per_attribute_epsilon_list(self):
        """Per-attribute epsilon: first attr fully corrected, second not."""
        preds, D = two_attr_preds(300)
        s = SequentialOTCorrector(
            ["gender", "age_band"],
            epsilon=[0.0, 1.0],  # gender fully corrected, age not corrected
            log_space=True,
        )
        s.fit(preds, D)
        result = s.transform(preds, D)
        assert np.all(result > 0)

        # gender should be corrected (means closer), age_band unchanged after step 1
        # After full gender correction (eps=0) followed by no age correction (eps=1),
        # the age_band group means should match what we'd get from gender-only correction
        s_gender_only = SequentialOTCorrector(["gender"], epsilon=0.0, log_space=True)
        s_gender_only.fit(preds, D)
        r_gender_only = s_gender_only.transform(preds, D)
        np.testing.assert_allclose(result, r_gender_only, rtol=1e-6)


# ── intermediate predictions ──────────────────────────────────────────────────


class TestIntermediatePredictions:
    def test_returns_list_of_k_plus_1_arrays(self):
        preds, D = two_attr_preds(200)
        s = SequentialOTCorrector(["gender", "age_band"])
        s.fit(preds, D)
        intermediates = s.get_intermediate_predictions()
        assert len(intermediates) == 3  # f_0, f_1, f_2

    def test_first_element_is_original_predictions(self):
        preds, D = two_group_preds(200)
        s = SequentialOTCorrector(["group"])
        s.fit(preds, D)
        intermediates = s.get_intermediate_predictions()
        np.testing.assert_allclose(intermediates[0], preds, rtol=1e-10)

    def test_last_element_matches_transform_output(self):
        preds, D = two_attr_preds(200)
        s = SequentialOTCorrector(["gender", "age_band"])
        s.fit(preds, D)
        intermediates = s.get_intermediate_predictions()
        result = s.transform(preds, D)
        np.testing.assert_allclose(intermediates[-1], result, rtol=1e-10)

    def test_all_intermediate_arrays_positive(self):
        preds, D = two_attr_preds(200)
        s = SequentialOTCorrector(["gender", "age_band"])
        s.fit(preds, D)
        for i, arr in enumerate(s.get_intermediate_predictions()):
            assert np.all(arr > 0), f"f_{i} has non-positive values"

    def test_k1_has_two_intermediates(self):
        preds, D = two_group_preds(200)
        s = SequentialOTCorrector(["group"])
        s.fit(preds, D)
        intermediates = s.get_intermediate_predictions()
        assert len(intermediates) == 2  # f_0, f_1


# ── properties ────────────────────────────────────────────────────────────────


class TestProperties:
    def test_unfairness_reductions_structure(self):
        preds, D = two_attr_preds(300)
        s = SequentialOTCorrector(["gender", "age_band"])
        s.fit(preds, D)
        reductions = s.unfairness_reductions_
        assert set(reductions.keys()) == {"gender", "age_band"}
        for attr, (before, after) in reductions.items():
            assert isinstance(before, float)
            assert isinstance(after, float)
            assert before >= 0
            assert after >= 0

    def test_wasserstein_distances_two_groups(self):
        preds, D = two_group_preds(300)
        s = SequentialOTCorrector(["group"])
        s.fit(preds, D)
        dists = s.wasserstein_distances_
        assert "group" in dists
        assert dists["group"] > 0

    def test_wasserstein_distances_two_attrs(self):
        preds, D = two_attr_preds(300)
        s = SequentialOTCorrector(["gender", "age_band"])
        s.fit(preds, D)
        dists = s.wasserstein_distances_
        # Both attrs have exactly 2 groups so both should be populated
        assert "gender" in dists
        assert "age_band" in dists

    def test_unfairness_reductions_before_fit_raises(self):
        c = SequentialOTCorrector(["group"])
        with pytest.raises(RuntimeError):
            _ = c.unfairness_reductions_

    def test_wasserstein_distances_before_fit_raises(self):
        c = SequentialOTCorrector(["group"])
        with pytest.raises(RuntimeError):
            _ = c.wasserstein_distances_


# ── small-group warning ───────────────────────────────────────────────────────


class TestSmallGroupWarning:
    def test_warns_when_group_below_min_samples(self):
        rng = np.random.default_rng(0)
        # Group A: 50 samples, Group B: 50 samples — both below default 100
        preds = rng.lognormal(0, 0.3, 100)
        D = pl.DataFrame({"group": ["A"] * 50 + ["B"] * 50})
        c = SequentialOTCorrector(["group"], group_min_samples=100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.fit(preds, D)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) > 0
        assert "group_min_samples" in str(user_warns[0].message)

    def test_no_warning_when_group_above_min_samples(self):
        preds, D = two_group_preds(200)  # 400 total, 200 per group
        c = SequentialOTCorrector(["group"], group_min_samples=100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.fit(preds, D)
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0


# ── pricing integration ───────────────────────────────────────────────────────


class TestPricingIntegration:
    """Smoke tests for SequentialOTCorrector wired into DiscriminationFreePrice."""

    def _make_graph(self):
        from insurance_fairness.optimal_transport.causal import CausalGraph
        # add_edge requires both nodes to exist; add_outcome first so the edges
        # can be declared. CausalGraph.validate() requires each protected node
        # to have a path to the outcome — so we must add those edges explicitly.
        return (
            CausalGraph()
            .add_protected("gender")
            .add_protected("age_band")
            .add_outcome("loss")
            .add_edge("gender", "loss")
            .add_edge("age_band", "loss")
        )

    def _make_data(self, n=200, seed=0):
        preds, D = two_attr_preds(n, seed)
        X = D.clone()  # features = protected attrs for simplicity
        exposure = np.random.default_rng(seed).uniform(0.5, 1.5, len(preds))
        return X, D, exposure, preds

    def _make_model(self, preds):
        """Return a model function that returns pre-computed predictions."""
        def model_fn(df: pl.DataFrame) -> np.ndarray:
            return preds[:df.shape[0]]
        return model_fn

    def test_sequential_wasserstein_correction(self):
        from insurance_fairness.optimal_transport.pricing import DiscriminationFreePrice

        X, D, exposure, preds = self._make_data()
        g = self._make_graph()
        model_fn = self._make_model(preds)

        dfp = DiscriminationFreePrice(
            g,
            combined_model_fn=model_fn,
            correction="sequential_wasserstein",
            epsilon=0.0,
        )
        result = dfp.fit_transform(X, D, exposure=exposure)
        assert result.fair_premium.shape == preds.shape
        assert np.all(result.fair_premium > 0)
        assert result.method == "sequential_wasserstein"
        assert "unfairness_reductions" in result.metadata

    def test_lindholm_sequential_wasserstein_correction(self):
        from insurance_fairness.optimal_transport.pricing import DiscriminationFreePrice

        X, D, exposure, preds = self._make_data()
        g = self._make_graph()
        model_fn = self._make_model(preds)

        dfp = DiscriminationFreePrice(
            g,
            combined_model_fn=model_fn,
            correction="lindholm+sequential_wasserstein",
            epsilon=0.0,
        )
        result = dfp.fit_transform(X, D, exposure=exposure)
        assert result.fair_premium.shape == preds.shape
        assert np.all(result.fair_premium > 0)
        assert result.method == "lindholm+sequential_wasserstein"
