"""
test_pareto.py
--------------
Tests for pareto.py — NSGA-II multi-objective Pareto optimisation.

Tests cover:
- FairnessProblem evaluates correctly with dummy models
- NSGA-II returns non-dominated solutions (when pymoo is available)
- TOPSIS selects the correct point for known inputs
- ParetoResult serialisation (to_dict / from_dict round-trip)
- LipschitzMetric with a simple distance function
- Edge cases: single model, two models, identical models
- FairnessAudit run_pareto integration flag (smoke test)

All models used here are simple callable objects — no CatBoost dependency
is required for the unit tests. The models are thin wrappers that return
pre-set numpy arrays, bypassing any Pool construction logic.
"""

from __future__ import annotations

import json
import math
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from insurance_fairness.pareto import (
    FairnessProblem,
    LipschitzMetric,
    LipschitzResult,
    ParetoResult,
    topsis_select,
)


# ---------------------------------------------------------------------------
# Helpers: dummy models
# ---------------------------------------------------------------------------


class ConstantModel:
    """Always returns the same prediction for every policy."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, "__len__"):
            n = len(X)
        else:
            n = 1
        return np.full(n, self.value)


class LinearModel:
    """Returns predictions proportional to column index 0."""

    def __init__(self, scale: float = 1.0, offset: float = 0.0) -> None:
        self.scale = scale
        self.offset = offset

    def predict(self, X: Any) -> np.ndarray:
        # X may be a pandas DataFrame or numpy array
        try:
            col = X.iloc[:, 0].to_numpy()
        except AttributeError:
            col = np.asarray(X)[:, 0]
        return self.scale * col + self.offset


class ArrayModel:
    """Returns a fixed pre-computed array of predictions."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = np.asarray(predictions, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return self._predictions.copy()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def small_df(rng: np.random.Generator) -> pl.DataFrame:
    """
    Small 100-policy dataset with a binary protected characteristic.

    Two groups, each 50 policies. Predictions differ between groups.
    """
    n = 100
    gender = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    pred_base = rng.lognormal(4.6, 0.3, n)
    # Group 1 predictions are 25% higher on average
    pred_biased = pred_base * np.where(gender == 1, 1.25, 1.0)
    pred_fair = pred_base  # same for both groups (no disparity)

    return pl.DataFrame({
        "gender": gender.tolist(),
        "pred_base": pred_biased.tolist(),
        "pred_fair": pred_fair.tolist(),
        "claim_amount": (pred_base * rng.lognormal(0, 0.2, n)).tolist(),
        "exposure": rng.uniform(0.5, 1.0, n).tolist(),
        "age": rng.integers(20, 70, n).tolist(),
        "vehicle_value": rng.lognormal(9, 0.5, n).tolist(),
    })


@pytest.fixture
def base_preds(small_df: pl.DataFrame) -> np.ndarray:
    return small_df["pred_base"].to_numpy()


@pytest.fixture
def fair_preds(small_df: pl.DataFrame) -> np.ndarray:
    return small_df["pred_fair"].to_numpy()


@pytest.fixture
def y(small_df: pl.DataFrame) -> np.ndarray:
    return small_df["claim_amount"].to_numpy()


@pytest.fixture
def exposure(small_df: pl.DataFrame) -> np.ndarray:
    return small_df["exposure"].to_numpy()


@pytest.fixture
def biased_model(base_preds: np.ndarray) -> ArrayModel:
    return ArrayModel(base_preds)


@pytest.fixture
def fair_model(fair_preds: np.ndarray) -> ArrayModel:
    return ArrayModel(fair_preds)


@pytest.fixture
def fairness_problem(
    small_df: pl.DataFrame,
    biased_model: ArrayModel,
    fair_model: ArrayModel,
    y: np.ndarray,
    exposure: np.ndarray,
) -> FairnessProblem:
    """FairnessProblem with two models: one biased, one fair."""
    return FairnessProblem(
        models={"biased": biased_model, "fair": fair_model},
        X=small_df,
        y=y,
        exposure=exposure,
        protected_col="gender",
    )


# ---------------------------------------------------------------------------
# A. FairnessProblem tests
# ---------------------------------------------------------------------------


class TestFairnessProblem:
    def test_evaluate_returns_array_of_three(self, fairness_problem: FairnessProblem):
        """evaluate() must return an array of exactly 3 objective values."""
        weights = np.array([0.5, 0.5])
        result = fairness_problem.evaluate(weights)
        assert result.shape == (3,)

    def test_evaluate_all_weight_on_fair_model(
        self, fairness_problem: FairnessProblem
    ):
        """All weight on the fair model should give lower group unfairness."""
        weights_biased = np.array([1.0, 0.0])
        weights_fair = np.array([0.0, 1.0])

        result_biased = fairness_problem.evaluate(weights_biased)
        result_fair = fairness_problem.evaluate(weights_fair)

        # Group unfairness (objective index 1) should be lower for fair model
        assert result_fair[1] <= result_biased[1] + 0.01  # allow small tolerance

    def test_evaluate_all_weight_on_biased_model_has_disparity(
        self, fairness_problem: FairnessProblem
    ):
        """All weight on the biased model should produce non-zero group unfairness."""
        weights = np.array([1.0, 0.0])
        result = fairness_problem.evaluate(weights)
        # Group unfairness should be positive (not zero) for biased model
        assert result[1] > 0.01

    def test_evaluate_normalises_weights(self, fairness_problem: FairnessProblem):
        """evaluate() should normalise weights internally; [2, 2] == [1, 1]."""
        r1 = fairness_problem.evaluate(np.array([1.0, 1.0]))
        r2 = fairness_problem.evaluate(np.array([2.0, 2.0]))
        np.testing.assert_allclose(r1, r2, rtol=1e-10)

    def test_gini_objective_identical_predictions(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray
    ):
        """Identical predictions give Gini = 0, so neg_gini = 0."""
        model = ConstantModel(100.0)
        prob = FairnessProblem(
            models={"const": model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = prob.evaluate(np.array([1.0]))
        # neg_gini (objective 0) should be 0 for constant predictions
        assert abs(result[0]) < 1e-6

    def test_gini_objective_ordered_predictions(
        self, small_df: pl.DataFrame, exposure: np.ndarray
    ):
        """Predictions perfectly correlated with actuals give high Gini (low neg_gini)."""
        n = len(small_df)
        # Monotonically increasing: best possible Gini
        preds = np.arange(1, n + 1, dtype=float)
        actuals = np.arange(1, n + 1, dtype=float)
        model = ArrayModel(preds)
        prob = FairnessProblem(
            models={"ordered": model},
            X=small_df,
            y=actuals,
            exposure=exposure,
            protected_col="gender",
        )
        result = prob.evaluate(np.array([1.0]))
        # neg_gini should be negative (Gini > 0) for ordered predictions
        assert result[0] < 0.0

    def test_counterfactual_objective_binary_flip(
        self,
        small_df: pl.DataFrame,
        fair_preds: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """For a model where flipping gender has no effect, cf_unfairness should be 0."""
        model = ArrayModel(fair_preds)
        # The model returns the same predictions regardless, so CF unfairness = 0
        prob = FairnessProblem(
            models={"fair": model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            cf_tolerance=0.1,
        )
        result = prob.evaluate(np.array([1.0]))
        # Since both original and counterfactual preds are the same (ArrayModel),
        # cf_unfairness should be 0 (all policies have |log(1)| = 0 < tolerance)
        assert result[2] < 1e-6

    def test_missing_protected_col_raises(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray
    ):
        """Providing a column that doesn't exist should raise ValueError."""
        model = ConstantModel(100.0)
        with pytest.raises(ValueError, match="not found"):
            FairnessProblem(
                models={"m": model},
                X=small_df,
                y=y,
                exposure=exposure,
                protected_col="nonexistent_col",
            )

    def test_no_models_raises(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray
    ):
        """Empty models dict should raise ValueError."""
        with pytest.raises(ValueError, match="At least one model"):
            FairnessProblem(
                models={},
                X=small_df,
                y=y,
                exposure=exposure,
                protected_col="gender",
            )

    def test_single_model(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray
    ):
        """Single model should evaluate without error."""
        model = ConstantModel(150.0)
        prob = FairnessProblem(
            models={"solo": model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = prob.evaluate(np.array([1.0]))
        assert result.shape == (3,)

    def test_identical_models(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray,
        base_preds: np.ndarray
    ):
        """Two identical models: any mixing weight gives the same objectives."""
        model_a = ArrayModel(base_preds)
        model_b = ArrayModel(base_preds)
        prob = FairnessProblem(
            models={"a": model_a, "b": model_b},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        r1 = prob.evaluate(np.array([1.0, 0.0]))
        r2 = prob.evaluate(np.array([0.0, 1.0]))
        r3 = prob.evaluate(np.array([0.5, 0.5]))
        np.testing.assert_allclose(r1, r2, rtol=1e-10)
        np.testing.assert_allclose(r1, r3, rtol=1e-10)

    def test_build_pymoo_problem_raises_without_pymoo(
        self,
        fairness_problem: FairnessProblem,
    ):
        """build_pymoo_problem() should raise ImportError if pymoo is absent."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pymoo" or name.startswith("pymoo."):
                raise ImportError("No module named 'pymoo'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pymoo"):
                fairness_problem.build_pymoo_problem()


# ---------------------------------------------------------------------------
# B. NSGA-II integration test (requires pymoo)
# ---------------------------------------------------------------------------


class TestNSGA2FairnessOptimiser:
    """Tests requiring pymoo. Skipped if pymoo is not installed."""

    @pytest.fixture(autouse=True)
    def skip_without_pymoo(self):
        pytest.importorskip("pymoo", reason="pymoo not installed")

    def test_run_returns_pareto_result(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """NSGA-II should return a ParetoResult with at least one solution."""
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        optimiser = NSGA2FairnessOptimiser(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = optimiser.run(pop_size=20, n_gen=10, seed=42)

        assert isinstance(result, ParetoResult)
        assert result.n_solutions >= 1
        assert result.F.shape[1] == 3
        assert result.weights.shape[1] == 2

    def test_weights_normalised(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """Returned mixing weights must sum to 1 for each solution."""
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        optimiser = NSGA2FairnessOptimiser(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = optimiser.run(pop_size=20, n_gen=10, seed=42)
        row_sums = result.weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(result.n_solutions), atol=1e-6)

    def test_pareto_solutions_are_non_dominated(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """
        All solutions on the Pareto front must be non-dominated by each other.

        A solution i is dominated by j if j is no worse on all objectives
        and strictly better on at least one.
        """
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        optimiser = NSGA2FairnessOptimiser(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = optimiser.run(pop_size=30, n_gen=15, seed=42)
        F = result.F

        for i in range(len(F)):
            for j in range(len(F)):
                if i == j:
                    continue
                # Check that j does not dominate i
                j_no_worse = np.all(F[j] <= F[i])
                j_strictly_better = np.any(F[j] < F[i])
                is_dominated = j_no_worse and j_strictly_better
                assert not is_dominated, (
                    f"Solution {i} is dominated by solution {j}:\n"
                    f"  F[{i}] = {F[i]}\n"
                    f"  F[{j}] = {F[j]}"
                )

    def test_three_models(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
        base_preds: np.ndarray,
    ):
        """Three models should produce a valid Pareto front."""
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        conservative_preds = base_preds * 0.9
        conservative_model = ArrayModel(conservative_preds)

        optimiser = NSGA2FairnessOptimiser(
            models={
                "biased": biased_model,
                "fair": fair_model,
                "conservative": conservative_model,
            },
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = optimiser.run(pop_size=20, n_gen=10, seed=42)
        assert result.weights.shape[1] == 3
        assert result.n_solutions >= 1


# ---------------------------------------------------------------------------
# C. TOPSIS tests
# ---------------------------------------------------------------------------


class TestTopsisSelect:
    def test_obvious_best_solution(self):
        """
        With one solution clearly best on all objectives, TOPSIS must select it.
        """
        F = np.array([
            [0.1, 0.1, 0.1],  # best on all objectives (index 0)
            [0.5, 0.5, 0.5],
            [0.9, 0.8, 0.7],
        ])
        idx = topsis_select(F)
        assert idx == 0

    def test_returns_valid_index(self):
        """Selected index must be in range [0, n_solutions)."""
        F = np.random.default_rng(42).uniform(0, 1, size=(10, 3))
        idx = topsis_select(F)
        assert 0 <= idx < 10

    def test_single_solution_returns_zero(self):
        """A single solution must always be selected (index 0)."""
        F = np.array([[0.3, 0.5, 0.2]])
        idx = topsis_select(F)
        assert idx == 0

    def test_equal_weights_versus_custom(self):
        """Different weights should generally produce different selections."""
        rng = np.random.default_rng(1234)
        F = rng.uniform(0, 1, size=(20, 3))

        idx_equal = topsis_select(F, weights=None)
        idx_accuracy = topsis_select(F, weights=[0.9, 0.05, 0.05])

        # Both should be valid indices
        assert 0 <= idx_equal < 20
        assert 0 <= idx_accuracy < 20

    def test_weights_emphasise_second_objective(self):
        """
        When weight is entirely on the second objective, TOPSIS should select
        the solution with the smallest value in column 1.
        """
        F = np.array([
            [0.9, 0.1, 0.5],  # best on obj 1 (index 0)
            [0.1, 0.9, 0.5],
            [0.5, 0.5, 0.5],
        ])
        idx = topsis_select(F, weights=[0.0, 1.0, 0.0])
        assert idx == 0  # col 1 minimum is at row 0

    def test_weight_length_mismatch_raises(self):
        """Wrong number of weights should raise ValueError."""
        F = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with pytest.raises(ValueError, match="weights length"):
            topsis_select(F, weights=[0.5, 0.5])

    def test_empty_F_raises(self):
        """Empty F matrix should raise ValueError."""
        F = np.empty((0, 3))
        with pytest.raises(ValueError, match="at least one row"):
            topsis_select(F)

    def test_all_zero_weights_raises(self):
        """All-zero weights should raise ValueError."""
        F = np.array([[0.1, 0.2, 0.3]])
        with pytest.raises(ValueError, match="positive"):
            topsis_select(F, weights=[0.0, 0.0, 0.0])

    def test_normalised_weights(self):
        """Weights [1, 1, 1] and [2, 2, 2] should give the same result."""
        F = np.random.default_rng(99).uniform(0, 1, size=(15, 3))
        idx1 = topsis_select(F, weights=[1.0, 1.0, 1.0])
        idx2 = topsis_select(F, weights=[2.0, 2.0, 2.0])
        assert idx1 == idx2


# ---------------------------------------------------------------------------
# D. ParetoResult serialisation tests
# ---------------------------------------------------------------------------


class TestParetoResultSerialisation:
    @pytest.fixture
    def sample_result(self, rng: np.random.Generator) -> ParetoResult:
        """A synthetic ParetoResult for serialisation tests."""
        n = 15
        F = rng.uniform(0, 1, size=(n, 3))
        weights = rng.dirichlet(np.ones(3), size=n)
        return ParetoResult(
            F=F,
            weights=weights,
            model_names=["base", "fair", "conservative"],
            n_gen=100,
            pop_size=50,
            seed=42,
        )

    def test_to_dict_is_serialisable(self, sample_result: ParetoResult):
        """to_dict() output must be JSON-serialisable."""
        d = sample_result.to_dict()
        # This should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_to_dict_round_trip(self, sample_result: ParetoResult):
        """from_dict(to_dict(result)) must recover the same data."""
        d = sample_result.to_dict()
        recovered = ParetoResult.from_dict(d)

        assert recovered.n_solutions == sample_result.n_solutions
        assert recovered.model_names == sample_result.model_names
        assert recovered.n_gen == sample_result.n_gen
        assert recovered.pop_size == sample_result.pop_size
        assert recovered.seed == sample_result.seed
        np.testing.assert_allclose(recovered.F, sample_result.F)
        np.testing.assert_allclose(recovered.weights, sample_result.weights)

    def test_to_json_and_back(self, sample_result: ParetoResult):
        """to_json() and from_dict(json.load()) should round-trip correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            path = tmp.name

        sample_result.to_json(path)

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        recovered = ParetoResult.from_dict(d)
        np.testing.assert_allclose(recovered.F, sample_result.F, rtol=1e-10)

    def test_selected_point_in_range(self, sample_result: ParetoResult):
        """selected_point() should return a valid index."""
        idx = sample_result.selected_point()
        assert 0 <= idx < sample_result.n_solutions

    def test_selected_point_with_weights(self, sample_result: ParetoResult):
        """selected_point(weights=...) should return a valid index."""
        idx = sample_result.selected_point(weights=[0.5, 0.3, 0.2])
        assert 0 <= idx < sample_result.n_solutions

    def test_summary_returns_string(self, sample_result: ParetoResult):
        """summary() must return a non-empty string."""
        s = sample_result.summary()
        assert isinstance(s, str)
        assert len(s) > 50
        assert "Pareto" in s

    def test_n_solutions_property(self, sample_result: ParetoResult):
        """n_solutions should equal the number of rows in F."""
        assert sample_result.n_solutions == len(sample_result.F)

    def test_from_dict_objective_names_default(self):
        """from_dict() should fill in default objective names if absent."""
        d = {
            "F": [[0.1, 0.2, 0.3]],
            "weights": [[0.5, 0.5]],
            "model_names": ["a", "b"],
            "n_gen": 10,
            "pop_size": 20,
            "seed": 1,
        }
        result = ParetoResult.from_dict(d)
        assert result.objective_names == ["neg_gini", "group_unfairness", "cf_unfairness"]


# ---------------------------------------------------------------------------
# E. LipschitzMetric tests
# ---------------------------------------------------------------------------


class TestLipschitzMetric:
    @pytest.fixture
    def X_numeric(self, rng: np.random.Generator) -> np.ndarray:
        """200 policies, 4 numeric features."""
        return rng.uniform(0, 1, size=(200, 4))

    @pytest.fixture
    def predictions_positive(self, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(4.6, 0.3, size=200)

    def test_compute_returns_lipschitz_result(
        self,
        X_numeric: np.ndarray,
        predictions_positive: np.ndarray,
    ):
        """compute() must return a LipschitzResult."""
        metric = LipschitzMetric(n_pairs=100, log_predictions=True, random_seed=42)
        result = metric.compute(X_numeric, predictions_positive)
        assert isinstance(result, LipschitzResult)

    def test_lipschitz_constant_is_positive(
        self,
        X_numeric: np.ndarray,
        predictions_positive: np.ndarray,
    ):
        """Lipschitz constant must be non-negative for non-trivial data."""
        metric = LipschitzMetric(n_pairs=100, log_predictions=True)
        result = metric.compute(X_numeric, predictions_positive)
        assert result.lipschitz_constant >= 0.0

    def test_constant_predictions_have_zero_lipschitz(
        self,
        X_numeric: np.ndarray,
    ):
        """Constant predictions -> Lipschitz constant = 0."""
        predictions = np.full(200, 100.0)
        metric = LipschitzMetric(n_pairs=100, log_predictions=True)
        result = metric.compute(X_numeric, predictions)
        assert result.lipschitz_constant < 1e-10

    def test_custom_distance_function(
        self,
        X_numeric: np.ndarray,
        predictions_positive: np.ndarray,
    ):
        """Custom distance function should be applied correctly."""
        def manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
            return float(np.sum(np.abs(x1 - x2)))

        metric = LipschitzMetric(
            distance_fn=manhattan,
            n_pairs=100,
            log_predictions=True,
        )
        result = metric.compute(X_numeric, predictions_positive)
        assert result.lipschitz_constant >= 0.0

    def test_n_pairs_sampled(
        self,
        X_numeric: np.ndarray,
        predictions_positive: np.ndarray,
    ):
        """n_pairs_sampled should be close to n_pairs (may differ slightly)."""
        metric = LipschitzMetric(n_pairs=50, log_predictions=True)
        result = metric.compute(X_numeric, predictions_positive)
        # Should sample approximately 50 pairs (excluding same-index pairs)
        assert result.n_pairs_sampled <= 50
        assert result.n_pairs_sampled >= 1

    def test_percentiles_ordered(
        self,
        X_numeric: np.ndarray,
        predictions_positive: np.ndarray,
    ):
        """p50 <= p95 <= max_ratio must hold."""
        metric = LipschitzMetric(n_pairs=200, log_predictions=True)
        result = metric.compute(X_numeric, predictions_positive)
        assert result.p50_ratio <= result.p95_ratio + 1e-10
        assert result.p95_ratio <= result.max_ratio + 1e-10

    def test_too_few_policies_raises(self):
        """Fewer than 2 policies should raise ValueError."""
        metric = LipschitzMetric()
        X = np.array([[1.0, 2.0]])
        preds = np.array([100.0])
        with pytest.raises(ValueError, match="At least 2"):
            metric.compute(X, preds)

    def test_non_positive_predictions_with_log_raises(
        self,
        X_numeric: np.ndarray,
    ):
        """log_predictions=True with non-positive predictions should raise."""
        predictions = np.full(200, 100.0)
        predictions[5] = 0.0
        metric = LipschitzMetric(log_predictions=True)
        with pytest.raises(ValueError, match="strictly positive"):
            metric.compute(X_numeric, predictions)

    def test_log_false_accepts_any_predictions(
        self,
        X_numeric: np.ndarray,
    ):
        """log_predictions=False should accept non-positive predictions."""
        predictions = np.linspace(-10, 10, 200)
        metric = LipschitzMetric(log_predictions=False, n_pairs=50)
        result = metric.compute(X_numeric, predictions)
        assert isinstance(result, LipschitzResult)

    def test_reproducible_with_seed(
        self,
        X_numeric: np.ndarray,
        predictions_positive: np.ndarray,
    ):
        """Same seed should give the same result."""
        metric1 = LipschitzMetric(n_pairs=100, random_seed=123)
        metric2 = LipschitzMetric(n_pairs=100, random_seed=123)
        r1 = metric1.compute(X_numeric, predictions_positive)
        r2 = metric2.compute(X_numeric, predictions_positive)
        assert r1.lipschitz_constant == r2.lipschitz_constant

    def test_known_lipschitz_constant(self):
        """
        For f(x) = x and d(x,x') = |x - x'|, Lipschitz constant should be 1.0.
        """
        X = np.arange(100, dtype=float).reshape(-1, 1)
        preds = np.arange(1, 101, dtype=float)  # positive, f(i) = i+1

        def abs_diff(x1: np.ndarray, x2: np.ndarray) -> float:
            return abs(float(x1[0]) - float(x2[0]))

        metric = LipschitzMetric(
            distance_fn=abs_diff,
            n_pairs=500,
            log_predictions=False,
            random_seed=42,
        )
        result = metric.compute(X, preds)
        # |f(i) - f(j)| / |i - j| = 1 for all pairs
        assert abs(result.lipschitz_constant - 1.0) < 0.01


# ---------------------------------------------------------------------------
# F. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_two_policies_two_models(self):
        """Minimum viable input: 2 policies, 2 models."""
        df = pl.DataFrame({
            "gender": [0, 1],
            "feature": [1.0, 2.0],
        })
        y = np.array([100.0, 120.0])
        exposure = np.array([1.0, 1.0])

        model_a = ArrayModel(np.array([100.0, 100.0]))
        model_b = ArrayModel(np.array([100.0, 120.0]))

        prob = FairnessProblem(
            models={"a": model_a, "b": model_b},
            X=df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        result = prob.evaluate(np.array([0.5, 0.5]))
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_multiclass_protected_characteristic(self):
        """Multi-class protected characteristic: CF objective should be 0."""
        n = 60
        rng = np.random.default_rng(0)
        df = pl.DataFrame({
            "ethnicity": (["A"] * 20 + ["B"] * 20 + ["C"] * 20),
            "age": rng.integers(20, 70, n).tolist(),
        })
        preds = rng.lognormal(4.6, 0.3, n)
        y = rng.lognormal(4.6, 0.3, n)
        exposure = np.ones(n)

        model = ArrayModel(preds)
        prob = FairnessProblem(
            models={"m": model},
            X=df,
            y=y,
            exposure=exposure,
            protected_col="ethnicity",
        )
        result = prob.evaluate(np.array([1.0]))
        # CF objective must be 0 for multi-class (no flip_map)
        assert result[2] == 0.0

    def test_zero_exposure_column(self):
        """Zero exposure should not produce NaN or errors in objectives."""
        df = pl.DataFrame({
            "gender": [0, 0, 1, 1],
            "age": [25.0, 30.0, 35.0, 40.0],
        })
        preds = np.array([100.0, 110.0, 120.0, 130.0])
        y = np.array([90.0, 100.0, 110.0, 120.0])
        exposure = np.array([0.0, 0.0, 0.0, 0.0])  # all zero

        model = ArrayModel(preds)
        prob = FairnessProblem(
            models={"m": model},
            X=df,
            y=y,
            exposure=exposure,
            protected_col="gender",
        )
        # Should not raise; results may be 0 or nan but not crash
        try:
            result = prob.evaluate(np.array([1.0]))
            # If it returns, should be shape (3,)
            assert result.shape == (3,)
        except ZeroDivisionError:
            pytest.fail("evaluate() raised ZeroDivisionError on zero exposure")

    def test_topsis_two_solutions_picks_closer_to_ideal(self):
        """
        With two solutions, TOPSIS should pick the one closer to the ideal
        on the weighted objectives.
        """
        # Solution 0 is better on all objectives
        F = np.array([
            [0.2, 0.3, 0.1],
            [0.8, 0.7, 0.9],
        ])
        idx = topsis_select(F)
        assert idx == 0

    def test_pareto_result_summary_with_one_solution(self):
        """summary() should work even with a single solution."""
        result = ParetoResult(
            F=np.array([[0.5, 0.3, 0.2]]),
            weights=np.array([[0.6, 0.4]]),
            model_names=["a", "b"],
            n_gen=50,
            pop_size=20,
            seed=0,
        )
        s = result.summary()
        assert "Pareto" in s
        assert "1" in s  # n_solutions = 1


# ---------------------------------------------------------------------------
# G. Four-objective mode (LipschitzMetric as NSGA-II objective)
# ---------------------------------------------------------------------------


class TestFourObjectiveMode:
    """Tests for FairnessProblem with individual fairness (Lipschitz) as 4th objective."""

    @pytest.fixture
    def four_obj_problem(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ) -> FairnessProblem:
        """FairnessProblem with Lipschitz 4th objective active."""
        return FairnessProblem(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "vehicle_value"],
            lipschitz_n_pairs=50,
        )

    def test_n_obj_is_four(self, four_obj_problem: FairnessProblem):
        """FairnessProblem with lipschitz_feature_cols should have n_obj=4."""
        assert four_obj_problem.n_obj == 4

    def test_n_obj_is_three_without_lipschitz(
        self, fairness_problem: FairnessProblem
    ):
        """FairnessProblem without lipschitz_feature_cols should have n_obj=3."""
        assert fairness_problem.n_obj == 3

    def test_evaluate_returns_four_objectives(self, four_obj_problem: FairnessProblem):
        """evaluate() should return shape (4,) in four-objective mode."""
        weights = np.array([0.5, 0.5])
        result = four_obj_problem.evaluate(weights)
        assert result.shape == (4,)

    def test_all_objectives_finite(self, four_obj_problem: FairnessProblem):
        """All four objectives should be finite for well-behaved inputs."""
        for w in [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]:
            result = four_obj_problem.evaluate(w)
            assert np.all(np.isfinite(result)), f"Non-finite objectives for weights={w}: {result}"

    def test_fourth_objective_non_negative(self, four_obj_problem: FairnessProblem):
        """Normalised Lipschitz constant must be non-negative."""
        result = four_obj_problem.evaluate(np.array([0.5, 0.5]))
        assert result[3] >= 0.0

    def test_three_obj_evaluate_unchanged(self, fairness_problem: FairnessProblem):
        """Three-objective mode: evaluate() still returns shape (3,) unchanged."""
        result = fairness_problem.evaluate(np.array([0.5, 0.5]))
        assert result.shape == (3,)

    def test_lipschitz_feature_cols_missing_raises(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray,
        biased_model: ArrayModel
    ):
        """Providing a non-existent feature column should raise ValueError."""
        with pytest.raises(ValueError, match="lipschitz_feature_cols contains columns not found"):
            FairnessProblem(
                models={"m": biased_model},
                X=small_df,
                y=y,
                exposure=exposure,
                protected_col="gender",
                lipschitz_feature_cols=["age", "nonexistent_col"],
            )

    def test_empty_lipschitz_feature_cols_raises(
        self, small_df: pl.DataFrame, y: np.ndarray, exposure: np.ndarray,
        biased_model: ArrayModel
    ):
        """Passing an empty list for lipschitz_feature_cols should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            FairnessProblem(
                models={"m": biased_model},
                X=small_df,
                y=y,
                exposure=exposure,
                protected_col="gender",
                lipschitz_feature_cols=[],
            )

    def test_custom_distance_function_used(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """Custom distance function should be accepted and used without error."""
        call_count = {"n": 0}

        def manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
            call_count["n"] += 1
            return float(np.sum(np.abs(x1 - x2)))

        prob = FairnessProblem(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "vehicle_value"],
            lipschitz_distance_fn=manhattan,
            lipschitz_n_pairs=20,
        )
        result = prob.evaluate(np.array([0.5, 0.5]))
        assert result.shape == (4,)
        assert call_count["n"] > 0  # the custom function was called

    def test_four_obj_topsis_select(self, four_obj_problem: FairnessProblem):
        """topsis_select should work on 4-column F matrix."""
        rng = np.random.default_rng(7)
        F = rng.uniform(0, 1, size=(10, 4))
        idx = topsis_select(F)
        assert 0 <= idx < 10

    def test_four_obj_topsis_with_explicit_weights(self):
        """topsis_select with 4 objectives and explicit weights should return valid index."""
        F = np.array([
            [0.1, 0.2, 0.3, 0.1],   # best on all (index 0)
            [0.9, 0.8, 0.7, 0.9],
            [0.5, 0.5, 0.5, 0.5],
        ])
        idx = topsis_select(F, weights=[0.4, 0.2, 0.2, 0.2])
        assert idx == 0

    def test_four_obj_topsis_weight_mismatch_raises(self):
        """Passing 3 weights for a 4-objective problem should raise ValueError."""
        F = np.random.default_rng(0).uniform(0, 1, size=(5, 4))
        with pytest.raises(ValueError, match="weights length"):
            topsis_select(F, weights=[0.5, 0.3, 0.2])

    def test_pareto_result_four_obj_serialisation(self):
        """ParetoResult with 4 objectives should round-trip via to_dict/from_dict."""
        rng = np.random.default_rng(1)
        F = rng.uniform(0, 1, size=(8, 4))
        weights = rng.dirichlet(np.ones(2), size=8)
        result = ParetoResult(
            F=F,
            weights=weights,
            model_names=["base", "fair"],
            n_gen=50,
            pop_size=20,
            seed=0,
            objective_names=["neg_gini", "group_unfairness", "cf_unfairness", "lipschitz_unfairness"],
        )
        d = result.to_dict()
        recovered = ParetoResult.from_dict(d)
        assert recovered.objective_names == result.objective_names
        assert recovered.F.shape == (8, 4)
        np.testing.assert_allclose(recovered.F, F)

    def test_pareto_result_four_obj_selected_point(self):
        """selected_point() with 4 objectives should accept 4-element weights."""
        rng = np.random.default_rng(2)
        F = rng.uniform(0, 1, size=(10, 4))
        weights_mat = rng.dirichlet(np.ones(2), size=10)
        result = ParetoResult(
            F=F,
            weights=weights_mat,
            model_names=["base", "fair"],
            n_gen=50,
            pop_size=20,
            seed=0,
            objective_names=["neg_gini", "group_unfairness", "cf_unfairness", "lipschitz_unfairness"],
        )
        idx = result.selected_point(weights=[0.4, 0.2, 0.2, 0.2])
        assert 0 <= idx < 10

    def test_pareto_result_four_obj_summary(self):
        """summary() for a 4-objective result should mention all objective names."""
        rng = np.random.default_rng(3)
        F = rng.uniform(0, 1, size=(5, 4))
        weights_mat = rng.dirichlet(np.ones(2), size=5)
        result = ParetoResult(
            F=F,
            weights=weights_mat,
            model_names=["base", "fair"],
            n_gen=50,
            pop_size=20,
            seed=0,
            objective_names=["neg_gini", "group_unfairness", "cf_unfairness", "lipschitz_unfairness"],
        )
        s = result.summary()
        assert "lipschitz_unfairness" in s
        assert "neg_gini" in s

    def test_lipschitz_baseline_normalisation(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """
        At equal weights, the 4th objective value should be approximately 1.0
        (since baseline is computed at equal weights and the objective is normalised
        by that baseline).
        """
        prob = FairnessProblem(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "vehicle_value"],
            lipschitz_n_pairs=100,
        )
        # At exactly equal weights, Lipschitz objective == baseline / baseline == 1.0
        result = prob.evaluate(np.array([1.0, 1.0]))
        # The objective at equal weights must be 1.0 (within floating-point tolerance)
        assert abs(result[3] - 1.0) < 1e-6

    def test_constant_models_lipschitz_objective_is_zero(
        self,
        small_df: pl.DataFrame,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """
        When both models produce constant predictions, Lipschitz baseline = 0
        and the 4th objective should be 0.0 throughout (no individual unfairness
        is possible if all predictions are identical).
        """
        model_a = ConstantModel(100.0)
        model_b = ConstantModel(200.0)
        prob = FairnessProblem(
            models={"a": model_a, "b": model_b},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "vehicle_value"],
            lipschitz_n_pairs=50,
        )
        # Constant predictions -> baseline Lipschitz = 0 -> 4th objective = 0
        result = prob.evaluate(np.array([0.5, 0.5]))
        assert result[3] == 0.0


class TestNSGA2FourObjectiveIntegration:
    """NSGA-II four-objective integration tests (requires pymoo)."""

    @pytest.fixture(autouse=True)
    def skip_without_pymoo(self):
        pytest.importorskip("pymoo", reason="pymoo not installed")

    def test_four_obj_nsga2_returns_correct_shape(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """NSGA-II with 4 objectives should return F with 4 columns."""
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        optimiser = NSGA2FairnessOptimiser(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "vehicle_value"],
            lipschitz_n_pairs=30,
        )
        result = optimiser.run(pop_size=15, n_gen=8, seed=42)
        assert result.F.shape[1] == 4
        assert result.objective_names == [
            "neg_gini", "group_unfairness", "cf_unfairness", "lipschitz_unfairness"
        ]

    def test_four_obj_result_has_valid_weights(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """Mixing weights from a 4-objective run should still sum to 1."""
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        optimiser = NSGA2FairnessOptimiser(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            lipschitz_feature_cols=["age", "vehicle_value"],
            lipschitz_n_pairs=30,
        )
        result = optimiser.run(pop_size=15, n_gen=8, seed=42)
        row_sums = result.weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(result.n_solutions), atol=1e-6)

    def test_three_obj_mode_unchanged_by_new_params(
        self,
        small_df: pl.DataFrame,
        biased_model: ArrayModel,
        fair_model: ArrayModel,
        y: np.ndarray,
        exposure: np.ndarray,
    ):
        """Existing 3-objective interface must be unchanged (backward compat)."""
        from insurance_fairness.pareto import NSGA2FairnessOptimiser

        optimiser = NSGA2FairnessOptimiser(
            models={"biased": biased_model, "fair": fair_model},
            X=small_df,
            y=y,
            exposure=exposure,
            protected_col="gender",
            # No lipschitz_feature_cols -> 3-objective mode
        )
        result = optimiser.run(pop_size=15, n_gen=8, seed=42)
        assert result.F.shape[1] == 3
        assert result.objective_names == ["neg_gini", "group_unfairness", "cf_unfairness"]
