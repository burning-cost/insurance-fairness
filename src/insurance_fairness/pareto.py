"""
pareto.py
---------
NSGA-II multi-objective Pareto optimisation for fairness-accuracy trade-offs
in insurance pricing models.

This module implements the framework from:

    Bellamy et al. (2024). Multi-Objective Fairness Optimisation for Insurance
    Pricing Models. arXiv:2512.24747.

The core idea is that fairness and accuracy are genuinely competing objectives
for a pricing model: a model trained to maximise predictive lift will often
produce more group-level disparity than one that sacrifices some accuracy for
fairness. NSGA-II finds the full Pareto front of non-dominated solutions, so
the pricing team can make an explicit, documented choice about where on that
front to sit — rather than discovering the trade-off after deployment.

Three-objective mode (default)
------------------------------
Objectives minimised simultaneously:

1. Negative Gini coefficient (accuracy proxy)
2. Group unfairness (1 − demographic parity ratio)
3. Counterfactual unfairness (1 − counterfactual fairness score)

Four-objective mode (individual fairness)
-----------------------------------------
Pass ``lipschitz_feature_cols`` (plus optionally ``lipschitz_distance_fn``,
``lipschitz_n_pairs``, ``lipschitz_log_predictions``) to activate a fourth
objective:

4. Normalised Lipschitz constant — the sampled Lipschitz constant of the
   ensemble, normalised by the baseline Lipschitz constant (computed once at
   construction for equal weights). This objective is 1.0 at the baseline and
   decreases as the ensemble becomes smoother. Values above 1.0 mean the
   ensemble is *less* individually fair than the equal-weight baseline.

The Lipschitz objective is deliberately opt-in because it requires a
user-defined distance function that is meaningful for your rating factors.
The default Euclidean distance is almost never appropriate for heterogeneous
insurance data. Pass a custom ``lipschitz_distance_fn`` — for example, one
that uses Gower distance or a weighted combination of actuarial feature
distances.

Workflow::

    from insurance_fairness.pareto import NSGA2FairnessOptimiser

    # Three-objective mode (default)
    optimiser = NSGA2FairnessOptimiser(
        models={'base': model_a, 'fair': model_b, 'conservative': model_c},
        X=X_test,
        y=y_test,
        exposure=exposure,
        protected_col='gender',
    )
    result = optimiser.run(pop_size=100, n_gen=200, seed=42)
    selected_idx = result.selected_point(weights=[0.4, 0.3, 0.3])
    result.plot_front(highlight=selected_idx)

    # Four-objective mode (individual fairness via Lipschitz)
    import numpy as np
    def gower_dist(x1, x2):
        return float(np.mean(np.abs(x1 - x2)))

    optimiser_4obj = NSGA2FairnessOptimiser(
        models={'base': model_a, 'fair': model_b},
        X=X_test,
        y=y_test,
        exposure=exposure,
        protected_col='gender',
        lipschitz_feature_cols=['age', 'vehicle_value', 'ncd'],
        lipschitz_distance_fn=gower_dist,
    )
    result = optimiser_4obj.run(pop_size=100, n_gen=200, seed=42)
    selected_idx = result.selected_point(weights=[0.4, 0.2, 0.2, 0.2])

Optional dependency
-------------------
This module requires ``pymoo>=0.6.1``:

    pip install insurance-fairness[pareto]

or:

    pip install pymoo>=0.6.1

Regulatory context
------------------
FCA Consumer Duty requires pricing teams to demonstrate they have considered
fairness outcomes. Documenting the Pareto front and the chosen operating point
provides an auditable record of that consideration. The selected_point() method
with explicit weights forces the team to state their preference ordering rather
than simply accepting the default.

References
----------
Bellamy et al. (2024). arXiv:2512.24747.
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.
"""

from __future__ import annotations

import json
import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

# np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# ---------------------------------------------------------------------------
# Lazy import guard for pymoo
# ---------------------------------------------------------------------------


def _require_pymoo() -> None:
    """Raise ImportError with an actionable message if pymoo is not installed."""
    try:
        import pymoo  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pymoo is required for Pareto optimisation. Install it with:\n"
            "    pip install insurance-fairness[pareto]\n"
            "or:\n"
            "    pip install 'pymoo>=0.6.1'"
        ) from exc


# ---------------------------------------------------------------------------
# A. FairnessProblem
# ---------------------------------------------------------------------------


class FairnessProblem:
    """
    Multi-objective optimisation problem over an ensemble of pricing models.

    Wraps ``pymoo.core.problem.ElementwiseProblem`` and defines three or four
    objectives to minimise simultaneously:

    1. Negative accuracy — specifically, negative Gini coefficient of the
       ensemble's predictions. Minimising this maximises discrimination.
       (Negative because NSGA-II minimises all objectives.)

    2. Group unfairness — ``1 - demographic_parity_ratio``, clipped to [0, 1].
       Zero means perfect demographic parity; one means maximum disparity.

    3. Counterfactual unfairness — ``1 - counterfactual_fairness_score``, where
       counterfactual_fairness_score is the proportion of policies for which
       flipping the protected characteristic changes the prediction by less
       than a tolerance threshold.

    4. (Optional) Individual unfairness — the sampled Lipschitz constant of the
       ensemble, normalised by the baseline value at equal mixing weights.
       Activated by passing ``lipschitz_feature_cols``. A value of 1.0 matches
       the equal-weight baseline; lower is better. See the module docstring for
       guidance on choosing a distance function.

    Decision variables are the mixing weights w_k over K pre-trained models.
    The weights are constrained to sum to 1.0 via normalisation inside
    ``_evaluate``. Raw decision variables are in [0, 1].

    Parameters
    ----------
    models:
        Dict mapping model name to fitted model. Each model must implement
        ``predict(X)`` where X is a pandas DataFrame (for CatBoost compatibility).
    X:
        Policy-level Polars DataFrame of features.
    y:
        Array-like of actual outcomes (claims, frequencies, etc.). Used for
        accuracy objective.
    exposure:
        Array-like of exposure values. Used for exposure-weighted metrics.
    protected_col:
        Name of the protected characteristic column in X.
    prediction_col:
        Name of an existing prediction column in X, if predictions are
        pre-computed. If None, predictions are generated from each model.
    cf_tolerance:
        Threshold for counting a prediction change as material when computing
        counterfactual unfairness. A policy is "counterfactually fair" if the
        absolute log-ratio of flipped to original prediction is below this
        value. Default 0.05 (approximately 5% premium change).
    cat_features:
        List of categorical feature column names. Required for CatBoost Pool
        construction if predictions are not pre-computed.
    lipschitz_feature_cols:
        List of numeric column names in X to use as the feature matrix for
        Lipschitz estimation. Passing this list activates the fourth objective.
        Columns must be numeric and must not contain nulls. Do not include the
        protected characteristic column — individual fairness is assessed
        conditional on non-protected features.
    lipschitz_distance_fn:
        Callable ``(x1, x2) -> float`` defining the distance between two
        policy feature vectors. If None, Euclidean distance is used (rarely
        appropriate for heterogeneous insurance features — pass a custom
        function such as Gower distance). Only used when
        ``lipschitz_feature_cols`` is provided.
    lipschitz_n_pairs:
        Number of random policy pairs to sample when estimating the Lipschitz
        constant. Larger values give a more stable estimate. Default 500.
    lipschitz_log_predictions:
        If True, compute ``|log(f(x)) - log(f(x'))|`` in the numerator.
        Appropriate for multiplicative pricing models with a log link.
        Default True.

    Notes
    -----
    The Lipschitz objective is normalised by the baseline value (equal
    mixing weights across all models). This prevents the objective value from
    varying with the absolute scale of predictions and makes the Pareto front
    easier to interpret: a value of 1.0 means the ensemble is as individually
    fair as the equal-weight ensemble; lower is better.

    If all models produce identical predictions, the baseline Lipschitz
    constant is zero and normalisation is ill-defined. In that case the
    Lipschitz objective is set to 0.0 throughout (the ensemble cannot be
    made more individually fair by changing weights).
    """

    def __init__(
        self,
        models: Dict[str, Any],
        X: pl.DataFrame,
        y: np.ndarray,
        exposure: np.ndarray,
        protected_col: str,
        prediction_col: Optional[str] = None,
        cf_tolerance: float = 0.05,
        cat_features: Optional[List[str]] = None,
        lipschitz_feature_cols: Optional[List[str]] = None,
        lipschitz_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        lipschitz_n_pairs: int = 500,
        lipschitz_log_predictions: bool = True,
    ) -> None:
        if protected_col not in X.columns:
            raise ValueError(
                f"protected_col '{protected_col}' not found in X. "
                f"Available columns: {X.columns}"
            )
        if len(models) < 1:
            raise ValueError("At least one model must be provided.")

        self.models = models
        self.model_names = list(models.keys())
        self.X = X
        self.y = np.asarray(y, dtype=float)
        self.exposure = np.asarray(exposure, dtype=float)
        self.protected_col = protected_col
        self.prediction_col = prediction_col
        self.cf_tolerance = cf_tolerance
        self.cat_features = cat_features or []

        # Pre-compute predictions from each model once (expensive)
        self._model_preds: Dict[str, np.ndarray] = {}
        self._model_cf_preds: Dict[str, np.ndarray] = {}
        self._precompute_predictions()

        # Protected characteristic groups (for demographic parity)
        self._groups = X[protected_col].unique().sort().to_list()
        self._group_masks: Dict[str, np.ndarray] = {
            str(g): (X[protected_col] == g).to_numpy()
            for g in self._groups
        }
        self._group_exposures: Dict[str, float] = {
            str(g): float(self.exposure[mask].sum())
            for g, mask in self._group_masks.items()
        }

        self._K = len(self.model_names)

        # -- Lipschitz (individual fairness) objective setup --
        self._lipschitz: Optional[LipschitzMetric] = None
        self._X_numeric: Optional[np.ndarray] = None
        self._lipschitz_baseline: float = 0.0

        if lipschitz_feature_cols is not None:
            self._setup_lipschitz(
                lipschitz_feature_cols=lipschitz_feature_cols,
                lipschitz_distance_fn=lipschitz_distance_fn,
                lipschitz_n_pairs=lipschitz_n_pairs,
                lipschitz_log_predictions=lipschitz_log_predictions,
            )

        self.n_obj: int = 4 if self._lipschitz is not None else 3

    def _setup_lipschitz(
        self,
        lipschitz_feature_cols: List[str],
        lipschitz_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]],
        lipschitz_n_pairs: int,
        lipschitz_log_predictions: bool,
    ) -> None:
        """
        Validate Lipschitz parameters, build the numeric feature matrix, and
        compute the baseline Lipschitz constant at equal mixing weights.
        """
        missing = [c for c in lipschitz_feature_cols if c not in self.X.columns]
        if missing:
            raise ValueError(
                f"lipschitz_feature_cols contains columns not found in X: {missing}. "
                f"Available columns: {self.X.columns}"
            )
        if not lipschitz_feature_cols:
            raise ValueError("lipschitz_feature_cols must not be empty.")

        try:
            X_numeric = self.X.select(lipschitz_feature_cols).to_numpy().astype(float)
        except Exception as exc:
            raise ValueError(
                f"Could not convert lipschitz_feature_cols to a numeric matrix. "
                f"Ensure all listed columns are numeric and contain no nulls. "
                f"Original error: {exc}"
            ) from exc

        if np.any(~np.isfinite(X_numeric)):
            raise ValueError(
                "lipschitz_feature_cols contains NaN or Inf values. "
                "Please impute or drop affected rows before optimisation."
            )

        self._X_numeric = X_numeric
        self._lipschitz = LipschitzMetric(
            distance_fn=lipschitz_distance_fn,
            n_pairs=lipschitz_n_pairs,
            log_predictions=lipschitz_log_predictions,
            random_seed=42,
        )

        # Compute the baseline Lipschitz constant at equal mixing weights.
        # We use this to normalise the objective throughout optimisation.
        equal_weights = np.ones(self._K) / self._K
        baseline_preds = self._ensemble_preds(equal_weights)

        if lipschitz_log_predictions and np.any(baseline_preds <= 0):
            warnings.warn(
                "Baseline ensemble predictions contain non-positive values. "
                "Lipschitz objective with log_predictions=True may be unreliable. "
                "Consider setting lipschitz_log_predictions=False.",
                UserWarning,
                stacklevel=3,
            )
            # Fall back gracefully: set baseline to 0 so objective is always 0
            self._lipschitz_baseline = 0.0
        else:
            result = self._lipschitz.compute(X_numeric, baseline_preds)
            self._lipschitz_baseline = result.lipschitz_constant

    def _precompute_predictions(self) -> None:
        """
        Pre-compute base and counterfactual predictions for each model.

        This runs once at construction time so that _evaluate() is fast
        enough for NSGA-II's many function evaluations.
        """
        X_pd = self.X.to_pandas()

        # Identify protected characteristic values for counterfactual flip
        unique_vals = self.X[self.protected_col].unique().to_list()
        if len(unique_vals) == 2:
            flip_map = {unique_vals[0]: unique_vals[1], unique_vals[1]: unique_vals[0]}
        else:
            # For multi-class, we cannot auto-flip. Disable counterfactual objective.
            flip_map = None

        self._flip_map = flip_map

        for name, model in self.models.items():
            if self.prediction_col is not None and self.prediction_col in self.X.columns:
                # Use pre-computed predictions if a column is provided
                # (for single model; for ensemble we always predict)
                preds = self.X[self.prediction_col].to_numpy().astype(float)
            else:
                preds = self._predict_model(model, X_pd)
            self._model_preds[name] = preds

            # Counterfactual predictions (flip protected characteristic)
            if flip_map is not None:
                X_cf = self.X.with_columns(
                    self.X[self.protected_col].map_elements(
                        lambda v: flip_map.get(v, v),
                        return_dtype=self.X[self.protected_col].dtype,
                    ).alias(self.protected_col)
                )
                X_cf_pd = X_cf.to_pandas()
                cf_preds = self._predict_model(model, X_cf_pd)
            else:
                cf_preds = preds.copy()

            self._model_cf_preds[name] = cf_preds

    def _predict_model(self, model: Any, X_pd: Any) -> np.ndarray:
        """Generate predictions from a single model."""
        try:
            # CatBoost: use Pool for proper categorical handling
            from catboost import Pool  # noqa: PLC0415

            pool = Pool(X_pd, cat_features=self.cat_features if self.cat_features else None)
            return np.asarray(model.predict(pool), dtype=float)
        except (ImportError, Exception):
            # Fallback: sklearn-compatible interface
            try:
                return np.asarray(model.predict(X_pd), dtype=float)
            except Exception:
                return np.asarray(model.predict(X_pd.values), dtype=float)

    def _ensemble_preds(self, weights: np.ndarray) -> np.ndarray:
        """Return exposure-weighted ensemble predictions for normalised weights."""
        # Normalise weights to sum to 1
        w = weights / weights.sum()
        preds = np.zeros(len(self.X), dtype=float)
        for i, name in enumerate(self.model_names):
            preds += w[i] * self._model_preds[name]
        return preds

    def _ensemble_cf_preds(self, weights: np.ndarray) -> np.ndarray:
        """Return counterfactual ensemble predictions for normalised weights."""
        w = weights / weights.sum()
        preds = np.zeros(len(self.X), dtype=float)
        for i, name in enumerate(self.model_names):
            preds += w[i] * self._model_cf_preds[name]
        return preds

    def _gini_objective(self, preds: np.ndarray) -> float:
        """
        Return negative weighted Gini coefficient of predictions vs actuals.

        The Gini coefficient measures how well predictions rank the actual
        outcomes. A higher Gini means better discrimination. We return the
        negative because NSGA-II minimises all objectives.

        Uses the standard trapezoid approximation of the Lorenz curve.
        """
        if np.all(preds == preds[0]):
            return 0.0  # All predictions identical -> Gini = 0, negative = 0

        sorted_idx = np.argsort(preds)
        sorted_actual = self.y[sorted_idx]
        sorted_exposure = self.exposure[sorted_idx]

        cum_exposure = np.cumsum(sorted_exposure)
        cum_actual = np.cumsum(sorted_actual * sorted_exposure)

        total_exposure = cum_exposure[-1]
        total_actual = cum_actual[-1]

        if total_actual == 0 or total_exposure == 0:
            return 0.0

        lorenz_x = np.concatenate([[0.0], cum_exposure / total_exposure])
        lorenz_y = np.concatenate([[0.0], cum_actual / total_actual])
        area_under = float(_trapz(lorenz_y, lorenz_x))
        gini = 1.0 - 2.0 * area_under

        # Return negative (minimisation -> maximise Gini)
        return -gini

    def _demographic_parity_objective(self, preds: np.ndarray) -> float:
        """
        Return group unfairness as 1 - demographic_parity_ratio, clipped to [0, 1].

        Demographic parity ratio is the ratio of exposure-weighted mean predictions
        between groups. A ratio of 1.0 (perfect parity) gives objective value 0.0.

        For multi-group characteristics, uses the maximum ratio deviation from 1.0.
        """
        if len(self._groups) < 2:
            return 0.0

        group_means: Dict[str, float] = {}
        for g_str, mask in self._group_masks.items():
            g_preds = preds[mask]
            g_exp = self.exposure[mask]
            total_g_exp = g_exp.sum()
            if total_g_exp > 0:
                group_means[g_str] = float(np.average(g_preds, weights=g_exp))
            else:
                group_means[g_str] = float("nan")

        valid_means = [v for v in group_means.values() if not np.isnan(v)]
        if len(valid_means) < 2:
            return 0.0

        if len(self._groups) == 2:
            keys = list(group_means.keys())
            m0, m1 = group_means[keys[0]], group_means[keys[1]]
            if m0 <= 0 or m1 <= 0:
                return 1.0
            ratio = min(m0, m1) / max(m0, m1)
        else:
            # Multi-group: ratio of min to max group mean
            ratio = min(valid_means) / max(valid_means) if max(valid_means) > 0 else 0.0

        # Clip ratio to [0, 1] and return 1 - ratio (so 0 = perfect parity)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return 1.0 - ratio

    def _counterfactual_objective(self, preds: np.ndarray, cf_preds: np.ndarray) -> float:
        """
        Return counterfactual unfairness as 1 - counterfactual_fairness_score.

        Counterfactual fairness score is the exposure-weighted proportion of
        policies where |log(cf_pred / pred)| < cf_tolerance. A policy is
        considered counterfactually fair if the premium change from flipping
        the protected characteristic is less than cf_tolerance in log-space.

        If flip_map is None (multi-class protected characteristic), returns 0.0.
        """
        if self._flip_map is None:
            return 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratios = np.where(
                (preds > 0) & (cf_preds > 0),
                np.abs(np.log(cf_preds / preds)),
                float("inf"),
            )

        is_fair = log_ratios < self.cf_tolerance
        total_exp = self.exposure.sum()
        if total_exp <= 0:
            return 0.0

        fair_exposure = float(np.dot(is_fair.astype(float), self.exposure))
        cf_score = fair_exposure / total_exp

        return 1.0 - cf_score

    def _lipschitz_objective(self, preds: np.ndarray) -> float:
        """
        Return the normalised Lipschitz constant for individual fairness.

        The raw sampled Lipschitz constant is divided by the baseline value
        (computed once at equal mixing weights). This gives an objective that
        is 1.0 at the baseline and lower for smoother ensembles.

        If the baseline Lipschitz constant is zero (e.g. all models produce
        identical predictions), returns 0.0 throughout.

        Parameters
        ----------
        preds:
            Ensemble predictions of shape (n_policies,).

        Returns
        -------
        float
            Normalised Lipschitz constant. Lower is better.
        """
        assert self._lipschitz is not None
        assert self._X_numeric is not None

        if self._lipschitz_baseline <= 0.0:
            return 0.0

        if self._lipschitz.log_predictions and np.any(preds <= 0):
            # Cannot compute log-space Lipschitz for non-positive predictions;
            # treat as maximally unfair (1.0 relative to baseline is conservative).
            return 1.0

        result = self._lipschitz.compute(self._X_numeric, preds)
        return result.lipschitz_constant / self._lipschitz_baseline

    def evaluate(self, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives for a given weight vector.

        This is the core function called by NSGA-II during optimisation.
        It is separated from the pymoo ``_evaluate`` method so it can be
        called independently in tests.

        Parameters
        ----------
        weights:
            Array of shape (K,) with raw mixing weights. Will be normalised
            internally to sum to 1.

        Returns
        -------
        Array of shape (3,) or (4,) depending on whether the Lipschitz
        objective is active. Objectives are in the order:
        [neg_gini, group_unfairness, cf_unfairness] or
        [neg_gini, group_unfairness, cf_unfairness, lipschitz_unfairness].
        """
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)

        preds = self._ensemble_preds(weights)
        cf_preds = self._ensemble_cf_preds(weights)

        obj1 = self._gini_objective(preds)
        obj2 = self._demographic_parity_objective(preds)
        obj3 = self._counterfactual_objective(preds, cf_preds)

        if self._lipschitz is not None:
            obj4 = self._lipschitz_objective(preds)
            return np.array([obj1, obj2, obj3, obj4], dtype=float)

        return np.array([obj1, obj2, obj3], dtype=float)

    def build_pymoo_problem(self) -> Any:
        """
        Construct and return a pymoo ElementwiseProblem for use with NSGA-II.

        The number of objectives is determined by ``self.n_obj`` (3 or 4,
        depending on whether ``lipschitz_feature_cols`` was provided).

        Returns
        -------
        pymoo.core.problem.ElementwiseProblem subclass instance.
        """
        _require_pymoo()
        from pymoo.core.problem import ElementwiseProblem  # noqa: PLC0415

        parent = self

        class _PymooWrapper(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=parent._K,
                    n_obj=parent.n_obj,
                    n_ieq_constr=0,
                    xl=np.zeros(parent._K),
                    xu=np.ones(parent._K),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = parent.evaluate(x)

        return _PymooWrapper()


# ---------------------------------------------------------------------------
# B. LipschitzMetric
# ---------------------------------------------------------------------------


class LipschitzMetric:
    """
    Sampled Lipschitz constant approximation for insurance pricing models.

    EXPERIMENTAL — requires a user-defined distance metric that is meaningful
    for your rating factors. The default Euclidean distance in raw feature
    space is rarely appropriate for heterogeneous insurance data (mixing
    continuous numeric factors and categorical variables). See the
    ``distance_fn`` parameter.

    The Lipschitz constant of a function f is:

        L = sup_{x != x'} |f(x) - f(x')| / d(x, x')

    For a pricing model, L bounds the maximum premium change per unit of
    feature distance. A lower L indicates a smoother, more stable model.
    Models with very high L values near protected-characteristic boundaries
    in feature space may be at higher risk of proxy discrimination.

    This class estimates L by sampling K random pairs of policies and
    computing the maximum ratio of prediction difference to feature distance.

    Parameters
    ----------
    distance_fn:
        Callable(x1, x2) -> float, where x1 and x2 are numpy arrays of
        the same shape. The distance must be non-negative and return 0.0
        only when x1 == x2. Log-space distances are often more appropriate
        for multiplicative features (e.g. vehicle value, sum insured).
        Pass None to use Euclidean distance in raw feature space, which
        is only appropriate if all features are on the same scale.
    n_pairs:
        Number of random policy pairs to sample. More pairs give a more
        stable estimate at higher computational cost.
    log_predictions:
        If True, compute |log(f(x)) - log(f(x'))| in the numerator instead
        of |f(x) - f(x')|. Appropriate for multiplicative pricing models
        with a log link.
    random_seed:
        Random seed for reproducibility.

    Notes
    -----
    This is a sample-based lower bound on the true Lipschitz constant, not
    an exact computation. It will underestimate L for models with
    concentrated sensitivity in regions of feature space not covered by
    the sampled pairs. For a tighter bound, increase ``n_pairs``.
    """

    def __init__(
        self,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        n_pairs: int = 1000,
        log_predictions: bool = True,
        random_seed: int = 42,
    ) -> None:
        self.distance_fn = distance_fn or self._euclidean
        self.n_pairs = n_pairs
        self.log_predictions = log_predictions
        self.random_seed = random_seed

    @staticmethod
    def _euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
        """Euclidean distance between two feature vectors."""
        return float(np.linalg.norm(x1 - x2))

    def compute(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
    ) -> "LipschitzResult":
        """
        Estimate the Lipschitz constant from a sample of policy pairs.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_policies, n_features). Must be numeric.
            Categorical features should be encoded before passing.
        predictions:
            Model predictions of shape (n_policies,). Must be positive if
            ``log_predictions=True``.

        Returns
        -------
        LipschitzResult
        """
        X = np.asarray(X, dtype=float)
        predictions = np.asarray(predictions, dtype=float)
        n = len(X)

        if n < 2:
            raise ValueError(
                f"At least 2 policies required for Lipschitz estimation. Got {n}."
            )

        if self.log_predictions and np.any(predictions <= 0):
            raise ValueError(
                "log_predictions=True requires strictly positive predictions. "
                f"Found {(predictions <= 0).sum()} non-positive values."
            )

        rng = np.random.default_rng(self.random_seed)
        actual_pairs = min(self.n_pairs, n * (n - 1) // 2)
        idx_i = rng.integers(0, n, size=actual_pairs)
        idx_j = rng.integers(0, n, size=actual_pairs)

        # Avoid identical pairs
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % n

        ratios: List[float] = []
        for i, j in zip(idx_i, idx_j):
            d = self.distance_fn(X[i], X[j])
            if d <= 0:
                continue

            if self.log_predictions:
                pred_diff = abs(np.log(predictions[i]) - np.log(predictions[j]))
            else:
                pred_diff = abs(predictions[i] - predictions[j])

            ratios.append(pred_diff / d)

        if not ratios:
            return LipschitzResult(
                lipschitz_constant=0.0,
                n_pairs_sampled=0,
                max_ratio=0.0,
                p95_ratio=0.0,
                p50_ratio=0.0,
                log_predictions=self.log_predictions,
            )

        ratios_arr = np.array(ratios)
        return LipschitzResult(
            lipschitz_constant=float(ratios_arr.max()),
            n_pairs_sampled=len(ratios),
            max_ratio=float(ratios_arr.max()),
            p95_ratio=float(np.percentile(ratios_arr, 95)),
            p50_ratio=float(np.percentile(ratios_arr, 50)),
            log_predictions=self.log_predictions,
        )


@dataclass
class LipschitzResult:
    """
    Result of a sampled Lipschitz constant estimation.

    Attributes
    ----------
    lipschitz_constant:
        Estimated Lipschitz constant (sample maximum of |f(x)-f(x')|/d(x,x')).
    n_pairs_sampled:
        Number of policy pairs used in the estimation.
    max_ratio:
        Same as lipschitz_constant; included for clarity.
    p95_ratio:
        95th percentile of sampled |f(x)-f(x')|/d(x,x') ratios.
    p50_ratio:
        Median of sampled ratios.
    log_predictions:
        Whether predictions were compared in log-space.
    """

    lipschitz_constant: float
    n_pairs_sampled: int
    max_ratio: float
    p95_ratio: float
    p50_ratio: float
    log_predictions: bool


# ---------------------------------------------------------------------------
# C. NSGA2FairnessOptimiser
# ---------------------------------------------------------------------------


class NSGA2FairnessOptimiser:
    """
    NSGA-II multi-objective optimiser for fairness-accuracy trade-offs.

    Wraps the pymoo NSGA-II implementation with the FairnessProblem to
    find the Pareto front of mixing weights over an ensemble of pricing models.

    Each point on the Pareto front represents a different trade-off between:
    - Accuracy (Gini coefficient of predictions)
    - Group fairness (demographic parity)
    - Counterfactual fairness
    - Individual fairness via Lipschitz constant (optional, 4th objective)

    Parameters
    ----------
    models:
        Dict mapping model name to fitted model. Provide at least two models
        that represent different points on the accuracy/fairness trade-off —
        for example, a standard loss-minimising model and a fairness-constrained
        model. The optimiser finds the best ensemble weights between them.
    X:
        Policy-level Polars DataFrame of features. Must contain protected_col.
    y:
        Array-like of actual outcomes. Used for the Gini accuracy objective.
    exposure:
        Array-like of exposure values. Used for exposure-weighted metrics.
    protected_col:
        Name of the protected characteristic column in X.
    prediction_col:
        If provided and present in X, use this column as pre-computed predictions
        for the first model instead of calling model.predict(). Rarely needed.
    cf_tolerance:
        Log-space tolerance for counterfactual fairness classification.
        Default 0.05 (~5% premium change).
    cat_features:
        Categorical feature column names for CatBoost Pool construction.
    lipschitz_feature_cols:
        List of numeric column names in X to use for individual fairness
        estimation. Passing this activates the fourth NSGA-II objective.
        See ``FairnessProblem`` for full documentation.
    lipschitz_distance_fn:
        Custom distance function for Lipschitz estimation. Defaults to
        Euclidean (rarely appropriate — pass a domain-specific function).
    lipschitz_n_pairs:
        Number of policy pairs to sample per Lipschitz evaluation. Default 500.
    lipschitz_log_predictions:
        Whether to compare predictions in log-space. Default True.

    Examples
    --------
    Three-objective mode (default)::

        optimiser = NSGA2FairnessOptimiser(
            models={'base': model_a, 'fair': model_b},
            X=X_test,
            y=y_test,
            exposure=exposure,
            protected_col='gender',
        )
        result = optimiser.run(pop_size=50, n_gen=100, seed=42)
        idx = result.selected_point(weights=[0.5, 0.3, 0.2])
        result.plot_front(highlight=idx)

    Four-objective mode (with individual fairness)::

        optimiser = NSGA2FairnessOptimiser(
            models={'base': model_a, 'fair': model_b},
            X=X_test,
            y=y_test,
            exposure=exposure,
            protected_col='gender',
            lipschitz_feature_cols=['age', 'vehicle_value', 'ncd'],
        )
        result = optimiser.run(pop_size=50, n_gen=100, seed=42)
        idx = result.selected_point(weights=[0.4, 0.2, 0.2, 0.2])
    """

    def __init__(
        self,
        models: Dict[str, Any],
        X: pl.DataFrame,
        y: "np.ndarray | Sequence",
        exposure: "np.ndarray | Sequence",
        protected_col: str,
        prediction_col: Optional[str] = None,
        cf_tolerance: float = 0.05,
        cat_features: Optional[List[str]] = None,
        lipschitz_feature_cols: Optional[List[str]] = None,
        lipschitz_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        lipschitz_n_pairs: int = 500,
        lipschitz_log_predictions: bool = True,
    ) -> None:
        self.models = models
        self.X = X
        self.y = np.asarray(y, dtype=float)
        self.exposure = np.asarray(exposure, dtype=float)
        self.protected_col = protected_col
        self.prediction_col = prediction_col
        self.cf_tolerance = cf_tolerance
        self.cat_features = cat_features or []

        # Build FairnessProblem (pre-computes predictions)
        self.problem = FairnessProblem(
            models=models,
            X=X,
            y=self.y,
            exposure=self.exposure,
            protected_col=protected_col,
            prediction_col=prediction_col,
            cf_tolerance=cf_tolerance,
            cat_features=self.cat_features,
            lipschitz_feature_cols=lipschitz_feature_cols,
            lipschitz_distance_fn=lipschitz_distance_fn,
            lipschitz_n_pairs=lipschitz_n_pairs,
            lipschitz_log_predictions=lipschitz_log_predictions,
        )

    def run(
        self,
        pop_size: int = 100,
        n_gen: int = 200,
        seed: int = 42,
        verbose: bool = False,
    ) -> "ParetoResult":
        """
        Run NSGA-II and return the Pareto front.

        Parameters
        ----------
        pop_size:
            Population size for NSGA-II. Larger populations explore the front
            more thoroughly but take proportionally longer. 50-100 is adequate
            for 2-5 model ensembles.
        n_gen:
            Number of generations to run. Increase if convergence plots show
            the front is still evolving at the last generation.
        seed:
            Random seed for reproducibility.
        verbose:
            If True, print pymoo progress to stdout.

        Returns
        -------
        ParetoResult
        """
        _require_pymoo()
        from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: PLC0415
        from pymoo.optimize import minimize  # noqa: PLC0415
        from pymoo.termination import get_termination  # noqa: PLC0415

        pymoo_problem = self.problem.build_pymoo_problem()
        algorithm = NSGA2(pop_size=pop_size)
        termination = get_termination("n_gen", n_gen)

        res = minimize(
            pymoo_problem,
            algorithm,
            termination,
            seed=seed,
            verbose=verbose,
        )

        # Extract Pareto front solutions
        n_obj = self.problem.n_obj
        F = res.F  # Shape: (n_pareto, n_obj)
        X_weights = res.X  # Shape: (n_pareto, K) — raw mixing weights

        # Normalise weights so they sum to 1
        row_sums = X_weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        X_weights_normalised = X_weights / row_sums

        objective_names = ["neg_gini", "group_unfairness", "cf_unfairness"]
        if n_obj == 4:
            objective_names.append("lipschitz_unfairness")

        return ParetoResult(
            F=F,
            weights=X_weights_normalised,
            model_names=list(self.models.keys()),
            n_gen=n_gen,
            pop_size=pop_size,
            seed=seed,
            objective_names=objective_names,
        )


# ---------------------------------------------------------------------------
# D. TOPSIS selector
# ---------------------------------------------------------------------------


def topsis_select(
    F: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> int:
    """
    Select the best Pareto point using TOPSIS (Technique for Order of
    Preference by Similarity to Ideal Solution).

    TOPSIS identifies the solution that is simultaneously closest to the
    positive ideal solution (best value on each objective) and farthest
    from the negative ideal solution (worst value on each objective).

    For insurance pricing, where objectives are on different scales
    (Gini, parity ratio, counterfactual score, Lipschitz constant),
    TOPSIS normalises before computing distances so that no single objective
    dominates through scale.

    Works with any number of objectives (3 for the default mode, 4 for
    the mode with individual fairness enabled).

    Parameters
    ----------
    F:
        Objective matrix of shape (n_solutions, n_objectives). All objectives
        must be oriented for minimisation (as returned by NSGA-II).
    weights:
        Weight vector of shape (n_objectives,). If None, equal weights are
        used. Weights are normalised to sum to 1 internally. Use these to
        express the relative importance of each objective.
        Example for 3 objectives: [0.5, 0.3, 0.2] weights accuracy highest,
        then group fairness, then counterfactual fairness.
        Example for 4 objectives: [0.4, 0.2, 0.2, 0.2].

    Returns
    -------
    int
        Index into F of the selected solution.

    References
    ----------
    Hwang, C.L. and Yoon, K. (1981). Multiple Attribute Decision Making:
    Methods and Applications. Springer-Verlag, Berlin.
    """
    F = np.asarray(F, dtype=float)
    n_solutions, n_obj = F.shape

    if n_solutions == 0:
        raise ValueError("F must have at least one row.")

    # Validate and normalise weights before early returns so that invalid
    # weight inputs always raise, regardless of the number of solutions.
    if weights is None:
        w = np.ones(n_obj) / n_obj
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n_obj:
            raise ValueError(
                f"weights length ({len(w)}) must match number of objectives ({n_obj})."
            )
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("weights must contain at least one positive value.")
        w = w / w_sum

    if n_solutions == 1:
        return 0

    # Step 1: Normalise the decision matrix (column-wise vector normalisation)
    col_norms = np.linalg.norm(F, axis=0)
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    F_norm = F / col_norms

    # Step 2: Weighted normalised matrix
    V = F_norm * w[np.newaxis, :]

    # Step 3: Ideal (A+) and anti-ideal (A-) solutions
    A_plus = V.min(axis=0)   # minimisation: ideal is the minimum
    A_minus = V.max(axis=0)  # anti-ideal is the maximum

    # Step 4: Separation measures
    D_plus = np.linalg.norm(V - A_plus[np.newaxis, :], axis=1)
    D_minus = np.linalg.norm(V - A_minus[np.newaxis, :], axis=1)

    # Step 5: Relative closeness to the ideal solution
    denom = D_plus + D_minus
    denom = np.where(denom == 0, 1e-12, denom)
    C = D_minus / denom

    # Higher C is better (closer to ideal, farther from anti-ideal)
    return int(np.argmax(C))


# ---------------------------------------------------------------------------
# E. ParetoResult dataclass + plotting
# ---------------------------------------------------------------------------


@dataclass
class ParetoResult:
    """
    Result of an NSGA-II Pareto front search.

    Attributes
    ----------
    F:
        Objective matrix of shape (n_pareto, n_objectives). Each row is a
        non-dominated solution. Objectives are oriented for minimisation.
        Three-objective mode: [neg_gini, group_unfairness, cf_unfairness].
        Four-objective mode: [neg_gini, group_unfairness, cf_unfairness,
        lipschitz_unfairness].
    weights:
        Normalised ensemble weight matrix of shape (n_pareto, K). Each row
        gives the mixing weights over the K pre-trained models that produce
        the corresponding row in F.
    model_names:
        Names of the K models, in the same order as the columns of weights.
    n_gen:
        Number of NSGA-II generations run.
    pop_size:
        NSGA-II population size used.
    seed:
        Random seed used.
    objective_names:
        Names of the objectives in order. Defaults to the three-objective
        names; automatically extended to four names when the Lipschitz
        objective is active.
    """

    F: np.ndarray
    weights: np.ndarray
    model_names: List[str]
    n_gen: int
    pop_size: int
    seed: int
    objective_names: List[str] = field(
        default_factory=lambda: ["neg_gini", "group_unfairness", "cf_unfairness"]
    )

    @property
    def n_solutions(self) -> int:
        """Number of Pareto-optimal solutions found."""
        return len(self.F)

    def selected_point(
        self, weights: Optional[Sequence[float]] = None
    ) -> int:
        """
        Select a single operating point from the Pareto front using TOPSIS.

        Parameters
        ----------
        weights:
            Preference weights over the objectives. Default is equal weights.
            Length must match the number of objectives (3 or 4 depending on
            whether individual fairness is active).
            Pass e.g. [0.5, 0.3, 0.2] for three objectives to weight accuracy
            most heavily, or [0.4, 0.2, 0.2, 0.2] for four objectives.

        Returns
        -------
        int
            Index of the selected solution in self.F and self.weights.
        """
        w = np.asarray(weights, dtype=float) if weights is not None else None
        return topsis_select(self.F, w)

    def plot_front(
        self,
        highlight: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Pareto Front: Fairness vs Accuracy",
    ) -> Any:
        """
        Plot the Pareto front as 2D scatter plots of all pairs of objectives.

        For three objectives, produces three subplots arranged in a single row.
        For four objectives, produces six subplots arranged in two rows of three.

        Requires matplotlib. If not installed, raises ImportError with an
        actionable message.

        Parameters
        ----------
        highlight:
            Index of a solution to highlight (e.g. from selected_point()).
            Shown as a red star marker.
        figsize:
            Figure size in inches. If None, defaults to (14, 5) for three
            objectives or (18, 10) for four objectives.
        title:
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_front(). Install it with:\n"
                "    pip install matplotlib"
            ) from exc

        obj_labels = {
            "neg_gini": "Negative Gini (higher = less accurate)",
            "group_unfairness": "Group Unfairness (1 - parity ratio)",
            "cf_unfairness": "Counterfactual Unfairness",
            "lipschitz_unfairness": "Individual Unfairness (normalised Lipschitz)",
        }
        labels = [obj_labels.get(n, n) for n in self.objective_names]
        n_obj = len(self.objective_names)

        # All pairs of objective indices
        pairs = list(itertools.combinations(range(n_obj), 2))
        n_pairs = len(pairs)

        # Layout: for 3 objectives (3 pairs) use 1 row; for 4 objectives (6 pairs) 2 rows
        if n_obj <= 3:
            n_rows, n_cols = 1, n_pairs
            default_figsize = (14, 5)
        else:
            n_cols = 3
            n_rows = (n_pairs + n_cols - 1) // n_cols
            default_figsize = (18, 5 * n_rows)

        if figsize is None:
            figsize = default_figsize

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Flatten axes to a 1-D array for uniform indexing
        axes_flat = np.array(axes).flatten()

        for ax_idx, (i, j) in enumerate(pairs):
            ax = axes_flat[ax_idx]
            ax.scatter(self.F[:, i], self.F[:, j], c="steelblue", alpha=0.7, s=30)
            if highlight is not None and 0 <= highlight < self.n_solutions:
                ax.scatter(
                    self.F[highlight, i],
                    self.F[highlight, j],
                    c="red",
                    marker="*",
                    s=200,
                    zorder=5,
                    label="Selected",
                )
                ax.legend()
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.set_title(f"{self.objective_names[i]} vs {self.objective_names[j]}")

        # Hide any unused subplots (when n_pairs < n_rows * n_cols)
        for ax_idx in range(n_pairs, len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dict of the Pareto result.

        The F matrix and weights matrix are serialised as lists. Use this
        for storing results in a model review document or database.

        Returns
        -------
        dict
        """
        return {
            "n_solutions": self.n_solutions,
            "n_gen": self.n_gen,
            "pop_size": self.pop_size,
            "seed": self.seed,
            "model_names": self.model_names,
            "objective_names": self.objective_names,
            "F": self.F.tolist(),
            "weights": self.weights.tolist(),
        }

    def to_json(self, path: str) -> None:
        """
        Write the Pareto result to a JSON file at *path*.

        Parameters
        ----------
        path:
            File path for the output JSON.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ParetoResult":
        """
        Reconstruct a ParetoResult from a dict (e.g. loaded from JSON).

        Parameters
        ----------
        d:
            Dict as returned by ``to_dict()``.

        Returns
        -------
        ParetoResult
        """
        return cls(
            F=np.array(d["F"]),
            weights=np.array(d["weights"]),
            model_names=d["model_names"],
            n_gen=d["n_gen"],
            pop_size=d["pop_size"],
            seed=d["seed"],
            objective_names=d.get(
                "objective_names", ["neg_gini", "group_unfairness", "cf_unfairness"]
            ),
        )

    def summary(self) -> str:
        """
        Return a plain-text summary of the Pareto front.

        Returns
        -------
        str
        """
        lines = [
            "=" * 60,
            f"Pareto Front Summary",
            f"Models: {', '.join(self.model_names)}",
            f"Solutions on front: {self.n_solutions}",
            f"NSGA-II: {self.n_gen} generations, pop_size={self.pop_size}",
            "-" * 60,
            "Objective ranges (all minimisation):",
        ]
        for i, obj_name in enumerate(self.objective_names):
            col = self.F[:, i]
            lines.append(
                f"  {obj_name}: "
                f"min={col.min():.4f}, max={col.max():.4f}, "
                f"mean={col.mean():.4f}"
            )

        idx = self.selected_point()
        lines.append("-" * 60)
        lines.append(f"TOPSIS-selected solution (equal weights): index {idx}")
        lines.append(
            f"  Weights: "
            + ", ".join(
                f"{n}={w:.3f}"
                for n, w in zip(self.model_names, self.weights[idx])
            )
        )
        lines.append(
            f"  Objectives: "
            + ", ".join(
                f"{n}={v:.4f}"
                for n, v in zip(self.objective_names, self.F[idx])
            )
        )
        return "\n".join(lines)
