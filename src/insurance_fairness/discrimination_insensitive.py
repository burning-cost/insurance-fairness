"""
discrimination_insensitive.py
-----------------------------
DiscriminationInsensitiveReweighter: sample reweighting for discrimination-insensitive
training.

The standard approach to proxy discrimination — removing the protected attribute from
the feature matrix — does not work when other features are correlated with it. A model
trained on postcodes, vehicle group, and occupation will still learn gender or ethnicity
even after you drop those columns. This is the proxy problem.

Reweighting solves it differently. Rather than changing the features, it reweights the
training examples so that, under the reweighted distribution, the features X are
statistically independent of the protected attribute A. Any downstream model trained
with these sample weights will not be able to exploit A (or its proxies) for prediction
without being penalised by the reweighting.

The approach is grounded in KL divergence minimisation (Miao & Pesenti, 2026,
arXiv:2603.16720): find the probability measure Q closest to the empirical measure P
(in KL divergence) such that X is independent of A under Q. The solution is:

    dQ/dP proportional to P(A = a_i) / P(A = a_i | X_i)

The denominator — the propensity score P(A = a_i | X_i) — is estimated by logistic
regression or a random forest. The numerator is the unconditional (marginal) proportion
of group a_i in the training data.

The resulting weights integrate directly with sklearn's ``sample_weight`` parameter::

    from insurance_fairness import DiscriminationInsensitiveReweighter

    rw = DiscriminationInsensitiveReweighter(protected_col="gender")
    weights = rw.fit_transform(X_train)

    model.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)

References
----------
Miao, K. E. & Pesenti, S. M. (2026). Discrimination-Insensitive Pricing.
arXiv:2603.16720.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ReweighterDiagnostics:
    """
    Diagnostics from a fitted DiscriminationInsensitiveReweighter.

    Attributes
    ----------
    protected_col :
        Name of the protected attribute column.
    n_groups :
        Number of unique values of the protected attribute.
    group_labels :
        Array of unique group labels, in the order used internally.
    marginal_proportions :
        Empirical proportion P(A = a) for each group.
    mean_propensity_by_group :
        Mean P(A = a_i | X_i) estimated by the propensity model, per group.
        Values far below the marginal proportion indicate the group is highly
        predictable from X — meaning proxy discrimination risk is elevated.
    weight_stats :
        Dict with summary statistics of the final weights:
        ``min``, ``max``, ``mean``, ``std``, ``effective_n``.
        effective_n = (sum w)^2 / sum(w^2) — the effective sample size after
        reweighting. Values much less than n indicate highly influential
        observations; consider truncating via ``clip_quantile``.
    propensity_model_score :
        Accuracy of the propensity model on the training data. A score near
        the chance level (1 / n_groups) means X is weakly predictive of A;
        a high score means X is a strong proxy for A.
    """

    protected_col: str
    n_groups: int
    group_labels: np.ndarray
    marginal_proportions: np.ndarray
    mean_propensity_by_group: np.ndarray
    weight_stats: dict[str, float]
    propensity_model_score: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DiscriminationInsensitiveReweighter:
    """
    Reweight training data so that features X are independent of protected
    attribute A, minimising KL divergence from the original distribution.

    The optimal sample weights under the KL divergence criterion (Miao &
    Pesenti 2026, arXiv:2603.16720) are:

        w_i = P(A = a_i) / P(A = a_i | X_i)

    where P(A = a_i) is the unconditional group proportion and
    P(A = a_i | X_i) is the propensity score estimated from data.

    After reweighting, the joint distribution of (X, A) under the sample
    weights has X ⊥ A marginally — any downstream model trained with these
    weights cannot achieve a reduction in loss by exploiting A, either
    directly or via proxy features.

    The weights are normalised to sum to n (the number of training
    observations), so they are compatible with sklearn's
    ``sample_weight`` parameter without changing the effective learning rate.

    Parameters
    ----------
    protected_col :
        Name of the column containing the protected attribute.
    method :
        Propensity estimation method. Options:

        - ``'logistic'``: Logistic regression (default). Fast, interpretable.
          Works well for low-dimensional X and binary A. Multinomial logistic
          regression is used automatically when A has more than two values.
        - ``'forest'``: Random forest classifier. Captures non-linear
          relationships between X and A. Slower but more flexible.

    propensity_model_kwargs :
        Additional keyword arguments passed to the underlying propensity model
        constructor. For example, ``{'C': 0.1}`` reduces logistic regression
        regularisation strength, or ``{'n_estimators': 200}`` increases the
        forest size.

    clip_quantile :
        Clip weights above this quantile to prevent extreme observations from
        dominating the reweighted distribution. For example, ``0.99`` clips
        the top 1% of weights. Set to 1.0 (default) for no clipping. Values
        should be in (0, 1].

    normalise :
        If True (default), normalise weights to sum to n so that the total
        weight equals the original sample size. Set to False to return raw
        importance weights (numerator / denominator).

    random_state :
        Random seed for reproducibility of the propensity model.

    Notes
    -----
    All columns in X (except ``protected_col``) are used as features when
    fitting the propensity model. Categorical columns (dtype object or
    category) are one-hot encoded automatically. Columns with missing values
    raise a ValueError — impute before calling fit().

    References
    ----------
    Miao, K. E. & Pesenti, S. M. (2026). Discrimination-Insensitive Pricing.
    arXiv:2603.16720.
    """

    _VALID_METHODS = ("logistic", "forest")

    def __init__(
        self,
        protected_col: str,
        method: str = "logistic",
        propensity_model_kwargs: dict[str, Any] | None = None,
        clip_quantile: float = 1.0,
        normalise: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if not isinstance(protected_col, str) or not protected_col:
            raise ValueError("protected_col must be a non-empty string.")
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}. Got: {method!r}"
            )
        if not 0.0 < clip_quantile <= 1.0:
            raise ValueError("clip_quantile must be in (0, 1].")

        self.protected_col = protected_col
        self.method = method
        self.propensity_model_kwargs = propensity_model_kwargs or {}
        self.clip_quantile = clip_quantile
        self.normalise = normalise
        self.random_state = random_state

        # Set after fit()
        self._is_fitted: bool = False
        self._propensity_model = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._feature_cols: list[str] = []
        self._group_labels: np.ndarray | None = None
        self._marginal_proportions: np.ndarray | None = None
        self._propensity_model_score: float = float("nan")
        self._ohe_categories: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "DiscriminationInsensitiveReweighter":
        """
        Fit the propensity model on the training data.

        Parameters
        ----------
        X :
            Training feature matrix including the protected attribute column.
            Shape (n, p). The protected attribute must be present as a column
            named ``protected_col``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``protected_col`` is not in X, if X has fewer than 2 rows,
            if the protected attribute has only one unique value, or if any
            column contains missing values.
        TypeError
            If X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame. Got {type(X).__name__}."
            )
        if self.protected_col not in X.columns:
            raise ValueError(
                f"protected_col {self.protected_col!r} not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        if len(X) < 2:
            raise ValueError("X must have at least 2 rows.")

        A = X[self.protected_col]
        unique_groups = A.unique()
        if len(unique_groups) < 2:
            raise ValueError(
                f"protected_col {self.protected_col!r} has only one unique value "
                f"({unique_groups[0]!r}). Cannot compute propensity scores."
            )

        # Check for missing values
        missing_cols = [c for c in X.columns if X[c].isnull().any()]
        if missing_cols:
            raise ValueError(
                f"X contains missing values in columns: {missing_cols}. "
                "Impute before calling fit()."
            )

        # Encode protected attribute
        A_encoded = self._label_encoder.fit_transform(A.values)
        self._group_labels = self._label_encoder.classes_

        # Compute marginal proportions P(A = a)
        n = len(X)
        group_counts = np.bincount(A_encoded)
        self._marginal_proportions = group_counts / n

        # Prepare feature matrix (exclude protected column)
        self._feature_cols = [c for c in X.columns if c != self.protected_col]
        X_features = self._prepare_features(X, fit=True)

        # Fit propensity model
        self._propensity_model = self._build_propensity_model()
        self._propensity_model.fit(X_features, A_encoded)
        self._propensity_model_score = float(
            self._propensity_model.score(X_features, A_encoded)
        )

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute discrimination-insensitive sample weights.

        For each observation i, returns:

            w_i = P(A = a_i) / P(A = a_i | X_i)

        normalised to sum to n.

        Parameters
        ----------
        X :
            Feature matrix including the protected attribute column.
            Must have the same columns as the training data passed to fit().

        Returns
        -------
        weights : np.ndarray, shape (n,)
            Sample weights. Pass to ``model.fit(..., sample_weight=weights)``.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If ``protected_col`` is not in X or X contains missing values.
        TypeError
            If X is not a pandas DataFrame.
        """
        self._check_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame. Got {type(X).__name__}."
            )
        if self.protected_col not in X.columns:
            raise ValueError(
                f"protected_col {self.protected_col!r} not found in X."
            )

        missing_cols = [c for c in X.columns if X[c].isnull().any()]
        if missing_cols:
            raise ValueError(
                f"X contains missing values in columns: {missing_cols}. "
                "Impute before calling transform()."
            )

        A = X[self.protected_col]
        n = len(X)

        # Encode protected attribute — handle unseen labels gracefully
        try:
            A_encoded = self._label_encoder.transform(A.values)
        except ValueError as exc:
            raise ValueError(
                f"protected_col contains values not seen during fit(): {exc}"
            ) from exc

        # Prepare features and get propensity scores P(A | X)
        X_features = self._prepare_features(X, fit=False)
        propensity_proba = self._propensity_model.predict_proba(X_features)
        # propensity_proba shape: (n, n_groups)

        # Extract P(A = a_i | X_i) for each observation
        propensity_scores = propensity_proba[np.arange(n), A_encoded]

        # Clip to avoid division by near-zero propensities
        propensity_scores = np.clip(propensity_scores, 1e-6, 1.0)

        # Numerator: marginal proportion P(A = a_i)
        marginal = self._marginal_proportions[A_encoded]

        # Raw importance weights: dQ/dP
        weights = marginal / propensity_scores

        # Clip extreme weights at specified quantile
        if self.clip_quantile < 1.0:
            cap = float(np.quantile(weights, self.clip_quantile))
            weights = np.minimum(weights, cap)

        # Normalise to sum to n
        if self.normalise:
            total = weights.sum()
            if total > 0:
                weights = weights * (n / total)

        return weights.astype(np.float64)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the propensity model and return sample weights in one step.

        Equivalent to ``fit(X).transform(X)``.

        Parameters
        ----------
        X :
            Training feature matrix including the protected attribute column.

        Returns
        -------
        weights : np.ndarray, shape (n,)
            Sample weights. Pass to ``model.fit(..., sample_weight=weights)``.
        """
        return self.fit(X).transform(X)

    def diagnostics(self, X: pd.DataFrame) -> ReweighterDiagnostics:
        """
        Return diagnostics for a fitted reweighter applied to X.

        Use this to assess whether the propensity model is well-calibrated
        and whether the weights have acceptable variance.

        Parameters
        ----------
        X :
            Feature matrix (same format as fit/transform).

        Returns
        -------
        ReweighterDiagnostics
        """
        self._check_fitted()

        weights = self.transform(X)
        A = X[self.protected_col]
        n = len(X)

        A_encoded = self._label_encoder.transform(A.values)
        X_features = self._prepare_features(X, fit=False)
        propensity_proba = self._propensity_model.predict_proba(X_features)
        propensity_scores = propensity_proba[np.arange(n), A_encoded]

        # Mean propensity per group
        n_groups = len(self._group_labels)
        mean_propensity_by_group = np.array([
            float(np.mean(propensity_scores[A_encoded == g]))
            if np.any(A_encoded == g) else float("nan")
            for g in range(n_groups)
        ])

        # Weight summary statistics
        w_sum = float(weights.sum())
        w_sq_sum = float((weights ** 2).sum())
        effective_n = (w_sum ** 2) / w_sq_sum if w_sq_sum > 0 else 0.0

        weight_stats = {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "effective_n": effective_n,
        }

        return ReweighterDiagnostics(
            protected_col=self.protected_col,
            n_groups=n_groups,
            group_labels=self._group_labels.copy(),
            marginal_proportions=self._marginal_proportions.copy(),
            mean_propensity_by_group=mean_propensity_by_group,
            weight_stats=weight_stats,
            propensity_model_score=self._propensity_model_score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_propensity_model(self):
        """
        Build the propensity estimation model.

        Returns a fitted sklearn-compatible classifier.
        """
        if self.method == "logistic":
            defaults: dict[str, Any] = {
                "max_iter": 1000,
                "random_state": self.random_state,
            }
            defaults.update(self.propensity_model_kwargs)
            return LogisticRegression(**defaults)

        if self.method == "forest":
            try:
                from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn RandomForestClassifier required for method='forest'. "
                    "Install scikit-learn: pip install scikit-learn"
                ) from exc
            defaults = {
                "n_estimators": 100,
                "random_state": self.random_state,
            }
            defaults.update(self.propensity_model_kwargs)
            return RandomForestClassifier(**defaults)

        raise ValueError(f"Unknown method: {self.method!r}")

    def _prepare_features(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """
        Prepare the feature matrix for the propensity model.

        Numeric columns are passed through as-is. Categorical and object
        columns are one-hot encoded. The protected column is excluded.

        Parameters
        ----------
        X :
            Full feature matrix including the protected column.
        fit :
            If True, learn the one-hot encoding categories from X.
            If False, use previously learnt categories (transform mode).

        Returns
        -------
        np.ndarray, shape (n, p_encoded)
        """
        X_sub = X[self._feature_cols].copy()

        cat_cols = [
            c for c in X_sub.columns
            if X_sub[c].dtype == object  # legacy numpy object strings
            or isinstance(X_sub[c].dtype, pd.StringDtype)  # pandas StringDtype
            or isinstance(X_sub[c].dtype, pd.CategoricalDtype)
        ]
        num_cols = [c for c in X_sub.columns if c not in cat_cols]

        parts: list[np.ndarray] = []

        if num_cols:
            parts.append(X_sub[num_cols].values.astype(np.float64))

        for col in cat_cols:
            if fit:
                categories = sorted(X_sub[col].unique().tolist())
                self._ohe_categories[col] = categories
            else:
                categories = self._ohe_categories.get(col, [])

            for cat in categories[:-1]:  # drop last to avoid perfect multicollinearity
                parts.append((X_sub[col] == cat).values.astype(np.float64).reshape(-1, 1))

        if not parts:
            # Edge case: no usable features after excluding protected col
            warnings.warn(
                "No usable feature columns found after excluding the protected column. "
                "All weights will equal 1.0 (marginal proportions only).",
                UserWarning,
                stacklevel=3,
            )
            return np.ones((len(X_sub), 1), dtype=np.float64)

        return np.hstack(parts)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "DiscriminationInsensitiveReweighter has not been fitted. "
                "Call fit() first."
            )

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "unfitted"
        clip = f", clip_quantile={self.clip_quantile}" if self.clip_quantile < 1.0 else ""
        return (
            f"DiscriminationInsensitiveReweighter("
            f"protected_col={self.protected_col!r}, "
            f"method={self.method!r}"
            f"{clip}, "
            f"{fitted})"
        )
