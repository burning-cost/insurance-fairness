"""
discrimination_insensitive.py
-----------------------------
DiscriminationInsensitiveReweighter: propensity-based sample reweighting to
achieve statistical independence between features X and protected attribute A
without removing A from the model.

The standard approach to proxy discrimination is to drop correlated features or
apply fairness constraints at training time. Both create accuracy/fairness
trade-offs. An alternative is to reweight the training sample so that the
*weighted* distribution of X is independent of A — then a model trained on
reweighted data learns P(Y|X) not P(Y|X,A).

Miao & Pesenti (2026) formalise this as KL divergence minimisation. The solution
is surprisingly clean: the optimal sample weight for observation i is

    w_i = P(A = a_i) / P(A = a_i | X_i)

where P(A|X) is the propensity score — the probability of group membership given
features. Observations where A is easy to predict from X get down-weighted
(they carry excess group signal); observations where A is hard to predict get
up-weighted (they are group-ambiguous and carry more useful X→Y signal).

This does *not* require dropping A from the model. After reweighting, you can
still include any feature — the reweighting has already broken the X-A dependence
in the training distribution.

Practical guidance
------------------
The choice of propensity model matters. Logistic regression is fast and
interpretable; random forest captures non-linear group separability better.
Use the ``diagnostics`` property to check whether your propensity model found
real group separation — if ``effective_n`` is close to n, the reweighting is
mild (X and A were weakly correlated to begin with).

Usage::

    from insurance_fairness import DiscriminationInsensitiveReweighter

    reweighter = DiscriminationInsensitiveReweighter(propensity_model="logistic")
    reweighter.fit(X_train, A_train)
    weights = reweighter.transform(X_train, A_train)

    model.fit(X_train, y_train, sample_weight=weights)

References
----------
Miao, W. & Pesenti, S. M. (2026). Discrimination-Insensitive Insurance Pricing
via KL Divergence Minimisation. arXiv:2603.16720.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ReweighterDiagnostics:
    """Diagnostics from a fitted DiscriminationInsensitiveReweighter.

    Attributes
    ----------
    effective_n:
        Effective sample size after reweighting: (sum w_i)^2 / sum(w_i^2).
        If this is close to n, the X-A dependence was weak and reweighting had
        little effect. If it is much lower than n, the propensity model found
        strong group separation and the reweighting is substantial.
    per_group_propensity:
        Dict mapping each group label to the mean propensity score P(A=a|X)
        within that group. High mean propensity means the group is easily
        predicted from X; values near the marginal P(A=a) indicate independence.
    propensity_scores:
        Raw propensity scores P(A=a_i|X_i) for each training observation.
        Length n. Useful for further diagnostics or visualisation.
    n_samples:
        Total number of training observations.
    n_groups:
        Number of distinct groups in A.
    """

    effective_n: float
    per_group_propensity: dict[str | int, float]
    propensity_scores: np.ndarray
    n_samples: int
    n_groups: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DiscriminationInsensitiveReweighter:
    """Propensity-based sample reweighter for discrimination-insensitive training.

    Computes per-observation sample weights w_i = P(A=a_i) / P(A=a_i|X_i)
    that, when passed as ``sample_weight`` to any sklearn-compatible model,
    train on a distribution where X and A are (approximately) independent.

    Parameters
    ----------
    propensity_model:
        Which classifier to use for estimating P(A|X). Either ``"logistic"``
        (LogisticRegression, fast, interpretable) or ``"random_forest"``
        (RandomForestClassifier, captures non-linear separability). Default
        is ``"logistic"``.
    clip_weights:
        Upper percentile at which to clip extreme weights. Default 99.0. Set to
        100.0 to disable clipping. Extreme weights occur when some observations
        are near-perfectly predicted by A from X.
    min_propensity:
        Floor for propensity scores before division to prevent numerical blow-up.
        Default 1e-6.
    logistic_kwargs:
        Extra kwargs forwarded to LogisticRegression. Ignored when
        ``propensity_model="random_forest"``.
    rf_kwargs:
        Extra kwargs forwarded to RandomForestClassifier. Ignored when
        ``propensity_model="logistic"``.
    random_state:
        Random seed for reproducibility.

    Examples
    --------
    Binary protected attribute::

        reweighter = DiscriminationInsensitiveReweighter(propensity_model="logistic")
        reweighter.fit(X_train, gender)
        weights = reweighter.transform(X_train, gender)
        glm.fit(X_train, y_train, sample_weight=weights)

    Multi-class protected attribute::

        reweighter = DiscriminationInsensitiveReweighter(propensity_model="random_forest")
        reweighter.fit(X_train, region)          # 10 UK regions
        weights = reweighter.transform(X_train, region)
    """

    def __init__(
        self,
        propensity_model: Literal["logistic", "random_forest"] = "logistic",
        clip_weights: float = 99.0,
        min_propensity: float = 1e-6,
        logistic_kwargs: dict | None = None,
        rf_kwargs: dict | None = None,
        random_state: int = 42,
    ) -> None:
        if propensity_model not in ("logistic", "random_forest"):
            raise ValueError(
                f"propensity_model must be 'logistic' or 'random_forest', "
                f"got {propensity_model!r}."
            )
        if not (0.0 < clip_weights <= 100.0):
            raise ValueError(
                f"clip_weights must be in (0, 100], got {clip_weights}."
            )
        if min_propensity <= 0.0:
            raise ValueError(
                f"min_propensity must be positive, got {min_propensity}."
            )

        self.propensity_model = propensity_model
        self.clip_weights = clip_weights
        self.min_propensity = min_propensity
        self.logistic_kwargs = logistic_kwargs or {}
        self.rf_kwargs = rf_kwargs or {}
        self.random_state = random_state

        self._clf: LogisticRegression | RandomForestClassifier | None = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._marginal_probs: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._training_X_arr: np.ndarray | None = None
        self._training_A_enc: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_classifier(self) -> LogisticRegression | RandomForestClassifier:
        """Instantiate the propensity classifier."""
        if self.propensity_model == "logistic":
            defaults: dict = {
                "max_iter": 1000,
                "random_state": self.random_state,
                "C": 1.0,
                "solver": "lbfgs",
            }
            defaults.update(self.logistic_kwargs)
            return LogisticRegression(**defaults)
        else:
            defaults = {
                "n_estimators": 100,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            defaults.update(self.rf_kwargs)
            return RandomForestClassifier(**defaults)

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Convert X to a float numpy array, handling pandas DataFrames.

        Handles the pandas 2.x behaviour change where string columns may report
        as ``StringDtype`` (new) rather than ``object`` dtype (old). Uses both
        ``is_object_dtype`` and ``is_string_dtype`` guards.
        """
        try:
            import pandas as pd
            from pandas.api.types import is_object_dtype, is_string_dtype

            if isinstance(X, pd.DataFrame):
                X = X.copy()
                for col in X.columns:
                    if is_object_dtype(X[col]) or is_string_dtype(X[col]):
                        # Encode string/object columns as integer codes.
                        # fillna before encoding so LabelEncoder doesn't choke.
                        X[col] = X[col].fillna("__missing__")
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                return X.to_numpy(dtype=float)
        except ImportError:
            pass

        return np.asarray(X, dtype=float)

    def _encode_protected(self, A: np.ndarray) -> np.ndarray:
        """Encode protected attribute to integer labels 0..K-1."""
        try:
            import pandas as pd

            if isinstance(A, pd.Series):
                A = A.to_numpy()
        except ImportError:
            pass
        return self._label_encoder.transform(np.asarray(A).ravel())

    def _compute_weights(
        self, X_arr: np.ndarray, A_enc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute raw weights and propensity scores.

        Returns
        -------
        weights:
            Per-observation weights w_i = P(A=a_i) / P(A=a_i|X_i), normalised
            so that sum(weights) == n.
        propensity_scores:
            P(A=a_i|X_i) for each observation i.
        """
        proba = self._clf.predict_proba(X_arr)  # shape (n, K)
        n = len(A_enc)

        # Vectorised: select the column for each observation's class
        idx = np.arange(n)
        raw_propensity = proba[idx, A_enc]
        propensity_scores = np.maximum(raw_propensity, self.min_propensity)

        # Marginal P(A=a_i) for each observation
        marginal_i = self._marginal_probs[A_enc]

        weights = marginal_i / propensity_scores

        # Clip extreme weights
        if self.clip_weights < 100.0:
            cap = float(np.percentile(weights, self.clip_weights))
            weights = np.minimum(weights, cap)

        # Normalise so weights sum to n (consistent with sklearn's expectation)
        weights = weights * (n / weights.sum())

        return weights, propensity_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, A: np.ndarray) -> "DiscriminationInsensitiveReweighter":
        """Fit the propensity model P(A|X).

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p). May be a numpy array or pandas
            DataFrame. Categorical columns in DataFrames are label-encoded
            automatically.
        A:
            Protected attribute, shape (n,). May be binary or multi-class,
            integer or string labels.

        Returns
        -------
        self
        """
        X_arr = self._prepare_features(X)

        try:
            import pandas as pd

            A_raw = A.to_numpy().ravel() if isinstance(A, pd.Series) else np.asarray(A).ravel()
        except ImportError:
            A_raw = np.asarray(A).ravel()

        if len(X_arr) != len(A_raw):
            raise ValueError(
                f"X and A must have the same number of observations. "
                f"Got X.shape[0]={len(X_arr)}, len(A)={len(A_raw)}."
            )
        if len(A_raw) < 2:
            raise ValueError("At least 2 observations are required.")

        self._label_encoder.fit(A_raw)
        self._classes = self._label_encoder.classes_
        A_enc = self._label_encoder.transform(A_raw)

        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError(
                f"Protected attribute A must have at least 2 distinct values. "
                f"Found only: {self._classes.tolist()}."
            )

        # Marginal class probabilities P(A=k) for k in 0..K-1
        self._marginal_probs = np.bincount(A_enc, minlength=n_classes) / len(A_enc)

        # Fit propensity classifier
        self._clf = self._build_classifier()
        self._clf.fit(X_arr, A_enc)

        # Store training data for the diagnostics property
        self._training_X_arr = X_arr
        self._training_A_enc = A_enc

        self._fitted = True
        return self

    def transform(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute sample weights for (X, A).

        Parameters
        ----------
        X:
            Feature matrix, same format as passed to ``fit``.
        A:
            Protected attribute, same format as passed to ``fit``.

        Returns
        -------
        weights:
            Float array of shape (n,) suitable for passing directly to
            ``model.fit(..., sample_weight=weights)``.

        Raises
        ------
        RuntimeError:
            If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError(
                "DiscriminationInsensitiveReweighter is not fitted. "
                "Call fit() before transform()."
            )

        X_arr = self._prepare_features(X)
        A_enc = self._encode_protected(A)

        if len(X_arr) != len(A_enc):
            raise ValueError(
                f"X and A must have the same number of observations. "
                f"Got X.shape[0]={len(X_arr)}, len(A)={len(A_enc)}."
            )

        weights, _ = self._compute_weights(X_arr, A_enc)
        return weights

    def fit_transform(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Fit propensity model and return sample weights in one step.

        Equivalent to ``fit(X, A).transform(X, A)``.

        Parameters
        ----------
        X:
            Feature matrix.
        A:
            Protected attribute.

        Returns
        -------
        weights:
            Float array of shape (n,).
        """
        return self.fit(X, A).transform(X, A)

    @property
    def diagnostics(self) -> ReweighterDiagnostics:
        """Diagnostics from the most recent ``fit`` call.

        Returns
        -------
        ReweighterDiagnostics
            Dataclass with effective_n, per_group_propensity, propensity_scores,
            n_samples, and n_groups.

        Raises
        ------
        RuntimeError:
            If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError(
                "DiscriminationInsensitiveReweighter is not fitted. "
                "Call fit() first."
            )

        weights, propensity_scores = self._compute_weights(
            self._training_X_arr, self._training_A_enc
        )
        n = len(weights)
        effective_n = float(weights.sum() ** 2 / (weights ** 2).sum())

        per_group_propensity: dict = {}
        for k, label in enumerate(self._classes):
            mask = self._training_A_enc == k
            if mask.any():
                per_group_propensity[label] = float(propensity_scores[mask].mean())

        return ReweighterDiagnostics(
            effective_n=effective_n,
            per_group_propensity=per_group_propensity,
            propensity_scores=propensity_scores,
            n_samples=n,
            n_groups=len(self._classes),
        )
