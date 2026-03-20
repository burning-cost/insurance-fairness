"""
privatized_audit.py
-------------------
PrivatizedFairnessAudit: discrimination-free pricing when protected attributes
are privatised via local differential privacy or estimated from proxies.

Implements the MPTP-LDP protocol from:

    Zhang, Liu & Shi (2025). Discrimination-Free Insurance Pricing with
    Privatized Sensitive Attributes. arXiv:2504.11775.

The core idea is that the fair premium averages K group-specific models over a
FIXED reference distribution P*, not the data-conditional P(D|X):

    h*(X_tilde) = sum_k f_k(X_tilde) * P*(D=k)

By fixing P*, indirect discrimination through the observed covariate distribution
is broken. With uniform P*, every individual is priced as if drawn from a
population with equal group proportions — which satisfies UK Equality Act
requirements for gender-neutral motor pricing.

The challenge addressed here is that in practice the sensitive attribute D is
either:
  (a) collected but privatised via randomised response before sharing (LDP), or
  (b) never observed and must be estimated from proxy covariates.

In both cases, the group-specific models f_k cannot be trained directly. Lemma
4.2 in the paper shows the training objective can be reformulated as a
reweighted sum over the noised labels S, using correction matrices derived from
the LDP noise rate pi.

When pi is unknown (case b), Procedure 4.5 provides an anchor-point estimator:
find covariate regions where P(D=k|X) approx 1 and read off pi from the
predicted probability there.

References
----------
Zhang, Z., Liu, J. & Shi, Y. (2025). Discrimination-Free Insurance Pricing
with Privatized Sensitive Attributes. arXiv:2504.11775.

Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PrivatizedAuditResult:
    """
    Results of a PrivatizedFairnessAudit run.

    Attributes
    ----------
    fair_premium :
        Discrimination-free predictions h*(X) for all training observations.
        Shape (n,).
    group_models :
        List of K fitted estimators [f_0, f_1, ..., f_{K-1}]. Each is a
        sklearn-compatible model with a ``predict`` method.
    pi_estimated :
        The estimated (or given) correct-response probability pi.
    pi_known :
        True if pi was supplied directly or derived from a known epsilon.
        False if pi was estimated via anchor points (Procedure 4.5).
    p_star :
        Reference distribution P*(D=k) used when computing the fair premium.
        Shape (K,).
    p_corrected :
        Noise-corrected group prevalences P_hat(D=k), recovered by applying
        T_inv to the raw noised-label marginals. Shape (K,).
    bound_95 :
        Theorem 4.3 generalisation bound at delta=0.05. Scales with noise
        amplification factor C1 and complexity of function class.
    anchor_quality :
        max_i P_hat(S=j*|X_i*) from the anchor-point classifier. The closer
        to 1.0, the more reliable the pi estimate. None if pi was supplied.
    negative_weight_frac :
        Fraction of (observation, group) pairs with negative reweighting
        before clipping. Values above 0.05 indicate the LDP noise is too
        heavy for reliable correction.
    """

    fair_premium: np.ndarray
    group_models: list
    pi_estimated: float
    pi_known: bool
    p_star: np.ndarray
    p_corrected: np.ndarray
    bound_95: float
    anchor_quality: float | None
    negative_weight_frac: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PrivatizedFairnessAudit:
    """
    Discrimination-free pricing with privatised sensitive attributes (LDP).

    Implements the MPTP-LDP protocol from Zhang, Liu & Shi (2025).

    The fair premium is::

        h*(X) = sum_k f_k(X) * P*(D=k)

    where f_k is a group-specific model trained on reweighted samples and
    P* is a fixed reference distribution (uniform or empirical).

    Noise Rate (pi)
    ---------------
    Provide one of:

    - ``pi``: the correct-response probability directly (e.g. from an auditor
      who applied randomised response with known parameters).
    - ``epsilon``: the LDP privacy budget; pi is derived as
      ``exp(epsilon) / (K - 1 + exp(epsilon))``.
    - ``X_anchor``: covariate matrix for anchor observations. A CatBoost
      classifier is trained to predict S from X_anchor, then Procedure 4.5
      recovers pi from the maximum predicted probabilities in groups.

    Parameters
    ----------
    n_groups :
        Number of protected groups K. Must match the number of distinct values
        in S passed to ``fit``.
    epsilon :
        LDP privacy budget. Mutually exclusive with ``pi``.
    pi :
        Known correct-response probability. Overrides ``epsilon``.
    reference_distribution :
        How to construct P*:
        - ``"uniform"``: P*(D=k) = 1/K for all k. Recommended for UK gender
          neutrality, eliminates indirect discrimination via composition.
        - ``"empirical"``: P*(D=k) = noise-corrected marginal. Preserves the
          population risk structure.
        - numpy array: custom distribution summing to 1. Shape (K,).
    loss :
        Loss family for group-specific models:
        ``"poisson"`` (default), ``"gaussian"``, or ``"bernoulli"``.
    n_anchor_groups :
        Number of partitions n1 for Procedure 4.5. Default 20.
    nuisance_backend :
        ``"catboost"`` uses CatBoostRegressor/Classifier for group models.
        ``"sklearn"`` uses scikit-learn GLMs (PoissonRegressor, Ridge,
        LogisticRegression).
    random_state :
        Random seed for reproducibility.

    References
    ----------
    Zhang, Z., Liu, J. & Shi, Y. (2025). arXiv:2504.11775.
    """

    def __init__(
        self,
        n_groups: int = 2,
        epsilon: float | None = None,
        pi: float | None = None,
        reference_distribution: Union[str, np.ndarray] = "empirical",
        loss: str = "poisson",
        n_anchor_groups: int = 20,
        nuisance_backend: str = "catboost",
        random_state: int | None = None,
    ) -> None:
        if loss not in ("poisson", "gaussian", "bernoulli"):
            raise ValueError(f"loss must be 'poisson', 'gaussian', or 'bernoulli', got {loss!r}")
        if nuisance_backend not in ("catboost", "sklearn"):
            raise ValueError(
                f"nuisance_backend must be 'catboost' or 'sklearn', got {nuisance_backend!r}"
            )
        if isinstance(reference_distribution, str) and reference_distribution not in (
            "empirical",
            "uniform",
        ):
            raise ValueError(
                "reference_distribution must be 'empirical', 'uniform', or a numpy array"
            )

        self.n_groups = n_groups
        self.epsilon = epsilon
        self.pi = pi
        self.reference_distribution = reference_distribution
        self.loss = loss
        self.n_anchor_groups = n_anchor_groups
        self.nuisance_backend = nuisance_backend
        self.random_state = random_state

        # Fitted attributes (set during fit)
        self.pi_: float | None = None
        self.pi_bar_: float | None = None
        self.T_inv_: np.ndarray | None = None
        self.Pi_inv_: np.ndarray | None = None
        self.p_corrected_: np.ndarray | None = None
        self.p_star_: np.ndarray | None = None
        self.group_models_: list = []
        self.negative_weight_frac_: float = 0.0
        self.anchor_quality_: float | None = None
        self.pi_known_: bool = False
        self.fair_predictions_: np.ndarray | None = None
        self._n_fit_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        S: np.ndarray,
        X_anchor: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
    ) -> "PrivatizedFairnessAudit":
        """
        Fit K group-specific models and compute the fair premium.

        Parameters
        ----------
        X :
            Non-sensitive feature matrix. Shape (n, p).
        Y :
            Outcome vector (claim frequency, pure premium, etc.). Shape (n,).
        S :
            Privatised sensitive attribute. Integer labels in {0, ..., K-1}.
            Shape (n,).
        X_anchor :
            Anchor-point features. Required when neither ``pi`` nor ``epsilon``
            is specified. Shape (n, q) — can be the same as X or a subset of
            features relevant for predicting group membership.
        exposure :
            Exposure weights (policy years, risk months, etc.). Shape (n,).
            If None, all observations are weighted equally.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        S = np.asarray(S, dtype=int)
        n = len(Y)
        K = self.n_groups
        self._n_fit_ = n

        if len(X) != n or len(S) != n:
            raise ValueError("X, Y, and S must have the same length.")
        if np.any((S < 0) | (S >= K)):
            raise ValueError(f"S must contain integer labels in {{0, ..., {K - 1}}}.")

        # ------------------------------------------------------------------
        # Step 1: Noise rate estimation
        # ------------------------------------------------------------------
        if self.pi is not None:
            self.pi_ = float(self.pi)
            self.pi_known_ = True
        elif self.epsilon is not None:
            exp_e = np.exp(float(self.epsilon))
            self.pi_ = exp_e / (K - 1 + exp_e)
            self.pi_known_ = True
        elif X_anchor is not None:
            self.pi_, self.anchor_quality_ = self._estimate_pi_anchor(
                np.asarray(X_anchor, dtype=float), S, K
            )
            self.pi_known_ = False
        else:
            raise ValueError(
                "Must supply one of: pi, epsilon, or X_anchor. "
                "Use pi= or epsilon= for known LDP parameters; pass X_anchor "
                "when the noise rate must be estimated from anchor points."
            )

        # pi_bar: probability of any given incorrect response
        # For K groups: K-1 incorrect outcomes, each with prob pi_bar
        # such that pi + (K-1)*pi_bar = 1
        self.pi_bar_ = (1.0 - self.pi_) / (K - 1) if K > 1 else 0.0

        # Validate K*pi - 1 != 0 (matrix would be singular)
        denom = K * self.pi_ - 1.0
        if abs(denom) < 1e-10:
            raise ValueError(
                f"K*pi - 1 is near zero (K={K}, pi={self.pi_:.4f}). "
                "Cannot invert the LDP correction matrix. Try a larger epsilon."
            )

        # ------------------------------------------------------------------
        # Step 2: Build correction matrices
        # ------------------------------------------------------------------
        T_inv, Pi_inv, p_hat_raw, p_corrected = self._build_correction_matrices(S, K)

        self.T_inv_ = T_inv
        self.Pi_inv_ = Pi_inv
        self.p_corrected_ = p_corrected

        # ------------------------------------------------------------------
        # Step 3: Compute reference distribution P*
        # ------------------------------------------------------------------
        if isinstance(self.reference_distribution, str):
            if self.reference_distribution == "uniform":
                p_star = np.ones(K) / K
            else:  # "empirical"
                p_star = p_corrected.copy()
        else:
            p_star = np.asarray(self.reference_distribution, dtype=float)
            if p_star.shape != (K,):
                raise ValueError(
                    f"reference_distribution array must have shape ({K},), "
                    f"got {p_star.shape}."
                )
            if abs(p_star.sum() - 1.0) > 1e-5:
                raise ValueError(
                    f"reference_distribution must sum to 1, got {p_star.sum():.6f}."
                )

        self.p_star_ = p_star

        # ------------------------------------------------------------------
        # Step 4: Train K group-specific models via reweighted samples
        # ------------------------------------------------------------------
        self.group_models_ = []
        self.negative_weight_frac_ = 0.0

        for k in range(K):
            # Scale factor for group k: T_inv[k] @ p_hat_raw (scalar)
            scale = float(T_inv[k] @ p_hat_raw)

            # Per-observation weight when training f_k
            weights_k = Pi_inv[k, S] * scale

            # Record and clip negative weights
            n_neg = int((weights_k < 0).sum())
            self.negative_weight_frac_ += n_neg / (K * n)

            weights_k = np.maximum(weights_k, 0.0)
            w_sum = weights_k.sum()
            if w_sum < 1e-12:
                # Degenerate — all weight zero; fall back to uniform
                weights_k = np.ones(n) / n
            else:
                weights_k = weights_k / w_sum

            # Incorporate exposure if provided
            if exposure is not None:
                exp_arr = np.asarray(exposure, dtype=float)
                weights_k = weights_k * exp_arr

            f_k = self._fit_group_model(X, Y, weights_k)
            self.group_models_.append(f_k)

        # Warn if negative weight fraction is large
        if self.negative_weight_frac_ > 0.05:
            warnings.warn(
                f"negative_weight_frac={self.negative_weight_frac_:.3f} > 5%. "
                "The LDP noise is too heavy for reliable correction — consider a "
                "larger epsilon or more data. Clipped weights introduce bias.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 5: Compute fair premiums on training data
        # ------------------------------------------------------------------
        preds = np.stack([m.predict(X) for m in self.group_models_], axis=1)  # (n, K)
        self.fair_predictions_ = preds @ p_star

        return self

    def predict_fair_premium(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the discrimination-free premium for new observations.

        Returns h*(X) = sum_k f_k(X) * P*(D=k).

        Parameters
        ----------
        X :
            Feature matrix, shape (n_new, p).

        Returns
        -------
        np.ndarray
            Fair premium predictions, shape (n_new,).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        preds = np.stack([m.predict(X) for m in self.group_models_], axis=1)
        return preds @ self.p_star_

    def correction_matrices(self) -> dict:
        """
        Return the LDP correction matrices and noise parameters.

        Returns
        -------
        dict with keys:
            - ``Pi_inv``: (K, K) array — full reweighting matrix (Lemma 4.2)
            - ``T_inv``: (K, K) array — noise-inversion matrix
            - ``pi``: float — correct-response probability
            - ``pi_bar``: float — incorrect-response probability per group
            - ``C1``: float — noise amplification factor (K*pi-1)^{-1}*(pi+K-2)
        """
        self._check_fitted()
        K = self.n_groups
        C1 = (self.pi_ + K - 2) / (K * self.pi_ - 1)
        return {
            "Pi_inv": self.Pi_inv_.copy(),
            "T_inv": self.T_inv_.copy(),
            "pi": self.pi_,
            "pi_bar": self.pi_bar_,
            "C1": C1,
        }

    def statistical_bound(self, delta: float = 0.05, vc_dim: int | None = None) -> float:
        """
        Generalisation bound from Theorem 4.3 (Zhang et al., 2025).

        Bound = K * sqrt((vc_dim + ln(2/delta)) / (2*n)) * 2 * C1 * K

        This is the excess risk guarantee for the LDP-corrected model relative
        to the oracle model trained on true (unnoised) attributes. The bound
        scales O(C1 * K^2 / sqrt(n)), so heavier noise or more groups requires
        more data to maintain the same guarantee quality.

        Parameters
        ----------
        delta :
            Failure probability. Default 0.05 (95% confidence).
        vc_dim :
            VC dimension of the function class F. If None, defaults to
            ``min(n_features + 1, 100)`` — a conservative heuristic.

        Returns
        -------
        float
            Upper bound on excess empirical risk.
        """
        self._check_fitted()
        K = self.n_groups
        n = self._n_fit_
        C1 = (self.pi_ + K - 2) / (K * self.pi_ - 1)

        if vc_dim is None:
            # Heuristic: linear model in p features has VC dim p+1
            # Cap at 100 to avoid unreasonably loose bounds
            vc_dim = min(100, 10)  # conservative default

        inner = (vc_dim + np.log(2.0 / delta)) / (2.0 * n)
        bound = K * np.sqrt(inner) * 2.0 * C1 * K
        return float(bound)

    def audit_report(self) -> PrivatizedAuditResult:
        """
        Return a structured audit result.

        Returns
        -------
        PrivatizedAuditResult
        """
        self._check_fitted()
        return PrivatizedAuditResult(
            fair_premium=self.fair_predictions_.copy(),
            group_models=list(self.group_models_),
            pi_estimated=self.pi_,
            pi_known=self.pi_known_,
            p_star=self.p_star_.copy(),
            p_corrected=self.p_corrected_.copy(),
            bound_95=self.statistical_bound(delta=0.05),
            anchor_quality=self.anchor_quality_,
            negative_weight_frac=self.negative_weight_frac_,
        )

    def minimum_n_recommended(self, delta: float = 0.05, target_bound: float = 0.05) -> int:
        """
        Minimum sample size for the statistical bound to meet target_bound.

        Derived by inverting Theorem 4.3: n >= (vc_dim + ln(2/delta)) / 2 *
        (2*C1*K^2 / target_bound)^2.

        Parameters
        ----------
        delta :
            Failure probability. Default 0.05.
        target_bound :
            Desired maximum excess risk bound. Default 0.05.

        Returns
        -------
        int
            Recommended minimum n.
        """
        self._check_fitted()
        K = self.n_groups
        C1 = (self.pi_ + K - 2) / (K * self.pi_ - 1)
        vc_dim = 10  # same heuristic as statistical_bound
        factor = (2.0 * C1 * K * K / target_bound) ** 2
        n_min = int(np.ceil((vc_dim + np.log(2.0 / delta)) / 2.0 * factor))
        return n_min

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_pi_anchor(
        self, X_anchor: np.ndarray, S: np.ndarray, K: int
    ) -> tuple[float, float]:
        """
        Estimate pi via anchor points (Procedure 4.5).

        Train a classifier P_hat(S=j*|X) on (X_anchor, S), then partition
        samples into n_anchor_groups groups, compute C1 in each, average, and
        invert to recover pi.

        Returns (pi_hat, anchor_quality).
        """
        # Reference group: majority class in S
        ref_group = int(np.bincount(S, minlength=K).argmax())

        # Train anchor classifier
        clf = self._make_anchor_classifier()
        clf.fit(X_anchor, S)

        probs = clf.predict_proba(X_anchor)  # (n, K)
        probs_ref = probs[:, ref_group]

        anchor_quality = float(probs_ref.max())
        if anchor_quality < 0.90:
            warnings.warn(
                f"anchor_quality={anchor_quality:.3f} < 0.90. "
                "No covariate region achieves near-certain group identification; "
                "C1 estimation will be biased. Consider adding more informative "
                "anchor observations.",
                UserWarning,
                stacklevel=3,
            )

        # Partition indices into n_anchor_groups groups
        n = len(S)
        n1 = min(self.n_anchor_groups, n)
        group_indices = np.array_split(np.arange(n), n1)

        C1_estimates = []
        for idx in group_indices:
            if len(idx) == 0:
                continue
            pi_hat_k = float(probs_ref[idx].max())
            denom_k = K * pi_hat_k - 1.0
            if abs(denom_k) < 1e-10:
                continue
            C1_k = (pi_hat_k + K - 2) / denom_k
            C1_estimates.append(C1_k)

        if not C1_estimates:
            raise RuntimeError(
                "Could not estimate pi from anchor points: all partition groups "
                "produced degenerate C1 estimates. Check that X_anchor is informative."
            )

        C1 = float(np.mean(C1_estimates))

        # Algebraic inverse of C1 = (pi + K - 2) / (K*pi - 1):
        # C1*(K*pi - 1) = pi + K - 2
        # C1*K*pi - C1 = pi + K - 2
        # pi*(C1*K - 1) = C1 + K - 2
        # pi = (C1 + K - 2) / (C1*K - 1)
        denom_inv = C1 * K - 1.0
        if abs(denom_inv) < 1e-10:
            raise RuntimeError(
                f"Degenerate anchor-point estimate: C1={C1:.4f}, K={K}. "
                "C1*K - 1 is near zero."
            )
        pi_hat = (C1 + K - 2) / denom_inv
        pi_hat = float(np.clip(pi_hat, 1.0 / K + 1e-6, 1.0 - 1e-6))

        return pi_hat, anchor_quality

    def _build_correction_matrices(
        self, S: np.ndarray, K: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build T_inv, Pi_inv, raw marginals p_hat_raw, and corrected p_corrected.

        T_inv[i,i] = (pi + K - 2) / (K*pi - 1)
        T_inv[i,j] = (pi - 1)     / (K*pi - 1)  for i != j
        """
        pi = self.pi_
        pi_bar = self.pi_bar_
        n = len(S)
        denom = K * pi - 1.0

        T_inv = np.full((K, K), (pi - 1.0) / denom)
        np.fill_diagonal(T_inv, (pi + K - 2.0) / denom)

        # Raw noised-label marginals
        p_hat_raw = np.bincount(S, minlength=K).astype(float) / n

        # De-noise: p_corrected = T_inv @ p_hat_raw
        p_corrected = T_inv @ p_hat_raw
        p_corrected = np.maximum(p_corrected, 1e-6)
        p_corrected = p_corrected / p_corrected.sum()

        # Pi_inv: full reweighting matrix (Lemma 4.2)
        # Pi_inv[i,i] = T_inv[i,i] * (pi*p[i] + sum_{l!=i} pi_bar*p[l]) / p[i]
        # Pi_inv[i,j] = T_inv[i,j] * (pi_bar*p[i] + sum_{l!=i: l!=j actually
        #               we need sum of remaining} ...) / p[i]  for i != j
        #
        # From the spec (Lemma 4.2 rearrangement):
        # Pi_inv[i,j] for i!=j:
        #   = T_inv[i,j] * (pi_bar*p[i] + sum_{d'!=i} pi*p[d']) / p[i]
        # Note: "sum_{d'!=i} pi*p[d']" sums pi*p[l] for l != i
        Pi_inv = np.zeros((K, K))
        p = p_corrected

        for i in range(K):
            sum_others = sum(p[l] for l in range(K) if l != i)

            # Diagonal
            Pi_inv[i, i] = (
                T_inv[i, i]
                * (pi * p[i] + pi_bar * sum_others)
                / p[i]
            )

            # Off-diagonal
            for j in range(K):
                if j != i:
                    # sum over d' != i of pi*p[d']
                    sum_others_pi = sum(pi * p[l] for l in range(K) if l != i)
                    Pi_inv[i, j] = (
                        T_inv[i, j]
                        * (pi_bar * p[i] + sum_others_pi)
                        / p[i]
                    )

        return T_inv, Pi_inv, p_hat_raw, p_corrected

    def _fit_group_model(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray,
    ):
        """
        Fit a single group-specific model f_k.

        Uses CatBoost or sklearn depending on nuisance_backend.
        """
        if self.nuisance_backend == "catboost":
            return self._fit_catboost(X, Y, sample_weight)
        else:
            return self._fit_sklearn(X, Y, sample_weight)

    def _fit_catboost(self, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray):
        """Fit group model using CatBoost."""
        try:
            import catboost  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "catboost is required for nuisance_backend='catboost'. "
                "Install it with: pip install catboost"
            ) from e

        common_params = {
            "iterations": 200,
            "depth": 4,
            "learning_rate": 0.1,
            "verbose": 0,
            "random_seed": self.random_state if self.random_state is not None else 42,
            "allow_writing_files": False,
        }

        if self.loss == "poisson":
            model = catboost.CatBoostRegressor(loss_function="Poisson", **common_params)
        elif self.loss == "bernoulli":
            model = catboost.CatBoostClassifier(loss_function="Logloss", **common_params)
        else:  # gaussian
            model = catboost.CatBoostRegressor(loss_function="RMSE", **common_params)

        model.fit(X, Y, sample_weight=sample_weight)

        if self.loss == "bernoulli":
            # Wrap to return probabilities as predict
            return _CatBoostClassifierWrapper(model)

        return model

    def _fit_sklearn(self, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray):
        """Fit group model using scikit-learn GLMs."""
        from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge  # noqa: PLC0415

        if self.loss == "poisson":
            model = PoissonRegressor(max_iter=300, alpha=1e-4)
        elif self.loss == "bernoulli":
            model = LogisticRegression(max_iter=300, C=1e4)
        else:  # gaussian
            model = Ridge(alpha=1.0)

        model.fit(X, Y, sample_weight=sample_weight)
        return model

    def _make_anchor_classifier(self):
        """Create anchor-point classifier (always CatBoost if available, else sklearn)."""
        try:
            import catboost  # noqa: PLC0415

            return catboost.CatBoostClassifier(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                verbose=0,
                random_seed=self.random_state if self.random_state is not None else 42,
                allow_writing_files=False,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415

            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
            )

    def _check_fitted(self) -> None:
        """Raise if fit() has not been called."""
        if self.pi_ is None:
            raise RuntimeError(
                "PrivatizedFairnessAudit has not been fitted. Call fit() first."
            )


# ---------------------------------------------------------------------------
# Helper: wrap CatBoostClassifier to expose .predict() returning proba[:,1]
# ---------------------------------------------------------------------------


class _CatBoostClassifierWrapper:
    """Thin wrapper so CatBoostClassifier exposes a regression-style predict()."""

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
