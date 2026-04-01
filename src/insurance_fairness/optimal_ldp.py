"""
optimal_ldp.py
--------------
OptimalLDPMechanism: asymmetric LDP noise that minimises data unfairness
subject to epsilon-privacy constraints.

LDPEpsilonAdvisor: recommend epsilon given dataset size and number of groups.

Based on:

    Ghoukasian, A. & Asoodeh, S. (2025). Optimal Local Differential Privacy
    Mechanisms for Data Unfairness Minimisation. arXiv:2511.16377.

Design rationale
----------------
The standard k-ary randomised response (k-RR) applies *symmetric* noise: every
group gets the same correct-response probability pi. This is convenient but
wasteful — it applies the same privacy cost to the majority and minority alike.
Theorem 2 of Ghoukasian & Asoodeh (2025) shows that for binary attributes, the
unfairness-minimising mechanism under eps-LDP is *asymmetric*: the minority
class gets a higher probability of receiving a correct response, and the
majority class absorbs more noise.

The intuition: data unfairness is driven by how well the noised distribution
D_Q matches the true distribution. Noise on the minority distorts the
distribution more (proportionally), so it is better to protect the majority
more aggressively and let the minority signal through.

For the multi-valued (K > 2) case, we solve a linear fractional program via
scipy's linprog after a Charnes-Cooper transformation that makes it linear.

References
----------
Ghoukasian, A. & Asoodeh, S. (2025). arXiv:2511.16377.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


# ---------------------------------------------------------------------------
# OptimalLDPMechanism
# ---------------------------------------------------------------------------


class OptimalLDPMechanism:
    """
    Asymmetric LDP mechanism minimising data unfairness (Ghoukasian & Asoodeh, 2025).

    For binary attributes (K=2), Theorem 2 gives a closed-form solution.
    For K > 2, the perturbation matrix is found via a linear fractional program
    (Charnes-Cooper transformation) solved with scipy.optimize.linprog.

    The resulting matrix Q is *row-stochastic*: Q[j, i] = probability that
    a respondent in group j reports group i. The diagonal entries Q[j,j] are
    the correct-response probabilities; the off-diagonal entries carry the
    false-response mass.

    Parameters
    ----------
    epsilon :
        LDP privacy budget. Larger epsilon = less noise = better utility.
        Typical range: [0.5, 5.0]. Values below 0.3 give C1 > 3 and are
        rarely useful in practice.
    k :
        Number of groups. k=2 for binary protected attributes (gender, smoker).
    group_prevalences :
        Array of shape (k,) giving the true group prevalences P(S=j). If None,
        uniform prevalences are assumed. These drive which group absorbs more
        noise in the asymmetric optimisation. Shape (k,).

    Attributes
    ----------
    perturbation_matrix_ : np.ndarray
        Fitted K×K noise matrix Q, set after calling ``fit`` or on first use
        of ``privatise``. Available immediately if group_prevalences is
        supplied to the constructor.
    """

    def __init__(
        self,
        epsilon: float,
        k: int = 2,
        group_prevalences: Optional[np.ndarray] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")

        self.epsilon = float(epsilon)
        self.k = int(k)

        if group_prevalences is not None:
            group_prevalences = np.asarray(group_prevalences, dtype=float)
            if group_prevalences.shape != (k,):
                raise ValueError(
                    f"group_prevalences must have shape ({k},), got {group_prevalences.shape}"
                )
            if abs(group_prevalences.sum() - 1.0) > 1e-5:
                raise ValueError(
                    f"group_prevalences must sum to 1, got {group_prevalences.sum():.6f}"
                )
            if np.any(group_prevalences <= 0):
                raise ValueError("group_prevalences must be strictly positive")

        self.group_prevalences = group_prevalences
        self._Q: np.ndarray | None = None

        # Pre-compute if we have prevalences
        if group_prevalences is not None:
            self._Q = self._solve(group_prevalences)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def privatise(
        self,
        true_labels: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Apply the optimal LDP mechanism to true group labels.

        Each label j is replaced by a random draw from the j-th row of Q.

        Parameters
        ----------
        true_labels :
            Integer labels in {0, ..., K-1}. Shape (n,).
        rng :
            Optional numpy random Generator for reproducibility.

        Returns
        -------
        np.ndarray
            Privatised labels, same shape as true_labels.
        """
        true_labels = np.asarray(true_labels, dtype=int)
        if np.any((true_labels < 0) | (true_labels >= self.k)):
            raise ValueError(
                f"true_labels must be in {{0, ..., {self.k - 1}}}, "
                f"got range [{true_labels.min()}, {true_labels.max()}]."
            )

        Q = self.perturbation_matrix  # triggers solve if needed

        if rng is None:
            rng = np.random.default_rng()

        n = len(true_labels)
        privatised = np.empty(n, dtype=int)
        for j in range(self.k):
            mask = true_labels == j
            count = int(mask.sum())
            if count > 0:
                privatised[mask] = rng.choice(self.k, size=count, p=Q[j])

        return privatised

    @property
    def perturbation_matrix(self) -> np.ndarray:
        """
        K×K perturbation matrix Q.

        Q[j, i] = probability that group-j individual reports group i.
        Diagonal entries are the correct-response probabilities.

        For uniform prevalences (the default), this matches k-RR for K=2
        but is asymmetric for K > 2 when prevalences differ.

        Raises
        ------
        RuntimeError
            If group_prevalences was not provided and fit() has not been called.
        """
        if self._Q is None:
            # Default: uniform prevalences
            uniform = np.ones(self.k) / self.k
            self._Q = self._solve(uniform)
        return self._Q.copy()

    def fit(self, group_prevalences: np.ndarray) -> "OptimalLDPMechanism":
        """
        Compute the optimal Q given empirical group prevalences.

        Call this when you learn the prevalences from the data rather than
        specifying them in advance.

        Parameters
        ----------
        group_prevalences :
            Empirical P(S=j) for j=0,...,K-1. Shape (k,).

        Returns
        -------
        self
        """
        group_prevalences = np.asarray(group_prevalences, dtype=float)
        if group_prevalences.shape != (self.k,):
            raise ValueError(
                f"group_prevalences must have shape ({self.k},), got {group_prevalences.shape}"
            )
        if abs(group_prevalences.sum() - 1.0) > 1e-5:
            raise ValueError(
                f"group_prevalences must sum to 1, got {group_prevalences.sum():.6f}"
            )
        self.group_prevalences = group_prevalences
        self._Q = self._solve(group_prevalences)
        return self

    def unfairness_bound(self) -> float:
        """
        Marginal distribution shift: TV distance between noised and true label distributions.

        Computes 0.5 * ||Q^T p - p||_1, the total variation distance between
        the noised marginal distribution (p_noised = Q^T p) and the true distribution p.

        This measures how much the privatisation distorts the observable group
        frequencies. For the binary optimal mechanism, the asymmetric noise allocation
        shifts the minority group's observed frequency less than k-RR would, at the
        cost of shifting the majority group's frequency more.

        Note: for K=3 with the LP-optimal Q, this is often 0.0 because the LP can
        find mechanisms that perfectly preserve the marginal distribution.

        Returns
        -------
        float
            TV distance in [0, 0.5].
        """
        Q = self.perturbation_matrix
        p = self.group_prevalences
        if p is None:
            p = np.ones(self.k) / self.k

        # Compute noised distribution: p_noised[i] = sum_j Q[j,i] * p[j]
        p_noised = Q.T @ p

        # TV distance = 0.5 * ||p_noised - p||_1
        return float(0.5 * np.sum(np.abs(p_noised - p)))

    # ------------------------------------------------------------------
    # Private solvers
    # ------------------------------------------------------------------

    def _solve(self, p: np.ndarray) -> np.ndarray:
        """Dispatch to binary closed-form or multi-valued LP."""
        if self.k == 2:
            return self._solve_binary(p)
        else:
            return self._solve_lp(p)

    def _solve_binary(self, p: np.ndarray) -> np.ndarray:
        """
        Closed-form optimal mechanism for K=2 (Theorem 2, Ghoukasian & Asoodeh 2025).

        When P(S=0) < P(S=1): minority is group 0.
            p* = 1 - exp(-eps)/2  (correct response for minority)
            q* = 1/2              (correct response for majority)

        When P(S=1) < P(S=0): minority is group 1.
            p* = 1/2
            q* = 1 - exp(-eps)/2

        The matrix is:
            Q = [[p,   1-p],
                 [1-q, q  ]]
        where p, q are correct-response probabilities for groups 0, 1.
        """
        exp_neg_eps = np.exp(-self.epsilon)
        p0, p1 = p[0], p[1]

        if p0 < p1:
            # Group 0 is minority — it gets preferential noise allocation
            p_correct_0 = 1.0 - exp_neg_eps / 2.0
            p_correct_1 = 0.5
        elif p1 < p0:
            # Group 1 is minority
            p_correct_0 = 0.5
            p_correct_1 = 1.0 - exp_neg_eps / 2.0
        else:
            # Equal prevalences: symmetric (same as k-RR for K=2)
            p_correct_0 = 1.0 - exp_neg_eps / 2.0
            p_correct_1 = 1.0 - exp_neg_eps / 2.0

        Q = np.array([
            [p_correct_0, 1.0 - p_correct_0],
            [1.0 - p_correct_1, p_correct_1],
        ])
        return Q

    def _solve_lp(self, p: np.ndarray) -> np.ndarray:
        """
        Solve the min-max linear fractional program for K > 2.

        Objective: minimise Delta(D_Q) = 0.5 * ||Q^T p - p||_1
        Subject to: Standard epsilon-LDP column constraints:
                        Q[j,i] - exp(eps)*Q[j\'i] <= 0 for all j != j\', all i
                    Row-stochastic: sum_i Q[j,i] = 1, Q[j,i] >= 0

        The epsilon-LDP constraint in the standard (column-wise) form ensures:
        for any two inputs j, j\' and any output i:
            P(Z=i|X=j) / P(Z=i|X=j\') <= exp(eps)

        This is stricter than the row-wise spec constraint (q_jj - exp(eps)*q_ij <= 0)
        but correctly implements standard local differential privacy.

        We reformulate the L1 objective using auxiliary variables t_i >= 0:
            minimise sum_i t_i
            subject to: (Q^T p)_i - p_i <= t_i
                        p_i - (Q^T p)_i <= t_i

        Variables are flattened Q (K^2 entries) and t (K entries).
        """
        from scipy.optimize import linprog  # noqa: PLC0415

        K = self.k
        e = self.epsilon
        exp_e = np.exp(e)

        # Variable layout: [Q_00, Q_01, ..., Q_{K-1,K-1}, t_0, ..., t_{K-1}]
        n_q = K * K
        n_t = K
        n_vars = n_q + n_t

        # Objective: minimise sum of t_i
        c = np.zeros(n_vars)
        c[n_q:] = 1.0

        # Build inequality constraints A_ub @ x <= b_ub
        constraints_ub = []
        b_ub = []

        # Standard LDP constraints (column-wise):
        # Q[j,i] / Q[j',i] <= exp(eps) for all j != j', all i
        # => Q[j,i] - exp(eps)*Q[j',i] <= 0
        for i in range(K):
            for j in range(K):
                for jp in range(K):
                    if j != jp:
                        row = np.zeros(n_vars)
                        row[j * K + i] = 1.0           # Q[j,i]
                        row[jp * K + i] = -exp_e        # -exp(eps)*Q[j',i]
                        constraints_ub.append(row)
                        b_ub.append(0.0)

        # L1 unfairness constraints:
        # (Q^T p)_i - p_i <= t_i   =>  sum_j Q[j,i]*p[j] - p_i - t_i <= 0
        # p_i - (Q^T p)_i <= t_i   =>  -sum_j Q[j,i]*p[j] + p_i - t_i <= 0
        for i in range(K):
            # Positive direction
            row_pos = np.zeros(n_vars)
            for j in range(K):
                row_pos[j * K + i] = p[j]   # Q[j,i]*p[j]
            row_pos[n_q + i] = -1.0          # -t_i
            constraints_ub.append(row_pos)
            b_ub.append(p[i])

            # Negative direction
            row_neg = np.zeros(n_vars)
            for j in range(K):
                row_neg[j * K + i] = -p[j]  # -Q[j,i]*p[j]
            row_neg[n_q + i] = -1.0          # -t_i
            constraints_ub.append(row_neg)
            b_ub.append(-p[i])

        A_ub = np.array(constraints_ub)
        b_ub = np.array(b_ub)

        # Equality constraints: row-stochastic Q
        # sum_i Q[j,i] = 1 for each j
        A_eq = np.zeros((K, n_vars))
        b_eq = np.ones(K)
        for j in range(K):
            for i in range(K):
                A_eq[j, j * K + i] = 1.0

        # Bounds: Q[j,i] in [0,1], t_i >= 0
        bounds = [(0.0, 1.0)] * n_q + [(0.0, None)] * n_t

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if not result.success:
            warnings.warn(
                f"OptimalLDPMechanism LP solver did not find an optimal solution "
                f"(status={result.status}: {result.message}). "
                "Falling back to symmetric k-RR mechanism.",
                UserWarning,
                stacklevel=3,
            )
            return self._fallback_krr()

        Q_flat = result.x[:n_q]
        Q = Q_flat.reshape(K, K)

        # Clip to [0,1] and re-normalise rows (numerical clean-up)
        Q = np.clip(Q, 0.0, 1.0)
        row_sums = Q.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        Q = Q / row_sums

        return Q

    def _fallback_krr(self) -> np.ndarray:
        """Symmetric k-RR fallback for LP solver failure."""
        K = self.k
        exp_e = np.exp(self.epsilon)
        pi = exp_e / (K - 1 + exp_e)
        pi_bar = (1.0 - pi) / (K - 1)
        Q = np.full((K, K), pi_bar)
        np.fill_diagonal(Q, pi)
        return Q


# ---------------------------------------------------------------------------
# LDPEpsilonAdvisor
# ---------------------------------------------------------------------------


class LDPEpsilonAdvisor:
    """
    Recommend epsilon given dataset size and number of groups.

    The noise amplification factor C1 quantifies how much the LDP correction
    inflates statistical bounds. It is defined as:

        C1 = (pi + K - 2) / (K*pi - 1)

    where pi = exp(eps) / (K - 1 + exp(eps)) is the correct-response
    probability under k-RR.

    At K=2:
        eps=0.5  -> C1 ~ 2.45  (145% inflation of the bound)
        eps=1.0  -> C1 ~ 1.73
        eps=2.0  -> C1 ~ 1.27
        eps=5.0  -> C1 ~ 1.01  (effectively no inflation)

    The generalisation bound scales O(C1 * K^2 / sqrt(n)), so the
    target_bound_inflation parameter sets the maximum tolerable C1 - 1.

    Parameters
    ----------
    n_samples :
        Number of training observations.
    k :
        Number of protected groups.
    target_bound_inflation :
        Maximum acceptable fractional inflation of the generalisation bound
        due to LDP noise. Default 0.30 (30% inflation: C1 = 1.30).
        Lower is stricter.
    """

    def __init__(
        self,
        n_samples: int,
        k: int = 2,
        target_bound_inflation: float = 0.30,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if target_bound_inflation <= 0:
            raise ValueError(f"target_bound_inflation must be positive, got {target_bound_inflation}")

        self.n_samples = int(n_samples)
        self.k = int(k)
        self.target_bound_inflation = float(target_bound_inflation)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pi_from_eps(eps: float, k: int) -> float:
        """Convert epsilon to correct-response probability."""
        exp_e = np.exp(float(eps))
        return exp_e / (k - 1 + exp_e)

    @staticmethod
    def _c1_from_pi(pi: float, k: int) -> float:
        """Compute noise amplification factor C1."""
        denom = k * pi - 1.0
        if abs(denom) < 1e-12:
            return float("inf")
        return (pi + k - 2) / denom

    @staticmethod
    def _c1_from_eps(eps: float, k: int) -> float:
        """Compute C1 directly from epsilon."""
        pi = LDPEpsilonAdvisor._pi_from_eps(eps, k)
        return LDPEpsilonAdvisor._c1_from_pi(pi, k)

    def _gen_bound(self, eps: float, delta: float = 0.05, vc_dim: int = 10) -> float:
        """Generalisation bound: K * sqrt((vc_dim + log(2/delta)) / (2n)) * 2 * C1 * K."""
        C1 = self._c1_from_eps(eps, self.k)
        inner = (vc_dim + np.log(2.0 / delta)) / (2.0 * self.n_samples)
        return float(self.k * np.sqrt(inner) * 2.0 * C1 * self.k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self) -> dict:
        """
        Recommend a minimum epsilon that meets the target_bound_inflation.

        Searches over epsilon in [0.1, 10.0] and returns the smallest value
        where C1 <= 1 + target_bound_inflation.

        Returns
        -------
        dict with keys:
            - ``epsilon``: recommended epsilon (float)
            - ``pi``: correct-response probability at recommended epsilon
            - ``C1``: noise amplification factor
            - ``gen_bound``: generalisation bound estimate
            - ``rationale``: human-readable explanation string
        """
        target_c1 = 1.0 + self.target_bound_inflation

        # Grid search over epsilon
        candidates = np.linspace(0.1, 10.0, 1000)
        chosen_eps = None
        for eps in candidates:
            c1 = self._c1_from_eps(eps, self.k)
            if c1 <= target_c1:
                chosen_eps = float(eps)
                break

        if chosen_eps is None:
            # Even eps=10 doesn't meet target; return eps=10 as best effort
            chosen_eps = 10.0
            warnings.warn(
                f"No epsilon in [0.1, 10.0] achieves C1 <= {target_c1:.2f} for K={self.k}. "
                "The target_bound_inflation may be too strict. Returning eps=10.",
                UserWarning,
                stacklevel=2,
            )

        pi = self._pi_from_eps(chosen_eps, self.k)
        c1 = self._c1_from_pi(pi, self.k)
        gen_bound = self._gen_bound(chosen_eps)

        rationale = (
            f"With K={self.k} groups and n={self.n_samples:,} samples, "
            f"epsilon={chosen_eps:.2f} gives pi={pi:.4f} (correct-response rate), "
            f"C1={c1:.3f} ({(c1 - 1) * 100:.1f}% bound inflation vs. non-private). "
            f"Generalisation bound ~ {gen_bound:.4f}."
        )

        return {
            "epsilon": chosen_eps,
            "pi": pi,
            "C1": c1,
            "gen_bound": gen_bound,
            "rationale": rationale,
        }

    def sweep(self, epsilons: Optional[np.ndarray] = None) -> "pl.DataFrame":
        """
        Sweep over epsilon values and return a summary DataFrame.

        Parameters
        ----------
        epsilons :
            Array of epsilon values to evaluate. If None, uses a log-spaced
            grid from 0.1 to 10.

        Returns
        -------
        polars.DataFrame
            Columns: epsilon, pi, C1, gen_bound, bound_inflation_pct.

        Raises
        ------
        ImportError
            If polars is not installed.
        """
        if not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is required for LDPEpsilonAdvisor.sweep(). "
                "Install it with: pip install polars"
            )

        if epsilons is None:
            epsilons = np.logspace(np.log10(0.1), np.log10(10.0), 50)

        rows = []
        for eps in epsilons:
            pi = self._pi_from_eps(float(eps), self.k)
            c1 = self._c1_from_pi(pi, self.k)
            gen_bound = self._gen_bound(float(eps))
            rows.append({
                "epsilon": float(eps),
                "pi": float(pi),
                "C1": float(c1),
                "gen_bound": float(gen_bound),
                "bound_inflation_pct": float((c1 - 1.0) * 100.0),
            })

        return pl.DataFrame(rows)
