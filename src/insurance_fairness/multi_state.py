"""
multi_state.py
--------------
MultiStateTransitionFairness: discrimination-free pricing in multi-state
insurance models (critical illness, income protection, long-term care).

Standard discrimination-free pricing (Lindholm et al. 2022) assumes a single
transition from "insured" to "claim". Most long-tail insurance products involve
multiple states with multiple transition intensities: healthy -> sick -> dead,
or healthy -> disabled -> recovered / dead. The protected attribute may affect
different transitions very differently — excluding gender from a critical
illness model that affects recovery rates but not incidence rates requires
transition-level fairness correction, not a single blended premium adjustment.

This module extends the Lindholm marginalisation to multi-state Markov models.
Each transition intensity mu_{ij}(age, x) is fitted separately as a Poisson GLM.
LindholmCorrector is applied per-transition to obtain discrimination-free
intensity rates, and then KolmogorovPremiumCalculator integrates the resulting
state occupancy probabilities to a net premium via the matrix differential
equation dp/dt = p(t) Q(t), where Q is the generator matrix.

The approach handles:
- Competing transition states (e.g., sick -> recovered vs. sick -> dead)
- Age-dependent intensities via log-link Poisson regression on exposures
- Interval-split observation records (each row = one age-band for one policy)
- Protected attribute marginalisation per-transition via LindholmCorrector

Usage::

    import polars as pl
    from insurance_fairness.multi_state import (
        TransitionDataBuilder,
        PoissonTransitionFitter,
        MultiStateTransitionFairness,
        KolmogorovPremiumCalculator,
        MultiStateFairnessReport,
    )

    # Build transition-level DataFrames from raw observation data
    builder = TransitionDataBuilder(
        state_from_col="state_from",
        state_to_col="state_to",
        age_col="age_entry",
        exposure_col="exposure",
    )
    transition_data = builder.build(df)

    # Fit Poisson GLMs per transition
    fitter = PoissonTransitionFitter(feature_cols=["age", "gender", "occupation"])
    fitted = fitter.fit(transition_data)

    # Run full fairness audit
    audit = MultiStateTransitionFairness(
        protected_attrs=["gender"],
        feature_cols=["age", "occupation"],
        transitions=["healthy->sick", "sick->dead"],
        discount_rate=0.05,
    )
    report = audit.run(df, D)
    print(report.summary())

References
----------
Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
Pricing. ASTIN Bulletin 52(1), 55-89.

Norberg, R. (1991). Reserves in life and pension insurance. Scandinavian
Actuarial Journal, 1991(1), 3-24.

Christiansen, M. (2012). Multistate models in health insurance. AStA Advances
in Statistical Analysis, 96(2), 155-186.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable

import numpy as np
import polars as pl
from scipy.linalg import expm
from scipy.optimize import minimize

from insurance_fairness.optimal_transport.correction import LindholmCorrector


__all__ = [
    "TransitionDataBuilder",
    "PoissonTransitionFitter",
    "MultiStateTransitionFairness",
    "KolmogorovPremiumCalculator",
    "MultiStateFairnessReport",
]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MultiStateFairnessReport:
    """
    Before/after fairness audit results for a multi-state insurance model.

    Attributes
    ----------
    transitions :
        Names of transitions audited (e.g. ["healthy->sick", "sick->dead"]).
    premium_before :
        Dict mapping group label -> mean premium before fairness correction.
    premium_after :
        Dict mapping group label -> mean premium after per-transition
        Lindholm marginalisation.
    transition_corrections :
        Dict mapping transition name -> fractional change in mean intensity
        after marginalisation. 0.0 means no change; 0.1 means 10% reduction.
    n_policies :
        Number of policies in the audit dataset.
    protected_attrs :
        Protected attribute column names that were marginalised.
    """

    transitions: list[str]
    premium_before: dict[str, float]
    premium_after: dict[str, float]
    transition_corrections: dict[str, float]
    n_policies: int
    protected_attrs: list[str]

    def summary(self) -> str:
        """Return a plain-text summary suitable for an FCA evidence pack."""
        lines = [
            "MultiState Fairness Report",
            "=" * 40,
            f"Policies audited: {self.n_policies}",
            f"Protected attributes: {', '.join(self.protected_attrs)}",
            f"Transitions: {', '.join(self.transitions)}",
            "",
            "Premium by group (before -> after):",
        ]
        all_groups = sorted(
            set(list(self.premium_before.keys()) + list(self.premium_after.keys()))
        )
        for grp in all_groups:
            before = self.premium_before.get(grp, float("nan"))
            after = self.premium_after.get(grp, float("nan"))
            lines.append(f"  {grp}: {before:.4f} -> {after:.4f}")
        lines.append("")
        lines.append("Intensity corrections per transition:")
        for tr, corr in self.transition_corrections.items():
            lines.append(f"  {tr}: {corr:+.4f} (fractional change)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TransitionDataBuilder
# ---------------------------------------------------------------------------


class TransitionDataBuilder:
    """
    Prepare per-transition DataFrames from raw multi-state observation data.

    Each input row represents one policy-period: the policyholder was in
    state_from at entry, was observed for ``exposure`` years, and either
    transitioned to state_to or was censored (state_to == state_from or a
    sentinel value like "censored").

    The builder groups by (state_from, state_to) pair to produce one
    DataFrame per transition. Each output DataFrame contains:

    - All covariate columns (age plus any extras)
    - ``event``: 1 if this row ended in the relevant transition, else 0
    - ``exposure``: person-years of exposure

    Parameters
    ----------
    state_from_col :
        Name of the column holding the origin state.
    state_to_col :
        Name of the column holding the destination state (or "censored").
    age_col :
        Name of the column holding entry age (numeric).
    exposure_col :
        Name of the column holding exposure in years.
    censor_value :
        Sentinel value in state_to_col indicating no transition (censored).
        Default "censored". Can also pass None to treat same-state as censored.
    """

    def __init__(
        self,
        state_from_col: str = "state_from",
        state_to_col: str = "state_to",
        age_col: str = "age",
        exposure_col: str = "exposure",
        censor_value: str | None = "censored",
    ) -> None:
        self.state_from_col = state_from_col
        self.state_to_col = state_to_col
        self.age_col = age_col
        self.exposure_col = exposure_col
        self.censor_value = censor_value

    def build(
        self,
        df: pl.DataFrame,
        covariate_cols: list[str] | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Build per-transition DataFrames.

        Parameters
        ----------
        df :
            Raw observation DataFrame. Must contain state_from_col,
            state_to_col, age_col, exposure_col, and any covariate columns.
        covariate_cols :
            Covariate columns to retain in the output. If None, all columns
            except the state/exposure columns are treated as covariates.

        Returns
        -------
        dict mapping transition name (e.g. "healthy->sick") to a DataFrame
        with columns: covariate_cols + ["event", "exposure"].
        """
        required = {
            self.state_from_col,
            self.state_to_col,
            self.age_col,
            self.exposure_col,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {sorted(missing)}"
            )

        if covariate_cols is None:
            covariate_cols = [
                c for c in df.columns if c not in required
            ]
        # Always include age in covariates if not already there
        if self.age_col not in covariate_cols:
            covariate_cols = [self.age_col] + covariate_cols

        # Identify all distinct (from, to) transitions (excluding censored)
        state_from = df[self.state_from_col]
        state_to = df[self.state_to_col]

        transitions: set[tuple[str, str]] = set()
        for sf, st in zip(state_from.to_list(), state_to.to_list()):
            sf_s = str(sf)
            st_s = str(st)
            if st_s == self.censor_value:
                continue
            if sf_s == st_s:
                # Same-state: treat as censored unless censor_value is None
                if self.censor_value is not None:
                    continue
            transitions.add((sf_s, st_s))

        result: dict[str, pl.DataFrame] = {}

        for (sf, st) in sorted(transitions):
            transition_name = f"{sf}->{st}"

            # Indicator: did this observation end in transition sf->st?
            event_expr = (
                (pl.col(self.state_from_col).cast(pl.String) == sf)
                & (pl.col(self.state_to_col).cast(pl.String) == st)
            ).cast(pl.Int32).alias("event")

            # Only keep rows where the origin state is sf (at-risk rows)
            # Plus rows from other states that may be in the data — we keep
            # all at-risk rows for sf (even if they go elsewhere or censor)
            at_risk_mask = pl.col(self.state_from_col).cast(pl.String) == sf

            transition_df = (
                df.filter(at_risk_mask)
                .with_columns(event_expr)
                .select(covariate_cols + ["event", self.exposure_col])
                .rename({self.exposure_col: "exposure"})
            )

            result[transition_name] = transition_df

        if not result:
            raise ValueError(
                "No transitions found in data. Check state_from_col, "
                "state_to_col, and censor_value."
            )

        return result


# ---------------------------------------------------------------------------
# PoissonTransitionFitter
# ---------------------------------------------------------------------------


def _col_to_numeric(series: pl.Series, encoding: dict | None) -> tuple[np.ndarray, dict]:
    """
    Convert a polars Series to a float64 numpy array.

    For numeric dtypes: direct cast.
    For string/categorical dtypes: label-encode using an existing or new encoding map.

    Returns (array, encoding_map). encoding_map maps str -> int; None for numeric cols.
    """
    dtype = series.dtype
    if dtype in (pl.Categorical, pl.String):
        if encoding is None:
            unique_vals = sorted(series.drop_nulls().unique().to_list())
            encoding = {v: i for i, v in enumerate(unique_vals)}
        arr = np.array(
            [encoding.get(str(v), 0) for v in series.to_list()],
            dtype=np.float64,
        )
        return arr, encoding
    else:
        return series.cast(pl.Float64).to_numpy(), encoding


class _FittedTransitionModel:
    """Internal: a single fitted Poisson GLM for one transition."""

    def __init__(
        self,
        coefficients: np.ndarray,
        feature_cols: list[str],
        transition_name: str,
        encodings: dict[str, dict | None] | None = None,
    ) -> None:
        self.coefficients = coefficients
        self.feature_cols = feature_cols
        self.transition_name = transition_name
        # encodings[col_name] = {str_val: int_code} or None for numeric cols
        self.encodings: dict[str, dict | None] = encodings or {}
        self._is_fitted = True

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Predict intensity rates (events per unit exposure) for each row.

        Parameters
        ----------
        df :
            DataFrame containing feature_cols. May also contain other columns.

        Returns
        -------
        np.ndarray of shape (n,), the predicted intensity mu = exp(X @ beta).
        """
        X = self._build_design_matrix(df)
        eta = X @ self.coefficients
        return np.exp(eta)

    def _build_design_matrix(self, df: pl.DataFrame) -> np.ndarray:
        """Extract feature matrix with intercept from DataFrame."""
        n = df.shape[0]
        cols = []
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in DataFrame for transition "
                    f"'{self.transition_name}'. Available: {df.columns}"
                )
            enc = self.encodings.get(col, None)
            arr, _ = _col_to_numeric(df[col], enc)
            cols.append(arr)
        if not cols:
            return np.ones((n, 1))
        X = np.column_stack([np.ones(n)] + cols)
        return X


class PoissonTransitionFitter:
    """
    Fit independent Poisson GLMs for each transition intensity.

    Each transition mu_{ij}(x) = exp(beta_ij^T x) is fitted separately via
    maximum Poisson log-likelihood with log-link. Uses scipy.optimize.minimize
    (L-BFGS-B) directly — no statsmodels dependency.

    The Poisson log-likelihood for n observations is:

        L(beta) = sum_k [ e_k * exp(x_k^T beta) - y_k * (x_k^T beta) ]

    where y_k is the event count (0/1) and e_k is the exposure.

    Parameters
    ----------
    feature_cols :
        Covariate columns to use as predictors. An intercept is always added.
    max_iter :
        Maximum iterations for the L-BFGS-B solver.
    tol :
        Convergence tolerance passed to scipy.optimize.minimize.
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.feature_cols = feature_cols or []
        self.max_iter = max_iter
        self.tol = tol
        self._fitted_models: dict[str, _FittedTransitionModel] = {}

    def fit(
        self,
        transition_data: dict[str, pl.DataFrame],
    ) -> "PoissonTransitionFitter":
        """
        Fit one Poisson GLM per transition.

        Parameters
        ----------
        transition_data :
            Output of TransitionDataBuilder.build(). Keys are transition names;
            values are DataFrames with columns feature_cols + ["event", "exposure"].

        Returns
        -------
        self
        """
        self._fitted_models = {}

        for transition_name, df in transition_data.items():
            model = self._fit_one(transition_name, df)
            self._fitted_models[transition_name] = model

        return self

    def _fit_one(
        self,
        transition_name: str,
        df: pl.DataFrame,
    ) -> _FittedTransitionModel:
        """Fit a single Poisson GLM for one transition."""
        n = df.shape[0]
        y = df["event"].cast(pl.Float64).to_numpy()
        exposure = df["exposure"].cast(pl.Float64).to_numpy()

        # Build design matrix (intercept + features)
        # String/categorical columns are label-encoded; numeric columns cast directly.
        feature_cols_present = [c for c in self.feature_cols if c in df.columns]
        encodings: dict[str, dict | None] = {}
        cols = []
        for c in feature_cols_present:
            arr, enc = _col_to_numeric(df[c], None)
            encodings[c] = enc
            cols.append(arr)
        if cols:
            X = np.column_stack([np.ones(n)] + cols)
        else:
            X = np.ones((n, 1))

        p = X.shape[1]
        beta0 = np.zeros(p)

        def neg_log_likelihood(beta: np.ndarray) -> float:
            eta = X @ beta
            # Clip to avoid overflow
            eta = np.clip(eta, -30.0, 30.0)
            mu = np.exp(eta)
            # Poisson: -L = sum(exposure * mu - y * eta)
            nll = float(np.sum(exposure * mu - y * eta))
            return nll

        def gradient(beta: np.ndarray) -> np.ndarray:
            eta = np.clip(X @ beta, -30.0, 30.0)
            mu = np.exp(eta)
            residual = exposure * mu - y  # (n,)
            grad = X.T @ residual  # (p,)
            return grad

        result = minimize(
            neg_log_likelihood,
            beta0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        if not result.success:
            warnings.warn(
                f"Poisson GLM for transition '{transition_name}' did not fully "
                f"converge: {result.message}",
                RuntimeWarning,
                stacklevel=3,
            )

        return _FittedTransitionModel(
            coefficients=result.x,
            feature_cols=feature_cols_present,
            transition_name=transition_name,
            encodings=encodings,
        )

    def predict(
        self,
        transition_name: str,
        df: pl.DataFrame,
    ) -> np.ndarray:
        """
        Predict intensity rates for a specific transition.

        Parameters
        ----------
        transition_name :
            One of the transitions fitted (must match key in training data).
        df :
            DataFrame with required feature columns.

        Returns
        -------
        np.ndarray of shape (n,) with predicted intensities.
        """
        self._check_fitted()
        if transition_name not in self._fitted_models:
            raise KeyError(
                f"No model fitted for transition '{transition_name}'. "
                f"Available: {list(self._fitted_models.keys())}"
            )
        return self._fitted_models[transition_name].predict(df)

    def get_model(self, transition_name: str) -> _FittedTransitionModel:
        """Return the fitted model object for a transition."""
        self._check_fitted()
        if transition_name not in self._fitted_models:
            raise KeyError(
                f"No model for transition '{transition_name}'. "
                f"Available: {list(self._fitted_models.keys())}"
            )
        return self._fitted_models[transition_name]

    @property
    def transitions(self) -> list[str]:
        """List of fitted transition names."""
        return list(self._fitted_models.keys())

    def _check_fitted(self) -> None:
        if not self._fitted_models:
            raise RuntimeError(
                "PoissonTransitionFitter has not been fitted. Call fit() first."
            )


# ---------------------------------------------------------------------------
# KolmogorovPremiumCalculator
# ---------------------------------------------------------------------------


class KolmogorovPremiumCalculator:
    """
    Compute net premiums via the Kolmogorov forward equations.

    Integrates the state occupancy probability vector p(t) forward through
    time using the generator matrix Q(t):

        dp/dt = p(t) Q(t)

    Solution at each time-step uses the matrix exponential:

        p(t + dt) ≈ p(t) expm(Q * dt)

    The net premium is the expected present value of the sum of cash flows
    triggered by each transition, discounted at ``discount_rate``.

    Parameters
    ----------
    states :
        Ordered list of state names. First state is assumed to be the
        initial state (probability 1 at t=0).
    discount_rate :
        Continuous discount rate (force of interest). Default 0.05.
    dt :
        Time step for numerical integration (years). Default 0.1 year.
    max_age :
        Maximum age (or duration) for integration. Default 65.0.
    """

    def __init__(
        self,
        states: list[str],
        discount_rate: float = 0.05,
        dt: float = 0.1,
        max_age: float = 65.0,
    ) -> None:
        if len(states) < 2:
            raise ValueError("At least 2 states are required.")
        if discount_rate < 0:
            raise ValueError("discount_rate must be non-negative.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.states = states
        self.discount_rate = discount_rate
        self.dt = dt
        self.max_age = max_age
        self._n_states = len(states)
        self._state_idx: dict[str, int] = {s: i for i, s in enumerate(states)}

    def compute_premium(
        self,
        intensity_fns: dict[str, Callable[[float], float]],
        cash_flows: dict[str, float],
        entry_age: float = 0.0,
    ) -> float:
        """
        Compute the expected present value of all cash flows.

        Parameters
        ----------
        intensity_fns :
            Dict mapping transition name (e.g. "healthy->sick") to a callable
            that takes age (float) and returns the transition intensity mu(age).
        cash_flows :
            Dict mapping transition name to the lump-sum benefit paid on that
            transition. E.g. {"healthy->sick": 1.0, "sick->dead": 0.0}.
            Ongoing cash flows (e.g. annuity while sick) should be expressed
            as the present value per unit exposure, or use dt-scaled amounts.
        entry_age :
            Age at policy inception.

        Returns
        -------
        float — expected present value of benefits (net premium).
        """
        n_states = self._n_states
        p = np.zeros(n_states)
        p[0] = 1.0  # Start in first state

        t = entry_age
        epv = 0.0
        discount = 1.0
        n_steps = int((self.max_age - entry_age) / self.dt)

        for _ in range(n_steps):
            # Build generator matrix Q at time t
            Q = self._build_generator(intensity_fns, t)

            # Accumulate cash flows: epv += sum_ij p_i * mu_ij * b_ij * dt * discount
            for tr_name, benefit in cash_flows.items():
                if benefit == 0.0:
                    continue
                parts = tr_name.split("->")
                if len(parts) != 2:
                    continue
                s_from, s_to = parts
                i = self._state_idx.get(s_from)
                j = self._state_idx.get(s_to)
                if i is None or j is None:
                    continue
                mu_ij = Q[i, j]
                epv += float(p[i] * mu_ij * benefit * self.dt * discount)

            # Advance occupancy probabilities
            Q_dt = Q * self.dt
            p = p @ expm(Q_dt)
            t += self.dt
            discount *= np.exp(-self.discount_rate * self.dt)

        return epv

    def _build_generator(
        self,
        intensity_fns: dict[str, Callable[[float], float]],
        age: float,
    ) -> np.ndarray:
        """
        Build the instantaneous generator matrix Q at a given age.

        Q[i, j] = mu_{ij}(age) for i != j.
        Q[i, i] = -sum_{j != i} mu_{ij}(age).
        """
        Q = np.zeros((self._n_states, self._n_states))

        for tr_name, fn in intensity_fns.items():
            parts = tr_name.split("->")
            if len(parts) != 2:
                continue
            s_from, s_to = parts
            i = self._state_idx.get(s_from)
            j = self._state_idx.get(s_to)
            if i is None or j is None:
                continue
            mu_val = max(float(fn(age)), 0.0)
            Q[i, j] = mu_val

        # Diagonal: row sums to zero
        for i in range(self._n_states):
            Q[i, i] = -np.sum(Q[i, :])

        return Q


# ---------------------------------------------------------------------------
# MultiStateTransitionFairness
# ---------------------------------------------------------------------------


class MultiStateTransitionFairness:
    """
    Discrimination-free pricing for multi-state insurance models.

    Orchestrates the full pipeline:

    1. Build per-transition DataFrames via TransitionDataBuilder.
    2. Fit Poisson GLMs per transition via PoissonTransitionFitter.
    3. Apply LindholmCorrector per transition to marginalise the protected
       attribute from each intensity.
    4. Compute before/after premiums via KolmogorovPremiumCalculator.
    5. Return MultiStateFairnessReport with per-group premium comparison.

    The key design choice: marginalisation happens at the intensity level,
    not at the premium level. This matters because premium non-linearity
    (via the matrix exponential) means that averaging premiums across groups
    is not the same as marginalising the rates that drive them.

    Parameters
    ----------
    protected_attrs :
        Column names of protected attributes in the observation DataFrame.
    feature_cols :
        Non-protected covariate columns to include as predictors.
    states :
        Ordered list of states in the model. First = initial state.
    cash_flows :
        Dict mapping transition name -> benefit amount for premium calculation.
    discount_rate :
        Force of interest for EPV calculation. Default 0.05.
    dt :
        Integration step for Kolmogorov equations (years). Default 0.1.
    max_age :
        Maximum age for integration. Default 65.0.
    bias_correction :
        LindholmCorrector bias correction method. Default "proportional".
    """

    def __init__(
        self,
        protected_attrs: list[str],
        feature_cols: list[str],
        states: list[str],
        cash_flows: dict[str, float] | None = None,
        discount_rate: float = 0.05,
        dt: float = 0.1,
        max_age: float = 65.0,
        bias_correction: str = "proportional",
    ) -> None:
        self.protected_attrs = protected_attrs
        self.feature_cols = feature_cols
        self.states = states
        self.cash_flows = cash_flows or {}
        self.discount_rate = discount_rate
        self.dt = dt
        self.max_age = max_age
        self.bias_correction = bias_correction

        self._fitter: PoissonTransitionFitter | None = None
        self._correctors: dict[str, LindholmCorrector] = {}
        self._is_fitted = False

    def run(
        self,
        df: pl.DataFrame,
        D: pl.DataFrame,
        state_from_col: str = "state_from",
        state_to_col: str = "state_to",
        age_col: str = "age",
        exposure_col: str = "exposure",
        group_col: str | None = None,
    ) -> MultiStateFairnessReport:
        """
        Run the full fairness audit pipeline.

        Parameters
        ----------
        df :
            Observation DataFrame. Must contain state_from_col, state_to_col,
            age_col, exposure_col, feature_cols, and protected_attrs columns.
        D :
            Protected attribute DataFrame (same row order as df). Must contain
            protected_attrs columns.
        state_from_col, state_to_col, age_col, exposure_col :
            Column name overrides (default values work for standard schemas).
        group_col :
            Column in D (or df) to group premiums by. If None, uses the first
            protected attribute.

        Returns
        -------
        MultiStateFairnessReport
        """
        n = df.shape[0]
        if D.shape[0] != n:
            raise ValueError(
                f"df ({n} rows) and D ({D.shape[0]} rows) must have the same "
                "number of observations."
            )
        for attr in self.protected_attrs:
            if attr not in D.columns and attr not in df.columns:
                raise ValueError(
                    f"Protected attribute '{attr}' not found in D or df."
                )

        # Step 1: build transition data
        covariate_cols = list(
            dict.fromkeys(
                [age_col]
                + self.feature_cols
                + self.protected_attrs
            )
        )
        # Include protected attrs in the training data so Lindholm can marginalise
        builder = TransitionDataBuilder(
            state_from_col=state_from_col,
            state_to_col=state_to_col,
            age_col=age_col,
            exposure_col=exposure_col,
        )
        transition_data = builder.build(df, covariate_cols=covariate_cols)

        # Step 2: fit Poisson GLMs (including protected attrs as features)
        fitter_feature_cols = self.feature_cols + [
            a for a in self.protected_attrs if a not in self.feature_cols
        ]
        fitter = PoissonTransitionFitter(feature_cols=fitter_feature_cols)
        fitter.fit(transition_data)
        self._fitter = fitter

        transitions = fitter.transitions

        # Step 3: fit LindholmCorrector per transition
        correctors: dict[str, LindholmCorrector] = {}
        for tr_name in transitions:
            tr_df = transition_data[tr_name]
            # X: non-protected features; D_tr: protected cols for this transition
            protected_in_tr = [a for a in self.protected_attrs if a in tr_df.columns]
            if not protected_in_tr:
                continue

            model = fitter.get_model(tr_name)
            model_fn = _make_model_fn(model)

            X_tr = tr_df.drop(["event", "exposure"] + protected_in_tr)
            D_tr = tr_df.select(protected_in_tr)
            exposure_arr = tr_df["exposure"].to_numpy()

            corrector = LindholmCorrector(
                protected_attrs=protected_in_tr,
                bias_correction=self.bias_correction,
            )
            corrector.fit(
                model_fn=model_fn,
                X_calib=X_tr,
                D_calib=D_tr,
                exposure=exposure_arr,
            )
            correctors[tr_name] = corrector

        self._correctors = correctors
        self._is_fitted = True

        # Step 4: compute before and after premiums per policy
        # For each row in df, we compute the premium using:
        #   before: raw Poisson model intensities
        #   after:  Lindholm-corrected intensities

        # Determine group labels
        grp_col = group_col or self.protected_attrs[0]
        if grp_col in D.columns:
            group_labels = D[grp_col].to_list()
        elif grp_col in df.columns:
            group_labels = df[grp_col].to_list()
        else:
            group_labels = ["all"] * n

        # Compute per-row premiums before/after
        premiums_before = self._compute_row_premiums(
            df, D, transition_data, fitter, correctors,
            corrected=False,
            age_col=age_col,
        )
        premiums_after = self._compute_row_premiums(
            df, D, transition_data, fitter, correctors,
            corrected=True,
            age_col=age_col,
        )

        # Aggregate by group
        premium_before_by_group: dict[str, list[float]] = {}
        premium_after_by_group: dict[str, list[float]] = {}
        for i, grp in enumerate(group_labels):
            grp_s = str(grp)
            premium_before_by_group.setdefault(grp_s, []).append(premiums_before[i])
            premium_after_by_group.setdefault(grp_s, []).append(premiums_after[i])

        premium_before_mean = {
            g: float(np.mean(v)) for g, v in premium_before_by_group.items()
        }
        premium_after_mean = {
            g: float(np.mean(v)) for g, v in premium_after_by_group.items()
        }

        # Step 5: transition-level correction summary
        transition_corrections = self._compute_transition_corrections(
            df, D, transition_data, fitter, correctors
        )

        return MultiStateFairnessReport(
            transitions=transitions,
            premium_before=premium_before_mean,
            premium_after=premium_after_mean,
            transition_corrections=transition_corrections,
            n_policies=n,
            protected_attrs=self.protected_attrs,
        )

    def _compute_row_premiums(
        self,
        df: pl.DataFrame,
        D: pl.DataFrame,
        transition_data: dict[str, pl.DataFrame],
        fitter: PoissonTransitionFitter,
        correctors: dict[str, LindholmCorrector],
        corrected: bool,
        age_col: str,
    ) -> np.ndarray:
        """
        Compute per-row premiums using portfolio-average intensities.

        We compute one EPV for the portfolio using mean intensities and mean
        entry age. All policies get the same value — the audit purpose is
        the between-group comparison (before vs after marginalisation), not
        per-policy pricing. A production system would want per-policy
        integration via the Kolmogorov ODE.

        The before/after split is meaningful: ``corrected=False`` uses raw
        Poisson predictions (which encode gender effects), ``corrected=True``
        uses LindholmCorrector output (which has marginalised gender out).
        """
        n = df.shape[0]
        calculator = KolmogorovPremiumCalculator(
            states=self.states,
            discount_rate=self.discount_rate,
            dt=self.dt,
            max_age=self.max_age,
        )

        transitions = fitter.transitions
        ages = df[age_col].cast(pl.Float64).to_numpy()
        entry_age = float(np.mean(ages))

        # Build constant intensity functions using portfolio-mean predictions
        intensity_fns: dict[str, Callable[[float], float]] = {}

        for tr_name in transitions:
            if tr_name not in transition_data:
                continue
            tr_df = transition_data[tr_name]
            protected_in_tr = [a for a in self.protected_attrs if a in tr_df.columns]
            model = fitter.get_model(tr_name)
            model_fn_raw = _make_model_fn(model)

            if corrected and tr_name in correctors and protected_in_tr:
                corrector = correctors[tr_name]
                X_tr = tr_df.drop(["event", "exposure"] + protected_in_tr)
                D_tr = tr_df.select(protected_in_tr)
                mu_arr = corrector.transform(
                    model_fn=model_fn_raw,
                    X=X_tr,
                    D=D_tr,
                )
                mean_mu = float(np.mean(mu_arr))
            else:
                mean_mu = float(np.mean(model_fn_raw(tr_df)))

            _mu = mean_mu  # capture for closure

            def _make_const_fn(val: float) -> Callable[[float], float]:
                def fn(_age: float) -> float:
                    return val
                return fn

            intensity_fns[tr_name] = _make_const_fn(_mu)

        if not self.cash_flows:
            cash_flows = {tr: 1.0 for tr in transitions}
        else:
            cash_flows = self.cash_flows

        epv = calculator.compute_premium(
            intensity_fns=intensity_fns,
            cash_flows=cash_flows,
            entry_age=entry_age,
        )

        return np.full(n, epv)

    def _compute_transition_corrections(
        self,
        df: pl.DataFrame,
        D: pl.DataFrame,
        transition_data: dict[str, pl.DataFrame],
        fitter: PoissonTransitionFitter,
        correctors: dict[str, LindholmCorrector],
    ) -> dict[str, float]:
        """
        Compute fractional change in mean intensity per transition.

        Returns dict mapping transition name -> (after - before) / before.
        """
        corrections: dict[str, float] = {}

        for tr_name in fitter.transitions:
            if tr_name not in transition_data:
                corrections[tr_name] = 0.0
                continue

            tr_df = transition_data[tr_name]
            protected_in_tr = [a for a in self.protected_attrs if a in tr_df.columns]
            model = fitter.get_model(tr_name)
            model_fn = _make_model_fn(model)

            mu_before = float(np.mean(model_fn(tr_df)))

            if tr_name in correctors and protected_in_tr:
                corrector = correctors[tr_name]
                X_tr = tr_df.drop(["event", "exposure"] + protected_in_tr)
                D_tr = tr_df.select(protected_in_tr)
                mu_after_arr = corrector.transform(
                    model_fn=model_fn,
                    X=X_tr,
                    D=D_tr,
                )
                mu_after = float(np.mean(mu_after_arr))
                corr = (mu_after - mu_before) / mu_before if mu_before > 0 else 0.0
            else:
                corr = 0.0

            corrections[tr_name] = corr

        return corrections


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_model_fn(model: _FittedTransitionModel) -> Callable[[pl.DataFrame], np.ndarray]:
    """
    Wrap a _FittedTransitionModel as a model_fn compatible with LindholmCorrector.

    LindholmCorrector.fit() takes model_fn: Callable[[pl.DataFrame], np.ndarray].
    The DataFrame passed to model_fn includes both X and D columns.
    """

    def model_fn(df: pl.DataFrame) -> np.ndarray:
        return model.predict(df)

    return model_fn
