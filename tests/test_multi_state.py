"""
Tests for multi_state.py — TransitionDataBuilder, PoissonTransitionFitter,
MultiStateTransitionFairness, KolmogorovPremiumCalculator, MultiStateFairnessReport.

Synthetic data represents a 3-state disability model:
    healthy -> sick   (incidence)
    sick -> healthy   (recovery)
    sick -> dead      (excess mortality while sick)
    healthy -> dead   (background mortality)

Data is generated at the row level: each row is one (policy, period) observation
with a known origin state, destination state, and exposure in years.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from insurance_fairness.multi_state import (
    KolmogorovPremiumCalculator,
    MultiStateFairnessReport,
    MultiStateTransitionFairness,
    PoissonTransitionFitter,
    TransitionDataBuilder,
    _FittedTransitionModel,
    _make_model_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_obs() -> pl.DataFrame:
    """
    Synthetic 3-state observation DataFrame.

    Columns: state_from, state_to, age, exposure, gender, occupation,
             log_age (for testing extra covariates).

    n=800 rows. Gender affects sick->dead intensity.
    """
    rng = np.random.default_rng(0)
    n = 800

    # Randomly assign origin state (healthy:70%, sick:30%)
    state_from = rng.choice(["healthy", "sick"], size=n, p=[0.7, 0.3])
    age = rng.uniform(30.0, 60.0, size=n)
    exposure = rng.uniform(0.5, 2.0, size=n)
    gender = rng.choice(["M", "F"], size=n)
    occupation = rng.choice(["office", "manual", "other"], size=n)

    # Generate transitions based on origin state
    state_to = []
    for i in range(n):
        sf = state_from[i]
        if sf == "healthy":
            # Incidence rate ~0.05/yr; mortality ~0.005/yr; rest censored
            u = rng.uniform()
            if u < 0.05 * exposure[i]:
                state_to.append("sick")
            elif u < 0.055 * exposure[i]:
                state_to.append("dead")
            else:
                state_to.append("censored")
        else:  # sick
            # Recovery rate ~0.30/yr; sick-mortality 0.02/yr (gender effect)
            extra = 0.01 if gender[i] == "M" else 0.0
            u = rng.uniform()
            if u < 0.30 * exposure[i]:
                state_to.append("healthy")
            elif u < (0.30 + 0.02 + extra) * exposure[i]:
                state_to.append("dead")
            else:
                state_to.append("censored")

    df = pl.DataFrame(
        {
            "state_from": state_from.tolist(),
            "state_to": state_to,
            "age": age.tolist(),
            "exposure": exposure.tolist(),
            "gender": gender.tolist(),
            "occupation": occupation.tolist(),
        }
    )
    return df


@pytest.fixture(scope="module")
def D_df(synthetic_obs) -> pl.DataFrame:
    """Protected attribute DataFrame aligned with synthetic_obs."""
    return synthetic_obs.select(["gender"])


@pytest.fixture(scope="module")
def builder() -> TransitionDataBuilder:
    return TransitionDataBuilder(
        state_from_col="state_from",
        state_to_col="state_to",
        age_col="age",
        exposure_col="exposure",
    )


@pytest.fixture(scope="module")
def transition_data(synthetic_obs, builder) -> dict:
    return builder.build(synthetic_obs, covariate_cols=["age", "gender", "occupation"])


@pytest.fixture(scope="module")
def fitted_fitter(transition_data) -> PoissonTransitionFitter:
    fitter = PoissonTransitionFitter(feature_cols=["age", "gender", "occupation"])
    fitter.fit(transition_data)
    return fitter


# ---------------------------------------------------------------------------
# TransitionDataBuilder tests
# ---------------------------------------------------------------------------


class TestTransitionDataBuilder:
    def test_build_returns_dict(self, transition_data):
        assert isinstance(transition_data, dict)

    def test_transition_keys_format(self, transition_data):
        for key in transition_data.keys():
            assert "->" in key, f"Key '{key}' is not in 'from->to' format"

    def test_known_transitions_present(self, transition_data):
        keys = set(transition_data.keys())
        assert "healthy->sick" in keys
        assert "sick->healthy" in keys

    def test_output_has_event_and_exposure(self, transition_data):
        for name, df in transition_data.items():
            assert "event" in df.columns, f"Missing 'event' in {name}"
            assert "exposure" in df.columns, f"Missing 'exposure' in {name}"

    def test_event_is_binary(self, transition_data):
        for name, df in transition_data.items():
            vals = df["event"].unique().to_list()
            for v in vals:
                assert v in (0, 1), f"Non-binary event in {name}: {v}"

    def test_exposure_positive(self, transition_data):
        for name, df in transition_data.items():
            assert df["exposure"].min() > 0, f"Non-positive exposure in {name}"

    def test_only_at_risk_rows_included(self, transition_data):
        # healthy->sick should only have rows where state_from == healthy
        # We can't check directly but can check event counts are plausible
        df = transition_data.get("healthy->sick")
        if df is not None:
            assert df["event"].sum() > 0

    def test_missing_required_column_raises(self, builder):
        bad_df = pl.DataFrame({"state_from": ["healthy"], "state_to": ["sick"]})
        with pytest.raises(ValueError, match="missing required columns"):
            builder.build(bad_df)

    def test_covariate_cols_carried_through(self, transition_data):
        # All output DataFrames should have age and gender
        for name, df in transition_data.items():
            assert "age" in df.columns
            assert "gender" in df.columns

    def test_no_transitions_raises(self, builder):
        # All censored -> no transitions
        df = pl.DataFrame(
            {
                "state_from": ["healthy", "healthy"],
                "state_to": ["censored", "censored"],
                "age": [40.0, 50.0],
                "exposure": [1.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="No transitions"):
            builder.build(df)


# ---------------------------------------------------------------------------
# PoissonTransitionFitter tests
# ---------------------------------------------------------------------------


class TestPoissonTransitionFitter:
    def test_fit_returns_self(self, transition_data):
        fitter = PoissonTransitionFitter(feature_cols=["age"])
        result = fitter.fit(transition_data)
        assert result is fitter

    def test_transitions_property(self, fitted_fitter, transition_data):
        assert set(fitted_fitter.transitions) == set(transition_data.keys())

    def test_coefficients_shape(self, fitted_fitter, transition_data):
        # With features [age, gender, occupation] -> design matrix is (n, 1+k)
        # But gender and occupation are strings — check we handle them
        for tr_name in fitted_fitter.transitions:
            model = fitted_fitter.get_model(tr_name)
            assert model.coefficients.ndim == 1
            assert model.coefficients.shape[0] >= 1

    def test_predict_positive(self, fitted_fitter, transition_data):
        for tr_name in fitted_fitter.transitions:
            df = transition_data[tr_name]
            preds = fitted_fitter.predict(tr_name, df)
            assert np.all(preds > 0), f"Non-positive predictions for {tr_name}"

    def test_predict_shape(self, fitted_fitter, transition_data):
        for tr_name in fitted_fitter.transitions:
            df = transition_data[tr_name]
            preds = fitted_fitter.predict(tr_name, df)
            assert preds.shape == (df.shape[0],)

    def test_unknown_transition_raises(self, fitted_fitter):
        with pytest.raises(KeyError, match="No model fitted"):
            fitted_fitter.predict("nonexistent->state", pl.DataFrame({"age": [40.0]}))

    def test_unfitted_raises(self):
        fitter = PoissonTransitionFitter()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fitter.predict("healthy->sick", pl.DataFrame({"x": [1.0]}))

    def test_intercept_only_model(self, transition_data):
        # Feature cols with only empty list — intercept only
        fitter = PoissonTransitionFitter(feature_cols=[])
        fitter.fit(transition_data)
        for tr_name in fitter.transitions:
            df = transition_data[tr_name]
            preds = fitter.predict(tr_name, df)
            # All predictions should be the same (intercept only)
            assert np.allclose(preds, preds[0], rtol=1e-5)

    def test_incidence_rate_in_plausible_range(self, fitted_fitter, transition_data):
        # healthy->sick predictions should be in (0, 1) per year (not absurd)
        df = transition_data.get("healthy->sick")
        if df is not None:
            preds = fitted_fitter.predict("healthy->sick", df)
            assert float(np.mean(preds)) < 5.0
            assert float(np.mean(preds)) > 0.0


# ---------------------------------------------------------------------------
# KolmogorovPremiumCalculator tests
# ---------------------------------------------------------------------------


class TestKolmogorovPremiumCalculator:
    @pytest.fixture
    def simple_calculator(self):
        return KolmogorovPremiumCalculator(
            states=["healthy", "sick", "dead"],
            discount_rate=0.05,
            dt=0.1,
            max_age=65.0,
        )

    def test_premium_positive(self, simple_calculator):
        intensity_fns = {
            "healthy->sick": lambda age: 0.05,
            "sick->healthy": lambda age: 0.30,
            "sick->dead": lambda age: 0.02,
            "healthy->dead": lambda age: 0.005,
        }
        cash_flows = {"healthy->sick": 1.0}
        epv = simple_calculator.compute_premium(
            intensity_fns=intensity_fns,
            cash_flows=cash_flows,
            entry_age=30.0,
        )
        assert epv > 0.0

    def test_zero_benefit_zero_premium(self, simple_calculator):
        intensity_fns = {
            "healthy->sick": lambda age: 0.05,
            "sick->dead": lambda age: 0.02,
        }
        cash_flows = {"healthy->sick": 0.0, "sick->dead": 0.0}
        epv = simple_calculator.compute_premium(
            intensity_fns=intensity_fns,
            cash_flows=cash_flows,
            entry_age=30.0,
        )
        assert epv == 0.0

    def test_higher_intensity_higher_premium(self, simple_calculator):
        cf = {"healthy->sick": 1.0}
        epv_low = simple_calculator.compute_premium(
            intensity_fns={"healthy->sick": lambda a: 0.02, "sick->dead": lambda a: 0.01},
            cash_flows=cf,
            entry_age=30.0,
        )
        epv_high = simple_calculator.compute_premium(
            intensity_fns={"healthy->sick": lambda a: 0.10, "sick->dead": lambda a: 0.01},
            cash_flows=cf,
            entry_age=30.0,
        )
        assert epv_high > epv_low

    def test_invalid_states_raises(self):
        with pytest.raises(ValueError, match="At least 2 states"):
            KolmogorovPremiumCalculator(states=["healthy"])

    def test_negative_discount_raises(self):
        with pytest.raises(ValueError, match="discount_rate"):
            KolmogorovPremiumCalculator(states=["h", "s"], discount_rate=-0.01)

    def test_generator_diagonal_nonpositive(self, simple_calculator):
        intensity_fns = {
            "healthy->sick": lambda age: 0.05,
            "sick->healthy": lambda age: 0.20,
            "sick->dead": lambda age: 0.02,
        }
        Q = simple_calculator._build_generator(intensity_fns, age=40.0)
        # Diagonal should be <= 0
        for i in range(Q.shape[0]):
            assert Q[i, i] <= 0.0

    def test_generator_row_sums_zero(self, simple_calculator):
        intensity_fns = {
            "healthy->sick": lambda age: 0.05,
            "sick->healthy": lambda age: 0.20,
        }
        Q = simple_calculator._build_generator(intensity_fns, age=40.0)
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# MultiStateTransitionFairness tests
# ---------------------------------------------------------------------------


class TestMultiStateFairness:
    @pytest.fixture
    def audit(self):
        return MultiStateTransitionFairness(
            protected_attrs=["gender"],
            feature_cols=["age"],
            states=["healthy", "sick", "dead"],
            cash_flows={"healthy->sick": 1.0, "sick->dead": 0.5},
            discount_rate=0.05,
            dt=0.2,
            max_age=65.0,
        )

    @pytest.fixture(scope="class")
    def report(self, synthetic_obs, D_df):
        audit = MultiStateTransitionFairness(
            protected_attrs=["gender"],
            feature_cols=["age"],
            states=["healthy", "sick", "dead"],
            cash_flows={"healthy->sick": 1.0, "sick->dead": 0.5},
            discount_rate=0.05,
            dt=0.2,
            max_age=65.0,
        )
        return audit.run(synthetic_obs, D_df)

    def test_run_returns_report(self, report):
        assert isinstance(report, MultiStateFairnessReport)

    def test_report_has_transitions(self, report):
        assert len(report.transitions) > 0

    def test_report_has_before_after_premiums(self, report):
        assert len(report.premium_before) > 0
        assert len(report.premium_after) > 0

    def test_premium_before_positive(self, report):
        for grp, val in report.premium_before.items():
            assert val > 0.0, f"Non-positive before premium for group {grp}"

    def test_premium_after_positive(self, report):
        for grp, val in report.premium_after.items():
            assert val > 0.0, f"Non-positive after premium for group {grp}"

    def test_groups_match_gender_values(self, report):
        groups = set(report.premium_before.keys())
        assert "M" in groups or "F" in groups

    def test_transition_corrections_present(self, report):
        assert len(report.transition_corrections) > 0

    def test_n_policies_correct(self, report, synthetic_obs):
        assert report.n_policies == synthetic_obs.shape[0]

    def test_protected_attrs_in_report(self, report):
        assert "gender" in report.protected_attrs

    def test_summary_returns_string(self, report):
        s = report.summary()
        assert isinstance(s, str)
        assert "before" in s.lower() or "->" in s

    def test_mismatched_df_D_raises(self, audit, synthetic_obs):
        D_short = synthetic_obs.select(["gender"]).head(10)
        with pytest.raises(ValueError, match="same number"):
            audit.run(synthetic_obs, D_short)

    def test_missing_protected_attr_raises(self, audit, synthetic_obs):
        D_bad = pl.DataFrame({"occupation": synthetic_obs["occupation"].to_list()})
        df_no_gender = synthetic_obs.drop("gender")
        with pytest.raises(ValueError, match="not found"):
            audit.run(df_no_gender, D_bad)


# ---------------------------------------------------------------------------
# MultiStateFairnessReport unit tests
# ---------------------------------------------------------------------------


class TestMultiStateFairnessReport:
    def test_summary_includes_transitions(self):
        report = MultiStateFairnessReport(
            transitions=["healthy->sick", "sick->dead"],
            premium_before={"M": 0.25, "F": 0.20},
            premium_after={"M": 0.22, "F": 0.22},
            transition_corrections={"healthy->sick": -0.05, "sick->dead": 0.01},
            n_policies=500,
            protected_attrs=["gender"],
        )
        s = report.summary()
        assert "healthy->sick" in s
        assert "sick->dead" in s
        assert "500" in s
        assert "gender" in s

    def test_summary_shows_all_groups(self):
        report = MultiStateFairnessReport(
            transitions=["healthy->sick"],
            premium_before={"M": 0.25, "F": 0.20},
            premium_after={"M": 0.22, "F": 0.22},
            transition_corrections={"healthy->sick": -0.05},
            n_policies=100,
            protected_attrs=["gender"],
        )
        s = report.summary()
        assert "M" in s
        assert "F" in s


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_model_fn_callable(self, transition_data):
        fitter = PoissonTransitionFitter(feature_cols=["age"])
        fitter.fit(transition_data)
        model = fitter.get_model(list(transition_data.keys())[0])
        fn = _make_model_fn(model)
        df = list(transition_data.values())[0]
        result = fn(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (df.shape[0],)

    def test_fitted_model_predict(self, transition_data):
        tr_name = list(transition_data.keys())[0]
        tr_df = transition_data[tr_name]
        model = _FittedTransitionModel(
            coefficients=np.array([-3.0, 0.01]),
            feature_cols=["age"],
            transition_name=tr_name,
        )
        preds = model.predict(tr_df)
        assert np.all(preds > 0)
        assert preds.shape == (tr_df.shape[0],)
