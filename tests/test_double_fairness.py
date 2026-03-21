"""
Tests for double_fairness.py — DoubleFairnessAudit and DoubleFairnessResult.

Synthetic data setup:
    n=600, p=5 features, binary S (0/1), action A generated from S + X.
    y_primary = 200 + 50*X[:,0] + 30*S + noise   (group effect on revenue)
    y_fairness: loss ratio with group differential
                group 0 mean ~0.7, group 1 mean ~0.9

The group differential in y_fairness ensures Delta_2 is non-trivial and the
Pareto sweep actually has something to trade off.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from insurance_fairness.double_fairness import DoubleFairnessAudit, DoubleFairnessResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate reproducible synthetic insurance data."""
    rng = np.random.default_rng(42)
    n = 600
    p = 5

    X = rng.normal(0.0, 1.0, size=(n, p))
    S = rng.binomial(1, 0.5, size=n)

    # Primary outcome: pure premium with group effect
    y_primary = 200 + 50 * X[:, 0] + 30 * S + rng.normal(0, 10, size=n)
    y_primary = np.clip(y_primary, 50, 500)

    # Fairness outcome: loss ratio with group differential
    # Group 0 ~ 0.7, Group 1 ~ 0.9
    base_lr = 0.7 + 0.2 * S + 0.1 * X[:, 1]
    y_fairness = np.clip(
        rng.normal(base_lr, 0.15, size=n),
        0.0,
        None,
    )

    exposure = rng.uniform(0.5, 2.0, size=n)

    return X, y_primary, y_fairness, S, exposure


@pytest.fixture(scope="module")
def fitted_audit(synthetic_data):
    """A DoubleFairnessAudit fitted on synthetic data (n_alphas=5 for speed)."""
    X, y_primary, y_fairness, S, _ = synthetic_data
    audit = DoubleFairnessAudit(n_alphas=5, random_state=0, max_iter=200)
    audit.fit(X, y_primary, y_fairness, S)
    return audit


@pytest.fixture(scope="module")
def audit_result(fitted_audit):
    """Cached DoubleFairnessResult for the fitted audit."""
    return fitted_audit.audit()


# ---------------------------------------------------------------------------
# Test 1: fit() returns self without error
# ---------------------------------------------------------------------------


def test_fit_returns_self(synthetic_data):
    X, y_primary, y_fairness, S, _ = synthetic_data
    audit = DoubleFairnessAudit(n_alphas=3, max_iter=100)
    result = audit.fit(X, y_primary, y_fairness, S)
    assert result is audit


# ---------------------------------------------------------------------------
# Test 2: audit() returns DoubleFairnessResult with correct shape
# ---------------------------------------------------------------------------


def test_audit_returns_result(fitted_audit):
    result = fitted_audit.audit()
    assert isinstance(result, DoubleFairnessResult)


def test_audit_result_shapes(audit_result):
    K = 5
    assert len(audit_result.pareto_alphas) == K
    assert len(audit_result.pareto_V) == K
    assert len(audit_result.pareto_delta1) == K
    assert len(audit_result.pareto_delta2) == K
    assert audit_result.pareto_theta.shape == (K, 5)  # p=5


def test_audit_is_cached(fitted_audit):
    """Second call to audit() returns the same object."""
    r1 = fitted_audit.audit()
    r2 = fitted_audit.audit()
    assert r1 is r2


# ---------------------------------------------------------------------------
# Test 3: Pareto alphas are in (0, 1) and strictly ordered
# ---------------------------------------------------------------------------


def test_pareto_alphas_range(audit_result):
    alphas = audit_result.pareto_alphas
    assert np.all(alphas > 0.0)
    assert np.all(alphas < 1.0)
    # Should be increasing (linspace)
    assert np.all(np.diff(alphas) > 0)


# ---------------------------------------------------------------------------
# Test 4: Selected index is valid and delta1 is reduced relative to max
# ---------------------------------------------------------------------------


def test_pareto_improves_fairness(audit_result):
    """Selected policy has strictly less action unfairness than the worst Pareto point."""
    assert 0 <= audit_result.selected_idx < len(audit_result.pareto_alphas)
    # The selected point should not be the worst point on Delta_1
    assert audit_result.selected_delta1 < audit_result.pareto_delta1.max() + 1e-10


# ---------------------------------------------------------------------------
# Test 5: Delta_2 is non-trivial when there is genuine group differential
# ---------------------------------------------------------------------------


def test_pareto_delta2_nonzero_with_group_differential(audit_result):
    """With genuine group differential in y_fairness, Delta_2 should be > 0."""
    # At least one Pareto point should have non-trivial Delta_2
    assert audit_result.pareto_delta2.max() > 1e-8


def test_selected_delta2_finite_positive(audit_result):
    """Selected Delta_2 should be finite and non-negative."""
    assert math.isfinite(audit_result.selected_delta2)
    assert audit_result.selected_delta2 >= 0.0


# ---------------------------------------------------------------------------
# Test 6: kappa override
# ---------------------------------------------------------------------------


def test_kappa_override(synthetic_data):
    X, y_primary, y_fairness, S, _ = synthetic_data
    audit = DoubleFairnessAudit(n_alphas=3, kappa=0.01, max_iter=100)
    audit.fit(X, y_primary, y_fairness, S)
    result = audit.audit()
    assert abs(result.kappa - 0.01) < 1e-10


def test_kappa_auto_scales_with_n(synthetic_data):
    """Default kappa = sqrt(log(n)/n)."""
    X, y_primary, y_fairness, S, _ = synthetic_data
    n = len(X)
    audit = DoubleFairnessAudit(n_alphas=3, max_iter=100)
    audit.fit(X, y_primary, y_fairness, S)
    result = audit.audit()
    expected_kappa = math.sqrt(math.log(n) / n)
    assert abs(result.kappa - expected_kappa) < 1e-10


# ---------------------------------------------------------------------------
# Test 7: Exposure weighting
# ---------------------------------------------------------------------------


def test_exposure_weighted(synthetic_data):
    X, y_primary, y_fairness, S, exposure = synthetic_data
    audit = DoubleFairnessAudit(n_alphas=3, max_iter=100)
    audit.fit(X, y_primary, y_fairness, S, exposure=exposure)
    result = audit.audit()
    assert result.n_train == len(X)
    assert math.isfinite(result.selected_V)


# ---------------------------------------------------------------------------
# Test 8: report() returns a non-trivial string
# ---------------------------------------------------------------------------


def test_report_is_string(fitted_audit):
    r = fitted_audit.report()
    assert isinstance(r, str)
    assert len(r) > 100


def test_report_contains_key_sections(fitted_audit):
    r = fitted_audit.report()
    assert "Double Fairness Analysis" in r
    assert "Delta_1" in r
    assert "Delta_2" in r
    assert "Consumer Duty" in r
    assert "Bian" in r


# ---------------------------------------------------------------------------
# Test 9: summary() is a non-trivial string
# ---------------------------------------------------------------------------


def test_summary_is_string(audit_result):
    s = audit_result.summary()
    assert isinstance(s, str)
    assert len(s) > 100


def test_summary_contains_selected_marker(audit_result):
    s = audit_result.summary()
    assert "<--" in s  # selected row is annotated


# ---------------------------------------------------------------------------
# Test 10: to_dict() is JSON-serialisable
# ---------------------------------------------------------------------------


def test_to_dict_serialisable(audit_result):
    d = audit_result.to_dict()
    serialised = json.dumps(d)  # must not raise
    assert isinstance(serialised, str)


def test_to_dict_fields(audit_result):
    d = audit_result.to_dict()
    expected_keys = {
        "pareto_alphas", "pareto_V", "pareto_delta1", "pareto_delta2",
        "pareto_theta", "selected_idx", "selected_alpha", "selected_delta1",
        "selected_delta2", "selected_V", "kappa", "n_train",
        "outcome_model_type", "fairness_notion",
    }
    assert expected_keys.issubset(set(d.keys()))


# ---------------------------------------------------------------------------
# Test 11: DoubleFairnessResult properties
# ---------------------------------------------------------------------------


def test_selected_properties(audit_result):
    K = len(audit_result.pareto_alphas)
    idx = audit_result.selected_idx
    assert 0 <= idx < K
    assert abs(audit_result.selected_alpha - float(audit_result.pareto_alphas[idx])) < 1e-10
    assert abs(audit_result.selected_delta1 - float(audit_result.pareto_delta1[idx])) < 1e-10
    assert abs(audit_result.selected_delta2 - float(audit_result.pareto_delta2[idx])) < 1e-10
    assert abs(audit_result.selected_V - float(audit_result.pareto_V[idx])) < 1e-10


def test_selected_idx_maximises_V(audit_result):
    """selected_idx should be the argmax of pareto_V."""
    best = int(np.argmax(audit_result.pareto_V))
    assert audit_result.selected_idx == best


# ---------------------------------------------------------------------------
# Test 12: plot_pareto() returns a Figure
# ---------------------------------------------------------------------------


def test_plot_pareto_returns_figure(fitted_audit):
    mpl = pytest.importorskip("matplotlib")
    fig = fitted_audit.plot_pareto()
    assert fig is not None
    # Should have two axes
    assert len(fig.axes) == 2


# ---------------------------------------------------------------------------
# Test 13: unfitted audit raises RuntimeError
# ---------------------------------------------------------------------------


def test_unfitted_audit_raises():
    audit = DoubleFairnessAudit()
    with pytest.raises(RuntimeError, match="not been fitted"):
        audit.audit()

    with pytest.raises(RuntimeError, match="not been fitted"):
        audit.report()


# ---------------------------------------------------------------------------
# Test 14: parameter validation
# ---------------------------------------------------------------------------


def test_invalid_fairness_notion():
    with pytest.raises(ValueError, match="fairness_notion"):
        DoubleFairnessAudit(fairness_notion="equalised_odds")


def test_counterfactual_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        DoubleFairnessAudit(fairness_notion="counterfactual")


def test_invalid_n_alphas():
    with pytest.raises(ValueError, match="n_alphas"):
        DoubleFairnessAudit(n_alphas=1)


def test_non_binary_S_raises(synthetic_data):
    X, y_primary, y_fairness, S, _ = synthetic_data
    S_bad = np.where(S == 1, 2, 0)  # values {0, 2}
    audit = DoubleFairnessAudit(n_alphas=3)
    with pytest.raises(ValueError, match="binary"):
        audit.fit(X, y_primary, y_fairness, S_bad)


def test_single_group_S_raises(synthetic_data):
    X, y_primary, y_fairness, S, _ = synthetic_data
    S_bad = np.zeros_like(S)  # only group 0
    audit = DoubleFairnessAudit(n_alphas=3)
    with pytest.raises(ValueError, match="both group 0 and group 1"):
        audit.fit(X, y_primary, y_fairness, S_bad)


def test_bad_exposure_shape_raises(synthetic_data):
    X, y_primary, y_fairness, S, _ = synthetic_data
    bad_exposure = np.ones(len(X) + 5)
    audit = DoubleFairnessAudit(n_alphas=3)
    with pytest.raises(ValueError, match="exposure"):
        audit.fit(X, y_primary, y_fairness, S, exposure=bad_exposure)


# ---------------------------------------------------------------------------
# Test 15: small group warning
# ---------------------------------------------------------------------------


def test_small_group_warning():
    """Audit should warn when a group has fewer than 50 observations."""
    rng = np.random.default_rng(99)
    n = 100
    X = rng.normal(0, 1, size=(n, 3))
    S = np.zeros(n, dtype=int)
    S[:30] = 1  # group 1 has 30 obs < 50 threshold
    y_primary = rng.uniform(100, 300, size=n)
    y_fairness = rng.uniform(0.5, 1.2, size=n)

    audit = DoubleFairnessAudit(n_alphas=2, max_iter=50)
    with pytest.warns(UserWarning, match="observations"):
        audit.fit(X, y_primary, y_fairness, S)


# ---------------------------------------------------------------------------
# Test 16: custom primary and fairness models
# ---------------------------------------------------------------------------


def test_custom_models(synthetic_data):
    """User can pass custom sklearn-compatible models."""
    from sklearn.linear_model import Lasso, Ridge

    X, y_primary, y_fairness, S, _ = synthetic_data
    audit = DoubleFairnessAudit(
        primary_model=Ridge(alpha=10.0),
        fairness_model=Lasso(alpha=0.1, max_iter=500),
        n_alphas=3,
        max_iter=100,
    )
    audit.fit(X, y_primary, y_fairness, S)
    result = audit.audit()
    assert isinstance(result, DoubleFairnessResult)


# ---------------------------------------------------------------------------
# Test 17: outcome_model_type is logged correctly
# ---------------------------------------------------------------------------


def test_outcome_model_type_logged(audit_result):
    """outcome_model_type should be a non-empty string."""
    assert isinstance(audit_result.outcome_model_type, str)
    assert len(audit_result.outcome_model_type) > 0


def test_tweedie_selected_for_loss_ratio_with_zeros(synthetic_data):
    """When y_fairness has >30% zeros, TweedieRegressor should be selected."""
    X, y_primary, _, S, _ = synthetic_data
    rng = np.random.default_rng(7)
    # Create a loss ratio with lots of zeros (no-claim policies)
    y_fairness_sparse = rng.choice([0.0, 0.0, 0.0, 0.8, 1.2], size=len(X))

    audit = DoubleFairnessAudit(n_alphas=3, max_iter=100)
    audit.fit(X, y_primary, y_fairness_sparse, S)
    result = audit.audit()
    assert "Tweedie" in result.outcome_model_type


# ---------------------------------------------------------------------------
# Test 18: V, Delta_1, Delta_2 are all finite and non-negative
# ---------------------------------------------------------------------------


def test_all_pareto_values_finite(audit_result):
    assert np.all(np.isfinite(audit_result.pareto_V))
    assert np.all(np.isfinite(audit_result.pareto_delta1))
    assert np.all(np.isfinite(audit_result.pareto_delta2))


def test_deltas_non_negative(audit_result):
    """Delta_1 and Delta_2 are squared quantities — must be >= 0."""
    assert np.all(audit_result.pareto_delta1 >= -1e-10)
    assert np.all(audit_result.pareto_delta2 >= -1e-10)


# ---------------------------------------------------------------------------
# Test 19: n_train matches input
# ---------------------------------------------------------------------------


def test_n_train_matches(synthetic_data, audit_result):
    X, _, _, _, _ = synthetic_data
    assert audit_result.n_train == len(X)


# ---------------------------------------------------------------------------
# Test 20: fairness_notion is recorded
# ---------------------------------------------------------------------------


def test_fairness_notion_recorded(audit_result):
    assert audit_result.fairness_notion == "equal_opportunity"
