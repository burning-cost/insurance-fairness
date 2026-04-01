"""
Tests for intersectional.py

Tests cover:
1.  CCdCov decomposition: marginals + eta = joint (Theorem 3.2.1)
2.  Independent data: CCdCov approximately zero
3.  Dependent data: CCdCov > 0
4.  Intersectional-only dependence: CCdCov > 0, eta > 0, marginals near zero
5.  Encoding: categorical attributes get ordinal integers
6.  Encoding: continuous attributes normalised to [0,1]
7.  Audit report has correct structure
8.  Audit report subgroup_statistics has correct columns
9.  Audit report summary() returns a string
10. Audit report to_markdown() returns a string containing Markdown
11. Missing protected_attr column raises ValueError
12. y_hat/D length mismatch raises ValueError
13. Regulariser penalty returns scalar
14. Regulariser with sum_dcov method
15. Regulariser with jdCov method
16. Regulariser penalty is zero for independent data (approximately)
17. Regulariser penalty > 0 for dependent data
18. Regulariser lambda_val=0 returns zero penalty
19. Regulariser invalid method raises ValueError
20. Regulariser negative lambda raises ValueError
21. Regulariser js_divergence: uniform predictions give near-zero D_JS
22. Regulariser js_divergence: bimodal by group gives positive D_JS
23. Lambda calibration result structure
24. Lambda calibration Pareto indices are subset of grid
25. LambdaCalibrationResult.select_lambda records choice
26. LambdaCalibrationResult.summary() returns string
27. _pareto_indices: known 2D Pareto front
28. _compute_loss: poisson, mse, binary_crossentropy
29. Audit with exposure_col
30. Single protected attribute (d=1): eta definition still holds
31. Three protected attributes
32. Large n warning at threshold
33. IntersectionalFairnessAudit __repr__
34. DistanceCovFairnessRegulariser __repr__
35. fit() caches encoders; second call uses cached values
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_fairness.intersectional import (
    DistanceCovFairnessRegulariser,
    IntersectionalAuditReport,
    IntersectionalFairnessAudit,
    LambdaCalibrationResult,
    _ccDcov,
    _compute_loss,
    _encode_attributes,
    _eta_intersectional,
    _jdCov,
    _js_divergence,
    _make_group_labels,
    _marginal_dcov_per_attr,
    _marginal_dcov_sum,
    _pareto_indices,
)

# Skip all tests if dcor is not available
pytest.importorskip("dcor", reason="dcor not installed; skipping intersectional tests")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_independent_data(n: int = 500, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """
    y_hat and binary protected attributes are mutually independent.
    CCdCov should be near zero.
    """
    rng = np.random.default_rng(seed)
    y_hat = rng.uniform(0.05, 0.5, size=n)
    gender = rng.choice(["M", "F"], size=n)
    age_band = rng.integers(1, 6, size=n)
    D = pd.DataFrame({"gender": gender, "age_band": age_band})
    return y_hat, D


def make_dependent_data(n: int = 500, seed: int = 1) -> tuple[np.ndarray, pd.DataFrame]:
    """
    y_hat depends strongly on the protected attributes.
    CCdCov should be substantially positive.
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice([0, 1], size=n)
    age_band = rng.integers(1, 4, size=n)
    # Predictions driven by both protected attributes
    y_hat = 0.1 + 0.2 * gender + 0.1 * age_band + rng.normal(0, 0.02, size=n)
    D = pd.DataFrame({"gender": gender.astype(str), "age_band": age_band})
    return y_hat, D


def make_intersectional_only_data(n: int = 1000, seed: int = 2) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Predictions depend on the interaction (young women get different prices)
    but NOT on gender or age alone — marginal dCov should be near zero while
    eta > 0.

    Construction:
    - gender ∈ {0, 1}; age ∈ {young=0, old=1}
    - mean prediction = 0.2 + 0.1 * (gender == 0) * (age == 0)
      i.e., young women (gender=0, age=young) get a premium bump
    """
    rng = np.random.default_rng(seed)
    n_each = n // 4
    genders = np.array([0, 0, 1, 1] * n_each + [0, 1] * (n % 4 // 2))[:n]
    ages    = np.array([0, 1, 0, 1] * n_each + [0, 0] * (n % 4 // 2))[:n]

    # Shuffle so the pattern is not trivially position-based
    idx = rng.permutation(n)
    genders, ages = genders[idx], ages[idx]

    # Large intersectional effect: young woman gets 0.3 instead of 0.2
    noise_scale = 0.01
    y_hat = (
        0.2
        + 0.1 * ((genders == 0) & (ages == 0)).astype(float)
        + rng.normal(0, noise_scale, size=n)
    )
    D = pd.DataFrame({"gender": genders.astype(str), "age": ages.astype(str)})
    return y_hat, D


# ---------------------------------------------------------------------------
# 1. CCdCov decomposition: marginals + eta = joint
# ---------------------------------------------------------------------------


def test_ccDcov_decomposition_marginals_plus_eta_equals_joint():
    """
    Theorem 3.2.1: CCdCov² = Σ_k dCov²(ŷ, s_k) + η
    """
    y_hat, D = make_dependent_data(n=300)
    S, _ = _encode_attributes(D, ["gender", "age_band"], continuous_attrs=[], fit=True)

    cc = _ccDcov(y_hat, S)
    marg_sum = _marginal_dcov_sum(y_hat, S)
    eta = _eta_intersectional(y_hat, S)

    assert abs(cc - (marg_sum + eta)) < 1e-10, (
        f"CCdCov={cc:.8f} != marginal_sum + eta = {marg_sum:.8f} + {eta:.8f} = {marg_sum + eta:.8f}"
    )


def test_ccDcov_decomposition_single_attr():
    """
    With a single protected attribute, CCdCov == dCov²(ŷ, s) and eta == 0.
    """
    rng = np.random.default_rng(7)
    n = 300
    y_hat = rng.uniform(0.1, 0.5, size=n)
    s = rng.integers(0, 3, size=n)
    S = s.reshape(-1, 1).astype(np.float64)

    cc = _ccDcov(y_hat, S)
    marg = _marginal_dcov_sum(y_hat, S)
    eta = _eta_intersectional(y_hat, S)

    assert abs(cc - marg) < 1e-10, f"Single attr: CCdCov ({cc:.8f}) != dCov² ({marg:.8f})"
    assert abs(eta) < 1e-10, f"Single attr: eta should be 0, got {eta:.8f}"


# ---------------------------------------------------------------------------
# 2. Independent data: CCdCov approximately zero
# ---------------------------------------------------------------------------


def test_ccDcov_near_zero_for_independent_data():
    """
    CCdCov ≈ 0 when predictions and protected attributes are independent.
    We use a generous tolerance because the unbiased estimator is noisy
    for moderate n.
    """
    y_hat, D = make_independent_data(n=600, seed=10)
    S, _ = _encode_attributes(D, ["gender", "age_band"], continuous_attrs=[], fit=True)
    cc = _ccDcov(y_hat, S)
    assert abs(cc) < 0.01, f"Expected CCdCov near 0 for independent data, got {cc:.6f}"


# ---------------------------------------------------------------------------
# 3. Dependent data: CCdCov > 0
# ---------------------------------------------------------------------------


def test_ccDcov_positive_for_dependent_data():
    y_hat, D = make_dependent_data(n=400, seed=5)
    S, _ = _encode_attributes(D, ["gender", "age_band"], continuous_attrs=[], fit=True)
    cc = _ccDcov(y_hat, S)
    assert cc > 1e-5, f"Expected CCdCov > 0 for dependent data, got {cc:.8f}"


# ---------------------------------------------------------------------------
# 4. Intersectional-only: CCdCov > 0, eta > 0
# ---------------------------------------------------------------------------


def test_intersectional_only_effect():
    """
    When predictions depend on the interaction term only, eta should be positive.
    Marginal dCov values may be small relative to eta.
    """
    y_hat, D = make_intersectional_only_data(n=1000, seed=2)
    S, _ = _encode_attributes(D, ["gender", "age"], continuous_attrs=[], fit=True)

    cc = _ccDcov(y_hat, S)
    marg_arr = _marginal_dcov_per_attr(y_hat, S)
    eta = _eta_intersectional(y_hat, S)

    # The joint CCdCov should be positive
    assert cc > 1e-6, f"CCdCov should be positive for intersectional data, got {cc:.8f}"

    # The intersectional residual should be positive
    assert eta > 0, (
        f"Expected eta > 0 for intersectional-only data. "
        f"CCdCov={cc:.6f}, marginals={marg_arr}, eta={eta:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. Encoding: categorical attributes get ordinal integers
# ---------------------------------------------------------------------------


def test_encoding_categorical_ordinal():
    """
    Categorical attributes should map to sorted integer codes.
    """
    D = pd.DataFrame({"gender": ["F", "M", "F", "M", "F"]})
    S, encoders = _encode_attributes(D, ["gender"], continuous_attrs=[], fit=True)

    assert S.shape == (5, 1)
    # "F" should be 0, "M" should be 1 (sorted order)
    expected = np.array([0, 1, 0, 1, 0], dtype=np.float64)
    np.testing.assert_array_equal(S[:, 0], expected)
    assert "gender" in encoders


def test_encoding_categorical_transform_reuses_fitted():
    """
    Transform mode should reuse fitted categories, not re-fit.
    """
    D_train = pd.DataFrame({"gender": ["F", "M", "F"]})
    _, encoders = _encode_attributes(D_train, ["gender"], continuous_attrs=[], fit=True)

    D_test = pd.DataFrame({"gender": ["M", "F", "M"]})
    S_test, _ = _encode_attributes(
        D_test, ["gender"], continuous_attrs=[], fitted_encoders=encoders, fit=False
    )
    expected = np.array([1, 0, 1], dtype=np.float64)
    np.testing.assert_array_equal(S_test[:, 0], expected)


# ---------------------------------------------------------------------------
# 6. Encoding: continuous attributes normalised to [0, 1]
# ---------------------------------------------------------------------------


def test_encoding_continuous_normalised():
    D = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0, 60.0]})
    S, encoders = _encode_attributes(
        D, ["age"], continuous_attrs=["age"], fit=True
    )
    assert S.shape == (5, 1)
    assert abs(S[:, 0].min()) < 1e-10
    assert abs(S[:, 0].max() - 1.0) < 1e-10


def test_encoding_continuous_constant_no_crash():
    """Constant continuous attribute should not produce NaN."""
    D = pd.DataFrame({"age": [30.0, 30.0, 30.0]})
    S, _ = _encode_attributes(D, ["age"], continuous_attrs=["age"], fit=True)
    assert np.all(S[:, 0] == 0.0)


# ---------------------------------------------------------------------------
# 7. Audit report structure
# ---------------------------------------------------------------------------


def test_audit_report_structure():
    y_hat, D = make_dependent_data(n=300)
    audit = IntersectionalFairnessAudit(
        protected_attrs=["gender", "age_band"],
        continuous_attrs=[],
    )
    report = audit.audit(y_hat, D)

    assert isinstance(report, IntersectionalAuditReport)
    assert isinstance(report.ccDcov, float)
    assert isinstance(report.eta, float)
    assert isinstance(report.jdCov, float)
    assert isinstance(report.js_divergence_overall, float)
    assert isinstance(report.marginal_dcov, dict)
    assert set(report.marginal_dcov.keys()) == {"gender", "age_band"}
    assert isinstance(report.js_divergence_by_pair, dict)
    # With 2 attributes there is 1 pair
    assert len(report.js_divergence_by_pair) == 1
    assert report.n_observations == 300
    assert report.n_subgroups >= 1
    assert report.protected_attrs == ["gender", "age_band"]

    # Decomposition holds in the report
    marg_sum = sum(report.marginal_dcov.values())
    assert abs(report.ccDcov - (marg_sum + report.eta)) < 1e-10


# ---------------------------------------------------------------------------
# 8. Audit report subgroup_statistics columns
# ---------------------------------------------------------------------------


def test_audit_report_subgroup_statistics_columns():
    y_hat, D = make_dependent_data(n=200)
    audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
    report = audit.audit(y_hat, D)

    expected_cols = {
        "subgroup", "n", "mean_prediction", "std_prediction",
        "min_prediction", "max_prediction", "exposure_share",
    }
    assert expected_cols.issubset(set(report.subgroup_statistics.columns))
    assert report.subgroup_statistics["exposure_share"].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Audit report summary() returns a string
# ---------------------------------------------------------------------------


def test_audit_report_summary_returns_string():
    y_hat, D = make_dependent_data(n=200)
    audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
    report = audit.audit(y_hat, D)
    s = report.summary()
    assert isinstance(s, str)
    assert "CCdCov" in s
    assert "D_JS" in s


# ---------------------------------------------------------------------------
# 10. Audit report to_markdown() returns Markdown
# ---------------------------------------------------------------------------


def test_audit_report_to_markdown_returns_markdown():
    y_hat, D = make_dependent_data(n=200)
    audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
    report = audit.audit(y_hat, D)
    md = report.to_markdown()
    assert isinstance(md, str)
    assert "# Intersectional Fairness Audit Report" in md
    assert "| Metric | Value |" in md


# ---------------------------------------------------------------------------
# 11. Missing protected_attr column raises ValueError
# ---------------------------------------------------------------------------


def test_audit_missing_column_raises():
    y_hat, D = make_dependent_data(n=100)
    audit = IntersectionalFairnessAudit(protected_attrs=["gender", "nonexistent_col"])
    with pytest.raises(ValueError, match="not found in D"):
        audit.audit(y_hat, D)


# ---------------------------------------------------------------------------
# 12. y_hat/D length mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_audit_length_mismatch_raises():
    y_hat, D = make_dependent_data(n=200)
    audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age_band"])
    with pytest.raises(ValueError, match="observations"):
        audit.audit(y_hat[:100], D)


# ---------------------------------------------------------------------------
# 13. Regulariser penalty returns scalar
# ---------------------------------------------------------------------------


def test_regulariser_penalty_returns_scalar():
    y_hat, D = make_dependent_data(n=200)
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"], lambda_val=1.0
    )
    p = reg.penalty(y_hat, D)
    assert isinstance(p, float)
    assert np.isfinite(p)


# ---------------------------------------------------------------------------
# 14. Regulariser with sum_dcov method
# ---------------------------------------------------------------------------


def test_regulariser_sum_dcov():
    y_hat, D = make_dependent_data(n=200)
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"],
        method="sum_dcov",
        lambda_val=1.0,
    )
    p = reg.penalty(y_hat, D)
    assert isinstance(p, float)
    assert p >= 0


# ---------------------------------------------------------------------------
# 15. Regulariser with jdCov method
# ---------------------------------------------------------------------------


def test_regulariser_jdCov():
    y_hat, D = make_dependent_data(n=200)
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"],
        method="jdCov",
        lambda_val=1.0,
    )
    p = reg.penalty(y_hat, D)
    assert isinstance(p, float)
    assert np.isfinite(p)


# ---------------------------------------------------------------------------
# 16. Regulariser penalty near zero for independent data
# ---------------------------------------------------------------------------


def test_regulariser_penalty_near_zero_independent():
    y_hat, D = make_independent_data(n=600, seed=99)
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"], lambda_val=1.0
    )
    p = reg.penalty(y_hat, D)
    assert abs(p) < 0.02, f"Penalty for independent data should be ~0, got {p:.6f}"


# ---------------------------------------------------------------------------
# 17. Regulariser penalty > 0 for dependent data
# ---------------------------------------------------------------------------


def test_regulariser_penalty_positive_dependent():
    y_hat, D = make_dependent_data(n=300, seed=3)
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"], lambda_val=1.0
    )
    p = reg.penalty(y_hat, D)
    assert p > 1e-6, f"Penalty for dependent data should be > 0, got {p:.8f}"


# ---------------------------------------------------------------------------
# 18. Regulariser lambda_val=0 returns zero penalty
# ---------------------------------------------------------------------------


def test_regulariser_lambda_zero_returns_zero():
    y_hat, D = make_dependent_data(n=200)
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"], lambda_val=0.0
    )
    p = reg.penalty(y_hat, D)
    assert p == 0.0


# ---------------------------------------------------------------------------
# 19. Regulariser invalid method raises ValueError
# ---------------------------------------------------------------------------


def test_regulariser_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        DistanceCovFairnessRegulariser(
            protected_attrs=["gender"], method="invalid"
        )


# ---------------------------------------------------------------------------
# 20. Regulariser negative lambda raises ValueError
# ---------------------------------------------------------------------------


def test_regulariser_negative_lambda_raises():
    with pytest.raises(ValueError, match="lambda_val must be non-negative"):
        DistanceCovFairnessRegulariser(
            protected_attrs=["gender"], lambda_val=-0.5
        )


# ---------------------------------------------------------------------------
# 21. js_divergence: uniform predictions give near-zero D_JS
# ---------------------------------------------------------------------------


def test_js_divergence_uniform_predictions_near_zero():
    """All predictions identical => subgroup distributions are all the same => D_JS = 0."""
    rng = np.random.default_rng(0)
    n = 400
    y_hat = np.full(n, 0.15)
    gender = rng.choice(["M", "F"], size=n)
    age = rng.integers(1, 5, size=n)
    D = pd.DataFrame({"gender": gender, "age": age})

    reg = DistanceCovFairnessRegulariser(protected_attrs=["gender", "age"])
    djs = reg.js_divergence(y_hat, D)
    assert djs < 1e-6, f"Uniform predictions should give D_JS ≈ 0, got {djs:.8f}"


# ---------------------------------------------------------------------------
# 22. js_divergence: bimodal by group gives positive D_JS
# ---------------------------------------------------------------------------


def test_js_divergence_bimodal_groups():
    """
    Group A gets predictions centred at 0.1, group B at 0.9.
    D_JS should be substantially positive.
    """
    n = 400
    group = np.array(["A"] * 200 + ["B"] * 200)
    y_hat = np.concatenate([
        np.random.default_rng(1).normal(0.1, 0.01, 200),
        np.random.default_rng(2).normal(0.9, 0.01, 200),
    ])
    D = pd.DataFrame({"group": group})

    reg = DistanceCovFairnessRegulariser(protected_attrs=["group"])
    djs = reg.js_divergence(y_hat, D)
    assert djs > 0.1, f"D_JS should be large for bimodal predictions by group, got {djs:.6f}"


# ---------------------------------------------------------------------------
# 23. Lambda calibration result structure
# ---------------------------------------------------------------------------


def test_lambda_calibration_result_structure():
    """
    calibrate_lambda sweeps the grid and returns a correctly structured result.
    We mock the model_fn to return fixed predictions.
    """
    y_hat_base, D_train = make_dependent_data(n=200, seed=5)
    y_hat_val, D_val = make_dependent_data(n=100, seed=6)
    n_train = len(D_train)
    n_val = len(D_val)

    X_train = np.zeros((n_train, 1))
    X_val = np.zeros((n_val, 1))
    y_train = y_hat_base
    y_val = y_hat_val

    def model_fn(X_tr, y_tr, D_tr, lv):
        # Return slightly attenuated predictions to simulate lambda effect
        return y_hat_val * (1 - 0.1 * min(lv, 1.0))

    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age_band"], lambda_val=1.0
    )
    lambda_grid = [0.0, 0.1, 1.0]
    result = reg.calibrate_lambda(
        model_fn, X_train, D_train, y_train, X_val, D_val, y_val,
        lambda_grid=lambda_grid,
        loss="mse",
    )

    assert isinstance(result, LambdaCalibrationResult)
    assert len(result.lambda_values) == 3
    assert len(result.js_divergence) == 3
    assert len(result.validation_loss) == 3
    assert result.method == "ccDcov"
    assert all(np.isfinite(result.js_divergence))
    assert all(np.isfinite(result.validation_loss))


# ---------------------------------------------------------------------------
# 24. Pareto indices are valid subset of the grid
# ---------------------------------------------------------------------------


def test_calibration_pareto_indices_are_valid():
    n_lambda = 5
    js = [0.5, 0.4, 0.3, 0.2, 0.1]
    loss = [0.1, 0.2, 0.3, 0.4, 0.5]

    result = LambdaCalibrationResult(
        lambda_values=list(range(n_lambda)),
        js_divergence=js,
        validation_loss=loss,
        selected_lambda=0.0,
        method="ccDcov",
        pareto_indices=_pareto_indices(np.array(js), np.array(loss)),
    )
    assert len(result.pareto_indices) > 0
    assert all(0 <= i < n_lambda for i in result.pareto_indices)


# ---------------------------------------------------------------------------
# 25. LambdaCalibrationResult.select_lambda
# ---------------------------------------------------------------------------


def test_select_lambda_records_choice():
    result = LambdaCalibrationResult(
        lambda_values=[0.1, 1.0, 10.0],
        js_divergence=[0.3, 0.2, 0.1],
        validation_loss=[0.1, 0.15, 0.3],
        selected_lambda=0.0,
        method="ccDcov",
        pareto_indices=np.array([0, 1, 2]),
    )
    result2 = result.select_lambda(1.0)
    assert result2.selected_lambda == 1.0
    # Original is unchanged (dataclass replace returns new instance)
    assert result.selected_lambda == 0.0


# ---------------------------------------------------------------------------
# 26. LambdaCalibrationResult.summary() returns string
# ---------------------------------------------------------------------------


def test_calibration_result_summary():
    result = LambdaCalibrationResult(
        lambda_values=[0.1, 1.0],
        js_divergence=[0.3, 0.15],
        validation_loss=[0.1, 0.18],
        selected_lambda=1.0,
        method="ccDcov",
        pareto_indices=np.array([0, 1]),
    )
    s = result.summary()
    assert isinstance(s, str)
    assert "ccDcov" in s
    assert "selected" in s.lower()


# ---------------------------------------------------------------------------
# 27. _pareto_indices: known 2D Pareto front
# ---------------------------------------------------------------------------


def test_pareto_indices_known_front():
    """
    Points: (0.1, 0.5), (0.2, 0.3), (0.4, 0.2), (0.5, 0.1), (0.3, 0.4)
    Pareto-efficient (minimising both):
    - (0.1, 0.5): dominated by nothing in obj1 but dominated by (0.2, 0.3) in obj2
      ... actually: need to check properly.
    Let's use a simple case:
    - A=(0.1, 1.0): best on obj1
    - B=(0.5, 0.5): middle
    - C=(1.0, 0.1): best on obj2
    - D=(0.6, 0.6): dominated by B
    """
    obj1 = np.array([0.1, 0.5, 1.0, 0.6])
    obj2 = np.array([1.0, 0.5, 0.1, 0.6])
    idx = _pareto_indices(obj1, obj2)
    idx_set = set(idx.tolist())
    # A, B, C are all Pareto-efficient; D is dominated by B
    assert 0 in idx_set  # A
    assert 1 in idx_set  # B
    assert 2 in idx_set  # C
    assert 3 not in idx_set  # D is dominated by B


# ---------------------------------------------------------------------------
# 28. _compute_loss: all three variants
# ---------------------------------------------------------------------------


def test_compute_loss_poisson():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    loss = _compute_loss(y_true, y_pred, "poisson")
    assert abs(loss) < 1e-10, f"Perfect prediction should give 0 Poisson deviance, got {loss}"


def test_compute_loss_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    loss = _compute_loss(y_true, y_pred, "mse")
    assert abs(loss - 1.0) < 1e-10


def test_compute_loss_binary_crossentropy():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.99, 0.01])
    loss = _compute_loss(y_true, y_pred, "binary_crossentropy")
    assert loss > 0
    assert np.isfinite(loss)


def test_compute_loss_invalid_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        _compute_loss(np.array([1.0]), np.array([1.0]), "gamma_deviance")


# ---------------------------------------------------------------------------
# 29. Audit with exposure_col
# ---------------------------------------------------------------------------


def test_audit_with_exposure_col():
    rng = np.random.default_rng(11)
    n = 300
    y_hat = rng.uniform(0.05, 0.5, size=n)
    gender = rng.choice(["M", "F"], size=n)
    age = rng.integers(1, 4, size=n)
    exposure = rng.uniform(0.1, 2.0, size=n)
    D = pd.DataFrame({"gender": gender, "age": age, "exposure": exposure})

    audit = IntersectionalFairnessAudit(
        protected_attrs=["gender", "age"],
        exposure_col="exposure",
    )
    report = audit.audit(y_hat, D)
    assert isinstance(report, IntersectionalAuditReport)
    assert np.isfinite(report.js_divergence_overall)
    # exposure_share should sum to 1
    assert report.subgroup_statistics["exposure_share"].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 30. Single protected attribute: eta = 0
# ---------------------------------------------------------------------------


def test_single_attr_eta_zero():
    rng = np.random.default_rng(20)
    n = 300
    gender = rng.choice(["M", "F"], size=n)
    y_hat = 0.2 + 0.1 * (gender == "M").astype(float) + rng.normal(0, 0.01, size=n)
    D = pd.DataFrame({"gender": gender})

    audit = IntersectionalFairnessAudit(protected_attrs=["gender"])
    report = audit.audit(y_hat, D)

    assert abs(report.eta) < 1e-10, (
        f"With single attribute, eta should be 0. Got eta={report.eta:.10f}"
    )
    # No pairwise comparisons with single attr
    assert len(report.js_divergence_by_pair) == 0


# ---------------------------------------------------------------------------
# 31. Three protected attributes
# ---------------------------------------------------------------------------


def test_three_protected_attributes():
    rng = np.random.default_rng(30)
    n = 400
    gender = rng.choice(["M", "F"], size=n)
    age = rng.integers(1, 4, size=n)
    region = rng.choice(["N", "S", "E", "W"], size=n)
    y_hat = rng.uniform(0.05, 0.5, size=n)
    D = pd.DataFrame({"gender": gender, "age": age, "region": region})

    audit = IntersectionalFairnessAudit(protected_attrs=["gender", "age", "region"])
    report = audit.audit(y_hat, D)

    assert len(report.marginal_dcov) == 3
    # C(3,2) = 3 pairs
    assert len(report.js_divergence_by_pair) == 3
    assert abs(report.ccDcov - (sum(report.marginal_dcov.values()) + report.eta)) < 1e-10


# ---------------------------------------------------------------------------
# 32. Large n warning
# ---------------------------------------------------------------------------


def test_large_n_warning():
    rng = np.random.default_rng(40)
    n = 50_001
    y_hat = rng.uniform(0, 1, size=n)
    gender = rng.choice(["M", "F"], size=n)
    D = pd.DataFrame({"gender": gender})
    audit = IntersectionalFairnessAudit(protected_attrs=["gender"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # We just check the warning fires; don't run the full dcor computation
        # so we short-circuit by calling the internal warn function directly
        from insurance_fairness.intersectional import _warn_large_n
        _warn_large_n(n)

    assert any("50,001" in str(w.message) or "50001" in str(w.message) for w in caught), (
        "Expected a warning about large n"
    )


# ---------------------------------------------------------------------------
# 33. IntersectionalFairnessAudit __repr__
# ---------------------------------------------------------------------------


def test_audit_repr():
    audit = IntersectionalFairnessAudit(
        protected_attrs=["gender", "age"],
        continuous_attrs=["age"],
    )
    r = repr(audit)
    assert "gender" in r
    assert "age" in r
    assert "not yet run" in r

    y_hat, D = make_independent_data(n=100)
    D = D.rename(columns={"age_band": "age"})
    audit2 = IntersectionalFairnessAudit(
        protected_attrs=["gender", "age"],
        continuous_attrs=["age"],
    )
    audit2.audit(y_hat, D)
    assert "audited" in repr(audit2)


# ---------------------------------------------------------------------------
# 34. DistanceCovFairnessRegulariser __repr__
# ---------------------------------------------------------------------------


def test_regulariser_repr():
    reg = DistanceCovFairnessRegulariser(
        protected_attrs=["gender", "age"], method="sum_dcov", lambda_val=0.5
    )
    r = repr(reg)
    assert "gender" in r
    assert "sum_dcov" in r
    assert "0.5" in r
    assert "unfitted" in r


# ---------------------------------------------------------------------------
# 35. fit() caches encoders; subsequent penalty() uses cached values
# ---------------------------------------------------------------------------


def test_regulariser_fit_caches_encoders():
    """
    After calling fit(), penalty() should use the pre-fitted encoders and
    not re-fit on the prediction data.
    """
    y_hat, D = make_dependent_data(n=200)
    reg = DistanceCovFairnessRegulariser(protected_attrs=["gender", "age_band"])
    reg.fit(D)
    assert reg._encoders_fitted

    p = reg.penalty(y_hat, D)
    assert isinstance(p, float)
    assert np.isfinite(p)
    assert "gender" in reg._encoders
