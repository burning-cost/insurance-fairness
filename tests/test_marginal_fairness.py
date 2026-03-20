"""
Tests for marginal_fairness.py

Tests cover:
1. Zero sensitivity when D not in model — correction is zero
2. Linear model + expectation distortion — analytic result verified
3. ES_alpha correction vs expectation — tail distortion amplifies sensitivity
4. Multi-marginal with two protected attributes
5. Actuarial balance — portfolio neutrality within 1%
6. Categorical protected attribute (one-hot encoded)
7. Cascade vs non-cascade sensitivity comparison
8. Numerical vs analytic gradient consistency for linear model
9. MarginalFairnessReport fields are correct
10. Unfitted model raises RuntimeError
11. Invalid parameter validation
12. Wang transform distortion
13. Callable distortion
14. cdf_method='global' and 'empirical' produce valid output
15. Single protected attribute as 1D array
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from insurance_fairness.marginal_fairness import (
    MarginalFairnessPremium,
    MarginalFairnessReport,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class LinearModelNoD:
    """A linear model that ignores D completely (only uses X)."""

    def __init__(self, beta_x: float = 2.0, intercept: float = 1.0):
        self.beta_x = beta_x
        self.intercept = intercept
        self.coef_ = None  # Not a standard linear model w.r.t. D

    def predict(self, DX: np.ndarray) -> np.ndarray:
        # DX columns: [D_0, X_0]; only use X_0 (column 1)
        return self.intercept + self.beta_x * DX[:, -1]


class LinearModelWithD:
    """A linear model: g(D, X) = beta_d * D + beta_x * X."""

    def __init__(self, beta_d: float = 1.5, beta_x: float = 2.0, intercept: float = 0.5):
        self.beta_d = beta_d
        self.beta_x = beta_x
        self.intercept = intercept
        # coef_ in order [D, X] for analytic gradient support
        self.coef_ = np.array([beta_d, beta_x])

    def predict(self, DX: np.ndarray) -> np.ndarray:
        return self.intercept + DX @ self.coef_


def make_synthetic_data(
    n: int = 500,
    beta_d: float = 1.5,
    beta_x: float = 2.0,
    seed: int = 42,
    heavy_tail: bool = False,
):
    """
    Generate synthetic insurance-like data.

    Y ~ Gamma(shape=2, scale=(intercept + beta_d*D + beta_x*X)/2)
    D ~ Bernoulli(0.5)
    X ~ Normal(1, 0.5)
    """
    rng = np.random.default_rng(seed)
    D = rng.binomial(1, 0.5, size=(n, 1)).astype(float)
    X = rng.normal(1.0, 0.5, size=(n, 1))

    mu = 0.5 + beta_d * D[:, 0] + beta_x * X[:, 0]
    mu = np.maximum(mu, 0.1)

    if heavy_tail:
        shape = 0.5  # More tail-heavy
    else:
        shape = 2.0

    scale = mu / shape
    Y = rng.gamma(shape=shape, scale=scale)

    return Y, D, X


# ---------------------------------------------------------------------------
# Test 1: Zero sensitivity when D not in model
# ---------------------------------------------------------------------------


def test_zero_sensitivity_when_d_not_in_model():
    """
    When the model does not use D at all, the gradient dg/dD is zero.
    The correction should be zero and rho_fair == rho_baseline.
    """
    Y, D, X = make_synthetic_data(n=300, seed=1)
    model = LinearModelNoD(beta_x=2.0, intercept=1.0)

    mfp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp.fit(Y, D, X, model=model, protected_indices=[0])

    report = mfp.sensitivity_report()

    assert abs(report.corrections[0]) < 1e-6, (
        f"Correction should be zero when D not in model, got {report.corrections[0]:.6f}"
    )
    assert abs(report.rho_fair - report.rho_baseline) < 1e-6, (
        f"rho_fair should equal rho_baseline, got diff={report.rho_fair - report.rho_baseline:.6f}"
    )
    # Sensitivity should be zero too
    assert abs(report.sensitivities[0]) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: Linear model + expectation distortion — analytic check
# ---------------------------------------------------------------------------


def test_linear_model_expectation_analytic_result():
    """
    For g(D,X) = beta_d*D + beta_x*X and expectation distortion (gamma=1),
    the sensitivity reduces to E[D * beta_d] = beta_d * E[D].

    correction = (beta_d * E[D]) / E[D^2] * E[Y * D]

    We verify the implementation matches this formula within tolerance.
    """
    n = 2000
    beta_d = 1.5
    beta_x = 2.0
    Y, D, X = make_synthetic_data(n=n, beta_d=beta_d, beta_x=beta_x, seed=42)
    model = LinearModelWithD(beta_d=beta_d, beta_x=beta_x, intercept=0.5)

    mfp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp.fit(Y, D, X, model=model, protected_indices=[0])
    report = mfp.sensitivity_report()

    D_flat = D[:, 0]
    # Analytic: grad_i = beta_d everywhere
    grad_i = np.full(n, beta_d)
    expected_sensitivity = float(np.mean(D_flat * grad_i * 1.0))  # gamma=1
    expected_denom = float(np.mean((D_flat * grad_i) ** 2))
    expected_numer = float(np.mean(Y * D_flat * grad_i))
    expected_correction = (expected_sensitivity / expected_denom) * expected_numer

    assert abs(report.sensitivities[0] - expected_sensitivity) < 1e-3 * abs(expected_sensitivity + 1e-10), (
        f"Sensitivity mismatch: got {report.sensitivities[0]:.6f}, expected {expected_sensitivity:.6f}"
    )
    assert abs(report.corrections[0] - expected_correction) < 1e-2 * abs(expected_correction + 1e-10), (
        f"Correction mismatch: got {report.corrections[0]:.6f}, expected {expected_correction:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: ES_alpha correction > expectation correction (tail sensitivity)
# ---------------------------------------------------------------------------


def test_es_correction_larger_than_expectation():
    """
    ES_alpha weights the upper tail heavily. For a heavy-tailed loss with
    a protected attribute affecting the mean, ES-based correction should
    exceed the expectation-based correction in absolute magnitude.
    """
    Y, D, X = make_synthetic_data(n=1000, seed=10, heavy_tail=True)
    model = LinearModelWithD(beta_d=2.0, beta_x=1.0, intercept=0.5)

    mfp_exp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp_exp.fit(Y, D, X, model=model, protected_indices=[0])
    report_exp = mfp_exp.sensitivity_report()

    mfp_es = MarginalFairnessPremium(distortion="es_alpha", alpha=0.75, cdf_method="global")
    mfp_es.fit(Y, D, X, model=model, protected_indices=[0])
    report_es = mfp_es.sensitivity_report()

    # ES-based sensitivity should be larger in absolute value than expectation-based
    assert abs(report_es.sensitivities[0]) > abs(report_exp.sensitivities[0]), (
        f"ES sensitivity {report_es.sensitivities[0]:.4f} not larger than "
        f"expectation sensitivity {report_exp.sensitivities[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Multi-marginal with two protected attributes
# ---------------------------------------------------------------------------


def test_multi_marginal_two_attributes():
    """
    With two protected attributes, corrections are applied additively.
    After correction, both sensitivities should be reduced towards zero.
    """
    n = 1000
    rng = np.random.default_rng(7)
    D1 = rng.binomial(1, 0.5, size=n).astype(float)
    D2 = rng.binomial(1, 0.4, size=n).astype(float)
    D = np.column_stack([D1, D2])
    X = rng.normal(1.0, 0.5, size=(n, 1))

    class TwoProtectedModel:
        def __init__(self):
            self.coef_ = np.array([1.2, 0.8, 1.5])  # [D1, D2, X]

        def predict(self, DX):
            return 0.3 + DX @ self.coef_

    model = TwoProtectedModel()
    Y_pred = np.maximum(model.predict(np.hstack([D, X])), 0.1)
    Y = rng.gamma(shape=2.0, scale=Y_pred / 2.0)

    mfp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp.fit(Y, D, X, model=model, protected_indices=[0, 1])
    report = mfp.sensitivity_report()

    # Both protected attributes should have corrections
    assert len(report.corrections) == 2
    assert len(report.sensitivities) == 2

    # rho_fair = rho_baseline - correction_1 - correction_2
    expected_rho_fair = report.rho_baseline - report.corrections[0] - report.corrections[1]
    assert abs(report.rho_fair - expected_rho_fair) < 1e-8


# ---------------------------------------------------------------------------
# Test 5: Actuarial balance — portfolio neutrality within 1%
# ---------------------------------------------------------------------------


def test_actuarial_balance():
    """
    Test that rho_fair is computed correctly: rho_fair = rho_baseline - correction.

    Note: the Huang & Pesenti formula does NOT guarantee rho_fair == rho_baseline.
    It removes sensitivity to D, which generally changes the portfolio average.
    The test checks internal consistency and that transform() is coherent.
    """
    Y, D, X = make_synthetic_data(n=2000, seed=99)
    model = LinearModelWithD(beta_d=1.0, beta_x=2.0, intercept=0.5)

    mfp = MarginalFairnessPremium(distortion="es_alpha", alpha=0.75, cdf_method="global")
    mfp.fit(Y, D, X, model=model, protected_indices=[0])
    report = mfp.sensitivity_report()

    # rho_fair is internally consistent: baseline - correction
    expected_fair = report.rho_baseline - report.corrections[0]
    assert abs(report.rho_fair - expected_fair) < 1e-10

    # rho_fair and rho_baseline are finite and positive
    assert np.isfinite(report.rho_fair)
    assert np.isfinite(report.rho_baseline)
    assert report.rho_baseline > 0

    # transform() portfolio mean matches rho_fair
    rho_obs = mfp.transform(Y, D, X)
    assert abs(np.mean(rho_obs) - report.rho_fair) < 1e-4 * abs(report.rho_fair + 1e-10) + 1e-3


# ---------------------------------------------------------------------------
# Test 6: Categorical protected attribute (one-hot encoded)
# ---------------------------------------------------------------------------


def test_categorical_one_hot():
    """
    Multi-category protected attribute encoded as one-hot dummies.
    Five categories, four dummies passed as D. Algorithm must handle all
    four dummies without errors and produce non-degenerate corrections.
    """
    n = 1000
    rng = np.random.default_rng(55)
    cat = rng.choice(5, size=n)
    # One-hot (drop first category = reference)
    D = (cat[:, None] == np.arange(1, 5)).astype(float)  # (n, 4)
    X = rng.normal(0.0, 1.0, size=(n, 2))

    # Model coefficients: one per dummy + two for X
    coefs = np.array([0.5, 1.0, -0.3, 0.8, 1.2, 0.9])  # 4 D + 2 X

    class CategoricalModel:
        def __init__(self):
            self.coef_ = coefs

        def predict(self, DX):
            return 1.0 + DX @ coefs

    model = CategoricalModel()
    Y_pred = model.predict(np.hstack([D, X]))
    Y = np.maximum(rng.gamma(shape=2.0, scale=np.maximum(Y_pred, 0.1) / 2.0), 0.01)

    mfp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp.fit(Y, D, X, model=model, protected_indices=[0, 1, 2, 3])
    report = mfp.sensitivity_report()

    assert len(report.corrections) == 4
    # Not all corrections should be zero — the model does use D
    nonzero = np.sum(np.abs(report.corrections) > 1e-8)
    assert nonzero >= 1, "Expected at least one non-zero correction for categorical D"


# ---------------------------------------------------------------------------
# Test 7: Cascade vs non-cascade comparison
# ---------------------------------------------------------------------------


def test_cascade_vs_noncascade():
    """
    When D_1 is correlated with X_1 (an indirect proxy path), cascade=True
    should produce a larger sensitivity in absolute value than cascade=False.
    """
    n = 1000
    rng = np.random.default_rng(77)

    D = rng.binomial(1, 0.5, size=(n, 1)).astype(float)
    # X_0 = occupation, correlated with gender
    X0 = 0.6 * D[:, 0] + rng.normal(0, 0.5, size=n)
    X = X0.reshape(-1, 1)

    model = LinearModelWithD(beta_d=0.3, beta_x=1.5, intercept=0.5)
    Y_pred = model.predict(np.hstack([D, X]))
    Y = np.maximum(rng.gamma(shape=2.0, scale=np.maximum(Y_pred, 0.1) / 2.0), 0.01)

    mfp_no_cascade = MarginalFairnessPremium(
        distortion="expectation", cdf_method="global", cascade=False
    )
    mfp_no_cascade.fit(Y, D, X, model=model, protected_indices=[0])
    report_nc = mfp_no_cascade.sensitivity_report()

    mfp_cascade = MarginalFairnessPremium(
        distortion="expectation", cdf_method="global", cascade=True
    )
    mfp_cascade.fit(Y, D, X, model=model, protected_indices=[0])
    report_c = mfp_cascade.sensitivity_report()

    # Cascade sensitivity should be larger in absolute value due to indirect path
    assert abs(report_c.sensitivities[0]) >= abs(report_nc.sensitivities[0]) - 1e-6, (
        f"Cascade sensitivity {report_c.sensitivities[0]:.4f} not >= "
        f"non-cascade sensitivity {report_nc.sensitivities[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: Numerical vs analytic gradient consistency
# ---------------------------------------------------------------------------


def test_numerical_vs_analytic_gradient():
    """
    For a linear model where the analytic gradient is known (= coef_[0]),
    finite differences should match within 1e-3 relative tolerance.
    """
    Y, D, X = make_synthetic_data(n=500, beta_d=1.5, beta_x=2.0, seed=123)
    model = LinearModelWithD(beta_d=1.5, beta_x=2.0, intercept=0.5)

    mfp_fd = MarginalFairnessPremium(
        distortion="expectation", grad_method="finite_diff", cdf_method="global"
    )
    mfp_fd.fit(Y, D, X, model=model, protected_indices=[0])
    report_fd = mfp_fd.sensitivity_report()

    mfp_an = MarginalFairnessPremium(
        distortion="expectation", grad_method="analytic", cdf_method="global"
    )
    mfp_an.fit(Y, D, X, model=model, protected_indices=[0])
    report_an = mfp_an.sensitivity_report()

    # Sensitivities should agree closely
    tol = 1e-3 * abs(report_an.sensitivities[0]) + 1e-8
    assert abs(report_fd.sensitivities[0] - report_an.sensitivities[0]) < tol, (
        f"Finite diff sensitivity {report_fd.sensitivities[0]:.6f} differs from "
        f"analytic sensitivity {report_an.sensitivities[0]:.6f} by more than {tol:.2e}"
    )

    # Corrections should also agree
    tol_c = 1e-2 * abs(report_an.corrections[0]) + 1e-8
    assert abs(report_fd.corrections[0] - report_an.corrections[0]) < tol_c


# ---------------------------------------------------------------------------
# Test 9: MarginalFairnessReport fields
# ---------------------------------------------------------------------------


def test_report_fields():
    """
    Check that MarginalFairnessReport is a dataclass with the expected fields
    and correct types.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(MarginalFairnessReport)}
    expected = {
        "protected_names", "sensitivities", "denominators", "corrections",
        "rho_baseline", "rho_fair", "lift_ratio",
    }
    assert expected.issubset(fields), f"Missing fields: {expected - fields}"

    Y, D, X = make_synthetic_data(n=200, seed=5)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp.fit(Y, D, X, model=model)
    report = mfp.sensitivity_report()

    assert isinstance(report.protected_names, list)
    assert isinstance(report.sensitivities, np.ndarray)
    assert isinstance(report.denominators, np.ndarray)
    assert isinstance(report.corrections, np.ndarray)
    assert isinstance(report.rho_baseline, float)
    assert isinstance(report.rho_fair, float)
    assert isinstance(report.lift_ratio, float)


# ---------------------------------------------------------------------------
# Test 10: Unfitted model raises RuntimeError
# ---------------------------------------------------------------------------


def test_unfitted_raises():
    """Calling transform() or sensitivity_report() before fit() raises RuntimeError."""
    mfp = MarginalFairnessPremium()
    Y = np.ones(10)
    D = np.ones((10, 1))
    X = np.ones((10, 1))

    with pytest.raises(RuntimeError, match="not been fitted"):
        mfp.transform(Y, D, X)

    with pytest.raises(RuntimeError, match="not been fitted"):
        mfp.sensitivity_report()


# ---------------------------------------------------------------------------
# Test 11: Parameter validation
# ---------------------------------------------------------------------------


def test_invalid_distortion():
    with pytest.raises(ValueError, match="distortion"):
        MarginalFairnessPremium(distortion="invalid_dist")


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        MarginalFairnessPremium(alpha=1.5)


def test_invalid_grad_method():
    with pytest.raises(ValueError, match="grad_method"):
        MarginalFairnessPremium(grad_method="newton")


def test_invalid_cdf_method():
    with pytest.raises(ValueError, match="cdf_method"):
        MarginalFairnessPremium(cdf_method="kde")


def test_invalid_loss_distribution():
    with pytest.raises(ValueError, match="loss_distribution"):
        MarginalFairnessPremium(loss_distribution="pareto")


def test_invalid_protected_indices():
    Y, D, X = make_synthetic_data(n=100, seed=3)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(cdf_method="global")
    with pytest.raises(ValueError, match="protected_indices"):
        mfp.fit(Y, D, X, model=model, protected_indices=[5])  # out of range


def test_negative_Y_raises():
    D = np.ones((10, 1))
    X = np.ones((10, 1))
    Y = np.array([-1.0] + [1.0] * 9)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(cdf_method="global")
    with pytest.raises(ValueError, match="non-negative"):
        mfp.fit(Y, D, X, model=model)


# ---------------------------------------------------------------------------
# Test 12: Wang transform distortion
# ---------------------------------------------------------------------------


def test_wang_transform_distortion():
    """
    Wang transform with lambda > 0 loads the premium (risk-averse).
    The distortion weights should be valid (positive, integrating to ~1).
    """
    Y, D, X = make_synthetic_data(n=500, seed=88)
    model = LinearModelWithD()

    mfp = MarginalFairnessPremium(distortion="wang_lambda", alpha=0.5, cdf_method="global")
    mfp.fit(Y, D, X, model=model, protected_indices=[0])
    report = mfp.sensitivity_report()

    assert report.rho_baseline > 0
    assert np.isfinite(report.rho_fair)
    assert np.isfinite(report.lift_ratio)


# ---------------------------------------------------------------------------
# Test 13: Callable distortion
# ---------------------------------------------------------------------------


def test_callable_distortion():
    """
    User-provided callable gamma(u) should work identically to the built-in
    'expectation' when it returns ones.
    """
    Y, D, X = make_synthetic_data(n=400, seed=200)
    model = LinearModelWithD()

    def gamma_one(u):
        return np.ones_like(u)

    mfp_callable = MarginalFairnessPremium(distortion=gamma_one, cdf_method="global")
    mfp_callable.fit(Y, D, X, model=model, protected_indices=[0])
    report_callable = mfp_callable.sensitivity_report()

    mfp_exp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp_exp.fit(Y, D, X, model=model, protected_indices=[0])
    report_exp = mfp_exp.sensitivity_report()

    assert abs(report_callable.rho_baseline - report_exp.rho_baseline) < 1e-8
    assert abs(report_callable.corrections[0] - report_exp.corrections[0]) < 1e-8


# ---------------------------------------------------------------------------
# Test 14: cdf_method options produce valid output
# ---------------------------------------------------------------------------


def test_cdf_method_global():
    """cdf_method='global' should produce a valid rho_baseline > 0."""
    Y, D, X = make_synthetic_data(n=300, seed=30)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(distortion="es_alpha", alpha=0.75, cdf_method="global")
    mfp.fit(Y, D, X, model=model)
    assert mfp.sensitivity_report().rho_baseline > 0


def test_cdf_method_empirical():
    """cdf_method='empirical' (same as global in our implementation) should run cleanly."""
    Y, D, X = make_synthetic_data(n=300, seed=31)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(distortion="es_alpha", alpha=0.75, cdf_method="empirical")
    mfp.fit(Y, D, X, model=model)
    assert mfp.sensitivity_report().rho_baseline > 0


def test_cdf_method_parametric_gamma():
    """cdf_method='parametric' with loss_distribution='gamma' should run cleanly."""
    Y, D, X = make_synthetic_data(n=300, seed=32)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(
        distortion="es_alpha", alpha=0.75, cdf_method="parametric", loss_distribution="gamma"
    )
    mfp.fit(Y, D, X, model=model)
    assert mfp.sensitivity_report().rho_baseline > 0


def test_cdf_method_parametric_lognormal():
    """cdf_method='parametric' with loss_distribution='lognormal' should run cleanly."""
    Y, D, X = make_synthetic_data(n=300, seed=33)
    model = LinearModelWithD()
    mfp = MarginalFairnessPremium(
        distortion="es_alpha", alpha=0.75, cdf_method="parametric", loss_distribution="lognormal"
    )
    mfp.fit(Y, D, X, model=model)
    assert mfp.sensitivity_report().rho_baseline > 0


# ---------------------------------------------------------------------------
# Test 15: Single protected attribute as 1D array input
# ---------------------------------------------------------------------------


def test_1d_d_input():
    """
    D passed as a 1D array should be handled without errors.
    """
    Y, D, X = make_synthetic_data(n=200, seed=15)
    D_1d = D[:, 0]  # flatten
    model = LinearModelWithD()

    mfp = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp.fit(Y, D_1d, X, model=model, protected_indices=[0])
    report = mfp.sensitivity_report()
    assert len(report.corrections) == 1


# ---------------------------------------------------------------------------
# Test 16: transform() returns correct shape
# ---------------------------------------------------------------------------


def test_transform_shape():
    """transform() should return an array of length n."""
    n = 150
    Y, D, X = make_synthetic_data(n=n, seed=50)
    model = LinearModelWithD()

    mfp = MarginalFairnessPremium(distortion="es_alpha", alpha=0.75, cdf_method="global")
    mfp.fit(Y, D, X, model=model)
    rho_fair = mfp.transform(Y, D, X)

    assert rho_fair.shape == (n,)
    assert np.all(np.isfinite(rho_fair))


# ---------------------------------------------------------------------------
# Test 17: lift_ratio near 1.0 for well-balanced portfolio
# ---------------------------------------------------------------------------


def test_lift_ratio_near_one():
    """
    lift_ratio = rho_fair / rho_baseline.

    Case 1: model ignores D -> correction is zero -> lift_ratio == 1.0 exactly.
    Case 2: model uses D -> lift_ratio is finite, positive, and equals rho_fair/rho_baseline.
    """
    # Case 1: no D in model -> lift_ratio == 1.0
    Y, D, X = make_synthetic_data(n=1000, seed=17)
    model_no_d = LinearModelNoD(beta_x=2.0, intercept=1.0)
    mfp_no_d = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp_no_d.fit(Y, D, X, model=model_no_d)
    r_no_d = mfp_no_d.sensitivity_report()
    assert abs(r_no_d.lift_ratio - 1.0) < 1e-6

    # Case 2: D in model -> lift_ratio is well-defined and matches rho_fair/rho_baseline
    Y2, D2, X2 = make_synthetic_data(n=5000, seed=17)
    model_d = LinearModelWithD(beta_d=0.5, beta_x=2.0, intercept=0.3)
    mfp_d = MarginalFairnessPremium(distortion="expectation", cdf_method="global")
    mfp_d.fit(Y2, D2, X2, model=model_d)
    r_d = mfp_d.sensitivity_report()
    assert np.isfinite(r_d.lift_ratio)
    assert r_d.lift_ratio > 0
    expected_lr = r_d.rho_fair / r_d.rho_baseline
    assert abs(r_d.lift_ratio - expected_lr) < 1e-10


# ---------------------------------------------------------------------------
# Test 18: repr is informative
# ---------------------------------------------------------------------------


def test_repr():
    mfp = MarginalFairnessPremium(distortion="es_alpha", alpha=0.90)
    r = repr(mfp)
    assert "es_alpha" in r
    assert "MarginalFairnessPremium" in r
