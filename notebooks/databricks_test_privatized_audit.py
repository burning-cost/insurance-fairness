# Databricks notebook source
# Test PrivatizedFairnessAudit — Zhang, Liu & Shi (2025) arXiv:2504.11775

# COMMAND ----------

# MAGIC %pip install --quiet scikit-learn numpy

# COMMAND ----------

# Install the package from the uploaded source wheel or directly from source
import subprocess
result = subprocess.run(
    ["pip", "install", "--quiet", "-e", "/Workspace/insurance-fairness"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

import insurance_fairness
print("version:", insurance_fairness.__version__)

from insurance_fairness import PrivatizedFairnessAudit, PrivatizedAuditResult
import dataclasses
print("PrivatizedFairnessAudit imported OK")
print("PrivatizedAuditResult is dataclass:", dataclasses.is_dataclass(PrivatizedAuditResult))

# COMMAND ----------

import numpy as np
import warnings

def randomised_response(D, pi, K, rng):
    n = len(D)
    S = D.copy()
    for i in range(n):
        if rng.random() > pi:
            others = [k for k in range(K) if k != D[i]]
            S[i] = rng.choice(others)
    return S

# COMMAND ----------

# ============================================================
# TEST 1: Known epsilon, two groups
# ============================================================
print("=" * 60)
print("TEST 1: Known epsilon, binary groups")
print("=" * 60)

epsilon = 2.0
K = 2
exp_e = np.exp(epsilon)
pi_true = exp_e / (K - 1 + exp_e)

rng = np.random.default_rng(0)
n = 3000
D = rng.integers(0, K, n)
X = rng.normal(0, 1, (n, 4))
Y = rng.poisson(0.08 + 0.04 * (D == 1)).astype(float)
S = randomised_response(D, pi_true, K, rng)

audit1 = PrivatizedFairnessAudit(
    n_groups=2,
    epsilon=epsilon,
    reference_distribution="uniform",
    loss="poisson",
    nuisance_backend="sklearn",
    random_state=42,
)
audit1.fit(X, Y, S)

mats = audit1.correction_matrices()
print(f"pi estimated: {mats['pi']:.4f} (true: {pi_true:.4f})")
print(f"C1 (noise amplification): {mats['C1']:.4f}")
print(f"T_inv diagonal: {mats['T_inv'][0,0]:.4f}")
print(f"T_inv off-diag: {mats['T_inv'][0,1]:.4f}")
print(f"p_corrected: {audit1.p_corrected_}")
print(f"p_star (uniform): {audit1.p_star_}")
print(f"negative_weight_frac: {audit1.negative_weight_frac_:.4f}")

fair = audit1.fair_predictions_
mean_g0 = fair[S == 0].mean()
mean_g1 = fair[S == 1].mean()
ratio = max(mean_g0, mean_g1) / min(mean_g0, mean_g1)
print(f"Group mean ratio (fair): {ratio:.4f} (should be < 1.05)")
assert ratio < 1.05, f"Premium gap too large: {ratio:.4f}"

bound = audit1.statistical_bound(delta=0.05)
print(f"Statistical bound (95%): {bound:.4f}")
assert bound > 0

print("TEST 1 PASSED")

# COMMAND ----------

# ============================================================
# TEST 2: Anchor-point pi recovery
# ============================================================
print("=" * 60)
print("TEST 2: Anchor-point pi recovery")
print("=" * 60)

pi_true2 = 0.85
K = 2
rng2 = np.random.default_rng(7)
n2 = 2000

D2 = rng2.integers(0, K, n2)
X2 = np.column_stack([
    rng2.normal(0, 1, n2),
    rng2.normal(0, 1, n2),
    D2.astype(float) + rng2.normal(0, 0.1, n2),
])
Y2 = rng2.poisson(0.1 + 0.05 * D2).astype(float)
S2 = randomised_response(D2, pi_true2, K, rng2)

# Add 300 anchor observations with strong group signal
n_anchor = 300
D_anchor = rng2.integers(0, K, n_anchor)
X_anchor = np.column_stack([
    rng2.normal(0, 0.1, n_anchor),
    rng2.normal(0, 0.1, n_anchor),
    D_anchor.astype(float) * 10.0,
])
S_anchor = randomised_response(D_anchor, pi_true2, K, rng2)

X_all = np.vstack([X2, X_anchor])
Y_all = np.concatenate([Y2, rng2.poisson(0.1, n_anchor).astype(float)])
S_all = np.concatenate([S2, S_anchor])

audit2 = PrivatizedFairnessAudit(
    n_groups=2,
    reference_distribution="empirical",
    loss="poisson",
    nuisance_backend="sklearn",
    n_anchor_groups=20,
    random_state=99,
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    audit2.fit(X_all, Y_all, S_all, X_anchor=X_all)

print(f"pi_true: {pi_true2:.4f}")
print(f"pi_estimated: {audit2.pi_:.4f}")
print(f"pi_known: {audit2.pi_known_}")
print(f"anchor_quality: {audit2.anchor_quality_:.4f}")
diff = abs(audit2.pi_ - pi_true2)
print(f"|pi_est - pi_true| = {diff:.4f} (threshold: 0.04)")
assert diff < 0.04, f"pi estimation off by {diff:.4f}"
assert audit2.pi_known_ is False
print("TEST 2 PASSED")

# COMMAND ----------

# ============================================================
# TEST 3: Uniform vs empirical reference
# ============================================================
print("=" * 60)
print("TEST 3: Uniform vs empirical reference distribution")
print("=" * 60)

rng3 = np.random.default_rng(21)
n3 = 2000
K = 2
D3 = rng3.choice([0, 1], size=n3, p=[0.7, 0.3])
X3 = rng3.normal(0, 1, (n3, 3))
Y3 = rng3.poisson(0.08 + 0.04 * D3).astype(float)
S3 = randomised_response(D3, 0.88, K, rng3)

audit3_uniform = PrivatizedFairnessAudit(
    n_groups=2, pi=0.88, reference_distribution="uniform",
    loss="poisson", nuisance_backend="sklearn", random_state=42,
)
audit3_uniform.fit(X3, Y3, S3)

audit3_empirical = PrivatizedFairnessAudit(
    n_groups=2, pi=0.88, reference_distribution="empirical",
    loss="poisson", nuisance_backend="sklearn", random_state=42,
)
audit3_empirical.fit(X3, Y3, S3)

fair_uniform = audit3_uniform.fair_predictions_
mean0_u = fair_uniform[S3 == 0].mean()
mean1_u = fair_uniform[S3 == 1].mean()
gap_uniform = abs(mean0_u - mean1_u) / max(mean0_u, mean1_u)

p_star_emp = audit3_empirical.p_star_
print(f"Uniform p_star: {audit3_uniform.p_star_}")
print(f"Empirical p_star: {p_star_emp}")
print(f"Group mean ratio (uniform): {max(mean0_u,mean1_u)/min(mean0_u,mean1_u):.4f}")
print(f"Uniform gap: {gap_uniform:.4f} (should be < 0.10)")
assert gap_uniform < 0.10, f"Uniform gap too large: {gap_uniform:.4f}"
assert p_star_emp[0] > 0.5, "Empirical p_star should weight majority group more"
print("TEST 3 PASSED")

# COMMAND ----------

# ============================================================
# TEST 4: Negative weight warning for small epsilon
# ============================================================
print("=" * 60)
print("TEST 4: Negative weight warning (epsilon=0.3)")
print("=" * 60)

epsilon4 = 0.3
K = 2
exp_e4 = np.exp(epsilon4)
pi_noisy = exp_e4 / (K - 1 + exp_e4)
print(f"pi for epsilon=0.3: {pi_noisy:.4f}")

rng4 = np.random.default_rng(31)
n4 = 1000
D4 = rng4.integers(0, K, n4)
X4 = rng4.normal(0, 1, (n4, 3))
Y4 = rng4.poisson(0.1, n4).astype(float)
S4 = randomised_response(D4, pi_noisy, K, rng4)

audit4 = PrivatizedFairnessAudit(
    n_groups=2, epsilon=epsilon4, reference_distribution="uniform",
    loss="poisson", nuisance_backend="sklearn", random_state=42,
)

caught4 = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    audit4.fit(X4, Y4, S4)
    caught4 = [str(x.message) for x in w if issubclass(x.category, UserWarning)]

print(f"Warnings caught: {caught4}")
has_neg_warning = any("negative_weight_frac" in m for m in caught4)
assert has_neg_warning, f"Expected negative_weight_frac warning. Got: {caught4}"

result4 = audit4.audit_report()
print(f"negative_weight_frac: {result4.negative_weight_frac:.4f}")
assert result4.negative_weight_frac > 0
print("TEST 4 PASSED")

# COMMAND ----------

# ============================================================
# TEST 5: Poisson deviance
# ============================================================
print("=" * 60)
print("TEST 5: Poisson deviance check")
print("=" * 60)

def poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-10)
    y_safe = np.where(y_true > 0, y_true, 1e-10)
    d = 2.0 * (y_safe * np.log(y_safe / y_pred) - y_true + y_pred)
    return float(d.mean())

rng5 = np.random.default_rng(51)
n5 = 3000
K = 2
D5 = rng5.integers(0, K, n5)
X5 = rng5.normal(0, 1, (n5, 4))
Y5 = rng5.poisson(0.08 + 0.04 * D5 + 0.02 * X5[:, 0]).astype(float)
S5 = randomised_response(D5, 0.88, K, rng5)

audit5 = PrivatizedFairnessAudit(
    n_groups=2, pi=0.88, reference_distribution="uniform",
    loss="poisson", nuisance_backend="sklearn", random_state=42,
)
audit5.fit(X5, Y5, S5)
fair5 = audit5.fair_predictions_
naive5 = np.full(n5, Y5.mean())

dev_fair = poisson_deviance(Y5, fair5)
dev_naive = poisson_deviance(Y5, naive5)
print(f"Naive deviance: {dev_naive:.6f}")
print(f"Fair model deviance: {dev_fair:.6f}")
print(f"Ratio (fair/naive): {dev_fair/dev_naive:.3f} (should be < 3.0)")
assert dev_fair < dev_naive * 3.0
assert np.all(fair5 >= 0)
print("TEST 5 PASSED")

# COMMAND ----------

# ============================================================
# TEST 6: K=3 groups smoke test
# ============================================================
print("=" * 60)
print("TEST 6: K=3 groups smoke test")
print("=" * 60)

K6 = 3
rng6 = np.random.default_rng(91)
n6 = 1500
D6 = rng6.integers(0, K6, n6)
X6 = rng6.normal(0, 1, (n6, 3))
Y6 = rng6.poisson(0.05 + 0.02 * D6).astype(float)
S6 = randomised_response(D6, 0.80, K6, rng6)

audit6 = PrivatizedFairnessAudit(
    n_groups=K6, pi=0.80, reference_distribution="uniform",
    loss="poisson", nuisance_backend="sklearn", random_state=42,
)
audit6.fit(X6, Y6, S6)

assert audit6.p_star_.shape == (K6,)
assert len(audit6.group_models_) == K6
mats6 = audit6.correction_matrices()
assert mats6["T_inv"].shape == (K6, K6)

# Verify T_inv @ T = I
pi6 = mats6["pi"]
pi_bar6 = mats6["pi_bar"]
T6 = np.full((K6, K6), pi_bar6)
np.fill_diagonal(T6, pi6)
product6 = mats6["T_inv"] @ T6
np.testing.assert_allclose(product6, np.eye(K6), atol=1e-8)

preds6 = audit6.predict_fair_premium(X6)
assert preds6.shape == (n6,)
assert np.all(np.isfinite(preds6))
print(f"K=3 fair premium: mean={preds6.mean():.4f}, min={preds6.min():.4f}")
print("TEST 6 PASSED")

# COMMAND ----------

# ============================================================
# TEST 7: ValueError for missing noise params
# ============================================================
print("=" * 60)
print("TEST 7: ValueError when pi/epsilon/X_anchor all missing")
print("=" * 60)

rng7 = np.random.default_rng(61)
n7 = 100
X7 = rng7.normal(0, 1, (n7, 2))
Y7 = rng7.poisson(0.1, n7).astype(float)
S7 = rng7.integers(0, 2, n7)

audit7 = PrivatizedFairnessAudit(n_groups=2, nuisance_backend="sklearn")
try:
    audit7.fit(X7, Y7, S7)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"Got expected ValueError: {e}")
print("TEST 7 PASSED")

# COMMAND ----------

# ============================================================
# TEST 8: audit_report() structure
# ============================================================
print("=" * 60)
print("TEST 8: audit_report() structure")
print("=" * 60)

rng8 = np.random.default_rng(81)
n8 = 500
D8 = rng8.integers(0, 2, n8)
X8 = rng8.normal(0, 1, (n8, 3))
Y8 = rng8.poisson(0.1, n8).astype(float)
S8 = randomised_response(D8, 0.88, 2, rng8)

audit8 = PrivatizedFairnessAudit(n_groups=2, pi=0.88, nuisance_backend="sklearn", random_state=0)
audit8.fit(X8, Y8, S8)
result8 = audit8.audit_report()

assert dataclasses.is_dataclass(result8)
assert isinstance(result8, PrivatizedAuditResult)
assert len(result8.group_models) == 2
assert abs(result8.p_star.sum() - 1.0) < 1e-6
assert abs(result8.p_corrected.sum() - 1.0) < 1e-6
assert result8.anchor_quality is None  # pi supplied directly
assert result8.pi_known is True
assert result8.bound_95 > 0
print(f"pi_estimated: {result8.pi_estimated:.4f}")
print(f"p_star: {result8.p_star}")
print(f"p_corrected: {result8.p_corrected}")
print(f"bound_95: {result8.bound_95:.6f}")
print(f"negative_weight_frac: {result8.negative_weight_frac:.4f}")
print("TEST 8 PASSED")

# COMMAND ----------

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ALL 8 FUNCTIONAL TESTS PASSED")
print("PrivatizedFairnessAudit — insurance-fairness v0.3.8")
print("Zhang, Liu & Shi (2025) arXiv:2504.11775")
print("=" * 60)
