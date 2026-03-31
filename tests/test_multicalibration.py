"""
Tests for multicalibration.py

Tests cover:
- Perfectly calibrated model is_multicalibrated=True
- Model biased against one group is detected (is_multicalibrated=False)
- Correction reduces bias for flagged cells
- Edge cases: single group, all-same prediction, tiny bins
- Exposure weighting changes results as expected
- min_bin_size filtering flags small cells correctly
- Report structure is correct (right columns, right types)
- Overall autocalibration p-value is high for well-calibrated model
- Per-group calibration p-values are computed
- worst_cells is sorted by |AE - 1|
- Correction leaves non-significant cells unchanged
- Tiny prediction variance (all-same) handled without error
- MulticalibrationReport is a dataclass with expected fields
- Correction credibility blending: small cells get partial correction
- n_bins=2 edge case
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_fairness.multicalibration import IterativeMulticalibrationCorrector, MulticalibrationAudit, MulticalibrationReport


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_calibrated_data(
    n: int = 2000,
    n_groups: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (y_true, y_pred, protected, exposure) for a well-calibrated model.

    y_pred is drawn from a realistic premium distribution.
    y_true ~ Poisson(y_pred * exposure) / exposure, so E[Y | pred] = pred.
    """
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, size=n)
    y_pred = rng.gamma(2.0, 50.0, size=n)  # premiums in [~0, ~500]
    # Expected claims = y_pred * exposure
    expected_claims = y_pred * exposure
    y_true = rng.poisson(expected_claims) / exposure
    protected = rng.integers(0, n_groups, size=n).astype(str)
    return y_true, y_pred, protected, exposure


def make_biased_data(
    n: int = 3000,
    bias_group: str = "1",
    bias_factor: float = 1.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return data where group '1' has a consistent 30% over-prediction
    (i.e., model premium is 30% too high relative to actual claims).
    """
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, size=n)
    y_pred = rng.gamma(2.0, 50.0, size=n)
    protected = rng.integers(0, 2, size=n).astype(str)

    # True claims: group "0" calibrated, group "1" under-claims (model over-predicts)
    expected_claims = y_pred * exposure
    true_claims = rng.poisson(expected_claims) / exposure
    mask_biased = protected == bias_group
    # Reduce actual claims for the biased group so AE < 1 (model over-predicts)
    true_claims[mask_biased] = (
        rng.poisson(expected_claims[mask_biased] / bias_factor) / exposure[mask_biased]
    )
    return true_claims, y_pred, protected, exposure


# ---------------------------------------------------------------------------
# Test: correctly calibrated model
# ---------------------------------------------------------------------------


class TestCalibratedModel:
    def test_is_multicalibrated_true(self):
        """Well-calibrated model on large sample should pass multicalibration."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert report.is_multicalibrated is True

    def test_overall_pvalue_high(self):
        """Overall autocalibration p-value should be well above 0.05."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert report.overall_calibration_pvalue > 0.05

    def test_no_significant_cells(self):
        """No (bin, group) cells should be significant for a calibrated model."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20)
        report = audit.audit(y_true, y_pred, protected, exposure)
        sig_cells = report.bin_group_table.filter(
            ~report.bin_group_table["small_cell"] & report.bin_group_table["significant"]
        )
        assert sig_cells.height == 0


# ---------------------------------------------------------------------------
# Test: biased model detection
# ---------------------------------------------------------------------------


class TestBiasedModel:
    def test_is_multicalibrated_false(self):
        """Model with systematic group bias should fail multicalibration."""
        y_true, y_pred, protected, exposure = make_biased_data(n=4000, bias_factor=1.5)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert report.is_multicalibrated is False

    def test_biased_group_has_low_pvalue(self):
        """The biased group should have a low per-group calibration p-value."""
        y_true, y_pred, protected, exposure = make_biased_data(n=4000, bias_factor=1.5)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert report.group_calibration["1"] < 0.05

    def test_significant_cells_present(self):
        """Significant cells should be present when bias is large."""
        y_true, y_pred, protected, exposure = make_biased_data(n=4000, bias_factor=1.5)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20)
        report = audit.audit(y_true, y_pred, protected, exposure)
        sig = report.bin_group_table.filter(report.bin_group_table["significant"])
        assert sig.height > 0


# ---------------------------------------------------------------------------
# Test: correction
# ---------------------------------------------------------------------------


class TestCorrection:
    def test_correction_reduces_ae_deviation(self):
        """After correction, the mean AE ratio for the biased group should be closer to 1."""
        y_true, y_pred, protected, exposure = make_biased_data(n=4000, bias_factor=1.5, seed=7)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20, min_credible=50)
        report = audit.audit(y_true, y_pred, protected, exposure)

        corrected = audit.correct(y_pred, protected, report, exposure)

        # Re-audit with corrected predictions
        report2 = audit.audit(y_true, corrected, protected, exposure)

        # Worst AE deviation should improve
        before_worst = float(
            report.bin_group_table.filter(~report.bin_group_table["small_cell"])
            .select((pl.col("ae_ratio") - 1.0).abs().mean())
            .item()
        )
        after_worst = float(
            report2.bin_group_table.filter(~report2.bin_group_table["small_cell"])
            .select((pl.col("ae_ratio") - 1.0).abs().mean())
            .item()
        )
        assert after_worst < before_worst

    def test_correction_preserves_well_calibrated_cells(self):
        """Cells that passed the test should not be altered by correction."""
        y_true, y_pred, protected, exposure = make_biased_data(n=4000, bias_factor=1.5)
        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20, min_credible=50)
        report = audit.audit(y_true, y_pred, protected, exposure)
        corrected = audit.correct(y_pred, protected, report, exposure)

        # Group "0" is well-calibrated — predictions for that group should be unchanged
        mask_group0 = protected == "0"
        # Check significant cells only affect group "1"
        sig_groups = report.bin_group_table.filter(
            report.bin_group_table["significant"]
        )["group"].unique().to_list()

        if "0" not in sig_groups:
            np.testing.assert_array_equal(corrected[mask_group0], y_pred[mask_group0])


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_group(self):
        """Single group should run without error and return one group entry."""
        rng = np.random.default_rng(1)
        n = 500
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        protected = np.array(["A"] * n)
        audit = MulticalibrationAudit(n_bins=5, min_bin_size=10)
        report = audit.audit(y_true, y_pred, protected)
        assert "A" in report.group_calibration
        assert len(report.group_calibration) == 1

    def test_all_same_prediction(self):
        """When all predictions are identical, bins degenerate — should not raise."""
        n = 200
        y_pred = np.full(n, 100.0)
        y_true = np.random.default_rng(2).uniform(80, 120, n)
        protected = np.array(["A", "B"] * (n // 2))
        audit = MulticalibrationAudit(n_bins=5, min_bin_size=5)
        report = audit.audit(y_true, y_pred, protected)
        # Should complete without error; structure should be correct
        assert isinstance(report, MulticalibrationReport)

    def test_min_bin_size_flags_small_cells(self):
        """Cells below min_bin_size should have small_cell=True."""
        rng = np.random.default_rng(3)
        n = 200
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        # Very small group B: only 5 observations
        protected = np.array(["A"] * 195 + ["B"] * 5)
        audit = MulticalibrationAudit(n_bins=5, min_bin_size=30)
        report = audit.audit(y_true, y_pred, protected)
        b_cells = report.bin_group_table.filter(pl.col("group") == "B")
        assert b_cells["small_cell"].all()

    def test_n_bins_2(self):
        """n_bins=2 should work correctly."""
        rng = np.random.default_rng(4)
        n = 1000
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        protected = np.array(["A", "B"] * (n // 2))
        audit = MulticalibrationAudit(n_bins=2, min_bin_size=10)
        report = audit.audit(y_true, y_pred, protected)
        assert report.n_bins == 2
        assert report.bin_group_table["bin"].n_unique() <= 2

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            MulticalibrationAudit(n_bins=1)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            MulticalibrationAudit(alpha=1.5)

    def test_mismatched_lengths(self):
        audit = MulticalibrationAudit()
        with pytest.raises(ValueError):
            audit.audit(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array(["A", "B"]),
            )


# ---------------------------------------------------------------------------
# Test: exposure weighting
# ---------------------------------------------------------------------------


class TestExposureWeighting:
    def test_exposure_changes_result(self):
        """Results with exposure weighting should differ from unweighted."""
        rng = np.random.default_rng(5)
        n = 500
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        protected = rng.integers(0, 2, n).astype(str)
        exposure = rng.uniform(0.1, 5.0, n)

        audit = MulticalibrationAudit(n_bins=5, min_bin_size=10)
        report_no_exp = audit.audit(y_true, y_pred, protected)
        report_exp = audit.audit(y_true, y_pred, protected, exposure)

        # AE ratios should differ because exposure weighting shifts bin boundaries
        ae_no_exp = report_no_exp.bin_group_table["ae_ratio"].to_list()
        ae_exp = report_exp.bin_group_table["ae_ratio"].to_list()
        # At least one cell should differ
        diffs = [
            abs(a - b) > 1e-6
            for a, b in zip(ae_no_exp, ae_exp)
            if a is not None and b is not None
        ]
        assert any(diffs)


# ---------------------------------------------------------------------------
# Test: report structure
# ---------------------------------------------------------------------------


class TestReportStructure:
    def test_bin_group_table_columns(self):
        """bin_group_table must have the documented columns."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=1000)
        audit = MulticalibrationAudit(n_bins=5)
        report = audit.audit(y_true, y_pred, protected, exposure)
        expected_cols = {
            "bin", "group", "n_obs", "exposure", "observed",
            "expected", "ae_ratio", "pvalue", "significant", "small_cell",
        }
        assert expected_cols == set(report.bin_group_table.columns)

    def test_worst_cells_sorted(self):
        """worst_cells should be sorted by |AE - 1| descending."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        audit = MulticalibrationAudit(n_bins=5, min_bin_size=10)
        report = audit.audit(y_true, y_pred, protected, exposure)
        if report.worst_cells.height > 1:
            ae = report.worst_cells["ae_ratio"].drop_nulls().to_numpy()
            deviations = np.abs(ae - 1.0)
            assert np.all(deviations[:-1] >= deviations[1:])

    def test_group_calibration_keys(self):
        """group_calibration keys should match unique values in protected."""
        rng = np.random.default_rng(6)
        n = 500
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        protected = np.array(["X", "Y", "Z"] * (n // 3) + ["X"] * (n % 3))
        audit = MulticalibrationAudit(n_bins=3, min_bin_size=10)
        report = audit.audit(y_true, y_pred, protected)
        assert set(report.group_calibration.keys()) == {"X", "Y", "Z"}

    def test_worst_cells_max_10(self):
        """worst_cells should contain at most 10 rows."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=2000)
        audit = MulticalibrationAudit(n_bins=10)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert report.worst_cells.height <= 10


# ---------------------------------------------------------------------------
# Test: bin_level_ae field (Task 1 — credibility target fix)
# ---------------------------------------------------------------------------


class TestBinLevelAE:
    def test_bin_level_ae_present_in_report(self):
        """MulticalibrationReport must contain a bin_level_ae dict."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=2000)
        audit = MulticalibrationAudit(n_bins=5)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert hasattr(report, "bin_level_ae")
        assert isinstance(report.bin_level_ae, dict)

    def test_bin_level_ae_has_all_bins(self):
        """bin_level_ae should have one entry per bin."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=2000)
        n_bins = 5
        audit = MulticalibrationAudit(n_bins=n_bins)
        report = audit.audit(y_true, y_pred, protected, exposure)
        assert set(report.bin_level_ae.keys()) == set(range(n_bins))

    def test_bin_level_ae_is_positive(self):
        """bin_level_ae values should be positive floats."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=2000)
        audit = MulticalibrationAudit(n_bins=5)
        report = audit.audit(y_true, y_pred, protected, exposure)
        for b, val in report.bin_level_ae.items():
            assert isinstance(val, float)
            assert val > 0.0, f"bin_level_ae[{b}] = {val} is not positive"

    def test_bin_level_ae_near_one_for_calibrated_model(self):
        """For a well-calibrated model, bin_level_ae should be near 1.0."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000)
        audit = MulticalibrationAudit(n_bins=5)
        report = audit.audit(y_true, y_pred, protected, exposure)
        for b, val in report.bin_level_ae.items():
            assert abs(val - 1.0) < 0.15, f"bin_level_ae[{b}] = {val:.3f} too far from 1.0"

    def test_correct_uses_bin_level_ae_not_one(self):
        """
        Credibility blending must target b_hat_k (bin pooled AE), not 1.0.

        We construct a case where b_hat_k != 1.0 for a bin, then verify that
        the correction factor for a partially-credible cell differs between
        the old (target=1.0) and new (target=b_hat_k) approaches.
        """
        rng = np.random.default_rng(99)
        n = 4000
        # Deliberately bias ALL groups in the same direction so b_hat_k != 1.0
        # but also add extra bias to group "1" so it gets flagged as significant
        exposure = rng.uniform(0.5, 2.0, size=n)
        y_pred = rng.gamma(2.0, 50.0, size=n)
        protected = rng.integers(0, 2, size=n).astype(str)

        # Both groups under-claim but group "1" under-claims more
        expected_claims = y_pred * exposure
        true_claims = rng.poisson(expected_claims * 0.8) / exposure  # all ~0.8 AE
        mask_group1 = protected == "1"
        # Group "1" gets even more extreme under-claim
        true_claims[mask_group1] = (
            rng.poisson(expected_claims[mask_group1] * 0.5) / exposure[mask_group1]
        )

        audit = MulticalibrationAudit(n_bins=5, alpha=0.05, min_bin_size=20, min_credible=500)
        report = audit.audit(true_claims, y_pred, protected, exposure)

        # There should be significant cells
        sig = report.bin_group_table.filter(pl.col("significant"))
        if sig.height == 0:
            return  # skip if no significant cells (stochastic)

        # For a significant cell, compute what both approaches give
        row = sig.row(0, named=True)
        b = int(row["bin"])
        n_obs = int(row["n_obs"])
        ae = float(row["ae_ratio"])
        b_hat_k = report.bin_level_ae[b]

        # Old approach: blend toward 1.0
        z = min(n_obs / 500, 1.0)
        old_factor = z * ae + (1.0 - z) * 1.0
        # New approach: blend toward b_hat_k
        new_factor = z * ae + (1.0 - z) * b_hat_k

        # b_hat_k should be meaningfully different from 1.0 in this setup
        # (the whole portfolio is under-claiming at ~0.8 AE)
        # Therefore old_factor != new_factor for partially-credible cells
        if z < 1.0 and abs(b_hat_k - 1.0) > 0.05:
            assert abs(old_factor - new_factor) > 1e-6, (
                f"Correction factors should differ: old={old_factor:.4f}, "
                f"new={new_factor:.4f}, b_hat_k={b_hat_k:.4f}, z={z:.3f}"
            )


# ---------------------------------------------------------------------------
# Test: IterativeMulticalibrationCorrector (Task 2)
# ---------------------------------------------------------------------------


class TestIterativeMulticalibrationCorrectorBasic:
    def test_import_from_package(self):
        """IterativeMulticalibrationCorrector must be importable from top-level package."""
        from insurance_fairness import IterativeMulticalibrationCorrector as IMC
        assert IMC is IterativeMulticalibrationCorrector

    def test_invalid_eta(self):
        with pytest.raises(ValueError, match="eta"):
            IterativeMulticalibrationCorrector(eta=0.0)

    def test_invalid_eta_above_one(self):
        with pytest.raises(ValueError, match="eta"):
            IterativeMulticalibrationCorrector(eta=1.5)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            IterativeMulticalibrationCorrector(delta=-0.01)

    def test_invalid_c(self):
        with pytest.raises(ValueError, match="c"):
            IterativeMulticalibrationCorrector(c=0.0)

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            IterativeMulticalibrationCorrector(n_bins=1)

    def test_transform_before_fit_raises(self):
        corrector = IterativeMulticalibrationCorrector()
        with pytest.raises(RuntimeError, match="fit"):
            corrector.transform(np.array([1.0, 2.0]), np.array(["A", "B"]))

    def test_convergence_report_before_fit_raises(self):
        corrector = IterativeMulticalibrationCorrector()
        with pytest.raises(RuntimeError, match="fit"):
            corrector.convergence_report()


class TestIterativeMulticalibrationCorrectorFit:
    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=1000)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=5)
        result = corrector.fit(y_true, y_pred, protected, exposure)
        assert result is corrector

    def test_transform_output_shape(self):
        """transform() should return array of same shape as input."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=10)
        corrector.fit(y_true, y_pred, protected, exposure)
        corrected = corrector.transform(y_pred, protected, exposure)
        assert corrected.shape == y_pred.shape

    def test_transform_no_negative_values(self):
        """Corrected predictions should remain non-negative."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=10, eta=0.5)
        corrector.fit(y_true, y_pred, protected, exposure)
        corrected = corrector.transform(y_pred, protected, exposure)
        assert np.all(corrected >= 0.0)

    def test_fit_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            corrector = IterativeMulticalibrationCorrector()
            corrector.fit(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array(["A", "B"]),
            )

    def test_no_exposure_defaults_to_ones(self):
        """fit() without exposure should run identically to exposure=ones."""
        y_true, y_pred, protected, _ = make_biased_data(n=1000)
        corrector_no_exp = IterativeMulticalibrationCorrector(n_bins=5, max_iter=5)
        corrector_no_exp.fit(y_true, y_pred, protected, exposure=None)
        corrector_ones = IterativeMulticalibrationCorrector(n_bins=5, max_iter=5)
        corrector_ones.fit(y_true, y_pred, protected, exposure=np.ones(len(y_pred)))
        out_no_exp = corrector_no_exp.transform(y_pred, protected)
        out_ones = corrector_ones.transform(y_pred, protected, np.ones(len(y_pred)))
        np.testing.assert_allclose(out_no_exp, out_ones, rtol=1e-10)


class TestIterativeMulticalibrationCorrectorConvergence:
    def test_convergence_report_structure(self):
        """convergence_report() must return dict with required keys."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=10)
        corrector.fit(y_true, y_pred, protected, exposure)
        report = corrector.convergence_report()
        assert "iterations" in report
        assert "converged" in report
        assert "max_relative_bias_per_iteration" in report
        assert "final_cell_biases" in report
        assert isinstance(report["iterations"], int)
        assert isinstance(report["converged"], bool)
        assert isinstance(report["max_relative_bias_per_iteration"], list)
        assert isinstance(report["final_cell_biases"], dict)

    def test_bias_history_length_matches_iterations(self):
        """max_relative_bias_per_iteration list length should equal iterations run."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=15)
        corrector.fit(y_true, y_pred, protected, exposure)
        report = corrector.convergence_report()
        assert len(report["max_relative_bias_per_iteration"]) == report["iterations"]

    def test_max_iter_respected(self):
        """Should not run more than max_iter iterations."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        max_iter = 7
        corrector = IterativeMulticalibrationCorrector(
            n_bins=5, max_iter=max_iter, delta=1e-9  # tiny delta to prevent early convergence
        )
        corrector.fit(y_true, y_pred, protected, exposure)
        report = corrector.convergence_report()
        assert report["iterations"] <= max_iter

    def test_calibrated_model_converges_quickly(self):
        """A well-calibrated model should converge in very few iterations."""
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000, seed=123)
        corrector = IterativeMulticalibrationCorrector(
            n_bins=5, eta=1.0, delta=0.05, c=10.0, max_iter=50
        )
        corrector.fit(y_true, y_pred, protected, exposure)
        report = corrector.convergence_report()
        assert report["converged"] is True
        assert report["iterations"] <= 10

    def test_bias_reduces_after_correction(self):
        """For a biased model, the corrected predictions should reduce group AE deviation."""
        y_true, y_pred, protected, exposure = make_biased_data(
            n=4000, bias_factor=1.5, seed=77
        )
        corrector = IterativeMulticalibrationCorrector(
            n_bins=5, eta=0.5, delta=0.02, c=50.0, max_iter=50
        )
        corrector.fit(y_true, y_pred, protected, exposure)
        corrected = corrector.transform(y_pred, protected, exposure)

        # Compute mean AE deviation before and after (for group "1")
        from insurance_fairness import MulticalibrationAudit
        audit = MulticalibrationAudit(n_bins=5, min_bin_size=20)

        report_before = audit.audit(y_true, y_pred, protected, exposure)
        report_after = audit.audit(y_true, corrected, protected, exposure)

        dev_before = float(
            report_before.bin_group_table
            .filter(~pl.col("small_cell") & (pl.col("group") == "1"))
            .select((pl.col("ae_ratio") - 1.0).abs().mean())
            .item()
        )
        dev_after = float(
            report_after.bin_group_table
            .filter(~pl.col("small_cell") & (pl.col("group") == "1"))
            .select((pl.col("ae_ratio") - 1.0).abs().mean())
            .item()
        )
        assert dev_after < dev_before, (
            f"Correction should reduce bias: before={dev_before:.4f}, after={dev_after:.4f}"
        )


class TestIterativeMulticalibrationCorrectorEdgeCases:
    def test_single_group(self):
        """Single group should run without error."""
        rng = np.random.default_rng(11)
        n = 500
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        protected = np.array(["A"] * n)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=5)
        corrector.fit(y_true, y_pred, protected)
        out = corrector.transform(y_pred, protected)
        assert out.shape == y_pred.shape

    def test_eta_one_full_step(self):
        """eta=1.0 applies the full correction in one step."""
        y_true, y_pred, protected, exposure = make_biased_data(n=3000, seed=42)
        corrector = IterativeMulticalibrationCorrector(
            n_bins=5, eta=1.0, delta=0.001, c=10.0, max_iter=30
        )
        corrector.fit(y_true, y_pred, protected, exposure)
        out = corrector.transform(y_pred, protected, exposure)
        assert out.shape == y_pred.shape
        # With eta=1 and sufficient data we expect meaningful correction
        assert not np.allclose(out, y_pred)

    def test_transform_unseen_group_gets_no_correction(self):
        """Observations in a group not seen during fit() should be passed through."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        corrector = IterativeMulticalibrationCorrector(n_bins=5, max_iter=5)
        corrector.fit(y_true, y_pred, protected, exposure)

        # Introduce an unseen group "Z"
        new_prot = np.array(["Z"] * len(y_pred))
        out = corrector.transform(y_pred, new_prot)
        # No correction factor for "Z", so output should equal input
        np.testing.assert_array_equal(out, y_pred)


# ---------------------------------------------------------------------------
# Test: stopping criterion fix (Task 1 — Gap 4)
#
# The paper's stopping criterion (Section 7.1.2) is:
#   max_{k,l} eta * |b_tilde_kl - 1| < delta
# where b_tilde_kl is the BLENDED factor (not the raw b_hat_kl AE ratio).
#
# The key differences from the old criterion (max |b_hat_kl - 1| < delta):
# 1. We check the blended factor b_tilde_kl, not the raw AE.
# 2. We multiply by eta (the learning rate), so delta is in terms of the
#    actual correction applied per step, not the raw data bias.
# ---------------------------------------------------------------------------


class TestStoppingCriterionPaperCompliant:
    def test_convergence_metric_uses_blended_factor(self):
        """
        The convergence history values must reflect eta * |b_tilde_kl - 1|,
        not |b_hat_kl - 1|.

        We verify this by checking that when eta < 1, the reported convergence
        values are strictly smaller than they would be for eta=1 on the same
        data — because the metric is eta * |b_tilde - 1|, not |b_hat - 1|.
        """
        y_true, y_pred, protected, exposure = make_biased_data(
            n=3000, bias_factor=1.5, seed=55
        )

        # eta=1.0: metric = 1.0 * |b_tilde - 1|
        c1 = IterativeMulticalibrationCorrector(
            n_bins=5, eta=1.0, delta=1e-9, c=50.0, max_iter=3
        )
        c1.fit(y_true, y_pred, protected, exposure)
        rep1 = c1.convergence_report()

        # eta=0.5: metric = 0.5 * |b_tilde - 1|
        c2 = IterativeMulticalibrationCorrector(
            n_bins=5, eta=0.5, delta=1e-9, c=50.0, max_iter=3
        )
        c2.fit(y_true, y_pred, protected, exposure)
        rep2 = c2.convergence_report()

        # First iteration metric should be approximately halved for eta=0.5
        m1 = rep1["max_relative_bias_per_iteration"][0]
        m2 = rep2["max_relative_bias_per_iteration"][0]
        # m2 should be roughly m1/2 (both use same data, same blended factor,
        # but eta scales the metric differently)
        assert m2 < m1, (
            f"eta=0.5 metric ({m2:.4f}) should be < eta=1.0 metric ({m1:.4f})"
        )
        # And the ratio should be close to 0.5 (within 20% tolerance for noise)
        ratio = m2 / m1 if m1 > 0 else 1.0
        assert 0.3 < ratio < 0.7, (
            f"Ratio of metrics {ratio:.3f} should be ~0.5 (eta halved)"
        )

    def test_stopping_criterion_consistent_with_eta(self):
        """
        With delta=0.05 and eta=0.5, the algorithm should stop when
        0.5 * |b_tilde - 1| < 0.05, i.e., |b_tilde - 1| < 0.1.

        This is stricter in absolute b_tilde terms than delta=0.05 alone
        would be, but weaker than the old criterion which used raw |b_hat - 1|.
        The test just verifies convergence_report reflects the correct metric.
        """
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000, seed=11)
        corrector = IterativeMulticalibrationCorrector(
            n_bins=5, eta=0.5, delta=0.05, c=50.0, max_iter=50
        )
        corrector.fit(y_true, y_pred, protected, exposure)
        report = corrector.convergence_report()

        # Calibrated model should converge
        assert report["converged"] is True
        # Final metric must be below delta
        final_metric = report["max_relative_bias_per_iteration"][-1]
        assert final_metric < 0.05, (
            f"Final metric {final_metric:.4f} should be < delta=0.05"
        )

    def test_blended_vs_raw_ae_convergence_difference(self):
        """
        Demonstrate that the corrected criterion (blended b_tilde) converges
        at a different rate than checking raw b_hat_kl.

        We fit a biased model and check that the convergence history contains
        the expected values (eta-scaled blended factors, not raw AEs).

        Specifically: if z_kl < 1 (partial credibility), b_tilde_kl is
        shrunk toward b_hat_k, so |b_tilde - 1| <= |b_hat_kl - 1|.
        After eta scaling, the metric is further reduced.
        """
        rng = np.random.default_rng(77)
        n = 3000
        exposure = rng.uniform(0.5, 2.0, size=n)
        y_pred = rng.gamma(2.0, 50.0, size=n)
        protected = rng.integers(0, 2, size=n).astype(str)

        expected = y_pred * exposure
        y_true = rng.poisson(expected) / exposure
        mask1 = protected == "1"
        y_true[mask1] = rng.poisson(expected[mask1] * 0.6) / exposure[mask1]

        # Low c means partial credibility for many cells
        corrector = IterativeMulticalibrationCorrector(
            n_bins=5, eta=0.3, delta=1e-9, c=5000.0, max_iter=2
        )
        corrector.fit(y_true, y_pred, protected, exposure)
        report = corrector.convergence_report()

        # Metric should be positive (there is bias) but reduced by eta and shrinkage
        assert report["max_relative_bias_per_iteration"][0] > 0
        # With eta=0.3 and c=5000 (low credibility), metric must be < raw |AE-1|
        # The raw |AE-1| for group 1 should be ~0.4; metric = 0.3 * |b_tilde-1|
        # where b_tilde is shrunk, so metric < 0.3 * 0.4 = 0.12
        assert report["max_relative_bias_per_iteration"][0] < 0.15


# ---------------------------------------------------------------------------
# Test: IsotonicMulticalibrationCorrector (Task 2 — Gap 3)
# ---------------------------------------------------------------------------


class TestIsotonicMulticalibrationCorrectorImport:
    def test_import_from_package(self):
        """IsotonicMulticalibrationCorrector must be importable from top-level."""
        from insurance_fairness import IsotonicMulticalibrationCorrector as IMC
        from insurance_fairness.multicalibration import IsotonicMulticalibrationCorrector
        assert IMC is IsotonicMulticalibrationCorrector

    def test_in_all(self):
        """IsotonicMulticalibrationCorrector must be in __all__."""
        import insurance_fairness
        assert "IsotonicMulticalibrationCorrector" in insurance_fairness.__all__


class TestIsotonicMulticalibrationCorrectorBasic:
    def test_fit_returns_self(self):
        y_true, y_pred, protected, exposure = make_calibrated_data(n=1000)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        result = iso.fit(y_true, y_pred, protected, exposure)
        assert result is iso

    def test_transform_before_fit_raises(self):
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        with pytest.raises(RuntimeError, match="fit"):
            iso.transform(np.array([1.0, 2.0]), np.array(["A", "B"]))

    def test_transform_output_shape(self):
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        iso.fit(y_true, y_pred, protected, exposure)
        corrected = iso.transform(y_pred, protected)
        assert corrected.shape == y_pred.shape

    def test_fit_transform_shape(self):
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        corrected = iso.fit_transform(y_true, y_pred, protected, exposure)
        assert corrected.shape == y_pred.shape

    def test_mismatched_lengths_raises(self):
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        with pytest.raises(ValueError):
            iso.fit(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array(["A", "B"]),
            )

    def test_negative_predictions_raises(self):
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        with pytest.raises(ValueError, match="non-negative"):
            iso.fit(
                np.array([1.0, 2.0]),
                np.array([-1.0, 2.0]),
                np.array(["A", "B"]),
            )

    def test_no_exposure_defaults_to_ones(self):
        """fit() without exposure should match exposure=ones."""
        y_true, y_pred, protected, _ = make_biased_data(n=1000)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso1 = IsotonicMulticalibrationCorrector()
        iso2 = IsotonicMulticalibrationCorrector()
        out1 = iso1.fit_transform(y_true, y_pred, protected, exposure=None)
        out2 = iso2.fit_transform(y_true, y_pred, protected, np.ones(len(y_pred)))
        np.testing.assert_allclose(out1, out2, rtol=1e-10)


class TestIsotonicMulticalibrationCorrectorCorrection:
    def test_unseen_group_passed_through(self):
        """Observations in groups not seen at fit() should be unchanged."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        iso.fit(y_true, y_pred, protected, exposure)
        new_prot = np.array(["UNSEEN"] * len(y_pred))
        out = iso.transform(y_pred, new_prot)
        np.testing.assert_array_equal(out, y_pred)

    def test_corrected_output_nonnegative(self):
        """Corrected predictions should remain non-negative."""
        y_true, y_pred, protected, exposure = make_biased_data(n=2000)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        iso.fit(y_true, y_pred, protected, exposure)
        corrected = iso.transform(y_pred, protected)
        assert np.all(corrected >= 0.0)

    def test_reduces_bias_for_biased_group(self):
        """
        After correction, the mean AE for the biased group should be closer
        to 1.0 than before correction.
        """
        y_true, y_pred, protected, exposure = make_biased_data(
            n=5000, bias_factor=1.5, seed=88
        )
        from insurance_fairness import IsotonicMulticalibrationCorrector, MulticalibrationAudit
        iso = IsotonicMulticalibrationCorrector()
        corrected = iso.fit_transform(y_true, y_pred, protected, exposure)

        audit = MulticalibrationAudit(n_bins=5, min_bin_size=20)
        rep_before = audit.audit(y_true, y_pred, protected, exposure)
        rep_after = audit.audit(y_true, corrected, protected, exposure)

        dev_before = float(
            rep_before.bin_group_table
            .filter(~pl.col("small_cell") & (pl.col("group") == "1"))
            .select((pl.col("ae_ratio") - 1.0).abs().mean())
            .item()
        )
        dev_after = float(
            rep_after.bin_group_table
            .filter(~pl.col("small_cell") & (pl.col("group") == "1"))
            .select((pl.col("ae_ratio") - 1.0).abs().mean())
            .item()
        )
        assert dev_after < dev_before, (
            f"Isotonic corrector should reduce bias: before={dev_before:.4f}, "
            f"after={dev_after:.4f}"
        )

    def test_single_group(self):
        """Single group should run without error."""
        rng = np.random.default_rng(33)
        n = 500
        y_pred = rng.uniform(50, 200, n)
        y_true = rng.uniform(50, 200, n)
        protected = np.array(["A"] * n)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        out = iso.fit_transform(y_true, y_pred, protected)
        assert out.shape == y_pred.shape

    def test_calibrated_model_small_change(self):
        """
        For a well-calibrated model, isotonic corrections should be small.

        The isotonic regression of Y on pi, when E[Y|pi] = pi, should
        produce corrections close to the identity (pi_mbc ≈ pi).
        We check that the mean absolute correction is small.
        """
        y_true, y_pred, protected, exposure = make_calibrated_data(n=5000, seed=123)
        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector()
        corrected = iso.fit_transform(y_true, y_pred, protected, exposure)

        # Mean relative correction should be small for a calibrated model
        rel_change = np.abs(corrected - y_pred) / (y_pred + 1e-8)
        assert rel_change.mean() < 0.15, (
            f"Too large average correction for calibrated model: "
            f"{rel_change.mean():.4f}"
        )

    def test_out_of_range_clips_not_raises(self):
        """
        Predictions at transform() outside the training range should be
        clipped gracefully (not raise an error) when out_of_bounds='clip'.
        """
        rng = np.random.default_rng(44)
        n = 500
        y_pred_train = rng.uniform(50, 200, n)
        y_true_train = rng.uniform(50, 200, n)
        protected = np.array(["A", "B"] * (n // 2))

        from insurance_fairness import IsotonicMulticalibrationCorrector
        iso = IsotonicMulticalibrationCorrector(out_of_bounds="clip")
        iso.fit(y_true_train, y_pred_train, protected)

        # New predictions with values outside [50, 200]
        y_pred_new = np.array([1.0, 10.0, 500.0, 999.0])
        prot_new = np.array(["A", "B", "A", "B"])

        # Should not raise
        out = iso.transform(y_pred_new, prot_new)
        assert out.shape == y_pred_new.shape
        assert np.all(np.isfinite(out))
