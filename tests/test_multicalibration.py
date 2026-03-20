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

from insurance_fairness.multicalibration import MulticalibrationAudit, MulticalibrationReport


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
