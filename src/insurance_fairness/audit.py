"""
audit.py
--------
FairnessAudit: the main entry point for a full discrimination audit.

FairnessAudit takes a CatBoost model, a policy-level dataset, and a
specification of protected characteristics. It runs all proxy detection,
fairness metric, and counterfactual checks, and returns a structured
FairnessReport.

Typical usage::

    from insurance_fairness import FairnessAudit

    audit = FairnessAudit(
        model=catboost_model,
        data=df,
        protected_cols=["gender"],
        prediction_col="predicted_premium",
        outcome_col="claim_amount",
        exposure_col="exposure",
        factor_cols=["age_band", "vehicle_group", "postcode_district", "ncd_years"],
    )
    report = audit.run()
    report.summary()
    report.to_markdown("audit_2024.md")

To run Pareto front analysis (requires ``pymoo``), pass a dict of alternative
models and set ``run_pareto=True``::

    audit = FairnessAudit(
        model=catboost_model,
        data=df,
        protected_cols=["gender"],
        prediction_col="predicted_premium",
        outcome_col="claim_amount",
        exposure_col="exposure",
        pareto_models={"base": model_a, "fair": model_b},
        run_pareto=True,
    )
    report = audit.run()
    print(report.pareto_result.summary())

The output FairnessReport is a structured dataclass. All constituent metric
objects are accessible for custom analysis. The to_markdown() and to_dict()
methods provide audit-ready outputs.

Regulatory context
------------------
FCA Consumer Duty requires firms to demonstrate ongoing monitoring of
differential outcomes by customer group. This class produces the evidence
record that pricing committees and compliance functions need. Run it as part
of the model review cycle, not as a one-off exercise.

References
----------
FCA Consumer Duty Finalised Guidance FG22/5 (2023).
FCA Multi-Firm Review of Outcomes Monitoring under the Consumer Duty (2024).
FCA Thematic Review TR24/2 (2024).
FCA Evaluation Paper EP25/2 (2025).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import polars as pl

from insurance_fairness._utils import validate_columns
from insurance_fairness.bias_metrics import (
    CalibrationResult,
    DisparateImpactResult,
    DemographicParityResult,
    GiniResult,
    TheilResult,
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
    gini_by_group,
    theil_index,
)
from insurance_fairness.counterfactual import CounterfactualResult, counterfactual_fairness
from insurance_fairness.proxy_detection import ProxyDetectionResult, detect_proxies


# ---------------------------------------------------------------------------
# FairnessReport
# ---------------------------------------------------------------------------


@dataclass
class ProtectedCharacteristicReport:
    """All metrics for a single protected characteristic."""

    protected_col: str
    proxy_detection: ProxyDetectionResult | None = None
    demographic_parity: DemographicParityResult | None = None
    calibration: CalibrationResult | None = None
    disparate_impact: DisparateImpactResult | None = None
    gini: GiniResult | None = None
    theil: TheilResult | None = None
    counterfactual: CounterfactualResult | None = None


@dataclass
class FairnessReport:
    """
    Full fairness audit report.

    Attributes
    ----------
    model_name:
        Human-readable model identifier, included in report headers.
    audit_date:
        Date the audit was run (ISO 8601).
    protected_cols:
        Protected characteristics audited.
    factor_cols:
        Rating factors tested for proxy correlation.
    n_policies:
        Number of policies in the audit dataset.
    total_exposure:
        Total exposure in the audit dataset.
    results:
        Dict mapping protected characteristic name to its
        ProtectedCharacteristicReport.
    flagged_factors:
        Rating factors flagged as potential proxies (amber or red status)
        across any protected characteristic.
    overall_rag:
        Overall traffic-light status: 'green', 'amber', or 'red'.
    pareto_result:
        Pareto front result from NSGA-II, if run_pareto=True was set on
        FairnessAudit. None if Pareto optimisation was not run.
    """

    model_name: str
    audit_date: str
    protected_cols: list[str]
    factor_cols: list[str]
    n_policies: int
    total_exposure: float
    results: dict[str, ProtectedCharacteristicReport]
    flagged_factors: list[str] = field(default_factory=list)
    overall_rag: str = "unknown"
    pareto_result: Any = None  # ParetoResult | None, typed as Any to avoid hard import

    def summary(self) -> None:
        """Print a plain-text summary of the audit."""
        print(self._summary_text())

    def _summary_text(self) -> str:
        lines = [
            "=" * 60,
            f"Fairness Audit: {self.model_name}",
            f"Date: {self.audit_date}",
            f"Policies: {self.n_policies:,} | Exposure: {self.total_exposure:,.1f}",
            f"Overall status: {self.overall_rag.upper()}",
            "=" * 60,
        ]

        for pc, result in self.results.items():
            lines.append(f"\nProtected characteristic: {pc}")
            lines.append("-" * 40)

            if result.demographic_parity is not None:
                dp = result.demographic_parity
                lines.append(
                    f"  Demographic parity log-ratio: {dp.log_ratio:+.4f} "
                    f"(ratio: {dp.ratio:.4f}) [{dp.rag.upper()}]"
                )

            if result.calibration is not None:
                cal = result.calibration
                lines.append(
                    f"  Max calibration disparity: {cal.max_disparity:.4f} "
                    f"[{cal.rag.upper()}]"
                )

            if result.disparate_impact is not None:
                di = result.disparate_impact
                lines.append(
                    f"  Disparate impact ratio: {di.ratio:.4f} [{di.rag.upper()}]"
                )

            if result.proxy_detection is not None:
                flagged = result.proxy_detection.flagged_factors
                if flagged:
                    lines.append(
                        f"  Flagged proxy factors ({len(flagged)}): "
                        + ", ".join(flagged[:5])
                        + ("..." if len(flagged) > 5 else "")
                    )
                else:
                    lines.append("  No factors flagged as proxies.")

            if result.counterfactual is not None:
                cf = result.counterfactual
                lines.append(
                    f"  Counterfactual premium impact: "
                    f"{(cf.premium_impact_ratio - 1) * 100:+.1f}%"
                )

        if self.flagged_factors:
            lines.append("\nFactors with proxy concerns (across all protected characteristics):")
            for f in self.flagged_factors:
                lines.append(f"  - {f}")
        else:
            lines.append("\nNo rating factors flagged with proxy concerns.")

        if self.pareto_result is not None:
            lines.append("\nPareto front analysis:")
            lines.append(f"  Solutions on front: {self.pareto_result.n_solutions}")
            idx = self.pareto_result.selected_point()
            lines.append(
                f"  TOPSIS-selected solution (equal weights): index {idx}"
            )

        return "\n".join(lines)

    def to_markdown(self, path: str) -> None:
        """Write a Markdown audit report to *path*."""
        from insurance_fairness.report import generate_markdown_report  # noqa: PLC0415

        md = generate_markdown_report(self)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all audit results."""
        output: dict = {
            "model_name": self.model_name,
            "audit_date": self.audit_date,
            "protected_cols": self.protected_cols,
            "factor_cols": self.factor_cols,
            "n_policies": self.n_policies,
            "total_exposure": self.total_exposure,
            "overall_rag": self.overall_rag,
            "flagged_factors": self.flagged_factors,
            "results": {},
        }

        for pc, result in self.results.items():
            pc_dict: dict = {"protected_col": pc}

            if result.demographic_parity is not None:
                dp = result.demographic_parity
                pc_dict["demographic_parity"] = {
                    "log_ratio": dp.log_ratio,
                    "ratio": dp.ratio,
                    "group_means": dp.group_means,
                    "rag": dp.rag,
                }

            if result.calibration is not None:
                cal = result.calibration
                pc_dict["calibration"] = {
                    "max_disparity": cal.max_disparity,
                    "rag": cal.rag,
                }

            if result.disparate_impact is not None:
                di = result.disparate_impact
                pc_dict["disparate_impact"] = {
                    "ratio": di.ratio,
                    "group_means": di.group_means,
                    "rag": di.rag,
                }

            if result.proxy_detection is not None:
                pc_dict["proxy_detection"] = {
                    "flagged_factors": result.proxy_detection.flagged_factors,
                    "scores": [
                        {
                            "factor": s.factor,
                            "proxy_r2": s.proxy_r2,
                            "mutual_information": s.mutual_information,
                            "rag": s.rag,
                        }
                        for s in result.proxy_detection.scores
                    ],
                }

            if result.counterfactual is not None:
                cf = result.counterfactual
                pc_dict["counterfactual"] = {
                    "premium_impact_ratio": cf.premium_impact_ratio,
                    "premium_impact_log": cf.premium_impact_log,
                    "method": cf.method,
                }

            output["results"][pc] = pc_dict

        if self.pareto_result is not None:
            output["pareto"] = self.pareto_result.to_dict()

        return output


# ---------------------------------------------------------------------------
# FairnessAudit
# ---------------------------------------------------------------------------


class FairnessAudit:
    """
    Run a full fairness audit of a CatBoost pricing model.

    Parameters
    ----------
    model:
        Fitted CatBoost model (CatBoostRegressor or CatBoostClassifier).
        Pass None to run metrics on existing predictions without re-predicting.
    data:
        Policy-level Polars DataFrame. Must contain all specified columns.
    protected_cols:
        Names of protected characteristic columns. Each will be audited
        independently. Supported types: binary (0/1), string categories,
        or continuous proxies (e.g. ONS ethnicity proportions).
    prediction_col:
        Name of the model prediction column (expected loss rate, pure premium,
        or frequency). Must contain the model's output; not re-computed unless
        model is provided and prediction_col is absent.
    outcome_col:
        Name of the actual outcome column (claim amount, claim count, or
        combined loss). Used for calibration checks.
    exposure_col:
        Name of the exposure column. All metrics are exposure-weighted.
        If absent, all policies are weighted equally (generally inappropriate
        for insurance data).
    factor_cols:
        Rating factor columns to test for proxy correlation with protected
        characteristics. If None, all columns in *data* that are not
        protected_cols, prediction_col, outcome_col, or exposure_col are used.
    model_name:
        Human-readable model identifier for the report.
    run_proxy_detection:
        Whether to run proxy R-squared and mutual information tests.
    run_counterfactual:
        Whether to run counterfactual fairness tests. Requires *model* to be
        provided and the protected characteristic to be a direct model input.
    counterfactual_method:
        'direct_flip' or 'lrtw_marginalisation'.
    n_calibration_deciles:
        Number of prediction deciles for calibration checks.
    n_bootstrap:
        Bootstrap replicates for confidence intervals on parity metrics.
        Set to 0 to skip (faster).
    proxy_catboost_iterations:
        CatBoost iterations for proxy R-squared computation.
    run_pareto:
        If True, run NSGA-II multi-objective Pareto optimisation and attach
        the result to FairnessReport.pareto_result. Requires ``pymoo`` to
        be installed (``pip install insurance-fairness[pareto]``).
        Only meaningful when *pareto_models* contains two or more models.
    pareto_models:
        Dict mapping model name to fitted model. Required when run_pareto=True.
        Should contain at least two models representing different positions
        on the accuracy/fairness trade-off (e.g. standard model + fairness-
        constrained model).
    pareto_pop_size:
        NSGA-II population size. Default 50 is adequate for 2-3 model ensembles.
    pareto_n_gen:
        NSGA-II number of generations. Default 100.
    pareto_seed:
        NSGA-II random seed. Default 42.
    """

    def __init__(
        self,
        model,
        data: "pl.DataFrame | pd.DataFrame",
        protected_cols: Sequence[str],
        prediction_col: str,
        outcome_col: str,
        exposure_col: str | None = None,
        factor_cols: Sequence[str] | None = None,
        model_name: str = "Pricing Model",
        run_proxy_detection: bool = True,
        run_counterfactual: bool = False,
        counterfactual_method: str = "direct_flip",
        n_calibration_deciles: int = 10,
        n_bootstrap: int = 0,
        proxy_catboost_iterations: int = 100,
        run_pareto: bool = False,
        pareto_models: Optional[Dict[str, Any]] = None,
        pareto_pop_size: int = 50,
        pareto_n_gen: int = 100,
        pareto_seed: int = 42,
    ) -> None:
        from insurance_fairness._utils import to_polars  # noqa: PLC0415

        self.model = model
        self.data = to_polars(data)
        self.protected_cols = list(protected_cols)
        self.prediction_col = prediction_col
        self.outcome_col = outcome_col
        self.exposure_col = exposure_col
        self.model_name = model_name
        self.run_proxy_detection = run_proxy_detection
        self.run_counterfactual = run_counterfactual
        self.counterfactual_method = counterfactual_method
        self.n_calibration_deciles = n_calibration_deciles
        self.n_bootstrap = n_bootstrap
        self.proxy_catboost_iterations = proxy_catboost_iterations
        self.run_pareto = run_pareto
        self.pareto_models = pareto_models or {}
        self.pareto_pop_size = pareto_pop_size
        self.pareto_n_gen = pareto_n_gen
        self.pareto_seed = pareto_seed

        # Validate required columns
        required = [prediction_col, outcome_col] + self.protected_cols
        if exposure_col:
            required.append(exposure_col)
        validate_columns(self.data, *required)

        # Determine factor columns
        if factor_cols is not None:
            self.factor_cols = list(factor_cols)
        else:
            exclude = set(self.protected_cols) | {prediction_col, outcome_col}
            if exposure_col:
                exclude.add(exposure_col)
            self.factor_cols = [c for c in self.data.columns if c not in exclude]

    def run(self) -> FairnessReport:
        """
        Run the full audit and return a FairnessReport.

        This runs all enabled checks for each protected characteristic, then
        assembles the results into a structured report with an overall RAG
        status.

        If run_pareto=True, also runs NSGA-II Pareto optimisation and attaches
        the result to FairnessReport.pareto_result. This requires pymoo to be
        installed and pareto_models to contain at least two models.
        """
        from insurance_fairness._utils import resolve_exposure  # noqa: PLC0415

        exposure = resolve_exposure(self.data, self.exposure_col)
        n_policies = len(self.data)
        total_exposure = float(exposure.sum())

        audit_date = datetime.date.today().isoformat()
        results: dict[str, ProtectedCharacteristicReport] = {}
        all_flagged: set[str] = set()
        rag_values: list[str] = []

        for pc in self.protected_cols:
            pc_report = ProtectedCharacteristicReport(protected_col=pc)

            # --- Proxy detection ---
            if self.run_proxy_detection and self.factor_cols:
                pc_report.proxy_detection = detect_proxies(
                    df=self.data,
                    protected_col=pc,
                    factor_cols=self.factor_cols,
                    exposure_col=self.exposure_col,
                    model=self.model,
                    run_proxy_r2=True,
                    run_mutual_info=True,
                    run_partial_corr=True,
                    run_shap=False,
                    catboost_iterations=self.proxy_catboost_iterations,
                )
                for f in pc_report.proxy_detection.flagged_factors:
                    all_flagged.add(f)

            # --- Demographic parity ---
            pc_report.demographic_parity = demographic_parity_ratio(
                df=self.data,
                protected_col=pc,
                prediction_col=self.prediction_col,
                exposure_col=self.exposure_col,
                log_space=True,
                n_bootstrap=self.n_bootstrap,
            )
            rag_values.append(pc_report.demographic_parity.rag)

            # --- Calibration by group ---
            pc_report.calibration = calibration_by_group(
                df=self.data,
                protected_col=pc,
                prediction_col=self.prediction_col,
                outcome_col=self.outcome_col,
                exposure_col=self.exposure_col,
                n_deciles=self.n_calibration_deciles,
            )
            rag_values.append(pc_report.calibration.rag)

            # --- Disparate impact ratio ---
            pc_report.disparate_impact = disparate_impact_ratio(
                df=self.data,
                protected_col=pc,
                prediction_col=self.prediction_col,
                exposure_col=self.exposure_col,
            )
            rag_values.append(pc_report.disparate_impact.rag)

            # --- Gini by group ---
            pc_report.gini = gini_by_group(
                df=self.data,
                protected_col=pc,
                prediction_col=self.prediction_col,
                exposure_col=self.exposure_col,
            )

            # --- Theil index ---
            try:
                pc_report.theil = theil_index(
                    df=self.data,
                    protected_col=pc,
                    prediction_col=self.prediction_col,
                    exposure_col=self.exposure_col,
                )
            except ValueError:
                # Theil requires positive predictions; skip if not satisfied
                pc_report.theil = None

            # --- Counterfactual fairness ---
            if self.run_counterfactual and self.model is not None:
                try:
                    pc_report.counterfactual = counterfactual_fairness(
                        model=self.model,
                        df=self.data,
                        protected_col=pc,
                        feature_cols=self.factor_cols + [pc]
                        if pc not in self.factor_cols
                        else self.factor_cols,
                        prediction_col=self.prediction_col,
                        exposure_col=self.exposure_col,
                        method=self.counterfactual_method,
                    )
                except Exception as exc:
                    # Non-fatal: warn and continue
                    import warnings
                    warnings.warn(
                        f"Counterfactual test for '{pc}' failed: {exc}",
                        stacklevel=2,
                    )

            results[pc] = pc_report

        # Overall RAG: worst across all metrics
        rag_priority = {"red": 2, "amber": 1, "green": 0, "unknown": -1}
        overall_rag = max(rag_values, key=lambda r: rag_priority.get(r, -1), default="green")

        # --- Optional Pareto front analysis ---
        pareto_result = None
        if self.run_pareto:
            pareto_result = self._run_pareto_analysis(exposure.to_numpy())

        return FairnessReport(
            model_name=self.model_name,
            audit_date=audit_date,
            protected_cols=self.protected_cols,
            factor_cols=self.factor_cols,
            n_policies=n_policies,
            total_exposure=total_exposure,
            results=results,
            flagged_factors=sorted(all_flagged),
            overall_rag=overall_rag,
            pareto_result=pareto_result,
        )

    def _run_pareto_analysis(self, exposure_arr: "np.ndarray") -> Any:
        """
        Run NSGA-II Pareto optimisation over pareto_models.

        Returns ParetoResult or None if optimisation fails.
        """
        import numpy as np  # noqa: PLC0415

        from insurance_fairness.pareto import NSGA2FairnessOptimiser  # noqa: PLC0415

        if len(self.pareto_models) < 1:
            print(
                "Warning: run_pareto=True but pareto_models is empty. "
                "Pareto analysis skipped."
            )
            return None

        if len(self.pareto_models) < 2:
            print(
                "Warning: run_pareto=True but only one model in pareto_models. "
                "Pareto analysis requires at least two models for meaningful "
                "trade-off exploration. Running with single model."
            )

        y = self.data[self.outcome_col].to_numpy().astype(float)

        # Use the first protected characteristic for the Pareto objectives
        pc = self.protected_cols[0]

        try:
            optimiser = NSGA2FairnessOptimiser(
                models=self.pareto_models,
                X=self.data,
                y=y,
                exposure=exposure_arr,
                protected_col=pc,
                prediction_col=self.prediction_col,
            )
            return optimiser.run(
                pop_size=self.pareto_pop_size,
                n_gen=self.pareto_n_gen,
                seed=self.pareto_seed,
            )
        except Exception as exc:
            print(f"Warning: Pareto analysis failed: {exc}")
            return None
