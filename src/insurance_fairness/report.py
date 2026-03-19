"""
report.py
---------
Generate Markdown audit reports from FairnessReport objects.

The output is structured for two audiences:

1. Pricing committees: executive summary, traffic-light statuses, flagged
   factors. No statistical notation.

2. FCA / compliance review: methodology description, metric definitions,
   regulatory mapping, data caveats. Sufficient detail for a third party to
   evaluate the methodology.

The report includes an explicit mapping to FCA Consumer Duty requirements
and Equality Act 2010 Section 19. This framing is appropriate for use in
board reporting, pricing committee packs, and FCA file reviews.

Usage::

    report = audit.run()
    report.to_markdown("fairness_audit_2024q4.md")

    # Or generate the string directly:
    from insurance_fairness.report import generate_markdown_report
    md = generate_markdown_report(report)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insurance_fairness.audit import FairnessReport


_RAG_SYMBOLS = {"green": "[GREEN]", "amber": "[AMBER]", "red": "[RED]", "unknown": "[N/A]"}

_REGULATORY_MAPPING = """
## Regulatory Compliance Framework

This audit is designed to provide evidence of compliance with the following
regulatory requirements:

**FCA Consumer Duty (PRIN 2A, effective July 2023)**

- PRIN 2A.4 (Price and Value Outcome): Firms must ensure the price a customer
  pays is reasonable relative to the benefits received. The FCA has confirmed
  this applies to differential pricing across groups sharing protected
  characteristics.
- PRIN 2A.4.6: Manufacturers must regularly review whether their products
  provide fair value for all groups of customers.
- FCA Multi-Firm Review (2024): "Most firms were monitoring vulnerable
  customers but had limited, often inadequate, monitoring of differential
  outcomes by demographic group."

**Equality Act 2010, Section 19 (Indirect Discrimination)**

- Applies a provision, criterion or practice that puts persons with a
  protected characteristic at a particular disadvantage relative to others.
- A pricing model that uses rating factors correlated with protected
  characteristics (e.g. postcode correlated with ethnicity) may constitute
  indirect discrimination.
- The legitimate aim defence: use of a correlated factor is lawful if it
  genuinely and independently predicts risk, and if less discriminatory
  alternatives would not achieve adequate risk differentiation.
- This audit identifies which factors are correlated with protected
  characteristics, enabling the firm to assess whether this correlation
  constitutes indirect discrimination.

**FCA Consumer Duty record-keeping (PRIN 2A.4.6)**

- Firms must regularly review and document their monitoring of fair value
  outcomes across customer groups. Keeping written records of this analysis
  satisfies the Consumer Duty governance expectation.
- This report constitutes part of that written record.

**ICOBS and SYSC requirements**

- ICOBS 6B.1 and related rules require that firms' pricing practices do not
  result in customers with protected characteristics being treated less
  favourably.
- SYSC 4.1.1 requires robust governance and controls over pricing models,
  including discrimination risk.
"""

_METHODOLOGY = """
## Methodology

### Data and scope

The audit analyses policy-level data from the model training or monitoring
dataset. All metrics are exposure-weighted: policies are weighted by their
exposure period so that short-period policies do not count equally to
full-year policies.

### Proxy detection

Proxy correlation is measured by two complementary methods:

**CatBoost proxy R-squared**: For each rating factor X_j, a CatBoost model
is trained to predict the protected characteristic S from X_j alone. The
R-squared on a held-out validation set measures how well the factor predicts
the protected characteristic. Higher values indicate stronger proxy
correlation.

**Mutual information**: The mutual information I(X_j; S) measures the
statistical dependence between X_j and S. It is model-free and captures
non-linear relationships that R-squared may miss.

**Thresholds**: Factors with proxy R-squared above 0.05 are flagged amber;
above 0.10 they are flagged red. These thresholds are not prescribed by the
FCA and should be interpreted in conjunction with the substantive analysis
of whether the factor's use is justified.

### Fairness metrics

**Calibration by group (sufficiency)**: For each combination of protected
characteristic group and prediction decile, the actual-to-expected (A/E)
ratio is computed. A value of 1.0 indicates the model is well-calibrated
for that group at that pricing level. Systematic deviations indicate the
model over-prices (A/E < 1) or under-prices (A/E > 1) a particular group.

Calibration by group is the most defensible metric under the Equality Act:
if the model is equally well-calibrated for all groups, any price differences
reflect genuine risk differences. The FCA has not prescribed a calibration
threshold; material deviations (more than 10-20%) warrant investigation.

**Demographic parity ratio**: The ratio of exposure-weighted mean predicted
price for each group relative to the overall mean, measured in log-space
(appropriate for multiplicative pricing models). A ratio of 1.0 indicates
equal average prices. Note that demographic parity is not required by the
Equality Act; risk-based differences are permissible. This metric is a
directional diagnostic, not a compliance test.

**Disparate impact ratio**: The ratio of mean prices between the highest- and
lowest-priced groups. The US EEOC 4/5ths rule uses 0.80 as a threshold;
this is a US regulatory concept and is used here as an indicative benchmark
only.

### Counterfactual testing

Where the protected characteristic is a direct model input, counterfactual
fairness testing computes the change in predicted premium when the
characteristic is flipped (e.g. male to female). A zero impact indicates
the characteristic has no direct price effect. A non-zero impact may or
may not constitute discrimination depending on whether the characteristic
is a permitted direct rating factor.

### Limitations

1. The proxy correlation tests cannot distinguish causal from confounded
   relationships. A rating factor correlated with both ethnicity and claims
   may be correlated because the characteristic itself causes claims, or
   because it is a proxy for unmeasured legitimate risk factors. Causal
   attribution requires additional data and methods not implemented here.

2. Demographic parity, equalised odds, and calibration cannot all be
   satisfied simultaneously when base rates differ across groups
   (Chouldechova, 2017). This audit focuses on calibration as the primary
   criterion, consistent with actuarial fairness principles.

3. Where protected characteristics are not directly observed (e.g. ethnicity),
   analysis is conducted using proxy measures (e.g. ONS LSOA-level ethnicity
   proportions). Proxy-based analysis has lower statistical power and may
   misclassify individual policyholders.

### References

- Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance
  Pricing. ASTIN Bulletin 52(1), 55-89.
- Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of
  Discrimination in Insurance Pricing. European Journal of Operational Research.
- FCA Consumer Duty Finalised Guidance FG22/5 (2023).
- FCA Multi-Firm Review: Outcomes Monitoring under Consumer Duty (2024).
- FCA Thematic Review TR24/2 (2024).
- FCA Evaluation Paper EP25/2 (2025).
- Equality Act 2010, Section 19.
"""


def generate_markdown_report(report: "FairnessReport") -> str:
    """
    Generate a Markdown audit report from a FairnessReport.

    The report is structured for both pricing committee review (executive
    summary, traffic-light statuses) and FCA/compliance review (methodology,
    regulatory mapping).

    Parameters
    ----------
    report:
        A FairnessReport returned by FairnessAudit.run().

    Returns
    -------
    Markdown string.
    """
    lines: list[str] = []

    # Header
    rag_sym = _RAG_SYMBOLS.get(report.overall_rag, "[N/A]")
    lines.append(f"# Fairness Audit Report: {report.model_name}")
    lines.append("")
    lines.append(f"**Date:** {report.audit_date}  ")
    lines.append(f"**Overall status:** {rag_sym}  ")
    lines.append(f"**Policies audited:** {report.n_policies:,}  ")
    lines.append(f"**Exposure:** {report.total_exposure:,.1f}  ")
    lines.append(f"**Protected characteristics:** {', '.join(report.protected_cols)}  ")
    lines.append(f"**Rating factors assessed:** {len(report.factor_cols)}  ")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    if report.overall_rag == "green":
        lines.append(
            "No material fairness concerns were identified in this audit. "
            "All metrics are within acceptable thresholds. Continued monitoring "
            "is recommended as part of the regular model review cycle."
        )
    elif report.overall_rag == "amber":
        lines.append(
            "Potential fairness concerns were identified that require further "
            "investigation. Amber-status metrics do not necessarily indicate "
            "discrimination but warrant substantive review and documentation. "
            "Relevant factors are listed in the detailed findings below."
        )
    elif report.overall_rag == "red":
        lines.append(
            "Material fairness concerns were identified. Red-status metrics "
            "indicate potential proxy discrimination that requires immediate "
            "investigation, documentation of the legitimate aim, and consideration "
            "of mitigating action. Escalation to the pricing committee and "
            "compliance function is recommended."
        )
    lines.append("")

    # Flagged factors
    if report.flagged_factors:
        lines.append("### Flagged Rating Factors")
        lines.append("")
        lines.append(
            "The following rating factors have been identified as potential "
            "proxies for one or more protected characteristics:"
        )
        lines.append("")
        for f in report.flagged_factors:
            lines.append(f"- `{f}`")
        lines.append("")
        lines.append(
            "These factors warrant substantive review of their actuarial "
            "justification and less discriminatory alternatives."
        )
        lines.append("")

    # Per-characteristic results
    lines.append("## Detailed Findings by Protected Characteristic")
    lines.append("")

    for pc, result in report.results.items():
        lines.append(f"### {pc}")
        lines.append("")

        # Metric summary table
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")

        if result.demographic_parity is not None:
            dp = result.demographic_parity
            sym = _RAG_SYMBOLS.get(dp.rag, "[N/A]")
            lines.append(
                f"| Demographic parity log-ratio | {dp.log_ratio:+.4f} (ratio: {dp.ratio:.4f}) | {sym} |"
            )

        if result.calibration is not None:
            cal = result.calibration
            sym = _RAG_SYMBOLS.get(cal.rag, "[N/A]")
            lines.append(
                f"| Max calibration disparity (A/E) | {cal.max_disparity:.4f} | {sym} |"
            )

        if result.disparate_impact is not None:
            di = result.disparate_impact
            sym = _RAG_SYMBOLS.get(di.rag, "[N/A]")
            lines.append(
                f"| Disparate impact ratio | {di.ratio:.4f} | {sym} |"
            )

        if result.gini is not None:
            gini_range = (
                max(result.gini.group_ginis.values())
                - min(result.gini.group_ginis.values())
            ) if result.gini.group_ginis else 0.0
            lines.append(
                f"| Gini coefficient range (across groups) | {gini_range:.4f} | [N/A] |"
            )

        if result.theil is not None:
            t = result.theil
            lines.append(
                f"| Theil index (between-group / total) | "
                f"{t.theil_between:.4f} / {t.theil_total:.4f} | [N/A] |"
            )

        if result.counterfactual is not None:
            cf = result.counterfactual
            lines.append(
                f"| Counterfactual premium impact | "
                f"{(cf.premium_impact_ratio - 1) * 100:+.1f}% | [N/A] |"
            )

        lines.append("")

        # Group means for demographic parity
        if result.demographic_parity is not None and result.demographic_parity.group_means:
            lines.append("**Group-level mean predictions (log-space):**")
            lines.append("")
            lines.append("| Group | Mean (log-space) | Exposure |")
            lines.append("|-------|-----------------|---------|")
            gm = result.demographic_parity.group_means
            ge = result.demographic_parity.group_exposures
            for g in sorted(gm.keys()):
                lines.append(f"| {g} | {gm[g]:.4f} | {ge.get(g, 0):,.1f} |")
            lines.append("")

        # Proxy detection results
        if result.proxy_detection is not None:
            prox = result.proxy_detection
            lines.append("**Proxy detection - top factors:**")
            lines.append("")
            lines.append("| Factor | Proxy R-squared | Mutual Info | Partial Corr | Status |")
            lines.append("|--------|----------------|-------------|--------------|--------|")
            for s in prox.scores[:10]:  # Top 10 only
                r2_str = f"{s.proxy_r2:.4f}" if s.proxy_r2 is not None else "N/A"
                mi_str = f"{s.mutual_information:.4f}" if s.mutual_information is not None else "N/A"
                pc_str = f"{s.partial_correlation:.4f}" if s.partial_correlation is not None else "N/A"
                sym = _RAG_SYMBOLS.get(s.rag, "[N/A]")
                lines.append(f"| `{s.factor}` | {r2_str} | {mi_str} | {pc_str} | {sym} |")
            if len(prox.scores) > 10:
                lines.append(f"| ... ({len(prox.scores) - 10} more) | | | | |")
            lines.append("")

        # Calibration table (abbreviated)
        if result.calibration is not None:
            cal = result.calibration
            lines.append("**Calibration by group and prediction decile (A/E ratios):**")
            lines.append("")
            groups = sorted(set(
                g for d_vals in cal.actual_to_expected.values() for g in d_vals.keys()
            ))
            header = "| Decile | " + " | ".join(str(g) for g in groups) + " |"
            sep = "|--------|" + "---|" * len(groups)
            lines.append(header)
            lines.append(sep)
            for d in sorted(cal.actual_to_expected.keys()):
                row = f"| {d} |"
                for g in groups:
                    val = cal.actual_to_expected[d].get(g)
                    if val is None or (isinstance(val, float) and val != val):
                        row += " N/A |"
                    else:
                        row += f" {val:.3f} |"
                lines.append(row)
            lines.append("")

    # Regulatory mapping
    lines.append(_REGULATORY_MAPPING)

    # Methodology
    lines.append(_METHODOLOGY)

    # Sign-off section
    lines.append("## Sign-off")
    lines.append("")
    lines.append(
        "This audit was produced using the `insurance-fairness` library. "
        "The methodology is documented above. The results are based on the "
        "data and model specified in the audit configuration."
    )
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| **Prepared by** | |")
    lines.append("| **Reviewed by (Pricing Actuary)** | |")
    lines.append("| **Approved by (Head of Pricing)** | |")
    lines.append("| **Date of sign-off** | |")
    lines.append("| **Next review due** | |")
    lines.append("")
    lines.append(
        "*This document constitutes part of the firm's written record of "
        "compliance with the FCA Consumer Duty price and value outcome and "
        "the Equality Act 2010 indirect discrimination provisions.*"
    )
    lines.append("")

    return "\n".join(lines)
