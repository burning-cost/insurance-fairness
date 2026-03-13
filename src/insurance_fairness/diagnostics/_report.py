"""
Audit report generation for proxy discrimination diagnostics.

Produces:
  - HTML report for human review (audit trail, regulatory documentation)
  - JSON export for programmatic consumption (downstream tooling, dashboards)

Regulatory references embedded in reports:
  - Equality Act 2010, Section 19 (Indirect Discrimination)
  - FCA PRIN 2A.4 (Consumer Duty -- Fair Value)
  - FCA FG22/5 paras 8.8-8.12 (guidance on pricing practices)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._audit import ProxyDiscriminationResult

try:
    from jinja2 import Environment, BaseLoader
    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Proxy Discrimination Audit Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2em; color: #222; }
    h1 { color: #003366; }
    h2 { color: #005599; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
    th { background: #f0f4f8; }
    .green { color: #006600; font-weight: bold; }
    .amber { color: #cc6600; font-weight: bold; }
    .red { color: #cc0000; font-weight: bold; }
    .summary-box { background: #f8f8ff; border: 1px solid #aab; padding: 1em 2em;
                   border-radius: 4px; margin: 1em 0; }
    .regulatory { background: #fff8e8; border-left: 4px solid #e8a000;
                  padding: 0.5em 1em; margin: 1em 0; font-size: 0.92em; }
    .footer { margin-top: 3em; font-size: 0.85em; color: #888; }
  </style>
</head>
<body>
<h1>Proxy Discrimination Audit Report</h1>
<p><strong>Generated:</strong> {{ report_date }}
   &nbsp;|&nbsp; <strong>Library:</strong> insurance-fairness-diag v0.1.0</p>

<div class="summary-box">
  <h2>Executive Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
    <tr>
      <td>D_proxy (normalised L2-distance)</td>
      <td>{{ "%.4f"|format(d_proxy) }}</td>
      <td class="{{ rag }}">{{ rag|upper }}</td>
    </tr>
    <tr>
      <td>D_proxy 95% CI</td>
      <td>[{{ "%.4f"|format(d_proxy_ci[0]) }}, {{ "%.4f"|format(d_proxy_ci[1]) }}]</td>
      <td></td>
    </tr>
    <tr>
      <td>D_proxy monetary (mean premium &times; D_proxy)</td>
      <td>&pound;{{ "%.2f"|format(d_proxy_monetary) }}</td>
      <td></td>
    </tr>
    <tr>
      <td>Sensitive attribute</td>
      <td>{{ sensitive_col }}</td>
      <td></td>
    </tr>
    <tr>
      <td>Portfolio size</td>
      <td>{{ n_policies|int }}</td>
      <td></td>
    </tr>
  </table>
</div>

<div class="regulatory">
  <strong>Regulatory context:</strong>
  This report supports compliance documentation under the Equality Act 2010 s.19
  (indirect discrimination), FCA PRIN 2A.4 (Consumer Duty -- fair value), and
  FCA FG22/5 paras 8.8-8.12 (pricing practices guidance). D_proxy measures the
  L2-distance of the fitted premium to the admissible (discrimination-free) price
  set. A value above 0.15 (red) warrants immediate investigation and possible
  remediation before the pricing model is deployed or renewed.
</div>

<h2>Shapley Effects Attribution</h2>
<p>Which rating factors drive proxy discrimination? Shapley effects decompose
D_proxy variance across factors using the Owen (2014) permutation estimator.</p>
<table>
  <tr><th>Rank</th><th>Rating Factor</th><th>Shapley Effect (&phi;)</th>
      <th>&phi; Monetary (&pound;)</th><th>Status</th></tr>
  {% for name, se in shapley_effects.items() %}
  <tr>
    <td>{{ se.rank }}</td>
    <td>{{ name }}</td>
    <td>{{ "%.4f"|format(se.phi) }}</td>
    <td>&pound;{{ "%.2f"|format(se.phi_monetary) }}</td>
    <td class="{{ se.rag }}">{{ se.rag|upper }}</td>
  </tr>
  {% endfor %}
</table>

<h2>Premium Benchmarks</h2>
<table>
  <tr><th>Benchmark</th><th>Mean Premium (&pound;)</th><th>Description</th></tr>
  <tr>
    <td>Unaware (current model)</td>
    <td>&pound;{{ "%.2f"|format(mean_unaware) }}</td>
    <td>Current fitted premium, no sensitive attribute in model</td>
  </tr>
  <tr>
    <td>Aware (marginalised)</td>
    <td>&pound;{{ "%.2f"|format(mean_aware) }}</td>
    <td>Refitted model with S, then S marginalised out</td>
  </tr>
  <tr>
    <td>Mean proxy vulnerability</td>
    <td>&pound;{{ "%.2f"|format(mean_proxy_vulnerability) }}</td>
    <td>Mean per-policyholder (unaware &minus; aware)</td>
  </tr>
</table>

<h2>Per-policyholder Distribution</h2>
<table>
  <tr><th>d_proxy_local</th><th>Count</th><th>Share</th></tr>
  <tr>
    <td class="green">Green (&lt; 0.05)</td>
    <td>{{ rag_counts.green }}</td>
    <td>{{ "%.1f"|format(100.0 * rag_counts.green / n_policies) }}%</td>
  </tr>
  <tr>
    <td class="amber">Amber (0.05&ndash;0.15)</td>
    <td>{{ rag_counts.amber }}</td>
    <td>{{ "%.1f"|format(100.0 * rag_counts.amber / n_policies) }}%</td>
  </tr>
  <tr>
    <td class="red">Red (&gt; 0.15)</td>
    <td>{{ rag_counts.red }}</td>
    <td>{{ "%.1f"|format(100.0 * rag_counts.red / n_policies) }}%</td>
  </tr>
</table>

<h2>Methodology</h2>
<ul>
  <li><strong>Admissible price h*</strong>: exposure-weighted mean prediction
      within each S-group, per Lindholm et al. (2022).</li>
  <li><strong>D_proxy</strong>: normalised L2-distance, sqrt(E[(h-h*)^2]) / sqrt(E[h^2]).
      Bootstrap 95% CI from 200 replicates.</li>
  <li><strong>Shapley effects</strong>: Owen (2014) random permutation estimator
      with {{ n_perms }} permutations. Surrogate: RandomForestRegressor on
      discrimination residual D = h - h*.</li>
  <li><strong>Aware premium</strong>: model refitted with S included (distillation
      target = unaware predictions), then S marginalised using empirical distribution.</li>
</ul>

<h2>References</h2>
<ul>
  <li>Lindholm, Richman, Tsanakas, W&uuml;thrich (2022). Discrimination-Free Insurance Pricing. <em>ASTIN Bulletin</em> 52(1).</li>
  <li>Lindholm, Richman, Tsanakas, W&uuml;thrich (2026). Sensitivity-Based Measures of Proxy Discrimination. <em>EJOR</em> (SSRN 4897265).</li>
  <li>Owen, A.B. (2014). Sobol' indices and Shapley value. <em>SIAM/ASA JUQ</em> 2(1).</li>
  <li>C&ocirc;t&eacute;, C&ocirc;t&eacute;, Charpentier (2025). Five premium benchmarks for proxy discrimination.</li>
  <li>Biessy (2024). Revisiting the Discrimination-Free Principle Through Shapley Values. <em>ASTIN Bulletin</em>.</li>
</ul>

<div class="footer">
  <p>Generated by insurance-fairness-diag v0.1.0 (Burning Cost).
  This report is a statistical diagnostic tool. It does not constitute legal
  advice. Consult your compliance team and legal counsel before making
  product or pricing decisions based on this output.</p>
</div>
</body>
</html>
"""


def _simple_render(template_str: str, context: dict) -> str:
    """
    Minimal template rendering without jinja2.

    Only handles {{ expr }} substitution and basic iteration.
    Used as fallback when jinja2 is not available.
    """
    import re

    result = template_str

    # Remove jinja2 block tags and their content for simple fallback
    result = re.sub(r"\{%-?\s*for.*?%\}.*?\{%-?\s*endfor\s*-?%\}", "[see JSON output]", result, flags=re.DOTALL)
    result = re.sub(r"\{%-?\s*if.*?%\}.*?\{%-?\s*endif\s*-?%\}", "", result, flags=re.DOTALL)

    # Handle "%.Nf"|format(expr) patterns -- simplified
    def fmt_replace(m):
        fmt = m.group(1)
        key = m.group(2).strip()
        try:
            val = _nested_get(context, key.split("."))
            return fmt % float(val)
        except Exception:
            return key
    result = re.sub(r'"(%-?\.\d+f)"\|format\(([^)]+)\)', fmt_replace, result)

    # Simple {{ var }} substitution
    def var_replace(m):
        key = m.group(1).strip()
        try:
            parts = key.split(".")
            val = _nested_get(context, parts)
            return str(val)
        except Exception:
            return m.group(0)
    result = re.sub(r"\{\{\s*([^}]+)\s*\}\}", var_replace, result)

    return result


def _nested_get(obj, keys):
    """Retrieve nested attribute/dict value."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    return obj


def generate_html_report(result: "ProxyDiscriminationResult") -> str:
    """
    Generate an HTML audit report from a ProxyDiscriminationResult.

    Parameters
    ----------
    result:
        Fitted ProxyDiscriminationResult.

    Returns
    -------
    HTML string.
    """
    from ._audit import ProxyDiscriminationResult

    # Compute summary stats for the template
    local_df = result.local_scores
    rag_counts_raw = local_df.group_by("rag").agg(pl.len().alias("count"))
    rag_dict = {"green": 0, "amber": 0, "red": 0}
    for row in rag_counts_raw.iter_rows(named=True):
        rag_dict[row["rag"]] = int(row["count"])

    # Simple namespace for rag_counts
    class _RAGCounts:
        def __init__(self, d):
            self.green = d.get("green", 0)
            self.amber = d.get("amber", 0)
            self.red = d.get("red", 0)

    context = {
        "report_date": str(date.today()),
        "d_proxy": result.d_proxy,
        "d_proxy_ci": result.d_proxy_ci,
        "d_proxy_monetary": result.d_proxy_monetary,
        "rag": result.rag,
        "sensitive_col": result.sensitive_col,
        "n_policies": len(local_df),
        "shapley_effects": result.shapley_effects,
        "mean_unaware": float(result.benchmarks.unaware.mean()),
        "mean_aware": float(result.benchmarks.aware.mean()),
        "mean_proxy_vulnerability": float(result.benchmarks.proxy_vulnerability.mean()),
        "rag_counts": _RAGCounts(rag_dict),
        "n_perms": result.n_perms,
    }

    if _HAS_JINJA2:
        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(_HTML_TEMPLATE)
        return tmpl.render(**context)
    else:
        return _simple_render(_HTML_TEMPLATE, context)


def generate_json_report(result: "ProxyDiscriminationResult") -> str:
    """
    Generate a JSON audit report from a ProxyDiscriminationResult.

    The JSON output is structured for programmatic consumption.
    All monetary values are in the same unit as the input premiums.

    Parameters
    ----------
    result:
        Fitted ProxyDiscriminationResult.

    Returns
    -------
    JSON string.
    """
    shapley_data = {
        name: {
            "phi": se.phi,
            "phi_monetary": se.phi_monetary,
            "rank": se.rank,
            "rag": se.rag,
        }
        for name, se in result.shapley_effects.items()
    }

    report = {
        "report_date": str(date.today()),
        "library_version": "0.1.0",
        "regulatory_references": [
            "Equality Act 2010 s.19",
            "FCA PRIN 2A.4",
            "FCA FG22/5 paras 8.8-8.12",
        ],
        "sensitive_col": result.sensitive_col,
        "n_policies": len(result.local_scores),
        "d_proxy": result.d_proxy,
        "d_proxy_ci_lower": result.d_proxy_ci[0],
        "d_proxy_ci_upper": result.d_proxy_ci[1],
        "d_proxy_monetary": result.d_proxy_monetary,
        "rag": result.rag,
        "shapley_effects": shapley_data,
        "benchmarks": {
            "mean_unaware": float(result.benchmarks.unaware.mean()),
            "mean_aware": float(result.benchmarks.aware.mean()),
            "mean_proxy_vulnerability": float(
                result.benchmarks.proxy_vulnerability.mean()
            ),
        },
        "local_scores_summary": {
            "green_count": int((result.local_scores["rag"] == "green").sum()),
            "amber_count": int((result.local_scores["rag"] == "amber").sum()),
            "red_count": int((result.local_scores["rag"] == "red").sum()),
        },
        "methodology": {
            "admissible_price": "Within-S-group exposure-weighted mean (Lindholm et al. 2022)",
            "d_proxy_formula": "sqrt(E[(h-h*)^2]) / sqrt(E[h^2])",
            "shapley_estimator": "Owen (2014) random permutation",
            "n_perms": result.n_perms,
            "surrogate": "RandomForestRegressor(n_estimators=100, max_depth=6)",
            "aware_benchmark": "Distillation refit with S, then S marginalised",
        },
    }

    return json.dumps(report, indent=2, default=str)


# Import polars here to keep top-level imports clean
import polars as pl  # noqa: E402
