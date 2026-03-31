# Changelog

## v0.9.0 (2026-03-31)
- feat: add SequentialOTCorrector — fixes calibration bug in WassersteinCorrector
  for K >= 2 protected attributes. WassersteinCorrector fitted all attribute ECDFs
  on f* then applied them sequentially; this is wrong because the OT map for
  attribute k is applied to f_{k-1} which has a different distribution from f*.
  SequentialOTCorrector calibrates each step's ECDF on f_{k-1}, making the OT map
  well-specified. For K=1 the two classes are identical. API: fit/transform follow
  WassersteinCorrector conventions. Per-attribute epsilon list supported.
  get_intermediate_predictions() returns [f_0, ..., f_K] for diagnostics.
  unfairness_reductions_ and wasserstein_distances_ properties for reporting.
  group_min_samples warning when group sizes are too small for reliable ECDF.
  55 tests in test_ot_sequential_corrector.py.
- feat: add correction='sequential_wasserstein' and 'lindholm+sequential_wasserstein'
  options to DiscriminationFreePrice. These wire SequentialOTCorrector into the main
  pricing pipeline. 'lindholm+sequential_wasserstein' is the recommended option when
  both conditional fairness and demographic parity are required.
- docs: add calibration bug warning to WassersteinCorrector docstring.

## v0.6.6 (2026-03-26)
- fix: remove `scikit-learn<1.6` upper cap — this constraint blocked co-installation
  with insurance-causal and other stack libraries that require sklearn>=1.6. Audit of
  all source files confirmed no usage of `check_X_y(force_all_finite=...)` or any
  other API removed in sklearn 1.6. The cap was added defensively but was never
  required. New constraint: `scikit-learn>=1.5`. This unblocks full Burning Cost stack
  installation in a single environment.

## v0.6.5 (2026-03-25)
- fix: IndexError in _build_summary where wmean closure applied full-dataset
  boolean mask to already-sliced group arrays. The wmean helper was defined
  inside the per-group loop and captured `mask` via closure, then applied
  `arr[mask]` to callers that were already passing group-sliced arrays
  (h_A_g, h_U_g, etc.). Fixed by passing w_g and gw as default arguments so
  wmean operates directly on the pre-sliced array without re-indexing.
  20 tests in TestBasicFit, TestProxyVulnerabilityFormula, TestProxyFreeModel,
  TestExposureWeighting, TestSegmentReport, TestCustomModel, TestDeterminism
  now passing. 26 tests total in test_indirect.py all pass.

## v0.6.4 (2026-03-25)
- feat: add IndirectDiscriminationAudit — end-to-end partition-based audit of
  indirect discrimination implementing the five benchmark premiums from Côté,
  Côté & Charpentier (CAS Working Paper, October 2025). Fits aware (h_A),
  unaware (h_U), unawareness (h_UN), proxy-free (h_PV), and parity-cost (h_C)
  models from raw training data. Computes proxy vulnerability = mean |h_U - h_A|
  per segment, parity-cost gap, and unawareness gap. No causal graph required.
  LightGBM as optional default model with GradientBoostingRegressor fallback.
  Exposure-weighted metrics throughout. 23 tests.

## v0.6.3 (2026-03-25)
- feat: add DiscriminationInsensitiveReweighter — KL divergence-minimising sample
  reweighting for discrimination-insensitive training (Miao & Pesenti 2026,
  arXiv:2603.16720). Propensity-based weights achieve X ⊥ A without removing the
  protected attribute. Supports logistic and random forest propensity models,
  weight clipping, and diagnostics including effective sample size.
  28 tests. Weights integrate with any sklearn sample_weight API.

## v0.6.0 (2026-03-22) [unreleased]
- Fix licence footer: BSD-3 was wrong, LICENSE file is MIT
- Benchmark fixes Round 3: rename, financial impact, Monte Carlo sensitivity
- docs: fix README review issues

## v0.6.0 (2026-03-21)
- Add cross-links to related libraries in README
- Add DoubleFairnessAudit benchmark notebook and README section
- fix(tests): correct test_pareto_delta2 to account for Delta_2 math
- feat: add DoubleFairnessAudit module (v0.6.0)
- docs: replace pip install with uv add in README
- docs: document v0.5.0 MarginalFairnessPremium
- feat: add MarginalFairnessPremium — v0.5.0
- docs: add ProxyVulnerabilityScore and ParityCost README section
- Add ProxyVulnerabilityScore and ParityCost (v0.4.0)
- Fix: pass DataFrame directly to model.predict() in _predict_best_estimate_both_groups
- docs(notebook): full ProxyVulnerabilityScore demo workflow
- fix: guard against overflow in proxy_vulnerability_pct when aware premium is zero
- feat: add ProxyVulnerabilityScore and ParityCost (v0.4.0)
- Reframe README with FCA Consumer Duty compliance-first positioning
- Add PrivatizedFairnessAudit: discrimination-free pricing with LDP privatisation
- Add Databricks test notebook for v0.3.8
- Add MulticalibrationAudit: calibration-aware fairness auditing (Denuit et al. 2026)
- bump: v0.3.7 — export detect_proxies at top level
- Export detect_proxies at top-level (add to __all__)
- fix: remove false EP25/2 citations (v0.3.6)
- docs(notebook): three presentation improvements for AIDSET demo
- fix: five correctness fixes from reviewer (v0.3.5)
- fix: update quickstart output example to match actual n=1,000 output
- fix: address 6 quality-review issues in AIDSET demo notebook
- fix: update license badge from BSD-3 to MIT
- Revise AIDSET demo: switch to continuous diversity_score DGP
- Add IFoA AIDSET talk demo notebook
- Add MIT license
- Add discussions link and star CTA
- Add benchmark results from live Databricks run to README
- Add PyPI classifiers for financial/insurance audience
- Add Google Colab quickstart notebook and Open-in-Colab badge
- Add CONTRIBUTING.md with bug reporting, feature request, and dev setup guidance
- refresh benchmark numbers post-P0 fixes
- fix: P0 bugs CRIT-1 through CRIT-4 (v0.3.4)
- Fix docs workflow: use pdoc not pdoc3 syntax (no --html flag)
- Add pdoc API documentation workflow with GitHub Pages deployment
- Add consulting CTA to README
- Add benchmark: proxy discrimination detection vs manual Spearman inspection
- pin statsmodels>=0.14.5 for scipy compat
- fix: handle CatBoost random_seed/use_best_model conflicts in _clone_model (v0.3.3)
- Polish flagship README: badges, benchmark table, problem statement
- docs: add Databricks notebook link
- fix: encode string protected columns before float conversion in proxy detection
- Add Related Libraries section to README
- fix: update cross-references to consolidated repos
- Add CITATION.cff for academic and software citation
- fix: replace external file references in quick-start with inline data
- docs: fix stale insurance-elasticity reference; add Performance section
- Add Capabilities Demo section to README
- Add Databricks benchmark: proxy discrimination demo
- Fix conftest.py: merge diag fixtures cleanly without syntax errors
- Update run_tests notebook for v0.3.0 dependencies
- Absorb insurance-fairness-ot and insurance-fairness-diag as subpackages
- fix: replace np.trapz with np.trapezoid for NumPy 2.0 compatibility
- Add Pareto optimisation demo notebook
- Add NSGA-II Pareto front optimisation for fairness-accuracy trade-offs (v0.2.0)

## v0.1.0 (2026-03-09)
- fix: replace np.trapz with np.trapezoid for NumPy 2.0 compatibility
- Add GitHub Actions CI workflow and test badge
- Add GitHub Actions CI workflow and test badge
- docs: README quality pass — fix URLs, correct blog link
- docs: README quality pass — fix URLs, add cross-references
- fix: replace pip with uv in README
- Add badges and cross-links to README
- Fix 8 test failures from initial Databricks run
- Initial release of insurance-fairness library
