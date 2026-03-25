# Contributing to insurance-fairness

This library is built for UK pricing actuaries working on FCA Consumer Duty and Equality Act compliance. Contributions that make it more useful in that context are welcome.

## Reporting bugs

Open a GitHub Issue. Include:

- The Python and library version (`import insurance_fairness; print(insurance_fairness.__version__)`)
- A minimal reproducible example — ideally using the synthetic data generator bundled with the library
- What you expected to happen and what actually happened

If the bug involves unexpected fairness metric values, include the model type (GLM, GBM), the link function, and whether you are working on frequency, severity, or combined.

## Requesting features

Open a GitHub Issue with the label `enhancement`. Describe the pricing or compliance problem you are trying to solve, not just the feature you want. The most useful feature requests explain: what decision are you making, what data you have, and what the current tool cannot do.

Current gaps we know about: commercial lines support, home insurance-specific metrics, and protected characteristics beyond ethnicity and gender. If you are working in those areas, your use case is directly relevant.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-fairness.git
cd insurance-fairness
uv sync --dev
uv run pytest
```

That is it. The library uses `uv` for dependency management and `hatchling` as the build backend. Python 3.10+ is required.

To run the full test suite including slow proxy detection tests:

```bash
uv run pytest --run-slow
```

## Code style

- Type hints on all public functions and methods
- UK English in docstrings and documentation (e.g., "optimise" not "optimize", "licence" not "license" for the noun)
- Docstrings follow NumPy format
- No line length limit enforced, but keep it readable — 100 characters is a reasonable guide
- Tests go in `tests/` and should use the synthetic data generators so they do not depend on external data

If you are adding a new fairness metric, add a docstring that cites the source (paper, FCA guidance, or Equality Act section) and explains the interpretation in plain language a pricing actuary would use.

---

For questions or to discuss ideas before opening an issue, start a [Discussion](https://github.com/burning-cost/insurance-fairness/discussions).
