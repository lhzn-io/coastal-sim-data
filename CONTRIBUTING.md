# Contributing to coastal-sim-data

First off, thank you for considering contributing to `coastal-sim-data`!
This repository is meant to be a community resource for providing clean,
standardized ocean and atmospheric forcing boundaries for high-fidelity coastal
hydrodynamic models.

## Development Workflow

### 1. Environment Setup

We use `uv` for lightning-fast Python package management.

1. Install `uv`.
2. Sync the environment: `uv sync`
3. We use `pre-commit` to ensure code format standardisation. Run
   `pre-commit install` to set up your git hooks.

### 2. Making Changes

- All code must pass `ruff` (for formatting and linting) and `mypy` (for typing).
- Ensure your changes are covered by tests where applicable (we use `pytest`).
  Tests are located in the `tests/` directory.
- `coastal_sim_data.fetchers` logic handles API integrations. If adding a new
  telemetry source (e.g., a new regional IOOS node), follow the patterns
  established in `maracoos.py` or `neracoos.py`.
- Keep the Sphinx documentation up to date!

### 3. Pull Requests

1. Create a descriptive branch name: `git checkout -b feature/ioos-integration`
2. Make your commits clear and logical.
3. Open a Pull Request! A maintainer will review your code.

### 4. Code of Conduct

Please be respectful and patient with fellow collaborators. We look forward to
your contributions!
