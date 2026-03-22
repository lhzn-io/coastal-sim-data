# CoastalSim Data — AI Agent Instructions

## Mission

Microservice responsible for fetching, harmonizing, regridding, and serving real-time and historical coastal boundary conditions, initial conditions, and structural nudging telemetry to the `coastal-sim` Julia physics engine.

## Environment

- **Python 3.10+** — All fetcher and routing code is Python.
- **Dependency Management** — Uses `pyproject.toml` and `uv`. Use `uv run` to execute scripts and run standard tooling in the environment.
- **Microservices** — FastAPI application serving data arrays (Zarr/NetCDF) and JSON telemetry to the simulation engine.

## Architecture

### Core Pipeline

```text
HTTP Request → main.py (FastAPI) → dispatcher.py → [fetchers/*.py] → regridder.py → Zarr/JSON Response
```

- **Entry point**: `service/coastal_data_serve/main.py` — The FastAPI application.
- **Dispatcher**: `src/coastal_sim_data/dispatcher.py` — Interprets incoming requests (e.g., station IDs like "WLIS") and routes them to the appropriate fetcher module.
- **Fetchers**: `src/coastal_sim_data/fetchers/` — Contains isolated modules for specific data sources (e.g., `erddap.py` for UConn WLIS profiling, `era5.py`, etc.).
- **Regridder**: `src/coastal_sim_data/regridder.py` — Handles spatial interpolation for 2D/3D fields before transmitting to `coastal-sim`.

### Key Dependencies

| Package | Purpose |
|---|---|
| `FastAPI` | HTTP API framework |
| `xarray` / `pandas` | Data manipulation and I/O |
| `Zarr` | Chunked N-dimensional array output |
| `pytest` / `pytest-mock` | Unit and integration testing |
| `Sphinx` | Documentation generation |
| `ruff` | Fast Python linting and formatting |

### Validation & Testing

- **Unit Tests**: `tests/unit/` — Mock external API calls. Ensure routing and transformations are tested without network IO (e.g., `test_dispatcher.py`).
- **Integration Tests**: `tests/integration/` — Live tests against external APIs (e.g., `test_erddap_fetch.py`).
- **Execution**: Run tests strictly using `uv run pytest`.

## Conventions

- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`). No AI attribution footers. **CRITICAL: NEVER execute a commit without first presenting the proposed commit message and change set to the user and receiving explicit, positive confirmation.**
- **Git**: Stage files explicitly (`git add <file>`), NEVER use `git add .` to prevent bleeding context.
- **Formatting Constraints**: Run `uv run pre-commit run --files <modified-files>` (which triggers `ruff` and `ruff-format`) before committing any changes.
- **Documentation**: Updates to Sphinx docs in `docs/source/` (e.g., `fetchers.rst`) are required when adding new data provenance or API endpoints.

## Gotchas

- **Telemetry Performance**: Use `.csvp` endpoints when pulling tabular time-series (like ERDDAP station profiles) to avoid heavy NetCDF binary parsing overhead.
- **Mocking Strategy**: When modifying test assertions on mocked fetchers, closely track the keyword-arguments versus dictionary returns. The dispatcher tests check tuple/dict boundaries carefully.
- **Grid Normalization**: `coastal-sim` expects rigorous adherence to standard units (Elevations/Water Levels are positive up). Ensure external dataset quirks are normalized in the fetcher tier before entering the dispatcher.
