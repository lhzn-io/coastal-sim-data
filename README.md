# coastal-sim-data

`coastal-sim-data` is a highly-optimized data acquisition and transformation
service designed to provide boundary and initial conditions for 3D hydrodynamic
coastal simulations (like `coastal-sim`).

It acts as a caching and regridding intermediary between large-scale operational
forecasting API endpoints (NOAA, Copernicus ERA5, HRRR, HYCOM, and IOOS
Regional Nodes) and local high-fidelity Navier-Stokes solvers like
`Oceananigans.jl`.

## Architecture overview

The service exposes HTTP APIs (via FastAPI) which:

1. Deterministically hash bounding boxes and time windows to manage an automated
   local Zarr cache (`~/.cache/coastal-sim-data`).
2. Dispatch fetch operations via tiered fallbacks (e.g., HRRR -> ERA5T -> ERA5
   for atmospheric conditions, NERACOOS -> MARACOOS -> HYCOM for initial ocean
   states).
3. Regrid heterogeneous GRIB/NetCDF outputs to CF-compliant `xarray` Zarr stores
   for seamless consumption.

## Setup

```bash
uv sync
pre-commit install
uv run uvicorn service.coastal_data_serve.main:app --port 9598
```

## Licensing

This project is licensed under the Apache 2.0 License. See the `LICENSE` file
for details.
