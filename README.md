# coastal-sim-data

`coastal-sim-data` is a highly-optimized data acquisition and transformation
service designed to provide boundary and initial conditions for 3D hydrodynamic
coastal simulations.

It acts as a caching and regridding intermediary between large-scale operational
forecasting API endpoints (NOAA, Copernicus ERA5, HRRR, HYCOM, and IOOS
Regional Nodes) and local high-fidelity Navier-Stokes solvers like
`Oceananigans.jl`.

Supported OBC/IC Models:

- **NYOFS** (NOAA NY/NJ Operational Forecast System) — 70–150m
  resolution, primary for NY Harbor
- NECOFS (FVCOM) — 200m resolution for New England
- HYCOM — Global 9km fallback
- HRRR — Atmospheric forcing
- ERA5 — Historical atmospheric conditions

## Architecture overview

The service exposes HTTP APIs (via FastAPI) which:

1. Deterministically hash bounding boxes and time windows to manage an automated
   local Zarr cache (`~/.cache/coastal-sim-data`).
2. Dispatch fetch operations via tiered fallbacks (e.g., HRRR -> ERA5T -> ERA5
   for atmospheric conditions, **NYOFS -> NECOFS -> HYCOM** for initial ocean
   states, and ERDDAP for structural telemetry and nudging profiles).
3. Regrid heterogeneous GRIB/NetCDF outputs to CF-compliant `xarray` Zarr stores
   for seamless consumption.

## Setup

```bash
uv sync
pre-commit install
uv run uvicorn service.coastal_data_serve.main:app --port 9598
```

## Docker

The `Dockerfile` defaults to `ubuntu:24.04` (x86_64).
Override `BASE_IMAGE` for other targets:

```bash
# x86_64 (default)
docker build -t coastal-sim-data .

# NVIDIA Jetson AGX Orin (JetPack 6.x)
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r36.2.0 \
  -t coastal-sim-data .

# Run the service
docker run --rm -p 9598:9598 coastal-sim-data
```

Validated platforms: x86_64 (Ubuntu 24.04) and Jetson AGX Orin
(JetPack 6.1 / L4T r36.2).

## Licensing

This project is licensed under the Apache 2.0 License. See the `LICENSE` file
for details.
