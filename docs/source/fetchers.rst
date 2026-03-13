Data Fetchers
=============

The Data Fetchers module contains implementations for communicating with external data providers. We use tiered fallback logic and cache everything locally to minimize bandwidth usage and guarantee reproducible simulations.

Available Sources
-----------------

Atmospheric Forcing:
- **HRRR**: High-Resolution Rapid Refresh (AWS S3)
- **ERA5T / ERA5**: ECMWF Copernicus Reanalysis

Ocean Boundary & Initial Conditions:
- **NERACOOS** / **MARACOOS**: Regional IOOS endpoints
- **HYCOM**: Global fallback

Point Telemetry & Nudging:
- **ERDDAP**: Support for tabular/point-source telemetry (e.g. 3-depth temperature profiles) from platforms like the UConn ERDDAP server. Bypasses bulky NetCDF dependencies by utilizing the ``.csvp`` endpoints with ``pandas``.
