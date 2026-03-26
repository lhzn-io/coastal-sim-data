import xarray as xr
import pandas as pd
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def process_and_regrid_grib(
    grib_path: list[str] | str,
    bbox: list[float],
    output_zarr_path: str,
    tide_data: Optional[dict] = None,
) -> str:
    """
    Reads a native GRIB2 file using xarray and cfgrib, clips it to the bounding box,
    and saves it as a CF-compliant Zarr store. Optionally merges NOAA tide data.

    Args:
        grib_path: Path to the downloaded GRIB file.
        bbox: [North, West, South, East]
        output_zarr_path: Path to write the output Zarr store.
        tide_data: Optional JSON-parsed dictionary from NOAA CO-OPS API.
    """
    logger.info(f"Processing meteorological GRIB file: {grib_path}...")

    # Load GRIB
    paths = grib_path if isinstance(grib_path, list) else [grib_path]

    # Try filter strategies in order of specificity:
    #   1. HRRR surface (full GRIB2 with many variables)
    #   2. HRRR 10m wind (wind-only GRIB2 from selective .idx fetch)
    #   3. Unfiltered fallback (ERA5 or other sources)
    filter_strategies: list[tuple[str, dict[str, Any]]] = [
        (
            "HRRR surface",
            {"filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"}},
        ),
        (
            "HRRR 10m wind",
            {"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10}},
        ),
        ("unfiltered fallback", {}),
    ]

    ds = None
    for strategy_name, backend_kwargs in filter_strategies:
        try:
            datasets = []
            for gp in paths:
                d = xr.open_dataset(gp, engine="cfgrib", backend_kwargs=backend_kwargs)
                # Drop scalar coords that vary per file (forecast step, reference time)
                # to prevent concat conflicts. valid_time is the true timestamp.
                drop_coords = [
                    c for c in ["time", "step"] if c in d.coords and d[c].dims == ()
                ]
                if drop_coords:
                    d = d.drop_vars(drop_coords)
                datasets.append(d)

            if "valid_time" in datasets[0].coords:
                datasets.sort(key=lambda d: d.valid_time.values)

            dim = "valid_time" if "valid_time" in datasets[0].coords else "time"
            ds = xr.concat(datasets, dim=dim, coords="minimal")
            if "valid_time" in ds.dims or "valid_time" in ds.coords:
                ds = ds.rename({"valid_time": "time"})
            if len(ds.data_vars) == 0:
                raise ValueError(f"Empty dataset with {strategy_name} filter.")
            logger.info(
                f"Loaded GRIB with '{strategy_name}' strategy: {list(ds.data_vars)}"
            )
            break
        except Exception as e:
            logger.debug(f"Filter '{strategy_name}' failed: {e}")
            continue

    if ds is None:
        raise RuntimeError(f"Could not load any variables from GRIB file(s): {paths}")
    # Rename coords if they use standard GRIB aliases
    rename_dict = {}
    if "lon" in ds.coords:
        rename_dict["lon"] = "longitude"
    if "lat" in ds.coords:
        rename_dict["lat"] = "latitude"
    if rename_dict:
        ds = ds.rename(rename_dict)

    north, west, south, east = bbox

    # Normalize longitude if necessary
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

    if "y" in ds.dims and "x" in ds.dims:
        logger.info("2D (y, x) topology detected. Filtering by mask...")
        mask = (
            (ds.latitude >= south - 0.3)
            & (ds.latitude <= north + 0.3)
            & (ds.longitude >= west - 0.3)
            & (ds.longitude <= east + 0.3)
        )
        ds_cropped = ds.where(mask, drop=True)
    else:
        logger.info("1D (lat, lon) topology detected. Slicing coordinate vectors...")
        if ds.longitude.max() > 180:
            ds = ds.sortby("longitude")
        if ds.latitude[0] > ds.latitude[-1]:
            ds = ds.sel(latitude=slice(None, None, -1))

        lat_slice = slice(south - 0.3, north + 0.3)
        lon_slice = slice(west - 0.3, east + 0.3)
        ds_cropped = ds.sel(latitude=lat_slice, longitude=lon_slice)

    # 2. Merge NOAA Tide Data if available
    if tide_data and "data" in tide_data:
        logger.info("Merging NOAA tide data into forcing tensors...")

        # Parse NOAA timeseries
        t_vals = []
        h_vals = []
        for entry in tide_data["data"]:
            # NOAA format: "2026-03-05 12:00"
            t_dt = pd.to_datetime(entry["t"])
            h_val = float(entry["v"])
            t_vals.append(t_dt)
            h_vals.append(h_val)

        tide_ds = xr.Dataset(
            data_vars=dict(
                water_level=(["time"], h_vals),
            ),
            coords=dict(
                time=("time", pd.to_datetime(t_vals).values.astype("datetime64[us]")),
            ),
        )

        # Cast the meteorological time coordinate to microseconds to match
        # and prevent xarray extrapolation bounds from exploding due to ns vs s type discrepancies.
        ds_cropped = ds_cropped.assign_coords(
            time=ds_cropped.time.astype("datetime64[us]")
        )

        # We'll interpolate for the MVP to ensure a consistent 'time' dimension
        # broadcastable to (time, lat, lon) if needed (though water_level is spaital uniform here).
        # We use fill_value="nearest" or omit extrapolate to prevent runaway tsunami gradients
        tide_interp = tide_ds.interp(
            time=ds_cropped.time, method="linear", kwargs={"fill_value": "extrapolate"}
        )

        # Add to main dataset
        ds_cropped["water_level"] = tide_interp.water_level

    logger.info(
        f"Saving forcing tensors to Zarr cache (V2 format for Julia): {output_zarr_path}"
    )
    ds_cropped.to_zarr(output_zarr_path, mode="w", zarr_format=2)

    return output_zarr_path
