import xarray as xr
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def process_and_regrid_grib(
    grib_path: str,
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
    # First try filtering by HRRR keys
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "stepType": "instant",
                    "level": 10,
                }
            },
        )
        if len(ds.data_vars) == 0:
            raise ValueError("Empty dataset returned with HRRR filter keys.")
    except Exception as e:
        logger.debug(
            f"Failed to load with HRRR keys ({e}), attempting fallback for surface-level datasets (e.g. ERA5)..."
        )
        # Fallback for single-level ERA5
        ds = xr.open_dataset(grib_path, engine="cfgrib")

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
                time=t_vals,
            ),
        )

        # If the HRRR/ERA5 dataset has a 'time' dimension, we can interpolate
        # the tide data to match the meteorological time steps for simplicity.
        # Alternatively, we just add it as a separate variable and let Julia
        # handle the multi-rate interpolation. Julia's Zarr.jl handles this fine.

        # We'll interpolate for the MVP to ensure a consistent 'time' dimension
        # broadcastable to (time, lat, lon) if needed (though water_level is spaital uniform here).
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
