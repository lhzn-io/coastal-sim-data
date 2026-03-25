import xarray as xr
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_metadata() -> dict:
    return {
        "id": "nyhops",
        "name": "Stevens NYHOPS",
        "resolution_approx_m": 500.0,
        "type_desc": "Curvilinear grid",
    }


def supports_bbox(bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Nyhops coverage rough approx
    if max_lat < 39.0 or min_lat > 42.0 or max_lon < -75.0 or min_lon > -71.0:
        return False
    return True


def fetch_nyhops_initial_conditions(
    target_date: str,
    bbox: list[float],
) -> Optional[xr.Dataset]:
    """
    Fetches 3D Initial Conditions (u, v, temp, salt) from Stevens Institute NYHOPS THREDDS server.
    NYHOPS is a high-resolution regional coastal ocean model for the NY/NJ harbor and Long Island Sound.
    Returns None if the requested bounding box is outside the NYHOPS domain.
    """
    # NYHOPS bounds roughly cover the regional bight and estuaries
    min_lon, min_lat, max_lon, max_lat = bbox

    # Rough check for NYHOPS bounds avoiding unnecessary OPeNDAP requests
    if max_lat < 39.0 or min_lat > 42.0 or max_lon < -74.5 or min_lon > -71.5:
        logger.warning(
            f"Bounding box {bbox} is completely outside NYHOPS domain. Aborting fetch."
        )
        return None

    # Parse target date and enforce timezone naivety (UTC base)
    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")

    # NYHOPS THREDDS OPeNDAP Endpoint
    # We attempt to hit the archive or the latest forecast depending on the date
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    dataset_url = ""
    if (now - target_dt).days < 2 and (now - target_dt).days >= 0:
        # Recent/Forecast (use the operational run)
        dataset_url = "http://colossus.dl.stevens-tech.edu:8080/thredds/dodsC/latest/Complete_gcm_run.nc"
    else:
        # NYHOPS Archive
        # Format: http://colossus.dl.stevens-tech.edu:8080/thredds/dodsC/Archive/2026/03/nyhops_parsed_20260304.nc (approximate)
        # Note: the exact pathing changes, this needs robust fallback checking in production,
        # but for this POC we will attempt a best-effort pattern or fallback to HYCOM.
        date_str = target_dt.strftime("%Y%m%d")
        dataset_url = f"http://colossus.dl.stevens-tech.edu:8080/thredds/dodsC/Archive/NYHOPS/{year}/{month}/nyhops_{date_str}.nc"

    logger.info(f"Attempting to fetch initial conditions from NYHOPS: {dataset_url}")

    try:
        # Open the dataset lazily with pydap (netcdf4 C-bindings can hang on OPeNDAP coords)
        # pydap expects dap2:// or dap4:// instead of http:// or https://
        dap_url = dataset_url.replace("https://", "dap2://").replace(
            "http://", "dap2://"
        )
        ds = xr.open_dataset(dap_url, engine="pydap")

        # NYHOPS native variables:
        # u (eastward_sea_water_velocity), v (northward_sea_water_velocity)
        # temp (sea_water_temperature), salt (sea_water_salinity)

        # Time alignment
        # NYHOPS time might be 'time' or 'ocean_time'. We select the nearest snapshot to start_time.
        time_var = "time" if "time" in ds.coords else "ocean_time"
        ds_t = ds.sel({time_var: target_dt}, method="nearest")

        # Spatial Subsetting (we use lat/lon 2D arrays so we have to use where)
        # Creating a mask for the bounding box
        lon_var = "lon" if "lon" in ds.coords else "lon_rho"
        lat_var = "lat" if "lat" in ds.coords else "lat_rho"

        # Subsetting massive curvalinear grids over OPeNDAP is notoriously slow.
        # We slice index ranges natively if possible, but here we do a lazy drop.
        mask = (
            (ds_t[lon_var] >= min_lon)
            & (ds_t[lon_var] <= max_lon)
            & (ds_t[lat_var] >= min_lat)
            & (ds_t[lat_var] <= max_lat)
        )

        ds_subset = ds_t.where(mask, drop=True)

        # Load the subset into memory (triggers the OPeNDAP download)
        logger.info("Executing OPeNDAP download for NYHOPS subset...")
        ds_subset = ds_subset.compute()

        logger.info(
            f"Successfully fetched NYHOPS Initial Conditions. Shape: {ds_subset.u.shape}"
        )
        return ds_subset

    except Exception as e:
        logger.error(f"Failed to fetch from NYHOPS ({dataset_url}): {e}")
        return None
