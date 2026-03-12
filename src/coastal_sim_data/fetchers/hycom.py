import xarray as xr
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_hycom_initial_conditions(
    target_date: str,
    bbox: list[float],
) -> Optional[xr.Dataset]:
    """
    Fetches 3D Initial Conditions (u, v, temp, salt) from the HYCOM Global Ocean Forecasting System.
    HYCOM (GLBy0.08) runs at a 1/12 degree equatorial resolution (~9km). It serves as the
    global fallback when regional high-fidelity models (like NYHOPS or NGOFS) are unavailable.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Normalize longitudes for HYCOM (0 to 360) rather than (-180 to 180)
    # Most HYCOM servers use 0-360 standard eastward.
    hycom_min_lon = min_lon if min_lon >= 0 else 360 + min_lon
    hycom_max_lon = max_lon if max_lon >= 0 else 360 + max_lon

    # Handle bounding box crossing the prime meridian backward
    if hycom_min_lon > hycom_max_lon:
        logger.warning(
            f"Bounding box {bbox} crosses longitude 0 or 180 boundary awkwardly for HYCOM. Results may be discontinuous."
        )

    target_dt = pd.to_datetime(target_date)

    # Base HYCOM THREDDS OPeNDAP Endpoint
    # GLBy0.08/expt_93.0 is the latest comprehensive global GOFS 3.1 run
    # https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0

    # HYCOM endpoints often split by variable to save time on the master catalog
    dataset_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"

    logger.info(
        f"Attempting to fetch global initial conditions from HYCOM: {dataset_url}"
    )

    try:
        # Open lazily. HYCOM THREDDS handles slice optimizations very well because coords are 1D.
        # We use pydap because netcdf4's C-bindings hang on massive global coordinate grids via HTTP
        # decode_times=False prevents "hours since analysis" errors on non-compliant forecast variables
        ds = xr.open_dataset(dataset_url, engine="pydap", decode_times=False)

        # HYCOM natively maps variables:
        # water_u, water_v, water_temp, salinity

        # Find nearest time snapshot manually (decode_times=False turns time into a float array)
        # HYCOM time axis is typically "hours since 2000-01-01 00:00:00"
        time_var = "time"
        try:
            target_hours = (
                target_dt - pd.Timestamp("2000-01-01 00:00:00").tz_localize("UTC")
            ).total_seconds() / 3600.0
            ds_t = ds.sel({time_var: target_hours}, method="nearest")
        except KeyError:
            logger.error(
                f"Cannot find time {target_dt} in HYCOM catalog. Available range: {ds[time_var].values[0]} to {ds[time_var].values[-1]}"
            )
            return None

        # Slice spatially using standard 1D index slicing - significantly faster via OPeNDAP than `.where()`
        lon_var = "lon" if "lon" in ds.coords else "longitude"
        lat_var = "lat" if "lat" in ds.coords else "latitude"

        # OPeNDAP subset slice request
        logger.info(
            f"Subsetting HYCOM OPeNDAP grid: lat [{min_lat}, {max_lat}], lon [{hycom_min_lon}, {hycom_max_lon}]"
        )
        ds_subset = ds_t.sel(
            {
                lat_var: slice(min_lat - 0.1, max_lat + 0.1),
                lon_var: slice(hycom_min_lon - 0.1, hycom_max_lon + 0.1),
            }
        )

        # Execute the download over the network
        logger.info("Executing OPeNDAP download for HYCOM subset...")
        ds_subset = ds_subset.compute()

        logger.info(
            f"Successfully fetched HYCOM Initial Conditions. Shape: {ds_subset.water_u.shape}"
        )

        return ds_subset

    except Exception as e:
        logger.error(f"Failed to fetch from HYCOM ({dataset_url}): {e}")
        return None
