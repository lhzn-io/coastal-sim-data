import xarray as xr
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_metadata() -> dict:
    return {
        "id": "maracoos",
        "name": "MARACOOS Rutgers DOPPIO",
        "resolution_approx_m": 7000.0,
        "type_desc": "ROMS structured grid",
    }


def supports_bbox(bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Rough check for DOPPIO bounds (MAB / LIS / Gulf of Maine)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        return False
    return True


# MARACOOS (Rutgers DOPPIO) OPeNDAP Endpoint
# Operational 7km resolution ROMS model with 40 vertical levels
# Covering the Mid-Atlantic Bight and Gulf of Maine
DOPPIO_THREDDS_URL = (
    "https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best"
)


def fetch_maracoos_initial_conditions(
    target_date: str, bbox: list
) -> Optional[xr.Dataset]:
    """
    Fetches the 3D Z-level ocean state from the MARACOOS (Rutgers DOPPIO) OPeNDAP server.

    Args:
        target_date: ISO 8601 datestring (e.g. "2026-03-04T00:00:00Z")
        bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Rough check for DOPPIO bounds (MAB / LIS / Gulf of Maine)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.warning(
            f"Bounding box {bbox} is completely outside MARACOOS DOPPIO domain. Aborting fetch."
        )
        return None

    # Parse target date and enforce timezone naivety (UTC base)
    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    logger.info("Attempting to fetch initial conditions from MARACOOS DOPPIO...")

    try:
        # Open the dataset lazily with pydap
        # pydap expects dap2:// or dap4:// instead of http:// or https://
        dap_url = DOPPIO_THREDDS_URL.replace("https://", "dap2://").replace(
            "http://", "dap2://"
        )
        ds = xr.open_dataset(dap_url, engine="pydap")

        # DOPPIO/ROMS coords
        lon_var = "lon_rho" if "lon_rho" in ds.coords else "lon"
        lat_var = "lat_rho" if "lat_rho" in ds.coords else "lat"

        # Time Selection
        time_var = "time"
        try:
            # Select the nearest time snapshot
            ds_t = ds.sel({time_var: target_dt}, method="nearest")
        except KeyError:
            logger.error(f"Cannot find time {target_dt} in MARACOOS catalog.")
            return None
        except Exception as e:
            logger.error(f"Error accessing time dimension in MARACOOS catalog: {e}")
            return None

        # Spatial Subsetting
        # DOPPIO uses a curvilinear grid so lat/lon are 2D arrays (eta_rho, xi_rho).
        # Slicing natively is complex, so we drop via a mask.
        mask = (
            (ds_t[lon_var] >= min_lon)
            & (ds_t[lon_var] <= max_lon)
            & (ds_t[lat_var] >= min_lat)
            & (ds_t[lat_var] <= max_lat)
        )

        # We need to extract the rho variables (temp, salt) and maybe u, v which are on u, v grids
        # But for simplification and Oceananigans initial conditions, we extract the core state
        target_vars = ["temp", "salt", "u", "v", "zeta"]
        keep_vars = [v for v in target_vars if v in ds_t.data_vars]

        if not keep_vars:
            logger.error(
                "No target variables (temp, salt, u, v) found in DOPPIO dataset."
            )
            return None

        ds_subset = ds_t[keep_vars].where(mask, drop=True)

        # Load the subset into memory (triggers the OPeNDAP download)
        logger.info("Executing MARACOOS OPeNDAP download for target domain bounds...")
        ds_subset = ds_subset.compute()

        logger.info(
            f"Successfully downloaded MARACOOS DOPPIO data. Dimensions: {ds_subset.dims}"
        )
        return ds_subset

    except Exception as e:
        logger.error(f"Failed to fetch from MARACOOS DOPPIO: {str(e)}")
        return None
