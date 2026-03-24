import xarray as xr
import pandas as pd
import requests
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# NERACOOS THREDDS Catalog Endpoint
NERACOOS_CATALOG_URL = "https://data.neracoos.org/thredds/catalog/catalog.xml"
NERACOOS_DODS_BASE = "https://data.neracoos.org/thredds/dodsC/"


def find_latest_gyx_dataset() -> list[str]:
    """
    Parses the NERACOOS catalog to find all NOAA NWPS gyx dataset URL paths, sorted by recency.
    """
    try:
        resp = requests.get(NERACOOS_CATALOG_URL, timeout=10)
        resp.raise_for_status()

        # Regex to find all urlPaths for GYX models: urlPath="datasets/gyx/gyx_nwps_YYYY-MM-DDTHH.nc"
        pattern = r'urlPath="([^"]*gyx_nwps_[^"]*\.nc)"'
        matches = re.findall(pattern, resp.text)

        if not matches:
            logger.error("No NERACOOS GYX NWPS datasets found in catalog.")
            return []

        # Extract dates and sort to find the latest available execution
        valid_datasets = []
        for match in matches:
            # Extract the date part assuming the format gyx_nwps_YYYY-MM-DDTHH.nc
            date_match = re.search(r"gyx_nwps_(\d{4}-\d{2}-\d{2}T\d{2})", match)
            if date_match:
                date_str = date_match.group(1)
                try:
                    dt = pd.to_datetime(date_str, format="%Y-%m-%dT%H").tz_localize(
                        "UTC"
                    )
                    valid_datasets.append((dt, match))
                except Exception:
                    pass

        if not valid_datasets:
            logger.error("Could not parse dates from any GYX dataset URLs.")
            return []

        # Sort by datetime, descending
        valid_datasets.sort(key=lambda x: x[0], reverse=True)
        return [NERACOOS_DODS_BASE + ds_path for _, ds_path in valid_datasets]

    except Exception as e:
        logger.error(f"Failed to crawl NERACOOS catalog: {e}")
        return []


def fetch_neracoos_initial_conditions(
    target_date: str, bbox: list
) -> Optional[xr.Dataset]:
    """
    Fetches the 3D Z-level ocean state from the NERACOOS OPeNDAP server (NOAA NWPS gyx).

    Args:
        target_date: ISO 8601 datestring (e.g. "2026-03-04T00:00:00Z")
        bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Rough check for NERACOOS GYX bounds (NH / Southern Maine coast)
    if max_lat < 42.0 or min_lat > 44.5 or max_lon < -71.5 or min_lon > -69.5:
        logger.warning(
            f"Bounding box {bbox} is completely outside NERACOOS domain. Aborting fetch."
        )
        return None

    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    logger.info("Attempting to fetch initial conditions from NERACOOS NWPS (gyx)...")

    dataset_urls = find_latest_gyx_dataset()
    if not dataset_urls:
        return None

    ds = None
    for url in dataset_urls:
        try:
            # Test DODS metadata endpoint before passing to pydap/xarray to avoid hard hangs
            test_resp = requests.get(url + ".dds", timeout=5)
            test_resp.raise_for_status()

            logger.info(f"Connected to valid dataset: {url}")
            ds = xr.open_dataset(url, engine="pydap")
            break
        except Exception:
            # 404 means the dataset was deleted from THREDDS, continue to next
            continue

    if ds is None:
        logger.error(
            "All parsed NERACOOS NWPS (gyx) datasets returned HTTP 404 (Stale Catalog)."
        )
        return None

    try:
        # NERACOOS GYX coords (usually lon, lat)
        lon_var = "lon" if "lon" in ds.coords else "longitude"
        lat_var = "lat" if "lat" in ds.coords else "latitude"

        # Time Selection
        time_var = "time"
        try:
            # Select the nearest time snapshot
            ds_t = ds.sel({time_var: target_dt}, method="nearest")
        except KeyError:
            logger.error(f"Cannot find time {target_dt} in NERACOOS catalog.")
            return None

        # Determine if the grid is regular or curvilinear
        if len(ds_t[lon_var].shape) == 1:
            # Regular 1D grid
            logger.info("Slicing regular 1D NERACOOS grid...")
            ds_subset = ds_t.sel(
                {lon_var: slice(min_lon, max_lon), lat_var: slice(min_lat, max_lat)}
            )
        else:
            # Curvilinear 2D grid
            logger.info("Slicing curvilinear 2D NERACOOS grid via index bounds...")
            import numpy as np

            mask = (
                (
                    (ds_t[lon_var] >= min_lon)
                    & (ds_t[lon_var] <= max_lon)
                    & (ds_t[lat_var] >= min_lat)
                    & (ds_t[lat_var] <= max_lat)
                )
                .compute()
                .values
            )

            # Find index bounds where mask is true
            if not mask.any():
                logger.error("Bounding box mask is entirely empty for this dataset.")
                return None

            y_indices, x_indices = np.where(mask)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # ROMS grids use eta_rho, xi_rho (and eta_u, xi_v etc). We slice all related dimensions safely
            slice_dict = {}
            for dim in ds_t.dims:
                if "eta" in str(dim) or "y" in str(dim):
                    slice_dict[dim] = slice(y_min, y_max)
                elif "xi" in str(dim) or "x" in str(dim):
                    slice_dict[dim] = slice(x_min, x_max)

            ds_subset = ds_t.isel(slice_dict)

        # Extract the core state - NWPS is a wave/surge model so variables might be 'u', 'v', 'zeta' or wave components
        target_vars = [
            "temp",
            "salt",
            "u",
            "v",
            "zeta",
            "water_u",
            "water_v",
            "elevation",
            "hs",
            "dir",
            "tp",
        ]
        keep_vars = [v for v in target_vars if v in ds_t.data_vars]

        if not keep_vars:
            logger.error(
                f"No target variables found in NERACOOS dataset. Found: {list(ds_t.data_vars.keys())}"
            )
            return None

        ds_subset = ds_subset[keep_vars]

        logger.info("Executing NERACOOS OPeNDAP download for target domain bounds...")
        ds_subset = ds_subset.compute()

        logger.info(
            f"Successfully downloaded NERACOOS NWPS data. Dimensions: {ds_subset.dims}"
        )
        return ds_subset

    except Exception as e:
        logger.error(f"Failed to fetch from NERACOOS NWPS: {str(e)}")
        return None
