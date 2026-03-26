import numpy as np
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
        "domain_bbox": [-75.0, 39.0, -71.0, 42.0],
    }


def supports_bbox(bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Nyhops coverage rough approx
    if max_lat < 39.0 or min_lat > 42.0 or max_lon < -75.0 or min_lon > -71.0:
        return False
    return True


def _get_nyhops_url(target_dt: pd.Timestamp) -> str:
    """Resolve the best NYHOPS OPeNDAP URL for a given target datetime.

    Uses the operational forecast for recent dates (< 2 days old),
    otherwise falls back to the THREDDS archive.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if (now - target_dt).days < 2 and (now - target_dt).days >= 0:
        return "http://colossus.dl.stevens-tech.edu:8080/thredds/dodsC/latest/Complete_gcm_run.nc"
    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")
    date_str = target_dt.strftime("%Y%m%d")
    return f"http://colossus.dl.stevens-tech.edu:8080/thredds/dodsC/Archive/NYHOPS/{year}/{month}/nyhops_{date_str}.nc"


def _to_dap_url(url: str) -> str:
    """Convert http(s) URL to pydap dap2:// scheme."""
    return url.replace("https://", "dap2://").replace("http://", "dap2://")


def fetch_nyhops_initial_conditions(
    target_date: str,
    bbox: list[float],
) -> Optional[xr.Dataset]:
    """
    Fetches 3D Initial Conditions (u, v, temp, salt) from Stevens Institute NYHOPS THREDDS server.
    NYHOPS is a high-resolution regional coastal ocean model for the NY/NJ harbor and Long Island Sound.
    Returns None if the requested bounding box is outside the NYHOPS domain.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if max_lat < 39.0 or min_lat > 42.0 or max_lon < -74.5 or min_lon > -71.5:
        logger.warning(
            f"Bounding box {bbox} is completely outside NYHOPS domain. Aborting fetch."
        )
        return None

    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    dataset_url = _get_nyhops_url(target_dt)
    logger.info(f"Attempting to fetch initial conditions from NYHOPS: {dataset_url}")

    try:
        dap_url = _to_dap_url(dataset_url)
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


def fetch_nyhops_boundary_conditions(
    start_date: str, duration_hours: int, bbox: list[float]
) -> Optional[xr.Dataset]:
    """
    Fetches 4D ocean state (u, v) from Stevens NYHOPS over a time range,
    subsets to the request bbox, and returns a structured Dataset suitable
    for open boundary conditions.

    Output dims: (time, depth, eta, xi) matching the NECOFS OBC contract.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if not supports_bbox(bbox):
        logger.info(f"Bounding box {bbox} outside NYHOPS domain.")
        return None

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)
    end_dt = target_dt + pd.Timedelta(hours=duration_hours)

    dataset_url = _get_nyhops_url(target_dt)
    dap_url = _to_dap_url(dataset_url)
    logger.info(f"Attempting to fetch OBC from NYHOPS: {dataset_url}")

    try:
        ds = xr.open_dataset(dap_url, engine="pydap")

        # Detect coordinate names (NYHOPS datasets vary)
        time_var = "time" if "time" in ds.coords else "ocean_time"
        lon_var = "lon" if "lon" in ds.coords else "lon_rho"
        lat_var = "lat" if "lat" in ds.coords else "lat_rho"

        # Time slicing — select the full requested range
        ds_t = ds.sel({time_var: slice(target_dt, end_dt)})
        if ds_t.sizes[time_var] == 0:
            # Fall back to nearest if exact slice is empty
            ds_t = ds.sel({time_var: target_dt}, method="nearest").expand_dims(time_var)
            logger.warning(
                "Exact time range empty, fell back to nearest single snapshot."
            )

        # Spatial subsetting via curvilinear mask
        mask = (
            (ds_t[lon_var] >= min_lon)
            & (ds_t[lon_var] <= max_lon)
            & (ds_t[lat_var] >= min_lat)
            & (ds_t[lat_var] <= max_lat)
        )
        ds_sub = ds_t.where(mask, drop=True)

        logger.info("Executing OPeNDAP download for NYHOPS OBC subset...")
        ds_sub = ds_sub.compute()

        # Detect sigma/depth dimension
        depth_dim = None
        for candidate in ["siglay", "s_rho", "depth", "sigma"]:
            if candidate in ds_sub.dims:
                depth_dim = candidate
                break

        if depth_dim is None:
            logger.error(
                f"No recognized depth dimension in NYHOPS dataset. Dims: {list(ds_sub.dims)}"
            )
            return None

        n_sigma = ds_sub.sizes[depth_dim]
        # Map sigma layers to pseudo-depth (matching NECOFS convention)
        depths = np.linspace(-50, 0, n_sigma).astype(np.float32)

        # Extract velocity arrays
        u_raw = ds_sub["u"].values
        v_raw = ds_sub["v"].values

        # Detect spatial dim names from the subset
        # After .where(drop=True) on curvilinear grid, remaining dims are
        # typically: (time_var, depth_dim, eta_dim, xi_dim)
        all_dims = list(ds_sub["u"].dims)
        spatial_dims = [d for d in all_dims if d != time_var and d != depth_dim]

        if len(spatial_dims) >= 2:
            eta_dim, xi_dim = spatial_dims[0], spatial_dims[1]
            eta_coords = np.arange(ds_sub.sizes[eta_dim], dtype=np.float32)
            xi_coords = np.arange(ds_sub.sizes[xi_dim], dtype=np.float32)
        else:
            logger.error(f"Unexpected spatial dims after subset: {spatial_dims}")
            return None

        out_times = ds_sub[time_var].values

        ds_out = xr.Dataset(
            data_vars={
                "u": (("time", "depth", "eta", "xi"), u_raw.astype(np.float32)),
                "v": (("time", "depth", "eta", "xi"), v_raw.astype(np.float32)),
            },
            coords={
                "time": out_times,
                "depth": depths,
                "eta": eta_coords,
                "xi": xi_coords,
            },
            attrs={
                "type": "Stevens NYHOPS OBC",
                "source": "Stevens Institute of Technology",
            },
        )

        logger.info(
            f"Successfully processed NYHOPS OBC data. "
            f"Shape: u{ds_out['u'].shape}, {len(out_times)} time steps"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to process NYHOPS OBC ({dataset_url}): {e}")
        return None
