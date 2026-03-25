import warnings
import logging
import pandas as pd
import numpy as np
import xarray as xr
from typing import Optional
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

logger = logging.getLogger(__name__)


def get_metadata() -> dict:
    return {
        "id": "necofs",
        "name": "NECOFS FVCOM GOM7",
        "resolution_approx_m": 200.0,
        "type_desc": "Unstructured Triangular Mesh",
        "domain_bbox": [-77.0, 35.0, -65.0, 46.0],
    }


def supports_bbox(bbox: list[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Rough check for GOM3 bounds (NECOFS FVCOM)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        return False
    return True


NECOFS_GOM7_URL = "http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Forecasts/NECOFS_GOM7_FORECAST.nc"


def get_necofs_url(target_dt: pd.Timestamp) -> str:
    """
    Determine best NECOFS GOM7 URL by falling back to daily history archives.
    SMAST daily archives (e.g. 2026_03_03.nc) contain [Mar 2 01:00 to Mar 3 00:00].
    """
    import requests

    if target_dt == target_dt.normalize():
        file_dt = target_dt
    else:
        file_dt = target_dt.normalize() + pd.Timedelta(days=1)

    date_str = file_dt.strftime("%Y_%m_%d")
    history_url = f"http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/necofs_history/NECOFS_GOM7_{date_str}.nc"

    try:
        resp = requests.get(history_url + ".dds", timeout=5)
        if resp.status_code == 200:
            logger.info(f"Using historic NECOFS GOM7 archive: {history_url}")
            return history_url
    except requests.RequestException:
        pass

    logger.info("Falling back to NECOFS GOM7 rolling forecast.")
    return NECOFS_GOM7_URL


def fetch_necofs_initial_conditions(
    target_date: str, bbox: list
) -> Optional[xr.Dataset]:
    """
    Fetches the 3D unstructured ocean state from NECOFS (GOM3),
    and regrids it to a structured Z-level grid suitable for Oceananigans.

    Args:
        target_date: ISO 8601 datestring (e.g. "2026-03-04T00:00:00Z")
        bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Rough check for GOM3 bounds (NECOFS FVCOM)
    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.info(
            f"Bounding box {bbox} is completely outside NECOFS GOM3 domain. Aborting fetch."
        )
        return None

    # Enforce UTC timezone naivety
    target_dt = pd.to_datetime(target_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    logger.info("Attempting to fetch initial conditions from NECOFS GOM3...")

    try:
        # Determine the correct URL based on target date (history archive vs rolling forecast)
        dap_url = get_necofs_url(target_dt)

        # Pydap might hang on SMAST, so we should rely on dispatcher catching/logging it,
        # or we could use requests to check if it's alive first.
        import requests

        try:
            # Quick alive check
            resp = requests.get(dap_url + ".dds", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"NECOFS server appears unreachable at {dap_url}: {e}")
            return None

        # Open the dataset
        # We must disable decode_times because GOM7 contains broken Itime/Itime2 variables
        # specifying "msec since 00:00:00" which cftime cannot parse.
        # We load without decoding, drop them, and decode the rest properly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
        drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
        ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))

        # Extract unstructured coordinate variables
        lon = ds["lon"].values
        lat = ds["lat"].values
        lonc = ds["lonc"].values
        latc = ds["latc"].values

        # FVCOM nv is 1-indexed, shape can be (nele, 3) or (3, nele)
        # Often it is (3, nele). We transpose to (nele, 3) and subtract 1.
        nv_raw = ds["nv"].values
        if nv_raw.shape[0] == 3:
            nv = nv_raw.T - 1
        else:
            nv = nv_raw - 1

        _h = ds["h"].values  # noqa: F841 — needed for future depth mapping
        _siglay = ds["siglay"].values  # noqa: F841 — shape: (siglay, node)

        # Check nodes within bbox + buffer
        buffer = 0.05
        node_mask = (
            (lon >= min_lon - buffer)
            & (lon <= max_lon + buffer)
            & (lat >= min_lat - buffer)
            & (lat <= max_lat + buffer)
        )

        if not np.any(node_mask):
            logger.warning("No NECOFS nodes found within the padded bounding box.")
            return None

        # Temporal sub-selection
        try:
            ds_t = ds.sel(time=target_dt, method="nearest")
        except KeyError:
            logger.error(f"Cannot find time {target_dt} in NECOFS GOM3.")
            return None

        # Load required fields into memory (for the region or full domain depending on size)
        # Fetching full domain of one timestep might be faster/easier than fancy indexing with pydap
        logger.info("Downloading variables from NECOFS...")
        try:
            # We explicitly need node variables: temp, salinity, zeta
            # And element variables: u, v (and ww if available, but we'll stick to u, v)
            vars_to_get = ["temp", "salinity", "zeta", "u", "v"]
            ds_sub = ds_t[vars_to_get].compute()
            zeta = ds_sub["zeta"].values
            temp = ds_sub["temp"].values
            salt = ds_sub["salinity"].values
            u = ds_sub["u"].values
            v = ds_sub["v"].values
        except Exception as e:
            logger.error(f"Failed to extract variables from NECOFS: {e}")
            return None

        # Determine dimensions of regular grid
        # 0.002 degrees ~ roughly 200 meters resolution matching coastal NECOFS
        d_spacing = 0.002
        lon_rho = np.arange(min_lon, max_lon, d_spacing)
        lat_rho = np.arange(min_lat, max_lat, d_spacing)
        xi_rho = np.arange(len(lon_rho))
        eta_rho = np.arange(len(lat_rho))

        # Create target structured meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_rho, lat_rho)

        logger.info("Building explicit FVCOM triangulations for regridding...")
        import matplotlib.tri as mtri

        # Node-based triangulation mapping exactly to physical bounds
        fvcom_tri = mtri.Triangulation(lon, lat, triangles=nv)

        # Element-based triangulation (for U, V, unconstrained convex hull)
        pts_elem = np.column_stack((lonc, latc))
        tri_elem = Delaunay(pts_elem)

        logger.info("Interpolating variables to structured grid...")

        # Helper for node interpolation strictly inside real mesh boundaries
        def interpolate_nodes(field_vals):
            # Returns a numpy MaskedArray, use filled(nan) to return normal floats
            interp = mtri.LinearTriInterpolator(fvcom_tri, field_vals)
            return interp(lon_grid, lat_grid).filled(np.nan)

        # Helper for element interpolation
        def interpolate_elems(field_vals, strict_mask):
            interp = LinearNDInterpolator(tri_elem, field_vals)
            val = interp(lon_grid, lat_grid)
            # Clip unconstrained elements convex hull using the strict node boundary mask
            val[strict_mask] = np.nan
            return val

        # 2D field
        zeta_interp = interpolate_nodes(zeta)
        land_mask = np.isnan(zeta_interp)

        s_rho_dim = ds_t.dims.get("siglay", 45)
        s_rho = np.linspace(-1, 0, s_rho_dim)

        # Preallocate 3D arrays
        # shape: (s_rho, eta_rho, xi_rho)  - matching our target schema conventions
        ny, nx = len(lat_rho), len(lon_rho)
        nz = s_rho_dim

        temp_out = np.zeros((nz, ny, nx), dtype=np.float32)
        salt_out = np.zeros((nz, ny, nx), dtype=np.float32)
        u_out = np.zeros((nz, ny, nx), dtype=np.float32)
        v_out = np.zeros((nz, ny, nx), dtype=np.float32)

        for k in range(nz):
            # node based
            t_k = interpolate_nodes(temp[k, :])
            s_k = interpolate_nodes(salt[k, :])
            # elem based
            u_k = interpolate_elems(u[k, :], land_mask)
            v_k = interpolate_elems(v[k, :], land_mask)

            temp_out[k, :, :] = t_k
            salt_out[k, :, :] = s_k
            u_out[k, :, :] = u_k
            v_out[k, :, :] = v_k

        # Assemble Output Dataset matching general ROMS/DOPPIO output layout
        # (Variables: u, v, temp, salt, zeta, s_rho, lon_rho, lat_rho)

        ds_out = xr.Dataset(
            data_vars={
                "temp": (("s_rho", "eta_rho", "xi_rho"), temp_out),
                "salt": (("s_rho", "eta_rho", "xi_rho"), salt_out),
                "u": (("s_rho", "eta_rho", "xi_rho"), u_out),
                "v": (("s_rho", "eta_rho", "xi_rho"), v_out),
                "zeta": (("eta_rho", "xi_rho"), zeta_interp),
            },
            coords={
                "s_rho": s_rho,
                "eta_rho": eta_rho,
                "xi_rho": xi_rho,
                "lon_rho": (("eta_rho", "xi_rho"), lon_grid),
                "lat_rho": (("eta_rho", "xi_rho"), lat_grid),
            },
            attrs={"type": "NECOFS/FVCOM GOM3", "source": "UMass Dartmouth SMAST"},
        )

        logger.info(
            f"Successfully processed NECOFS data. Target grid dims: {ds_out.dims}"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to fetch or process NECOFS data: {str(e)}")
        return None
