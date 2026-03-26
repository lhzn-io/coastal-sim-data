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

    # NECOFS GOM7 daily history files contain data for the PREVIOUS day up to 00:00 of the CURRENT day.
    # Therefore to get data forward-looking from target_dt, we must ALWAYS fetch the file for the NEXT day.
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

        s_rho_dim = ds_t.sizes.get("siglay", 45)
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
            f"Successfully processed NECOFS data. Target grid dims: {dict(ds_out.sizes)}"
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to fetch or process NECOFS data: {str(e)}")
        return None


def fetch_necofs_boundary_conditions(
    start_date: str, duration_hours: int, bbox: list[float]
) -> Optional[xr.Dataset]:
    """
    Fetches the 4D unstructured ocean state from NECOFS (GOM3/MASSBAY),
    and regrids it to a structured grid suitable for OBCs.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if max_lat < 35.0 or min_lat > 46.0 or max_lon < -77.0 or min_lon > -65.0:
        logger.info(f"Bounding box {bbox} outside NECOFS domain.")
        return None

    target_dt = pd.to_datetime(start_date)
    if target_dt.tzinfo is not None:
        target_dt = target_dt.tz_convert("UTC").tz_localize(None)

    # In a full implementation we'd merge archives if spans multiple days.
    # For now, let's grab the best single file.
    dap_url = get_necofs_url(target_dt)

    import requests

    try:
        requests.get(dap_url + ".dds", timeout=10).raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"NECOFS server unreachable: {e}")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_raw = xr.open_dataset(dap_url, engine="pydap", decode_times=False)
        drop_vars = [v for v in ["Itime", "Itime2"] if v in ds_raw.variables]
        ds = xr.decode_cf(ds_raw.drop_vars(drop_vars))

        ds_times = pd.DatetimeIndex(ds.time.values)
        if ds_times.tz is not None:
            ds_times = ds_times.tz_convert("UTC").tz_localize(None)

        # Slice time to requested bounds (with nearest if exact doesn't match)
        # Select indices where ds_times overlaps with requested times
        # For simplicity, we just find start and end indices
        start_idx = np.abs(ds_times - target_dt).argmin()
        end_idx = min(start_idx + duration_hours, len(ds_times) - 1)
        ds_t = ds.isel(time=slice(start_idx, end_idx + 1))
        out_times = ds_t.time.values

        lon = ds["lon"].values
        lat = ds["lat"].values
        lonc = ds["lonc"].values
        latc = ds["latc"].values

        nv_raw = ds["nv"].values
        if nv_raw.shape[0] == 3:
            nv = nv_raw.T - 1
        else:
            nv = nv_raw - 1

        d_spacing = 0.002
        lon_rho = np.arange(min_lon, max_lon, d_spacing)
        lat_rho = np.arange(min_lat, max_lat, d_spacing)
        lon_grid, lat_grid = np.meshgrid(lon_rho, lat_rho)

        import matplotlib.tri as mtri
        from scipy.spatial import Delaunay
        from scipy.interpolate import LinearNDInterpolator

        logger.info("Building NECOFS explicit interpolators...")
        fvcom_tri = mtri.Triangulation(lon, lat, triangles=nv)
        pts_elem = np.column_stack((lonc, latc))
        tri_elem = Delaunay(pts_elem)

        s_rho_dim = ds_t.sizes.get("siglay", 45)
        # Approximate depth or keep it strictly sigma like ICs?
        # The new julia OBC code extracts `zg_obc["depth"][:]` and maps it via best_z_idx
        # Let's map siglay to a pseudo-depth (e.g., 0 to 50m) or keep s_rho
        depths = np.linspace(-50, 0, s_rho_dim)

        nt = len(out_times)
        ny = len(lat_rho)
        nx = len(lon_rho)
        nz = s_rho_dim

        u_out = np.zeros((nt, nz, ny, nx), dtype=np.float32)
        v_out = np.zeros((nt, nz, ny, nx), dtype=np.float32)

        # Base land mask off full domain Zeta
        # We can evaluate zeta at time 0
        zeta_0 = ds_t["zeta"].isel(time=0).values
        interp = mtri.LinearTriInterpolator(fvcom_tri, zeta_0)
        zeta_grid = interp(lon_grid, lat_grid).filled(np.nan)
        land_mask = np.isnan(zeta_grid)

        logger.info(f"Extracting {nt} time steps for 4D OBC boundaries...")

        for t in range(nt):
            logger.info(f"Fetching time step {t + 1}/{nt} for OBC...")
            u_t = ds_t["u"].isel(time=t).values  # shape: (siglay, nele)
            v_t = ds_t["v"].isel(time=t).values

            for k in range(nz):
                interp_u = LinearNDInterpolator(tri_elem, u_t[k, :])
                interp_v = LinearNDInterpolator(tri_elem, v_t[k, :])

                u_k = interp_u(lon_grid, lat_grid)
                u_k[land_mask] = np.nan
                v_k = interp_v(lon_grid, lat_grid)
                v_k[land_mask] = np.nan

                u_out[t, k, :, :] = u_k
                v_out[t, k, :, :] = v_k

        ds_out = xr.Dataset(
            data_vars={
                "u": (("time", "depth", "eta", "xi"), u_out),
                "v": (("time", "depth", "eta", "xi"), v_out),
            },
            coords={
                "time": out_times,
                "depth": depths,
                "eta": lat_rho,
                "xi": lon_rho,
            },
            attrs={"type": "NECOFS/FVCOM GOM3 OBC", "source": "UMass Dartmouth SMAST"},
        )
        return ds_out

    except Exception as e:
        logger.error(f"Failed to process NECOFS OBC: {e}")
        return None
