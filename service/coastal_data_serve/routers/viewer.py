import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import io
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["Cache Viewer"])


class CacheItem(BaseModel):
    id: str
    type: str  # 'zarr', 'grib', 'nc'
    size_mb: float
    modified_time: float
    path: str
    # Enriched metadata (populated for zarr datasets)
    label: str | None = None  # Human-readable label
    variables: list[str] | None = None  # Data variable names
    grid_shape: str | None = None  # e.g. "3×4" or "point"
    time_steps: int | None = None  # Number of time steps
    time_range: str | None = None  # e.g. "Mar 2 00:00 → Mar 3 23:00"
    source: str | None = None  # e.g. "ERA5", "HRRR", "NOAA Water Level"
    spatial_extent: str | None = None  # e.g. "-71.25, 42.75 → -70.25, 43.50"
    target_date: str | None = None  # ISO timestamp from sidecar metadata (e.g. ICs)


def _enrich_zarr_metadata(zarr_path: Path) -> dict:
    """Extract grid shape, variables, time range, and source label from a zarr dataset."""
    try:
        import xarray as xr
        import numpy as np

        # Open with decode_times=True for proper datetime parsing
        try:
            ds = xr.open_zarr(str(zarr_path), consolidated=False, decode_times=True)
        except Exception:
            ds = xr.open_zarr(str(zarr_path), consolidated=False, decode_times=False)
        variables = list(ds.data_vars.keys())

        # Grid shape and spatial extent from coordinate arrays
        lat_dim = next((d for d in ["latitude", "lat", "y"] if d in ds.sizes), None)
        lon_dim = next((d for d in ["longitude", "lon", "x"] if d in ds.sizes), None)
        spatial_extent = None
        if lat_dim and lon_dim:
            nlat, nlon = ds.sizes[lat_dim], ds.sizes[lon_dim]
            grid_shape = f"{nlat}\u00d7{nlon}" if nlat > 0 and nlon > 0 else "point"
            # Extract coordinate bounds for spatial extent
            if lat_dim in ds.coords and lon_dim in ds.coords:
                lat_vals = ds[lat_dim].values
                lon_vals = ds[lon_dim].values
                if len(lat_vals) > 0 and len(lon_vals) > 0:
                    spatial_extent = (
                        f"{float(np.min(lon_vals)):.2f}, {float(np.min(lat_vals)):.2f}"
                        f" \u2192 {float(np.max(lon_vals)):.2f}, {float(np.max(lat_vals)):.2f}"
                    )
        else:
            grid_shape = "point"

        # Time info — use decoded timestamps when available
        time_dim = next(
            (d for d in ["time", "t", "step", "nt", "timeseries"] if d in ds.sizes),
            None,
        )
        time_steps = ds.sizes[time_dim] if time_dim else None
        time_range = None
        if time_dim and time_steps and time_steps > 0:
            try:
                from datetime import datetime as dt

                t_vals = ds[time_dim].values
                t0 = np.datetime64(t_vals[0], "ns")
                t1 = np.datetime64(t_vals[-1], "ns")
                d0 = t0.astype("datetime64[s]").astype(dt)
                d1 = t1.astype("datetime64[s]").astype(dt)
                time_range = (
                    f"{d0.strftime('%b %d %H:%M')} \u2192 {d1.strftime('%b %d %H:%M')}"
                )
            except Exception:
                time_range = f"{time_steps} steps"

        # Source detection
        stem = zarr_path.stem.lower()
        if stem.startswith("bc_"):
            # Detect source from variables present
            if "sp" in variables or "surface_pressure" in variables:
                source = "ERA5"
            elif any(v.startswith("HRRR") or "hrrr" in v for v in variables):
                source = "HRRR"
            else:
                source = "Atmos. Forcing"
            label = f"{source} Boundary Conditions"
        elif stem.startswith("ic_"):
            source = "Ocean IC"
            label = "Initial Conditions"
        elif "hrrr" in stem:
            source = "HRRR"
            label = f"HRRR Forecast ({stem.split('_')[1] if '_' in stem else ''})"
        elif "era5" in stem:
            source = "ERA5" if "era5t" not in stem else "ERA5T"
            label = f"{source} Reanalysis ({stem.split('_')[1] if '_' in stem else ''})"
        else:
            source = None
            label = None

        ds.close()

        # Read target_date from sidecar metadata JSON if available
        target_date = None
        sidecar = zarr_path.with_name(zarr_path.stem + "_metadata.json")
        if sidecar.exists():
            try:
                import json

                with open(sidecar) as f:
                    target_date = json.load(f).get("target_date")
            except Exception:
                pass

        return {
            "label": label,
            "variables": variables,
            "grid_shape": grid_shape,
            "time_steps": time_steps,
            "time_range": time_range,
            "source": source,
            "spatial_extent": spatial_extent,
            "target_date": target_date,
        }
    except Exception as e:
        logger.warning(f"Failed to enrich metadata for {zarr_path.name}: {e}")
        return {}


@router.get("/inventory", response_model=List[CacheItem])
async def get_cache_inventory():
    """Crawls the local data cache and returns an inventory of all processed forcing datasets."""
    cache_dir = Path(
        Path(
            os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
        ).expanduser()
    )
    if not cache_dir.exists():
        return []

    inventory = []

    # 1. Find Zarr datasets (directories)
    for zarr_path in cache_dir.glob("*.zarr"):
        if zarr_path.is_dir():
            size_bytes = sum(
                f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
            )
            meta = _enrich_zarr_metadata(zarr_path)
            inventory.append(
                CacheItem(
                    id=zarr_path.stem,
                    type="zarr",
                    size_mb=round(size_bytes / (1024 * 1024), 2),
                    modified_time=zarr_path.stat().st_mtime,
                    path=str(zarr_path),
                    **meta,
                )
            )

    # 2. Find GRIB/NC datasets (files)
    for ext in ["*.grib", "*.grib2", "*.nc"]:
        for file_path in cache_dir.glob(ext):
            if file_path.is_file():
                inventory.append(
                    CacheItem(
                        id=file_path.stem,
                        type=file_path.suffix.lstrip("."),
                        size_mb=round(file_path.stat().st_size / (1024 * 1024), 2),
                        modified_time=file_path.stat().st_mtime,
                        path=str(file_path),
                    )
                )

    # Sort by descending modification time (newest first)
    inventory.sort(key=lambda x: x.modified_time, reverse=True)
    return inventory


@router.get("/preview")
async def get_dataset_preview(
    dataset_id: str = Query(..., description="ID of the dataset (without extension)"),
    ext: str = Query(..., description="Extension of dataset (zarr, grib, nc)"),
    var_name: str = Query(..., description="Variable to plot (e.g., u10, v10, tp)"),
    time_idx: int = Query(0, description="Time step index relative to data slice"),
):
    """
    Renders a dynamic Matplotlib base64 preview of a cached multidimensional dataset.
    """

    # Deferred plotting imports for memory savings on edge devices
    import xarray as xr
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    cache_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    full_path = ""

    if ext == "zarr":
        full_path = os.path.join(cache_dir, f"{dataset_id}.zarr")
    else:
        full_path = os.path.join(cache_dir, f"{dataset_id}.{ext}")

    if not os.path.exists(full_path):
        raise HTTPException(
            status_code=404, detail=f"Dataset file {full_path} not found."
        )

    try:
        ds = None
        if ext == "zarr":
            ds = xr.open_zarr(full_path, consolidated=False, decode_times=True)
        else:
            # Handle ambiguous GRIB levels by attempting common filters
            try:
                ds = xr.open_dataset(full_path, decode_times=True)
            except Exception as e:
                if "multiple values for unique key" in str(e):
                    # Try surface or heightAboveGround
                    for filter_key in ["surface", "heightAboveGround"]:
                        try:
                            ds = xr.open_dataset(
                                full_path,
                                decode_times=True,
                                engine="cfgrib",
                                backend_kwargs={
                                    "filter_by_keys": {"typeOfLevel": filter_key}
                                },
                            )
                            if ds:
                                break
                        except Exception:
                            continue
                if not ds:
                    raise e

        # Map common IC variable names if missing
        if var_name not in ds.variables:
            ic_mapping = {
                "u": ["water_u", "u_current", "uo"],
                "v": ["water_v", "v_current", "vo"],
                "temp": ["water_temp", "temperature", "thetao"],
                "salt": ["salinity", "so"],
                "zeta": ["surf_el", "ssh", "zos"],
            }
            if var_name in ic_mapping:
                for alt_name in ic_mapping[var_name]:
                    if alt_name in ds.variables:
                        var_name = alt_name
                        break

        if var_name not in ds.variables:
            available = list(ds.data_vars.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Variable '{var_name}' not found. Try: {available}",
            )

        data_var = ds[var_name]

        # Detect time dimension
        time_dim = next(
            (
                d
                for d in ["time", "t", "step", "nt", "timeseries"]
                if d in data_var.dims
            ),
            None,
        )
        if not time_dim and len(data_var.dims) == 1:
            time_dim = str(data_var.dims[0])

        # Detect spatial dimensions
        lat_dim = next(
            (d for d in ["latitude", "lat", "y"] if d in data_var.dims), None
        )
        lon_dim = next(
            (d for d in ["longitude", "lon", "x"] if d in data_var.dims), None
        )
        has_spatial = (
            lat_dim
            and lon_dim
            and ds.sizes.get(lat_dim, 0) > 0
            and ds.sizes.get(lon_dim, 0) > 0
        )

        # Determine rendering mode:
        # - "timeseries" for 1D time-only data or point grids (0×0 spatial)
        # - "heatmap_small" for small grids (<= 20 cells per dim) with annotated values
        # - "heatmap" for larger spatial grids
        nlat = ds.sizes.get(lat_dim, 0) if lat_dim else 0
        nlon = ds.sizes.get(lon_dim, 0) if lon_dim else 0

        if not has_spatial and time_dim:
            render_mode = "timeseries"
        elif has_spatial and nlat <= 8 and nlon <= 8 and time_dim:
            render_mode = "timeseries_grid"
        elif has_spatial and nlat <= 20 and nlon <= 20:
            render_mode = "heatmap_small"
        else:
            render_mode = "heatmap"

        # Decode timestamp for the selected time index
        time_label = f"t={time_idx}"
        if time_dim and time_dim in ds.variables:
            try:
                t_vals = ds[time_dim].values
                idx = min(time_idx, len(t_vals) - 1)
                t_val = t_vals[idx]

                # Robust decoding to numpy datetime64
                if isinstance(t_val, np.datetime64):
                    d64 = t_val.astype("datetime64[us]")
                else:
                    try:
                        d64 = np.datetime64(t_val, "us")
                    except Exception:
                        # Fallback for floats: assume seconds from epoch
                        d64 = np.datetime64(int(float(t_val)), "s").astype(
                            "datetime64[us]"
                        )

                from datetime import datetime as dt

                d = d64.astype(dt)
                time_label = d.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        # Subtitle with decoded timestamp
        if time_label:
            fig.text(
                0.5,
                0.97,
                time_label,
                ha="center",
                va="top",
                fontsize=8.5,
                color="#8b949e",
                fontstyle="italic",
            )

        # Style for dark theme
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#f0f6fc")

        if render_mode == "timeseries":
            # Plot all time steps as a line chart
            values = data_var.values.flatten()
            valid = (
                ~np.isnan(values)
                if values.dtype.kind == "f"
                else np.ones_like(values, dtype=bool)
            )
            ax.plot(
                np.arange(len(values)),
                values,
                color="#58a6ff",
                linewidth=1.5,
                marker="o" if len(values) < 50 else None,
                markersize=3,
            )
            if time_idx < len(values):
                ax.axvline(
                    time_idx, color="#f85149", linewidth=1, linestyle="--", alpha=0.7
                )
                if valid[time_idx]:
                    ax.annotate(
                        f"{values[time_idx]:.3f}",
                        xy=(time_idx, values[time_idx]),
                        xytext=(5, 10),
                        textcoords="offset points",
                        color="#f0f6fc",
                        fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#21262d", ec="#30363d"),
                    )
            ax.set_xlabel("Time Step", fontsize=9)
            ax.set_ylabel(var_name, fontsize=9)
            ax.set_title(f"{dataset_id} — {var_name} ({time_label})", fontsize=10)
            ax.grid(True, alpha=0.15, color="#8b949e")

        elif render_mode == "timeseries_grid":
            # Small grid: plot timeseries for each cell as overlaid lines
            if time_dim:
                for ilat in range(nlat):
                    for ilon in range(nlon):
                        slice_opts = {}
                        if lat_dim:
                            slice_opts[lat_dim] = ilat
                        if lon_dim:
                            slice_opts[lon_dim] = ilon
                        ts = data_var.isel(slice_opts).values.flatten()
                        label = f"({ilat},{ilon})" if nlat * nlon <= 16 else None
                        ax.plot(ts, linewidth=1, alpha=0.7, label=label)
                if time_idx is not None:
                    ax.axvline(
                        time_idx,
                        color="#f85149",
                        linewidth=1,
                        linestyle="--",
                        alpha=0.7,
                    )
                ax.set_xlabel("Time Step", fontsize=9)
                ax.set_ylabel(var_name, fontsize=9)
                ax.set_title(
                    f"{dataset_id} \u2014 {var_name} ({nlat}\u00d7{nlon} grid)",
                    fontsize=10,
                )
                ax.grid(True, alpha=0.15, color="#8b949e")
                if nlat * nlon <= 16:
                    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.3)

        elif render_mode == "heatmap_small":
            # Small spatial grid with value annotations
            slice_opts = {}
            if time_dim:
                max_t = ds.sizes[time_dim] - 1
                slice_opts[time_dim] = min(time_idx, max_t)
            for z_name in ["depth", "zC", "level", "s_rho", "s_w", "isobaricInhPa"]:
                if z_name in data_var.dims:
                    slice_opts[z_name] = -1 if z_name in ("s_rho", "s_w") else 0
            valid_opts = {k: v for k, v in slice_opts.items() if k in data_var.dims}
            array = data_var.isel(valid_opts).values

            vmin = np.nanmin(array) if not np.all(np.isnan(array)) else 0.0
            vmax = np.nanmax(array) if not np.all(np.isnan(array)) else 1.0
            im = ax.imshow(
                array, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax
            )
            # Annotate cell values
            for yi in range(array.shape[0]):
                for xi in range(array.shape[1]):
                    v = array[yi, xi]
                    if not np.isnan(v):
                        ax.text(
                            xi,
                            yi,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
                        )
            cbar = plt.colorbar(im, ax=ax, label=var_name)
            cbar.ax.yaxis.set_tick_params(color="#8b949e")
            cbar.ax.yaxis.label.set_color("#c9d1d9")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
            ax.set_title(f"{dataset_id} \u2014 {var_name} ({time_label})", fontsize=10)

        else:
            # Standard heatmap for larger grids
            slice_opts = {}
            if time_dim:
                max_t = ds.sizes[time_dim] - 1
                slice_opts[time_dim] = min(time_idx, max_t)
            for z_name in ["depth", "zC", "level", "s_rho", "s_w", "isobaricInhPa"]:
                if z_name in data_var.dims:
                    slice_opts[z_name] = -1 if z_name in ("s_rho", "s_w") else 0
            valid_opts = {k: v for k, v in slice_opts.items() if k in data_var.dims}
            array = data_var.isel(valid_opts).values

            vmin = np.nanmin(array) if not np.all(np.isnan(array)) else 0.0
            vmax = np.nanmax(array) if not np.all(np.isnan(array)) else 1.0
            im = ax.imshow(
                array, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax
            )
            cbar = plt.colorbar(im, ax=ax, label=var_name)
            cbar.ax.yaxis.set_tick_params(color="#8b949e")
            cbar.ax.yaxis.label.set_color("#c9d1d9")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
            ax.set_title(f"{dataset_id} \u2014 {var_name} ({time_label})", fontsize=10)

        buf = io.BytesIO()
        plt.tight_layout(rect=(0, 0, 1, 0.95) if time_label else (0, 0, 1, 1))
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Failed to generate preview for {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview3d")
async def get_dataset_preview_3d(
    dataset_id: str = Query(..., description="ID of the dataset (without extension)"),
    time_idx: int = Query(0, description="Time step index"),
):
    """
    Returns downsampled 3D vector field JSON for Three.js rendering.
    Designed for IC/BC datasets with u,v velocity fields.
    """
    import xarray as xr
    import numpy as np
    import json

    cache_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    full_path = os.path.join(cache_dir, f"{dataset_id}.zarr")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

    try:
        ds = xr.open_zarr(full_path, consolidated=False, decode_times=True)
        vars_list = list(ds.data_vars)

        # Detect u/v variable pairs
        u_var = v_var = None
        for pair in [("u", "v"), ("water_u", "water_v"), ("u10", "v10")]:
            if pair[0] in vars_list and pair[1] in vars_list:
                u_var, v_var = pair
                break

        if u_var is None:
            raise HTTPException(
                status_code=400,
                detail=f"No u,v vector pair found. Variables: {vars_list}",
            )

        u_da = ds[u_var]
        v_da = ds[v_var]

        # Detect depth coordinate (s_rho, depth, z)
        depth_name = next(
            (
                n
                for n in ["s_rho", "depth", "z", "level"]
                if n in ds.coords or n in ds.dims
            ),
            None,
        )

        # Select time slice
        time_name = next((n for n in ["time", "ocean_time", "t"] if n in ds.dims), None)
        if time_name and time_name in u_da.dims:
            t_idx = min(time_idx, u_da.sizes[time_name] - 1)
            u_da = u_da.isel({time_name: t_idx})
            v_da = v_da.isel({time_name: t_idx})

        # Find the best lon/lat coordinate for each variable.
        # Staggered ROMS grids have per-variable coords (lon_u/lat_u, lon_v/lat_v).
        # We require the coord's dims to be a subset of the variable's dims so we
        # don't accidentally pick up a rho-grid coord for a u-grid variable.
        def _find_coord(da, candidates):
            for n in candidates:
                if n in ds.coords and set(ds[n].dims).issubset(set(da.dims)):
                    return n
            return None

        u_lon_name = _find_coord(
            u_da, [f"lon_{u_var}", "longitude", "lon", "lon_rho", "x"]
        )
        u_lat_name = _find_coord(
            u_da, [f"lat_{u_var}", "latitude", "lat", "lat_rho", "y"]
        )

        # Squeeze out broadcast/extra dims: keep only depth + the spatial dims that
        # belong to the variable's own coordinate (handles the rho-dim bleed-in from
        # ROMS curvilinear datasets).
        def _squeeze_broadcast(da, lon_cname, lat_cname):
            if lon_cname is None:
                return da
            spatial = set(ds[lon_cname].dims)
            if lat_cname and lat_cname in ds.coords:
                spatial |= set(ds[lat_cname].dims)
            keep = spatial | ({depth_name} if depth_name else set())
            extra = [d for d in da.dims if d not in keep]
            # Mean across broadcast dims (skipna so masked corners don't pollute)
            return da.mean(dim=extra, skipna=True) if extra else da

        u_da = _squeeze_broadcast(u_da, u_lon_name, u_lat_name)
        v_lon_name = _find_coord(
            v_da, [f"lon_{v_var}", "longitude", "lon", "lon_rho", "x"]
        )
        v_lat_name = _find_coord(
            v_da, [f"lat_{v_var}", "latitude", "lat", "lat_rho", "y"]
        )
        v_da = _squeeze_broadcast(v_da, v_lon_name, v_lat_name)

        # Get coordinate arrays (2D for curvilinear grids)
        lons = (
            ds[u_lon_name].values
            if u_lon_name
            else np.arange(u_da.shape[-1], dtype=float)
        )
        lats = (
            ds[u_lat_name].values
            if u_lat_name
            else np.arange(u_da.shape[-2] if u_da.ndim >= 2 else 1, dtype=float)
        )
        depths = ds[depth_name].values if depth_name else np.array([0.0])

        lons_2d = lons.ndim == 2
        lats_2d = lats.ndim == 2

        u_arr = u_da.values
        v_arr = v_da.values

        # Replace NaN/missing with 0
        u_arr = np.nan_to_num(u_arr, nan=0.0)
        v_arr = np.nan_to_num(v_arr, nan=0.0)

        def _lon(iy: int, ix: int) -> float:
            if lons_2d:
                return float(lons[iy, ix])
            return float(lons[ix]) if ix < len(lons) else float(ix)

        def _lat(iy: int, ix: int) -> float:
            if lats_2d:
                return float(lats[iy, ix])
            return float(lats[iy]) if iy < len(lats) else float(iy)

        # u and v may have different spatial shapes on staggered grids — clamp v indices
        v_shape = v_arr.shape

        def _v(iz_or_none, iy: int, ix: int) -> float:
            if v_arr.ndim == 2:
                return float(v_arr[min(iy, v_shape[0] - 1), min(ix, v_shape[1] - 1)])
            if v_arr.ndim == 3:
                iz = iz_or_none if iz_or_none is not None else 0
                return float(
                    v_arr[
                        min(iz, v_shape[0] - 1),
                        min(iy, v_shape[1] - 1),
                        min(ix, v_shape[2] - 1),
                    ]
                )
            return 0.0

        vectors = []
        max_vectors = 5000  # cap for UI performance

        if u_arr.ndim == 1:
            # 1D time-series (already time-sliced → scalar)
            vectors.append(
                {
                    "lon": float(lons[0]) if len(lons) > 0 else 0,
                    "lat": float(lats[0]) if len(lats) > 0 else 0,
                    "depth": 0,
                    "u": float(u_arr),
                    "v": float(v_arr),
                }
            )
        elif u_arr.ndim == 2:
            # 2D: (lat, lon) or (eta, xi) — curvilinear or rectilinear
            ny, nx = u_arr.shape
            step = max(1, int(np.sqrt(ny * nx / max_vectors)))
            for iy in range(0, ny, step):
                for ix in range(0, nx, step):
                    u_val = float(u_arr[iy, ix])
                    v_val = _v(None, iy, ix)
                    if abs(u_val) < 1e-10 and abs(v_val) < 1e-10:
                        continue
                    vectors.append(
                        {
                            "lon": _lon(iy, ix),
                            "lat": _lat(iy, ix),
                            "depth": 0,
                            "u": u_val,
                            "v": v_val,
                        }
                    )
        elif u_arr.ndim == 3:
            # 3D: (depth, lat, lon) or (s_rho, eta, xi)
            nz, ny, nx = u_arr.shape
            z_step = max(1, nz // 5)
            xy_step = max(
                1, int(np.sqrt(ny * nx / (max_vectors // max(1, nz // z_step))))
            )
            for iz in range(0, nz, z_step):
                d = float(depths[iz]) if iz < len(depths) else float(iz)
                # Map s_rho (-1 to 0) to physical depth
                if depth_name == "s_rho" and -1.5 < d < 0.5:
                    d = d * 50.0  # approximate 50m water column
                for iy in range(0, ny, xy_step):
                    for ix in range(0, nx, xy_step):
                        u_val = float(u_arr[iz, iy, ix])
                        v_val = _v(iz, iy, ix)
                        if abs(u_val) < 1e-10 and abs(v_val) < 1e-10:
                            continue
                        vectors.append(
                            {
                                "lon": _lon(iy, ix),
                                "lat": _lat(iy, ix),
                                "depth": d,
                                "u": u_val,
                                "v": v_val,
                            }
                        )

        # Compute bounds for Three.js scene setup
        all_lons = [v["lon"] for v in vectors] if vectors else [0]
        all_lats = [v["lat"] for v in vectors] if vectors else [0]

        result = {
            "vectors": vectors,
            "u_var": u_var,
            "v_var": v_var,
            "bounds": [min(all_lons), min(all_lats), max(all_lons), max(all_lats)],
            "depth_levels": sorted(set(v["depth"] for v in vectors)),
            "count": len(vectors),
        }

        return Response(
            content=json.dumps(result),
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate 3D preview for {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
