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
        time_dim = next((d for d in ["time", "t"] if d in ds.sizes), None)
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
        return {
            "label": label,
            "variables": variables,
            "grid_shape": grid_shape,
            "time_steps": time_steps,
            "time_range": time_range,
            "source": source,
            "spatial_extent": spatial_extent,
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
            ds = xr.open_zarr(full_path, consolidated=False, decode_times=False)
        else:
            # Handle ambiguous GRIB levels by attempting common filters
            try:
                ds = xr.open_dataset(full_path, decode_times=False)
            except Exception as e:
                if "multiple values for unique key" in str(e):
                    # Try surface or heightAboveGround
                    for filter_key in ["surface", "heightAboveGround"]:
                        try:
                            ds = xr.open_dataset(
                                full_path,
                                decode_times=False,
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
        time_dim = next((d for d in ["time", "t"] if d in data_var.dims), None)

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
        time_label = ""
        if time_dim and time_dim in ds.coords:
            try:
                t_vals = ds[time_dim].values
                idx = min(time_idx, len(t_vals) - 1)
                t_stamp = np.datetime64(t_vals[idx], "ns")  # type: ignore[call-overload]
                from datetime import datetime as dt

                d = t_stamp.astype("datetime64[s]").astype(dt)  # type: ignore[union-attr]
                time_label = d.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                time_label = f"t={time_idx}"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        # Subtitle with decoded timestamp
        if time_label:
            fig.text(
                0.5,
                0.97,
                f"Step {time_idx}: {time_label}",
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
            ax.set_title(f"{dataset_id} \u2014 {var_name}", fontsize=10)
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
            ax.set_title(f"{dataset_id} \u2014 {var_name} (t={time_idx})", fontsize=10)

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
            ax.set_title(f"{dataset_id} \u2014 {var_name} (t={time_idx})", fontsize=10)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Failed to generate preview for {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
