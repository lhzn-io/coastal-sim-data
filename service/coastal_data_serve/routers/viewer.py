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


@router.get("/inventory", response_model=List[CacheItem])
async def get_cache_inventory():
    """Crawls the local data cache and returns an inventory of all processed forcing datasets."""
    cache_dir = Path(os.path.expanduser("~/.cache/coastal-sim-data"))
    if not cache_dir.exists():
        return []

    inventory = []

    # 1. Find Zarr datasets (directories)
    for zarr_path in cache_dir.glob("*.zarr"):
        if zarr_path.is_dir():
            # Calculate rough directory size
            size_bytes = sum(
                f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
            )
            inventory.append(
                CacheItem(
                    id=zarr_path.stem,
                    type="zarr",
                    size_mb=round(size_bytes / (1024 * 1024), 2),
                    modified_time=zarr_path.stat().st_mtime,
                    path=str(zarr_path),
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

    cache_dir = os.path.expanduser("~/.cache/coastal-sim-data")
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

        # Time slicing
        data_var = ds[var_name]

        # Determine if there is a time dimension to slice
        if "time" in data_var.dims or "t" in data_var.dims:
            time_dim = "time" if "time" in data_var.dims else "t"
            max_t = ds.dims[time_dim] - 1
            idx = min(time_idx, max_t)
            slice_opts = {time_dim: idx}

            # Surface only for 3D/4D (like HYCOM / NYHOPS / MARACOOS)
            for z_name in ["depth", "zC", "level", "s_rho", "s_w", "isobaricInhPa"]:
                if z_name in data_var.dims:
                    if z_name in ["s_rho", "s_w"]:
                        slice_opts[
                            z_name
                        ] = -1  # Surface in ROMS is usually the last index
                    else:
                        slice_opts[z_name] = 0

            # Filter slice_opts to only include valid dimensions for this data_var
            valid_slice_opts = {
                k: v for k, v in slice_opts.items() if k in data_var.dims
            }
            array = data_var.isel(**valid_slice_opts).values  # type: ignore
        else:
            array = data_var.values

        fig, ax = plt.subplots(figsize=(6, 5))

        # Exclude NaN boundaries (land) from cmap
        vmax = np.nanmax(array) if not np.all(np.isnan(array)) else 1.0
        vmin = np.nanmin(array) if not np.all(np.isnan(array)) else 0.0

        # Map orientation
        im = ax.imshow(
            array, origin="lower", cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax
        )

        ax.set_title(f"{dataset_id}\n{var_name} (t={time_idx})")
        plt.colorbar(im, ax=ax, label=var_name)

        # Buffer PNG bytes
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=90)
        buf.seek(0)
        plt.close(fig)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Failed to generate preview for {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
