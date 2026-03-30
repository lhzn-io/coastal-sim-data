import os
from datetime import datetime
import logging
from typing import Optional
import numpy as np

from coastal_sim_data.fetchers.era5 import fetch_era5_surface_forcing
from coastal_sim_data.fetchers.hrrr import fetch_hrrr_surface_forcing

logger = logging.getLogger(__name__)


def predict_bc_donor(target_date: str) -> dict:
    """Predicts the boundary condition (forcing) donor based on target date delta."""
    from datetime import datetime, timezone

    # Parse target date
    try:
        if "T" in target_date:
            from dateutil import parser  # type: ignore[import-untyped]

            target_dt = parser.parse(target_date)
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
    except Exception:
        return {"name": "Unknown (Invalid Date)", "resolution_approx_m": 0}

    now = datetime.now(timezone.utc)
    delta_days = (now - target_dt).days
    # We now allow HRRR for historical dates since we pull from the AWS S3 archive
    # In a full implementation we'd check if the bbox is within CONUS, but for NH/GoM it is.
    if delta_days <= 1000:  # Assuming HRRR archive is robust for recent years
        return {"name": "HRRR (NOAA)", "resolution_approx_m": 3000, "tier": "IV"}
    elif delta_days > 1000:
        return {
            "name": "ERA5 Final (ECMWF CDS)",
            "resolution_approx_m": 31000,
            "tier": "I",
        }
    elif delta_days >= 5 and delta_days <= 90:
        return {
            "name": "ERA5T Preliminary (ECMWF CDS)",
            "resolution_approx_m": 31000,
            "tier": "II",
        }
    else:
        return {
            "name": "ERA5T Fallback (ECMWF CDS)",
            "resolution_approx_m": 31000,
            "tier": "III",
        }


def dispatch_forcing_request(
    target_date: str,
    bbox: list[float],
    duration_hours: int = 1,
    cache_dir: str = os.environ.get(
        "COASTAL_SIM_DATA_CACHE_DIR", os.path.expanduser("~/.cache/coastal-sim-data")
    ),
    cache_bust: bool = False,
) -> list[str]:
    """
    Tiered modality dispatcher. Evaluates the Delta T between 'Today' and the 'Target Date'
    to download the highest fidelity forcing model available.

    Tier I   (> 3 Months old) : ERA5 Final
    Tier II  (5 Days to 3 Mos): ERA5T Preliminary
    Tier III (Future to 5 Day): ECMWF HRES (Not implemented)
    Tier IV  (Last 18 Hours)  : HRRR High-Resolution Rapid Refresh
    """

    os.makedirs(cache_dir, exist_ok=True)

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    now = datetime.utcnow()

    delta_days = (now - target_dt).days
    delta_hours = (now - target_dt).total_seconds() / 3600.0

    # Tier IV: Edge/Tactical HRRR (Very recent or forecast)
    if delta_days <= 1000:
        logger.info("Dispatching to Tier IV (Historical/Live): HRRR")

        from datetime import timedelta

        # If it's a future forecast use 'now' as the base init time to find the latest run
        # If it's historical, use the exact target_dt (we'll fetch the analysis f00 for that hour)
        base_dt = now if delta_hours < 0 else target_dt

        for i in range(12):
            check_dt = base_dt - timedelta(hours=i)
            cycle_hr = check_dt.hour
            check_date_str = check_dt.strftime("%Y-%m-%d")

            out_path = os.path.join(
                cache_dir, f"hrrr_{check_date_str}_t{cycle_hr}z.grib2"
            )
            if not cache_bust and os.path.exists(out_path):
                logger.info(f"Checking cache sequence starting at: {out_path}")
                grib_paths = [out_path]
                all_cached = True
                for offset in range(1, duration_hours):
                    offset_path = os.path.join(
                        cache_dir,
                        f"hrrr_{check_date_str}_t{cycle_hr}z_f{offset:02d}.grib2",
                    )
                    if os.path.exists(offset_path):
                        grib_paths.append(offset_path)
                    else:
                        all_cached = False
                        break
                if all_cached:
                    logger.info("Found fully cached sequence!")
                    return grib_paths
                else:
                    logger.info(
                        "Sequence incomplete in cache, falling through to fetch."
                    )

            try:
                # First check if f00 exists for this cycle
                fetch_hrrr_surface_forcing(
                    check_date_str,
                    cycle_hr,
                    out_path,
                    forecast_offset=0,
                    cache_bust=cache_bust,
                )

                # If it does, fetch the rest of the sequence in parallel
                grib_paths = [out_path]
                import concurrent.futures

                def _fetch_offset(offset):
                    offset_path = os.path.join(
                        cache_dir,
                        f"hrrr_{check_date_str}_t{cycle_hr}z_f{offset:02d}.grib2",
                    )
                    fetch_hrrr_surface_forcing(
                        check_date_str,
                        cycle_hr,
                        offset_path,
                        forecast_offset=offset,
                        cache_bust=cache_bust,
                    )
                    return offset_path

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_offset = {
                        executor.submit(_fetch_offset, offset): offset
                        for offset in range(1, duration_hours)
                    }
                    results = {}
                    for future in concurrent.futures.as_completed(future_to_offset):
                        offset = future_to_offset[future]
                        try:
                            path = future.result()
                            results[offset] = path
                        except FileNotFoundError:
                            logger.warning(f"Offset f{offset:02d} missing.")
                        except Exception as e:
                            logger.error(f"Error fetching offset f{offset:02d}: {e}")

                # Append in correct order
                for offset in range(1, duration_hours):
                    if offset in results:
                        grib_paths.append(results[offset])
                    else:
                        break  # Stop at first missing

                return grib_paths
            except FileNotFoundError:
                logger.warning(
                    f"HRRR {check_date_str} {cycle_hr}z missing on S3. Trying older cycle..."
                )
                continue

        raise FileNotFoundError(
            "Exhausted 12-hour lookback for HRRR. S3 bucket might be down."
        )

    # Tier I: Validated ERA5 Final
    elif delta_days > 90:
        logger.info(
            f"Delta is {delta_days} days. Dispatching to Tier I: ERA5 Final (CDS)"
        )
        out_path = os.path.join(cache_dir, f"era5_{target_date}.grib")
        if not cache_bust and os.path.exists(out_path):
            return [out_path]  # WARNING: naive cache hit, may miss subsequent steps
        return [
            fetch_era5_surface_forcing(
                target_date, bbox, out_path, preliminary=False, cache_bust=cache_bust
            )
        ]

    # Tier II: Near-Past ERA5T
    elif delta_days >= 5 and delta_days <= 90:
        logger.info(
            f"Delta is {delta_days} days. Dispatching to Tier II: ERA5T Preliminary (CDS)"
        )
        out_path = os.path.join(cache_dir, f"era5t_{target_date}.grib")
        if not cache_bust and os.path.exists(out_path):
            return [out_path]  # WARNING: naive cache hit, may miss subsequent steps
        return [
            fetch_era5_surface_forcing(
                target_date, bbox, out_path, preliminary=True, cache_bust=cache_bust
            )
        ]

    else:
        logger.warning(
            f"Delta is {delta_days} days. Tier III HRES not yet implemented. Falling back to ERA5T if available, or failing."
        )
        # Fallback implementation
        out_path = os.path.join(cache_dir, f"ecmwf_fallback_{target_date}.grib")
        return [
            fetch_era5_surface_forcing(
                target_date, bbox, out_path, preliminary=True, cache_bust=cache_bust
            )
        ]


def get_ic_fetchers():
    """Return all registered IC fetcher modules and their fetch functions."""
    from coastal_sim_data.fetchers import (
        hycom,
        necofs,
        nyofs,
    )

    return [
        (nyofs, nyofs.fetch_nyofs_initial_conditions),
        (necofs, necofs.fetch_necofs_initial_conditions),
        (hycom, hycom.fetch_hycom_initial_conditions),
    ]


def _domain_area(domain_bbox: list[float]) -> float:
    """Compute approximate domain area in degree^2 from [min_lon, min_lat, max_lon, max_lat]."""
    return (domain_bbox[2] - domain_bbox[0]) * (domain_bbox[3] - domain_bbox[1])


def _rank_ic_candidates(bbox: list[float]) -> list[tuple]:
    """
    Rank IC fetchers for a given request bbox using spatial heuristics.

    Scoring strategy (smallest-enclosing-domain first, resolution tiebreak):
      1. Filter to models whose domain contains the request bbox.
      2. Sort by domain area ascending — the tightest enclosing domain is most
         likely purpose-built for the region (e.g. NYHOPS for NY harbor).
      3. Tiebreak by resolution_approx_m ascending (finer resolution wins).

    Returns an ordered list of (module, fetch_func, metadata) tuples.
    """
    candidates = []
    for module, fetch_func in get_ic_fetchers():
        if not module.supports_bbox(bbox):
            continue
        meta = module.get_metadata()
        domain_bbox = meta.get("domain_bbox")
        area = _domain_area(domain_bbox) if domain_bbox else float("inf")
        resolution = meta.get("resolution_approx_m", float("inf"))
        candidates.append((area, resolution, module, fetch_func, meta))

    # Sort: smallest domain first, then finest resolution
    candidates.sort(key=lambda c: (c[0], c[1]))

    ranked = [(c[2], c[3], c[4]) for c in candidates]
    if ranked:
        logger.info(
            f"IC donor ranking for bbox {bbox}: "
            + " > ".join(
                f"{c[4]['name']} (area={c[0]:.1f}°², ~{c[1]:.0f}m)" for c in candidates
            )
        )
    return ranked


def predict_ic_donor(bbox: list[float]) -> dict:
    """Predicts which model will be used for a given bounding box."""
    ranked = _rank_ic_candidates(bbox)
    if ranked:
        return ranked[0][2]
    return {}


def dispatch_ic_request(
    target_date: str,
    bbox: list[float],
    cache_dir: str = os.environ.get(
        "COASTAL_SIM_DATA_CACHE_DIR", os.path.expanduser("~/.cache/coastal-sim-data")
    ),
    cache_bust: bool = False,
    zarr_path: Optional[str] = None,
) -> str:
    """
    Tiered modality dispatcher for Initial Conditions. Evaluates regional high-res models,
    falling back to global HYCOM.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if not zarr_path:
        zarr_name = f"ic_{target_date}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.zarr"
        zarr_path = os.path.join(cache_dir, zarr_name)

    if not cache_bust and os.path.exists(zarr_path):
        logger.info(f"Cache hit for IC: {zarr_path}")
        return zarr_path

    ranked = _rank_ic_candidates(bbox)
    if not ranked:
        raise ValueError(f"No suitable IC fetcher found for bbox {bbox}")

    target_module, target_fetch_func, meta = ranked[0]
    logger.info(f"Selected primary IC Donor: {meta['name']}")

    try:
        ds = target_fetch_func(target_date, bbox)
    except Exception as e:
        logger.error(f"{meta['name']} fetch failed: {e}")
        ds = None

    if ds is None:
        raise RuntimeError(
            f"Primary IC donor {meta['name']} failed to return data. Silent failover to lower-res models is disabled."
        )

    # Apply standard provenance attributes directly to dataset
    ds.attrs.update(
        {
            "type": meta["name"],
            "donor_id": meta["id"],
            "source": "coastal-sim-data",
            "target_date": target_date,  # existing parser relies on this string being 'UMass Dartmouth SMAST' or similar, but let's standardize
        }
    )

    # Save to Zarr
    logger.info(f"Writing Initial Conditions to {zarr_path}...")

    import shutil

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    # Minimal cleanup for Zarr engine compatibility
    for var in list(ds.variables):
        ds[var].encoding.clear()

        # Prevent Zarr from omitting "zero" chunks which crashes Zarr.jl
        if ds[var].dtype.kind in "iu":
            ds[var].encoding["_FillValue"] = -9999
        else:
            ds[var].encoding["_FillValue"] = -9999.0

        # Force explicit Little-Endian Float32 for all physical variables
        # because Oceananigans Zarr.jl backend faults on Big-Endian types
        if ds[var].dtype == "float64" and "time" not in str(var):
            ds[var] = ds[var].astype("<f4")

    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)

    logger.info(
        f"Successfully generated IC Zarr with V2 strict protocol at {zarr_path}"
    )

    return zarr_path


def dispatch_station_profiles_request(
    station_id: str,
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        os.environ.get(
            "COASTAL_SIM_DATA_CACHE_DIR",
            os.path.expanduser("~/.cache/coastal-sim-data"),
        ),
        "erddap",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Tiered modality dispatcher for internal nudging profiles.
    Currently routes WLIS directly to the ERDDAP fetcher.
    """
    logger.info(
        f"Dispatching station profile request for {station_id} ({start_time} to {end_time})"
    )

    # We could implement a real fallback strategy here, but for now we dispatch straight to our ERDDAP module
    from coastal_sim_data.fetchers.erddap import fetch_erddap_station_profiles

    try:
        profiles = fetch_erddap_station_profiles(
            station_id=station_id,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            cache_bust=cache_bust,
        )
        return profiles
    except Exception as e:
        logger.error(f"Failed to fetch profiles for {station_id}: {e}")
        return {}


def dispatch_bounding_box_profiles_request(
    bbox: list[float],
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.join(
        os.environ.get(
            "COASTAL_SIM_DATA_CACHE_DIR",
            os.path.expanduser("~/.cache/coastal-sim-data"),
        ),
        "erddap",
    ),
    cache_bust: bool = False,
) -> dict:
    """
    Tiered modality dispatcher for internal nudging profiles across a domain.
    """
    logger.info(
        f"Dispatching bounded profile request for {bbox} ({start_time} to {end_time})"
    )

    from coastal_sim_data.fetchers.erddap import fetch_erddap_stations_in_bbox

    try:
        profiles = fetch_erddap_stations_in_bbox(
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
            cache_dir=cache_dir,
            cache_bust=cache_bust,
        )
        return profiles
    except Exception as e:
        logger.error(f"Failed to fetch bounded profiles: {e}")
        return {}


def _rank_obc_candidates(bbox: list[float]) -> list[tuple]:
    candidates = []
    for module, fetch_func in get_obc_fetchers():
        if not module.supports_bbox(bbox):
            continue
        meta = module.get_metadata()
        domain_bbox = meta.get("domain_bbox")
        area = _domain_area(domain_bbox) if domain_bbox else float("inf")
        resolution = meta.get("resolution_approx_m", float("inf"))
        candidates.append((area, resolution, module, fetch_func, meta))

    candidates.sort(key=lambda c: (c[0], c[1]))
    ranked = [(c[2], c[3], c[4]) for c in candidates]
    if ranked:
        logger.info(
            f"OBC donor ranking for bbox {bbox}: "
            + " > ".join(
                f"{c[4]['name']} (area={c[0]:.1f}°², ~{c[1]:.0f}m)" for c in candidates
            )
        )
    return ranked


def predict_obc_donor(bbox: list[float]) -> dict:
    ranked = _rank_obc_candidates(bbox)
    if ranked:
        return ranked[0][2]
    return {}


def dispatch_obc_request(
    start_date: str,
    duration_hours: int,
    bbox: list[float],
    cache_dir: str = os.environ.get(
        "COASTAL_SIM_DATA_CACHE_DIR", os.path.expanduser("~/.cache/coastal-sim-data")
    ),
    cache_bust: bool = False,
    zarr_path: Optional[str] = None,
    allow_donor_fallback: bool = False,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)

    if not zarr_path:
        zarr_name = f"obc_{start_date}_{duration_hours}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.zarr"
        zarr_path = os.path.join(cache_dir, zarr_name)

    if not cache_bust and os.path.exists(zarr_path):
        logger.info(f"Cache hit for OBC: {zarr_path}")
        return zarr_path

    ranked = _rank_obc_candidates(bbox)
    if not ranked:
        raise ValueError(f"No suitable OBC fetcher found for bbox {bbox}")

    ds = None
    target_module = None
    meta = None
    candidates_to_try = ranked if allow_donor_fallback else ranked[:1]
    for candidate_module, candidate_fetch_func, candidate_meta in candidates_to_try:
        logger.info(f"Trying OBC donor: {candidate_meta['name']}")
        try:
            ds = candidate_fetch_func(start_date, duration_hours, bbox)
        except Exception as e:
            logger.warning(f"{candidate_meta['name']} OBC fetch raised: {e}")
            ds = None
        if ds is not None:
            target_module = candidate_module
            meta = candidate_meta
            logger.info(f"OBC donor succeeded: {meta['name']}")
            break
        if allow_donor_fallback:
            logger.warning(
                f"{candidate_meta['name']} OBC fetch returned no data, trying next donor."
            )

    if ds is None or target_module is None:
        tried = [c[2]["name"] for c in candidates_to_try]
        raise RuntimeError(
            f"Primary OBC donor {tried[0]} failed to return data."
            if not allow_donor_fallback
            else f"All OBC donors failed for bbox {bbox}. Tried: {tried}"
        )

    # Cast strictly to Float32 for Oceananigans
    for var in ds.data_vars:
        if ds[var].dtype != np.float32:
            ds[var] = ds[var].astype(np.float32)

    # Convert Endianness for Julia
    # Minimal cleanup for Zarr engine compatibility
    for var in list(ds.variables):
        ds[var].encoding.clear()

        # Prevent Zarr from omitting "zero" chunks which crashes Zarr.jl
        if ds[var].dtype.kind in "iu":
            ds[var].encoding["_FillValue"] = -9999
        else:
            ds[var].encoding["_FillValue"] = -9999.0

        # Force explicit Little-Endian Float32 for all physical variables
        # because Oceananigans Zarr.jl backend faults on Big-Endian types
        if (
            ds[var].dtype == "float64" or ds[var].dtype == "float32"
        ) and "time" not in str(var):
            ds[var] = ds[var].astype("<f4")

    ds.attrs["source"] = target_module.__name__
    ds.attrs["type"] = "3D Time-Varying Free Surface + Hydrostatic Velocity"
    ds.attrs["duration_hours"] = duration_hours

    logger.info(f"Writing OBC data to Zarr: {zarr_path}")
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_format=2)
    return zarr_path


def get_obc_fetchers():
    from coastal_sim_data.fetchers import (
        hycom,
        necofs,
        nyhops,
        nyofs,
    )

    # Dynamically discover fetch_xxx_boundary_conditions functions.
    # Not all modules have implemented OBC yet, so we check with hasattr.
    fetchers = []
    for module in [nyofs, nyhops, necofs, hycom]:
        func_name = "fetch_" + module.__name__.split(".")[-1] + "_boundary_conditions"
        if hasattr(module, func_name):
            fetchers.append((module, getattr(module, func_name)))
    return fetchers
