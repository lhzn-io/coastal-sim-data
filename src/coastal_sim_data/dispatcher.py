import os
from datetime import datetime
import logging
from typing import Optional

from coastal_sim_data.fetchers.era5 import fetch_era5_surface_forcing
from coastal_sim_data.fetchers.hrrr import fetch_hrrr_surface_forcing

logger = logging.getLogger(__name__)


def dispatch_forcing_request(
    target_date: str,
    bbox: list[float],
    cache_dir: str = os.path.expanduser("~/.cache/coastal-sim-data"),
    cache_bust: bool = False,
) -> str:
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
    if delta_hours < 48 and delta_hours >= -24:
        logger.info(f"Delta is {delta_hours:.1f} hours. Dispatching to Tier IV: HRRR")

        from datetime import timedelta

        base_dt = now if delta_hours > 0 else target_dt

        for i in range(12):
            check_dt = base_dt - timedelta(hours=i)
            cycle_hr = check_dt.hour
            check_date_str = check_dt.strftime("%Y-%m-%d")

            out_path = os.path.join(
                cache_dir, f"hrrr_{check_date_str}_t{cycle_hr}z.grib2"
            )
            if not cache_bust and os.path.exists(out_path):
                logger.info(f"Cache hit: {out_path}")
                return out_path

            try:
                return fetch_hrrr_surface_forcing(
                    check_date_str, cycle_hr, out_path, cache_bust=cache_bust
                )
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
            return out_path
        return fetch_era5_surface_forcing(
            target_date, bbox, out_path, preliminary=False, cache_bust=cache_bust
        )

    # Tier II: Near-Past ERA5T
    elif delta_days >= 5 and delta_days <= 90:
        logger.info(
            f"Delta is {delta_days} days. Dispatching to Tier II: ERA5T Preliminary (CDS)"
        )
        out_path = os.path.join(cache_dir, f"era5t_{target_date}.grib")
        if not cache_bust and os.path.exists(out_path):
            return out_path
        return fetch_era5_surface_forcing(
            target_date, bbox, out_path, preliminary=True, cache_bust=cache_bust
        )

    else:
        logger.warning(
            f"Delta is {delta_days} days. Tier III HRES not yet implemented. Falling back to ERA5T if available, or failing."
        )
        # Fallback implementation
        out_path = os.path.join(cache_dir, f"ecmwf_fallback_{target_date}.grib")
        return fetch_era5_surface_forcing(
            target_date, bbox, out_path, preliminary=True, cache_bust=cache_bust
        )


def dispatch_ic_request(
    target_date: str,
    bbox: list[float],
    cache_dir: str = os.path.expanduser("~/.cache/coastal-sim-data"),
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

    ds = None

    from coastal_sim_data.fetchers.neracoos import fetch_neracoos_initial_conditions
    from coastal_sim_data.fetchers.maracoos import fetch_maracoos_initial_conditions
    from coastal_sim_data.fetchers.hycom import fetch_hycom_initial_conditions

    # 1. Try NERACOOS (NH/Piscataqua)
    try:
        ds = fetch_neracoos_initial_conditions(target_date, bbox)
    except Exception as e:
        logger.error(f"NERACOOS fetch failed: {e}")

    # 2. Try MARACOOS (NY/NJ/LIS)
    if ds is None:
        try:
            ds = fetch_maracoos_initial_conditions(target_date, bbox)
        except Exception as e:
            logger.error(f"MARACOOS fetch failed: {e}")

    # 3. Fallback to HYCOM
    if ds is None:
        logger.info("Regional models failed or out of bounds. Falling back to HYCOM.")
        try:
            ds = fetch_hycom_initial_conditions(target_date, bbox)
        except Exception as e:
            logger.error(f"HYCOM fetch failed: {e}")

    if ds is None:
        raise RuntimeError(
            "No Initial Conditions could be fetched for this domain and time."
        )

    # Save to Zarr
    logger.info(f"Writing Initial Conditions to {zarr_path}...")

    import shutil

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    # Minimal cleanup for Zarr engine compatibility
    for var in list(ds.variables):
        ds[var].encoding.clear()

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
    cache_dir: str = os.path.expanduser("~/.cache/coastal-sim-data/erddap"),
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
