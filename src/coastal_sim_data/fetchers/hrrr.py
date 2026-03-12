import os
import s3fs
import logging
import xarray as xr
from datetime import datetime

from .validation import verify_sources

logger = logging.getLogger(__name__)

# NOAA makes HRRR freely available on AWS S3 without authentication
fs = s3fs.S3FileSystem(anon=True)


def fetch_hrrr_surface_forcing(
    target_date: str, forecast_hour: int, output_path: str, cache_bust: bool = False
) -> str:
    """
    Fetches the High-Resolution Rapid Refresh (HRRR) CONUS surface dataset
    from NOAA's public S3 bucket for a specific target date and forecast hour.

    Args:
        target_date: YYYY-MM-DD
        forecast_hour: The hour of the day (0-23) the forecast was initialized
        output_path: Path to write the downloaded .grib2 file
    """
    logger.info(f"Fetching HRRR forcing for {target_date} cycle {forecast_hour}z...")

    dt = datetime.strptime(target_date, "%Y-%m-%d")
    date_str = dt.strftime("%Y%m%d")

    # S3 path structure: noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.t{CC}z.wrfsfcf{FF}.grib2
    # We'll pull the f00 (analysis/initialization file) for actual conditions
    s3_path = f"noaa-hrrr-bdp-pds/hrrr.{date_str}/conus/hrrr.t{str(forecast_hour).zfill(2)}z.wrfsfcf00.grib2"

    verify_sources(
        source_name="HRRR",
        url=f"s3://{s3_path}",
        timestamp=datetime.utcnow().isoformat(),
        spatial_res="3km",
        requires_auth=False,
    )

    if not cache_bust and os.path.exists(output_path):
        logger.info(
            f"Cache hit: {output_path} already exists. Skipping HRRR S3 download."
        )
    else:
        try:
            # Check if it exists before downloading (can take a few minutes to appear on AWS after the hour)
            if not fs.exists(s3_path):
                raise FileNotFoundError(
                    f"HRRR data not yet available on S3 at {s3_path}"
                )

            logger.info(f"Downloading s3://{s3_path} to {output_path}...")
            fs.get(s3_path, output_path)
            logger.info(f"Successfully downloaded HRRR GRIB2 to {output_path}")

        except Exception as e:
            logger.warning(f"Failed to fetch HRRR data from S3: {e}")
            raise e

    # Print summary statistics
    try:
        ds = xr.open_dataset(
            output_path, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface"}
        )
        logger.info(f"--- HRRR Forcing Field Summary ({output_path}) ---")
        for var_name in ds.data_vars:
            if ds[var_name].dtype.kind in "fi":
                logger.info(
                    f"  {var_name}: min={float(ds[var_name].min()):.2f}, "
                    f"max={float(ds[var_name].max()):.2f}, "
                    f"mean={float(ds[var_name].mean()):.2f}"
                )
        logger.info("----------------------------------------")
        ds.close()
    except Exception as e:
        logger.warning(f"Could not calculate summary statistics for {output_path}: {e}")

    return output_path
