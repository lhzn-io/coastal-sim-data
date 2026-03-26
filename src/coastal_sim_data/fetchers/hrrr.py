import os
import s3fs
import logging
import xarray as xr
from datetime import datetime

from .validation import verify_sources

logger = logging.getLogger(__name__)

# NOAA makes HRRR freely available on AWS S3 without authentication
fs = s3fs.S3FileSystem(anon=True)

# GRIB2 .idx variable patterns for 10-meter wind components.
# Each HRRR .grib2 on S3 has a sidecar .idx listing byte offsets per message.
WIND_IDX_PATTERNS = ("UGRD:10 m above ground", "VGRD:10 m above ground")


def _parse_idx(
    idx_text: str, patterns: tuple[str, ...]
) -> list[tuple[int, int | None]]:
    """
    Parse a GRIB2 .idx sidecar file and return byte ranges for matching messages.

    Each .idx line has the format:
        message_number:byte_offset:date:variable:level:forecast_type:

    Returns a list of (start_byte, end_byte) tuples. end_byte is None for the
    last message in the file (meaning read-to-EOF).
    """
    lines = [line.strip() for line in idx_text.strip().splitlines() if line.strip()]

    # Build list of (message_number, byte_offset, raw_line)
    entries = []
    for line in lines:
        parts = line.split(":")
        if len(parts) < 3:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        entries.append((offset, line))

    # Find matching messages and compute byte ranges
    ranges = []
    for i, (offset, line) in enumerate(entries):
        if any(pattern in line for pattern in patterns):
            # End byte is the start of the next message, or None for EOF
            end = entries[i + 1][0] if i + 1 < len(entries) else None
            ranges.append((offset, end))

    return ranges


def _fetch_s3_byte_ranges(
    s3_path: str, ranges: list[tuple[int, int | None]], output_path: str
) -> None:
    """
    Download specific byte ranges from an S3 object and concatenate into a
    single valid GRIB2 file. Each GRIB2 message is self-contained, so
    concatenating selected messages produces a valid file.
    """
    with open(output_path, "wb") as f:
        for start, end in ranges:
            if end is not None:
                length = end - start
                with fs.open(s3_path, "rb") as s3f:
                    s3f.seek(start)
                    f.write(s3f.read(length))
            else:
                # Read from start to EOF
                with fs.open(s3_path, "rb") as s3f:
                    s3f.seek(start)
                    f.write(s3f.read())


def fetch_hrrr_surface_forcing(
    target_date: str,
    forecast_hour: int,
    output_path: str,
    forecast_offset: int = 0,
    cache_bust: bool = False,
) -> str:
    """
    Fetches 10-meter wind components (u10, v10) from the HRRR CONUS surface
    dataset on NOAA's public S3 bucket using byte-range reads via the .idx
    sidecar file. Falls back to downloading the full GRIB2 if the .idx is
    unavailable.

    Args:
        target_date: YYYY-MM-DD
        forecast_hour: The hour of the day (0-23) the forecast was initialized
        output_path: Path to write the downloaded .grib2 file
        forecast_offset: Forecast offset hour (0 = analysis)
        cache_bust: Force re-download even if cached
    """
    logger.info(f"Fetching HRRR forcing for {target_date} cycle {forecast_hour}z...")

    dt = datetime.strptime(target_date, "%Y-%m-%d")
    date_str = dt.strftime("%Y%m%d")

    ff = str(forecast_offset).zfill(2)
    s3_path = f"noaa-hrrr-bdp-pds/hrrr.{date_str}/conus/hrrr.t{str(forecast_hour).zfill(2)}z.wrfsfcf{ff}.grib2"
    idx_path = f"{s3_path}.idx"

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
            if not fs.exists(s3_path):
                raise FileNotFoundError(
                    f"HRRR data not yet available on S3 at {s3_path}"
                )

            # Try selective byte-range download via .idx sidecar
            fetched_selectively = False
            if fs.exists(idx_path):
                try:
                    with fs.open(idx_path, "r") as f:
                        idx_text = f.read()

                    ranges = _parse_idx(idx_text, WIND_IDX_PATTERNS)

                    if ranges:
                        logger.info(
                            f"Selective fetch: {len(ranges)} wind messages "
                            f"from .idx ({idx_path})"
                        )
                        _fetch_s3_byte_ranges(s3_path, ranges, output_path)
                        fetched_selectively = True
                        size_kb = os.path.getsize(output_path) / 1024
                        logger.info(
                            f"Downloaded {size_kb:.0f} KB (wind-only) to {output_path}"
                        )
                    else:
                        logger.warning(
                            "No wind variables found in .idx — "
                            "falling back to full GRIB2 download."
                        )
                except Exception as e:
                    logger.warning(
                        f"Selective fetch failed ({e}) — "
                        "falling back to full GRIB2 download."
                    )

            if not fetched_selectively:
                logger.info(
                    f"Downloading full GRIB2 s3://{s3_path} to {output_path}..."
                )
                fs.get(s3_path, output_path)
                logger.info(f"Successfully downloaded HRRR GRIB2 to {output_path}")

        except Exception as e:
            logger.warning(f"Failed to fetch HRRR data from S3: {e}")
            raise e

    # Print summary statistics
    try:
        ds = xr.open_dataset(output_path, engine="cfgrib")
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
