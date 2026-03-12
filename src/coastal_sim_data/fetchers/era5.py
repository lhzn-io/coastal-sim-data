import os
import cdsapi
import logging
import xarray as xr
from datetime import datetime

from .validation import verify_sources

logger = logging.getLogger(__name__)


def fetch_era5_surface_forcing(
    target_date: str,
    bbox: list[float],
    output_path: str,
    preliminary: bool = False,
    cache_bust: bool = False,
) -> str:
    """
    Fetches ERA5 or ERA5T (Preliminary) hourly single levels for wind and pressure forcing.

    Args:
        target_date: YYYY-MM-DD
        bbox: [North, West, South, East]
        output_path: Path to save the downloaded .grib file
        preliminary: Use the ERA5T preliminary dataset (for requests < 3 months old)
    """
    verify_sources(
        source_name="ERA5/HRES" if not preliminary else "ERA5T",
        url="https://cds.climate.copernicus.eu/api",
        timestamp=datetime.utcnow().isoformat(),
        spatial_res="30km",
        requires_auth=True,
    )

    if not cache_bust and os.path.exists(output_path):
        logger.info(
            f"Cache hit: {output_path} already exists. Skipping CDS API request."
        )
    else:
        # The Copernicus Climate Data Store API client requires a .cdsapirc file or env vars:
        # CDSAPI_URL and CDSAPI_KEY. Since we load from .env, it will pick them up natively!
        client = cdsapi.Client()

        logger.info(
            f"Fetching {'ERA5T' if preliminary else 'ERA5'} forcing for {target_date} over bbox {bbox}..."
        )

        dt = datetime.strptime(target_date, "%Y-%m-%d")
        year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

        dataset = "reanalysis-era5-single-levels"

        # ERA5 grid is ~0.25 deg. Pad bbox by 0.5 deg to ensure we get at least 4x4 points for any small domain.
        pad = 0.5
        padded_bbox = [
            bbox[0] + pad,  # North (+ is North)
            bbox[1] - pad,  # West (- is West)
            bbox[2] - pad,  # South
            bbox[3] + pad,  # East
        ]

        # We request the 10m u/v wind components and surface pressure
        request_params = {
            "product_type": "reanalysis",
            "data_format": "grib",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": [f"{str(hour).zfill(2)}:00" for hour in range(24)],
            "area": padded_bbox,  # N, W, S, E
        }

        try:
            client.retrieve(dataset, request_params, output_path)
            logger.info(f"Successfully downloaded GRIB forcing to {output_path}")
        except Exception as e:
            logger.error(f"Failed to fetch ERA5 data: {e}")
            raise e

    # Print summary statistics
    try:
        ds = xr.open_dataset(output_path, engine="cfgrib")
        logger.info(f"--- ERA5 Forcing Field Summary ({output_path}) ---")
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
