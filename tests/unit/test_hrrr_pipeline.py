import os
import sys

# Configure for local namespace
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from coastal_sim_data.dispatcher import dispatch_forcing_request
from coastal_sim_data.regridder import process_and_regrid_grib
from datetime import datetime


def test_hrrr_pipeline():
    print("Testing HRRR Fetching and Regridding Pipeline...")

    # Throgs neck bbox
    bbox = [40.815, -73.815, 40.785, -73.775]  # North, West, South, East

    # Target date is today
    target_date = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        print(f"1. Dispatching Forcing Request for {target_date}...")
        grib_path = dispatch_forcing_request(target_date, bbox)
        print(f"   => Downloaded GRIB to {grib_path}")

        print("2. Processing and Regridding GRIB...")
        zarr_path = grib_path.replace(".grib2", ".zarr").replace(".grib", ".zarr")
        process_and_regrid_grib(grib_path, bbox, zarr_path)
        print(f"   => Successfully converted to Zarr at {zarr_path}")

    except Exception as e:
        print(f"Pipeline Failed: {e}")


if __name__ == "__main__":
    test_hrrr_pipeline()
