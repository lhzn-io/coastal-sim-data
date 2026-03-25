import sys
import os
import pandas as pd

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from coastal_sim_data.fetchers.nyhops import fetch_nyhops_initial_conditions
from coastal_sim_data.fetchers.hycom import fetch_hycom_initial_conditions
from coastal_sim_data.fetchers.maracoos import fetch_maracoos_initial_conditions
from coastal_sim_data.fetchers.neracoos import fetch_neracoos_initial_conditions
from coastal_sim_data.fetchers.necofs import fetch_necofs_initial_conditions


def test_nyhops():
    # Throgs Neck Bridge Bounding Box (Is well within NYHOPS)
    bbox = [-73.815, 40.785, -73.775, 40.815]
    print(f"Testing NYHOPS Fetcher for {bbox}...")

    # We use a date known to be in the recent archive or forecast
    # OPeNDAP can be slow on large dimensions, subsetting should limit it
    ds = fetch_nyhops_initial_conditions("2026-03-04T00:00:00Z", bbox)

    if ds is not None:
        print("NYHOPS Success! Variables:")
        print(list(ds.data_vars.keys()))
        print(f"U-Velocity Shape: {ds.u.shape}")
    else:
        print("NYHOPS returned None (Outside domain or error fetching).")


def test_hycom():
    # Deep Ocean / Miami (Outside NYHOPS)
    bbox = [-80.1, 25.7, -80.0, 25.8]
    print(f"\nTesting HYCOM Fetcher for {bbox}...")

    ds = fetch_hycom_initial_conditions("2026-03-04T00:00:00Z", bbox)

    if ds is not None:
        print("HYCOM Success! Variables:")
        print(list(ds.data_vars.keys()))
        print(f"Water U-Velocity Shape: {ds.water_u.shape}")
    else:
        print("HYCOM returned None (Error fetching).")


def test_maracoos():
    # Long Island Sound / NYC Bounding Box (Within DOPPIO)
    bbox = [-74.2, 40.5, -73.0, 41.3]
    print(f"\nTesting MARACOOS (DOPPIO) Fetcher for {bbox}...")

    # We use a recent date
    target_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    # The timezone tz_localize was crashing if not string, so we use string rep
    target_str = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    ds = fetch_maracoos_initial_conditions(target_str, bbox)

    if ds is not None:
        print("MARACOOS Success! Variables:")
        print(list(ds.data_vars.keys()))
        if "salt" in ds:
            print(f"Salinity Shape: {ds.salt.shape}")
    else:
        print("MARACOOS returned None (Error fetching).")


def test_neracoos():
    # New Castle NH / Piscataqua River Bounding Box (Within NERACOOS GYX)
    bbox = [-70.8, 42.9, -70.5, 43.1]
    print(f"\nTesting NERACOOS (NOAA NWPS GYX) Fetcher for {bbox}...")

    # We use a recent date
    target_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    target_str = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    ds = fetch_neracoos_initial_conditions(target_str, bbox)

    if ds is not None:
        print("NERACOOS Success! Variables:")
        print(list(ds.data_vars.keys()))
        if "dir" in ds:
            print(f"Wave Dir Shape: {ds.dir.shape}")
    else:
        print("NERACOOS returned None (Error fetching).")


def test_necofs():
    # Great Bay Estuary NH (Within NECOFS GOM3)
    bbox = [-70.9, 43.05, -70.8, 43.15]
    print(f"\nTesting NECOFS (FVCOM GOM3) Fetcher for {bbox}...")

    # We use a recent date
    target_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    target_str = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    ds = fetch_necofs_initial_conditions(target_str, bbox)

    if ds is not None:
        print("NECOFS Success! Variables:")
        print(list(ds.data_vars.keys()))
        if "salt" in ds:
            print(f"Salinity Shape: {ds.salt.shape}")
            print(f"Dataset Dims: {ds.dims}")
    else:
        print("NECOFS returned None (Error fetching).")


if __name__ == "__main__":
    # test_nyhops() # Stevens server is offline or blocking connection
    test_necofs()
    test_maracoos()
    test_neracoos()
    test_hycom()
