import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

BASE_ERDDAP_URL = "http://merlin.dms.uconn.edu:8080/erddap"


def fetch_erddap_station_profiles(
    station_id: str,
    start_time: str,
    end_time: str,
    cache_dir: str = os.path.expanduser("~/.cache/coastal-sim-data/erddap"),
    cache_bust: bool = False,
) -> dict:
    """
    Fetches 3-depth temperature profiles for a given station.
    Currently specifically targeting WLIS via UConn ERDDAP.

    Args:
        station_id: Station ID, e.g., "WLIS"
        start_time: ISO8601 string (e.g., "YYYY-MM-DDTHH:MM:SSZ")
        end_time: ISO8601 string

    Returns:
        dictionary with keys: "surface", "mid", "bottom", each containing a dictionary of time (ISO) -> temperature (float)
    """
    os.makedirs(cache_dir, exist_ok=True)

    if station_id.upper() != "WLIS":
        logger.warning(
            f"Station {station_id} not supported for 3-depth profiles via ERDDAP currently."
        )
        return {}

    datasets = {"surface": "WLIS_WQ_SFC", "mid": "WLIS_WQ_MID", "bottom": "WLIS_WQ_BTM"}

    profiles = {}

    for level, dataset_name in datasets.items():
        cache_key = f"{dataset_name}_{start_time}_{end_time}".replace(":", "-").replace(
            "Z", ""
        )
        cache_path = os.path.join(cache_dir, f"{cache_key}.csv")

        if not cache_bust and os.path.exists(cache_path):
            logger.info(f"Cache hit for {dataset_name}: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            query_url = f"{BASE_ERDDAP_URL}/tabledap/{dataset_name}.csvp?time,sea_water_temperature&time%3E={start_time}&time%3C={end_time}"

            logger.info(
                f"Fetching ERDDAP data for {dataset_name} ({start_time} to {end_time})..."
            )
            try:
                df = pd.read_csv(query_url)
                if df is not None and not df.empty:
                    # The headers will be like "time (UTC)" and "sea_water_temperature (celsius)"
                    # We can normalize them
                    df.columns = ["time", "sea_water_temperature"]
                    df = df.set_index("time")
                    df.index = pd.to_datetime(df.index, utc=True)
                    df.to_csv(cache_path)
            except Exception as e:
                logger.error(
                    f"Failed fetching or parsing ERDDAP for {dataset_name}: {e}"
                )
                df = None

        if df is not None and not df.empty:
            df = df.dropna(subset=["sea_water_temperature"])
            # We must convert timestamp back to isoformat strings to match json expectations if returned by API
            time_dict = {
                ts.isoformat(): temp for ts, temp in df["sea_water_temperature"].items()
            }
            profiles[level] = time_dict

    return profiles
