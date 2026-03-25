import logging
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Awaitable, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from coastal_sim_data.dispatcher import (  # noqa: E402
    dispatch_forcing_request,
    dispatch_station_profiles_request,
    dispatch_bounding_box_profiles_request,
)
from coastal_data_serve.routers import viewer, bathymetry  # noqa: E402
from coastal_sim_data.fetchers.noaa import fetch_noaa_tide_data  # noqa: E402

# Configure Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_level = logging.INFO
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# File Handler
file_handler = logging.FileHandler(log_dir / "service.log")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Stream Handler
has_console = False
for h in root_logger.handlers:
    if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
        h.setFormatter(formatter)
        h.setLevel(log_level)
        has_console = True
        break

if not has_console:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

logging.getLogger("coastal_data_serve").setLevel(log_level)
logger = logging.getLogger("coastal_data_serve")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# Initialize FastAPI application
app = FastAPI(
    title="coastal-sim-data",
    description="Microservice for fetching, parsing, and regridding ERA5/HRRR forcing data for coastal-sim.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Cache Viewer",
            "description": "Endpoints to navigate and preview static cached data.",
        }
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(viewer.router)
app.include_router(bathymetry.router)

# Ensure static UI directory exists
ui_dir = Path("static")
ui_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory="static", html=True), name="static_ui")


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )
    return response


class BoundingBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class ForcingRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    station_id: Optional[str] = None
    cache_bust: bool = False


class ICRequest(BaseModel):
    bbox: BoundingBox
    target_date: str  # ISO8601 string
    cache_bust: bool = False


class TideRequest(BaseModel):
    station_id: str
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


class TelemetryRequest(BaseModel):
    station_id: str
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


class HoTRequest(BaseModel):
    lat: float

    lon: float

    radius_km: float = 20.0


@app.post("/api/v1/hot_discovery")
async def hot_discovery(req: HoTRequest):
    """Discover head of tide limits using OSM API."""

    from coastal_sim_data.fetchers.hydrography import find_head_of_tide

    try:
        results = find_head_of_tide(req.lat, req.lon, req.radius_km)

        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "coastal-sim-data"}


@app.api_route("/api/v1/cache/purge", methods=["GET", "POST"])
async def purge_cache() -> Dict[str, str]:
    """Purges the forcing and IC data cache. Supports both GET (manual) and POST (UI)."""
    cache_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    if os.path.exists(cache_dir):
        logger.info(f"Purging cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            return {"status": "success", "message": "Cache purged successfully."}
        except Exception as e:
            logger.error(f"Failed to purge cache: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to purge cache: {e}")
    else:
        return {"status": "success", "message": "Cache directory does not exist."}


@app.post("/api/v1/tide")
async def get_tide_data(request: TideRequest) -> Dict[str, Any]:
    """Fetch and cache NOAA tide data for a given station and time window."""
    try:
        data = fetch_noaa_tide_data(
            station_id=request.station_id,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Tide fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TelemetryBBoxRequest(BaseModel):
    bbox: BoundingBox
    start_time: str  # ISO8601 string
    end_time: str  # ISO8601 string
    cache_bust: bool = False


@app.post("/api/v1/telemetry/station")
async def get_station_telemetry(request: TelemetryRequest) -> Dict[str, Any]:
    """Fetch 3-depth telemetry profile data for structural nudging."""
    logger.info(
        f"Received telemetry request for station {request.station_id} from {request.start_time} to {request.end_time}"
    )
    try:
        data = dispatch_station_profiles_request(
            station_id=request.station_id,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Telemetry fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/telemetry/bbox")
async def get_bbox_telemetry(request: TelemetryBBoxRequest) -> Dict[str, Any]:
    """Fetch structured nudge telemetry for all stations in bounding box."""
    logger.info(
        f"Received bbox telemetry request for {request.bbox} from {request.start_time} to {request.end_time}"
    )
    try:
        # Convert Request bbox to list [min_lon, min_lat, max_lon, max_lat]
        bbox_list = [
            request.bbox.min_lon,
            request.bbox.min_lat,
            request.bbox.max_lon,
            request.bbox.max_lat,
        ]
        data = dispatch_bounding_box_profiles_request(
            bbox=bbox_list,
            start_time=request.start_time,
            end_time=request.end_time,
            cache_bust=request.cache_bust,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"BBox telemetry fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/bc/generate")
async def generate_bc(request: ForcingRequest) -> Dict[str, Any]:
    """
    Generate boundary conditions (u_wind, v_wind, patm, water_level) for a given bounding box and time window.
    """
    logger.info(
        f"Received BC request for bbox {request.bbox} from {request.start_time} to {request.end_time}"
    )

    import hashlib
    import json

    # Generate a deterministic hash ID based on the exact spatial and temporal bounds
    hash_str = f"{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.start_time}_{request.end_time}_{request.station_id}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"bc_{md5_hash}"

    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    zarr_path = os.path.join(base_dir, f"{zarr_id}.zarr")
    meta_path = os.path.join(base_dir, f"{zarr_id}_metadata.json")

    # Native cache hit evaluation
    if not request.cache_bust and os.path.exists(zarr_path):
        logger.info(f"Deterministic Hash Cache Hit for BC: {zarr_id}")
        return {
            "status": "success",
            "message": "Tiered BC fetch successful. Loaded from hash cache.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }

    try:
        # 1. Dispatch Request: evaluate Tier (I-IV) and fetch native GRIB
        bbox_list = [
            request.bbox.max_lat,  # North
            request.bbox.min_lon,  # West
            request.bbox.min_lat,  # South
            request.bbox.max_lon,  # East
        ]

        # We only pass the start date string for now: "YYYY-MM-DD"
        start_date_str = request.start_time.split("T")[0]

        # 2. Fetch NOAA tides if requested
        tide_data = None
        if request.station_id:
            try:
                tide_data = fetch_noaa_tide_data(
                    request.station_id,
                    request.start_time,
                    request.end_time,
                    cache_bust=request.cache_bust,
                )
            except Exception as e:
                logger.warning(
                    f"NOAA tide fetch failed (proceeding without tides): {e}"
                )

        grib_path = dispatch_forcing_request(
            target_date=start_date_str, bbox=bbox_list, cache_bust=request.cache_bust
        )

        from coastal_sim_data.regridder import process_and_regrid_grib

        zarr_path = process_and_regrid_grib(
            grib_path, bbox_list, zarr_path, tide_data=tide_data
        )

        # Write metadata manifest for traceability
        with open(meta_path, "w") as f:
            json.dump(request.model_dump(), f, indent=4)

        return {
            "status": "success",
            "message": "Tiered BC fetch and regridding successful. Ready for Oceananigans.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }
    except Exception as e:
        logger.error(f"BC generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/bc/download/{zarr_id}")
async def download_bc(zarr_id: str):
    """
    Downloads the processed CF-compliant Zarr array as a ZIP file.
    """
    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    zarr_dir = os.path.join(base_dir, f"{zarr_id}.zarr")

    if not os.path.exists(zarr_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Zarr cache not found for ID: {zarr_id}. Call /generate first.",
        )

    zip_path = os.path.join(base_dir, f"{zarr_id}.zip")

    # We create the zip if it doesn't exist, or if the zarr directory is newer than the zip
    if not os.path.exists(zip_path) or os.path.getmtime(zarr_dir) > os.path.getmtime(
        zip_path
    ):
        logger.info(f"Packing Zarr directory into archive: {zip_path}")
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", zarr_dir)

    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{zarr_id}.zip"
    )


class TCPredictRequest(BaseModel):
    target_date: str


@app.post("/api/v1/bc/predict-donor")
async def predict_bc_donor_endpoint(request: TCPredictRequest) -> Dict[str, Any]:
    """Predicts which boundary condition donor will be used for a given target date."""
    try:
        from coastal_sim_data.dispatcher import predict_bc_donor

        donor_meta = predict_bc_donor(request.target_date)
        if donor_meta:
            return {"status": "success", "donor": donor_meta}
        else:
            return {
                "status": "error",
                "message": "No donor model supports this target date.",
            }
    except Exception as e:
        logger.error(f"Failed to predict BC donor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ic/predict-donor")
async def predict_ic_donor_endpoint(request: ICRequest) -> Dict[str, Any]:
    """Predicts which donor model will be used for a given bounding box."""
    try:
        from coastal_sim_data.dispatcher import predict_ic_donor

        bbox_list = [
            request.bbox.min_lon,
            request.bbox.min_lat,
            request.bbox.max_lon,
            request.bbox.max_lat,
        ]
        donor_meta = predict_ic_donor(bbox_list)
        if donor_meta:
            return {"status": "success", "donor": donor_meta}
        else:
            return {
                "status": "error",
                "message": "No donor model supports this bounding box.",
            }
    except Exception as e:
        logger.error(f"Failed to predict IC donor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ic/generate")
async def generate_ic(request: ICRequest) -> Dict[str, Any]:
    """
    Generate initial conditions (u, v, temp, salt, etc.) for a given bounding box and target date.
    Evaluates MARACOOS, NERACOOS, and HYCOM in tiered priority.
    """
    logger.info(f"Received IC request for bbox {request.bbox} at {request.target_date}")

    import hashlib
    import json
    from coastal_sim_data.dispatcher import predict_ic_donor

    # Include predicted donor in hash so different models produce different cache keys
    bbox_list = [
        request.bbox.min_lon,
        request.bbox.min_lat,
        request.bbox.max_lon,
        request.bbox.max_lat,
    ]
    donor_meta = predict_ic_donor(bbox_list)
    donor_id = donor_meta["id"] if donor_meta else "unknown"

    hash_str = f"{request.bbox.min_lon}_{request.bbox.max_lon}_{request.bbox.min_lat}_{request.bbox.max_lat}_{request.target_date}_{donor_id}"
    md5_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]
    zarr_id = f"ic_{md5_hash}"

    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    zarr_path = os.path.join(base_dir, f"{zarr_id}.zarr")
    meta_path = os.path.join(base_dir, f"{zarr_id}_metadata.json")

    # Native cache hit evaluation
    if not request.cache_bust and os.path.exists(zarr_path):
        logger.info(f"Deterministic Hash Cache Hit for IC: {zarr_id}")
        return {
            "status": "success",
            "message": "Hierarchical initial conditions fetch successful. Loaded from hash cache.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }

    try:
        from coastal_sim_data.dispatcher import dispatch_ic_request

        _ = dispatch_ic_request(
            target_date=request.target_date,
            bbox=bbox_list,
            cache_bust=request.cache_bust,
            zarr_path=zarr_path,
        )

        with open(meta_path, "w") as f:
            json.dump(request.model_dump(), f, indent=4)

        return {
            "status": "success",
            "message": "Hierarchical initial conditions fetch successful. Ready for Oceananigans.",
            "zarr_file": zarr_path,
            "zarr_id": zarr_id,
            "request": request.model_dump(),
        }
    except Exception as e:
        logger.error(f"IC generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ic/download/{zarr_id}")
async def download_ic(zarr_id: str):
    """
    Downloads the processed CF-compliant IC Zarr array as a ZIP file.
    """
    base_dir = Path(
        os.environ.get("COASTAL_SIM_DATA_CACHE_DIR", "~/.cache/coastal-sim-data")
    ).expanduser()
    zarr_dir = os.path.join(base_dir, f"{zarr_id}.zarr")

    if not os.path.exists(zarr_dir):
        raise HTTPException(
            status_code=404,
            detail=f"IC Zarr cache not found for ID: {zarr_id}. Call /api/v1/ic/generate first.",
        )

    zip_path = os.path.join(base_dir, f"{zarr_id}.zip")

    if not os.path.exists(zip_path) or os.path.getmtime(zarr_dir) > os.path.getmtime(
        zip_path
    ):
        import shutil

        logger.info(f"Packing IC Zarr directory into archive: {zip_path}")
        shutil.make_archive(
            zip_path.replace(".zip", ""), "zip", root_dir=zarr_dir, base_dir="."
        )

    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{zarr_id}.zip"
    )
