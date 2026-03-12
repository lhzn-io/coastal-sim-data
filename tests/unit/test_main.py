import os
import pytest
from httpx import AsyncClient, ASGITransport
import sys

# To enable importing src directly
sys.path.insert(0, "service")
from coastal_data_serve.main import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "coastal-sim-data"}


@pytest.mark.asyncio
async def test_forcing_generate_endpoint(mocker):
    # Mock the regridder and dispatcher to avoid triggering real downloads/calculations
    mocker.patch(
        "coastal_data_serve.main.dispatch_forcing_request",
        return_value=os.path.join(
            os.path.expanduser("~/.cache/coastal-sim-data"), "mock_era5.grib"
        ),
    )
    mocker.patch(
        "coastal_sim_data.regridder.process_and_regrid_grib",
        return_value=os.path.join(
            os.path.expanduser("~/.cache/coastal-sim-data"), "mock_era5.zarr"
        ),
    )
    mocker.patch("os.path.exists", return_value=False)

    payload = {
        "bbox": {"min_lon": -74.0, "min_lat": 40.0, "max_lon": -73.0, "max_lat": 41.0},
        "start_time": "2026-03-03T00:00:00Z",
        "end_time": "2026-03-04T00:00:00Z",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/api/v1/forcing/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["zarr_file"] == os.path.join(
        os.path.expanduser("~/.cache/coastal-sim-data"), "mock_era5.zarr"
    )
    assert data["request"]["start_time"] == "2026-03-03T00:00:00Z"


@pytest.mark.asyncio
async def test_ic_generate_endpoint(mocker):
    mocker.patch(
        "coastal_sim_data.dispatcher.dispatch_ic_request",
        return_value=os.path.join(
            os.path.expanduser("~/.cache/coastal-sim-data"), "ic_mock.zarr"
        ),
    )
    mocker.patch("os.path.exists", return_value=False)

    payload = {
        "bbox": {"min_lon": -74.0, "min_lat": 40.0, "max_lon": -73.0, "max_lat": 41.0},
        "target_date": "2026-03-03T00:00:00Z",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/api/v1/ic/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "ic_mock.zarr" in data["zarr_file"]
