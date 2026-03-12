import pytest
from datetime import datetime
from unittest.mock import patch
from coastal_sim_data.dispatcher import dispatch_forcing_request


# We will mock the actual fetchers so we don't hit Copernicus or S3
@pytest.fixture
def mock_fetchers():
    with (
        patch("coastal_sim_data.dispatcher.fetch_hrrr_surface_forcing") as mock_hrrr,
        patch("coastal_sim_data.dispatcher.fetch_era5_surface_forcing") as mock_era5,
    ):
        mock_hrrr.return_value = "/tmp/mocked_hrrr.grib2"
        mock_era5.return_value = "/tmp/mocked_era5.grib"

        yield mock_hrrr, mock_era5


@patch("coastal_sim_data.dispatcher.datetime")
def test_dispatch_tier_iv_hrrr(mock_datetime, mock_fetchers):
    mock_hrrr, mock_era5 = mock_fetchers

    # Mock 'now' to be today, and target to be 10 hours ago
    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    # 10 hours ago falls cleanly in Tier IV (HRRR)
    target_date = "2026-03-03"

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert mock_hrrr.called
    assert not mock_era5.called


@patch("coastal_sim_data.dispatcher.datetime")
def test_dispatch_tier_i_era5_final(mock_datetime, mock_fetchers):
    mock_hrrr, mock_era5 = mock_fetchers

    # Mock 'now' to be today, target 4 months ago
    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    target_date = "2025-10-01"  # > 90 days ago

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert not mock_hrrr.called
    assert mock_era5.called

    # ERA5 Final means preliminary=False
    import os

    mock_era5.assert_called_with(
        target_date,
        [-74.0, 40.0, -73.0, 41.0],
        os.path.join(
            os.path.expanduser("~/.cache/coastal-sim-data"), "era5_2025-10-01.grib"
        ),
        preliminary=False,
        cache_bust=False,
    )


@patch("coastal_sim_data.dispatcher.datetime")
def test_dispatch_tier_ii_era5t_preliminary(mock_datetime, mock_fetchers):
    mock_hrrr, mock_era5 = mock_fetchers

    now = datetime(2026, 3, 3, 12, 0, 0)
    mock_datetime.utcnow.return_value = now
    mock_datetime.strptime.side_effect = datetime.strptime

    target_date = "2026-02-15"  # ~16 days ago (between 5 and 90 days)

    dispatch_forcing_request(target_date, [-74.0, 40.0, -73.0, 41.0])

    assert not mock_hrrr.called
    assert mock_era5.called

    import os

    # ERA5T means preliminary=True
    mock_era5.assert_called_with(
        target_date,
        [-74.0, 40.0, -73.0, 41.0],
        os.path.join(
            os.path.expanduser("~/.cache/coastal-sim-data"), "era5t_2026-02-15.grib"
        ),
        preliminary=True,
        cache_bust=False,
    )


@pytest.fixture
def mock_ic_fetchers():
    with (
        patch(
            "coastal_sim_data.fetchers.neracoos.fetch_neracoos_initial_conditions"
        ) as mock_neracoos,
        patch(
            "coastal_sim_data.fetchers.maracoos.fetch_maracoos_initial_conditions"
        ) as mock_maracoos,
        patch(
            "coastal_sim_data.fetchers.hycom.fetch_hycom_initial_conditions"
        ) as mock_hycom,
    ):
        yield mock_neracoos, mock_maracoos, mock_hycom


@patch("coastal_sim_data.dispatcher.os.path.exists")
@patch("shutil.rmtree")
def test_dispatch_ic_neracoos(mock_rmtree, mock_exists, mock_ic_fetchers, mocker):
    mock_neracoos, mock_maracoos, mock_hycom = mock_ic_fetchers

    mock_exists.return_value = False

    mock_ds = mocker.MagicMock()
    mock_ds.variables = {"u": mocker.MagicMock()}
    mock_neracoos.return_value = mock_ds

    from coastal_sim_data.dispatcher import dispatch_ic_request

    zarr_path = dispatch_ic_request("2026-03-03", [-74.0, 40.0, -73.0, 41.0])

    assert mock_neracoos.called
    assert not mock_maracoos.called
    assert not mock_hycom.called
    assert mock_ds.to_zarr.called
    assert zarr_path.endswith(".zarr")


@patch("coastal_sim_data.dispatcher.os.path.exists")
@patch("shutil.rmtree")
def test_dispatch_ic_fallback_to_hycom(
    mock_rmtree, mock_exists, mock_ic_fetchers, mocker
):
    mock_neracoos, mock_maracoos, mock_hycom = mock_ic_fetchers

    mock_exists.return_value = False

    # Force regional failures
    mock_neracoos.side_effect = Exception("NERACOOS out of bounds")
    mock_maracoos.side_effect = Exception("MARACOOS out of bounds")

    mock_ds = mocker.MagicMock()
    mock_ds.variables = {"water_u": mocker.MagicMock()}
    mock_hycom.return_value = mock_ds

    from coastal_sim_data.dispatcher import dispatch_ic_request

    _ = dispatch_ic_request("2026-03-03", [-10.0, 10.0, -8.0, 12.0])

    assert mock_neracoos.called
    assert mock_maracoos.called
    assert mock_hycom.called
    assert mock_ds.to_zarr.called
