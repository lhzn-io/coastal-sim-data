"""Unit tests for NYOFS fetcher."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch

from coastal_sim_data.fetchers import nyofs


class TestNYOFSMetadata:
    """Test NYOFS metadata and domain support."""

    def test_get_metadata_has_required_keys(self):
        """Test that metadata contains all required keys."""
        meta = nyofs.get_metadata()
        assert meta["id"] == "nyofs"
        assert meta["name"] == "NOAA NYOFS (NY/NJ Harbor)"
        assert meta["resolution_approx_m"] == 100.0
        assert meta["type_desc"] == "Structured curvilinear POM grid"
        assert "domain_bbox" in meta
        assert len(meta["domain_bbox"]) == 4

    def test_supports_bbox_within_domain(self):
        """Test bbox validation for NY Harbor."""
        # Throgs Neck Bridge area — well within NYOFS domain
        bbox = [-73.815, 40.785, -73.775, 40.815]
        assert nyofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_west(self):
        """Test bbox outside domain (too far west)."""
        bbox = [-75.0, 40.6, -74.5, 40.8]
        assert not nyofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_north(self):
        """Test bbox outside domain (too far north)."""
        bbox = [-73.8, 41.2, -73.5, 41.5]
        assert not nyofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_gulf_of_maine(self):
        """Test bbox in Gulf of Maine (completely outside NYOFS)."""
        bbox = [-70.0, 43.0, -69.0, 44.0]
        assert not nyofs.supports_bbox(bbox)


class TestURLResolution:
    """Test URL resolution logic for FMRC vs NCEI."""

    def test_get_url_recent_returns_fmrc(self):
        """Test that recent data (< 31 days) returns FMRC URL."""
        target_dt = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=7)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "fmrc"
        assert "Aggregated_7_day_NYOFS_Fields_Forecast_best" in url
        assert "opendap.co-ops.nos.noaa.gov" in url

    def test_get_url_historical_returns_ncei(self):
        """Test that historical data (> 31 days) returns NCEI pattern."""
        target_dt = pd.Timestamp("2024-01-15", tz="UTC").tz_localize(None)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "ncei"
        assert "ncei.noaa.gov" in url

    def test_get_url_post_sept_2024_naming(self):
        """Test NCEI naming convention after Sept 9, 2024."""
        target_dt = pd.Timestamp("2024-10-01", tz="UTC").tz_localize(None)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "ncei"
        # Post Sept 9 naming: nyofs.tCCz.YYYYMMDD.fields.nHHH.nc (with placeholders)
        assert "{yyyy}" in url and "{mm}" in url and "{dd}" in url

    def test_get_url_pre_sept_2024_naming(self):
        """Test NCEI naming convention before Sept 9, 2024."""
        target_dt = pd.Timestamp("2024-08-01", tz="UTC").tz_localize(None)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "ncei"
        # Pre Sept 9 naming: nos.nyofs.fields.nHHH.YYYYMMDD.tCCz.nc (with placeholders)
        assert "{yyyy}" in url and "{mm}" in url


class TestEnumerateNCEIFiles:
    """Test NCEI file enumeration."""

    def test_enumerate_ncei_single_hour(self):
        """Test file enumeration for single hour request."""
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/"
            "{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        start_dt = pd.Timestamp("2024-10-15 12:30:00")
        end_dt = start_dt + pd.Timedelta(hours=1)

        files = nyofs._enumerate_ncei_nyofs_files(pattern, start_dt, end_dt)

        assert len(files) >= 1
        assert "2024" in files[0]
        assert "10" in files[0]
        assert "15" in files[0]

    def test_enumerate_ncei_multiple_hours(self):
        """Test file enumeration across multiple hours."""
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/"
            "{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        start_dt = pd.Timestamp("2024-10-15 12:30:00")
        end_dt = start_dt + pd.Timedelta(hours=6)

        files = nyofs._enumerate_ncei_nyofs_files(pattern, start_dt, end_dt)

        assert len(files) >= 6


class TestCGridInterpolation:
    """Test C-grid velocity interpolation."""

    def test_c_grid_to_rho_ic_shape(self):
        """Output shape matches input for IC (sigma, eta, xi) arrays."""
        nk, neta, nxi = 2, 6, 8
        u = np.ones((nk, neta, nxi))
        v = np.ones((nk, neta, nxi))

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        assert u_rho.shape == (nk, neta, nxi)
        assert v_rho.shape == (nk, neta, nxi)

    def test_c_grid_to_rho_obc_shape(self):
        """Output shape matches input for OBC (time, sigma, eta, xi) arrays."""
        nt, nk, neta, nxi = 3, 2, 6, 8
        u = np.ones((nt, nk, neta, nxi))
        v = np.ones((nt, nk, neta, nxi))

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        assert u_rho.shape == (nt, nk, neta, nxi)
        assert v_rho.shape == (nt, nk, neta, nxi)

    def test_c_grid_to_rho_averages_adjacent_cells(self):
        """Interior cells are averages of adjacent face values."""
        nk, neta, nxi = 1, 4, 4
        # u[..., i] = float(i), so average of i and i+1
        u = np.tile(np.arange(nxi, dtype=float), (nk, neta, 1))
        v = np.tile(np.arange(neta, dtype=float)[:, None], (nk, 1, nxi))

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        # Interior: 0.5*(i + i+1)
        np.testing.assert_allclose(u_rho[..., 0], 0.5)  # 0.5*(0+1)
        np.testing.assert_allclose(u_rho[..., 1], 1.5)  # 0.5*(1+2)
        # Boundary: copy of last edge value
        np.testing.assert_allclose(u_rho[..., -1], u[..., -1])
        np.testing.assert_allclose(v_rho[..., -1, :], v[..., -1, :])

    def test_c_grid_to_rho_uniform_field_unchanged(self):
        """Uniform velocity field should be unchanged after averaging."""
        u = np.full((2, 5, 7), 1.5)
        v = np.full((2, 5, 7), -0.3)

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        np.testing.assert_allclose(u_rho, u)
        np.testing.assert_allclose(v_rho, v)


class TestInitialConditions:
    """Test IC fetcher logic (mocked OPeNDAP)."""

    @patch("coastal_sim_data.fetchers.nyofs._open_nyofs_dataset")
    def test_fetch_nyofs_ic_success(self, mock_open):
        """Test successful IC fetch with mocked pydap."""
        # Create mock dataset with realistic dimensions
        # In actual NYOFS, u and v are on staggered grids but xarray combines them
        nk, neta, nxi = 20, 50, 60

        # Create the dataset with aligned dimensions at rho points
        mock_ds = xr.Dataset(
            data_vars={
                "u": (("sigma", "eta", "xi"), np.random.rand(nk, neta, nxi)),
                "v": (("sigma", "eta", "xi"), np.random.rand(nk, neta, nxi)),
                "temp": (("sigma", "eta", "xi"), np.random.rand(nk, neta, nxi) + 15),
                "salt": (("sigma", "eta", "xi"), np.random.rand(nk, neta, nxi) + 30),
                "zeta": (("eta", "xi"), np.random.rand(neta, nxi) * 0.5),
                "lon": (
                    ("eta", "xi"),
                    np.linspace(-74.0, -73.8, neta * nxi).reshape(neta, nxi),
                ),
                "lat": (
                    ("eta", "xi"),
                    np.linspace(40.6, 40.8, neta * nxi).reshape(neta, nxi),
                ),
                "mask": (("eta", "xi"), np.ones((neta, nxi))),
            },
            coords={"sigma": np.linspace(0, -1, nk)},
        )
        mock_open.return_value = mock_ds

        bbox = [-74.0, 40.6, -73.8, 40.8]
        target_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_initial_conditions(target_date, bbox)

        assert result is not None, "IC fetch returned None for valid mock data"
        assert "u" in result.data_vars
        assert "v" in result.data_vars
        assert "temp" in result.data_vars
        assert "salt" in result.data_vars
        assert "zeta" in result.data_vars

    def test_fetch_nyofs_ic_outside_bbox(self):
        """Test that IC fetch returns None for out-of-bbox request."""
        bbox = [-70.0, 43.0, -69.0, 44.0]  # Gulf of Maine
        target_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_initial_conditions(target_date, bbox)

        assert result is None

    @patch("coastal_sim_data.fetchers.nyofs.xr.open_dataset")
    def test_fetch_nyofs_ic_pydap_error(self, mock_xr_open):
        """Test IC fetch gracefully handles pydap errors."""
        mock_xr_open.side_effect = Exception("OPeNDAP connection failed")

        bbox = [-74.0, 40.6, -73.8, 40.8]
        target_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_initial_conditions(target_date, bbox)

        # Should return None when pydap fails
        assert result is None


class TestBoundaryConditions:
    """Test OBC fetcher logic (mocked OPeNDAP)."""

    @patch("coastal_sim_data.fetchers.nyofs._open_nyofs_dataset")
    def test_fetch_nyofs_obc_success(self, mock_open):
        """Test successful OBC fetch with mocked pydap."""
        # Create mock time-series dataset
        times = pd.date_range("2024-10-15", periods=3, freq="1h")
        mock_ds = xr.Dataset(
            data_vars={
                "u": (
                    ("time", "sigma", "eta", "xi"),
                    np.random.rand(3, 2, 3, 4) * 0.5,
                ),
                "v": (
                    ("time", "sigma", "eta", "xi"),
                    np.random.rand(3, 2, 3, 4) * 0.5,
                ),
                "lon": (("eta", "xi"), np.linspace(-74.0, -73.8, 12).reshape(3, 4)),
                "lat": (("eta", "xi"), np.linspace(40.6, 40.8, 12).reshape(3, 4)),
                "mask": (("eta", "xi"), np.ones((3, 4))),
            },
            coords={"time": times, "sigma": np.linspace(0, -1, 2)},
        )
        mock_open.return_value = mock_ds

        bbox = [-74.0, 40.6, -73.8, 40.8]
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 3, bbox)

        assert result is not None
        assert "u" in result.data_vars
        assert "v" in result.data_vars
        assert set(result.dims) == {"time", "depth", "eta", "xi"}
        assert result.u.dtype == np.float32

    def test_fetch_nyofs_obc_outside_bbox(self):
        """Test that OBC fetch returns None for out-of-bbox request."""
        bbox = [-70.0, 43.0, -69.0, 44.0]  # Gulf of Maine
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 3, bbox)

        assert result is None

    @patch("coastal_sim_data.fetchers.nyofs.xr.open_dataset")
    def test_fetch_nyofs_obc_pydap_error(self, mock_xr_open):
        """Test OBC fetch gracefully handles pydap errors."""
        mock_xr_open.side_effect = Exception("OPeNDAP connection failed")

        bbox = [-74.0, 40.6, -73.8, 40.8]
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 6, bbox)

        # Should return None when pydap fails
        assert result is None


class TestDispatcherIntegration:
    """Test dispatcher registration."""

    def test_nyofs_in_ic_fetchers(self):
        """Test that NYOFS is registered in IC fetchers."""
        from coastal_sim_data.dispatcher import get_ic_fetchers

        ic_fetchers = get_ic_fetchers()
        fetcher_ids = [m.get_metadata()["id"] for m, _ in ic_fetchers]

        assert "nyofs" in fetcher_ids
        assert fetcher_ids.index("nyofs") == 0  # NYOFS should be first (best ranking)

    def test_nyofs_in_obc_fetchers(self):
        """Test that NYOFS is registered in OBC fetchers."""
        from coastal_sim_data.dispatcher import get_obc_fetchers

        obc_fetchers = get_obc_fetchers()
        fetcher_ids = [m.get_metadata()["id"] for m, _ in obc_fetchers]

        assert "nyofs" in fetcher_ids

    def test_nyofs_ranks_above_necofs_for_harbor(self):
        """Test that NYOFS ranks above NECOFS for NY Harbor bbox."""
        from coastal_sim_data.dispatcher import _rank_ic_candidates

        bbox = [-73.815, 40.785, -73.775, 40.815]
        ranked = _rank_ic_candidates(bbox)

        assert len(ranked) > 0
        assert ranked[0][2]["id"] == "nyofs"  # NYOFS should be first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
