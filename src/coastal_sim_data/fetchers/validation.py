import os
import logging

logger = logging.getLogger(__name__)


def verify_sources(
    source_name: str,
    url: str,
    timestamp: str,
    spatial_res: str,
    requires_auth: bool = False,
):
    """
    Validates data source access and provenance telemetry.
    """
    if requires_auth:
        # Require CDS API credentials in environment (typically loaded from .env)
        if not os.environ.get("CDSAPI_URL") or not os.environ.get("CDSAPI_KEY"):
            logger.error(
                f"Access Denied: Production authentication required for {source_name}. Missing CDSAPI_URL or CDSAPI_KEY in environment."
            )
            raise ValueError(f"Missing CDS credentials required for {source_name}")
    logger.info(
        f"[PROVENANCE] Source: {source_name} | URL: {url} | Timestamp: {timestamp} | Spatial Resolution: {spatial_res}"
    )
