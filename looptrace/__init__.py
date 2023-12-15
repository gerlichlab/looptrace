"""Chromatin tracing in Python, from microscopy images of FISH probes"""

from pathlib import Path
import pkg_resources

__all__ = [
    "LOOPTRACE_JAR_PATH", 
    "LOOPTRACE_JAVA_PACKAGE",
    "MAX_DISTANCE_SPOT_FROM_REGION_NAME",
    "MINIMUM_SPOT_SEPARATION_KEY",
    "SIGMA_XY_MAX_NAME",
    "SIGMA_Z_MAX_NAME",
    "SIGNAL_NOISE_RATIO_NAME",
    "ZARR_CONVERSIONS_KEY",
    ]


# This is put into place by the docker build, as declared in the Dockerfile.
LOOPTRACE_JAR_PATH = Path(pkg_resources.resource_filename(__name__, "looptrace-assembly-0.2.0-SNAPSHOT.jar"))
LOOPTRACE_JAVA_PACKAGE = "at.ac.oeaw.imba.gerlich.looptrace"

MAX_DISTANCE_SPOT_FROM_REGION_NAME = "max_dist"
MINIMUM_SPOT_SEPARATION_KEY = "min_spot_dist"
SIGMA_XY_MAX_NAME = "sigma_xy_max"
SIGMA_Z_MAX_NAME = "sigma_z_max"
SIGNAL_NOISE_RATIO_NAME = "A_to_BG"
ZARR_CONVERSIONS_KEY = "zarr_conversions"
