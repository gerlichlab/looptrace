"""Chromatin tracing in Python, from microscopy images of FISH probes"""

import importlib.resources
from pathlib import Path
from typing import *
import pandas as pd

__all__ = [
    "LOOPTRACE_JAR_PATH", 
    "LOOPTRACE_JAVA_PACKAGE",
    "MAX_DISTANCE_SPOT_FROM_REGION_NAME",
    "MINIMUM_SPOT_SEPARATION_KEY",
    "SIGMA_XY_MAX_NAME",
    "SIGMA_Z_MAX_NAME",
    "SIGNAL_NOISE_RATIO_NAME",
    "TRACING_SUPPORT_EXCLUSIONS_KEY",
    "ZARR_CONVERSIONS_KEY",
    "IllegalSequenceOfOperationsError",
    "read_table_pandas",
    ]


# This is put into place by the docker build, as declared in the Dockerfile.
LOOPTRACE_JAR_PATH = importlib.resources.files(__name__).joinpath("looptrace-assembly-0.2.0-SNAPSHOT.jar")
LOOPTRACE_JAVA_PACKAGE = "at.ac.oeaw.imba.gerlich.looptrace"

MAX_DISTANCE_SPOT_FROM_REGION_NAME = "max_dist"
MINIMUM_SPOT_SEPARATION_KEY = "min_spot_dist"
SIGMA_XY_MAX_NAME = "sigma_xy_max"
SIGMA_Z_MAX_NAME = "sigma_z_max"
SIGNAL_NOISE_RATIO_NAME = "A_to_BG"
TRACING_SUPPORT_EXCLUSIONS_KEY = "illegal_frames_for_trace_support"
ZARR_CONVERSIONS_KEY = "zarr_conversions"


def read_table_pandas(f: Union[str, Path]) -> pd.DataFrame:
    """Read a pandas table from CSV, passing the argument indicating index is the first column.

    Parameters
    ----------
    f : str or Path
        The path to the file to parse
    
    Returns
    -------
    pd.DataFrame
        Table parsed from the given file
    """
    return pd.read_csv(f, index_col=0)


class LooptraceException(BaseException):
    "General base for exceptional situations related to the specifics of this project"
    pass


class IllegalSequenceOfOperationsError(LooptraceException):
    """Exception for when an operation's attempted before at least one of its dependencies is finished."""
    pass
