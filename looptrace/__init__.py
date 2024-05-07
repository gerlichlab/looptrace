"""Chromatin tracing in Python, from microscopy images of FISH probes"""

from dataclasses import dataclass
import importlib.resources
from pathlib import Path
from typing import *
import pandas as pd

__all__ = [
    "LOOPTRACE_JAR_PATH", 
    "LOOPTRACE_JAVA_PACKAGE",
    "MAX_DISTANCE_SPOT_FROM_REGION_NAME",
    "SIGMA_XY_MAX_NAME",
    "SIGMA_Z_MAX_NAME",
    "SIGNAL_NOISE_RATIO_NAME",
    "ZARR_CONVERSIONS_KEY",
    "ArrayDimensionalityError",
    "ConfigurationValueError",
    "DimensionalityError",
    "MissingImagesError",
    "RoiImageSize",
    "read_table_pandas",
    ]


# This is put into place by the docker build, as declared in the Dockerfile.
LOOPTRACE_JAR_PATH = importlib.resources.files(__name__).joinpath("looptrace-assembly-0.5-SNAPSHOT.jar")
LOOPTRACE_JAVA_PACKAGE = "at.ac.oeaw.imba.gerlich.looptrace"

MAX_DISTANCE_SPOT_FROM_REGION_NAME = "max_dist"
SIGMA_XY_MAX_NAME = "sigma_xy_max"
SIGMA_Z_MAX_NAME = "sigma_z_max"
SIGNAL_NOISE_RATIO_NAME = "A_to_BG"
ZARR_CONVERSIONS_KEY = "zarr_conversions"


@dataclass
class RoiImageSize:
    z: int
    y: int
    x: int

    def div_by(self, m: int) -> "RoiImageSize":
        return RoiImageSize(z=self.z // m, y=self.y // m, x=self.x // m)


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


class DimensionalityError(LooptraceException):
    """Error subtype for when one or more dimensions of an object are unexpected"""
    pass


class ArrayDimensionalityError(DimensionalityError):
    """Error subtype to represent an error in array dimensionality"""


class ConfigurationValueError(LooptraceException):
    "Exception subtype for when something's wrong with a config file value"
    pass


class MissingImagesError(LooptraceException):
    """Exception subtype for when a collection of images (usually subfolder of main images) is missing"""
    pass
