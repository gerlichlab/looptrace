"""Chromatin tracing in Python, from microscopy images of FISH probes"""

import attrs
import importlib.resources
from typing import *

__all__ = [
    "FIELD_OF_VIEW_COLUMN",
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
    ]


# This is put into place by the docker build, as declared in the Dockerfile.
LOOPTRACE_JAR_PATH = importlib.resources.files(__name__).joinpath("looptrace-assembly-0.12.0.jar")
LOOPTRACE_JAVA_PACKAGE = "at.ac.oeaw.imba.gerlich.looptrace"

FIELD_OF_VIEW_COLUMN = "fieldOfView"
MAX_DISTANCE_SPOT_FROM_REGION_NAME = "max_dist"
SIGMA_XY_MAX_NAME = "sigma_xy_max"
SIGMA_Z_MAX_NAME = "sigma_z_max"
SIGNAL_NOISE_RATIO_NAME = "A_to_BG"
ZARR_CONVERSIONS_KEY = "zarr_conversions"


_IS_POSITIVE_INTEGER = [attrs.validators.instance_of(int), attrs.validators.gt(0)]


@attrs.define(kw_only=True, frozen=True)
class RoiImageSize:
    z = attrs.field(validator=_IS_POSITIVE_INTEGER)
    y = attrs.field(validator=_IS_POSITIVE_INTEGER)
    x = attrs.field(validator=_IS_POSITIVE_INTEGER)

    def div_by(self, m: int) -> "RoiImageSize":
        return RoiImageSize(z=self.z // m, y=self.y // m, x=self.x // m)


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
