"""Chromatin tracing in Python, from microscopy images of FISH probes"""

import attrs
import importlib.resources
from typing import *

from expression import Result, result
from spotfishing import RoiCenterKeys

__all__ = [
    "FIELD_OF_VIEW_COLUMN",
    "LOOPTRACE_JAR_PATH", 
    "LOOPTRACE_JAVA_PACKAGE",
    "SIGMA_XY_MAX_NAME",
    "SIGMA_Z_MAX_NAME",
    "SIGNAL_NOISE_RATIO_NAME",
    "X_CENTER_COLNAME", 
    "Y_CENTER_COLNAME", 
    "Z_CENTER_COLNAME",
    "ZARR_CONVERSIONS_KEY",
    "ArrayDimensionalityError",
    "ConfigurationValueError",
    "DimensionalityError",
    "MissingImagesError",
    "RoiImageSize",
    "unsafe_extract_result",
    ]


# This is put into place by the docker build, as declared in the Dockerfile.
LOOPTRACE_JAR_PATH = importlib.resources.files(__name__).joinpath("looptrace-assembly-0.15.0.jar")
LOOPTRACE_JAVA_PACKAGE = "at.ac.oeaw.imba.gerlich.looptrace"

FIELD_OF_VIEW_COLUMN = "fieldOfView"
SIGMA_XY_MAX_NAME = "sigma_xy_max"
SIGMA_Z_MAX_NAME = "sigma_z_max"
SIGNAL_NOISE_RATIO_NAME = "A_to_BG"
X_CENTER_COLNAME = RoiCenterKeys.X.value
Y_CENTER_COLNAME = RoiCenterKeys.Y.value
Z_CENTER_COLNAME = RoiCenterKeys.Z.value
ZARR_CONVERSIONS_KEY = "zarr_conversions"


_IS_POSITIVE_INTEGER = [attrs.validators.instance_of(int), attrs.validators.gt(0)]

_E = TypeVar('_E', bound=BaseException)
_R = TypeVar('_R', covariant=True)


def unsafe_extract_result(res: Result[_R, _E]) -> _R:
    match res:
        case result.Result(tag="ok", ok=r):
            return r
        case result.Result(tag="error", error=e):
            raise e
        case unexpected:
            raise TypeError(f"Unexpected result type ({type(unexpected).__name__}), not expression.Result")


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
