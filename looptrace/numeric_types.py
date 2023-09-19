"""Groupings of numeric types and tools for working with them"""

from typing import *
import numpy as np

__author__ = "Vince Reuter"

__all__ = ["FloatLike", "IntegerLike", "NumberLike"]


FloatLike = Union[float, np.float16, np.float32, np.float64]
IntegerLike = Union[int, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
NumberLike = Union[IntegerLike, FloatLike]
