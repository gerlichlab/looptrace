"""Types and functions related to point-spread function (PSF)"""

from enum import Enum
from typing import *


class PointSpreadFunctionStrategy(Enum):
    EMPIRICAL = "exp"
    THEORETICAL = "gen"

    @classmethod
    def from_string(cls, s: str) -> Optional["PointSpreadFunctionStrategy"]:
        for member in cls:
            if member.value == s:
                return member
    
    @classmethod
    def from_string_unsafe(cls, s) -> "PointSpreadFunctionStrategy":
        result = cls.from_string(s)
        if result is None:
            raise ValueError(f"Illegal PSF strategy: {s}")
