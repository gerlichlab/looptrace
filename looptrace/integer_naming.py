"""Tools for naming integers

In particular, tools here are for padding to fixed-length string and going 0-based to 1-based.
"""

from enum import Enum
from math import log10
from typing import *

__author__ = "Vince Reuter"
__all__ = ["get_position_name_1", "get_position_names_N"]


class _IntegerNaming(Enum):
    """
    Apply a strategy for converting nonnegative integers to fixed-length string name, as natural number.
    
    These strategies map [0, n) into [1, N] in text reprentation, with each name the same length.
    """
    TenThousand = 10000

    @property
    def _text_size(self) -> int:
        return int(log10(self.value))

    @property
    def _num_values_possible(self) -> int:
        # Subtraction 1 to account for presence of 0 in the domain.
        return self.value - 1

    @property
    def _max_index_nameable(self) -> int:
        # Subtract 1 to account for increment of 1 (going from 0-based to 1-based).
        return self._num_values_possible - 1

    def get_name(self, i: int) -> str:
        """Get the (padded) name for the single value given."""
        _typecheck(i, ctx="Index to name")
        if i < 0 or i > self._max_index_nameable:
            raise ValueError(f"{i} is out-of-bounds [0, {self._max_index_nameable}] for for namer '{self.name}'")
        return self._get_name_unsafe(i)

    def _get_name_unsafe(self, n: int) -> str:
        return str(n + 1).zfill(self._text_size)


_DEFAULT_NAMER = _IntegerNaming.TenThousand


def get_position_name_1(pos_idx: int, namer: _IntegerNaming = _DEFAULT_NAMER) -> str:
    """Get the position-like (field of view) name for the given index."""
    return "P" + namer.get_name(pos_idx)


def get_position_names_N(num_names: int, namer: _IntegerNaming = _DEFAULT_NAMER) -> List[str]:
    """Get the position-like (field of view) name for the first n indices."""
    _typecheck(num_names, ctx="Number of names")
    if num_names < 0:
        raise ValueError(f"Number of names is negative: {num_names}")
    return [get_position_name_1(i, namer) for i in range(num_names)]


def _typecheck(i: int, ctx: str) -> bool:
    if isinstance(i, bool) or not isinstance(i, int):
        raise TypeError(f"{ctx} ({i}) (type={type(i).__name__}) is not integer-like!")
