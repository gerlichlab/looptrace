"""Tools for naming integers

In particular, tools here are for padding to fixed-length string and going 0-based to 1-based.
"""

from enum import Enum
from math import log10
from typing import *

from expression import Failure, Option, Result, Success

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

__all__ = [
    "get_position_name_short", 
    "get_position_names_N", 
    "parse_semantic_and_value",
    ]


class IndexToNaturalNumberText(Enum):
    """
    Apply a strategy for converting a 0-based index (nonnegative integer) to fixed-length string name, as natural number.
    
    These strategies map [0, n) into [1, N] in text representation, with each name the same length.
    """
    TenThousand = 10000

    @property
    def _text_size(self) -> int:
        return int(log10(self.value))

    @property
    def _num_values_possible(self) -> int:
        # Subtract 1 to account for fact that very last value spills over into next string length.
        return self.value - 1

    def get_name(self, i: int) -> str:
        """Get the (padded) name for the single value given."""
        _typecheck(i, ctx="Index to name")
        if i < 0 or i >= self._num_values_possible:
            raise ValueError(f"{i} is out-of-bounds [0, {self._num_values_possible}) for for namer '{self.name}'")
        return self._get_name_unsafe(i)

    def read_as_index(self, text: str) -> Result[int, str]:
        def check_size(s: str) -> Result[None, str]:
            return Result.Ok(s) if len(s) == self._text_size else Result.Error(f"Wrong text size: {len(s)}, not {self._text_size}")
        def to_int(s: str) -> Result[int, str]:
            try:
                z = int(s)
            except Exception as e:
                return Failure(f"Cannot convert value ({s}) to int: {e}")
            return Success(z)
        def in_bounds(z: int) -> Result[int, str]:
            return Result.Ok(z) if z >= 1 and z < self.value else Result.Error(f"Not in [0, {self.value}): {z}")
        return check_size(text).bind(to_int).bind(in_bounds)

    def _get_name_unsafe(self, n: int) -> str:
        return str(n + 1).zfill(self._text_size)


_DEFAULT_NAMER = IndexToNaturalNumberText.TenThousand


class NameableSemantic(Enum):
    Point = "P"
    Time = "T"
    Channel = "C"

    @classmethod
    def parse_exact(cls, s: str) -> Option["NameableSemantic"]:
        return cls._fetch_first(lambda m: s == m.value)
    
    @classmethod
    def parse_from_prefix(cls, s: str) -> Option["NameableSemantic"]:
        return cls._fetch_first(lambda m: s.startswith(m.value))

    @classmethod
    def _fetch_first(cls, p: Callable[["NameableSemantic"], bool]) -> Option["NameableSemantic"]:
        for m in cls:
            if p(m):
                return Option.Some(m)
        return Option.Nothing()


def _get_short_name(*, semantic: "NameableSemantic", i: int, namer: "IndexToNaturalNumberText") -> str:
    return semantic.value + namer.get_name(i)


def get_position_name_short(i: int, *, namer: IndexToNaturalNumberText = _DEFAULT_NAMER) -> str:
    """Get the position-like (field of view) name for the given index."""
    return _get_short_name(semantic=NameableSemantic.Point, i=i, namer=namer)


def get_position_names_N(num_names: int, namer: IndexToNaturalNumberText = _DEFAULT_NAMER) -> List[str]:
    """Get the position-like (field of view) name for the first n indices."""
    _typecheck(num_names, ctx="Number of names")
    if num_names < 0:
        raise ValueError(f"Number of names is negative: {num_names}")
    return [get_position_name_short(i, namer=namer) for i in range(num_names)]


def parse_semantic_and_value(s: str, namer: IndexToNaturalNumberText) -> Result[tuple["NameableSemantic", int], str]:
    return NameableSemantic.parse_from_prefix(s)\
        .to_result("Cannot parse semantic from prefix")\
        .bind(lambda sem: namer.read_as_index(s.removeprefix(sem.value)).map(lambda z: (sem, z)))


def _typecheck(i: int, ctx: str) -> bool:
    if isinstance(i, bool) or not isinstance(i, int):
        raise TypeError(f"{ctx} ({i}) (type={type(i).__name__}) is not integer-like!")
