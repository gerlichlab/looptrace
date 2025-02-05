"""Tools for naming integers

In particular, tools here are for padding to fixed-length string and going 0-based to 1-based.
"""

from enum import Enum
from math import log10
from typing import Any, Callable

from expression import Failure, Option, Result, Success
from gertils.types import FieldOfViewFrom1

from .utilities import find_first_option

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]


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
        _typecheck_as_int(i, ctx="Index to name")
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


DEFAULT_INTEGER_NAMER = IndexToNaturalNumberText.TenThousand


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
        return find_first_option(p)(cls)


def _get_short_name(*, semantic: "NameableSemantic", i: int, namer: "IndexToNaturalNumberText") -> str:
    return semantic.value + namer.get_name(i)


def get_fov_name_short(fov: int | FieldOfViewFrom1, *, namer: IndexToNaturalNumberText = DEFAULT_INTEGER_NAMER) -> str:
    """Get the field of view name for the given index."""
    def build(i: int) -> str:
        return _get_short_name(semantic=NameableSemantic.Point, i=i, namer=namer)
    match fov:
        case int():
            return build(fov)
        case FieldOfViewFrom1():
            return build(fov.get - 1)
        case _:
            raise TypeError(_illegal_type_message(ctx="Index to name", value=fov))


def get_fov_names_N(num_names: int, namer: IndexToNaturalNumberText = DEFAULT_INTEGER_NAMER) -> list[str]:
    """Get the field of view name for the first n indices."""
    _typecheck_as_int(num_names, ctx="Number of names")
    if num_names < 0:
        raise ValueError(f"Number of names is negative: {num_names}")
    return [get_fov_name_short(i, namer=namer) for i in range(num_names)]


def parse_semantic_and_value(s: str, *, namer: IndexToNaturalNumberText) -> Result[tuple["NameableSemantic", int], str]:
    return NameableSemantic.parse_from_prefix(s)\
        .to_result("Cannot parse semantic from prefix")\
        .bind(lambda sem: namer.read_as_index(s.removeprefix(sem.value)).map(lambda z: (sem, z)))


def _build_field_of_view_one_based(*, semantic: "NameableSemantic", raw_value: int) -> Result[FieldOfViewFrom1, str]:
    match semantic:
        case NameableSemantic.Point:
            try:
                return Result.Ok(FieldOfViewFrom1(raw_value))
            except Exception as e:
                return Result.Error(f"Failed to build 1-based FOV; error -- {e}")
        case _:
            return Result.Error(f"Cannot build 1-based FOV with given semantic for raw value: {semantic}f")


def parse_field_of_view_one_based_from_position_name_representation(
    s: str, 
    *, 
    namer: IndexToNaturalNumberText = DEFAULT_INTEGER_NAMER,
) -> Result[FieldOfViewFrom1, str]:
    return parse_semantic_and_value(s, namer=namer)\
        .bind(lambda sem_val: _build_field_of_view_one_based(semantic=sem_val[0], raw_value=sem_val[1]))


def _illegal_type_message(*, ctx: str, value: Any) -> str:
    return f"{ctx} is of illegal type: {type(value).__name__}"


def _typecheck_as_int(i: Any, ctx: str) -> bool:
    if not isinstance(i, int) or isinstance(i, bool): # Handle the fact that instance check of Boolean against int can be True.
        raise TypeError(_illegal_type_message(ctx=ctx, value=i))
