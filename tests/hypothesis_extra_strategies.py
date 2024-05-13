"""Extra hypothesis strategies for generation of examples of custom types"""

import more_itertools
from collections.abc import Iterable
from typing import Optional, TypeVar
from hypothesis import strategies as st

from looptrace.configuration import SEMANTIC_KEY

_T = TypeVar('_T')

MIN_SEP_KEY = "minimumPixelLikeSeparation"
GROUPS_KEY = "groups"


def gen_spot_separation_threshold(max_threhsold: Optional[int] = None):
    if max_threhsold is not None:
        if not isinstance(max_threhsold, int):
            raise TypeError(f"Max spot separation threshold isn't an int, but {type(max_threhsold).__name__}")
        if max_threhsold < 1:
            raise ValueError(f"Max spot separation threshold is less than 1: {max_threhsold}")
    return st.integers(min_value=1, max_value=max_threhsold)


class ProximityFilterStrategyData:
    """Strategies for generating proximity filter"""
    @staticmethod
    def universal_permission():
        return st.just({SEMANTIC_KEY: "UniversalProximityPermission"})
    
    @staticmethod
    def universal_prohibition(max_threshold: Optional[int] = None):
        return gen_spot_separation_threshold(max_threhsold=max_threshold).map(lambda x: {SEMANTIC_KEY: "UniveralProximityProhibition", MIN_SEP_KEY: x})

    @staticmethod
    def _gen_selective(*, eligible_timepoints: Iterable[int], semantic: str, max_threshold: Optional[int] = None):
        eligible_timepoints = list(eligible_timepoints)
        @st.composite
        def gen(draw):
            min_sep = draw(gen_spot_separation_threshold(max_threhsold=max_threshold))
            groups = gen_partition(eligible_timepoints)
            return {SEMANTIC_KEY: semantic, MIN_SEP_KEY: min_sep, GROUPS_KEY: groups}
        return gen


    @classmethod
    def selective_permission(cls, *, eligible_timepoints: Iterable[int], max_threshold: Optional[int] = None):
        return cls._gen_selective(eligible_timepoints=eligible_timepoints, semantic="SelectiveProximityPermission", max_threshold=max_threshold)

    @classmethod
    def selective_prohibition(cls, *, eligible_timepoints: Iterable[int], max_threshold: Optional[int] = None):
        return cls._gen_selective(eligible_timepoints=eligible_timepoints, semantic="SelectiveProximityProhibition", max_threshold=max_threshold)


def gen_partition(elements: list[_T], *, min_part_count: int = 1, max_part_count: Optional[int] = None):
    # First, validate the arguments both absolutely and in relation to one another.
    n: int = len(elements)
    if max_part_count is None:
        max_part_count: int = max(max_part_count, n)
    if min_part_count < 0:
        raise ValueError(f"Min part count must be nonnegative; got: {min_part_count}")
    if min_part_count > n:
        raise ValueError(f"Cannot partition {n} element(s) into {min_part_count} part(s)!")    
    if min_part_count > max_part_count:
        raise ValueError(f"Min part count exceeds max part count: {min_part_count} > {max_part_count}")
    
    @st.composite
    def part(draw):
        num_parts = draw(st.integers(min_value=min_part_count, max_value=max_part_count))
        result = draw(st.sampled_from(more_itertools.set_partitions(elements, k=num_parts)))
        return result

    return part


def gen_proximity_filter_strategy(*, eligible_timepoints: Iterable[int], max_threshold: Optional[int] = None):
    return st.one_of(
        ProximityFilterStrategyData.universal_permission(), 
        ProximityFilterStrategyData.universal_prohibition(max_threshold=max_threshold),
        ProximityFilterStrategyData.selective_permission(eligible_timepoints=eligible_timepoints, max_threshold=max_threshold), 
        ProximityFilterStrategyData.selective_prohibition(eligible_timepoints=eligible_timepoints, max_threshold=max_threshold),
    )
