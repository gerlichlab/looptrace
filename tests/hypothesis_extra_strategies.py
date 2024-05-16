"""Extra hypothesis strategies for generation of examples of custom types"""

from collections.abc import Iterable
from typing import Optional, TypeVar

import more_itertools
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from gertils.types import TimepointFrom0

from looptrace.configuration import SEMANTIC_KEY

_T = TypeVar('_T')

MIN_SEP_KEY = "minimumPixelLikeSeparation"
GROUPS_KEY = "groups"


class ProximityFilterStrategyData:
    """Strategies for generating proximity filter"""
    @staticmethod
    def universal_permission():
        return st.just({SEMANTIC_KEY: "UniversalProximityPermission"})
    
    @staticmethod
    def universal_prohibition(max_threshold: Optional[int] = None):
        return gen_spot_separation_threshold(max_threhsold=max_threshold).map(lambda x: {SEMANTIC_KEY: "UniveralProximityProhibition", MIN_SEP_KEY: x})

    @staticmethod
    @st.composite
    def _gen_selective(draw, *, eligible_timepoints: Iterable[int], semantic: str, max_threshold: Optional[int] = None):
        eligible_timepoints = list(eligible_timepoints)
        min_sep = draw(gen_spot_separation_threshold(max_threhsold=max_threshold))
        groups = gen_partition(eligible_timepoints)
        return {SEMANTIC_KEY: semantic, MIN_SEP_KEY: min_sep, GROUPS_KEY: groups}


    @classmethod
    def selective_permission(cls, *, eligible_timepoints: Iterable[int], max_threshold: Optional[int] = None):
        return cls._gen_selective(eligible_timepoints=eligible_timepoints, semantic="SelectiveProximityPermission", max_threshold=max_threshold)

    @classmethod
    def selective_prohibition(cls, *, eligible_timepoints: Iterable[int], max_threshold: Optional[int] = None):
        return cls._gen_selective(eligible_timepoints=eligible_timepoints, semantic="SelectiveProximityProhibition", max_threshold=max_threshold)


class gen_locus_grouping_data:
    @classmethod    
    def with_strategies_and_empty_flag(cls, *, gen_raw_reg_time: SearchStrategy, gen_raw_loc_time: SearchStrategy, max_size: int, allow_empty: bool):
        cls._check_max_size(max_size)
        return cls.with_strategies_and_min_group_size(
            gen_raw_reg_time=gen_raw_reg_time, 
            gen_raw_loc_time=gen_raw_loc_time, 
            max_size=max_size, 
            min_locus_times_per_reg_time=0 if allow_empty else 1,
        )

    @classmethod
    @st.composite
    def with_max_size_only(cls, draw, *, max_size: int, allow_empty: bool = False):
        cls._check_max_size(max_size)
        reg_times = draw(st.sets(st.integers(min_value=0), min_size=0 if allow_empty else 1, max_size=max_size))
        gen_loc_time = st.integers(min_value=0).filter(lambda n: n not in reg_times)
        locus_times = draw(st.sets(gen_loc_time, min_size=len(reg_times), max_size=len(reg_times)).map(list))
        return dict(zip(reg_times, locus_times, strict=True))

    @classmethod
    def with_strategies_and_min_group_size(cls, *, gen_raw_reg_time: SearchStrategy, gen_raw_loc_time: SearchStrategy, max_size: int, min_locus_times_per_reg_time: int):
        cls._check_max_size(max_size)
        return st.dictionaries(
            keys=gen_raw_reg_time.map(TimepointFrom0), 
            values=st.sets(gen_raw_loc_time.map(TimepointFrom0), min_size=min_locus_times_per_reg_time).map(list),
            max_size=max_size,
        )
    
    @staticmethod
    def _check_max_size(max_size) -> None:
        if max_size < 0:
            raise ValueError(f"Upper bound of locus grouping size can't be negative! {max_size}")


@st.composite
def gen_partition(draw, elements: list[_T], *, min_part_count: int = 1, max_part_count: Optional[int] = None):
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
    
    num_parts = draw(st.integers(min_value=min_part_count, max_value=max_part_count))
    result = draw(st.sampled_from(more_itertools.set_partitions(elements, k=num_parts)))
    return result


def gen_proximity_filter_strategy(*, eligible_timepoints: Iterable[int], max_threshold: Optional[int] = None):
    return st.one_of(
        ProximityFilterStrategyData.universal_permission(), 
        ProximityFilterStrategyData.universal_prohibition(max_threshold=max_threshold),
        ProximityFilterStrategyData.selective_permission(eligible_timepoints=eligible_timepoints, max_threshold=max_threshold), 
        ProximityFilterStrategyData.selective_prohibition(eligible_timepoints=eligible_timepoints, max_threshold=max_threshold),
    )


def gen_spot_separation_threshold(max_threhsold: Optional[int] = None):
    if max_threhsold is not None:
        if not isinstance(max_threhsold, int):
            raise TypeError(f"Max spot separation threshold isn't an int, but {type(max_threhsold).__name__}")
        if max_threhsold < 1:
            raise ValueError(f"Max spot separation threshold is less than 1: {max_threhsold}")
    return st.integers(min_value=1, max_value=max_threhsold)
