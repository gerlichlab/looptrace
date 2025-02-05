"""Utilities for tests"""

from string import whitespace
from expression import Option, compose
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from gertils.types import FieldOfViewFrom1
from looptrace.integer_naming import IndexToNaturalNumberText
from looptrace.trace_metadata import TraceGroupName


def _gen_valid_raw_trace_group_name() -> SearchStrategy[str]:
    return st.text(min_size=1).filter(
        lambda s: s[0] not in whitespace and s[-1] not in whitespace and "_" not in s
    )


# Here we subtract 2, allowing for the fact that this upper bound is inclusive rather than exclusive, 
# and for the fact that the limit itself represents one more than the representable maximum.
def gen_FieldOfViewFrom1(max_value: int = IndexToNaturalNumberText.TenThousand.value - 2) -> SearchStrategy[FieldOfViewFrom1]:
    return st.integers(min_value=1, max_value=max_value).map(FieldOfViewFrom1)


def gen_trace_group_maybe() -> SearchStrategy[Option[TraceGroupName]]:
    return st.booleans().flatmap(
        lambda p: _gen_valid_raw_trace_group_name().map(compose(TraceGroupName, Option.Some)) 
        if p 
        else st.just(Option.Nothing())
    )
