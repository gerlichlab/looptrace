"""Utilities for tests"""

from string import whitespace
from expression import Option, compose
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from looptrace.trace_metadata import TraceGroupName


def _gen_valid_raw_trace_group_name() -> SearchStrategy[str]:
    return st.text(min_size=1).filter(
        lambda s: s[0] not in whitespace and s[-1] not in whitespace and "_" not in s
    )


def gen_trace_group_maybe() -> SearchStrategy[Option[TraceGroupName]]:
    return st.booleans().flatmap(
        lambda p: _gen_valid_raw_trace_group_name().map(compose(TraceGroupName, Option.Some)) 
        if p 
        else st.just(Option.Nothing())
    )
