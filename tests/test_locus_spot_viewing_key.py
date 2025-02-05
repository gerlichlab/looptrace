"""Tests for the keys of locus spot visualisation data (i.e., what's dragged-and-dropped to Napari)"""

from pathlib import Path
from typing import Any, Callable

from expression import fst, result, snd
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import pytest

from looptrace.trace_metadata import LocusSpotViewingKey
from .utilities import gen_FieldOfViewFrom1, gen_trace_group_maybe


@pytest.fixture
def tmp_file(tmp_path) -> Path:
    return tmp_path / "tmp.json"


def gen_key() -> SearchStrategy[LocusSpotViewingKey]:
    return st.tuples(
        gen_FieldOfViewFrom1(),
        gen_trace_group_maybe(),
    ).map(lambda args: LocusSpotViewingKey(
        field_of_view=fst(args), 
        trace_group_maybe=snd(args),
    ))


@hyp.given(generated_key=gen_key())
def test_key_roundtrips_through_string(generated_key: LocusSpotViewingKey):
    match LocusSpotViewingKey.from_string(generated_key.to_string):
        case result.Result(tag="ok", ok=parsed_key):
            assert parsed_key == generated_key
        case result.Result(tag="error", err=err_msg):
            pytest.fail(f"Failed to re-parse key; message: {err_msg}")


@hyp.given(generated_key=gen_key())
def test_unsafe_for_good_input(generated_key: LocusSpotViewingKey):
    parsed_key: LocusSpotViewingKey = LocusSpotViewingKey.unsafe(generated_key.to_string)
    assert parsed_key == generated_key


def _anything_but_string() -> SearchStrategy[Any]:
    return st.from_type(type).flatmap(st.from_type).filter(lambda x: not isinstance(x, str))


@hyp.given(arg=st.one_of(_anything_but_string(), st.text()))
def test_unsafe_for_bad_input(arg: Any):
    check_error: Callable[[TypeError | ValueError], bool] = \
        (lambda e: isinstance(e, ValueError) and str(e).startswith("Failed to parse LocusSpotViewingKey:")) \
        if isinstance(arg, str) else \
        (lambda e: isinstance(e, TypeError) and f"Input to parse as LocusSpotViewingKey isn't str, but {type(arg).__name__}" in str(e))
    with pytest.raises((TypeError, ValueError)) as err_ctx:
        LocusSpotViewingKey.unsafe(arg)
    assert check_error(err_ctx.value)
