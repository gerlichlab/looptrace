"""Tests for the keys of locus spot visualisation data (i.e., what's dragged-and-dropped to Napari)"""

import json
from pathlib import Path
from typing import Mapping

from expression import fst, result, snd
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import pytest

from looptrace.trace_metadata import LocusSpotViewingKey
from .utilities import gen_trace_group_maybe


@pytest.fixture
def tmp_file(tmp_path) -> Path:
    return tmp_path / "tmp.json"


def gen_key() -> SearchStrategy[LocusSpotViewingKey]:
    return st.tuples(
        st.text().filter(lambda s: "_" not in s), # Avoid that the value contains the presumed delimiter.
        gen_trace_group_maybe(),
    ).map(lambda args: LocusSpotViewingKey(field_of_view=fst(args), trace_group_maybe=snd(args)))


@hyp.given(generated_key=gen_key())
def test_key_roundtrips_through_string(generated_key: LocusSpotViewingKey):
    match LocusSpotViewingKey.from_string(generated_key.to_string):
        case result.Result(tag="ok", ok=parsed_key):
            assert parsed_key == generated_key
        case result.Result(tag="error", err=err_msg):
            pytest.fail(f"Failed to re-parse key; message: {err_msg}")


@hyp.given(generated_key=gen_key())
@hyp.settings(suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ))
@pytest.mark.parametrize("kwargs", [{}, {"indent": 2}])
def test_key_roundtrips_through_json(tmp_file: Path, generated_key: LocusSpotViewingKey, kwargs: Mapping[str, object]):
    with tmp_file.open(mode="w") as fh:
        json.dump(fp=fh, obj=generated_key.to_mapping, **kwargs)
    with tmp_file.open(mode="r") as fh:
        data = json.load(fh)
    match LocusSpotViewingKey.from_mapping(data):
        case result.Result(tag="ok", ok=parsed_key):
            assert parsed_key == generated_key
        case result.Result(tag="error", error=err_msg):
            pytest.fail(f"Failed to re-parse locus spot viewing key; message: {err_msg}")
