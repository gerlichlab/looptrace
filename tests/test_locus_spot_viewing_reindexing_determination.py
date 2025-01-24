"""Tests for the data type which stores info about how to reindex trace IDs and timepoints for locus spot viewing"""

import json
from pathlib import Path
from typing import Mapping

from expression import fst, result, snd
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import pytest

from gertils.types import TimepointFrom0
from looptrace.trace_metadata import LocusSpotViewingReindexingDetermination


@pytest.fixture
def tmp_file(tmp_path) -> Path:
    """Provide a temporary file for each test to use."""
    return tmp_path / "metadata.json"


def gen_timepoints() -> SearchStrategy[TimepointFrom0]:
    """Generate a collection of timepoints to be wrapped in a reindexing determination."""
    return st.lists(st.integers(min_value=0).map(TimepointFrom0), min_size=1, unique=True)


def gen_trace_ids() -> SearchStrategy[list[int]]:
    """Generate a collection of trace IDs to be wrapped in a reindexing determination."""
    return st.lists(st.integers(min_value=0), unique=True)


def gen_determination() -> SearchStrategy[LocusSpotViewingReindexingDetermination]:
    """Randomly generate an instance of the class under test."""
    return st.tuples(gen_timepoints(), gen_trace_ids()).map(
        lambda args: LocusSpotViewingReindexingDetermination(timepoints=fst(args), traces=snd(args))
    )


@hyp.given(initial_determination=gen_determination())
@hyp.settings(suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ))
@pytest.mark.parametrize("kwargs", [{}, {"indent": 2}])
def test_reindexing_determination_roundtrips_through_json(
    tmp_file: Path, 
    initial_determination: LocusSpotViewingReindexingDetermination, 
    kwargs: Mapping[str, object],
):
    with tmp_file.open(mode="w") as fh:
        json.dump(fp=fh, obj=initial_determination.to_json, **kwargs)
    with tmp_file.open(mode="r") as fh:
        data = json.load(fh)
    match LocusSpotViewingReindexingDetermination.from_mapping(data):
        case result.Result(tag="ok", ok=parsed_determination):
            assert parsed_determination == initial_determination
        case result.Result(tag="error", error=err_msg):
            pytest.fail(f"Failed to re-parse determination instance; message: {err_msg}")
