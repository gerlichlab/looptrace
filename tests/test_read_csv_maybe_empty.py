"""Tests for the functionality of the maybe-empty CSV reader utility function"""

import pathlib
import string
from typing import Iterable, TypeAlias

import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.extra import pandas as hyp_pd
from hypothesis.extra.pandas import column as PandasColumn, data_frames
from hypothesis.strategies import SearchStrategy
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from looptrace.utilities import read_csv_maybe_empty


Colnames: TypeAlias = list[str]
ColumnsSpecLike: TypeAlias = int | Colnames


def gen_colnames(gen_one: SearchStrategy[str] = st.text()) -> SearchStrategy[Colnames]:
    """Generate a sequence of column names."""
    return st.sets(gen_one, min_size=1).map(list)


def gen_col_spec_like(gen_one: SearchStrategy[str] = st.text()) -> SearchStrategy[ColumnsSpecLike]:
    """Generate either a number of columns, or a sequence of column names."""
    return st.one_of(gen_colnames(gen_one), st.integers(min_value=1, max_value=10))


def gen_colspec_and_dtype(gen_one_name: SearchStrategy[str] = st.text()) -> SearchStrategy[tuple[ColumnsSpecLike, type]]:
    """Generate either a column count or sequence of column names, along with a data type."""
    return st.tuples(
        gen_col_spec_like(gen_one_name), 
        st.just(int), # Peg to int for now, as using general scalar_dtypes() was causing generation problems.
    )


def gen_columns(gen_one_name: SearchStrategy[str] = st.text()) -> SearchStrategy[Iterable[PandasColumn]]:
    """Generate a collection of columns with which to populate a pandas DataFrame."""
    return gen_colspec_and_dtype(gen_one_name).map(
        lambda spec_and_dtype: hyp_pd.columns(
            names_or_number=spec_and_dtype[0],
            dtype=spec_and_dtype[1],
        )
    )


def gen_random_frame(gen_one_name: SearchStrategy[str] = st.text()) -> SearchStrategy[pd.DataFrame]:
    """Generate a random pandas DataFrame."""
    return gen_columns(gen_one_name).flatmap(lambda cols: data_frames(columns=cols))


@pytest.fixture
def tmp_file(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "table.csv"


@hyp.given(frame=gen_random_frame())
@hyp.settings(suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ))
def test_equivalence_to_read_csv_when_input_is_not_empty(tmp_file, frame: pd.DataFrame):
    frame.to_csv(tmp_file)
    from_custom = read_csv_maybe_empty(tmp_file)
    from_pandas = pd.read_csv(tmp_file)
    assert_frame_equal(from_custom, from_pandas)


def test_empty_frame_with_no_columns_results_when_file_is_totally_empty(tmp_file):
    tmp_file.touch()
    from_custom = read_csv_maybe_empty(tmp_file)
    assert from_custom.empty
    assert list(from_custom.columns) == []


# Prevent edge cases due to single empty column name, or cases where column names are numbers and parsed as data.
@hyp.given(colnames=gen_colnames(st.text(min_size=1, alphabet=string.ascii_letters)))
@hyp.settings(suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ))
def test_empty_frame_with_correct_columns_results_when_input_is_file_with_just_header(tmp_file, colnames):
    with tmp_file.open(mode='w') as fh:
        fh.write(",".join(colnames))
    from_custom = read_csv_maybe_empty(tmp_file)
    assert from_custom.empty
    assert list(from_custom.columns) == colnames
