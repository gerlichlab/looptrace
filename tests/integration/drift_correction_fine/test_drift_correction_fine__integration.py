"""Tests for fine-grained drift correction"""

import os
from pathlib import Path
from typing import *

import pandas as pd
import pytest
from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Z_PX_COARSE, Y_PX_COARSE, X_PX_COARSE

__author__ = "Vince Reuter"


COARSE_SHIFT_COLUMNS = [Z_PX_COARSE, Y_PX_COARSE, X_PX_COARSE]
DATA_PATH = Path(os.path.dirname(__file__)) / "data"
REFERENCE_FRAME = 12


@pytest.fixture
def coarse_drift_table():
    # TODO: update with #104
    return pd.read_csv(get_data_file("drift_correction_coarse.csv").path, index_col=0)


@pytest.mark.parametrize("check_table", [
    pytest.param(test_func, id=test_name) for test_func, test_name in [
        (lambda df: 2 == len(df.position.unique()), "2_positions"), 
        (lambda df: 36 == len(df.frame.unique()), "36_frames"), 
        (lambda df: 72 == df.shape[0], "72_rows__36_frames_2_positions"), 
        (lambda df: all(col in df.columns for col in COARSE_SHIFT_COLUMNS), "all_shift_columns_present"), 
        (lambda df: (df[df.frame == REFERENCE_FRAME][COARSE_SHIFT_COLUMNS] == 0).all().all(), "all_coarse_shifts_zero_for_reference_frame"), 
        (lambda df: 1 < len(df[(df[COARSE_SHIFT_COLUMNS] == 0).apply(lambda r: r.all(), axis=1)].frame.unique()), "exists_nonreference_frame_with_all_zero_shift")
    ]
])
def test_data_precheck(coarse_drift_table, check_table):
    assert check_table(coarse_drift_table)


@pytest.mark.skip("not implemented")
def test_fine_drift_correction_fit_method__is_all_zeros_for_reference_frame__issue_103():
    """Drift correction uses one timepoint as reference, so by construction that timepoint should be unshifted."""
    pass


@pytest.mark.skip("not implemented")
def test_fine_drift_correction_fit_method__is_all_nonzero_for_non_reference_frame__issue_103():
    """Points other than the reference should be shifted, as there's almost-0 probability of alignment a priori, given continuous nature of the values."""
    pass


# Type variable for an extant path
XP = TypeVar('XP', bound=Union[ExtantFile, ExtantFolder])


def get_data_file(fn: str) -> ExtantFile:
    return _get_extant_data_path(name=fn, wrap_path=ExtantFile)


def get_data_folder(fn: str) -> ExtantFolder:
    return _get_extant_data_path(name=fn, wrap_path=ExtantFolder)

 
def _get_extant_data_path(name: str, wrap_path: XP) -> XP:
    return wrap_path(DATA_PATH / name)
