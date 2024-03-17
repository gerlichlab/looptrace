"""Tests for spot detection"""

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from looptrace.image_processing_functions import detect_spots as detect_spots_dog, detect_spots_int
from looptrace import read_table_pandas


__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]


@pytest.mark.parametrize(
    ["detect", "func_type_name", "threshold"],
    [
        pytest.param(
            detect, 
            func_type_name,
            threshold, 
            id=f"{func_type_name}__{threshold}"
            ) 
        for detect, func_type_name, threshold, in 
        [(detect_spots_dog, "diff_gauss", t) for t in (15, 10)] + 
        [(detect_spots_int, "intensity", t) for t in (300, 200)]
    ])
@pytest.mark.parametrize("data_name", ["p0_t57_c0", "p13_t57_c0"])
def test_spot_detection__matches_expectation_on_data_examples(detect, func_type_name, threshold, data_name):
    input_file_name = f"img__{data_name}__smaller.npy"
    output_file_name=f"expect__spots_table__{func_type_name}__threshold_{threshold}__{data_name}.csv"
    input_image = read_input_data(input_file_name)
    obs_table, _, _ = detect(input_image, threshold)
    obs_table = obs_table[["zc", "yc", "xc", "intensity_mean"]]
    exp_table = read_expected_output_table(output_file_name)
    try:
        assert_frame_equal(obs_table, exp_table)
    except AssertionError:
        print("OBS:\n")
        print(obs_table)
        print("EXP:\n")
        print(exp_table)
        raise


class InOrOut(Enum):
    """Designate if a file-like entity is an input or an output."""
    INPUT = "input"
    OUTPUT = "output"

    @property
    def plural(self) -> str:
        """Emit, for purpose of subfolder name, the enum value pluralised"""
        return self.value + "s"


def get_data_file_path(fn: str, *, in_or_out: InOrOut) -> Path:
    """Get the path to the data file associated with the given name."""
    return Path(__file__).parent / "data" / in_or_out.plural / fn


def read_expected_output_table(fn: str) -> "pd.DataFrame":
    """Read an expected output table as a DataFrame."""
    fp = get_data_file_path(fn, in_or_out=InOrOut.OUTPUT)
    return read_table_pandas(fp)


def read_input_data(fn: str) -> "npt.NDArray[Union[np.int8, np.int16]]":
    """Read input image data (numpy array) from file of given name."""
    fp = get_data_file_path(fn, in_or_out=InOrOut.INPUT)
    return np.load(fp)
