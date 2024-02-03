"""Tests for fine-grained drift correction"""

import dataclasses
import os
from pathlib import Path
import shutil
from typing import *

import pandas as pd
import pytest
from gertils import ExtantFile, ExtantFolder

from looptrace import read_table_pandas
from looptrace.Drifter import compute_fine_drifts, Drifter, FullDriftTableRow, Z_PX_COARSE, Y_PX_COARSE, X_PX_COARSE
from looptrace.ImageHandler import ImageHandler
from looptrace.numeric_types import NumberLike

__author__ = "Vince Reuter"


COARSE_SHIFT_COLUMNS = [Z_PX_COARSE, Y_PX_COARSE, X_PX_COARSE]
CONFIG_FILE_NAME = "tracing_processing_config__fine_drift_correction__integration.yaml"
IMAGES_FOLDER_NAME = "images_small"
REFERENCE_FRAME = 0
NUM_DRIFT_ROWS = 4
NUM_POSITIONS = 2


@dataclasses.dataclass
class DataPaths:
    """
    Manage pointers to the essential paths for a looptrace.ImageHandler / looptrace subprocesses.

    Critically, the config file and images subfolder should live in the same folder (i.e., have the
    same folder as a common direct parent).
    
    Parameters
    ----------
    conf : gertils.ExtantFile
        Path to the main looptrace configuration file
    imgs : gertils.ExtantFolder
        Path to the images subfolder

    Raises
    ------
    ValueError
        Raise a generic error with informative message if parent folder of this instance's config file 
        differs from that of its images subfolder.
    """
    conf: ExtantFile
    imgs: ExtantFolder

    def __post_init__(self):
        conf_parent = self.conf.path.parent
        imgs_parent = self.imgs.path.parent
        if conf_parent != imgs_parent:
            raise ValueError(f"Config and images live in different folders! ({conf_parent}, {imgs_parent})")

    @classmethod
    def from_root(cls, root: ExtantFolder, conf_name: str = CONFIG_FILE_NAME, imgs_name: str = IMAGES_FOLDER_NAME) -> "DataPaths":
        """Create an instance from root path and names for config file and images subfolder."""
        cfg = ExtantFile(root.path / conf_name)
        img = ExtantFolder(root.path / imgs_name)
        return cls(conf = cfg, imgs = img)

    @property
    def config_file(self) -> Path:
        return self.conf.path
    
    @property
    def images_folder(self) -> Path:
        return self.imgs.path
    
    @property
    def analysis_folder(self) -> Path:
        return self.root / "analysis"

    @property
    def root(self) -> ExtantFolder:
        """
        Return the path to folder that's common parent of this instance's config file and images subfolder.
        
        Returns
        -------
        gertils.ExtantFolder
            Path to common parent of this instance's config file and images subfolder.
        """
        return self.conf.path.parent


def get_frame(row) -> int:
    return row[0]


def get_reference_frame_rows(rows: Iterable[FullDriftTableRow]) -> List[FullDriftTableRow]:
    return [r for r in rows if get_frame(r) == REFERENCE_FRAME]


def get_non_reference_frame_rows(rows: Iterable[FullDriftTableRow]) -> List[FullDriftTableRow]:
    return [r for r in rows if get_frame(r) != REFERENCE_FRAME]


def get_fine_drift_components(row: FullDriftTableRow) -> Tuple[NumberLike, NumberLike, NumberLike]:
    return list(row)[-3:]


@pytest.fixture
def code_path() -> Path:
    """
    Return path to extant code folder.

    Returns
    -------
    pathlib.Path
        Path to code folder, if it's a directory; otherwise raise an error

    Raises
    ------
    TyoeError
        If the code path doesn't exist, or if neither "CODE" nor "HOME" exists as environment varaible
    """
    try:
        code = Path(os.getenv("CODE"))
    except TypeError:
        code = Path(os.getenv("HOME")) / "code"
    return ExtantFolder(code).path


@pytest.fixture
def coarse_drift_file(data_path) -> Path:
    return data_path / "analysis" / "TEST__seq_images_raw_decon_drift_correction_coarse.csv"


@pytest.fixture
def coarse_drift_table(coarse_drift_file) -> pd.DataFrame:
    return read_table_pandas(coarse_drift_file)


@pytest.fixture
def data_path(code_path) -> Path:
    return code_path / "looptrace-microtest" / "drift_correction_fine"


@pytest.fixture
def data_paths(data_path, tmp_path):
    # TODO: need to copy hidden files (.zattrs and .zgroup) also.
    root = shutil.copytree(data_path, tmp_path / "drift_correction_fine")
    return DataPaths.from_root(ExtantFolder(root))


@pytest.fixture
def drifter(image_handler) -> Drifter:
    return Drifter(image_handler=image_handler)


@pytest.fixture
def image_handler(data_paths) -> ImageHandler:
    H = ImageHandler(
        config_path=data_paths.config_file, 
        image_path=data_paths.images_folder,
        strict_load_tables=False,
        )
    analysis_folder = data_paths.analysis_folder
    H = set_analysis_path(H, analysis_folder)
    H.load_tables() # to repopulate, after updating analysis path, the collection of tables
    return H


def set_analysis_path(handler: ImageHandler, p: Union[str, Path, ExtantFolder]) -> ImageHandler:
    if isinstance(p, ExtantFolder):
        p = p.path
    handler.config["analysis_path"] = str(p)
    return handler


@pytest.mark.parametrize("check_table", [
    pytest.param(test_func, id=test_name) for test_func, test_name in [
        (lambda df: NUM_POSITIONS == len(df.position.unique()), f"{NUM_POSITIONS}_positions"), 
        (lambda df: 2 == len(df.frame.unique()), "2_frames"), 
        (lambda df: NUM_DRIFT_ROWS == df.shape[0], f"{NUM_DRIFT_ROWS}_rows__2_frames_2_positions"), 
        (lambda df: all(col in df.columns for col in COARSE_SHIFT_COLUMNS), "all_shift_columns_present"), 
        (lambda df: (df[df.frame == REFERENCE_FRAME][COARSE_SHIFT_COLUMNS] == 0).all().all(), "all_coarse_shifts_zero_for_reference_frame"), 
        (lambda df: 1 < len(df[(df[COARSE_SHIFT_COLUMNS] == 0).apply(lambda r: r.all(), axis=1)].frame.unique()), "exists_nonreference_frame_with_all_zero_shift")
    ]
])
@pytest.mark.slow
def test_data_precheck(coarse_drift_table, check_table):
    assert check_table(coarse_drift_table)


@pytest.mark.parametrize("check_drifts", [
    pytest.param(test_func, id=test_name) for test_func, test_name in [
        (lambda rows: NUM_DRIFT_ROWS == len(rows), f"{NUM_DRIFT_ROWS}_fine_drift_records"), 
        (lambda rows: NUM_POSITIONS == len(get_reference_frame_rows(rows)), f"{NUM_POSITIONS}_positions_for_reference_frame"),
        (lambda rows: NUM_POSITIONS == len(get_non_reference_frame_rows(rows)), f"{NUM_POSITIONS}_positions_for_non_reference_frame"),
        (lambda rows: all(all(0 == d for d in get_fine_drift_components(r)) for r in get_reference_frame_rows(rows)), "all_ref_frame_records_dont_shift"),
        (lambda rows: all(any(0 != d for d in get_fine_drift_components(r)) for r in get_non_reference_frame_rows(rows)), "each_non_ref_frame_record_shifts"),
    ]
])
@pytest.mark.slow
def test_fine_drift_correction_fit_method__properties__issue_103(drifter, check_drifts):
    """Drift correction uses one timepoint as reference, so by construction that timepoint should be unshifted."""
    fine_drifts = list(compute_fine_drifts(drifter))
    print("FINE DRIFTS")
    for drift in fine_drifts:
        print(drift)
    assert check_drifts(fine_drifts)
