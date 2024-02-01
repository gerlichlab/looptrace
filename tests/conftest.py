"""Test fixtures and utilities"""

import os
from pathlib import Path
import pytest

PIPE_NAME = "looptrace_pipeline"


#################################################################
# Fixtures
#################################################################
@pytest.fixture
def images_all_path(tmp_path):
    return tmp_path / "images_all"


@pytest.fixture
def prepped_minimal_config_data(tmp_path, seq_images_path):
    analysis_path = tmp_path / "analysis"
    analysis_path.mkdir()
    seq_images_path.mkdir(parents=True)
    return {
        "analysis_path": str(analysis_path),
        "analysis_prefix": "TESTING__",
        "decon_input_name": "seq_images_raw_decon", 
    }

@pytest.fixture
def seq_images_path(images_all_path):
    return images_all_path / "seq_images_raw"


#################################################################
# Other helpers
#################################################################
def prep_images_folder(folder: Path, create: bool) -> Path:
    return prep_subfolder(folder=folder, name="images", create=create)


def prep_subfolder(folder: Path, name: str, create: bool) -> Path:
    fp = folder / name
    if create:
        fp.mkdir()
    assert (create and fp.is_dir()) or (not create and not fp.is_dir())
    return fp
