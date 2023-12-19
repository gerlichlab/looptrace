"""Test fixtures and utilities"""

import importlib
import os
from pathlib import Path
import sys
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
def get_pipeline_path():
    return get_script_path("run_processing_pipeline.py")


def get_script_path(name: str) -> Path:
    return scripts_folder() / name


def import_pipeline_script():
    pipe_path = get_pipeline_path()
    sys.path.append(os.path.dirname(pipe_path))
    spec = importlib.util.spec_from_file_location(PIPE_NAME, pipe_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PIPE_NAME] = module
    spec.loader.exec_module(module)
    return module


def prep_images_folder(folder: Path, create: bool) -> Path:
    return _prep_subfolder(folder=folder, name="images", create=create)


def prep_output_folder(folder: Path, create: bool) -> Path:
    return _prep_subfolder(folder=folder, name="output", create=create)


def scripts_folder():
    tests_folder = os.path.dirname(__file__)
    project_folder = os.path.dirname(tests_folder)
    return Path(project_folder) / "bin" / "cli"


def _prep_subfolder(folder: Path, name: str, create: bool) -> Path:
    fp = folder / name
    if create:
        fp.mkdir()
    assert (create and fp.is_dir()) or (not create and not fp.is_dir())
    return fp
