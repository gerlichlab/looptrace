"""Test fixtures and utilities"""

import importlib
import os
from pathlib import Path
import sys

PIPE_NAME = "looptrace_pipeline"


def import_pipeline_script():
    pipe_path = get_pipeline_path()
    sys.path.append(os.path.dirname(pipe_path))
    spec = importlib.util.spec_from_file_location(PIPE_NAME, pipe_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PIPE_NAME] = module
    spec.loader.exec_module(module)
    return module


def get_pipeline_path():
    return get_script_path("run_processing_pipeline.py")


def get_script_path(name: str) -> Path:
    return scripts_folder() / name


def scripts_folder():
    tests_folder = os.path.dirname(__file__)
    project_folder = os.path.dirname(tests_folder)
    return Path(project_folder) / "bin" / "cli"
