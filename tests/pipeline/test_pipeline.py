"""Tests for the main processing pipeline"""

from dataclasses import dataclass
import importlib
import itertools
import logging
import os
from pathlib import Path
import sys
from typing import *

import pytest
import yaml
from gertils import PathWrapperException

from tests.conftest import PIPE_NAME, prep_images_folder, prep_subfolder


def scripts_folder():
    project_folder = Path(__file__).parent.parent.parent
    return project_folder / "bin" / "cli"


def import_pipeline_script():
    pipe_path = scripts_folder() / "run_processing_pipeline.py"
    sys.path.append(os.path.dirname(pipe_path))
    spec = importlib.util.spec_from_file_location(PIPE_NAME, pipe_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PIPE_NAME] = module
    spec.loader.exec_module(module)
    return module


looptrace_pipeline = import_pipeline_script()


@dataclass
class CliOptProvision:
    optname: Optional[str]
    get_path: Callable[[Path], Path]
    prepare_path: Optional[Callable[[Path], Optional[Path]]]

    @property
    def is_legitimate(self) -> bool:
        return self.is_specified and self.is_prepared

    @property
    def is_prepared(self) ->  bool:
        return self.prepare_path is not None

    @property
    def is_specified(self) -> bool:
        return self.optname is not None
    
    def __repr__(self) -> str:
        return f"(optname={self.optname}, prepared={self.is_prepared})"

    def __str__(self) -> str:
        return f"(optname={self.optname}, prepared={self.is_prepared})"


@dataclass
class CliSpecMinimal:
    params_config: CliOptProvision
    rounds_config: CliOptProvision
    images_folder: CliOptProvision
    output_folder: CliOptProvision

    @property
    def _members(self) -> List[CliOptProvision]:
        return [
            self.params_config,
            self.rounds_config,
            self.images_folder, 
            self.output_folder,
            ]

    @property
    def is_legitimate(self) -> bool:
        return all(m.is_legitimate for m in self._members)

    def prepare_list(self, folder: Path) -> List[str]:
        result = []
        for o in self._members:
            if o.optname is None:
                continue
            path = o.get_path(folder)
            if o.prepare_path is not None:
                o.prepare_path(path)
            result.extend([o.optname, str(path)])
        return result
    
    def __repr__(self) -> str:
        return repr(tuple(self._members))

    def __str__(self) -> str:
        return str(tuple(self._members))

BOOL_GRID = (False, True)

PARAMS_CONFIG_SPECS = [
    CliOptProvision(opt, lambda p: p / "params.yaml", (lambda p: p.write_text("", encoding='utf-8')) if create else None) 
    for opt, create in itertools.product((None, "--params-config"), BOOL_GRID)
    ]

ROUNDS_CONFIG_SPECS = [
    CliOptProvision(opt, lambda p: p / "rounds.json", (lambda p: p.write_text("\{\}", encoding='utf-8')) if create else None) 
    for opt, create in itertools.product((None, "--rounds-config"), BOOL_GRID)
    ]

IMAGE_FOLDER_SPECS = [
    CliOptProvision(opt, lambda p: p / "images", (lambda p: p.mkdir()) if create else None) 
    for opt, create in itertools.product((None, "-I", "--images-folder"), BOOL_GRID)
    ]

OUTPUT_FOLDER_SPECS = [
    CliOptProvision(opt, lambda p: p / "output", (lambda p: p.mkdir()) if create else None) 
    for opt, create in itertools.product((None, "--pypiper-folder"), BOOL_GRID)
    ]

COMMAND_LINES = [CliSpecMinimal(
    params_config=params, 
    rounds_config=rounds, 
    images_folder=imgs, 
    output_folder=out,
    ) 
    for params, rounds, imgs, out 
    in itertools.product(PARAMS_CONFIG_SPECS, ROUNDS_CONFIG_SPECS, IMAGE_FOLDER_SPECS, OUTPUT_FOLDER_SPECS)
]


@pytest.mark.parametrize(["cli", "expect_success"], [(cli, cli.is_legitimate) for cli in COMMAND_LINES], ids=str)
def test_required_inputs(tmp_path, cli, expect_success):
    args = cli.prepare_list(tmp_path)
    print(f"ARGS: {args}")
    try:
        looptrace_pipeline.parse_cli(args)
    except (PathWrapperException, SystemExit) as error:
        if expect_success:
            pytest.fail(f"Expected success but got error -- {error}")
        else:
            if isinstance(error, SystemExit):
                assert 2 == error.code
    else:
        if expect_success:
            pass
        else:
            pytest.fail("When testing, unexpected success is a failure!")


@pytest.mark.parametrize("params_config_option", ["--params-config"])
@pytest.mark.parametrize("rounds_config_option", ["--rounds-config"])
@pytest.mark.parametrize("images_folder_option", ["-I", "--images-folder"])
@pytest.mark.parametrize("output_folder_option", ["--pypiper-folder"])
def test_logging(tmp_path, dummy_rounds_config, prepped_minimal_config_data, caplog, params_config_option, rounds_config_option, images_folder_option, output_folder_option):
    caplog.set_level(logging.INFO)
    conf_path = tmp_path / "config.yaml"
    with open(conf_path, 'w') as fh:
        yaml.dump(prepped_minimal_config_data, fh)
    imgs_path = prep_images_folder(tmp_path, create=True)
    output_folder = prep_subfolder(folder=tmp_path, name="output", create=True)
    opts = looptrace_pipeline.parse_cli([
        params_config_option, str(conf_path), 
        rounds_config_option, str(dummy_rounds_config.path), 
        images_folder_option, str(imgs_path), 
        output_folder_option, str(output_folder), 
        looptrace_pipeline.NO_TEE_LOGS_OPTNAME,
        ])
    # Add a .stop_pipeline() call here to avoid (non-fatal) errors from logging system.
    # See: https://github.com/databio/pypiper/issues/186
    looptrace_pipeline.init(opts).manager.stop_pipeline()
    assert f"Building {looptrace_pipeline.PIPE_NAME} pipeline, using images from {imgs_path}" in caplog.text


@pytest.mark.skip("not implemented")
def test_deconvolution_gpu_requirement():
    pass
