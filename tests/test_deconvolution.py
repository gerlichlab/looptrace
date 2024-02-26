"""Tests for deconvolution step of pipeline"""

import json
import hypothesis as hyp
import pytest
import yaml

from gertils import ExtantFile, ExtantFolder

from looptrace.Deconvolver import Deconvolver, DECON_CHANNEL_KEY, DECON_ITER_KEY, NON_DECON_CHANNEL_KEY
from looptrace.ImageHandler import ImageHandler
from looptrace.exceptions import MissingInputException
from conftest import prep_images_folder

__author__ = "Vince Reuter"


@pytest.fixture
def deconvolver(tmp_path, dummy_rounds_config, prepped_minimal_config_data):
    conf_path = tmp_path / "config.yaml"
    conf_data = {**prepped_minimal_config_data}
    with open(conf_path, 'w') as fh:
        yaml.dump(conf_data, fh)
    params_config = ExtantFile(conf_path)
    imgs_path = ExtantFolder(prep_images_folder(folder=conf_path.parent, create=True))
    H = ImageHandler(rounds_config=dummy_rounds_config, params_config=params_config, image_path=imgs_path)
    return Deconvolver(H)


@pytest.mark.parametrize(
    "channels", 
    [{DECON_CHANNEL_KEY: 0, NON_DECON_CHANNEL_KEY: 1}, {DECON_CHANNEL_KEY: 1, NON_DECON_CHANNEL_KEY: 0}]
    )
@hyp.given(num_its=hyp.strategies.integers(min_value=1))
@hyp.settings(
    deadline=None, # Permit variable runtimes between randomisations (since only first builds Deconvolver).
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ) # Permit given + function-scoped fixture.
    )
def test_no_deconvolution_input(deconvolver, channels, num_its):
    """When declared input for deconvolution doesn't exist, the expected error should be thrown."""
    deconvolver.config.update({**channels, **{DECON_ITER_KEY: num_its}})
    with pytest.raises(MissingInputException):
        deconvolver.decon_seq_images()
