"""Tests for deconvolution step of pipeline"""

import pytest
import yaml
from looptrace.Deconvolver import Deconvolver, DECON_CHANNEL_KEY, DECON_ITER_KEY, NON_DECON_CHANNEL_KEY
from looptrace.ImageHandler import ImageHandler
from looptrace.exceptions import MissingInputException

from conftest import prep_images_folder

__author__ = "Vince Reuter"




@pytest.mark.parametrize("num_its", [30, 60])
@pytest.mark.parametrize(
    "channels", 
    [{DECON_CHANNEL_KEY: 0, NON_DECON_CHANNEL_KEY: 1}, {DECON_CHANNEL_KEY: 1, NON_DECON_CHANNEL_KEY: 0}]
    )
def test_no_deconvolution_input(tmp_path, prepped_minimal_config_data, channels, num_its):
    """When declared input for deconvolution doesn't exist, the expected error should be thrown."""
    conf_path = tmp_path / "config.yaml"
    conf_data = {**prepped_minimal_config_data, **channels, **{DECON_ITER_KEY: num_its}}
    with open(conf_path, 'w') as fh:
        yaml.dump(conf_data, fh)
    imgs_path = prep_images_folder(tmp_path, create=True)
    H = ImageHandler(config_path=conf_path, image_path=imgs_path)
    D = Deconvolver(H)
    with pytest.raises(MissingInputException):
        D.decon_seq_images()
