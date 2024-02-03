"""Tests for detection and segmentation of nuclei"""

import copy
from operator import itemgetter
from pathlib import Path
from typing import *

import numpy as np
import pytest
import yaml

from looptrace import ArrayDimensionalityError, MissingImagesError
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector

__author__ = "Vince Reuter"


@pytest.fixture
def base_conf() -> dict:
    """Basic configuration data"""
    fp = Path(__file__).parent / "data" / "complete_config_without_analysis_path.yaml"
    with open(fp, "r") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def complete_config_data(tmp_path, base_conf) -> dict:
    """Use temp folder to complete the basic config data."""
    conf_data = copy.deepcopy(base_conf)
    analysis_folder = tmp_path / "analysis_output"
    analysis_folder.mkdir()
    conf_data["analysis_path"] = str(analysis_folder)
    return conf_data


def get_image_handler(conf_data: dict, conf_file: Path, images_folder: Optional[Path] = None) -> ImageHandler:
    """Write config data to file, then parse to image handler."""
    with open(conf_file, "w") as fh:
        yaml.dump(conf_data, fh)
    return ImageHandler(conf_file, images_folder)


def get_nuc_detector(conf_data: dict, conf_file: Path, images_folder: Optional[Path] = None) -> NucDetector:
    """Write config data to file, then parse to nuclei detector, through image handler."""
    return NucDetector(get_image_handler(conf_data=conf_data, conf_file=conf_file, images_folder=images_folder))


@pytest.mark.parametrize(
    ["config_update", "attr_name", "expect"], 
    [
        ({NucDetector.KEY_3D: do_3d, "nuc_slice": z_slice, "nuc_downscaling_z": downscale}, attr_name, expect) 
        for do_3d, z_slice, downscale, attr_name, expect in [
            (False, 1, 2, "z_slice_for_segmentation", 1), 
            (False, 1, 2, "ds_z", NotImplementedError("3D nuclei detection is off, so downscaling in z (ds_z) is undefined!")), 
            (True, 0, 3, "z_slice_for_segmentation", NotImplementedError("z-slicing isn't allowed when doing nuclear segmentation in 3D!")), 
            (True, 1, 2, "ds_z", 2)
        ]
    ]
    )
def test_accessing_certain_attrs__is_defined_iff_certain_3d_status(complete_config_data, config_update, attr_name, expect, tmp_path):
    conf_data = copy.deepcopy(complete_config_data)
    conf_data.update(config_update)
    N = get_nuc_detector(conf_data=conf_data, conf_file=tmp_path / "conf.yaml")
    assert N.do_in_3d == config_update[NucDetector.KEY_3D]
    assert N.do_in_3d == conf_data[NucDetector.KEY_3D]
    run = lambda: getattr(N, attr_name)
    if isinstance(expect, BaseException):
        with pytest.raises(type(expect)) as err_ctx:
            run()
        assert str(err_ctx.value) == str(expect)
    else:
        assert run() == expect


def test_position_names_images_list_relationship(complete_config_data, tmp_path):
    input_key = complete_config_data["nuc_input_name"]
    N = get_nuc_detector(complete_config_data, tmp_path / "conf.yaml")
    # Initially, the input images don't exist, and so there are no position names.
    with pytest.raises(MissingImagesError):
        N.input_images
    with pytest.raises(AttributeError) as err_ctx:
        N.pos_list
    assert str(err_ctx.value) == "Position names list for nuclei isn't defined when there are no images!"
    # We can populate with an empty list of images.
    N.image_handler.images = {}
    N.image_handler.images[input_key] = []
    N.image_handler.image_lists = {}
    N.image_handler.image_lists[input_key] = []
    assert N.input_images == []
    assert N.pos_list == []
    # If position names list length differs from image list length, that's bad!
    pos_names = ["P0001.zarr", "P0002.zarr"]
    N.image_handler.image_lists[input_key] = pos_names
    assert N.pos_list == pos_names
    with pytest.raises(ArrayDimensionalityError) as err_ctx:
        N.input_images
    assert str(err_ctx.value) == f"0 images and {len(pos_names)} positions; these should be equal!"


@pytest.mark.parametrize(
    ["img_by_name", "expect"], [
        (
            {"P0004.zarr": np.zeros(shape=(2, 2, 2, 2)), "P0003.zarr": np.ones(shape=(3, 3, 3, 3))}, 
            [np.ones(shape=(3, 3, 3, 3)), np.zeros(shape=(2, 2, 2, 2))]
        ), 
        (
            {"P0002.zarr": np.zeros(shape=(2, 2, 2)), "P0001.zarr": np.ones(shape=(2, 2, 2))}, 
            ArrayDimensionalityError("2 images with bad shape (length not equal to 4)")
        ), 
        (
            {"P0002.zarr": np.zeros(shape=(2, 2, 2, 2, 2)), "P0001.zarr": np.ones(shape=(2, 2, 2, 2, 2))}, 
            ArrayDimensionalityError("2 images with bad shape (length not equal to 4)")
        ), 
    ]
    )
def test_bad_image_dimensionality__generates_expected_error(complete_config_data, img_by_name, expect, tmp_path):
    input_key = complete_config_data["nuc_input_name"]
    N = get_nuc_detector(complete_config_data, tmp_path / "conf.yaml")
    img_names, img_arrays = zip(*sorted(img_by_name.items(), key=itemgetter(0)))
    N.image_handler.images = {}
    N.image_handler.images[input_key] = img_arrays
    N.image_handler.image_lists = {}
    N.image_handler.image_lists[input_key] = img_names
    if isinstance(expect, BaseException):
        with pytest.raises(type(expect)) as err_ctx:
            N.input_images
        assert str(err_ctx.value).startswith(str(expect))
    else:
        assert len(N.input_images) == len(expect)
        assert all((obs == exp).all() for obs, exp in zip(N.input_images, expect))


@pytest.mark.skip("not implemented")
def test_nuc_detector__generates_image_of_proper_dimension():
    pass


@pytest.mark.skip("not implemented")
def test_nuclei_labels__are_contiguous_nonnegative_integers_from_zero():
    pass
