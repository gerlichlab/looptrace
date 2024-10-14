"""Tests for detection and segmentation of nuclei"""

import copy
import itertools
from operator import itemgetter
from pathlib import Path
import shutil
from typing import *

import dask.array as da
import hypothesis as hyp
import numpy as np
from numpy.random import uniform as runif
import pytest
import yaml

from gertils import ExtantFile

from looptrace import ArrayDimensionalityError, ConfigurationValueError, MissingImagesError
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector, SegmentationMethod
from looptrace.image_io import write_jvm_compatible_zarr_store
from looptrace.integer_naming import get_fov_name_short

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


def get_nuc_detector(
    rounds_config: ExtantFile,
    conf_data: dict, 
    conf_file: Path, 
    *, 
    images_folder: Optional[Path] = None, 
    ) -> NucDetector:
    """Write config data to file, then parse to nuclei detector, through image handler."""
    with open(conf_file, "w") as fh:
        yaml.dump(conf_data, fh)
    params_config = ExtantFile(conf_file)
    return NucDetector(ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder))


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
def test_accessing_certain_attrs__is_defined_iff_certain_3d_status(dummy_rounds_config, complete_config_data, config_update, attr_name, expect, tmp_path):
    conf_data = copy.deepcopy(complete_config_data)
    conf_data.update(config_update)
    N = get_nuc_detector(rounds_config=dummy_rounds_config, conf_data=conf_data, conf_file=tmp_path / "conf.yaml")
    assert N.do_in_3d == config_update[NucDetector.KEY_3D]
    assert N.do_in_3d == conf_data[NucDetector.KEY_3D]
    run = lambda: getattr(N, attr_name)
    if isinstance(expect, BaseException):
        with pytest.raises(type(expect)) as err_ctx:
            run()
        assert str(err_ctx.value) == str(expect)
    else:
        assert run() == expect


def test_fov_names_images_list_relationship(dummy_rounds_config, complete_config_data, tmp_path):
    input_key = complete_config_data["nuc_input_name"]
    N = get_nuc_detector(rounds_config=dummy_rounds_config, conf_data=complete_config_data, conf_file=tmp_path / "conf.yaml")
    # Initially, the input images don't exist, and so there are no FOV names.
    with pytest.raises(MissingImagesError):
        N.input_images
    with pytest.raises(AttributeError) as err_ctx:
        N.fov_list
    assert str(err_ctx.value) == "Position names list for nuclei isn't defined when there are no images!"
    # We can populate with an empty list of images.
    N.image_handler.images = {}
    N.image_handler.images[input_key] = []
    N.image_handler.image_lists = {}
    N.image_handler.image_lists[input_key] = []
    assert N.input_images == []
    assert N.fov_list == []
    # If FOV names list length differs from image list length, that's bad!
    pos_names = ["P0001.zarr", "P0002.zarr"]
    N.image_handler.image_lists[input_key] = pos_names
    assert N.fov_list == pos_names
    with pytest.raises(ArrayDimensionalityError) as err_ctx:
        N.input_images
    assert str(err_ctx.value) == f"0 images and {len(pos_names)} FOV(s); these should be equal!"


@pytest.mark.parametrize(
    ["img_by_name", "expect"], [
        # First, the good case, with 4D image arrays (channel, z, y, x)
        (
            {"P0004.zarr": np.zeros(shape=(2, 2, 2, 2)), "P0003.zarr": np.ones(shape=(3, 3, 3, 3))}, 
            [np.ones(shape=(3, 3, 3, 3)), np.zeros(shape=(2, 2, 2, 2))]
        ), 
        # Then, the bad case where there's no channel dimension.
        (
            {"P0002.zarr": np.zeros(shape=(2, 2, 2)), "P0001.zarr": np.ones(shape=(2, 2, 2))}, 
            ArrayDimensionalityError("2 images with bad shape (length not equal to 4)")
        ), 
        # Finally, the bad case where there's time dimension in addition to the 4 required ones.
        (
            {"P0002.zarr": np.zeros(shape=(2, 2, 2, 2, 2)), "P0001.zarr": np.ones(shape=(2, 2, 2, 2, 2))}, 
            ArrayDimensionalityError("2 images with bad shape (length not equal to 4)")
        ), 
    ]
    )
def test_bad_image_dimensionality__generates_expected_error__enforcing_single_timepoint_issue_250(
    dummy_rounds_config, complete_config_data, img_by_name, expect, tmp_path,
    ):
    input_key = complete_config_data["nuc_input_name"]
    N = get_nuc_detector(rounds_config=dummy_rounds_config, conf_data=complete_config_data, conf_file=tmp_path / "conf.yaml")
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


@pytest.mark.parametrize("config_update", [
    {NucDetector.DETECTION_METHOD_KEY: method, NucDetector.KEY_3D: do_3d, "nuc_slice": z_slice} 
    for method, do_3d, z_slice 
    in itertools.product(["cellpose", "threshold"], [False, True], [-1, 0, 1])
])
@pytest.mark.parametrize("num_pos", [1, 2, 3])
def test_nuc_detector__generates_image_of_proper_dimension(
    dummy_rounds_config, 
    complete_config_data, 
    config_update, 
    num_pos, 
    tmp_path,
    ):
    """This tests the generation of images for nuclei segmentation. Cellpose expects single-channel."""
    # Prepare the config data and the nucleus detector instance.
    conf_data = copy.deepcopy(complete_config_data)
    conf_data.update(config_update)
    input_key = conf_data["nuc_input_name"]
    N = get_nuc_detector(rounds_config=dummy_rounds_config, conf_data=conf_data, conf_file=tmp_path / "conf.yaml")
    
    # With no images, we should get the expected error.
    assert not N.has_images_for_segmentation
    with pytest.raises(MissingImagesError) as err_ctx:
        N.images_for_segmentation
    assert str(err_ctx.value) == f"No images available at all; was {NucDetector.__name__} created without an images folder?"
    
    # Generate the inputs.
    N.image_handler.images = {}
    N.image_handler.image_lists = {}
    gen_img = lambda: runif(size=(2, 3, 4, 6))
    inputs = [(get_fov_name_short(i), gen_img()) for i in range(num_pos)]
    images_folder = tmp_path / "images"
    nuc_subfolder = images_folder / "nuc_images_zarr"
    nuc_subfolder.mkdir(parents=True, exist_ok=True)
    write_jvm_compatible_zarr_store(inputs, root_path=nuc_subfolder, dtype=np.uint16)
    input_names, input_images = zip(*inputs)
    N.image_handler.images[input_key] = input_images
    N.image_handler.image_lists[input_key] = input_names
    
    # Still with just raw--no preprocessed--images, we should get the expected error.
    assert not N.has_images_for_segmentation
    with pytest.raises(MissingImagesError) as err_ctx:
        N.images_for_segmentation
    assert str(err_ctx.value) == f"No images available ({NucDetector.SEGMENTATION_IMAGES_KEY}) as preprocessed input for nuclei segmentation!"
    
    # Prepare for and do the image generation.
    N.image_handler.image_save_path = images_folder
    N.generate_images_for_segmentation()
    
    # Now the preprocessed images should be present and as expected...only AFTER image reloading.
    N.image_handler.images_folder = images_folder
    assert not N.has_images_for_segmentation
    N.image_handler.read_images()
    assert N.has_images_for_segmentation
    get_single_channel = lambda img: img[conf_data["nuc_channel"]]
    all_imgs_obs = list(N.images_for_segmentation)
    if conf_data[NucDetector.KEY_3D] or N.segmentation_method == SegmentationMethod.THRESHOLD:
        assert N.do_in_3d is True
        get_exp = get_single_channel
    else:
        assert N.do_in_3d is False
        get_single_z = lambda img: da.max(img, axis=0) if conf_data["nuc_slice"] == -1 else img[conf_data["nuc_slice"]]
        get_exp = lambda img: get_single_z(get_single_channel(img))
    
    # Based on the current parameterisation, get the proper pixels and then cast to 
    # the expected data type.
    all_imgs_exp = list(map(lambda img: get_exp(img).astype(np.uint16), input_images))
    
    assert len(all_imgs_obs) == len(all_imgs_exp)
    assert [img.shape for img in all_imgs_obs] == [img.shape for img in all_imgs_exp]
    try:
        assert all(da.equal(obs, exp).all() for obs, exp in zip(all_imgs_obs, all_imgs_exp))
    except AssertionError:
        print("EXP")
        print(all_imgs_exp)
        print("OBS")
        print(all_imgs_obs)
        raise
    shutil.rmtree(images_folder)


@pytest.mark.parametrize("parser_method_name", ["from_string", "unsafe_from_string"])
@pytest.mark.parametrize(
    "arg_exp_pair", [
        ("cellpose", SegmentationMethod.CELLPOSE), 
        ("threshold", SegmentationMethod.THRESHOLD),
        ]
)
def segmentation_method_parses_correct_values(parser_method_name, arg_exp_pair):
    arg, exp = arg_exp_pair
    parse = getattr(SegmentationMethod, parser_method_name)
    obs = parse(arg)
    assert obs == exp


@pytest.mark.parametrize(
    "parser_method_name_exp_pair", [
        ("from_string", None), 
        ("unsafe_from_string", ConfigurationValueError),
        ]
)
@hyp.given(arg=hyp.strategies.text().filter(lambda s: s not in ["cellpose", "threshold"]))
def segmentation_method_does_not_parse_incorrect_values(parser_method_name_exp_pair, arg):
    parser_method_name, exp = parser_method_name_exp_pair
    parse = getattr(SegmentationMethod, parser_method_name)
    if issubclass(exp, Exception):
        with pytest.raises(exp):
            parse(arg)
    else:
        obs = parse(arg)
        assert obs == exp


@pytest.mark.skip("not implemented")
def test_nuclei_labels__are_contiguous_nonnegative_integers_from_zero():
    # This is a cellpose property that could be enforced. See #241
    pass
