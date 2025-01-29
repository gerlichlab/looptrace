"""I/O operations with ND2 files"""

from collections import OrderedDict, defaultdict
import itertools
from operator import itemgetter
import os
from pathlib import Path
from typing import *

import dask.array as da
from expression import Result, result
import nd2
import tqdm

from looptrace.image_io import VOXEL_SIZE_KEY, parse_fields_of_view_from_text, parse_times_from_text
from looptrace.integer_naming import get_fov_names_N
from looptrace.voxel_stack import VoxelSize

__author__ = "Vince Reuter"
__all__ = [
    "EmptyImagesError", 
    "Nd2FileError", 
    "FieldOfViewTimeFilenameKeyError", 
    "key_image_file_names_by_point_and_time", 
    "parse_nd2_metadata", 
    "stack_nd2_to_dask",
    ]


AXIS_SIZES_KEY = "axis_sizes"

CHANNEL_COUNT_KEY = "channelCount"


class EmptyImagesError(Exception):
    """Exception subclass for when there are no images to read"""


class Nd2FileError(Exception):
    """Exception subtype for when there's a problem reading a ND2 file"""


class FieldOfViewTimeFilenameKeyError(Exception):
    """Exception subtype for when FOV and time are identical for multiple images in same subfolder"""


def key_image_file_names_by_point_and_time(image_files: Iterable[Union[str, Path]]) -> Mapping[str, Mapping[str, str]]:
    keyed = {}
    bad_names = {}
    collisions = defaultdict(list)
    for f in image_files:
        fn = f if isinstance(f, str) else str(f)
        pos_hits = parse_fields_of_view_from_text(fn)
        time_hits = parse_times_from_text(fn)
        if len(time_hits) != 1 or len(pos_hits) != 1:
            bad_names[fn] = (pos_hits, time_hits)
            continue
        k = (pos_hits[0], time_hits[0])
        if k in keyed:
            collisions[k].append(fn)
            continue
        keyed[k] = fn
    if bad_names or collisions:
        raise FieldOfViewTimeFilenameKeyError(
            f"Cannot uniquely key images on (FOV, time). bad_names = {bad_names}. collisions = {collisions}"
            )
    result = defaultdict(dict)
    for (p, t), f in keyed.items():
        result[p][t] = f
    return result


def parse_voxel_size(nd2_file: nd2.ND2File, channel: int = 0) -> VoxelSize:
    voxels = nd2_file.voxel_size(channel=channel)
    return VoxelSize(**{dim: getattr(voxels, dim) for dim in ("z", "y", "x")})


def parse_nd2_metadata(image_file: str) -> Mapping[str, Any]:
    metadata = {}
    with nd2.ND2File(image_file) as sample:
        metadata[VOXEL_SIZE_KEY] = parse_voxel_size(sample)
        metadata[AXIS_SIZES_KEY] = OrderedDict(sample.sizes)
        metadata[CHANNEL_COUNT_KEY] = getattr(sample.attributes, CHANNEL_COUNT_KEY)
        microscope = sample.metadata.channels[0].microscope
        metadata['microscope'] = {
            'objectiveMagnification': microscope.objectiveMagnification,
            'objectiveName': microscope.objectiveName,
            'objectiveNumericalAperture':microscope.objectiveNumericalAperture,
            'zoomMagnification': microscope.zoomMagnification,
            'immersionRefractiveIndex': microscope.immersionRefractiveIndex,
            'modalityFlags': microscope.modalityFlags
            }
        for channel in sample.metadata.channels:
            metadata['channel_' + str(channel.channel.index)] = {
                'name': channel.channel.name,
                'emissionLambdaNm': channel.channel.emissionLambdaNm,
                'excitationLambdaNm': channel.channel.excitationLambdaNm,
                }
    return metadata


def stack_nd2_to_dask(folder: str, fov_index: Optional[int] = None):
    """
    The function takes a folder path and returns a list of dask arrays and a 
    list of image folders by reading multiple nd2 images where each represents a 3D stack (split by FOV and time) in a single folder.
    Extracts some useful metadata from the first file in the folder.
    Args:
        folder (str): Input folder path

    Returns:
        list: list of dask arrays of the images
        list: names of fields of view
        dict: metadata dictionary
    """
    image_files = sorted([p.path for p in os.scandir(folder) if (p.name.endswith('.nd2') and not p.name.startswith('_'))])
    
    keyed_images_folders = key_image_file_names_by_point_and_time(image_files)
    pos_names = get_fov_names_N(len(keyed_images_folders))

    if fov_index is not None:
        # Allow caller to specify single FOV to use, and select it here.
        try:
            keyed_images_folders = dict([list(sorted(keyed_images_folders.items(), key=itemgetter(0)))[fov_index]])
            pos_names = [pos_names[fov_index]]
        except IndexError as e:
            raise IndexError(f"{len(pos_names)} FOV name(s) available, but tried to select index {fov_index}") from e

    try:
        sample_file = next(itertools.chain(*[by_time.values() for by_time in keyed_images_folders.values()]))
    except StopIteration as e:
        raise EmptyImagesError(f"No usable .nd2 files in folder: {folder}") from e
    metadata = parse_nd2_metadata(sample_file)

    pos_stack = []
    errors = {}
    for _, file_path_by_time_name in tqdm.tqdm(sorted(keyed_images_folders.items(), key=itemgetter(0))):
        t_stack = []
        for _, path in sorted(file_path_by_time_name.items(), key=itemgetter(0)):
            try:
                arr = read_nd2(path)
            except OSError as e:
                # Store the error, but try to create a dummy array to allow parse to continue, 
                # that way we can accumulate the totality of the errors.
                errors[path] = str(e)
                try:
                    arr = da.zeros_like(pos_stack[0][0])
                except IndexError:
                    raise Nd2FileError(f"Error reading first ND2 file from {folder}: {e}") from e
            t_stack.append(arr)
        pos_stack.append(da.stack(t_stack))
    
    if errors:
        raise Nd2FileError(f"{len(errors)} error(s) reading ND2 files from {folder}: {errors}")

    match _shift_axes_of_stacked_array_from_nd2(arr=da.stack(pos_stack), metadata=metadata):
        case result.Result(tag="error", error=err_msg):
            raise RuntimeError(f"Failed to finalize stacked array from ND2 parse -- {err_msg}")
        case result.Result(tag="ok", ok=final_array):
            print(f"Loaded nd2 arrays of shape {final_array.shape}")
            return final_array, pos_names, metadata
        case unexpected_structure:
            raise RuntimeError(f"Expected a Result-wrapped value, but got a value of type {type(unexpected_structure).__name__}")


def read_nd2(path: Path) -> da.Array:
    with nd2.ND2File(path, validate_frames=False) as imgdat:
        return imgdat.to_dask()


def _shift_axes_of_stacked_array_from_nd2(
    *,
    arr: da.Array, 
    metadata: Mapping[str, Any],
) -> Result[da.Array, str]:
    match list(metadata[AXIS_SIZES_KEY].keys()):
        case ["Z", "C", "Y", "X"]:
            return Result.Ok(da.moveaxis(arr, -4, -3))
        case ["Z", "Y", "X"]:
            return Result.Ok(arr)
        case axis_names:
            return Result.Error(f"Unxepected axis names from sample ND2: {axis_names}")
