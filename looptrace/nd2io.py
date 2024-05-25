"""I/O operations with ND2 files"""

from collections import defaultdict
import itertools
from operator import itemgetter
import os
from pathlib import Path
from typing import *

import dask.array as da
import nd2
import tqdm

from looptrace.image_io import parse_positions_from_text, parse_times_from_text
from looptrace.integer_naming import get_position_names_N

__author__ = "Vince Reuter"
__all__ = [
    "EmptyImagesError", 
    "Nd2FileError", 
    "PositionTimeFilenameKeyError", 
    "key_image_file_names_by_point_and_time", 
    "parse_nd2_metadata", 
    "stack_nd2_to_dask",
    ]


class EmptyImagesError(Exception):
    """Exception subclass for when there are no images to read"""


class Nd2FileError(Exception):
    """Exception subtype for when there's a problem reading a ND2 file"""


class PositionTimeFilenameKeyError(Exception):
    """Exception subtype for when position and time are identical for multiple images in same subfolder"""


def key_image_file_names_by_point_and_time(image_files: Iterable[Union[str, Path]]) -> Mapping[str, Mapping[str, str]]:
    keyed = {}
    bad_names = {}
    collisions = defaultdict(list)
    for f in image_files:
        fn = f if isinstance(f, str) else str(f)
        pos_hits = parse_positions_from_text(fn)
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
        raise PositionTimeFilenameKeyError(
            f"Cannot uniquely key images on (position, time). bad_names = {bad_names}. collisions = {collisions}"
            )
    result = defaultdict(dict)
    for (p, t), f in keyed.items():
        result[p][t] = f
    return result


def parse_nd2_metadata(image_file: str) -> Mapping[str, Any]:
    metadata = {}
    with nd2.ND2File(image_file) as sample:
        voxels = sample.voxel_size()
        metadata['voxel_size'] = [voxels.z, voxels.y, voxels.x]
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


def stack_nd2_to_dask(folder: str, position_id: int = None):
    '''The function takes a folder path and returns a list of dask arrays and a 
    list of image folders by reading multiple nd2 images where each represents a 3D stack (split by position and time) in a single folder.
    Extracts some useful metadata from the first file in the folder.
    Args:
        folder (str): Input folder path

    Returns:
        list: list of dask arrays of the images
        list: names of positions
        dict: metadata dictionary
    '''
    image_files = sorted([p.path for p in os.scandir(folder) if (p.name.endswith('.nd2') and not p.name.startswith('_'))])
    
    keyed_images_folders = key_image_file_names_by_point_and_time(image_files)
    pos_names = get_position_names_N(len(keyed_images_folders))

    if position_id is not None:
        # Allow caller to specify single FOV to use, and select it here.
        try:
            keyed_images_folders = dict([list(sorted(keyed_images_folders.items(), key=itemgetter(0)))[position_id]])
            pos_names = [pos_names[position_id]]
        except IndexError as e:
            raise IndexError(f"{len(pos_names)} position name(s) available, but tried to select index {position_id}") from e

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

    out = da.stack(pos_stack)
    out = da.moveaxis(out, 2, 3)
    print(f"Loaded nd2 arrays of shape {out.shape}")
    
    return out, pos_names, metadata


def read_nd2(path: Path) -> da.Array:
    with nd2.ND2File(path, validate_frames=False) as imgdat:
        return imgdat.to_dask()
