"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import os
from pathlib import *
from typing import *
import dask.array as da
import numpy as np
import tqdm

from expression import result
from gertils import ExtantFile, ExtantFolder
from looptrace import image_io, nd2io
from looptrace.ImageHandler import ImageHandler
from looptrace.integer_naming import get_fov_name_short
from looptrace.voxel_stack import VoxelSize


def workflow(n_pos: int, input_folders: Iterable[Path], output_folder: Path) -> None:
    for fov_index in tqdm.tqdm(range(int(n_pos))):
        imgs = []
        for f in input_folders:
            folder_imgs, _, folder_metadata = nd2io.stack_nd2_to_dask(f, fov_index=fov_index)
            imgs.append(folder_imgs[0])
        imgs = da.concatenate(imgs, axis=0)
        print(folder_metadata)
        voxel_size: VoxelSize
        match VoxelSize.from_list(folder_metadata["voxel_size"]):
            case result.Result(tag="ok", ok=vox_sz):
                voxel_size = vox_sz
            case result.Result(tag="error", error=err_msg):
                raise RuntimeError(f"Could not parse voxel size; message: {err_msg}")
        # TODO: why is it justified to use just the last folder_metadata value (associated with a 
        # single f in input_folders) in a function call where the concatenation of values from 
        # all input_folders is being passed to .zarr creation?
        # See: https://github.com/gerlichlab/looptrace/issues/118
        z = image_io.create_zarr_store(
            path = output_folder,
            name = os.path.basename(output_folder), 
            fov_name = get_fov_name_short(fov_index) + ".zarr",
            shape = imgs.shape, 
            dtype = np.uint16, 
            chunks = (1, 1, 1, imgs.shape[-2], imgs.shape[-1]), # 1 chunk per xy-plane (z-slice)
            metadata = folder_metadata,
            voxel_size = voxel_size,
        )
        n_t = imgs.shape[0]
        for t in tqdm.tqdm(range(n_t)):
            z[t] = imgs[t]


def one_to_one(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder):
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    for input_folder_name, output_folder_name in H.zarr_conversions.items():
        infolder = images_folder.path / input_folder_name
        outfolder = images_folder.path / output_folder_name
        assert not outfolder.exists(), f"Output folder for ZARR conversion already exists: {outfolder}"
        num_fov = len(H.image_lists[input_folder_name])
        print(f"Converting ({num_fov} FOV): {infolder} --> {outfolder}")
        workflow(n_pos=num_fov, input_folders=(infolder, ), output_folder=outfolder)
