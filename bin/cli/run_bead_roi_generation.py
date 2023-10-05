"""Driver for computing all fiducial bead ROIs for a particular imaging experiment"""

import argparse
from pathlib import Path
from typing import *

import pandas as pd
from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import ImageHandler, handler_from_cli
from looptrace.bead_roi_generation import generate_all_bead_rois

__author__ = "Vince Reuter"


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        output_folder: ExtantFolder, 
        **joblib_kwargs
        ) -> Iterable[Tuple[Path, pd.DataFrame]]:
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    D = Drifter(image_handler=H)
    return generate_all_bead_rois(
        image_array=H.images[get_images_key(H)], 
        output_folder=output_folder, 
        params=D.get_bead_roi_parameters, 
        channel=get_beads_channel(D),
        **joblib_kwargs,
        )


def get_images_key(handler: ImageHandler) -> str:
    key_ref = handler.reg_input_template
    key_mov = handler.reg_input_moving
    if key_mov != key_ref:
        raise Exception(f"Key for shifted images differs from key for template images! ({key_mov}, {key_ref})")
    return key_ref


def get_beads_channel(drifter: Drifter) -> int:
    ch_ref = drifter.reference_channel
    ch_mov = drifter.moving_channel
    if ch_mov != ch_ref:
        raise Exception(f"Key for shifted channel differs from key for template channel! ({ch_mov}, {ch_ref})")
    return ch_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for computing all fiducial bead ROIs for a particular imaging experiment")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("-O", "--output-folder", required=True, type=ExtantFolder.from_string, help="Path to folder in which to place output")
    parser.add_argument("--prefer-for-joblib", default="threads", help="Argument for joblib.Parallel's 'prefer' parameter")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, output_folder=args.output_folder)
