"""Driver for computing all fiducial bead ROIs for a particular imaging experiment"""

import argparse
from pathlib import Path
from typing import *

import pandas as pd
from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import handler_from_cli
from looptrace.bead_roi_generation import generate_all_bead_rois

__author__ = "Vince Reuter"


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        output_folder: ExtantFolder, 
        **joblib_kwargs
        ) -> Iterable[Tuple[Path, pd.DataFrame]]:
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    if H.reg_input_moving != H.reg_input_template:
        raise Exception(f"Key for shifted images differs from key for template! ({H.reg_input_moving}, {H.reg_input_template})")
    D = Drifter(image_handler=H)
    return generate_all_bead_rois(
        image_array=H.images[H.reg_input_moving], 
        output_folder=output_folder, 
        params=D.get_bead_roi_parameters, 
        **joblib_kwargs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for computing all fiducial bead ROIs for a particular imaging experiment")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("-O", "--output-folder", type=ExtantFolder, help="Path to folder in which to place output")
    parser.add_argument("--prefer-for-joblib", default="threads", help="Argument for joblib.Parallel's 'prefer' parameter")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
