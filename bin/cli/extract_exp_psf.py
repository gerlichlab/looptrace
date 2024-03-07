"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from pathlib import Path
from typing import *

from gertils import ExtantFile, ExtantFolder
from looptrace.Deconvolver import Deconvolver
from looptrace.ImageHandler import ImageHandler
from looptrace.point_spread_function import PointSpreadFunctionStrategy


def workflow(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder, image_save_path: Optional[ExtantFolder] = None) -> Optional[Path]:
    # TODO: simplify the procurement of the data needed to determine whether this step runs.
    image_handler = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder, image_save_path=image_save_path)
    decon = Deconvolver(image_handler=image_handler)
    if decon.point_spread_function_strategy == PointSpreadFunctionStrategy.EMPIRICAL:
        print("Empirical PSF selected; computing...")
        return decon.extract_exp_psf()
    else:
        print("Empirical PSF is not selected; skipping computation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract experimental PSF from bead images.')
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        image_save_path=args.image_save_path,
        )
