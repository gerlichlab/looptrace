"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import logging
from pathlib import Path
from typing import *

from gertils.pathtools import ExtantFile, ExtantFolder
from looptrace.Deconvolver import Deconvolver
from looptrace.ImageHandler import handler_from_cli
from looptrace.point_spread_function import PointSpreadFunctionStrategy

logger = logging.getLogger()


def workflow(config_file: ExtantFile, images_folder: ExtantFolder, image_save_path: Optional[ExtantFolder]) -> Optional[Path]:
    # TODO: simplify the procurement of the data needed to determine whether this step runs.
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    decon = Deconvolver(image_handler=image_handler)
    if decon.point_spread_function_strategy == PointSpreadFunctionStrategy.EMPIRICAL:
        logger.info("Empirical PSF selected; computing...")
        return decon.extract_exp_psf()
    else:
        logger.info("Empirical PSF is not selected; skipping computation")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract experimental PSF from bead images.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
