"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import os
from typing import *

from looptrace.Deconvolver import Deconvolver
from looptrace.ImageHandler import ImageHandler
from gertils import ExtantFile, ExtantFolder

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vincent Reuter"]


def workflow(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    image_save_path: Optional[ExtantFolder] = None,
    ) -> List[str]:
    image_handler = ImageHandler(
        rounds_config=rounds_config, 
        params_config=params_config, 
        images_folder=images_folder, 
        image_save_path=image_save_path,
        )
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    decon = Deconvolver(image_handler=image_handler, array_id=(None if array_id is None else int(array_id)))
    return decon.decon_seq_images()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deconvolve image data.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.image_path, 
        image_save_path=args.image_save_path,
        )
