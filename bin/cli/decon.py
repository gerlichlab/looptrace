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
from looptrace.ImageHandler import handler_from_cli
from looptrace.pathtools import ExtantFile, ExtantFolder


def workflow(config_file: ExtantFile, images_folder: ExtantFolder, image_save_path: Optional[ExtantFolder] = None) -> List[str]:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    decon = Deconvolver(image_handler=image_handler, array_id=(None if array_id is None else int(array_id)))
    return decon.decon_seq_images()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deconvolve image data.')
    parser.add_argument("config_path", type=ExtantFile, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
