"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import os

from looptrace.ImageHandler import handler_from_cli
from looptrace.SpotPicker import SpotPicker
from looptrace.pathtools import ExtantFile, ExtantFolder


def workflow(config_file: ExtantFile, images_folder: ExtantFolder, already_registered: bool = False) -> str:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=None)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    picker = SpotPicker(image_handler=image_handler, array_id=None if array_id is None else int(array_id))
    return picker.gen_roi_imgs_inmem_coursedc() if already_registered else picker.gen_roi_imgs_inmem()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", type=ExtantFile, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder, help="Path to folder with images to read.")
    parser.add_argument('--course_dc', action='store_true', help='Use this flag if images already coursely registered and saved.')
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, already_registered=args.course_dc)
