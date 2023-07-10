"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from typing import *

from looptrace.ImageHandler import handler_from_cli
from looptrace.SpotPicker import SpotPicker
from looptrace.pathtools import ExtantFile, ExtantFolder


def workflow(config_file: ExtantFile, images_folder: ExtantFolder) -> str:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=None)
    picker = SpotPicker(image_handler=image_handler)
    return picker.make_dc_rois_all_frames()    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path)
