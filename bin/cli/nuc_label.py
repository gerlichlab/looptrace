"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from typing import *

from looptrace.ImageHandler import handler_from_cli
from looptrace.NucDetector import NucDetector
from looptrace.pathtools import ExtantFile, ExtantFolder


def workflow(config_file: ExtantFile, images_folder: ExtantFolder, image_save_path: Optional[ExtantFolder] = None) -> str:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    detector = NucDetector(image_handler=image_handler)
    return detector.segment_nuclei()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nucleus detection on images.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
