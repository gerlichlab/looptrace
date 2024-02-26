"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from typing import *

from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from gertils import ExtantFile, ExtantFolder


def workflow(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder, image_save_path: Optional[ExtantFolder] = None) -> str:
    image_handler = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder, image_save_path=image_save_path)
    detector = NucDetector(image_handler=image_handler)
    return detector.segment_nuclei()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nucleus detection on images.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Path to file with declaration of imaging rounds, needed for ImageHandler")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Path to parameters configuration file, needed for ImageHandler")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.image_path, 
        image_save_path=args.image_save_path,
        )
