"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from typing import *

from gertils import ExtantFile, ExtantFolder

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vincent Reuter"]


def workflow(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder,
    ) -> Optional[str]:
    image_handler = ImageHandler(
        rounds_config=rounds_config, 
        params_config=params_config, 
        images_folder=images_folder, 
        image_save_path=None,
        )
    picker = SpotPicker(image_handler=image_handler)
    return picker.make_dc_rois_all_frames()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine bounding boxes for each detected spot for each timepoint.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.image_path,
        )
