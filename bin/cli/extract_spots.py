"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse

from gertils import ExtantFile, ExtantFolder

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vincent Reuter"]


def workflow(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    ) -> str:
    image_handler = ImageHandler(
        rounds_config=rounds_config, 
        params_config=params_config, 
        images_folder=images_folder, 
        image_save_path=None,
        )
    picker = SpotPicker(image_handler=image_handler)
    return picker.gen_roi_imgs_inmem()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run spot detection on all timepoints and channels listed in config.')
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        )
