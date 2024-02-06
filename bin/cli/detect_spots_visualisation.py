"""Visualisation of detected FISH spots"""

import argparse
import logging
from typing import *

from gertils import ExtantFile, ExtantFolder
import napari
import numpy as np

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker, compute_downsampled_image

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

logger = logging.getLogger(__name__)


# TODO: ideas...
# 1. Specify subset of positions
# 2. Combine timepoints
# 3. Use different filtered spot ROIs files (unfiltered, proximity-filtered, proximity- + nuclei-filtered)
# 4. Include background-subtracted image if available
# 5. Include transformed image file if available (i.e., for DoG detection method).


def workflow(
    config_file: ExtantFile, 
    images_folder: ExtantFolder, 
    image_save_path: Optional[ExtantFolder] = None,
    ):
    H = ImageHandler(config_path=config_file, image_path=images_folder, image_save_path=image_save_path)
    S = SpotPicker(H)
    for (i, frame), ch in S.iter_frames_and_channels():
        logger.info(f"Visualising spot detection in position {args.position}, frame {frame} with threshold {S.spot_threshold[i]}...")
        img = compute_downsampled_image(S.images[args.position], frame=frame, channel=ch, downsampling=S.downsampling)
        # TODO: resume here.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path, method=args.method, intensity_threshold=args.threshold)
