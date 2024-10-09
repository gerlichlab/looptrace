"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import *

from gertils import ExtantFile, ExtantFolder

from looptrace.configuration import MINIMUM_SPOT_SEPARATION_KEY
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import DetectionMethod, SpotPicker, CROSSTALK_SUBTRACTION_KEY, DETECTION_METHOD_KEY

__author__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


ConfigMapping = Dict[str, Union[str, List[int], DetectionMethod, int, bool]]


def workflow(
    *, 
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    image_save_path: Optional[ExtantFolder] = None, 
    outfile: Optional[Union[str, Path]] = None,
    # for additional provenance if desired, to see what the ImageHandler's config looked like
    write_config_path: Optional[str] = None, 
    ) -> Optional[Path]:
    image_handler = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder, image_save_path=image_save_path)
    if write_config_path:
        # for additional provenance if desired, to see what the ImageHandler's config looked like
        print(f"Writing config JSON: {write_config_path}")
        with open(write_config_path, 'w') as fh:
            json.dump(image_handler.config, fh, indent=4)
    S = SpotPicker(image_handler)
    return S.rois_from_spots(outfile=outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spot detection on all timepoints and channels listed in config.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        image_save_path=args.image_save_path,
        )
