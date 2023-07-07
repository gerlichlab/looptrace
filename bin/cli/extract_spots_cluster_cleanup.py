
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import os
from looptrace import image_io
from looptrace.SpotPicker import SPOT_IMG_ZIP_NAME


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("image_path", help="Path to folder containing 'spot_images_dir' folder")
    args = parser.parse_args()
    image_io.zip_folder(
        os.path.join(args.image_path, 'spot_images_dir'), 
        out_file=os.path.join(args.image_path, SPOT_IMG_ZIP_NAME), 
        remove_folder=True
        )