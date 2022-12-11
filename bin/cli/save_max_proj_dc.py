
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.Drifter import Drifter
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract experimental PSF from bead images.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", help="(Optional): Path to folder to save images to.", default=None)
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, image_save_path=args.image_save_path)

    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None
    D = Drifter(H, array_id = array_id)
    D.save_proj_dc_images()