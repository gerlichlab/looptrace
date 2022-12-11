
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import image_io
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("image_path", help="Path to folder containing 'spot_images_dir' folder")
    args = parser.parse_args()
    image_io.zip_folder(args.image_path+os.sep+'spot_images_dir', out_file=args.image_path+os.sep+'spot_images.npz', remove_folder=True)