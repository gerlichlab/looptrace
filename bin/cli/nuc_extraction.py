"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace import image_processing_functions as ip
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nucleus detection on images.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", help="(Optional): Path to folder to save images to.", default=None)
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, image_save_path=args.image_save_path)
    N = NucDetector(H)
    if 'nuc_rois' not in H.tables:
        N.gen_nuc_rois_prereg()
        H.load_tables()
    N.gen_single_nuc_images()
