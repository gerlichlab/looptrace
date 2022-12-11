"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""


from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Image folder path")
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    S = SpotPicker(H)
    S.make_dc_rois_all_frames()