"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument('--course_dc', help='Use this flag if images already coursely registered and saved.', action='store_true')
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None
    S = SpotPicker(H, array_id = array_id)
    #S.make_dc_rois_all_frames()
    if args.course_dc:
        S.gen_roi_imgs_inmem_coursedc()
    else:
        S.gen_roi_imgs_inmem()