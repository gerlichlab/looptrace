"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import logmuse
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preview spot detection in given position.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument('--position', type=int, help='(Optional): Index of position to view.', default=0)
    parser = logmuse.add_logging_options(parser)

    args = parser.parse_args()
    
    logger = logmuse.logger_via_cli(args)
    
    logger.info(f"Building image handler: {args.image_path}")
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    S = SpotPicker(H)
    preview_pos = H.image_lists[H.config['spot_input_name']][args.position]
    S.rois_from_spots(preview_pos=preview_pos)