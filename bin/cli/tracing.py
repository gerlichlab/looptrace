
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import os

from looptrace.ImageHandler import handler_from_cli
from looptrace.Tracer import Tracer
from gertils import ExtantFile, ExtantFolder


def workflow(config_file: ExtantFile, images_folder: ExtantFolder):
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=None)
    tracer = Tracer(image_handler)
    return tracer.trace_all_rois()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run chromatin tracing.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path)
