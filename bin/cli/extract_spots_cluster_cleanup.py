
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from pathlib import Path
from typing import *

from gertils import ExtantFile, ExtantFolder, PathWrapperException

from looptrace.filepaths import SPOT_IMAGES_SUBFOLDER, get_spot_images_path
from looptrace import image_io
from looptrace.SpotPicker import get_spot_images_zipfile


def workflow(images_folder: ExtantFolder, require_outfile: bool = True) -> Optional[ExtantFile]:
    folder_to_zip: ExtantFolder = ExtantFolder.from_string(get_spot_images_path(images_folder.path))
    file_to_zip_to: Path = get_spot_images_zipfile(images_folder)
    print(f"Zipping contents: {folder_to_zip} ---> {file_to_zip_to}")
    image_io.zip_folder(folder=str(folder_to_zip.path), out_file=str(file_to_zip_to), remove_folder=True)
    try:
        return ExtantFile(file_to_zip_to)
    except (PathWrapperException, TypeError) as e:
        if require_outfile:
            raise
        print(f"ERROR finalising spots zipfile ({file_to_zip_to}), so it may not exist! {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("image_path", type=ExtantFolder.from_string, help=f"Path to folder containing '{SPOT_IMAGES_SUBFOLDER}' folder")
    args = parser.parse_args()
    workflow(args.image_path)
    