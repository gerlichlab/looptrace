
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

from looptrace import image_io
from looptrace.configuration import read_parameters_configuration_file
from looptrace.filepaths import SPOT_BACKGROUND_SUBFOLDER, SPOT_IMAGES_SUBFOLDER
from looptrace.SpotPicker import get_spot_images_zipfile

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vincent Reuter"]


def workflow(
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    *, 
    is_background: bool,
    require_outfile: bool = True, 
    keep_folder: Optional[bool] = None,
    ) -> Optional[ExtantFile]:
    if keep_folder is None:
        conf_data = read_parameters_configuration_file(params_config)
        keep_folder = conf_data.get("keep_spot_images_folder", False)
    src_path_leaf = SPOT_BACKGROUND_SUBFOLDER if is_background else SPOT_IMAGES_SUBFOLDER
    folder_to_zip: ExtantFolder = ExtantFolder(images_folder.path / src_path_leaf)
    file_to_zip_to: Path = get_spot_images_zipfile(images_folder, is_background=is_background)
    print(f"Zipping contents: {folder_to_zip} ---> {file_to_zip_to}")
    image_io.zip_folder(folder=str(folder_to_zip.path), out_file=str(file_to_zip_to), remove_folder=not keep_folder)
    try:
        return ExtantFile(file_to_zip_to)
    except PathWrapperException as e:
        if require_outfile:
            raise
        print(f"ERROR finalising spots zipfile ({file_to_zip_to}), so it may not exist! {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help=f"Path to folder containing '{SPOT_IMAGES_SUBFOLDER}' folder")
    parser.add_argument("--keep", action="store_true", help="Keep the original data, even after zipping.")
    parser.add_argument("--is-background", action="store_true", help="ZIP the spot background image volumes")
    args = parser.parse_args()
    workflow(args.params_config, args.images_folder, is_background=args.is_background)
    
