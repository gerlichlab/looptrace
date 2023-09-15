"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import *

from gertils import ExtantFile, ExtantFolder

from looptrace.ImageHandler import handler_from_cli
from looptrace.SpotPicker import SpotPicker, CROSSTALK_SUBTRACTION_KEY, DETECTION_METHOD_KEY, DetectionMethod

__author__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


ConfigMapping = Dict[str, Union[str, List[int], DetectionMethod, int, bool]]


@dataclass
class Parameters:
    frames: List[int]
    method: DetectionMethod
    threshold: int
    downsampling: int
    only_in_nuclei: bool
    subtract_crosstalk: bool
    minimum_spot_separation: int

    # TODO: refinement typed for the case class members

    _config_keys = {
        "frames": "spot_frame", 
        "method": DETECTION_METHOD_KEY, 
        "threshold": "spot_threshold", 
        "downsampling": "spot_downsample", 
        "only_in_nuclei": "spot_in_nuc", 
        "subtract_crosstalk": CROSSTALK_SUBTRACTION_KEY, 
        "minimum_spot_separation": "min_spot_dist",
    }

    @classmethod
    def from_dict(cls, src: ConfigMapping) -> "Parameters":
        data = {k: DetectionMethod.parse(src[v]) if k == "method" else src[v] for k, v in cls._config_keys.items()}
        return cls(**data)

    @classmethod
    def from_json_file(cls, fp: ExtantFile) -> "Parameters":
        with open(fp.path, 'r') as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_dict_for_json(self) -> ConfigMapping:
        result = {}
        for member_key, config_key in self._config_keys.items():
            rawval = getattr(self, member_key)
            result[config_key] = rawval.value if member_key == "method" else rawval
        return result


ParamPatch = Union[Parameters, ConfigMapping]


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        image_save_path: Optional[ExtantFolder] = None, 
        params_update: Optional[Union[Parameters, ConfigMapping]] = None, 
        write_config_path: Optional[str] = None, 
        outfile: Optional[Union[str, Path]] = None,
        ) -> Optional[Path]:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    if params_update is not None:
        update_data = params_update if isinstance(params_update, dict) else params_update.to_dict_for_json()
        image_handler.config.update(update_data)
    if write_config_path:
        print(f"Writing config JSON: {write_config_path}")
        with open(write_config_path, 'w') as fh:
            json.dump(image_handler.config, fh, indent=4)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    S = SpotPicker(image_handler=image_handler, array_id=None if array_id is None else int(array_id))
    return S.rois_from_spots(outfile=outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path, method=args.method, intensity_threshold=args.threshold)
