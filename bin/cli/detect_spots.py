"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import copy
from dataclasses import dataclass
from enum import Enum
import json
import os
from typing import *

from looptrace.ImageHandler import handler_from_cli
from looptrace.SpotPicker import SpotPicker
from looptrace.pathtools import ExtantFile, ExtantFolder

__author__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


class Method(Enum):
    INTENSITY = 'intensity'
    DIFFERENCE_OF_GAUSSIANS = 'dog'

    @classmethod
    def parse(cls, name: str) -> "Method":
        try:
            return next(m for m in cls if m.value.lower() == name.lower())
        except StopIteration:
            raise ValueError(f"Unknown detection method: {name}")


ConfigMapping = Dict[str, Union[str, List[int], Method, int, bool]]


@dataclass
class Parameters:
    frames: List[int]
    method: Method
    threshold: int
    downsampling: int
    only_in_nuclei: bool
    subtract_crosstalk: bool
    minimum_spot_separation: int

    # TODO: refinement typed for the case class members

    @property
    def _config_keys(self):
        return {
            "frames": "spot_frame", 
            "method": "detection_method", 
            "threshold": "spot_threshold", 
            "downsampling": "spot_downsample", 
            "only_in_nuclei": "spot_in_nuc", 
            "subtract_crosstalk": "subtract_crosstalk", 
            "minimum_spot_separation": "min_spot_dist",
        }

    @classmethod
    def from_dict(cls, src: ConfigMapping) -> "Parameters":
        data = {k: Method.parse(src[v]) if k == "method" else src[v] for k, v in cls._config_keys.items()}
        cls(**data)

    @classmethod
    def from_json_file(cls, fp: ExtantFile) -> "Parameters":
        with open(fp, 'r') as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_dict(self) -> ConfigMapping:
        return {v: getattr(self, k) for k, v in self._config_keys.items()}
        
    def update(self, other: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(other)
        result.update(self.to_dict())
        return result


ParamPatch = Union[Parameters, ConfigMapping]


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        image_save_path: Optional[ExtantFolder], 
        params_update: Optional[Union[Parameters, ConfigMapping]] = None, 
        outfile: Optional[str] = None, 
        write_config_path: Optional[str] = None, 
        ) -> str:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    if params_update is not None:
        update_data = params_update if isinstance(params_update, dict) else params_update.to_dict()
        image_handler.config.update(update_data)
    if write_config_path:
        print(f"Writing config JSON: {write_config_path}")
        with open(write_config_path, 'w') as fh:
            json.dump(image_handler.config, fh)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    S = SpotPicker(image_handler=image_handler, array_id=None if array_id is None else int(array_id))
    outfile = outfile or S.roi_path
    S.rois_from_spots(roi_outfile=outfile) # Execute for effect.
    return outfile # This is the output file written in the effectful step.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", type=ExtantFile, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path, method=args.method, intensity_threshold=args.threshold)
