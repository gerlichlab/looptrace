"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import *

from gertils import ExtantFile, ExtantFolder

from looptrace.configuration import MINIMUM_SPOT_SEPARATION_KEY
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import DetectionMethod, SpotPicker, CROSSTALK_SUBTRACTION_KEY, DETECTION_METHOD_KEY

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
    # NB: this is only here to support the gridsearch (detect_spots_gridsearch). Otherwise, this could be removed

    _MIN_SEP_KEY = "minimum_spot_separation"

    _config_keys = {
        "frames": "spot_frame", 
        "method": DETECTION_METHOD_KEY, 
        "threshold": "spot_threshold", 
        "downsampling": "spot_downsample", 
        "only_in_nuclei": "spot_in_nuc", 
        "subtract_crosstalk": CROSSTALK_SUBTRACTION_KEY, 
        _MIN_SEP_KEY: MINIMUM_SPOT_SEPARATION_KEY,
    }

    @classmethod
    def update_config(cls, *, old_data: ConfigMapping, new_data: Union["Parameters", ConfigMapping]) -> None:
        # NB: this is executed strictly for its side effect, not for its return value.
        if isinstance(new_data, dict):
            pass
        elif isinstance(new_data, cls):
            new_data = new_data.to_dict_for_json()
        else:
            raise TypeError(f"Data with which to update config is of type {type(new_data).__name__}")
        new_data = copy.deepcopy(new_data) # Prepare for side-effecting action.
        try:
            new_min_sep = new_data.pop(Parameters.get_min_sep_key())
        except KeyError:
            # Nothing to do in this case, as there's no value to update
            pass
        else:
            old_data["regionGrouping"][cls._MIN_SEP_KEY] = new_min_sep
        old_data.update(new_data)

    def to_dict_for_json(self) -> ConfigMapping:
        """Convert this instance to a mapping that's amenable for writing to JSON."""
        result = {}
        for member_key, config_key in self._config_keys.items():
            rawval = getattr(self, member_key)
            result[config_key] = rawval.value if member_key == "method" else rawval
        return result


# A "patch" of looptrace's main configuration parameters, with new parameters for spot detection
ParamPatch = Union[Parameters, ConfigMapping] # TODO: remove if ever removing the spot detection gridsearch.


def workflow(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    image_save_path: Optional[ExtantFolder] = None, 
    params_update: Optional[Union[Parameters, ConfigMapping]] = None, 
    write_config_path: Optional[str] = None, 
    outfile: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
    image_handler = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder, image_save_path=image_save_path)
    if params_update is not None:
        print("Updating spot detection parameters")
        Parameters.update_config(old_data=image_handler.config, new_data=params_update)
    if write_config_path:
        print(f"Writing config JSON: {write_config_path}")
        with open(write_config_path, 'w') as fh:
            json.dump(image_handler.config, fh, indent=4)
    S = SpotPicker(image_handler)
    return S.rois_from_spots(outfile=outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spot detection on all frames and channels listed in config.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.image_path, 
        image_save_path=args.image_save_path,
        )
