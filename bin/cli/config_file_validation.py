"""Validation of the main looptrace configuration file"""

import argparse
from typing import *

import yaml

from gertils import ExtantFile

from looptrace.SpotPicker import DetectionMethod, CROSSTALK_SUBTRACTION_KEY, DETECTION_METHOD_KEY as SPOT_DETECTION_METHOD_KEY
from looptrace.Tracer import MASK_FITS_ERROR_MESSAGE


class ConfigFileError(Exception):
    """Base class for violations of configuration file rules."""
    pass


class ConfigFileCrash(Exception):
    """Class aggregating nonempty collection of config file errors"""
    def __init__(self, errors: Iterable[ConfigFileError]):
        super().__init__(f"{len(errors)} error(s):\n{'; '.join(errors)}")


def find_config_file_errors(config_file: ExtantFile) -> List[ConfigFileError]:
    with open(config_file.path, 'r') as fh:
        conf_data = yaml.safe_load(fh)
    errors = []
    if conf_data.get(CROSSTALK_SUBTRACTION_KEY, False):
        errors.append(ConfigFileError(f"Crosstalk subtraction ('{CROSSTALK_SUBTRACTION_KEY}') isn't currently supported."))
    spot_detection_method = conf_data.get(SPOT_DETECTION_METHOD_KEY)
    if spot_detection_method is None:
        errors.append(ConfigFileError(f"No spot detection method ('{SPOT_DETECTION_METHOD_KEY}') specified!"))
    elif spot_detection_method == DetectionMethod.INTENSITY.value:
        errors.append(ConfigFileError(f"Prohibited (or unsupported) spot detection method: '{spot_detection_method}'"))
    if conf_data.get("mask_fits", False):
        errors.append(ConfigFileError(MASK_FITS_ERROR_MESSAGE))
    return errors


def workflow(config_file: ExtantFile) -> None:
    errors = find_config_file_errors(config_file=config_file)
    if errors:
        raise ConfigFileCrash(errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the provided main looptrace configuration file.")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    args = parser.parse_args()
    workflow(config_file=args.config_path)
