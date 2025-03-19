"""Validation of the main looptrace configuration file"""

import argparse
import json
from pathlib import Path
from typing import *
import warnings

from expression import result
from gertils import ExtantFile

from looptrace import ConfigurationValueError, Drifter, LOOPTRACE_JAR_PATH, ZARR_CONVERSIONS_KEY
from looptrace.configuration import IMAGING_ROUNDS_KEY, KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS, get_minimum_regional_spot_separation, read_parameters_configuration_file
from looptrace.Deconvolver import REQ_GPU_KEY
from looptrace.ImageHandler import determine_bead_timepoint_for_spot_filtration
from looptrace.NucDetector import NucDetector, SegmentationMethod as NucSegMethod
from looptrace.SpotPicker import DetectionMethod, CROSSTALK_SUBTRACTION_KEY, DETECTION_METHOD_KEY as SPOT_DETECTION_METHOD_KEY
from looptrace.Tracer import MASK_FITS_ERROR_MESSAGE

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

TRACING_SUPPORT_EXCLUSIONS_KEY = "tracingExclusions"


class ConfigFileCrash(Exception):
    """Class aggregating nonempty collection of config file errors"""
    def __init__(self, errors: Iterable[ConfigurationValueError]):
        super().__init__(f"{len(errors)} error(s):\n{'; '.join(map(str, errors))}")


class MissingJarError(Exception):
    """For when the project's JAR is not an extant file"""
    def __init__(self, path: Path):
        super().__init__(str(path))
        if path.is_file():
            raise ValueError(f"Alleged missing JAR is a file: {path}")


def find_config_file_errors(rounds_config: ExtantFile, params_config: ExtantFile) -> List[ConfigurationValueError]:
    """
    Parse the given looptrace main processing configuration file, and build a collection of any errors.

    Parameters
    ----------
    params_config : ExtantFile
        Path to the main looptrace processing parameters configuration file to parse

    Returns
    -------
    list of ConfigurationValueError
        A collection of violations of prohibitions, or other faults, found in the parsed config data
    """
    
    parameters = read_parameters_configuration_file(params_config)

    print(f"Reading imaging config file: {rounds_config.path}")
    with open(rounds_config.path, "r") as fh:
        rounds_config = json.load(fh)
    
    errors = []
    
    # Zarr conversions
    if not parameters.get(ZARR_CONVERSIONS_KEY):
        errors.append(ConfigurationValueError(
            f"Conversion of image folders should be specified as mapping from src to dst folder, with key {ZARR_CONVERSIONS_KEY}."
            ))
    
    # Deconvolution
    errors.extend([
        ConfigurationValueError(f"Missing or null value for key: {k}") 
        for k in ("decon_input_name", "decon_output_name") if not parameters.get(k)
        ])
    if not parameters.get(REQ_GPU_KEY, False):
        errors.append(ConfigurationValueError(f"Requiring GPUs for deconvolution with key {REQ_GPU_KEY} is currently required."))
    
    # Nuclei detection
    if parameters.get(NucDetector.KEY_3D, False):
        errors.append(ConfigurationValueError(f"Nuclei detection in 3D isn't supported! Set key '{NucDetector.KEY_3D}' to False."))
    try:
        nuclei_detection_method = parameters[NucDetector.DETECTION_METHOD_KEY]
    except KeyError:
        errors.append(ConfigurationValueError(f"Missing nuclei detection method key: {NucDetector.DETECTION_METHOD_KEY}!"))
    else:
        if nuclei_detection_method != NucSegMethod.CELLPOSE.value:
            errors.append(f"Unsupported nuclei detection method (key '{NucDetector.DETECTION_METHOD_KEY}')! {nuclei_detection_method}")
    try:
        nuc_ref_timepoint = parameters["nuc_ref_timepoint"]
    except KeyError:
        pass
    else:
        msg_base = f"Nuclei timepoint ('nuc_ref_timepoint') is deprecated, as it's assumed that nuclei are imaged in exactly 1 timepoint."
        if nuc_ref_timepoint == 0:
            warnings.warn(msg_base, DeprecationWarning)
        else:
            errors.append(ConfigurationValueError(f"{msg_base}, but if present must be 0, not {nuc_ref_timepoint}"))
    try:
        segmentation_method = parameters[NucDetector.DETECTION_METHOD_KEY]
    except KeyError:
        errors.append(ConfigurationValueError(f"Missing nuclei detection method ('{NucDetector.DETECTION_METHOD_KEY}')"))
    else:
        if NucSegMethod.from_string(segmentation_method) != NucSegMethod.CELLPOSE:
            errors.append(ConfigurationValueError(
                f"Illegal or unsupported nuclei detection method (from '{NucDetector.DETECTION_METHOD_KEY}'): {segmentation_method}. Only '{NucSegMethod.CELLPOSE.value}' is supported."
                ))
    
    # Drift correction
    dc_method = Drifter.get_method_name(parameters)
    if dc_method and not Drifter.Methods.is_valid_name(dc_method):
        errors.append(ConfigurationValueError(f"Invalid drift correction method ({dc_method}); choose from: {', '.join(Drifter.Methods.values())}"))
    if "reg_ref_frame" in parameters:
        errors.append(ConfigurationValueError(f"The key for the timepoint as reference for drift correction has changed from reg_ref_frame to reg_ref_timepoint; update your config."))
    
    # Spot detection
    detection_timepoints_key: Literal["spot_frame"] = "spot_frame"
    if detection_timepoints_key in parameters:
        errors.append(ConfigurationValueError(f"From version 0.7, {detection_timepoints_key} is prohibited; use the imaging rounds configuration file."))
    # As of v0.14.0, no longer allow crosstalk subtraction since FISH spots will be filtered by proximity to beads.
    if CROSSTALK_SUBTRACTION_KEY in parameters:
        errors.append(ConfigurationValueError(f"Crosstalk subtraction ('{CROSSTALK_SUBTRACTION_KEY}') is no longer supported."))
    crosstalk_channel_key: Literal["crosstalk_ch"] = "crosstalk_ch"
    if crosstalk_channel_key in parameters:
        errors.append(ConfigurationValueError(f"Crosstalk subtraction channel ('{crosstalk_channel_key}') is no longer supported."))
    # Detection methods and parameters
    spot_detection_method = parameters.get(SPOT_DETECTION_METHOD_KEY)
    if spot_detection_method is None:
        errors.append(ConfigurationValueError(f"No spot detection method ('{SPOT_DETECTION_METHOD_KEY}') specified!"))
    elif spot_detection_method == DetectionMethod.INTENSITY.value:
        errors.append(ConfigurationValueError(f"Prohibited (or unsupported) spot detection method: '{spot_detection_method}'"))
    try:
        min_sep = get_minimum_regional_spot_separation(rounds_config)
    except KeyError:
        errors.append(ConfigurationValueError(f"No minimum spot separation specified in imaging rounds configuration!"))
    else:
        if not isinstance(min_sep, (int, float)) or min_sep < 0:
            errors.append(ConfigurationValueError(f"Illegal minimum spot separation value in imaging rounds configuration: {min_sep}"))

    # Spot merge
    try:
        min_pixels_to_avoid_spot_merge = parameters[KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS]
    except KeyError:
        errors.append(ConfigurationValueError(f"Missing key ('{KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS}') for minimal ROI separation to avoid merge"))
    else:
        if not isinstance(min_pixels_to_avoid_spot_merge, int):
            errors.append(ConfigurationValueError(
                f"Non-integer ({type(min_pixels_to_avoid_spot_merge).__name__}) value for min pixel count to avoid spot merge: {min_pixels_to_avoid_spot_merge}"
            ))
        elif min_pixels_to_avoid_spot_merge < 0:
            errors.append(ConfigurationValueError(
                f"Min pixel count to avoid spot merge must be nonnegative, not {min_pixels_to_avoid_spot_merge}"
            ))

    # Spot filtration
    match determine_bead_timepoint_for_spot_filtration(
        params_config=parameters, 
        image_rounds=rounds_config[IMAGING_ROUNDS_KEY],
    ):
        case result.Result(tag="ok", ok=_):
            pass # OK, nothing to do
        case result.Result(tag="error", error=err):
            errors.append(err)
        case outcome:
            raise Exception(f"Unexpected outcome from determination of beads timepoint for spot filtration: {outcome}")

    # Tracing
    if parameters.get("mask_fits", False):
        errors.append(ConfigurationValueError(MASK_FITS_ERROR_MESSAGE))
    try:
        probe_trace_exclusions = rounds_config[TRACING_SUPPORT_EXCLUSIONS_KEY]
    except KeyError:
        errors.append(f"Config (from {rounds_config}) lacks probes to exclude from tracing support ('{TRACING_SUPPORT_EXCLUSIONS_KEY}')!")
    else:
        if not isinstance(probe_trace_exclusions, list):
            typename = type(probe_trace_exclusions).__name__
            errors.append(f"Probes to exclude from tracing support ('{TRACING_SUPPORT_EXCLUSIONS_KEY}') isn't a list, but rather {typename}!")
        elif len(probe_trace_exclusions) == 0:
            errors.append(f"List of probes to exclude from tracing support ('{TRACING_SUPPORT_EXCLUSIONS_KEY}') is empty!")

    return errors


def workflow(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    if not LOOPTRACE_JAR_PATH.is_file():
        # Handle this separately since it's in the code itself and is not part of the config file.
        raise MissingJarError(LOOPTRACE_JAR_PATH)
    errors = find_config_file_errors(rounds_config=rounds_config, params_config=params_config)
    if errors:
        raise ConfigFileCrash(errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the provided looptrace configuration files.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    args = parser.parse_args()
    workflow(rounds_config=args.rounds_config, params_config=args.params_config)
