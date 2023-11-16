"""Quality control / analysis of the drift correction step"""

import argparse
import dataclasses
from joblib import Parallel, delayed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import *
import warnings

import numpy as np
import pandas as pd
import tqdm
import yaml

from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import ImageHandler, handler_from_cli
from looptrace.bead_roi_generation import extract_single_bead
from looptrace.filepaths import get_analysis_path
from looptrace.gaussfit import fitSymmetricGaussian3D
from looptrace import image_io, SIGNAL_NOISE_RATIO_NAME
from looptrace.numeric_types import NumberLike


def parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality control / analysis of the drift correction step; namely, how effective has drift correction been at reducing the distance between points which should coincide?", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        "-C", "--config-file", required=True, type=ExtantFile.from_string, 
        help="Path to the main looptrace config file used for current data processing and analysis",
        )
    parser.add_argument(
        "-I", "--images-folder", required=True, type=ExtantFolder.from_string, 
        help="Path to folder with images used for drift correction",
        )
    parser.add_argument(
        "--drift-correction-table", type=ExtantFile.from_string, 
        help="Path to drift correction table; if unspecified, infer from the config file and images folder",
        )
    parser.add_argument(
        "--reference-FOV", type=int, 
        help="Index of single FOV to use as reference; if unspecified, all FOVs will be used.",
        )
    return parser.parse_args(cmdl)


class IllegalParametersError(Exception):
    """Exception subtype for an illegal parameterisation"""
    def __init__(self, errors: List[Exception]) -> None:
        super().__init__(f"{len(errors)} error(s): {', '.join(errors)}")


@dataclasses.dataclass
class BeadDetectionParameters:
    """A bundle of parameters related to bead detection for assessment of drift correction accuracy"""
    reference_frame: int
    reference_channel: int
    threshold: int
    min_intensity: int
    roi_pixels: int

    def _invalidate(self):
        int_valued_attributes = ("reference_frame", "reference_channel", "threshold", "min_intensity", "roi_pixels")
        errors = []
        for attr_name in int_valued_attributes:
            attr_value = getattr(self, attr_name)
            if not isinstance(attr_value, int):
                errors.append(TypeError(f"Value for '{attr_name}' attribute is not int, but {type(attr_value).__name__}"))
            elif attr_value < 0:
                errors.append(ValueError(f"Value for '{attr_name}' attribute is negative: {attr_value}"))
        return errors
        
    def __post_init__(self):
        errors = self._invalidate()
        if errors:
            raise IllegalParametersError(errors)
        

@dataclasses.dataclass
class BeadFiltrationParameters:
    """A bundle of parameters related to filtration of beads for this fiducial accuracy analysis"""
    max_num_rois: int
    min_signal_to_noise: Union[int, float]

    def _invalidate(self):
        errors = []
        if not isinstance(self.max_num_rois, int):
            errors.append(TypeError(f"ROI count must be natural number, but got {type(self.max_num_rois).__name__}"))
        elif self.max_num_rois < 1:
            errors.append(ValueError(f"ROI count must be natural number, but got {self.max_num_rois}"))
        if not isinstance(self.min_signal_to_noise, (int, float)):
            errors.append(TypeError(f"Min signal-to-noise ratio must be int or float, but got {type(self.min_signal_to_noise).__name__}"))
        elif self.min_signal_to_noise < 0:
            errors.append(ValueError(f"Min signal-to-noise ratio cannot be negative: {self.min_signal_to_noise}"))
        return errors

    def __post_init__(self):
        errors = self._invalidate()
        if errors:
            raise IllegalParametersError(errors)


@dataclasses.dataclass
class CameraParameters:
    """A bundle of parameters related to properties of the camera used for imaging"""
    nanometers_xy: Union[int, float]
    nanometers_z: Union[int, float]

    def _invalidate(self):
        errors = []
        for attr_name in ("nanometers_xy", "nanometers_xy"):
            attr_value = getattr(self, attr_name)
            if not isinstance(attr_value, (int, float)):
                errors.append(TypeError(f"'{attr_name}' value must be int or float, but got {type(attr_value).__name__}"))
            elif not attr_value > 0:
                errors.append(ValueError(f"'{attr_name}' value must be positive, but got {attr_value}"))

    def __post_init__(self):
        errors = self._invalidate()
        if errors:
            raise IllegalParametersError(errors)


class DataclassCapableEncoder(json.JSONEncoder):
    """Facilitate serialisation of the parameters dataclasses in this module, for data provenance."""
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


@dataclasses.dataclass
class ReferenceImageStackDefinition:
    index: int # TODO: refine as nonnegative
    image_stack: np.ndarray

    def __post_init__(self):
        if 5 != len(self.image_stack.shape):
            raise TypeError(f"A reference image stack should be a 5-dimensional array; got {len(self.image_stack.shape)}")
    
    @property
    def num_channels(self) -> int:
        return self.image_stack.shape[1]

    @property
    def num_timepoints(self) -> int:
        return self.image_stack.shape[0]


def process_single_FOV_single_reference_frame(
    image_handler: ImageHandler,
    reference_image_stack_definition: ReferenceImageStackDefinition,
    drift_table: pd.DataFrame,
    bead_detection_params: BeadDetectionParameters, 
    bead_filtration_params: BeadFiltrationParameters, 
    camera_params: CameraParameters, 
    timepoints: Optional[Iterable[int]],
    ) -> pd.DataFrame:
    """
    Compute the drift correction accuracy for a single hybridisation round / imaging frame, within a single field-of-view.

    Parameters
    ----------
    reference_image_stack_definition : Sequence of np.ndarray
        The full collection of imaging data; nasmely, a list-/array-like indexed by position/FOV, in which each element is a 
        five-dimensional array itself (t, c, z, y, x) -- that is, a stack (z) of 2D images (y, x) for each imaging channel (c) 
        for each hybridisation round / timepoint (t). In other words, each array in this collection represents an entire 
        field of view / "position", with many timepoints (hybridisation rounds)--and potentially multiple imaging channels--of 
        data represented therein.\
    drift_table : pd.DataFrame
        The table of precomputed drift correction information; namely, the drift correction values whose accuracy / efficacy 
        is being assessed by this program
    reference_fov : int
        The index of the position/FOV in which to compute the drift correction accuracy in this particular function call. 
        The expectation would be that this function may be called multiple times, computing these data for different 
        timepoints also within this FOV, and perhaps in other FOVs as well.
    bead_detection_params : BeadDetectionParameters
        The parameters relevant to detection of the fiducial beads in the raw images
    bead_filtration_params : BeadFiltrationParameters
        The parameters relevant to which beads to use and which to discard
    camera_params : CameraParameters
        The parameters about the camera used to capture the imaging data passed here
    timepoints : Iterable of int, optional
        Specific hybridisation rounds/frames/timepoints to analyse; if unspecified, use all available
    
    Returns
    -------
    pd.DataFrame
        A table in which there is Gaussian fit information, and distance information, for each bead point sampled for the 
        indicated hybridisation timepoint within the indicated FOV
    """
    image_stack = reference_image_stack_definition.image_stack
    T = reference_image_stack_definition.num_timepoints
    iter_time = (lambda: range(T)) if timepoints is None else (lambda: iter(timepoints))
    
    # Get the bead ROIs for the current combo of FOV and frame/timepoint.
    fov_idx = reference_image_stack_definition.index
    roi_centers = image_handler.read_bead_rois_file_accuracy(pos_idx=fov_idx, frame=bead_detection_params.reference_frame)
    if len(roi_centers) != image_handler.num_bead_rois_for_drift_correction_accuracy:
        warnings.warn(RuntimeWarning(f"Fewer ROIs available ({len(roi_centers)}) than requested ({image_handler.num_bead_rois_for_drift_correction_accuracy}) for FOV {fov_idx}"))

    bead_roi_px = bead_detection_params.roi_pixels
    
    # TODO: this requires that the drift table be ordered such that the FOVs are as expected; need flexibility.
    pos = drift_table.position.unique()[fov_idx]
    print(f"Inferred position (for reference FOV index {fov_idx}): {pos}")
    curr_fov_drift_subtable = drift_table[drift_table.position == pos]

    # TODO: could type-refine the argument values to these parameters (which should be nonnegative).
    def proc1(frame_index: int, ref_ch: int, centroid: np.ndarray) -> Iterable[NumberLike]:
        coarse_shift = curr_fov_drift_subtable[curr_fov_drift_subtable.frame == frame_index][['z_px_course', 'y_px_course', 'x_px_course']].values[0]
        img = image_stack[frame_index, ref_ch].compute()
        bead_img = extract_single_bead(centroid, img, bead_roi_px=bead_roi_px, drift_course=coarse_shift)
        return fitSymmetricGaussian3D(bead_img, sigma=1, center='max')[0]

    skips = image_handler.position_frame_pairs_with_severe_problems
    print(f"(FOV, time) pairs to skip: {skips}")

    fits = Parallel(n_jobs=-1, prefer='threads')(
        delayed(lambda t, c, roi: [fov_idx, t, c, i] + list(proc1(frame_index=t, ref_ch=c, centroid=roi)))(t=t, c=c, roi=roi) 
        for t in tqdm.tqdm(filter(lambda t: (fov_idx, t) not in skips, iter_time()))
        for c in [bead_detection_params.reference_channel] 
        for i, roi in enumerate(roi_centers)
        )
    fits = pd.DataFrame(fits, columns=['reference_fov', 't', 'c', 'roi', 'BG', 'A', 'z_loc', 'y_loc', 'x_loc', 'sigma_z', 'sigma_xy'])
    fits = express_pixel_columns_as_nanometers(fits=fits, xy_cols=('y_loc', 'x_loc', 'sigma_xy'), z_cols=('z_loc', 'sigma_z'), camera_params=camera_params)
    
    # TODO: update if ever allowing channel (reg_ch_template) to be List[int] rather than simple int.
    ref_points = fits.loc[(fits.t == bead_detection_params.reference_frame) & (fits.c == bead_detection_params.reference_channel), ['z_loc', 'y_loc', 'x_loc']].to_numpy() # Fits of fiducial beads in ref frame
    print(f"Reference point count: {len(ref_points)}")
    res = []
    for t in tqdm.tqdm(iter_time()):
        # TODO: update if ever allowing channel (reg_ch_template) to be List[int] rather than simple int.
        mov_points = fits.loc[(fits.t == t) & (fits.c == bead_detection_params.reference_channel), ['z_loc', 'y_loc', 'x_loc']].to_numpy() # Fits of fiducial beads in moving frame
        print(f"mov_points shape: {mov_points.shape}")
        shift = drift_table.loc[(drift_table.position == pos) & (drift_table.frame == t), ['z_px_fine', 'y_px_fine', 'x_px_fine']].values[0]
        shift[0] =  shift[0] * camera_params.nanometers_z # Extract calculated drift correction from drift correction file.
        shift[1] =  shift[1] * camera_params.nanometers_xy
        shift[2] =  shift[2] * camera_params.nanometers_xy
        # TODO: some of these statements can be combined to share values without needing to reindex and reconvert between data types.
        fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']] = mov_points + shift #Apply precalculated drift correction to moving fits
        # TODO: what happens if there's more than 1 channel here (within each timepoint), but ref points are just from 1 channel (dimensionality problem?)
        fits.loc[(fits.t == t), ['z_dc_rel', 'y_dc_rel', 'x_dc_rel']] = np.abs(fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']].to_numpy() - ref_points)# Find offset between moving and reference points.
        fits.loc[(fits.t == t), ['euc_dc_rel']] = np.sqrt(np.sum((fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']].to_numpy() - ref_points)**2, axis=1)) # Calculate 3D eucledian distance between points and reference.
        res.append(shift) # NB: the shift values are expressed in units of nanometers rather than pixels.

    # TODO: consider returning (and/or writing to disk) the resulting shift array (1 shift (1D array of size 3) per timepoint, for this field of view).
    return finalise_fits_frame(fits=fits, min_signal_noise_ratio=bead_filtration_params.min_signal_to_noise)


def express_pixel_columns_as_nanometers(fits: pd.DataFrame, xy_cols: Iterable[str], z_cols: Iterable[str], camera_params: "CameraParameters") -> pd.DataFrame:
    """
    Convert the values in the relevant columns to be expressed in nanometers rather than in pixels.

    Parameters
    ----------
    fits : pd.DataFrame
        The frame in which values are to be converted
    xy_cols : Iterable of str
        Names of columns representing values in x- or y-direction
    z_cols : Iterable of str
        Names of columns representing values in z-direction
    camera_params : Camera
        Parameters bundle related to the camera used for imaging; here, for number of nanometers per pixel in each direction
    
    Returns
    -------
    pd.DataFrame
        Identical to input, just with the values in the relevant columns changed to reflect the change of units
    """
    xy_cols = list(xy_cols)
    z_cols = list(z_cols)
    fits.loc[:, xy_cols] = fits.loc[:, xy_cols] * camera_params.nanometers_xy # Scale xy coordinates to nm (use xy pixel size from exp).
    fits.loc[:, z_cols] = fits.loc[:, z_cols] * camera_params.nanometers_z #Scale z coordinates to nm (use slice spacing from exp)
    return fits


def finalise_fits_frame(fits: pd.DataFrame, min_signal_noise_ratio: NumberLike) -> pd.DataFrame:
    """
    Add the signal-to-noise ratio measurements and quality control pass/fail label

    Parameters
    ----------
    fits : pd.DataFrame
    The frame in which data are to be quality controlled
    min_signal_noise_ratio : NumberLike
        The minimum value of signal-to-noise ratio that a point can have and still pass QC

    """
    fits[SIGNAL_NOISE_RATIO_NAME] = fits['A'] / fits['BG']
    fits['QC'] = 0
    fits.loc[fits[SIGNAL_NOISE_RATIO_NAME] >= min_signal_noise_ratio, 'QC'] = 1
    return fits


def get_beads_channel(config: Dict[str, Any]) -> int:
    """Simple accessor for getting the fudicial beads' imaging channel"""
    return config["reg_ch_template"]


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        drift_correction_table_file: Union[None, Path, ExtantFile] = None, 
        reference_fov: Optional[int] = None, 
        timepoints: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
    """
    Pull random subset of beads and compute the distance that remains even after drift correction.

    Parameters
    ----------
    config_file : gertils.ExtantFile
        Path to the main looptrace configuration file
    images_folder : gertils.ExtantFolder
        Path to the folder with an experiment's imaging data
    drift_correction_table_file : gertils.ExtantFile, optional
        Path to the table of drift correction values; if unspecified, this can be inferred from config_file and images_folder
    reference_fov : int, optional
        Index (0-based) of position/field-of-view to use; if unspecified, use all FOVs
    timepoints : Iterable of int, optional
        Specific hybridisation rounds/frames/timepoints to analyse; if unspecified, use all available
        
    Returns
    -------
    pd.DataFrame
        A frame in which there is information about each point measured for remaining distance, 
        including values from a fit of a 3D Gaussian to the detected point which represents a bead.
    """
    # TODO: how to handle case when output already exists

    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    if drift_correction_table_file is None:
        print("Determining drift correction table path...")
        drift_correction_table_file = Drifter(H).dc_file_path__fine
    elif isinstance(drift_correction_table_file, ExtantFile):
        drift_correction_table_file = drift_correction_table_file.path
    if not os.path.isfile(drift_correction_table_file):
        raise FileNotFoundError(drift_correction_table_file)

    print(f"Parsing looptrace configuration file: {config_file.path}")
    with open(config_file.path, 'r') as fh:
        config = yaml.safe_load(fh)

    output_folder = Path(get_analysis_path(config))
    
    # Detection parameters
    bead_detection_params = BeadDetectionParameters(
        reference_frame = config["reg_ref_frame"],
        reference_channel = get_beads_channel(config), 
        threshold=config["bead_threshold"], 
        min_intensity=config["min_bead_intensity"],
        roi_pixels=config["bead_roi_size"],
    )
    print(f"Bead detection parameters: {bead_detection_params}")
    
    # Filtration parameters
    bead_filtration_params = BeadFiltrationParameters(
        max_num_rois=config["num_bead_rois_for_drift_correction_accuracy"],
        min_signal_to_noise=config[SIGNAL_NOISE_RATIO_NAME],
        )
    print(f"Bead filtration parameters: {bead_filtration_params}")

    # Camera parameters
    camera_params = CameraParameters(
        nanometers_xy=config["xy_nm"], 
        nanometers_z=config["z_nm"], 
    )
    print(f"Camera parameters: {camera_params}")

    # For provenance, write the collection of parameters to be used.
    realised_params_file = output_folder / "drift_correction_accuracy_analysis_parameters.json"
    print(f"Writing parameters file: {realised_params_file}")
    with open(realised_params_file, 'w') as fh:
        json.dump(
            {"bead_detection": bead_detection_params, "bead_filtration": bead_filtration_params, "camera": camera_params}, 
            fh, 
            indent=2,
            cls=DataclassCapableEncoder
            )

    # Read the actual FISH images.
    seqfish_images_folder = images_folder.path / config['reg_input_moving'] # TODO: reconcile with 'reg_input_template'
    print(f"Reading zarr to dask: {seqfish_images_folder}")
    imgs, _ = image_io.multi_ome_zarr_to_dask(str(seqfish_images_folder))
    
    # Read the table of precomputed drift correction values.
    print(f"Reading drift correction table: {drift_correction_table_file}")
    drift_table = pd.read_csv(drift_correction_table_file, index_col=0)

    # Subsample beads and compute remaining distance from reference point, even after drift correction.
    # Whether this is done in just a single FOV or across all FOVs is determined by the command-line specification.
    if reference_fov is not None:
        # TODO: parameterise with config.
        refspecs = [ReferenceImageStackDefinition(index=reference_fov, image_stack=imgs[reference_fov])]
    else:
        refspecs = (ReferenceImageStackDefinition(index=i, image_stack=imgs[i]) for i in range(len(drift_table.position.unique())))
    fits = (
        process_single_FOV_single_reference_frame(
            image_handler=H,
            reference_image_stack_definition=spec, 
            drift_table=drift_table, 
            bead_detection_params=bead_detection_params, 
            bead_filtration_params=bead_filtration_params, 
            camera_params=camera_params, 
            timepoints=timepoints,
            ) 
        for spec in refspecs
        )
    fits = pd.concat(fits)
    
    # Write the individual bead Gaussian fits, spatial coordinates, and distance values.
    fits_output_file = _get_dc_fits_filepath(output_folder)
    print(f"Writing fits file: {fits_output_file}")
    fits.to_csv(fits_output_file, index=False, sep=",")
    return fits


def run_visualisation(config_file: ExtantFile):
    with open(config_file.path, 'r') as fh:
        config = yaml.safe_load(fh)
    output_folder = get_analysis_path(config)
    fits_file = _get_dc_fits_filepath(output_folder)
    analysis_script_file = Path(os.path.dirname(__file__)) / "drift_correct_accuracy_analysis.R"
    if not analysis_script_file.is_file():
        raise FileNotFoundError(f"Missing drift correction analysis script: {analysis_script_file}")
    analysis_cmd_parts = [
        "Rscript", 
        str(analysis_script_file), 
        "-i", 
        str(fits_file), 
        "-o", 
        output_folder, 
        "--beads-channel", 
        str(get_beads_channel(config)),
        ]
    print(f"Analysis command: {' '.join(analysis_cmd_parts)}")
    return subprocess.check_call(analysis_cmd_parts)


def _get_dc_fits_filepath(folder: Union[str, Path]) -> str:
    return os.path.join(folder, "drift_correction_accuracy.fits.csv")


if __name__ == "__main__":
    opts = parse_cmdl(sys.argv[1:])
    workflow(
        config_file=opts.config_file,
        images_folder=opts.images_folder, 
        drift_correction_table_file=opts.drift_correction_table, 
        reference_fov=opts.reference_FOV,
    )
