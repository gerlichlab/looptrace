"""Quality control / analysis of the drift correction step"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import *

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tqdm
import yaml

from gertils.pathtools import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import ImageHandler
from looptrace.filepaths import simplify_path
from looptrace.gaussfit import fitSymmetricGaussian3D
from looptrace import image_io
from looptrace import image_processing_functions as ip


SIGNAL_NOISE_RATIO_NAME = "A_to_BG"

# TODO: switch from print() to logging


def parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality control / analysis of the drift correction step", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-C", "--config-file", required=True, type=ExtantFile.from_string, help="Path to the main looptrace config file used for current data processing and analysis")
    parser.add_argument("-I", "--images-folder", required=True, type=ExtantFolder.from_string, help="Path to folder with images used for drift correction")
    parser.add_argument("--drift-correction-table", type=ExtantFile.from_string, help="Path to drift correction table")
    parser.add_argument("--cores", type=int, help="Number of processors to use")
    parser.add_argument("--num-bead-rois", type=int, help="Number of bead ROIs to subsample")
    parser.add_argument("--reference-FOV", type=int, help="Index of single FOV to use as reference")
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
                errors.append(TypeError(f"Value for '{attr_name}' atttribute is not int, but {type(attr_value).__name__}"))
            elif attr_value < 0:
                errors.append(ValueError(f"Value for '{attr_name}' atttribute is negative: {attr_value}"))
        return errors
        
    def __post_init__(self):
        errors = self._invalidate()
        if errors:
            raise IllegalParametersError(errors)
        

@dataclasses.dataclass
class BeadFiltrationParameters:
    """A bundle of parameters related to filtration of beads for this fiducial accuracy analysis"""
    num_rois: int
    min_signal_to_noise: Union[int, float]

    def _invalidate(self):
        errors = []
        if not isinstance(self.num_rois, int):
            errors.append(TypeError(f"ROI count must be natural number, but got {type(self.num_rois).__name__}"))
        elif self.num_rois < 1:
            errors.append(ValueError(f"ROI count must be natural number, but got {self.num_rois}"))
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
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)


def process_single_FOV_single_reference_frame(imgs: List[np.ndarray], drift_table: pd.DataFrame, full_pos: int, bead_detection_params: BeadDetectionParameters, bead_filtration_params: BeadFiltrationParameters, camera_params: CameraParameters) -> pd.DataFrame:
    T = imgs[full_pos].shape[0]
    C = imgs[full_pos].shape[1]
    print(f"Generating bead ROIs for DC accuracy analysis, full_pos: {full_pos}")
    ref_rois = ip.generate_bead_rois(imgs[full_pos][bead_detection_params.reference_frame, bead_detection_params.reference_channel].compute(), threshold=bead_detection_params.threshold, min_bead_int=bead_detection_params.min_intensity, n_points=-1)
    rois = ref_rois[np.random.choice(ref_rois.shape[0], bead_filtration_params.num_rois, replace=False)]
    bead_roi_px = bead_detection_params.roi_pixels
    dims = (len(rois), T, C, bead_roi_px, bead_roi_px, bead_roi_px)
    print(f"Dims: {dims}")
    bead_imgs = np.zeros(dims)
    bead_imgs_dc = np.zeros(dims)

    fits = []
    for t in tqdm.tqdm(range(T)):
        print(f"Frame: {t}")
        # TODO: this requires that the drift table be ordered such that the FOVs are as expected; need flexibility.
        pos = drift_table.position.unique()[full_pos]
        print(f"Position: {pos}")
        course_shift = drift_table[(drift_table.position == pos) & (drift_table.frame == t)][['z_px_course', 'y_px_course', 'x_px_course']].values[0]
        fine_shift = drift_table[(drift_table.position == pos) & (drift_table.frame == t)][['z_px_fine', 'y_px_fine', 'x_px_fine']].values[0]
        for c in [bead_detection_params.reference_channel]:#range(C):
            img = imgs[full_pos][t, c].compute()
            for i, roi in enumerate(rois):
                bead_img = ip.extract_single_bead(roi, img, bead_roi_px=bead_roi_px, drift_course=course_shift)
                fit = fitSymmetricGaussian3D(bead_img, sigma=1, center='max')[0]
                fits.append([full_pos, t, c, i] + list(fit))
                bead_imgs[i, t, c] = bead_img.copy()
                bead_imgs_dc[i, t, c] = ndi.shift(bead_img, shift=fine_shift)

    fits = pd.DataFrame(fits, columns=['full_pos','t', 'c', 'roi', 'BG', 'A', 'z_loc', 'y_loc', 'x_loc', 'sigma_z', 'sigma_xy'])
    
    fits.loc[:, ['y_loc', 'x_loc', 'sigma_xy']] = fits.loc[:,  ['y_loc', 'x_loc', 'sigma_xy']] * camera_params.nanometers_xy # Scale xy coordinates to nm (use xy pixel size from exp).
    fits.loc[:, ['z_loc', 'sigma_z']] = fits.loc[:, ['z_loc', 'sigma_z']] * camera_params.nanometers_z #Scale z coordinates to nm (use slice spacing from exp)

    # TODO: update if ever allowing channel (reg_ch_template) to be List[int] rather than simple int.
    ref_points = fits.loc[(fits.t == bead_detection_params.reference_frame) & (fits.c == bead_detection_params.reference_channel), ['z_loc', 'y_loc', 'x_loc']].to_numpy() # Fits of fiducial beads in ref frame
    print(f"Reference point count: {len(ref_points)}")
    res = []
    for t in tqdm.tqdm(range(T)):
        # TODO: update if ever allowing channel (reg_ch_template) to be List[int] rather than simple int.
        mov_points = fits.loc[(fits.t == t) & (fits.c == bead_detection_params.reference_channel), ['z_loc', 'y_loc', 'x_loc']].to_numpy() # Fits of fiducial beads in moving frame
        print(f"mov_points shape: {mov_points.shape}")
        shift = drift_table.loc[(drift_table.position == pos) & (drift_table.frame == t), ['z_px_fine', 'y_px_fine', 'x_px_fine']].values[0]
        shift[0] =  shift[0] * camera_params.nanometers_z # Extract calculated drift correction from drift correction file.
        shift[1] =  shift[1] * camera_params.nanometers_xy
        shift[2] =  shift[2] * camera_params.nanometers_xy
        fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']] = mov_points + shift #Apply precalculated drift correction to moving fits
        fits.loc[(fits.t == t), ['z_dc_rel', 'y_dc_rel', 'x_dc_rel']] =  np.abs(fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']].to_numpy() - ref_points)# Find offset between moving and reference points.
        fits.loc[(fits.t == t), ['euc_dc_rel']] = np.sqrt(np.sum((fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']].to_numpy() - ref_points)**2, axis=1)) # Calculate 3D eucledian distance between points and reference.
        res.append(shift)

    # TODO: paramrterise by config and/or CLI.
    fits[SIGNAL_NOISE_RATIO_NAME] = fits['A'] / fits['BG']
    fits['QC'] = 0
    fits.loc[fits[SIGNAL_NOISE_RATIO_NAME] >= bead_filtration_params.min_signal_to_noise, 'QC'] = 1

    return fits


def get_beads_channel(config: Dict[str, Any]) -> int:
    return config["reg_ch_template"]


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        drift_correction_table_file: Optional[ExtantFile] = None, 
        cores: int = None, 
        num_bead_rois: Optional[int] = None, 
        full_pos: Optional[int] = None, 
    ) -> pd.DataFrame:
    # TODO: how to handle case when output already exists

    config_file = simplify_path(config_file)
    images_folder = simplify_path(images_folder)

    if drift_correction_table_file is None:
        print("Determining drift correction table path...")
        drift_correction_table_file = Drifter(ImageHandler(config_path=config_file, image_path=images_folder)).dc_file_path
    if not os.path.isfile(drift_correction_table_file):
        raise FileNotFoundError(drift_correction_table_file)

    print(f"Parsing looptrace configuration file: {config_file}")
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)

    output_folder = Path(config['analysis_path'])
    
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
    num_bead_rois = num_bead_rois or config.get("n_bead_rois_dc_accuracy", 500)
    min_signal_to_noise = config[SIGNAL_NOISE_RATIO_NAME]
    bead_filtration_params = BeadFiltrationParameters(num_rois=num_bead_rois, min_signal_to_noise=min_signal_to_noise)
    print(f"Bead filtration parameters: {bead_filtration_params}")

    # Camera parameters
    camera_params = CameraParameters(
        nanometers_xy=config["xy_nm"], 
        nanometers_z=config["z_nm"], 
    )
    print(f"Camera parameters: {camera_params}")

    realised_params_file = output_folder / "drift_correction_accuracy_analysis_parameters.json"
    print(f"Writing parameters file: {realised_params_file}")
    with open(realised_params_file, 'w') as fh:
        json.dump(
            {"bead_detection": bead_detection_params, "bead_filtration": bead_filtration_params, "camera": camera_params}, 
            fh, 
            indent=2,
            cls=DataclassCapableEncoder
            )

    seqfish_images_folder = images_folder / config['reg_input_moving'] # TODO: reconcile with 'reg_input_template'
    print(f"Reading zarr to dask: {seqfish_images_folder}")
    imgs, _ = image_io.multi_ome_zarr_to_dask(str(seqfish_images_folder))
    print(f"Reading drift correction table: {drift_correction_table_file}")
    drift_table = pd.read_csv(drift_correction_table_file, index_col=0)

    if full_pos is not None:
        # TODO: parameterise with config.
        fits = process_single_FOV_single_reference_frame(imgs, drift_table, full_pos, bead_detection_params, bead_filtration_params, camera_params)
    else:
        fov_indices = range(len(drift_table.position.unique()))
        # TODO: parameterise with config.
        func_args = ((imgs, drift_table, idx, bead_detection_params, bead_filtration_params, camera_params) for idx in fov_indices)
        cores = cores or config.get('num_cores_dc_analysis', 1)
        if cores == 1:
            single_fov_fits = (process_single_FOV_single_reference_frame(*args) for args in func_args)
        else:
            with mp.Pool(cores) as workers:
                single_fov_fits = workers.starmap(process_single_FOV_single_reference_frame, func_args)
        fits = pd.concat(single_fov_fits)
    
    fits_output_file = _get_dc_fits_filepath(output_folder)
    print(f"Writing fits file: {fits_output_file}")
    fits.to_csv(fits_output_file, index=False, sep="\t")
    return fits


def run_visualisation(config_file: ExtantFile):
    config_file = simplify_path(config_file)
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    output_folder = config['analysis_path']
    fits_file = _get_dc_fits_filepath(output_folder)
    # TODO: spin off this function to make pipeline checkpointable after long-running DC analysis.
    analysis_script_file = os.path.join(os.path.dirname(__file__), "drift_correct_accuracy_analysis.R")
    if not os.path.isfile(analysis_script_file):
        raise FileNotFoundError(f"Missing drift correction analysis script: {analysis_script_file}")
    analysis_cmd_parts = ["Rscript", analysis_script_file, "-i", str(fits_file), "-o", output_folder, "--beads-channel", str(get_beads_channel(config))]
    print(f"Analysis command: {' '.join(analysis_cmd_parts)}")
    return subprocess.check_call(analysis_cmd_parts)


def _get_dc_fits_filepath(folder: Union[str, Path]) -> str:
    return os.path.join(folder, "drift_correction_accuracy.fits.tsv")


if __name__ == "__main__":
    opts = parse_cmdl(sys.argv[1:])
    workflow(
        config_file=opts.config_file,
        images_folder=opts.images_folder, 
        drift_correction_table_file=opts.drift_correction_table, 
        cores=opts.cores,
        num_bead_rois=opts.num_bead_rois,
        full_pos=opts.reference_FOV,
    )
