# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import dataclasses
from typing import *

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from tqdm import tqdm

from looptrace.SpotPicker import RoiOrderingSpecification
import looptrace.image_processing_functions as ip
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from looptrace.numeric_types import NumberLike


ROI_FIT_COLUMNS = ["BG", "A", "z_px", "y_px", "x_px", "sigma_z", "sigma_xy"]
MASK_FITS_ERROR_MESSAGE = "Masking fits for tracing currently isn't supported!"


@dataclasses.dataclass
class BackgroundSpecification:
    frame_index: int
    drifts: Iterable[np.ndarray]


@dataclasses.dataclass
class FunctionalForm:
    function: callable
    dimensionality: int


class Tracer:

    def __init__(self, image_handler, trace_beads=False, array_id=None):
        '''
        Initialize Tracer class with config read in from YAML file.
        '''
        self.image_handler = image_handler
        self.config_path = image_handler.config_path
        self.config = image_handler.config
        self.drift_table = image_handler.tables[image_handler.spot_input_name + '_drift_correction_fine']
        self.images = self.image_handler.images[self.config['trace_input_name']]
        self.pos_list = self.image_handler.image_lists[image_handler.spot_input_name]
        if trace_beads:
            self.roi_table = image_handler.tables[image_handler.spot_input_name + '_bead_rois']
            finalise_suffix = lambda p: p.replace(".csv", "_beads.csv")
        else:
            self.roi_table = image_handler.tables[image_handler.spot_input_name + '_rois']
            finalise_suffix = lambda p: p
        self.all_rois = image_handler.tables[image_handler.spot_input_name + '_dc_rois']

        fit_func_specs = {
            'LS': FunctionalForm(function=fitSymmetricGaussian3D, dimensionality=3), 
            'MLE': FunctionalForm(function=fitSymmetricGaussian3DMLE, dimensionality=3)
            }
        fit_func_value = self.config['fit_func']
        try:
            self.fit_func_spec = fit_func_specs[fit_func_value]
        except KeyError as e:
            raise Exception(f"Unknown fitting function ('{fit_func_value}'); choose from: {', '.join(fit_func_specs.keys())}") from e

        self.array_id = array_id
        if self.array_id is not None:
            self.pos_list = [self.pos_list[int(self.array_id)]]
            self.roi_table = self.roi_table[self.roi_table.position.isin(self.pos_list)]
            self.all_rois = self.all_rois[self.all_rois.position.isin(self.pos_list)].reset_index(drop=True)
            traces_path = self.image_handler.out_path('traces.csv'[:-4] + '_' + str(self.array_id).zfill(4) + '.csv')
            self.images = self.images[self.roi_table.index.to_list()]
        else:
            traces_path = self.image_handler.out_path('traces.csv')
        
        # will be concatenated horizontally with fits; idempotent if already effectively unindexed
        self.all_rois = self.all_rois.reset_index(drop=True)
        
        self.traces_path = finalise_suffix(traces_path)

    @property
    def nanometers_per_pixel_xy(self) -> NumberLike:
        return self.config["xy_nm"]

    @property
    def nanometers_per_pixel_z(self) -> NumberLike:
        return self.config["z_nm"]

    def trace_all_rois(self) -> str:
        '''
        Fits 3D gaussian to previously detected ROIs across positions and timeframes.
    
        '''        
        try:
            #This only works for a single position at the time currently
            bg_frame_idx = self.image_handler.config['subtract_background']
        except KeyError:
            bg_spec = None
        else:
            pos_drifts = self.drift_table[self.drift_table.position.isin(self.pos_list)][['z_px_fine', 'y_px_fine', 'x_px_fine']].to_numpy()
            bg_spec = BackgroundSpecification(frame_index=bg_frame_idx, drifts=pos_drifts - pos_drifts[bg_frame_idx])

        spot_fits = find_trace_fits(
            fit_func_spec=self.fit_func_spec,
            images=(self.images[fn] for fn in sorted(self.images.files, key=RoiOrderingSpecification.get_file_key)), 
            mask_ref_frames=self.roi_table['frame'].to_list() if self.image_handler.config.get('mask_fits', False) else None, 
            background_specification=bg_spec, 
            cores=self.config.get("tracing_cores")
            )
        
        traces = finalise_traces(rois=self.all_rois, fits=spot_fits, z_nm=self.nanometers_per_pixel_z, xy_nm=self.nanometers_per_pixel_xy)
        
        print(f"Writing traces: {self.traces_path}")
        traces.to_csv(self.traces_path)

        return self.traces_path


# For parallelisation (multiprocessing) in the case of mask_fits being False.
def _iter_fit_args(
        fit_func_spec: FunctionalForm, 
        images: Iterable[np.ndarray], 
        bg_spec: Optional[BackgroundSpecification]
        ) -> Iterable[Tuple[FunctionalForm, np.ndarray]]:
    if bg_spec is None:
        # Iterating here over regional spots (pos_imgs)
        for pos_imgs in images:
            # Iterating here over individal timepoints / hybridisation rounds for each regional 
            for spot_img in pos_imgs:
                yield fit_func_spec, spot_img
    else:
        # Iterating here over regional spots (pos_imgs)
        for pos_imgs in images:
            # Iterating here over individal timepoints / hybridisation rounds for each regional 
            for spot_img in pos_imgs:
                yield fit_func_spec, spot_img.astype(np.int16) - pos_imgs[bg_spec.frame_index].astype(np.int16)                


def finalise_traces(rois: pd.DataFrame, fits: pd.DataFrame, z_nm: NumberLike, xy_nm: NumberLike) -> pd.DataFrame:
    """
    Pair ROIs (single spots) table with row-by-row fits, apply drift correction, convert to nanometers, sort, and name columns.

    Parameters
    ----------
    rois : pd.DataFrame
        The table of data for each spot in each hybridisation frame
    fits : pd.DataFrame
        The table of functional form fits for each row in the ROIs frame
    z_nm : NumberLike
        Number of nanometers per pixel in the z-direction
    xy_nm : NumberLike
        Number of nanometers per pixel in the x- and y-directions

    Returns
    -------
    pd.DataFrame
        The result of joining (horizontally) the frames, applying drift correction, sorting, and applying units
    """
    traces = pair_rois_with_fits(rois=rois, fits=fits)
    #Apply fine scale drift to fits, and physcial units.
    traces = apply_fine_scale_drift_correction(traces)
    #traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
    traces = apply_pixels_to_nanometers(traces, z_nm_per_px=z_nm, xy_nm_per_px=xy_nm)
    traces = traces.sort_values(RoiOrderingSpecification.row_order_columns())
    traces.rename(columns={"roi_id": "trace_id"}, inplace=True)
    return traces


def pair_rois_with_fits(rois: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge (horizontally) the data from the individual spots / ROIs (1 per frame per regional spot) and the Gaussian fits.

    Parameters
    ----------
    rois : pd.DataFrame
        Individual spot data (1 per frame per regional spot)
    fits : pd.DataFrame
        Parameters for function fit to each individual spot
    
    Returns
    -------
    pd.DataFrame
        A frame combining the individual spot data with parameters of functional form fit to that data
    
    Raises
    ------
    ValueError
        If the indexes of the frames to combine don't match
    """
    if any(rois.index != fits.index):
        raise ValueError("Indexes of sports table and fits table don't match!")
    # TODO: fix this incredibly error-prone thing; #84
    traces = pd.concat([rois, fits], axis=1)
    return traces


def find_trace_fits(
        fit_func_spec: FunctionalForm, 
        images: Iterable[np.ndarray], 
        mask_ref_frames: Optional[List[int]], 
        background_specification: Optional[BackgroundSpecification], 
        cores: Optional[int] = None
        ) -> pd.DataFrame:
    """
    Fit distributions to each of the regional spots, but over all hybridisation rounds.

    Parameters
    ----------
    fit_func_spec : FunctionalForm
        Pair of function to fit to each spot, and dimensionality of the data for that fit (e.g. 3)
    images : Iterable of np.ndarray
        The collection of 4D arrays from the spot_images.npz (1 for each FOV + ROI combo)
    mask_ref_frames : list of int, optional
        Frames to use for masking when fitting, indexed by FOV
    background_specification : BackgroundSpecification, optional
        Bundle of index of hybridisation round (e.g. 0) to define as background, and associated 
        positional drifts
    cores : int, optional
        How many CPUs to use
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with one row per hybridisation round per FOV, with columns named to 
        denote the parameters of the optimised functional form for each one
    """
    # NB: For these iterations, each is expected to be a 4D array (first dimension being hybridisation round, and (z, y, x) for each).
    if mask_ref_frames:
        raise NotImplementedError(MASK_FITS_ERROR_MESSAGE)
        if background_specification is None:
            def finalise_spot_img(img, _):
                return img
        else:
            def finalise_spot_img(img, fov_imgs):
                return img.astype(np.int16) - fov_imgs[background_specification.frame_index].astype(np.int16)
        fits = []
        for p, pos_imgs in tqdm(enumerate(images), total=len(images)):
            ref_img = pos_imgs[mask_ref_frames[p]]
            #print(ref_img.shape)
            for t, spot_img in enumerate(pos_imgs):
                #if background_specification is not None:
                    #shift = ndi.shift(pos_imgs[background_specification.frame_index], shift=background_specification.drifts[t])
                    #spot_img = np.clip(spot_img.astype(np.int16) - shift, a_min = 0, a_max = None)
                spot_img = finalise_spot_img(spot_img, pos_imgs)
                fits.append(trace_single_roi(fit_func_spec=fit_func_spec, roi_img=spot_img, mask=ref_img))
    else:
        fits = Parallel(n_jobs=cores or -1)(
            delayed(trace_single_roi)(fit_func_spec=ff_spec, roi_img=spot_img) 
            for ff_spec, spot_img in _iter_fit_args(fit_func_spec=fit_func_spec, images=images, bg_spec=background_specification)
            )
    return pd.DataFrame(fits, columns=ROI_FIT_COLUMNS)


def trace_single_roi(
        fit_func_spec: FunctionalForm, 
        roi_img: np.ndarray, 
        mask: Optional[np.ndarray] = None, 
        background: Optional[np.ndarray] = None,
        ) -> np.ndarray:
    """
    Fit a single roi with 3D gaussian (MLE or LS as defined in config).
    
    Masking by intensity or label image can be used to improve fitting correct spot (set in config).

    Parameters
    ----------
    fit_func : FunctionalForm
        The 3D functional form to fit, e.g. a 3D Gaussian -- bundle of function and dimensionality
    roi_img : np.ndarray
        The data to which the given functional form should be fit; namely, usually a 3D array of 
        signal intensity values, with each entry corresponding to a pixel
    mask : np.ndarray, optional
        Array of values which, after transformation, multiplies the ROI image, allegedly to perhaps 
        provide better tracing performance; if provided, the dimensions should match that of ROI image.
    background : np.ndarray, optional
        Data of values to subtract from main image, representing background fluorescence signal, optional; 
        if not provided, no background subtraction will be performed

    Returns
    -------
    np.ndarray
        Array-/vector-Like of values representing the optimised parameter values of the function to fit
    """
    if len(roi_img.shape) != fit_func_spec.dimensionality:
        raise ValueError(f"ROI image to trace isn't correct dimensionality ({fit_func_spec.dimensionality}); shape: {roi_img.shape}")
    if background:
        # TODO: check that dimension of background image matches that of main ROI image.s
        roi_img = roi_img - background
    if not np.any(roi_img) or any(d < 3 for d in roi_img.shape): # Check if empty or too small for fitting.
        return np.array([-1] * len(ROI_FIT_COLUMNS))
    if mask is None:
        center = 'max'
    else:
        roi_img_masked = roi_img * (mask / np.max(mask))**2
        center = list(np.unravel_index(np.argmax(roi_img_masked, axis=None), roi_img.shape))
    return fit_func_spec.function(roi_img, sigma=1, center=center)[0]


def apply_fine_scale_drift_correction(traces: pd.DataFrame) -> pd.DataFrame:
    """Shift pixel coordinates by the amount of fine-scale drift correction."""
    traces['z_px_dc'] = traces['z_px'] + traces['z_px_fine']
    traces['y_px_dc'] = traces['y_px'] + traces['y_px_fine']
    traces['x_px_dc'] = traces['x_px'] + traces['x_px_fine']
    return traces


def apply_pixels_to_nanometers(traces: pd.DataFrame, z_nm_per_px: float, xy_nm_per_px: float) -> pd.DataFrame:
    """Add columns for distance in nanometers, based on pixel-to-nanometer conversions."""
    traces['z'] = traces['z_px_dc'] * z_nm_per_px
    traces['y'] = traces['y_px_dc'] * xy_nm_per_px
    traces['x'] = traces['x_px_dc'] * xy_nm_per_px
    traces['sigma_z'] = traces['sigma_z'] * z_nm_per_px
    traces['sigma_xy'] = traces['sigma_xy'] * xy_nm_per_px
    return traces
