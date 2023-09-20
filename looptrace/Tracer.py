# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import dataclasses

from typing import *

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from tqdm import tqdm

import looptrace.image_processing_functions as ip
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE


ROI_FIT_COLUMNS = ["BG", "A", "z_px", "y_px", "x_px", "sigma_z", "sigma_xy"]


@dataclasses.dataclass
class BackgroundSpecification:
    frame_index: int
    drifts: Iterable[np.ndarray]


class Tracer:

    def __init__(self, image_handler, trace_beads=False, array_id=None):
        '''
        Initialize Tracer class with config read in from YAML file.
        '''
        self.image_handler = image_handler
        self.config_path = image_handler.config_path
        self.config = image_handler.config
        self.drift_table = image_handler.tables[image_handler.spot_input_name + '_drift_correction']
        self.images = self.image_handler.images[self.config['trace_input_name']]
        self.pos_list = self.image_handler.image_lists[image_handler.spot_input_name]
        if trace_beads:
            self.roi_table = image_handler.tables[image_handler.spot_input_name + '_bead_rois']
            finalise_suffix = lambda p: p.replace(".csv", "_beads.csv")
        else:
            self.roi_table = image_handler.tables[image_handler.spot_input_name + '_rois']
            finalise_suffix = lambda p: p
        self.all_rois = image_handler.tables[image_handler.spot_input_name + '_dc_rois']

        self.fit_funcs = {'LS': fitSymmetricGaussian3D, 'MLE': fitSymmetricGaussian3DMLE}
        self.fit_func = self.fit_funcs[self.config['fit_func']]

        self.array_id = array_id
        if self.array_id is not None:
            self.pos_list = [self.pos_list[int(self.array_id)]]
            self.roi_table = self.roi_table[self.roi_table.position.isin(self.pos_list)]
            self.all_rois = self.all_rois[self.all_rois.position.isin(self.pos_list)].reset_index(drop=True)
            traces_path = self.image_handler.out_path('traces.csv'[:-4] + '_' + str(self.array_id).zfill(4) + '.csv')
            self.images = self.images[self.roi_table.index.to_list()]
        else:
            traces_path = self.image_handler.out_path('traces.csv')
        
        self.traces_path = finalise_suffix(traces_path)

    def trace_all_rois(self) -> str:
        '''
        Fits 3D gaussian to previously detected ROIs across positions and timeframes.
    
        '''
        #fits = Parallel(n_jobs=-1, prefer='threads')(delayed(self.trace_single_roi)(roi_imgs[i]) for i in tqdm(range(roi_imgs.shape[0])))
        
        try:
            #This only works for a single position at the time currently
            bg_frame_idx = self.image_handler.config['subtract_background']
        except KeyError:
            bg_spec = None
        else:
            pos_drifts = self.drift_table[self.drift_table.position.isin(self.pos_list)][['z_px_fine', 'y_px_fine', 'x_px_fine']].to_numpy()
            bg_spec = BackgroundSpecification(frame_index=bg_frame_idx, drifts=pos_drifts - pos_drifts[bg_frame_idx])

        trace_res = find_trace_fits(
            fit_func=self.fit_func,
            images=self.images, 
            ref_frames=self.roi_table['frame'].to_list(), 
            mask_fits=self.image_handler.config.get('mask_fits', False), 
            background_specification=bg_spec
            )

        #trace_index = pd.DataFrame(fit_rois, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
        traces = pd.concat([self.all_rois, trace_res], axis=1)
        traces.rename(columns={"roi_id": "trace_id"}, inplace=True)

        #Apply fine scale drift to fits, and physcial units.
        traces = apply_fine_scale_drift_correction(traces)
        #traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces = apply_pixels_to_nanometers(traces, z_nm_per_px=self.config['z_nm'], xy_nm_per_px=self.config['xy_nm'])
        
        traces = traces.sort_values(['trace_id', 'frame'])
        print(f"Writing traces: {self.traces_path}")
        traces.to_csv(self.traces_path)

        return self.traces_path
    

def find_trace_fits(fit_func, images: Iterable[np.ndarray], ref_frames: List[int], mask_fits: bool, background_specification: Optional[BackgroundSpecification]) -> pd.DataFrame:
    fits = []
    if background_specification is None:
        def finalise_spot_img(img, _):
            return img
    else:
        def finalise_spot_img(img, fov_imgs):
            return img.astype(np.int16) - fov_imgs[background_specification.frame_index].astype(np.int16)
    if mask_fits:
        for p, pos_imgs in tqdm(enumerate(images), total=len(images)):
            ref_img = pos_imgs[ref_frames[p]]
            #print(ref_img.shape)
            for t, spot_img in enumerate(pos_imgs):
                #if background_specification is not None:
                    #shift = ndi.shift(pos_imgs[background_specification.frame_index], shift=background_specification.drifts[t])
                    #spot_img = np.clip(spot_img.astype(np.int16) - shift, a_min = 0, a_max = None)
                spot_img = finalise_spot_img(spot_img, pos_imgs)
                fits.append(trace_single_roi(fit_func=fit_func, roi_img=spot_img, mask=ref_img))
            #Parallel(n_jobs=1, prefer='threads')(delayed(self.trace_single_roi)(imgs[p, t], mask= ref_img) for t in range(imgs.shape[1]))
    else:
        for pos_imgs in tqdm(images, total=len(images)):
            for spot_img in pos_imgs:
                spot_img = finalise_spot_img(spot_img, pos_imgs)
                fits.append(trace_single_roi(fit_func=fit_func, roi_img=spot_img))
    return pd.DataFrame(fits, columns=ROI_FIT_COLUMNS)


def trace_single_roi(fit_func, roi_img, mask=None, background=None):
    #Fit a single roi with 3D gaussian (MLE or LS as defined in config).
    #Masking by intensity or label image can be used to improve fitting correct spot (set in config)
    if background is not None:
        roi_img = roi_img - background
    if np.any(roi_img) and np.all([d > 2 for d in roi_img.shape]): #Check if empty or too small for fitting
        if mask is None:
            center = 'max'
        else:
            roi_img_masked = roi_img * (mask / np.max(mask))**2
            center = list(np.unravel_index(np.argmax(roi_img_masked, axis=None), roi_img.shape))
        return fit_func(roi_img, sigma=1, center=center)[0]
    else:
        return np.array([-1] * len(ROI_FIT_COLUMNS))


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
