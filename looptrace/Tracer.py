# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import looptrace.image_processing_functions as ip
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from tqdm import tqdm

ROI_FIT_COLUMNS = ["BG", "A", "z_px", "y_px", "x_px", "sigma_z", "sigma_xy"]


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

    def trace_single_roi(self, roi_img, mask = None, background = None):
        #Fit a single roi with 3D gaussian (MLE or LS as defined in config).
        #Masking by intensity or label image can be used to improve fitting correct spot (set in config)
        if background is not None:
            roi_img = roi_img - background
        if np.any(roi_img) and np.all([d > 2 for d in roi_img.shape]): #Check if empty or too small for fitting
            if mask is None:
                center = 'max'
            else:
                roi_img_masked = (mask/np.max(mask)) * roi_img
                center = list(np.unravel_index(np.argmax(roi_img_masked, axis=None), roi_img.shape))
            return self.fit_func(roi_img, sigma=1, center=center)[0]
        else:
            return np.array([-1] * len(ROI_FIT_COLUMNS))
    
    def trace_all_rois(self) -> str:
        '''
        Fits 3D gaussian to previously detected ROIs across positions and timeframes.
    
        '''
        imgs = self.images

        fits = []

        #fits = Parallel(n_jobs=-1, prefer='threads')(delayed(self.trace_single_roi)(roi_imgs[i]) for i in tqdm(range(roi_imgs.shape[0])))
        mask_fits = self.image_handler.config.get('mask_fits', False)
        
        try:
            #This only works for a single position at the time currently
            background = self.image_handler.config['substract_background']
            pos_drifts = self.drift_table[self.drift_table.position.isin(self.pos_list)][['z_px_fine', 'y_px_fine', 'x_px_fine']].to_numpy()
            background_rel_drifts = pos_drifts - pos_drifts[background]
        except KeyError:
            background = None

        if mask_fits:
            ref_frames = self.roi_table['frame'].to_list()
            for p, pos_imgs in tqdm(enumerate(imgs), total=len(imgs)):
                ref_img = pos_imgs[ref_frames[p]]
                #print(ref_img.shape)
                for t, spot_img in enumerate(pos_imgs):
                    if background is not None:
                        spot_img = np.clip(spot_img.astype(np.int16) - ndi.shift(pos_imgs[background], shift = background_rel_drifts[t]), a_min = 0, a_max = None)
                    fits.append(self.trace_single_roi(spot_img, mask = ref_img))
                #Parallel(n_jobs=1, prefer='threads')(delayed(self.trace_single_roi)(imgs[p, t], mask= ref_img) for t in range(imgs.shape[1]))

        else:
            for pos_imgs in tqdm(imgs, total=len(imgs)):
                for spot_img in pos_imgs:
                    fits.append(self.trace_single_roi(spot_img))

        trace_res = pd.DataFrame(fits,columns=ROI_FIT_COLUMNS)
        #trace_index = pd.DataFrame(fit_rois, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
        traces = pd.concat([self.all_rois, trace_res], axis=1)
        traces.rename(columns={"roi_id": "trace_id"}, inplace=True)

        #Apply fine scale drift to fits, and physcial units.
        traces['z_px_dc'] = traces['z_px'] + traces['z_px_fine']
        traces['y_px_dc'] = traces['y_px'] + traces['y_px_fine']
        traces['x_px_dc'] = traces['x_px'] + traces['x_px_fine']
        #traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z'] = traces['z_px_dc'] * self.config['z_nm']
        traces['y'] = traces['y_px_dc'] * self.config['xy_nm']
        traces['x'] = traces['x_px_dc'] * self.config['xy_nm']
        traces['sigma_z'] = traces['sigma_z'] * self.config['z_nm']
        traces['sigma_xy'] = traces['sigma_xy'] * self.config['xy_nm']
        traces = traces.sort_values(['trace_id', 'frame'])
        
        print(f"Writing traces: {self.traces_path}")
        traces.to_csv(self.traces_path)

        return self.traces_path