# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

#from distutils.command.config import config
#from tkinter import image_names

import logging
import os
from typing import *

import numpy as np
import pandas as pd
import dask.array as da
from dask.delayed import delayed
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.stats import trim_mean
from skimage.measure import regionprops_table
from skimage.registration import phase_cross_correlation
import tqdm

from looptrace import image_processing_functions as ip
from looptrace import image_io
from looptrace.gaussfit import fitSymmetricGaussian3D

logger = logging.getLogger()

class Drifter():

    def __init__(self, image_handler, array_id = None):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.image_handler = image_handler
        self.config = self.image_handler.config
        self.images_template = self.image_handler.images[self.image_handler.reg_input_template]
        self.images_moving = self.image_handler.images[self.image_handler.reg_input_moving]
        self.full_pos_list = self.image_handler.image_lists[self.image_handler.reg_input_moving]
        self.pos_list = self.full_pos_list
        
        self.bead_roi_px = self.config.get('bead_roi_size', 15)
        
        if array_id is not None:
            self.dc_file_path = self.image_handler.out_path(self.image_handler.reg_input_moving + '_drift_correction.csv'[:-4]+'_' + str(array_id).zfill(4) + '.csv')
            self.pos_list = [self.pos_list[int(array_id)]]
        else:
            self.dc_file_path = self.image_handler.out_path(self.image_handler.reg_input_moving + '_drift_correction.csv')


    def fit_shift_single_bead(self, t_bead, o_bead):
        # Fit the center of two beads using 3D gaussian fit, and fit the shift between the centers.

        try:
            t_fit = np.array(fitSymmetricGaussian3D(t_bead, sigma=1, center=None)[0])
            o_fit = np.array(fitSymmetricGaussian3D(o_bead, sigma=1, center=None)[0])
            shift = t_fit[2:5] - o_fit[2:5]
        except (ValueError, AttributeError):
            shift = np.array([0,0,0])
        return shift

    def correlate_single_bead(self, t_bead, o_bead, upsampling):
        try:
            shift = phase_cross_correlation(t_bead, o_bead, upsample_factor=upsampling, return_error=False)
        except (ValueError, AttributeError):
            shift = np.array([0,0,0])
        return shift

    def drift_corr(self) -> Optional[str]:
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images/arrays.
        Settings set in config file.

        '''

        frame_t = self.config['reg_ref_frame']
        ch_t = self.config['reg_ch_template']
        ch_o = self.config['reg_ch_moving']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']
        #dc_bead_img_path = self.config['output_path']+os.sep+'dc_bead_images'
        roi_px = self.bead_roi_px
        ds = self.config['course_drift_downsample']

        dc_method = self.config.get('dc_method', 'cc')

        #try:
        #    save_dc_beads = self.config['save_dc_beads']
        #except KeyError:
        #    save_dc_beads = False

        if dc_method == 'course':
            all_drifts = []
            #out_imgs = []
            for pos in self.pos_list:
                logger.info(f'Running only course drift correction for position: {pos}.')
                i = self.full_pos_list.index(pos)
                #pos_imgs = []
                drifts_course = []
                drifts_fine = []

                t_img = np.array(self.images_template[i][frame_t, ch_t, ::ds, ::ds, ::ds])

                for t in tqdm.tqdm(range(self.images_moving[i].shape[0])):
                    o_img = np.array(self.images_moving[i][t, ch_o, ::ds, ::ds, ::ds])
                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)
                    drifts_course.append(drift_course)
                    drifts_fine.append([0,0,0])

                drifts = pd.concat([pd.DataFrame(drifts_course), pd.DataFrame(drifts_fine)], axis = 1)
                drifts['position'] = pos
                drifts.index.name = 'frame'
                all_drifts.append(drifts)
                logger.info(f'Finished drift correction for position: {pos}')
        else:
            #Run drift correction for each position and save results in table.
            all_drifts = []
            for pos in self.pos_list:
                i = self.full_pos_list.index(pos)
                logger.info(f'Running drift correction for position: {pos}')
                t_img = np.array(self.images_template[i][frame_t, ch_t])
                bead_rois = ip.generate_bead_rois(t_img, threshold, min_bead_int, roi_px, n_points)
                t_bead_imgs =  Parallel(n_jobs=-1, prefer='threads')(delayed(ip.extract_single_bead)(point, t_img) for point in bead_rois)
                method_lookup = {
                    'cc': (self.correlate_single_bead, lambda img_pair: img_pair + (100, )), 
                    'fit': (self.fit_shift_single_bead, lambda img_pair: img_pair)
                }
                try:
                    corr_func, get_args = method_lookup[dc_method]
                except KeyError:
                    raise NotImplementedError(f"Unknown drift correction method ({dc_method}); choose from: {', '.join(method_lookup.keys())}")
                for t in tqdm.tqdm(range(self.images_moving[i].shape[0])):
                    o_img = np.array(self.images_moving[i][t, ch_o])
                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)
                    o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.extract_single_bead)(point, o_img, drift_course=drift_course) for point in bead_rois)
                    if len(bead_rois) > 0:
                        logger.info("Computing fine drift")
                        drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(corr_func)(*get_args(img_pair)) for img_pair in zip(t_bead_imgs, o_bead_imgs))
                        #if dc_method == 'cc':
                        #    drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.correlate_single_bead)(t_bead, o_bead, 100) 
                        #                                                        for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                        #elif dc_method == 'fit':
                        #    drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.fit_shift_single_bead)(t_bead, o_bead) 
                        #                                                    for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                        #else:
                        #    raise NotImplementedError('Unknown dc method.')
                        drift_fine = np.array(drift_fine)
                        drift_fine = trim_mean(drift_fine, proportiontocut=0.2, axis=0)
                    else:
                        logger.info("No bead ROIs, setting fine drift to all-0s")
                        drift_fine = np.zeros_like(drift_course)

                    drifts = [t,pos] + list(drift_course) + list(drift_fine)
                    all_drifts.append(drifts) 

                    #if save_dc_beads:
                    #    if not os.path.isdir(dc_bead_img_path):
                    #        os.mkdir(dc_bead_img_path)
                    #    t_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in t_bead_imgs])
                    #    o_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in o_bead_imgs])
                        #out_imgs = np.stack([t_bead_imgs, o_bead_imgs])

                        #pos_imgs(dc_bead_img_path+os.sep+pos+'_T'+str(t).zfill(4)+'.npy', out_imgs)

                logger.info(f'Finished drift correction for position: {pos}')
                    
        
        all_drifts=pd.DataFrame(all_drifts, columns=['frame',
                                                    'position',
                                                    'z_px_course',
                                                    'y_px_course',
                                                    'x_px_course',
                                                    'z_px_fine',
                                                    'y_px_fine',
                                                    'x_px_fine',
                                                    ])
        
        outfile = self.dc_file_path
        all_drifts.to_csv(outfile)
        logger.info('Drift correction complete.')
        self.image_handler.drift_table = all_drifts
        return outfile

    def gen_dc_images(self, pos):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''
        n_t = self.images[0].shape[0]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
            pos_img.append(da.roll(self.images[pos_index][t], shift = shift, axis = (1,2,3)))
        self.dc_images = da.stack(pos_img)

        logger.info('DC images generated.')

    def save_proj_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)
        '''

        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.full_pos_list.index(pos)
            pos_img = self.images_moving[pos_index]
            proj_img = da.max(pos_img, axis=2)
            zarr_out_path = os.path.join(self.image_handler.image_save_path, self.image_handler.reg_input_moving + '_max_proj_dc')
            z = image_io.create_zarr_store(
                path=zarr_out_path,
                name = self.image_handler.reg_input_moving + '_max_proj_dc', 
                pos_name = pos,
                shape = proj_img.shape, 
                dtype = np.uint16,  
                chunks = (1,1,proj_img.shape[-2], proj_img.shape[-1]),
                )

            n_t = proj_img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.image_handler.reg_input_moving + '_drift_correction'].query('position == @pos').iloc[t][['y_px_course', 'x_px_course', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                z[t] = ndi.shift(proj_img[t].compute(), shift=(0,)+shift, order = 2)
        
        logger.info('DC images generated.')
    
    def save_course_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)
        '''

        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.image_handler.image_lists['seq_images'].index(pos)
            pos_img = self.images[pos_index]
            #proj_img = da.max(pos_img, axis=2)
            zarr_out_path = os.path.join(self.image_handler.image_save_path, self.image_handler.reg_input_moving + '_course_dc')
            z = image_io.create_zarr_store(path=zarr_out_path,
                                            name = self.image_handler.reg_input_moving + '_dc_images', 
                                            pos_name = pos,
                                            shape = pos_img.shape, 
                                            dtype = np.uint16,  
                                            chunks = (1,1,1,pos_img.shape[-2], pos_img.shape[-1]))

            n_t = pos_img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.image_handler.reg_input_moving + '_drift_correction'].query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']]
                shift = (shift[0], shift[1], shift[2])
                z[t] = ndi.shift(pos_img[t].compute(), shift=(0,)+shift, order = 0)
        
        logger.info('DC images generated.')