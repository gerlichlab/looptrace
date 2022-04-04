# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from datetime import date
import threading
from dask.delayed import delayed
from joblib.parallel import Parallel
import numpy as np
import pandas as pd
from looptrace import image_processing_functions as ip
from looptrace.gaussfit import fitSymmetricGaussian3D, symmetricGaussian3D
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops_table
from scipy import ndimage as ndi
from scipy.stats import trim_mean
from joblib import Parallel, delayed
import tifffile
import os
import tqdm

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.config = image_handler.config
        self.dc_file_path = image_handler.dc_file_path
        self.images, self.pos_list = image_handler.images, image_handler.pos_list
        try:
            self.bead_roi_px = self.config['bead_roi_size']
        except KeyError: #Legacy config
            self.bead_roi_px = 20

    def generate_bead_rois(self, t_img, threshold, min_bead_int, n_points):
        roi_px = self.bead_roi_px//2
        t_img_label,num_labels=ndi.label(t_img>threshold)
        print('Number of unfiltered beads found: ', num_labels)
        t_img_maxima = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
        #try:
        t_img_maxima = t_img_maxima[(t_img_maxima['centroid-0'] > roi_px) & (t_img_maxima['centroid-1'] > roi_px) & (t_img_maxima['centroid-2'] > roi_px)].query('max_intensity > @min_bead_int')
        if len(t_img_maxima) > n_points:
            t_img_maxima = t_img_maxima.sample(n=n_points, random_state=1)[['centroid-0', 'centroid-1', 'centroid-2']].to_numpy()
        else:
            t_img_maxima = t_img_maxima.sample(n=len(t_img_maxima), random_state=1)[['centroid-0', 'centroid-1', 'centroid-2']].to_numpy()
        #except ValueError: #Not enough beads found, make up bead to continue (usually imaging error resulting in no beads).
        #    t_img_maxima = np.array([10,100,100])
        t_img_maxima = np.round(t_img_maxima).astype(int)
        #print(t_img_maxima)
        return t_img_maxima

    def extract_single_bead(self, point, img, course_drift=None):
        #Calculate fine scale drift for all selected fiducials.
        roi_px = self.bead_roi_px//2
        if course_drift is not None:
            s = tuple([slice(ind-int(shift)-roi_px, ind-int(shift)+roi_px) for (ind, shift) in zip(point, course_drift)])
        else:
            s = tuple([slice(ind-roi_px, ind+roi_px) for ind in point])
        bead = img[s]

        if bead.shape != (2*roi_px, 2*roi_px, 2*roi_px):
            return np.zeros((2*roi_px, 2*roi_px, 2*roi_px))
        else:
            return bead

    def fit_shift_single_bead(self, t_bead, o_bead):
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

    def correlate_multi_beads(self, points):


        shifts = Parallel(n_jobs=-1, prefer='threads')(delayed(self.correlate_single_bead)(t_img, o_img, upsampling) for point in points)
        shifts = np.array(shifts)

        fine_drift = trim_mean(shifts, proportiontocut=0.2, axis=0)
        #print(f'Fine drift: {fine_drift} with untrimmed STD of {np.std(shifts, axis=0)} .')  

    def drift_corr(self):
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images.
        '''

        images = self.images
        pos_list = self.pos_list
        t_slice = self.config['bead_reference_frame']
        t_all = range(images[0].shape[0])
        ch = self.config['bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']
        dc_bead_img_path = self.config['output_path']+os.sep+'dc_bead_images'
        roi_px = self.bead_roi_px
        ds = self.config['course_drift_downsample']
        try:
            dc_method = self.config['dc_method']
        except KeyError:
            dc_method = 'cc'

        try:
            save_dc_beads = self.config['save_dc_beads']
        except KeyError:
            save_dc_beads = False

        decon_params = []
        try:
            decon_params.append(self.config['deconvolve_dc'])
        except KeyError:
            decon_params.append(0)
        

        if decon_params[0] > 0:
            algo, kernel, fd_data = ip.decon_RL_setup(res_lateral=self.config['xy_nm']/1000, 
                                                        res_axial=self.config['z_nm']/1000, 
                                                        wavelength=self.config['dc_spot_wavelength']/1000,
                                                        na=self.config['objective_na'])
            decon_params.append(algo)
            decon_params.append(kernel)
            decon_params.append(fd_data)

        #Run drift correction for each position and save results in table.
        all_drifts=[]
        from datetime import datetime
        for i, pos in enumerate(pos_list):
            print(f'Running drift correction for position {pos}.')
            drifts_course = []
            drifts_fine = []
            #print('Starting', datetime.now().time())
            t_img = np.array(images[i, t_slice, ch])
            #print('Loading t_imgs', datetime.now().time())
            bead_rois = self.generate_bead_rois(t_img, threshold, min_bead_int, n_points)
            #print('Generate bead rois', datetime.now().time())
            t_bead_imgs =  Parallel(n_jobs=-1, prefer='threads')(delayed(self.extract_single_bead)(point, t_img) for point in bead_rois)
            #print('Extracted single beads', datetime.now().time())
            if decon_params[0] > 0:
                t_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.decon_RL)(t_img, decon_params[2], decon_params[1], decon_params[3], niter=decon_params[0]) for t_img in t_bead_imgs)
                #print('Deconvolved single t beads', datetime.now().time())
            for t in tqdm.tqdm(t_all):
                o_img = np.array(images[i, t, ch])
                #print('Loaded o_img', datetime.now().time())
                drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)
                #print('Did course dc', datetime.now().time())
                drifts_course.append(drift_course)
                o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(self.extract_single_bead)(point, o_img, drift_course) for point in bead_rois)
                #print('Got o_bead_imgs', datetime.now().time())
                if decon_params[0] > 0:
                    o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.decon_RL)(o_img, decon_params[2], decon_params[1], decon_params[3], niter=decon_params[0]) for o_img in o_bead_imgs)
                    #print('Deconvolved single o beads', datetime.now().time())

                if dc_method == 'cc':
                    drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.correlate_single_bead)(t_bead, o_bead, 100) 
                                                                        for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                elif dc_method == 'fit':
                    drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.fit_shift_single_bead)(t_bead, o_bead) 
                                                                    for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                else:
                    raise NotImplementedError('Unknown dc method.')       

                drift_fine = np.array(drift_fine)
                drift_fine = trim_mean(drift_fine, proportiontocut=0.2, axis=0)

                drifts_fine.append(drift_fine) 
                #print('Calculated fine drift', datetime.now().time())
                if save_dc_beads:
                    if not os.path.isdir(dc_bead_img_path):
                        os.mkdir(dc_bead_img_path)
                    t_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in t_bead_imgs])
                    o_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in o_bead_imgs])
                    out_imgs = np.stack([t_bead_imgs, o_bead_imgs])
                    #print('Made out images', datetime.now().time())
                    tifffile.imsave(dc_bead_img_path+os.sep+pos+'_T'+str(t).zfill(4)+'.tif', out_imgs)
                    #print('Saved out images', datetime.now().time())

            drifts = pd.concat([pd.DataFrame(drifts_course), pd.DataFrame(drifts_fine)], axis = 1)
            drifts['position'] = pos
            drifts.index.name = 'frame'
            all_drifts.append(drifts)
            print('Finished drift correction for position ', pos)
            print('Drifts:', drifts)
        
        all_drifts=pd.concat(all_drifts).reset_index()
        
        all_drifts.columns=['frame',
                            'z_px_course',
                            'y_px_course',
                            'x_px_course',
                            'z_px_fine',
                            'y_px_fine',
                            'x_px_fine',
                            'position']
        all_drifts.to_csv(self.dc_file_path)
        print('Drift correction complete.')
        return all_drifts