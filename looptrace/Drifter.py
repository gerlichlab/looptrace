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
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops_table
from scipy import ndimage as ndi
from scipy.stats import trim_mean
from joblib import Parallel, delayed

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.config = image_handler.config
        self.dc_file_path = image_handler.dc_file_path
        self.images, self.pos_list = image_handler.images, image_handler.pos_list

    def generate_bead_rois(self, t_img, threshold, min_bead_int, n_points):
        t_img_label,num_labels=ndi.label(t_img>threshold)
        print('Number of unfiltered beads found: ', num_labels)
        t_img_maxima = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
        try:
            t_img_maxima = t_img_maxima.query('max_intensity > @min_bead_int').sample(n=n_points, random_state=1)[['centroid-0', 'centroid-1', 'centroid-2']].to_numpy()
        except ValueError: #Not enough beads found, make up bead to continue (usually imaging error resulting in no beads).
            t_img_maxima = np.array([10,100,100])
        t_img_maxima = np.round(t_img_maxima).astype(int)
        return t_img_maxima

    def extract_single_bead(self, point, img, course_drift=None):
        #Calculate fine scale drift for all selected fiducials.
        if course_drift is not None:
            s = tuple([slice(ind-int(shift)-8, ind-int(shift)+8) for (ind, shift) in zip(point, course_drift)])
        else:
            s = tuple([slice(ind-8, ind+8) for ind in point])
        bead = img[s]
        return bead

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
        print(f'Fine drift: {fine_drift} with untrimmed STD of {np.std(shifts, axis=0)} .')  

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

        #Run drift correction for each position and save results in table.
        all_drifts=[]
        import datetime
        for i, pos in enumerate(pos_list):
            print(f'Running drift correction for position {pos}.')
            drifts_course = []
            drifts_fine = []
            t_img = np.array(images[i, t_slice, ch])
            bead_rois = self.generate_bead_rois(t_img, threshold, min_bead_int, n_points)
            t_bead_imgs =  Parallel(n_jobs=-1, prefer='threads')(delayed(self.extract_single_bead)(point, t_img) for point in bead_rois)
            for t in t_all:
                print('Drift correcting frame', t)
                o_img = np.array(images[i, t, ch])
                drift_course = ip.drift_corr_course(t_img, o_img, downsample=2)
                drifts_course.append(drift_course)
                o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(self.extract_single_bead)(point, o_img, drift_course) for point in bead_rois)
                drift_fine = Parallel(n_jobs=-1, prefer='threads')(delayed(self.correlate_single_bead)(t_bead, o_bead, 100) 
                                                                    for t_bead, o_bead in zip(t_bead_imgs, o_bead_imgs))
                drift_fine = np.array(drift_fine)
                drift_fine = trim_mean(drift_fine, proportiontocut=0.2, axis=0)
                
                print(f'Fine drift: {drift_fine}.')  
                drifts_fine.append(drift_fine) 

            print('Drift correction complete in position.')
            drifts = pd.concat([pd.DataFrame(drifts_course), pd.DataFrame(drifts_fine)], axis = 1)
            drifts['position'] = pos
            drifts.index.name = 'frame'
            all_drifts.append(drifts)
            print('Finished drift correction for position ', pos)
        
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