# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

#from distutils.command.config import config
#from tkinter import image_names

from dask.delayed import delayed
from joblib.parallel import Parallel
import numpy as np
import pandas as pd
from looptrace import image_processing_functions as ip
from looptrace import image_io
from looptrace.gaussfit import fitSymmetricGaussian3D
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops_table
from scipy import ndimage as ndi
from scipy.stats import trim_mean
from joblib import Parallel, delayed
import os
import tqdm
import dask.array as da

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        Reads sequential images. NB! These need to be named 'seq_images'!
        '''
        self.image_handler = image_handler
        self.config = self.image_handler.config
        self.dc_file_path = self.image_handler.dc_file_path
        self.images = self.image_handler.images['seq_images']
        self.pos_list = self.image_handler.image_lists['seq_images']
        
        try:
            self.bead_roi_px = self.config['bead_roi_size']
        except KeyError: #Legacy config
            self.bead_roi_px = 20

    def generate_bead_rois(self, t_img, threshold, min_bead_int, n_points):
        '''Function for finding positions of beads in an image based on manually set thresholds in config file.

        Args:
            t_img (3D ndarray): Image
            threshold (float): Threshold for initial bead segmentation
            min_bead_int (float): Secondary filtering of segmented maxima.
            n_points (int): How many bead positions to return

        Returns:
            t_img_maxima: 3XN ndarray of 3D bead coordinates in t_img.
        '''
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

        return t_img_maxima

    def extract_single_bead(self, point, img, course_drift=None):
        #Exctract a cropped region of a single fiducial in an image, optionally including a pre-calucalated course drift to shift the cropped region.
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

    def drift_corr(self):
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images/arrays.
        Settings set in config file.

        '''

        images = self.images
        pos_list = self.pos_list
        t_slice = self.config['bead_reference_frame']
        t_all = range(images[0].shape[0])
        ch = self.config['bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']
        #dc_bead_img_path = self.config['output_path']+os.sep+'dc_bead_images'
        #roi_px = self.bead_roi_px
        ds = self.config['course_drift_downsample']

        try:
            dc_method = self.config['dc_method']
        except KeyError:
            dc_method = 'cc'

        #try:
        #    save_dc_beads = self.config['save_dc_beads']
        #except KeyError:
        #    save_dc_beads = False

        if dc_method == 'course':
            all_drifts=[]
            #out_imgs = []
            for i, pos in enumerate(pos_list):
                #pos_imgs = []
                print(f'Running only course drift correction for position {pos}.')
                drifts_course = []
                drifts_fine = []

                t_img = np.array(images[i][t_slice, ch, ::ds, ::ds, ::ds])

                for t in tqdm.tqdm(t_all):
                    o_img = np.array(images[i][t, ch, ::ds, ::ds, ::ds])
                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)
                    drifts_course.append(drift_course)
                    drifts_fine.append([0,0,0])

                drifts = pd.concat([pd.DataFrame(drifts_course), pd.DataFrame(drifts_fine)], axis = 1)
                drifts['position'] = pos
                drifts.index.name = 'frame'
                all_drifts.append(drifts)
                print('Finished drift correction for position ', pos)
                #print('Drifts:', drifts)
        else:
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
            #out_imgs = []
            for i, pos in enumerate(pos_list):
                #pos_imgs = []
                print(f'Running drift correction for position {pos}.')
                drifts_course = []
                drifts_fine = []

                t_img = np.array(images[i][t_slice, ch])

                bead_rois = self.generate_bead_rois(t_img, threshold, min_bead_int, n_points)

                t_bead_imgs =  Parallel(n_jobs=-1, prefer='threads')(delayed(self.extract_single_bead)(point, t_img) for point in bead_rois)

                if decon_params[0] > 0:
                    t_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.decon_RL)(t_img, decon_params[2], decon_params[1], decon_params[3], niter=decon_params[0]) for t_img in t_bead_imgs)

                for t in tqdm.tqdm(t_all):
                    o_img = np.array(images[i][t, ch])

                    drift_course = ip.drift_corr_course(t_img, o_img, downsample=ds)

                    drifts_course.append(drift_course)
                    o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(self.extract_single_bead)(point, o_img, drift_course) for point in bead_rois)

                    if decon_params[0] > 0:
                        o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.decon_RL)(o_img, decon_params[2], decon_params[1], decon_params[3], niter=decon_params[0]) for o_img in o_bead_imgs)

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

                    #if save_dc_beads:
                    #    if not os.path.isdir(dc_bead_img_path):
                    #        os.mkdir(dc_bead_img_path)
                    #    t_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in t_bead_imgs])
                    #    o_bead_imgs = np.stack([ip.pad_to_shape(img, (roi_px, roi_px, roi_px)) for img in o_bead_imgs])
                        #out_imgs = np.stack([t_bead_imgs, o_bead_imgs])

                        #pos_imgs(dc_bead_img_path+os.sep+pos+'_T'+str(t).zfill(4)+'.npy', out_imgs)


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
        self.image_handler.drift_table = all_drifts

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

        print('DC images generated.')

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

        imgs = []
        for pos in tqdm.tqdm(self.pos_list):
            n_t = self.images[0].shape[0]
            pos_img = []
            pos_index = self.pos_list.index(pos)
            for t in tqdm.tqdm(range(n_t)):
                shift = self.drift_table.query('position == @pos').iloc[t][['y_px_course', 'x_px_course', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                proj_img = da.max(self.images[pos_index][t], axis=1).compute()
                proj_img = ndi.shift(proj_img, shift=(0,)+shift, order = 2)
                pos_img.append(proj_img)
            pos_img = np.stack(pos_img)
            pos_img = pos_img[:,:,np.newaxis, :, :]
            imgs.append(pos_img)
        
        imgs = np.stack(imgs)
        self.image_handler.images['max_proj_dc'] = imgs
        image_io.images_to_ome_zarr(images=imgs, path=self.config['image_path']+os.sep+'max_proj_dc', name='max_proj_dc', axes=('p','t','c','z','y','x'))
        
        print('DC images generated.')