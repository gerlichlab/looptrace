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
#import dask
#from dask import delayed
from tqdm import tqdm

class Tracer:
    def __init__(self, image_handler):
        '''
        Initialize Tracer class with config read in from YAML file.
    '''
        self.image_handler = image_handler
        self.config_path = image_handler.config_path
        self.config = image_handler.config
        self.images, self.pos_list = image_handler.images, image_handler.pos_list
        self.images_shape = self.images.shape
        self.drift_table = image_handler.drift_table
        self.roi_table = image_handler.roi_table

        self.fit_funcs = {'LS': fitSymmetricGaussian3D, 'MLE': fitSymmetricGaussian3DMLE}
        self.fit_func = self.fit_funcs[self.config['fit_func']]

    def make_dc_rois_all_frames(self):
        print('Generating list of all ROIs for tracing.')
        Z, Y, X = self.images_shape[-3:]
        roi_size = self.config['roi_image_size']

        all_rois = []
        for pos in tqdm(self.pos_list):
            pos_index = self.pos_list.index(pos)
            sel_rois = self.roi_table.query('position == @pos')
            sel_dc = self.drift_table.query('position == @pos')
            for i, roi in sel_rois.iterrows():
                ref_frame = roi['frame']
                ref_offset = sel_dc.query('frame == @ref_frame')
                for j, dc_frame in sel_dc.iterrows():
                    z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                    y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                    x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])
                    zc = int(roi['zc'])-z_drift_course
                    yc = int(roi['yc'])-y_drift_course
                    xc = int(roi['xc'])-x_drift_course
                        
                    z_min = zc - roi_size[0]//2
                    z_max = zc + roi_size[0]//2
                    y_min = yc - roi_size[1]//2
                    y_max = yc + roi_size[1]//2
                    x_min = xc - roi_size[2]//2
                    x_max = xc + roi_size[2]//2

                    #Handling case of ROI extending beyond image edge after drift correction:
                    pad = ((abs(min(0,z_min)),abs(max(0,z_max-Z))),
                            (abs(min(0,y_min)),abs(max(0,y_max-Y))),
                            (abs(min(0,x_min)),abs(max(0,x_max-X))))

                    sz = (max(0,z_min),min(Z,z_max))
                    sy = (max(0,y_min),min(Y,y_max))
                    sx = (max(0,x_min),min(X,x_max))

                    #Create slice object after above corrections:
                    s = (slice(sz[0],sz[1]), 
                        slice(sy[0],sy[1]), 
                        slice(sx[0],sx[1]))

                    all_rois.append([pos, pos_index, roi.name, dc_frame['frame'], ref_frame, s, pad, z_drift_course, y_drift_course, x_drift_course, 
                                                                                          dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])
        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_id', 'frame', 'ref_frame', 'roi_slice', 'pad', 'z_px_course', 'y_px_course', 'x_px_course', 
                                                                                                  'z_px_fine', 'y_px_fine', 'x_px_fine'])


    def tracing_3d(self):
        '''
        Fits 3D gaussian to previously detected ROIs in all timeframes.
       
        Returns
        -------
        res : Pandas dataframe containing trace data
        imgs : Hyperstack image with raw image data of each ROI.
    
        '''
        #Extract parameters from config and predefined roi table.

        trace_ch = self.config['trace_ch']
        decon = self.config['deconvolve']
        roi_image_size = tuple(self.config['roi_image_size'])

        if decon != 0:
            algo, kernel, fd_data = ip.decon_RL_setup(  res_lateral=self.config['xy_nm']/1000, 
                                                        res_axial=self.config['z_nm']/1000, 
                                                        wavelength=self.config['spot_wavelength']/1000,
                                                        na=self.config['objective_na'])
        #roi_image_size = self.config['roi_image_size']
        #roi_table = self.roi_table[self.roi_table['position'].isin(self.pos_list)]

        #Setup loop over each timepoint.
        trace_index = []
        trace_res = []
        all_images = []
        for i, roi in tqdm(self.all_rois.iterrows(), total=self.all_rois.shape[0]):

            #Extract position and ROI coordinates, and extract roi image from 
            #slicable image object (typically a dask array)
            
            roi_slice = roi['roi_slice']
            pad = roi['pad']

            try:
                roi_image = np.array(self.images[roi['pos_index'],
                                                roi['frame'], 
                                                trace_ch,
                                                roi_slice[0], 
                                                roi_slice[1],
                                                roi_slice[2]])

                #If microscope drifted, ROI could be outside image. Correct for this:
                if pad != ((0,0),(0,0),(0,0)):
                    #print('Padding ', pad)
                    roi_image = np.pad(roi_image, pad, mode='edge')

            except ValueError: # ROI collection failed for some reason
                roi_image = np.zeros(roi_image_size, dtype=np.float32)
            
            if decon != 0:
                roi_image = ip.decon_RL(roi_image, kernel, algo, fd_data, niter=decon)

            #Perform 3D gaussian fit
            #trace_res.append(delayed(self.fit_func)(roi_image, sigma=1, center='max')[0])
            trace_res.append(self.fit_func(roi_image, sigma=1, center='max')[0])
            #Ensure images are compatible size for hyperstack.
            
            #roi_image_exp = delayed(ip.pad_to_shape)(roi_image, roi_image_size)
            if roi_image.shape != roi_image_size:
                print(roi_image.shape)
                roi_image = ip.pad_to_shape(roi_image, roi_image_size)

            #Extract fine drift from drift table and shift image for display.
            dz = float(roi['z_px_fine'])
            dy = float(roi['y_px_fine'])
            dx = float(roi['x_px_fine'])
            #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
            roi_image = ndi.shift(roi_image, (dz, dy, dx))
            all_images.append(roi_image)
            
            #Add some parameters for tracing table
            trace_index.append([roi['roi_id'], roi['frame'], roi['ref_frame'], roi['position'], dz, dy, dx])

            #Add all the results per timepoint, compute on delayed dask objects.
        #trace_res = dask.compute(*trace_res)
        #all_images = np.stack(dask.compute(*all_images))
        all_images = np.stack(all_images)
        #Cleanup of results into dataframe format
        trace_res = pd.DataFrame(trace_res,columns=["BG","A","z_px","y_px","x_px","sigma_z","sigma_xy"])
        trace_index = pd.DataFrame(trace_index, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
        traces = pd.concat([trace_index, trace_res], axis=1)

        #Apply fine scale drift to fits, and physcial units.
        traces['z_px']=traces['z_px']+traces['drift_z']
        traces['y_px']=traces['y_px']+traces['drift_y']
        traces['x_px']=traces['x_px']+traces['drift_x']
        traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_px']*self.config['z_nm']
        traces['y']=traces['y_px']*self.config['xy_nm']
        traces['x']=traces['x_px']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces['sigma_xy']=traces['sigma_xy']*self.config['xy_nm']
        traces = traces.sort_values(['trace_id', 'frame'])
        #Make final hyperstack of images in PTZYX order.
        all_images = np.reshape(np.stack(all_images), 
                                (max(traces.trace_id)+1,max(traces.frame)+1, 
                                roi_image_size[0], roi_image_size[1], roi_image_size[2]))
        
        self.image_handler.save_data(traces=traces, imgs=all_images)
        return traces, all_images
        
