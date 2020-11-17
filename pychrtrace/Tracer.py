"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
import os
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import pychrtrace.image_processing_functions as ip
from pychrtrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
import dask
from dask import delayed


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

    def slice_for_roi(self, roi, drift_table_row):
        '''
        Calculate the correct slice object based on a given ROI and drift table
        '''

        #Find size of image, and desired ROI size:
        Z, Y, X = self.images_shape[-3:]
        roi_size = self.config['roi_image_size']
        #Read values from drift table
        z_drift_course = int(drift_table_row['z_px_course'])
        y_drift_course = int(drift_table_row['y_px_course'])
        x_drift_course = int(drift_table_row['x_px_course'])
        
        #Course drift correct of the ROI: 
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

        #If drift correction has failed completely this checks if empty ROIs are generated:
        good = True
        if any([a == b for (a,b) in (sz, sy, sx)]):
            good = False

        #Create slice object after above corrections:
        s = (slice(sz[0],sz[1]), 
             slice(sy[0],sy[1]), 
             slice(sx[0],sx[1]))

        return s, pad, good

    def tracing_3d(self):
        '''
        Fits 3D gaussian to previously detected ROIs in all timeframes.
       
        Returns
        -------
        res : Pandas dataframe containing trace data
        imgs : Hyperstack image with raw image data of each ROI.
    
        '''
        #Extract parameters from config and predefined roi table.
        num_frames = self.images.shape[1]
        trace_ch = self.config['trace_ch']
        decon = self.config['deconvolve']
        if decon != 0:
            algo, kernel, fd_data = ip.decon_RL_setup(res_lateral=self.config['xy_nm']/1000, res_axial=self.config['z_nm']/1000)
        roi_image_size = self.config['roi_image_size']
        roi_table = self.roi_table[self.roi_table['position'].isin(self.pos_list)]

        #Setup loop over each timepoint.
        trace_index = []
        trace_res = []
        all_images = []
        for frame in range(num_frames):
            print(f'Tracing frame {frame}.')
            
            #Setup loop over each ROI.
            frame_result = []
            frame_index = []
            roi_images = []
            for index, roi in roi_table.iterrows():
                #Select matching row from drift table.
                drift_table_row = self.drift_table.loc[(self.drift_table['frame'] == frame) & 
                                                       (self.drift_table['pos_id'] == roi['position'])]
                #Extract position and ROI coordinates, and extract roi image from 
                #slicable image object (typically a dask array)
                pos_index = self.pos_list.index(roi['position'])
                roi_slice, pad, good = self.slice_for_roi(roi, drift_table_row)
                roi_image = np.array(self.images[pos_index, 
                                                frame, 
                                                trace_ch,
                                                roi_slice[0], 
                                                roi_slice[1],
                                                roi_slice[2]])

                #If microscope drifted, ROI could be outside image. Correct for this:
                if not good:
                    print('Bad image.')
                    roi_image=np.zeros((16,16,16), dtype=np.float32)
                elif pad != ((0,0),(0,0),(0,0)):
                    print('Padding ', pad)
                    try:
                        roi_image = np.pad(roi_image, pad, mode='edge')
                    except ValueError:
                        roi_image = np.zeros((16,16,16), dtype=np.float32)
                
                if decon != 0:
                    roi_image =ip.decon_RL(roi_image, kernel, algo, fd_data, niter=decon)
                #Perform 3D gaussian fit
                frame_result.append(delayed(self.fit_func)(roi_image, sigma=1, center='max')[0])
                #Expand the image to a standard size for hyperstack.
                roi_image_exp = delayed(ip.pad_to_shape)(roi_image, roi_image_size)
                #Extract fine drift from drift table and shift image for display.
                dz = float(drift_table_row['z_px_fine'])
                dy = float(drift_table_row['y_px_fine'])
                dx = float(drift_table_row['x_px_fine'])
                roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
                roi_images.append(roi_image_shifted)
                #Add some parameters for tracing table
                frame_index.append([roi.name, frame, roi['position'], roi['roi_id'], dz, dy, dx])

            #Add all the results per timepoint, compute on delayed dask objects.
            trace_res.append(dask.compute(*frame_result))
            trace_index.append(frame_index)
            all_images.append(np.stack(dask.compute(*roi_images)))
        
        #Cleanup of results into dataframe format
        trace_res = pd.DataFrame([i for t in trace_res for i in t], columns=["BG", 
                                                                    "A", 
                                                                    "z_px",
                                                                    "y_px",
                                                                    "x_px",
                                                                    "sigma_z",
                                                                    "sigma_xy"
                                                                    ])
        trace_index = pd.DataFrame([i for t in trace_index for i in t], columns=[
                                                                    "trace_ID",
                                                                    "frame",
                                                                    "position",
                                                                    "roi_ID",
                                                                    "drift_z",
                                                                    "drift_y",
                                                                    "drift_x"])
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
        traces = traces.sort_values(['trace_ID', 'frame'])
        traces = traces.set_index(['trace_ID'])
        #Make final hyperstack of images, will typically be in TPZYX order.
        all_images = np.stack(all_images)
        
        self.image_handler.save_data(traces=traces, imgs=all_images)
        return traces, all_images
        
