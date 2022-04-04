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
from joblib import Parallel, delayed
import os

class Tracer:
    def __init__(self, image_handler, trace_beads = False):
        '''
        Initialize Tracer class with config read in from YAML file.
    '''
        self.image_handler = image_handler
        self.config_path = image_handler.config_path
        self.config = image_handler.config
        self.drift_table = image_handler.drift_table
        self.trace_beads = trace_beads
        if trace_beads:
            self.roi_table = image_handler.bead_rois
        else:
            self.roi_table = image_handler.roi_table

        self.fit_funcs = {'LS': fitSymmetricGaussian3D, 'MLE': fitSymmetricGaussian3DMLE}
        self.fit_func = self.fit_funcs[self.config['fit_func']]

    def make_dc_rois_all_frames(self):
        print('Generating list of all ROIs for tracing:')
        Z, Y, X = self.image_handler.images.shape[-3:]
        positions = sorted(list(self.roi_table.position.unique()))

        all_rois = []
        for i, roi in tqdm(self.roi_table.iterrows()):
            pos = roi['position']
            pos_index = positions.index(pos)
            sel_dc = self.drift_table.query('position == @pos')
            ref_frame = roi['frame']
            ch = roi['ch']
            ref_offset = sel_dc.query('frame == @ref_frame')
            for j, dc_frame in sel_dc.iterrows():
                z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])

                z_min = roi['zmin'] - z_drift_course
                z_max = roi['zmax'] - z_drift_course
                y_min = roi['ymin'] - y_drift_course
                y_max = roi['ymax'] - y_drift_course
                x_min = roi['xmin'] - x_drift_course
                x_max = roi['xmax'] - x_drift_course

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

                all_rois.append([pos, pos_index, roi.name, dc_frame['frame'], ref_frame, ch, s, pad, z_drift_course, y_drift_course, x_drift_course, 
                                                                                        dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])

        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 'roi_slice', 'pad', 'z_px_course', 'y_px_course', 'x_px_course', 
                                                                                                  'z_px_fine', 'y_px_fine', 'x_px_fine'])

    def extract_single_roi_img(self, single_roi):

        roi_image_size = tuple(self.config['roi_image_size'])
        p = single_roi['pos_index']
        t = single_roi['frame']
        c = single_roi['ch']
        z, y, x = single_roi['roi_slice']
        pad = single_roi['pad']

        try:
            roi_img = np.array(self.image_handler.images[p, t, c, z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                #print('Padding ', pad)
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros(roi_image_size, dtype=np.float32)

        #print(p, t, c, z, y, x)
        return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def extract_single_roi_img_inmem(self, single_roi, images):

        roi_image_size = tuple(self.config['roi_image_size'])
        z, y, x = single_roi['roi_slice']
        pad = single_roi['pad']

        try:
            roi_img = np.array(images[z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                #print('Padding ', pad)
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros(roi_image_size, dtype=np.float32)

        #print(p, t, c, z, y, x)
        return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def gen_roi_imgs_inmem(self):
        #roi_imgs = Parallel(n_jobs=-1, verbose=0, prefer='threads')(delayed(self.extract_single_roi_img)(single_roi) for i, single_roi in tqdm(test.iterrows(), total=len(test)))#tqdm(self.all_rois.iterrows()))
        #roi_imgs = [self.extract_single_roi_img(single_roi) for i, single_roi in tqdm(test.iterrows())]#tqdm(self.all_rois.iterrows()))
        #roi_image_size = tuple(self.config['roi_image_size'])
        rois = self.all_rois#.iloc[0:500]

        P = max(rois['pos_index'])+1
        T = max(rois['frame'])+1
        roi_array = {}
        #roi_array_padded = []

        for pos in tqdm(range(P), total = P):
                for frame in tqdm(range(T), total = T):
                    rois_stack = rois.query('pos_index == @pos & frame == @frame')
                    #print(rois_stack)
                    roi_ch = int(rois['ch'].unique())
                    image_stack = np.array(self.image_handler.images[pos, frame, roi_ch])
                    #print(image_stack.shape)
                    for j, single_roi in rois_stack.iterrows():
                        id = single_roi['roi_id']
                        t = single_roi['frame']
                        roi = self.extract_single_roi_img_inmem(single_roi, image_stack).astype(np.uint16)
                        roi_array[id, t] = roi
                        #roi_array_padded.append(ip.pad_to_shape(roi, shape = roi_image_size, mode = 'minimum'))
        
        #print(roi_array.keys())
        
        pos_rois = np.empty(max(rois.roi_id)+1, object)
      
        for roi_id in rois.roi_id.unique():
            try:
                pos_rois[roi_id] = np.stack([roi_array[(roi_id, frame)] for frame in range(T)])
            except KeyError:
                break
        
        #self.temp_array = pos_rois
        #roi_array_padded = np.stack(roi_array_padded)

        print('ROIs generated.')
        self.image_handler.spot_images['spot_images'] = pos_rois
        #self.image_handler.spot_images['spot_images_padded'] = roi_array_padded
        np.save(self.image_handler.spot_images_path+os.sep+'spot_images.npy', pos_rois)
        #np.save(self.image_handler.spot_images_path+os.sep+'spot_images_padded.npy', roi_array_padded)
    
    def decon_roi_imgs(self):
        decon_iter = self.config['deconvolve']
        if decon_iter != 0:
            algo, kernel, fd_data = ip.decon_RL_setup(  res_lateral=self.config['xy_nm']/1000, 
                                                        res_axial=self.config['z_nm']/1000, 
                                                        wavelength=self.config['spot_wavelength']/1000,
                                                        na=self.config['objective_na'])
            
            if 'exp_psf' in self.image_handler.spot_images:
                kernel = self.image_handler.spot_images['exp_psf']
        
            imgs = self.image_handler.spot_images['spot_images']
            roi_array_decon = np.empty(len(imgs), object)
            for i, frame_stack in enumerate(imgs):
                roi_stack_decon = []
                for roi_stack in frame_stack:
                    if np.any(roi_stack):
                        if kernel.shape > roi_stack.shape:
                            new_shape = []
                            for ks, rs in zip(kernel.shape, roi_stack.shape):
                                if ks <= rs:
                                    new_shape.append(ks)
                                else:
                                    new_shape.append(rs-1 if rs%2 == 0 else rs)
                            kernel_crop = ip.crop_to_shape(kernel, new_shape)
                        else:
                            kernel_crop = kernel

                        roi_stack_decon.append(ip.decon_RL(roi_stack, kernel=kernel_crop, algo = algo, fd_data=fd_data, niter=decon_iter))
                        
                roi_array_decon[i] = np.stack(roi_stack_decon)

                        #print(p,t)
            #roi_array_decon = np.array(roi_array_decon, dtype='object')
            #ip.imgs_to_ome_zarr(images = roi_array, path=self.config['input_path']+os.sep+'spot_images', name = 'spot_imgs', axes=['p','t','z','y','x'])
            self.image_handler.spot_images['spot_images_decon'] = roi_array_decon
            np.save(self.image_handler.spot_images_path+os.sep+'spot_images_decon.npy', roi_array_decon)
            print('Images deconvolved')
        

    def fine_dc_single_roi_img(self, roi_img, roi):
        
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
        roi_img = ndi.shift(roi_img, (dz, dy, dx)).astype(np.uint16)
        return roi_img

    def gen_fine_dc_roi_imgs(self):
        if self.config['deconvolve'] != 0:
            imgs = self.image_handler.spot_images['spot_images_decon']
        else:
            imgs = self.image_handler.spot_images['spot_images']
        
        rois = self.all_rois

        i = 0
        roi_array_fine = np.empty(len(imgs), object)
        for j, frame_stack in enumerate(imgs):
            roi_stack_fine = []
            for roi_stack in frame_stack:
                roi_stack_fine.append(self.fine_dc_single_roi_img(roi_stack, rois.iloc[i]))
                i += 1
            roi_array_fine[j] = np.stack(roi_stack_fine)
        #roi_imgs_fine = Parallel(n_jobs=-1, verbose=1, prefer='threads')(delayed(self.fine_dc_single_roi_img)(roi_imgs[i], rois.iloc[i]) for i in tqdm(range(roi_imgs.shape[0])))
        #roi_imgs_fine = np.stack(roi_array_fine)
        #roi_array_fine = np.array(roi_array_fine, dtype='object')
        
        self.image_handler.spot_images['spot_images_fine'] = roi_array_fine
        np.save(self.image_handler.spot_images_path+os.sep+'spot_images_fine.npy', roi_array_fine)


    def trace_single_roi(self, roi_img, mask = None):
        if np.any(roi_img): #Check if empty due to error
            if mask is None:
                fit = self.fit_func(roi_img, sigma=1, center='max')[0]
            else:
                roi_img_masked = mask/np.max(mask) * roi_img
                center = list(np.unravel_index(np.argmax(roi_img_masked, axis=None), roi_img.shape))
                fit = self.fit_func(roi_img, sigma=1, center=center)[0]
            return fit
        else:
            fit = [-1, -1, -1, -1, -1, -1, -1]
            return fit
        


    def trace_all_rois(self):
        '''
        Fits 3D gaussian to previously detected ROIs in all timeframes.
       
        Returns
        -------
        res : Pandas dataframe containing trace data
        imgs : Hyperstack image with raw image data of each ROI.
    
        '''

        if self.config['deconvolve'] != 0:
            imgs = self.image_handler.spot_images['spot_images_decon']
        else:
            imgs = self.image_handler.spot_images['spot_images']
        
        rois = self.all_rois

        fits = []

        #fits = Parallel(n_jobs=-1, prefer='threads')(delayed(self.trace_single_roi)(roi_imgs[i]) for i in tqdm(range(roi_imgs.shape[0])))
        try:
            mask_fits = self.image_handler.config['mask_fits']
        except KeyError:
            mask_fits = False

        if mask_fits:
            for p, pos_imgs in tqdm(enumerate(imgs)):
                ref_img = pos_imgs[int(rois.query('roi_id == @p').iloc[0]['ref_frame'])]
                #print(ref_img.shape)
                for spot_img in pos_imgs:
                        fits.append(self.trace_single_roi(spot_img, mask = ref_img))
                #Parallel(n_jobs=1, prefer='threads')(delayed(self.trace_single_roi)(imgs[p, t], mask= ref_img) for t in range(imgs.shape[1]))

        else:
            for pos_imgs in tqdm(imgs):
                for spot_img in pos_imgs:
                    fits.append(self.trace_single_roi(spot_img))

        trace_res = pd.DataFrame(fits,columns=["BG","A","z_px","y_px","x_px","sigma_z","sigma_xy"])
        #trace_index = pd.DataFrame(fit_rois, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
        traces = pd.concat([rois, trace_res], axis=1)
        traces.rename(columns={"roi_id": "trace_id"}, inplace=True)

        #Apply fine scale drift to fits, and physcial units.
        traces['z_px']=traces['z_px']+traces['z_px_fine']
        traces['y_px']=traces['y_px']+traces['y_px_fine']
        traces['x_px']=traces['x_px']+traces['x_px_fine']
        #traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_px']*self.config['z_nm']
        traces['y']=traces['y_px']*self.config['xy_nm']
        traces['x']=traces['x_px']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces['sigma_xy']=traces['sigma_xy']*self.config['xy_nm']
        traces = traces.sort_values(['trace_id', 'frame'])

        if self.trace_beads: 
            suffix = '_beads'
        else:
            suffix = ''

        self.image_handler.traces = traces
        self.image_handler.save_data(traces=traces, suffix=suffix)

"""
    def trace_single_frame(self, trace_images, roi, decon_params):
        #trace_ch = self.config['trace_ch']
        roi_image_size = tuple(self.config['roi_image_size'])
        roi_slice = roi['roi_slice']
        pad = roi['pad']

        try:
            roi_image = np.array(trace_images[roi_slice[0], 
                                            roi_slice[1],
                                            roi_slice[2]])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                #print('Padding ', pad)
                roi_image = np.pad(roi_image, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_image = np.zeros(roi_image_size, dtype=np.float32)
        
        if decon_params[0] != 0:
            roi_image = ip.decon_RL(roi_image, decon_params[2], decon_params[1], decon_params[3], niter=decon_params[0])

        #Perform 3D gaussian fit
        #trace_res.append(delayed(self.fit_func)(roi_image, sigma=1, center='max')[0])
        fit = self.fit_func(roi_image, sigma=1, center='max')[0]
        #Ensure images are compatible size for hyperstack.
        
        #roi_image_exp = delayed(ip.pad_to_shape)(roi_image, roi_image_size)
        if roi_image.shape != roi_image_size:
            #print(roi_image.shape)
            roi_image = ip.pad_to_shape(roi_image, roi_image_size)

        #Extract fine drift from drift table and shift image for display.
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
        roi_image = ndi.shift(roi_image, (dz, dy, dx))

        return fit, roi[['roi_id', 'frame', 'ref_frame', 'position', 'z_px_fine', 'y_px_fine', 'x_px_fine']].values, roi_image

    def tracing_3d(self):
        
        Fits 3D gaussian to previously detected ROIs in all timeframes.
       
        Returns
        -------
        res : Pandas dataframe containing trace data
        imgs : Hyperstack image with raw image data of each ROI.
    

        #Extract parameters from config and predefined roi table.
        decon_params = []
        decon_params.append(self.config['deconvolve'])
        if decon_params[0] != 0:
            algo, kernel, fd_data = ip.decon_RL_setup(  res_lateral=self.config['xy_nm']/1000, 
                                                        res_axial=self.config['z_nm']/1000, 
                                                        wavelength=self.config['spot_wavelength']/1000,
                                                        na=self.config['objective_na'])
            decon_params.append(algo)
            decon_params.append(kernel)
            decon_params.append(fd_data)
        #roi_image_size = self.config['roi_image_size']
        #roi_table = self.roi_table[self.roi_table['position'].isin(self.pos_list)]

        fits = []
        fit_rois = []
        roi_imgs = []


        for pos in sorted(list(self.all_rois['position'].unique())):
            pos_index = self.pos_list.index(pos)
            for t in sorted(list(self.all_rois['frame'].unique())):
                for ch in sorted(list(self.all_rois['ch'].unique())):

                    #print('Loading all images in position ', pos)
                    trace_images = np.array(self.images[pos_index, t, ch]) #self.images[pos_index, t, ch] #
                    sel_rois = self.all_rois[(self.all_rois['position'] == pos) & (self.all_rois['ch'] == ch) & (self.all_rois['frame'] == t)]
                    print(f'Tracing {len(sel_rois)} ROIs in position', pos, ', frame ', t, ', channel ', ch)
                    out = Parallel(n_jobs=-1, prefer='threads')(delayed(self.trace_single_frame)(trace_images, roi, decon_params) for i, roi in tqdm(sel_rois.iterrows(), total=sel_rois.shape[0]))
                    for sublist in out:
                        fits.append(sublist[0])
                        fit_rois.append(sublist[1])
                        roi_imgs.append(sublist[2])
        

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
        


        #Cleanup of results into dataframe format
        trace_res = pd.DataFrame(fits,columns=["BG","A","z_px","y_px","x_px","sigma_z","sigma_xy"])
        trace_index = pd.DataFrame(fit_rois, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
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
        #roi_image_size = tuple(self.config['roi_image_size'])
        #all_images = np.reshape(np.stack(all_images), 
        #                        (max(traces.trace_id)+1,max(traces.frame)+1, 
        #                        roi_image_size[0], roi_image_size[1], roi_image_size[2]))
        
        T = self.images_shape[1]
        all_images = np.stack(roi_imgs)
        all_images = np.reshape(all_images, (all_images.shape[0]//T, T, all_images.shape[1], all_images.shape[2], all_images.shape[3])).astype(np.uint16)
        
        if self.trace_beads: 
            suffix = '_beads'
        else:
            suffix = ''
        self.image_handler.save_data(traces=traces, suffix=suffix)
        self.image_handler.save_data(imgs=all_images, suffix=suffix)
        return traces, all_images
""" 
