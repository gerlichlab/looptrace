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
from looptrace import image_io
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
#import dask
#from dask import delayed
from tqdm import tqdm
#from joblib import Parallel, delayed
import os

class Tracer:
    def __init__(self, image_handler, trace_beads = False):
        '''
        Initialize Tracer class with config read in from YAML file.
    '''
        self.image_handler = image_handler
        self.config_path = image_handler.config_path
        self.config = image_handler.config
        self.drift_table = image_handler.tables['drift_correction']
        self.images = self.image_handler.images['seq_images']
        self.image_lists = self.image_handler.image_lists['seq_images']
        self.trace_beads = trace_beads
        if trace_beads:
            self.roi_table = image_handler.tables['bead_rois']
        else:
            self.roi_table = image_handler.tables['rois']

        self.fit_funcs = {'LS': fitSymmetricGaussian3D, 'MLE': fitSymmetricGaussian3DMLE}
        self.fit_func = self.fit_funcs[self.config['fit_func']]

    def make_dc_rois_all_frames(self):
        #Precalculate all ROIs for extracting spot images, based on identified ROIs and precalculated drifts between time frames.
        print('Generating list of all ROIs for tracing:')
        #positions = sorted(list(self.roi_table.position.unique()))

        all_rois = []
        for i, roi in tqdm(self.roi_table.iterrows()):
            pos = roi['position']
            pos_index = self.image_lists.index(pos)#positions.index(pos)
            sel_dc = self.drift_table.query('position == @pos')
            ref_frame = roi['frame']
            ch = roi['ch']
            ref_offset = sel_dc.query('frame == @ref_frame')
            Z, Y, X = self.images[pos_index][0,ch].shape[-3:]
            for j, dc_frame in sel_dc.iterrows():
                z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])

                z_min = roi['z_min'] - z_drift_course
                z_max = roi['z_max'] - z_drift_course
                y_min = roi['y_min'] - y_drift_course
                y_max = roi['y_max'] - y_drift_course
                x_min = roi['x_min'] - x_drift_course
                x_max = roi['x_max'] - x_drift_course

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
                #print('Appending ', s)
                all_rois.append([pos, pos_index, roi.name, dc_frame['frame'], ref_frame, ch, s, pad, z_drift_course, y_drift_course, x_drift_course, 
                                                                                        dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])

        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 'roi_slice', 'pad', 'z_px_course', 'y_px_course', 'x_px_course', 
                                                                                                  'z_px_fine', 'y_px_fine', 'x_px_fine'])

    def extract_single_roi_img(self, single_roi):
        #Function to extract single ROI lazily without loading entire stack in RAM.
        #Depending on chunking of original data can be more or less performant.

        roi_image_size = tuple(self.config['roi_image_size'])
        p = single_roi['pos_index']
        t = single_roi['frame']
        c = single_roi['ch']
        z, y, x = single_roi['roi_slice']
        pad = single_roi['pad']

        try:
            roi_img = np.array(self.images[p][t, c, z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                #print('Padding ', pad)
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros(roi_image_size, dtype=np.float32)

        #print(p, t, c, z, y, x)
        return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def extract_single_roi_img_inmem(self, single_roi, images):
        # Function for extracting a single cropped region defined by ROI from a larger 3D image.

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
        # Load full stacks into memory to extract spots.
        # Not the most elegant, but depending on the chunking of the original data it is often more performant than loading segments from zarr.


        #roi_imgs = Parallel(n_jobs=-1, verbose=0, prefer='threads')(delayed(self.extract_single_roi_img)(single_roi) for i, single_roi in tqdm(test.iterrows(), total=len(test)))#tqdm(self.all_rois.iterrows()))
        #roi_imgs = [self.extract_single_roi_img(single_roi) for i, single_roi in tqdm(test.iterrows())]#tqdm(self.all_rois.iterrows()))
        #roi_image_size = tuple(self.config['roi_image_size'])
        rois = self.all_rois#.iloc[0:500]

        P = max(rois['pos_index'])+1
        T = max(rois['frame'])+1
        roi_array = {}
        #roi_array_padded = []

        for pos in tqdm(range(P), total = P):
                for frame in range(T):
                    rois_stack = rois.query('pos_index == @pos & frame == @frame')
                    #print(rois_stack)
                    roi_ch = int(rois['ch'].unique())
                    image_stack = np.array(self.images[pos][frame, roi_ch])
                    #print(image_stack.shape)
                    for j, single_roi in rois_stack.iterrows():
                        id = single_roi['roi_id']
                        t = single_roi['frame']
                        roi = self.extract_single_roi_img_inmem(single_roi, image_stack).astype(np.uint16)
                        roi_array[id, t] = roi
                        #roi_array_padded.append(ip.pad_to_shape(roi, shape = roi_image_size, mode = 'minimum'))
        
        #print(roi_array.keys())
        
        pos_rois = []
      
        for roi_id in rois.roi_id.unique():
            try:
                pos_rois.append(np.stack([roi_array[(roi_id, frame)] for frame in range(T)]))
            except KeyError:
                break
            except ValueError: #Edge case handling for rois very close to the edge, sometimes the intial padding does not work properly due to rounding errors.
                roi_size = roi_array[(roi_id, T-1)].shape
                pos_rois.append(np.stack([ip.pad_to_shape(roi_array[(roi_id, frame)], roi_size) for frame in range(T)]))
        
        #self.temp_array = pos_rois
        #roi_array_padded = np.stack(roi_array_padded)

        print('ROIs generated.')
        self.image_handler.images['spot_images'] = pos_rois
        #self.image_handler.spot_images['spot_images_padded'] = roi_array_padded
        np.savez_compressed(self.config['image_path']+os.sep+'spot_images.npz', *pos_rois)
        #np.save(self.image_handler.spot_images_path+os.sep+'spot_images_padded.npy', roi_array_padded)

    def gen_roi_imgs_inmem_coursedc(self):
        # Use this simplified function if the images that the spots are gathered from are already coursely drift corrected!
        #rois = self.roi_table#.iloc[0:500]
        #imgs = self.
        all_spots = {}
        for pos, group in tqdm(self.roi_table.groupby('position')):
            pos_index = self.image_lists.index(pos)
            full_image = np.array(self.images[pos_index])
            #print(full_image.shape)
            for roi in group.to_dict('records'):
                spot_stack = full_image[:, 
                                roi['ch'], 
                                roi['z_min']:roi['z_max'], 
                                roi['y_min']:roi['y_max'],
                                roi['x_min']:roi['x_max']].copy()
                #print(spot_stack.shape)
                all_spots[pos+'_'+str(roi['frame'])+'_'+str(roi['roi_id_pos']).zfill(2)] = spot_stack
        #self.image_handler.images['spot_images'] = all_spots
        np.savez_compressed(self.config['image_path']+os.sep+'spot_images.npz', **all_spots)
        self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.config['image_path']+os.sep+'spot_images.npz')
        
    
    def decon_roi_imgs(self):
        # Run deconvolution using generated or experimental PSF usign Flowdec.

        decon_iter = self.config['deconvolve']
        if decon_iter != 0:
            algo, kernel, fd_data = ip.decon_RL_setup(  res_lateral=self.config['xy_nm']/1000, 
                                                        res_axial=self.config['z_nm']/1000, 
                                                        wavelength=self.config['spot_wavelength']/1000,
                                                        na=self.config['objective_na'])
            
            if 'exp_psf' in self.image_handler.images:
                kernel = self.image_handler.images['exp_psf']
        
            imgs = self.image_handler.images['spot_images']
            roi_array_decon = []
            for i, frame_stack in tqdm(enumerate(imgs), total=len(imgs)):
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
                        
                roi_array_decon.append(np.stack(roi_stack_decon))

                        #print(p,t)
            #roi_array_decon = np.array(roi_array_decon, dtype='object')
            #ip.imgs_to_ome_zarr(images = roi_array, path=self.config['input_path']+os.sep+'spot_images', name = 'spot_imgs', axes=['p','t','z','y','x'])
            self.image_handler.images['spot_images_decon'] = roi_array_decon
            #self.image_handler.spot_images['spot_images_padded'] = roi_array_padded
            np.savez_compressed(self.config['image_path']+os.sep+'spot_images_decon.npz', *roi_array_decon)
            print('Images deconvolved')
        

    def fine_dc_single_roi_img(self, roi_img, roi):
        #Shift a single image according to precalculated drifts.
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
        roi_img = ndi.shift(roi_img, (dz, dy, dx)).astype(np.uint16)
        return roi_img

    def gen_fine_dc_roi_imgs(self):
        #Apply fine scale drift correction to spot images, used mainly for visualizing fits (these images are not used for fitting)
        print('Making fine drift-corrected spot images.')
        if self.config['deconvolve'] != 0:
            imgs = self.image_handler.images['spot_images_decon']
        else:
            imgs = self.image_handler.images['spot_images']
        
        rois = self.all_rois

        i = 0
        roi_array_fine = []
        for j, frame_stack in tqdm(enumerate(imgs)):
            roi_stack_fine = []
            for roi_stack in frame_stack:
                roi_stack_fine.append(self.fine_dc_single_roi_img(roi_stack, rois.iloc[i]))
                i += 1
            roi_array_fine.append(np.stack(roi_stack_fine))

        #roi_imgs_fine = Parallel(n_jobs=-1, verbose=1, prefer='threads')(delayed(self.fine_dc_single_roi_img)(roi_imgs[i], rois.iloc[i]) for i in tqdm(range(roi_imgs.shape[0])))
        #roi_imgs_fine = np.stack(roi_array_fine)
        #roi_array_fine = np.array(roi_array_fine, dtype='object')
        
        self.image_handler.images['spot_images_fine'] = roi_array_fine
        np.savez_compressed(self.config['image_path']+os.sep+'spot_images_fine.npz', *roi_array_fine)


    def trace_single_roi(self, roi_img, mask = None):
        
        #Fit a single roi with 3D gaussian (MLE or LS as defined in config).
        #Masking by intensity or label image can be used to improve fitting correct spot (set in config)


        if np.any(roi_img): #Check if empty due to error
            if mask is None:
                fit = self.fit_func(roi_img, sigma=1, center='max')[0]
            else:
                roi_img_masked = mask/np.max(mask) * roi_img
                center = list(np.unravel_index(np.argmax(roi_img_masked, axis=None), roi_img.shape))
                fit = self.fit_func(roi_img, sigma=1, center=center)[0]
            return fit
        else:
            fit = np.array([-1, -1, -1, -1, -1, -1, -1])
            return fit
        

    def trace_all_rois(self):
        '''
        Fits 3D gaussian to previously detected ROIs across positions and timeframes.
    
        '''

        if self.config['deconvolve'] != 0:
            imgs = self.image_handler.images['spot_images_decon']
        else:
            imgs = self.image_handler.images['spot_images']
        
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
        traces['z_px_dc']=traces['z_px']+traces['z_px_fine']
        traces['y_px_dc']=traces['y_px']+traces['y_px_fine']
        traces['x_px_dc']=traces['x_px']+traces['x_px_fine']
        #traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_px_dc']*self.config['z_nm']
        traces['y']=traces['y_px_dc']*self.config['xy_nm']
        traces['x']=traces['x_px_dc']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces['sigma_xy']=traces['sigma_xy']*self.config['xy_nm']
        traces = traces.sort_values(['trace_id', 'frame'])

        if self.trace_beads: 
            suffix = '_beads'
        else:
            suffix = ''

        self.image_handler.traces = traces
        traces.to_csv(self.image_handler.traces_path)