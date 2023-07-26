# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import logging
import os
from typing import *

import dask.array as da
import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels, relabel_sequential
from skimage.measure import regionprops_table
from skimage.transform import rescale
from skimage.morphology import remove_small_objects
import tqdm

from looptrace import image_io
from looptrace import image_processing_functions as ip

logger = logging.getLogger()


class NucDetector:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''
    def __init__(self, image_handler, array_id = None):
        self.image_handler = image_handler
        try:
            self.images = self.image_handler.images[self.input_name]
            self.pos_list = self.image_handler.image_lists[self.input_name]
        except KeyError as e:
            logger.warning("Error in nuclei detector setup: {e}")
        if array_id is not None:
            self.pos_list = [self.pos_list[int(array_id)]]

    @property
    def config(self) -> Dict[str, Any]:
        return self.image_handler.config

    @property
    def input_name(self) -> str:
        return self.config['nuc_input_name']

    @property
    def nuc_images_path(self) -> str:
        return self._save_img_path("nuc_images")
    
    @property
    def nuc_images_path(self) -> str:
        return self._save_img_path("nuc_masks")
    
    @property
    def nuc_images_path(self) -> str:
        return self._save_img_path("nuc_classes")
    
    def _get_img_save_path(self, name: str) -> str:
        return os.path.join(self._save_img_path, name)

    @property
    def _save_img_path(self):
        return self.image_handler.images_save_path

    def gen_nuc_images(self):
        '''
        Saves 2D/3D (defined in config) images of the nuclear channel into image folder for later analysis.
        '''
        try:
            nuc_slice = self.config['nuc_slice']
        except KeyError: #Legacy config
            nuc_slice = -1
        try:
            nuc_3d = self.config['nuc_3d']
        except KeyError:
            nuc_3d = False
        
        print('Generating nuclei images.')
        imgs = []
        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.pos_list.index(pos)

            if nuc_3d:
                img = self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel']].compute()
            elif nuc_slice == -1:
                img = da.max(self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel']], axis=0).compute()
            else:
                img = self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel'], self.config['nuc_slice']].compute()
            imgs.append(img.astype(np.uint16))

        if nuc_3d:
            image_io.images_to_ome_zarr(images=imgs, path=self.nuc_images_path, name='nuc_images', axes=('p','z','y','x'), chunk_split=(1,1), dtype = np.uint16)
        else:
            image_io.images_to_ome_zarr(images=imgs, path=self.nuc_images_path, name='nuc_images', axes=('p','y','x'), chunk_split=(1,1), dtype = np.uint16)
        
        del imgs #Cleanup RAM

    def segment_nuclei(self) -> str:
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''
        
        if 'nuc_images' not in self.image_handler.images:
            self.gen_nuc_images()
            self.image_handler.read_images()
        
        method = self.config.get('nuc_method', 'nuclei')
        
        try:
            nuc_3d = self.config['nuc_3d']
            ds_xy = self.config['nuc_downscaling_xy']
            
            if nuc_3d:
                anisotropy = self.config['nuc_anisotropy']
                ds_z = self.config['nuc_downscaling_z']
                nuc_min_size = self.config['nuc_min_size'] / (ds_z * ds_xy * ds_xy)
            else:
                nuc_min_size = self.config['nuc_min_size'] / (ds_xy * ds_xy)
                
            mitosis_class = self.config['nuc_mitosis_class']
        except KeyError:
            nuc_3d = False
            ds_xy = 4
            nuc_min_size = 10
            mitosis_class = False
        
        diameter = self.config['nuc_diameter'] / ds_xy
        
        nuc_imgs_in = self.image_handler.images['nuc_images']
        nuc_imgs = []

        #Remove unused dimensions and downscale input images.
        for img in nuc_imgs_in:        
            new_slice = tuple([0 if i == 1 else slice(None) for i in img.shape])
            img = img[new_slice]
            img = np.array(img[::ds_z, ::ds_xy, ::ds_xy]) if nuc_3d else np.array(img[::ds_xy,::ds_xy]) 
            nuc_imgs.append(img)
        
        print(f'Running nuclear segmentation using CellPose {method} model and diameter {diameter}.')

        if nuc_3d:
            print("Using all 3 dimensions for nuclei detection")
            masks = ip.nuc_segmentation_cellpose_3d(nuc_imgs, diameter = diameter, model_type = method, anisotropy=anisotropy)
            scaling = (ds_z, ds_xy, ds_xy)
            axes_for_zarr = ('p', 'z', 'y', 'x')
        else:
            print("Using just 2 dimensions for nuclei detection")
            masks = ip.nuc_segmentation_cellpose_2d(nuc_imgs, diameter = diameter, model_type = method)
            scaling = (ds_xy, ds_xy)
            axes_for_zarr = ('p', 'y', 'x')

        #Remove under-segmented nuclei and clean up:
        masks = [remove_small_objects(arr, min_size=nuc_min_size) for arr in masks]
        masks = [relabel_sequential(arr)[0] for arr in masks]

        if mitosis_class:
            print(f'Detecting mitotic cells on top of CellPose nuclei.')
            masks, mitotic_idx = zip(*[ip.mitotic_cell_extra_seg(np.array(nuc_imgs[i]), masks[i]) for i in range(len(nuc_imgs))])

        masks = [rescale(expand_labels(mask.astype(np.uint16), 3), scale=scaling, order=0) for mask in masks]
        #masks = np.stack(masks)

        print('Saving segmentations.')

        self.image_handler.images['nuc_masks'] = masks

        image_io.images_to_ome_zarr(images=masks, path=self.nuc_masks_path, name='nuc_masks', axes=axes_for_zarr, dtype = np.uint16, chunk_split=(1,1))
        
        if mitosis_class:
            nuc_class = []
            for i, mask in enumerate(masks):
                class_1 = ((mask > 0) & (mask < mitotic_idx[i])).astype(int)
                class_2 = (mask >= mitotic_idx[i]).astype(int)
                nuc_class.append(class_1 + class_2*2)
            #nuc_class = np.stack(nuc_class).astype(np.uint16)
            print('Saving classifications.')

            self.image_handler.images['nuc_classes'] = nuc_class
            image_io.images_to_ome_zarr(images=nuc_class, path=self.nuc_classes_path, name='nuc_classes', axes=axes_for_zarr, dtype = np.uint16, chunk_split=(1,1))

        return self.nuc_masks_path
            
    def update_masks_after_qc(self, new_mask, original_mask, mask_name, position):

        try:
            nuc_3d = self.config['nuc_3d']
        except KeyError:
            nuc_3d = False
        s = tuple([slice(None, None, 4) for i in range(new_mask.ndim)])
        if not np.allclose(new_mask[s], original_mask[s]):
            print('Segmentation labels changed, resaving.')
            nuc_mask = ip.relabel_nucs(new_mask)
            pos_index = self.image_handler.image_lists[mask_name].index(position)
            self.image_handler.images[mask_name] = nuc_mask.astype(np.uint16)
            if nuc_3d:
                image_io.single_position_to_zarr(images = self.image_handler.images[mask_name][pos_index], path=self.nuc_masks_path+position, name = mask_name, axes=('z','y','x'), dtype=np.uint16, chunk_split=(1,1))
            else:
                image_io.single_position_to_zarr(images = self.image_handler.images[mask_name][pos_index], path=self.nuc_masks_path+position, name = mask_name, axes=('y','x'), dtype=np.uint16, chunk_split=(1,1))


    def gen_nuc_rois(self):
        nuc_rois = []
        nuc_masks = self.image_handler.images['nuc_masks']
        #ch = self.config['trace_ch']
        #ref_frame = self.config['nuc_ref_frame']

        if 'nuc_classes' in self.image_handler.images:
            print('Adding classes.')
            nuc_classes = self.image_handler.images['nuc_classes']
        else:
            nuc_classes = [None]*len(nuc_masks)
        
        for i in tqdm.tqdm(range(len(nuc_masks))):

            mask = np.array(nuc_masks[i])
            nuc_class = np.array(nuc_classes[i])

            if mask.ndim == 2:
                nuc_props = pd.DataFrame(regionprops_table(mask, intensity_image=nuc_class, properties=['label', 'bbox', 'intensity_mean'])).rename(columns={'bbox-0':'y_min', 
                                                                                'bbox-1':'x_min', 
                                                                                'bbox-2':'y_max', 
                                                                                'bbox-3':'x_max'})
            else:
                nuc_props = pd.DataFrame(regionprops_table(mask, intensity_image=nuc_class, properties=['label', 'bbox', 'intensity_mean'])).rename(columns={'bbox-0':'z_min', 
                                                                'bbox-1':'y_min', 
                                                                'bbox-2':'x_min', 
                                                                'bbox-3':'z_max', 
                                                                'bbox-4':'y_max', 
                                                                'bbox-5':'x_max'})

            for j, roi in nuc_props.iterrows():
                old_pos = 'P'+str(i+1).zfill(4)
                new_pos = 'P'+str(i+1).zfill(4)+'_'+str(j+1).zfill(4)
                sel_dc = self.image_handler.tables['drift_correction_full_frame'].query('position == @old_pos')
                ref_offset = sel_dc.query('frame == @ref_frame')
                try:
                    Z, Y, X = self.images[i][0,ch].shape[-3:]
                except AttributeError: #Images not loaded for some reason
                    Z = 200
                    Y = nuc_masks[0].shape[-2]
                    X = nuc_masks[0].shape[-1]
                
                for k, dc_frame in sel_dc.iterrows():
                    z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                    y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                    x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])

                    if nuc_masks[0].ndim == 2:
                        z_min = 0
                        z_max = Z
                    else:
                        z_min = int(roi['z_min'] - z_drift_course)
                        z_max = int(roi['z_max'] - z_drift_course)
                    y_min = int(roi['y_min'] - y_drift_course)
                    y_max = int(roi['y_max'] - y_drift_course)
                    x_min = int(roi['x_min'] - x_drift_course)
                    x_max = int(roi['x_max'] - x_drift_course)

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

                    nuc_rois.append([old_pos, new_pos, i, roi.name, dc_frame['frame'], ref_frame, ch, s, pad, z_drift_course, y_drift_course, x_drift_course, 
                                                                                            dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine'], roi['intensity_mean']])

        nuc_rois = pd.DataFrame(nuc_rois, columns=['orig_position','position', 'orig_pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 'roi_slice', 'pad', 'z_px_course', 'y_px_course', 'x_px_course', 
                                                                                                'z_px_fine', 'y_px_fine', 'x_px_fine', 'nuc_class'])
        nuc_rois['nuc_class'] = nuc_rois['nuc_class'].round()
        self.nuc_rois = nuc_rois
        self.nuc_rois.to_csv(self.image_handler.out_path('nuc_table.csv'))

    def gen_nuc_rois_prereg(self):
        nuc_rois = []
        nuc_masks = self.image_handler.images['nuc_masks']
        #ch = self.config['trace_ch']
        #ref_frame = self.config['nuc_ref_frame']
        
        for i in tqdm.tqdm(range(len(nuc_masks))):

            mask = np.array(nuc_masks[i][0,0])
            if mask.shape[0] == 1:
                nuc_props = pd.DataFrame(regionprops_table(mask, properties=['label', 'bbox', 'area'])).rename(columns={'bbox-0':'y_min', 
                                                                                'bbox-1':'x_min', 
                                                                                'bbox-2':'y_max', 
                                                                                'bbox-3':'x_max'})
            else:
                nuc_props = pd.DataFrame(regionprops_table(mask, properties=['label', 'bbox', 'area'])).rename(columns={'bbox-0':'z_min', 
                                                                'bbox-1':'y_min', 
                                                                'bbox-2':'x_min', 
                                                                'bbox-3':'z_max', 
                                                                'bbox-4':'y_max', 
                                                                'bbox-5':'x_max'})

            nuc_props['orig_position'] = self.image_handler.image_lists['nuc_masks'][i]
            nuc_props['position'] = self.image_handler.image_lists['nuc_masks'][i]+'_'+'P'+nuc_props['label'].apply(str).str.zfill(3)
            nuc_rois.append(nuc_props)

        self.nuc_rois = pd.concat(nuc_rois).reset_index(drop=True)
        self.nuc_rois.to_csv(self.image_handler.out_path('nuc_rois.csv'))
                
                
    def extract_single_roi_img(self, single_roi, images):
        # Function for extracting a single cropped region defined by ROI from a larger 3D image.

        z = slice(single_roi['z_min'], single_roi['z_max'])
        y = slice(single_roi['y_min'], single_roi['y_max'])
        x = slice(single_roi['x_min'], single_roi['x_max'])
        #pad = single_roi['pad']

        try:
            roi_img = np.array(images[z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            #if pad != ((0,0),(0,0),(0,0)):
            #    roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros((z,y,x), dtype=np.float32)

        return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def gen_single_nuc_images(self):
        #Function to extract single nuclei images from full FOVs
        # A bit convoluted currently to ensure efficient loading of appropriate arrays into RAM for processing.
        print('Extracting nucs from ', self.config['nuc_extract_target'])
        rois = self.image_handler.tables['nuc_rois']#.iloc[0:500]
        rois_orig_all_pos = sorted(list(rois.orig_position.unique()))
        input_images = self.image_handler.images[self.config['nuc_extract_target']]

        T = input_images[0].shape[0]
        C = input_images[0].shape[1]

        
        for pos in self.pos_list:
            roi_array = {}
            print("extracting nucs in position ", pos)
            pos_index = self.image_handler.image_lists[self.config['nuc_extract_target']].index(pos)
            rois_orig_pos = rois[rois.orig_position == rois_orig_all_pos[pos_index]]

            for frame in tqdm.tqdm(range(T)):
                image_stack = np.array(input_images[pos_index][frame])
                for i, pos_roi in rois_orig_pos.iterrows():
                    roi_array[pos_roi['position'], frame] = np.stack([self.extract_single_roi_img(pos_roi, image_stack[c]).astype(np.uint16) for c in range(C)])
        
            for nuc_position in tqdm.tqdm(rois_orig_pos.position.to_list()):
                try:
                    nuc_pos_images = np.stack([roi_array[(nuc_position, frame)] for frame in range(T)])
                    #print('Made pos roi ', roi_id, pos_rois.shape)
                except KeyError:
                    break

                image_io.single_position_to_zarr(images=nuc_pos_images, 
                            path=self.image_handler.image_save_path+os.sep+self.config['nuc_extract_target']+'_single_nucs', 
                            name='single_nucs',
                            pos_name = nuc_position, 
                            axes=('t','c','z','y','x'), 
                            chunk_axes = ('t','z','y','x'),
                            dtype = np.uint16, 
                            chunk_split=(1,1,1,1))
        print('ROI images generated, please reinitialize to load them.')
