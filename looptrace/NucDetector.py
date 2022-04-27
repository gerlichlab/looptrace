# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import image_processing_functions as ip
from looptrace import image_io
import dask.array as da
import os
import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
from skimage.transform import rescale
from tqdm import tqdm

class NucDetector:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''

    def __init__(self, image_handler):
        self.image_handler = image_handler
        self.config = image_handler.config
        try:
            self.images = self.image_handler.images['seq_images_full']
            self.pos_list = self.image_handler.image_lists['seq_images_full']
        except KeyError:
            pass
        self.nuc_images_path = self.config['image_path']+os.sep+'nuc_images'
        self.nuc_masks_path = self.config['image_path']+os.sep+'nuc_masks'
        self.nuc_classes_path = self.config['image_path']+os.sep+'nuc_classes'

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
        

        imgs = []
        for pos in self.pos_list:
            pos_index = self.pos_list.index(pos)

            if nuc_3d:
                img = self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel']].compute()
            elif nuc_slice == -1:
                img = da.max(self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel']], axis=0).compute()
            else:
                img = self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel'], self.config['nuc_slice']].compute()
            imgs.append(img)
        imgs = np.stack(imgs).astype(np.uint16)
        self.image_handler.images['nuc_images']= imgs

        if nuc_3d:
            image_io.images_to_ome_zarr(images=imgs, path=self.nuc_images_path, name='nuc_images', axes=('p','z','y','x'), chunk_split=(1,1), dtype = np.uint16)
        else:
            image_io.images_to_ome_zarr(images=imgs, path=self.nuc_images_path, name='nuc_images', axes=('p','y','x'), chunk_split=(1,1), dtype = np.uint16)

    def segment_nuclei(self):
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''

        if 'nuc_images' not in self.image_handler.images:
            print('Generating nuclei images.')
            self.gen_nuc_images()

        diameter = self.config['nuc_diameter']/4
        
        try:
            method = self.config['nuc_method']
        except KeyError:
            method = 'nuclei'
        try:
            nuc_3d = self.config['nuc_3d']
        except KeyError:
            nuc_3d = False
        
        
        nuc_imgs_in = self.image_handler.images['nuc_images']
        nuc_imgs = []

        #Remove unused dimensions and downscale input images.
        for img in nuc_imgs_in:        
            new_slice = tuple([0 if i == 1 else slice(None) for i in img.shape])
            img = img[new_slice]
            if nuc_3d:
                img = np.array(img[::4, ::4, ::4])
            else:
                img = np.array(img[::4,::4])
            nuc_imgs.append(img)
        
        print(f'Running nuclear segmentation using CellPose {method} model and diameter {diameter}.')

        if nuc_3d:
            masks = ip.nuc_segmentation_cellpose_3d(nuc_imgs, diameter = diameter, model_type = method)
        else:
            masks = ip.nuc_segmentation_cellpose_2d(nuc_imgs, diameter = diameter, model_type = method)
            
        print(f'Detecting mitotic cells on top of CellPose nuclei.')
        masks, mitotic_idx = zip(*[ip.mitotic_cell_extra_seg(np.array(nuc_imgs[i]), masks[i]) for i in range(len(nuc_imgs))])

        masks = [rescale(expand_labels(mask.astype(np.uint16),3), scale = 4, order = 0) for mask in masks]
        masks = np.stack(masks)

        print('Saving segmentations.')

        self.image_handler.images['nuc_masks']= masks

        if nuc_3d:
            image_io.images_to_ome_zarr(images=masks, path=self.nuc_masks_path, name='nuc_masks', axes=('p','z','y','x'), dtype = np.uint16, chunk_split=(1,1))
        else:
            image_io.images_to_ome_zarr(images=masks, path=self.nuc_masks_path, name='nuc_masks', axes=('p','y','x'), dtype = np.uint16, chunk_split=(1,1))

        nuc_class = []
        for i, mask in enumerate(masks):
            class_1 = ((mask > 0) & (mask < mitotic_idx[i])).astype(int)
            class_2 = (mask >= mitotic_idx[i]).astype(int)
            nuc_class.append(class_1 + class_2*2)
        nuc_class = np.stack(nuc_class).astype(np.uint16)
        print('Saving classifications.')

        self.image_handler.images['nuc_classes'] = nuc_class
        if nuc_3d:
            image_io.images_to_ome_zarr(images=nuc_class, path=self.nuc_classes_path, name='nuc_classes', axes=('p','z', 'y','x'), dtype = np.uint16, chunk_split=(1,1))
        else:
            image_io.images_to_ome_zarr(images=nuc_class, path=self.nuc_classes_path, name='nuc_classes', axes=('p','y','x'), dtype = np.uint16, chunk_split=(1,1))

    def gen_nuc_rois(self):
        nuc_rois = []
        nuc_masks = self.image_handler.images['nuc_masks']
        ch = self.config['trace_ch']
        ref_frame = self.config['nuc_ref_frame']

        if 'nuc_classes' in self.image_handler.images:
            print('Adding classes.')
            nuc_classes = self.image_handler.images['nuc_classes']
        else:
            nuc_classes = [None]*len(nuc_masks)
        
        for i in tqdm(range(len(nuc_masks))):

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
                sel_dc = self.image_handler.tables['drift_correction_full'].query('position == @old_pos')
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

                    nuc_rois.append([old_pos, new_pos, i, roi.name, dc_frame['frame'], ref_frame, ch, s, pad, z_drift_course, y_drift_course, x_drift_course, 
                                                                                            dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine'], roi['intensity_mean']])

        nuc_rois = pd.DataFrame(nuc_rois, columns=['orig_position','position', 'orig_pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 'roi_slice', 'pad', 'z_px_course', 'y_px_course', 'x_px_course', 
                                                                                                'z_px_fine', 'y_px_fine', 'x_px_fine', 'nuc_class'])
        nuc_rois['nuc_class'] = nuc_rois['nuc_class'].round()
        self.nuc_rois = nuc_rois
        self.nuc_rois.to_csv(self.image_handler.out_path+'nuc_table.csv')
                
                
    def extract_single_roi_img(self, single_roi, images):
        # Function for extracting a single cropped region defined by ROI from a larger 3D image.

        z, y, x = single_roi['roi_slice']
        pad = single_roi['pad']

        try:
            roi_img = np.array(images[z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros((z,y,x), dtype=np.float32)

        return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def gen_single_nuc_images(self):
        #Function to extract single nuclei images from full FOVs
        # A bit convoluted currently to ensure efficient loading of appropriate arrays into RAM for processing.
        rois = self.nuc_rois#.iloc[0:500]
        P = max(rois['pos_index'])+1
        T = max(rois['frame'])+1

        roi_array = {}
        for pos in tqdm(range(P), total = P):
            for frame in range(T):
                rois_stack = rois.query('pos_index == @pos & frame == @frame')
                #print(rois_stack)
                #roi_ch = int(rois['ch'].unique())
                image_stack = np.array(self.images[pos][frame])
                #print(image_stack.shape)
                for j, single_roi in rois_stack.iterrows():
                    id = single_roi['roi_id']
                    t = single_roi['frame']

                    #TODO: channels hardcoded now, implement as config setting.
                    roi = np.stack([self.extract_single_roi_img(single_roi, image_stack[c]).astype(np.uint16) for c in (0,1)])

                    roi_array[id, t] = roi
                    #roi_array_padded.append(ip.pad_to_shape(roi, shape = roi_image_size, mode = 'minimum'))
        
        #print(roi_array.keys())
        
            #pos_rois = []
        
            for roi_id in rois_stack.roi_id.to_list():
                try:
                    pos_rois = np.stack([roi_array[(roi_id, frame)] for frame in range(T)])
                    #print('Made pos roi ', roi_id, pos_rois.shape)
                except KeyError:
                    break
                image_io.single_position_to_zarr(images=pos_rois, 
                            path=self.config['image_path']+os.sep+'single_nucs', 
                            name='single_nucs',
                            pos_name = 'P'+str(pos+1).zfill(4)+'_'+str(roi_id+1).zfill(4), 
                            axes=('t','c','z','y','x'), 
                            chunk_axes = ('t','z','y','x'),
                            dtype = np.uint16, 
                            chunk_split=(1,1,1,1))
            print('ROI images generated, please reinitialize to load them.')
