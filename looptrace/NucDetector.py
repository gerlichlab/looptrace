# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import image_processing_functions as ip
from pathlib import Path
import subprocess
import os
import numpy as np
import tifffile
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk
import dask.array as da

class NucDetector:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''

    def __init__(self, image_handler):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images, self.pos_list = image_handler.images, image_handler.pos_list
        self.nuc_folder = self.image_handler.nuc_folder
        self.image_handler.load_nucs()

    def segment_nuclei(self):
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''

        if not self.image_handler.nucs:
            print('Generating nuclei images.')
            self.image_handler.gen_nuc_images()
        nuc_imgs = self.image_handler.nucs
        masks = ip.nuc_segmentation(nuc_imgs, self.config['nuc_diameter'])
        self.mask_to_binary(masks)

        masks = [dilation(mask, disk(self.config['nuc_dilation'])) for mask in masks]
        self.image_handler.nucs = nuc_imgs
        self.image_handler.nuc_masks = masks
        self.image_handler.save_nucs(img_type='mask')
    
    def classify_nuclei(self):
        '''
        Runs nucleus classification after detection usign pre-trained ilastik model.
        Saves classified images.
        '''

        print('Running classification of nuclei with Ilastik.')
        raw_imgs = [str(p) for p in Path(self.nuc_folder).glob('nuc_raw_*.tiff')] #' '.join(
        seg_imgs = [str(p) for p in Path(self.nuc_folder).glob('nuc_binary_*.tiff')]
        
        ilastik_path = self.config['ilastik_path']
        project_path = self.config['ilastik_project_path']
        params = f' --headless --project=\"{project_path}\" --export_source=\"Object Predictions\" --output_format=numpy '
        for raw_img, seg_img in zip(raw_imgs, seg_imgs):
            raw_data = f'--raw_data {raw_img} '
            segmentation = f'--segmentation_image {seg_img}'
            command = ilastik_path+params+raw_data+segmentation
            subprocess.run(command)
        nuc_class = [np.load(img) for img in Path(self.nuc_folder).glob('nuc_raw_*_Object*.npy')]
        print('Nucleus classification done.')
        self.image_handler.nuc_class = nuc_class
    
    def mask_to_binary(self, masks):
        '''Converts masks from nuclear segmentation to masks with 
        single pixel background between separate, neighbouring features.
        Saves binary version of the masks as tiff.

        Args:
            masks ([np array]): Detected nuclear masks (label image)

        Returns:
            [np array]: Masks with single pixel seperation beteween neighboring features.
        '''
        masks_no_bound = [np.where(find_boundaries(mask)>0, 0, mask) for mask in masks]
        for i, img in enumerate(masks_no_bound):
            tifffile.imsave(self.nuc_folder+os.sep+'nuc_binary_'+self.pos_list[i]+'.tiff', data=((img>0)*255).astype(np.uint8))
        return masks_no_bound