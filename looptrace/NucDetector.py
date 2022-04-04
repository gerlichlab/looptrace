# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import image_processing_functions as ip
from pathlib import Path
import dask.array as da
import os
import numpy as np
import tifffile
from skimage.morphology import dilation, disk

class NucDetector:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''

    def __init__(self, image_handler):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images, self.pos_list = image_handler.images, image_handler.pos_list
        self.cell_images_path = self.image_handler.cell_images_path
        self.cell_images = self.image_handler.cell_images
        print(self.cell_images)
        self.nuc_images_path = self.cell_images_path  + os.sep + 'nuc_images'
        self.nuc_mask_path = self.cell_images_path + os.sep + 'nuc_masks'
        self.nuc_class_path = self.cell_images_path + os.sep + 'nuc_classes'

    def gen_nuc_images(self):
        '''
        Saves 2D max projected images of the nuclear channel into image folder for later analysis.
        '''
        try:
            nuc_slice = self.config['nuc_slice']
        except KeyError: #Legacy config
            nuc_slice = -1

        imgs = []
        for pos in self.pos_list:
            pos_index = self.pos_list.index(pos)
            if nuc_slice == -1:
                img = da.max(self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel']], axis=0).compute()
            else:
                img = self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel'], self.config['nuc_slice']].compute()
            imgs.append(img)
        imgs = np.stack(imgs).astype(np.uint16)
        self.cell_images['nuc_images']= imgs
        ip.imgs_to_ome_zarr(images = imgs, path=self.nuc_images_path, name = 'nuc_images', axes=['p','y','x'])

    def segment_nuclei(self):
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''

        if 'nuc_images' not in self.cell_images:
            print('Generating nuclei images.')
            self.gen_nuc_images()
        
        nuc_imgs = self.cell_images['nuc_images']

        diameter = self.config['nuc_diameter']
        try:
            model = self.config['nuc_model']
        except KeyError:
            model = 'nuclei'
        print(f'Running nuclear segmentation with CellPose using {model} model and diameter {diameter}.')
        masks = ip.nuc_segmentation(nuc_imgs, diameter=diameter, model=model)
        print(f'Detecting mitotic cells on top of CellPose nuclei.')
        masks, mitotic_idx = zip(*[ip.mitotic_cell_extra_seg(np.array(nuc_imgs[i]), masks[i]) for i in range(len(nuc_imgs))])


        masks = [dilation(mask, disk(self.config['nuc_dilation'])) for mask in masks]
        masks = np.stack(masks).astype(np.uint16)

        print('Saving segmentations.')
        ip.imgs_to_ome_zarr(images = masks, path=self.nuc_mask_path, name = 'nuc_masks', axes=['p','y','x'])
        self.image_handler.cell_images['nuc_masks'] = masks

        nuc_class = []
        for i, mask in enumerate(masks):
            class_1 = ((mask > 0) & (mask < mitotic_idx[i])).astype(int)
            class_2 = (mask >= mitotic_idx[i]).astype(int)
            nuc_class.append(class_1 + class_2*2)
        nuc_class = np.stack(nuc_class).astype(np.uint16)
        print('Saving classifications.')
        ip.imgs_to_ome_zarr(images = nuc_class, path=self.nuc_class_path, name = 'nuc_classes', axes=['p','y','x'])
        self.image_handler.cell_images['nuc_classes'] = nuc_class
      


    ## Legacy classification with ilastic.
    # def classify_nuclei(self):
    #     '''
    #     Runs nucleus classification after detection usign pre-trained ilastik model.
    #     Saves classified images.
    #     '''

    #     print('Running classification of nuclei with Ilastik.')
    #     raw_imgs = [str(p) for p in Path(self.nuc_folder).glob('nuc_raw_*.tiff')] #' '.join(
    #     seg_imgs = [str(p) for p in Path(self.nuc_folder).glob('nuc_binary_*.tiff')]
        
    #     ilastik_path = self.config['ilastik_path']
    #     project_path = self.config['ilastik_project_path']
    #     params = f' --headless --project=\"{project_path}\" --export_source=\"Object Predictions\" --output_format=numpy '
    #     for raw_img, seg_img in zip(raw_imgs, seg_imgs):
    #         raw_data = f'--raw_data {raw_img} '
    #         segmentation = f'--segmentation_image {seg_img}'
    #         command = ilastik_path+params+raw_data+segmentation
    #         subprocess.run(command)
    #     nuc_class = [np.load(img) for img in Path(self.nuc_folder).glob('nuc_raw_*_Object*.npy')]
    #     print('Nucleus classification done.')
    #     self.image_handler.nuc_class = nuc_class

        # def save_nucs(self, img_type):
    #     '''
    #     Function to save nuclear images, either raw or the masks, as tiff files in nucs folder.

    #     Args:
    #         img_type ([str]): Type of images to save, can be 'raw', 'mask' or 'class'.
    #     '''
    #     Path(self.nuc_folder).mkdir(parents=True, exist_ok=True)
    #     imgs = []
    #     for pos in self.pos_list:
    #         pos_index = self.pos_list.index(pos)
    #         if img_type=='raw':
    #             img = self.nucs[pos_index]
    #             tifffile.imsave(self.nuc_folder+os.sep+'nuc_raw_'+pos+'.tiff', data=img)
    #         elif img_type=='mask':
    #             img = self.nuc_masks[pos_index]
    #             tifffile.imsave(self.nuc_folder+os.sep+'nuc_labels_'+pos+'.tiff', data=img)
    #         elif img_type=='class':
    #             img = self.nuc_class[pos_index]
    #             np.save(self.nuc_folder+os.sep+'nuc_raw_'+pos+'_Object Predictions.npy', img)