# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import image_io
import os
import pandas as pd
import numpy as np


class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        
        self.config_path = config_path
        self.reload_config()
        self.image_path = self.config['image_path']
        self.out_path = self.config['output_path']+os.sep+self.config['output_prefix']
        self.dc_file_path = self.out_path+'drift_correction.csv'
        self.roi_file_path = self.out_path+'rois.csv'
        self.traces_path = self.out_path+'traces.csv'

        self.read_images()
        self.load_tables()

    def reload_config(self):
        self.config = image_io.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')

    def load_tables(self):
        self.tables = {}
        self.table_paths = {}
        for f in os.scandir(self.config['output_path']):
            if f.name.endswith('.csv'):
                table_name = os.path.splitext(f.name)[0].split(self.config['output_prefix'])[1]
                print('Loading table ', table_name)
                table = pd.read_csv(f.path, index_col = 0)
                self.tables[table_name] = table
                self.table_paths[table_name] = f.path

    def read_images(self):
        '''
        Function to load existing images from the input folder, and them into a dictionary (self.images{}),
        with folder name or image name (without extensions) as keys, images as values.
        Standardized to either folders with OME-ZARR, single NPY files or NPZ collections.
        More can be added as needed.
        '''
        self.images = {}
        self.image_lists = {}
        image_paths = [(p.name, p.path) for p in os.scandir(self.image_path)]
        for image_name, image_path in image_paths:

            if os.path.isdir(image_path):
                print(image_path)
                self.images[image_name], self.image_lists[image_name] = image_io.multi_ome_zarr_to_dask(image_path, remove_unused_dims = True)
                print('Loaded images: ', image_name)
            elif image_name.endswith('.npz'):
                self.images[os.path.splitext(image_name)[0]] = image_io.NPZ_wrapper(image_path)
                print('Loaded images: ', image_name)
            elif image_name.endswith('.npy'):
                try:
                    self.images[os.path.splitext(image_name)[0]] = np.load(image_path, mmap_mode = 'r')
                except ValueError: #This is for legacy datasets, will be removed after dataset cleanup!
                    self.images[os.path.splitext(image_name)[0]] = np.load(image_path, allow_pickle = True)
                print('Loaded images: ', image_name)