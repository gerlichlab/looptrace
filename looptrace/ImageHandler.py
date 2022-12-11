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
    def __init__(self, config_path, image_path = None, image_save_path = None):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        
        self.config_path = config_path
        self.reload_config()

        self.image_path = image_path

        if self.image_path is not None:
            self.read_images()

        if image_save_path is not None:
            self.image_save_path = image_save_path
        else:
            self.image_save_path = self.image_path
        
        self.out_path = self.config['analysis_path']+os.sep+self.config['analysis_prefix']

        self.load_tables()

    def reload_config(self):
        self.config = image_io.load_config(self.config_path)

    def load_tables(self):
        self.tables = {}
        self.table_paths = {}
        for f in os.scandir(self.config['analysis_path']):
            if f.name.endswith('.csv') and not f.name.startswith('_'):
                table_name = os.path.splitext(f.name)[0].split(self.config['analysis_prefix'])[1]
                print('Loading table ', table_name)
                table = pd.read_csv(f.path, index_col = 0)
                self.tables[table_name] = table
                self.table_paths[table_name] = f.path
            elif f.name.endswith('.pkl') and not f.name.startswith('_'):
                table_name = os.path.splitext(f.name)[0].split(self.config['analysis_prefix'])[1]
                print('Loading table ', table_name)
                table = pd.read_pickle(f.path)
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
            if image_name.startswith('_') or image_name == 'spot_images_dir':
                continue
            else:
                if os.path.isdir(image_path):
                    if len(os.listdir(image_path)) == 0:
                        continue
                    else:
                        sample_file = os.listdir(image_path)[0]
                        print(image_path)
                        if sample_file.endswith('.nd2'):
                            self.images[image_name], self.image_lists[image_name] = image_io.stack_nd2_to_dask(image_path)
                            print('Loaded images: ', image_name)
                        elif sample_file.endswith('.tiff') or sample_file.endswith('.tif'):
                            self.images[image_name], self.image_lists[image_name] = image_io.stack_tif_to_dask(image_path)
                            print('Loaded images: ', image_name)
                        else:
                            self.images[image_name], self.image_lists[image_name] = image_io.multi_ome_zarr_to_dask(image_path, remove_unused_dims = False)
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