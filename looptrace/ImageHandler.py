# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import os
from pathlib import Path
from typing import *
import numpy as np
import pandas as pd
import yaml
from looptrace.image_io import NPZ_wrapper, TIFF_EXTENSIONS, multi_ome_zarr_to_dask, stack_nd2_to_dask, stack_tif_to_dask


class ImageHandler:
    def __init__(self, config_path: Union[str, Path], image_path = None, image_save_path = None):
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

        self.image_save_path = image_save_path if image_save_path is not None else self.image_path
        
        self.out_path = os.path.join(self.config['analysis_path'], self.config['analysis_prefix'])

        self.load_tables()

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

    def read_images(self, is_eligible: Callable[[str], bool] = lambda path_name: path_name != "spot_images_dir" and not path_name.startswith("_")):
        '''
        Function to load existing images from the input folder, and them into a dictionary (self.images{}),
        with folder name or image name (without extensions) as keys, images as values.
        Standardized to either folders with OME-ZARR, single NPY files or NPZ collections.
        More can be added as needed.
        '''
        image_paths = ((p.name, p.path) for p in os.scandir(self.image_path) if is_eligible(p.name))
        self.images, self.image_lists = read_images(image_paths)

    def reload_config(self):
        with open(self.config_path, 'r') as fh:
            self.config = yaml.safe_load(fh)


def read_images(image_name_path_pairs: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images, image_lists = {}, {}
    for image_name, image_path in image_name_path_pairs:
        if os.path.isdir(image_path):
            exts = set(os.path.splitext(fn)[1] for fn in os.listdir(image_path))
            if len(exts) == 0:
                continue
            if len(exts) != 1:
                print(f"WARNING -- multiple ({len(exts)}) extensions found in folder {image_path}: {', '.join(exts)}")
            sample_ext = list(exts)[0]
            if sample_ext == '.nd2':
                parse = stack_nd2_to_dask
            elif sample_ext in TIFF_EXTENSIONS:
                parse = stack_tif_to_dask
            else:
                parse = multi_ome_zarr_to_dask
            images[image_name], image_lists[image_name] = parse(image_path)
            print('Loaded images: ', image_name)
        elif image_name.endswith('.npz'):
            images[os.path.splitext(image_name)[0]] = NPZ_wrapper(image_path)
            print('Loaded images: ', image_name)
        elif image_name.endswith('.npy'):
            try:
                images[os.path.splitext(image_name)[0]] = np.load(image_path, mmap_mode = 'r')
            except ValueError: #This is for legacy datasets, will be removed after dataset cleanup!
                images[os.path.splitext(image_name)[0]] = np.load(image_path, allow_pickle = True)
            print('Loaded images: ', image_name)
        else:
            print(f"WARNING -- cannot process image path: {image_path}")
    return images, image_lists
