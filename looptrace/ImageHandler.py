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

from looptrace.image_io import NPZ_wrapper, TIFF_EXTENSIONS
from looptrace.pathtools import ExtantFile, ExtantFolder

__all__ = ["ImageHandler", "handler_from_cli", "read_images"]


class ImageHandler:
    def __init__(self, config_path: Union[str, Path], image_path: Optional[str] = None, image_save_path: Optional[str] = None):
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
        get_table_name = lambda f: os.path.splitext(f.name)[0].split(self.config['analysis_prefix'])[1]
        is_eligible = lambda fp: not os.path.split(fp)[1].startswith('_')
        analysis_folder = self.config['analysis_path']
        try:
            table_files = os.scandir(analysis_folder)
        except FileNotFoundError:
            print(f"Declared analysis folder doesn't yet exist: {self.config['analysis_path']}")
            raise
        self.table_paths = {get_table_name(f): f.path for f in table_files}
        self.tables = {}
        for tn, fp in self.table_paths.items():
            if not is_eligible(fp):
                continue
            _, ext = os.path.splitext(fp)
            if ext == '.csv':
                parse = lambda f: pd.read_csv(f, index_col=0)
            elif ext == '.pkl':
                parse = pd.read_pickle
            else:
                print(f"Cannot load as table: {fp}")
                continue
            print(f"Loading table '{tn}': {fp}")
            self.tables[tn] = parse(fp)
            print(f"Loaded: {tn}")

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


def handler_from_cli(config_file: ExtantFile, images_folder: Optional[ExtantFolder], image_save_path: Optional[ExtantFolder] = None) -> ImageHandler:
    image_path = None if images_folder is None else images_folder.as_string()
    image_save_path = None if image_save_path is None else image_save_path.as_string()
    return ImageHandler(config_path=config_file.as_string(), image_path=image_path, image_save_path=image_save_path)


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
                from .image_io import stack_nd2_to_dask
                parse = stack_nd2_to_dask
            elif sample_ext in TIFF_EXTENSIONS:
                from .image_io import stack_tif_to_dask
                parse = stack_tif_to_dask
            else:
                from .image_io import multi_ome_zarr_to_dask
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
