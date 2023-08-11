# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import logging
import os
from pathlib import Path
from typing import *
import numpy as np
import pandas as pd
import yaml

from looptrace.filepaths import SPOT_IMAGES_SUBFOLDER, get_analysis_path, simplify_path
from looptrace.image_io import ignore_path, NPZ_wrapper, TIFF_EXTENSIONS
from gertils import ExtantFile, ExtantFolder

__all__ = ["ImageHandler", "handler_from_cli", "read_images"]

logger = logging.getLogger()


PathFilter = Callable[[Union[os.DirEntry, Path]], bool]


class ImageHandler:
    def __init__(
            self, 
            config_path: Union[str, Path, ExtantFile], 
            image_path: Union[str, Path, ExtantFolder, None] = None, 
            image_save_path: Union[str, Path, ExtantFolder, None] = None
            ):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        self.config_path = simplify_path(config_path)
        self.reload_config()
        self.image_path = simplify_path(image_path)
        if self.image_path is not None:
            self.read_images()
        self.image_save_path = simplify_path(image_save_path if image_save_path is not None else self.image_path)
        self.load_tables()

    @property
    def analysis_filename_prefix(self) -> str:
        return self.config['analysis_prefix']

    @property
    def analysis_path(self):
        return get_analysis_path(self.config)

    @property
    def decon_input_name(self) -> str:
        return self.config['decon_input_name']
    
    @property
    def decon_output_name(self) -> str:
        return self.config.get('decon_output_name', f"{self.decon_input_name}_decon")

    @property
    def decon_output_path(self) -> str:
        return os.path.join(self.image_save_path, self.decon_output_name)

    def out_path(self, fn_extra: str) -> str:
        return os.path.join(self.analysis_path, self.analysis_filename_prefix + fn_extra)

    @property
    def reg_input_template(self) -> str:
        return self.config.get('reg_input_template', self.decon_output_name)

    @property
    def reg_input_moving(self) -> str:
        return self.config.get('reg_input_moving', self.reg_input_template)

    @property
    def spot_input_name(self) -> str:
        return self.config.get('spot_input_name', self.decon_output_name)

    def load_tables(self):
        parsers = {".csv": lambda f: pd.read_csv(f, index_col=0), ".pkl": pd.read_pickle}
        try:
            table_files = os.scandir(self.analysis_path)
        except FileNotFoundError:
            logger.info(f"Declared analysis folder doesn't yet exist: {self.analysis_path}")
            raise
        self.table_paths = {}
        self.tables = {}
        for fp in table_files:
            try:
                tn = os.path.splitext(fp.name)[0].split(self.config['analysis_prefix'])[1]
            except IndexError:
                logger.warning(f"Cannot parse table name from filename: {fp.name}")
                continue
            fp = fp.path
            if ignore_path(fp):
                logger.debug(f"Not eligible as table: {fp}")
                continue
            try:
                parse = parsers[os.path.splitext(fp)[1]]
            except KeyError:
                logger.warning(f"Cannot load as table: {fp}")
                continue
            logger.info(f"Loading table '{tn}': {fp}")
            self.tables[tn] = parse(fp)
            logger.info(f"Loaded: {tn}")
            self.table_paths[tn] = fp
    
    def read_images(self, is_eligible: PathFilter = lambda p: p.name != SPOT_IMAGES_SUBFOLDER and not ignore_path(p)):
        '''
        Function to load existing images from the input folder, and them into a dictionary (self.images{}),
        with folder name or image name (without extensions) as keys, images as values.
        Standardized to either folders with OME-ZARR, single NPY files or NPZ collections.
        More can be added as needed.
        '''
        self.images, self.image_lists = read_images_folder(self.image_path, is_eligible=is_eligible)

    def reload_config(self):
        print(f"Loading config file: {self.config_path}")
        with open(self.config_path, 'r') as fh:
            self.config = yaml.safe_load(fh)
        return self.config


def handler_from_cli(config_file: ExtantFile, images_folder: Optional[ExtantFolder], image_save_path: Optional[ExtantFolder] = None) -> ImageHandler:
    image_path = None if images_folder is None else images_folder.to_string()
    image_save_path = None if image_save_path is None else image_save_path.to_string()
    return ImageHandler(config_path=config_file.to_string(), image_path=image_path, image_save_path=image_save_path)


def read_images_folder(folder: Path, is_eligible: PathFilter = lambda _: True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print(f"Finding image paths in folder: {folder}")
    image_paths = ((p.name, p.path) for p in os.scandir(folder) if is_eligible(p))
    print(f"Reading images from folder: {folder}")
    return read_images(image_paths)


def read_images(image_name_path_pairs: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images, image_lists = {}, {}
    for image_name, image_path in image_name_path_pairs:
        print(f"Attempting to read images: {image_path}...")
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
        elif image_name.endswith('.npz'):
            images[os.path.splitext(image_name)[0]] = NPZ_wrapper(image_path)
        elif image_name.endswith('.npy'):
            try:
                images[os.path.splitext(image_name)[0]] = np.load(image_path, mmap_mode = 'r')
            except ValueError: #This is for legacy datasets, will be removed after dataset cleanup!
                images[os.path.splitext(image_name)[0]] = np.load(image_path, allow_pickle = True)
        else:
            print(f"WARNING -- cannot process image path: {image_path}")
            continue
        print('Loaded images: ', image_name)
    return images, image_lists
