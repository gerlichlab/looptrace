# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import json
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


FolderLike = Union[str, Path, ExtantFolder]
PathFilter = Callable[[Union[os.DirEntry, Path]], bool]


def bead_rois_filename(pos_idx: int, frame: int, purpose: Optional[str]) -> str:
    prefix = f"bead_rois__{pos_idx}_{frame}"
    extension = ".csv" if purpose is None else f".{purpose}.json"
    return prefix + extension


def _read_bead_rois_file(fp: ExtantFile) -> np.ndarray[int]:
    with open(fp, 'r') as fh:
        data = json.load(fh)
    return np.round(np.array(list(map(data, lambda obj: np.array(obj["centroid"]))))).astype(int)


class ImageHandler:
    def __init__(
            self, 
            config_path: Union[str, Path, ExtantFile], 
            image_path: Union[str, Path, ExtantFolder, None] = None, 
            image_save_path: Union[str, Path, ExtantFolder, None] = None, 
            strict_load_tables: bool = True,
            ):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        self._strict_load_tables = strict_load_tables
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
    def bead_rois_path(self) -> Path:
        return Path(self.analysis_path) / "bead_rois"

    def get_bead_rois_file(self, pos_idx: int, frame: int, purpose: Optional[str]) -> ExtantFile:
        filename = bead_rois_filename(pos_idx=pos_idx, frame=frame, purpose=purpose)
        folder = self.bead_rois_path if purpose is None else self.bead_rois_path / purpose
        return ExtantFile(folder / filename)

    def read_bead_rois_file_accuracy(self, pos_idx: int, frame: int) -> np.ndarray[np.ndarray]:
        fp = self.get_bead_rois_file(pos_idx=pos_idx, frame=frame, purpose="accuracy").path
        return _read_bead_rois_file(fp)

    def read_bead_rois_file_shifting(self, pos_idx: int, frame: int) -> ExtantFile:
        fp = self.get_bead_rois_file(pos_idx=pos_idx, frame=frame, purpose="shifting").path
        return _read_bead_rois_file(fp)

    @property
    def decon_input_name(self) -> Optional[str]:
        return self.config.get('decon_input_name')
    
    @property
    def decon_output_name(self) -> Optional[str]:
        return self.config.get('decon_output_name')

    @property
    def decon_output_path(self) -> Optional[str]:
        outname = self.decon_output_name
        return outname and os.path.join(self.image_save_path, outname)

    @property
    def frame_names(self) -> List[str]:
        return self.config["frame_name"]
    
    @property
    def num_bead_rois_for_drift_correction(self) -> int:
        return self.config["bead_points"]

    @property
    def num_bead_rois_for_drift_correction_accuracy(self) -> int:
        return self.config["num_bead_rois_for_drift_correction_accuracy"]

    @property
    def num_frames(self) -> int:
        n1 = self.config.get("num_frames")
        if n1 is None: # no frame count defined, try counting names
            return len(self.frame_names)
        try:
            n2 = len(self.frame_names)
        except KeyError: # no frame names defined, use parsed count
            return n1
        if n1 == n2:
            return n1
        raise Exception(f"Declared frame count ({n1}) from config disagrees with frame name count ({n2})")
    
    @property
    def num_rounds(self) -> int:
        return self.num_frames

    @property
    def num_timepoints(self) -> int:
        return self.num_frames

    def set_analysis_path(self, newpath: FolderLike) -> None:
        """
        Update the value of the image handler's analysis subfolder path.

        Parameters
        ----------
        newpath : str
            The new value (possibly including environment and/or user variables) for the handler's analysis subfolder path
        """
        if isinstance(newpath, ExtantFolder):
            newpath = newpath.path
        self.config["analysis_path"] = str(newpath)

    def out_path(self, fn_extra: str) -> str:
        return os.path.join(self.analysis_path, self.analysis_filename_prefix + fn_extra)

    @property
    def reg_input_template(self) -> str:
        return self.config['reg_input_template']

    @property
    def reg_input_moving(self) -> str:
        return self.config['reg_input_moving']

    @property
    def spot_input_name(self) -> str:
        return self.config['spot_input_name']
    
    @property
    def tolerate_too_few_rois(self) -> bool:
        return self.config.get("tolerate_too_few_rois", False)

    @property
    def zarr_conversions(self) -> Mapping[str, str]:
        return self.config["zarr_conversions"]

    def load_tables(self):
        parsers = {".csv": lambda f: pd.read_csv(f, index_col=0), ".pkl": pd.read_pickle}
        try:
            table_files = os.scandir(self.analysis_path)
        except FileNotFoundError:
            logger.info(f"Declared analysis folder doesn't yet exist: {self.analysis_path}")
            if self._strict_load_tables:
                raise
            table_files = []
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
                def parse(p):
                    arrays, pos_names, _ = stack_nd2_to_dask(p)
                    return arrays, pos_names
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
