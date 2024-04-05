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

from looptrace import ZARR_CONVERSIONS_KEY, RoiImageSize, read_table_pandas
from looptrace.configuration import get_minimum_regional_spot_separation
from looptrace.filepaths import SPOT_IMAGES_SUBFOLDER, FilePathLike, FolderPathLike, get_analysis_path, simplify_path
from looptrace.image_io import ignore_path, NPZ_wrapper
from gertils import ExtantFile

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

__all__ = ["ImageHandler", "read_images"]

logger = logging.getLogger()

PathFilter = Callable[[Union[os.DirEntry, Path]], bool]


def bead_rois_filename(pos_idx: int, frame: int, purpose: Optional[str]) -> str:
    prefix = f"bead_rois__{pos_idx}_{frame}"
    extension = ".csv" if purpose is None else f".{purpose}.json"
    return prefix + extension


def _read_bead_rois_file(fp: ExtantFile) -> np.ndarray[int]:
    with open(fp, "r") as fh:
        data = json.load(fh)
    return np.round(np.array(list(map(lambda obj: np.array(obj["centroid"]), data)))).astype(int)


class ImageHandler:
    def __init__(
            self, 
            rounds_config: FilePathLike, 
            params_config: FilePathLike, 
            images_folder: Optional[FolderPathLike] = None, 
            image_save_path: Optional[FolderPathLike] = None, 
            strict_load_tables: bool = True,
            ):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        self._strict_load_tables = strict_load_tables
        self.rounds_config = simplify_path(rounds_config)
        self.params_config = simplify_path(params_config)
        
        print(f"Loading parameters config file: {self.params_config}")
        with open(self.params_config, "r") as fh:
            self.config = yaml.safe_load(fh)
        
        print(f"Loading imaging config file: {self.rounds_config}")
        with open(self.rounds_config, "r") as fh:
            rounds = json.load(fh)

        # Update the overall config with what's parsed from the imaging rounds one.
        if set(self.config) & set(rounds):
            raise ValueError(
                f"Overlap of keys from parameters config and imaging config: {', '.join(set(self.config) & set(rounds))}"
                )
        self.config.update(rounds)

        self.images_folder = simplify_path(images_folder)
        if self.images_folder is not None:
            self.read_images()
        self.image_save_path = simplify_path(image_save_path or self.images_folder)
        self.load_tables()

    @property
    def analysis_filename_prefix(self) -> str:
        return self.config["analysis_prefix"]

    @property
    def analysis_path(self) -> str:
        return get_analysis_path(self.config)

    @property
    def background_subtraction_frame(self) -> Optional[int]:
        return self.config.get("subtract_background")
    
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
    def _severe_bead_roi_partition_problems_file(self) -> Optional[ExtantFile]:
        fp = self.bead_rois_path / "roi_partition_warnings.severe.json"
        return ExtantFile(fp) if fp.exists() else None

    @property
    def position_frame_pairs_with_severe_problems(self) -> Set[Tuple[int, int]]:
        fp = self._severe_bead_roi_partition_problems_file
        if fp is None:
            return set()
        with open(fp.path, 'r') as fh:
            data = json.load(fh)
        return {(obj["position"], obj["time"]) for obj in data}

    @property
    def decon_input_name(self) -> str:
        return self.config["decon_input_name"]
    
    @property
    def decon_output_name(self) -> str:
        return self.config["decon_output_name"]

    @property
    def decon_output_path(self) -> Optional[str]:
        return os.path.join(self.image_save_path, self.decon_output_name)

    @property
    def drift_corrected_all_timepoints_rois_file(self) -> Path:
        return Path(self.out_path(self.spot_input_name + "_dc_rois" + ".csv"))

    @property
    def drift_correction_file__coarse(self) -> Path:
        return self.get_dc_filepath(prefix=self.reg_input_moving, suffix="_coarse.csv")

    @property
    def drift_correction_file__fine(self) -> Path:
        return self.get_dc_filepath(prefix=self.reg_input_moving, suffix="_fine.csv")

    @property
    def drift_correction_moving_channel(self) -> int:
        return self.config["reg_ch_moving"]

    @property
    def drift_correction_moving_images(self) -> Sequence[np.ndarray]:
        return self.images[self.reg_input_moving]

    @property
    def drift_correction_position_names(self) -> List[str]:
        return self.image_lists[self.reg_input_moving]

    @property
    def drift_correction_reference_channel(self) -> int:
        return self.config["reg_ch_template"]

    @property
    def drift_correction_reference_frame(self) -> int:
        return self.config["reg_ref_frame"]

    @property
    def drift_correction_reference_images(self) -> Sequence[np.ndarray]:
        return self.images[self.reg_input_template]

    def get_dc_filepath(self, prefix: str, suffix: str) -> Path:
        return Path(self.out_path(prefix + "_drift_correction" + suffix))

    @property
    def frame_names(self) -> List[str]:
        """The sequence of names corresponding to the imaging rounds used in the experiment"""
        names = []
        for round in self.config["imagingRounds"]:
            try:
                names.append(round["name"])
            except KeyError:
                probe = round["probe"]
                rep = round.get("repeat")
                n = probe + ("" if rep is None else f"_repeat{rep}")
                names.append(n)
        return names

    @property
    def locus_spot_images_root_path(self) -> Path:
        return Path(self.analysis_path) / "locus_spot_images"

    @property
    def minimum_spot_separation(self) -> Union[int, float]:
        return get_minimum_regional_spot_separation(self.config)

    @property
    def nuclear_masks_visualisation_data_path(self) -> Path:
        # Where to save data relevant to visualising nuclear masks.
        return Path(self.analysis_path) / "nuclear_masks_visualisation"

    @property
    def nuclei_channel(self) -> int:
        return self.config["nuc_channel"]

    @property
    def nuclei_filtered_spots_file_path(self) -> Path:
        return self.proximity_filtered_spots_file_path.with_suffix(".nuclei_filtered.csv")

    @property
    def nuclei_labeled_spots_file_path(self) -> Path:
        return self.proximity_filtered_spots_file_path.with_suffix(".nuclei_labeled.csv")

    @property
    def num_bead_rois_for_drift_correction(self) -> int:
        return self.config["num_bead_rois_for_drift_correction"]

    @property
    def num_bead_rois_for_drift_correction_accuracy(self) -> int:
        return self.config["num_bead_rois_for_drift_correction_accuracy"]

    @property
    def num_rounds(self) -> int:
        n1 = self.config.get("num_rounds")
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
    def num_timepoints(self) -> int:
        return self.num_rounds

    def out_path(self, fn_extra: str) -> str:
        return os.path.join(self.analysis_path, self.analysis_filename_prefix + fn_extra)

    @property
    def proximity_filtered_spots_file_path(self) -> Path:
        return self.raw_spots_file.with_suffix(".proximity_filtered.csv")

    @property
    def proximity_labeled_spots_file_path(self) -> Path:
        return self.raw_spots_file.with_suffix(".proximity_labeled.csv")

    @property
    def raw_spots_file(self) -> Path:
        return Path(self.out_path(self.spot_input_name + "_rois" + ".csv"))

    @property
    def reg_input_template(self) -> str:
        return self.config["reg_input_template"]

    @property
    def reg_input_moving(self) -> str:
        return self.config["reg_input_moving"]

    @property
    def roi_image_size(self) -> RoiImageSize:
        z, y, x = tuple(self.config["roi_image_size"])
        return RoiImageSize(z=z, y=y, x=x)

    @property
    def spot_image_extraction_skip_reasons_json_file(self) -> Path:
        return Path(self.out_path("_spot_image_extraction_skip_reasons.json"))

    @property
    def spot_input_name(self) -> str:
        return self.config["spot_input_name"]

    @property
    def traces_path(self) -> Path:
        # Written by Tracer.py
        return Path(self.out_path("traces.csv"))

    @property
    def traces_path_enriched(self) -> Path:
        # Written by Tracer.py, consumed by the label-and-filter traces QC program.
        # Still contains things like blank timepoints and regional barcode timepoints
        return Path(self.traces_path).with_suffix(".enriched.csv")
    
    @property
    def traces_file_qc_filtered(self) -> Path:
        # Written by the label-and-filter traces QC program, consumed by the spots plotter
        # Should not contain things like blank timepoints and regional barcode timepoints
        return self.traces_path.with_suffix(".enriched.filtered.csv")

    @property
    def traces_file_qc_unfiltered(self) -> Path:
        # Written by the label-and-filter traces QC program, consumed by the spots plotter
        # Should not contain things like blank timepoints and regional barcode timepoints
        return self.traces_path.with_suffix(".enriched.unfiltered.csv")

    @property
    def zarr_conversions(self) -> Mapping[str, str]:
        return self.config.get(ZARR_CONVERSIONS_KEY, dict())

    def load_tables(self):
        # TODO: the CSV parse needs to depend on whether the first column really is the index or not.
        # See: https://github.com/gerlichlab/looptrace/issues/261
        parsers = {".csv": read_table_pandas, ".pkl": pd.read_pickle}
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
                tn = os.path.splitext(fp.name)[0].split(self.analysis_filename_prefix)[1]
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
        self.images, self.image_lists = read_images_folder(self.images_folder, is_eligible=is_eligible)


def read_images_folder(folder: Path, is_eligible: PathFilter = lambda _: True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if folder is None:
        raise ValueError(f"To read images folder, a folder must be supplied.")
    print(f"Finding image paths in folder: {folder}")
    images_folders = ((p.name, p.path) for p in os.scandir(folder) if is_eligible(p))
    print(f"Reading images from folder: {folder}")
    return read_images(images_folders)


def read_images(image_name_path_pairs: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images, image_lists = {}, {}
    for image_name, images_folder in image_name_path_pairs:
        print(f"Attempting to read images: {images_folder}...")
        if os.path.isdir(images_folder):
            exts = set(os.path.splitext(fn)[1] for fn in os.listdir(images_folder))
            if len(exts) == 0:
                continue
            if len(exts) != 1:
                print(f"WARNING -- multiple ({len(exts)}) extensions found in folder {images_folder}: {', '.join(exts)}")
            sample_ext = list(exts)[0]
            if sample_ext == '.nd2':
                from .nd2io import stack_nd2_to_dask
                def parse(p):
                    arrays, pos_names, _ = stack_nd2_to_dask(p)
                    return arrays, pos_names
            elif sample_ext in [".tif", ".tiff"]:
                raise NotImplementedError(
                    f"Parsing TIFF-like isn't supported! Found extension '{sample_ext}' in folder {images_folder}"
                    )
            else:
                from .image_io import multi_ome_zarr_to_dask
                parse = multi_ome_zarr_to_dask
            images[image_name], image_lists[image_name] = parse(images_folder)
        elif image_name.endswith('.npz'):
            images[os.path.splitext(image_name)[0]] = NPZ_wrapper(images_folder)
        elif image_name.endswith('.npy'):
            try:
                images[os.path.splitext(image_name)[0]] = np.load(images_folder, mmap_mode = 'r')
            except ValueError: #This is for legacy datasets, will be removed after dataset cleanup!
                images[os.path.splitext(image_name)[0]] = np.load(images_folder, allow_pickle = True)
        else:
            print(f"WARNING -- cannot process image path: {images_folder}")
            continue
        print("Loaded images: ", image_name)
    return images, image_lists
