# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from dataclasses import dataclass
import json
import logging
from operator import itemgetter
import os
from pathlib import Path
from typing import *

import dask.array as da
from expression import Option, Result, option, result
import numpy as np
from numpydoc_decorator import doc
import pandas as pd
import yaml
from gertils import ExtantFile
from gertils.types import TimepointFrom0

from looptrace import (
    FIELD_OF_VIEW_COLUMN, 
    X_CENTER_COLNAME,
    Y_CENTER_COLNAME, 
    Z_CENTER_COLNAME, 
    ZARR_CONVERSIONS_KEY, 
    ConfigurationValueError, 
    RoiImageSize,
)
from looptrace.configuration import IMAGING_ROUNDS_KEY, TRACING_SUPPORT_EXCLUSIONS_KEY, get_minimum_regional_spot_separation
from looptrace.filepaths import SPOT_IMAGES_SUBFOLDER, FilePathLike, FolderPathLike, get_analysis_path, simplify_path
from looptrace.geometry import Point3D
from looptrace.image_io import ignore_path, NPZ_wrapper
from looptrace.image_processing_functions import CENTROID_KEY
from looptrace.numeric_types import NumberLike
from looptrace.utilities import read_csv_maybe_empty

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

__all__ = ["ImageHandler", "read_images"]

logger = logging.getLogger()

Times: TypeAlias = set[TimepointFrom0]
LocusGroupingData: TypeAlias = dict[TimepointFrom0, Times]
PathFilter: TypeAlias = Callable[[Union[os.DirEntry, Path]], bool]
ImageRound: TypeAlias = Mapping[str, object]

_BACKGROUND_SUBTRACTION_TIMEPOINT_KEY = "subtract_background"


def _get_bead_timepoint_for_spot_filtration(params_config: Mapping[str, object]) -> Result[Option[int], str]:
    bead_spot_filtration_key: Literal["proximityFiltrationBetweenBeadsAndSpots"] = "proximityFiltrationBetweenBeadsAndSpots"
    match params_config.get(bead_spot_filtration_key):
        case None:
            return Result.Error(f"Configuration is missing key for bead-to-spot proximity-based filtration: {bead_spot_filtration_key}")
        case False:
            return Result.Error(f"Bead-to-spot proximity-based filtration key ({bead_spot_filtration_key}) is set to False")
        case True:
            return Option\
                .of_optional(params_config.get(_BACKGROUND_SUBTRACTION_TIMEPOINT_KEY))\
                .to_result(
                    f"Bead spot filtration key ({bead_spot_filtration_key}) is True, but backgrond subtraction timepoint key ({_BACKGROUND_SUBTRACTION_TIMEPOINT_KEY}) is absent"
                )\
                .map(Option.Some)
        case int(t):
            return Result.Ok(Option.Some(t))
        case obj:
            return Result.Error(
                f"Bead-to-spot proximity-based filtration key ({bead_spot_filtration_key}) has value of illegal type: {type(obj).__name__}"
            )


def _invalidate_beads_timepoint_for_spot_filtration(
    *, 
    beads_timepoint: int, 
    rounds: Iterable[ImageRound],
) -> Result[int, str]:
    for r in rounds:
        if r["time"] == beads_timepoint:
            if r.get("isBlank", False):
                return Result.Ok(beads_timepoint)
            return Result.Error(f"The imaging round for beads timepoint {beads_timepoint} isn't tagged as being blank")
    return Result.Error(f"No imaging round corresponds to the given beads timepoint ({beads_timepoint})")


def determine_bead_timepoint_for_spot_filtration(
    *, 
    params_config: Mapping[str, object], 
    image_rounds: Iterable[ImageRound],
) -> Result[int, ConfigurationValueError]:
    return _get_bead_timepoint_for_spot_filtration(params_config)\
        .bind(lambda maybe_time: maybe_time.to_result("Could not get timepoint for filtration of spots by bead proximity"))\
        .bind(lambda t: _invalidate_beads_timepoint_for_spot_filtration(beads_timepoint=t, rounds=image_rounds))


@doc(
    summary="Store the parameters which uniquely specify the name of a file for fiducial bead ROIs.",
    parameters=dict(
        fov="0-based index of field of view",
        timepoint="0-based index of timepoint in imaging sequence",
        purpose="What the ROIs will be used for; set to null if these are generic ROIs",
    ),
    raises=dict(
        TypeError="If fov or timepoint is non-int, or if purpose is non-str",
        ValueError="If fov or timepoint is negative, or if purpose contains a period"
    ),
)
@dataclass(frozen=True, kw_only=True, order=True)
class BeadRoisFilenameSpecification:
    fov: int
    timepoint: int
    purpose: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.fov, int) or not isinstance(self.timepoint, int):
            raise TypeError(f"For bead ROIs filename spec, FOV and timepoint must be int; got {type(self.fov).__name__} and {type(self.timepoint).__name__}")
        if self.fov < 0 or self.timepoint < 0:
            raise ValueError(f"fov and timepoint must be nonnegative; got {self.fov} and {self.timepoint}")
        if self.purpose is not None:
            if not isinstance(self.purpose, str):
                raise TypeError(f"For bead ROIs filename spec, purpose must be null or str, not {type(self.purpose).__name__}")
            if "." in self.purpose:
                raise ValueError(f"In bead ROIs filename specification, purpose can't contain a period; got: {self.purpose}")
    
    prefix = "bead_rois__"

    @property
    def _suffix(self) -> str:
        return ".csv" if self.purpose is None else f".{self.purpose}.json"

    @property
    def get_filename(self) -> str:
        base = self.prefix + str(self.fov) + "_" + str(self.timepoint)
        return base + self._suffix
    
    @classmethod
    def from_filename(cls, fn: str) -> Optional["BeadRoisFilenameSpecification"]:
        fields = fn.split(".")
        if len(fields) == 2:
            if fields[1] != "csv":
                return None
            base = fields[0]
            purpose = None
        elif len(fields) == 3:
            if fields[2] != "json":
                return None
            base = fields[0]
            purpose = None
        else:
            return None
        if not base.startswith(cls.prefix):
            return None
        data = base.removeprefix(cls.prefix).split("_")
        if len(data) != 2:
            return None
        try:
            fov = int(data[0])
            timepoint = int(data[1])
        except:
            return None
        return cls(fov=fov, timepoint=timepoint, purpose=purpose)

    @classmethod
    def from_filepath(cls, fp: Path) -> Optional["BeadRoisFilenameSpecification"]:
        return cls.from_filename(fp.name)


def bead_rois_filename(fov_idx: int, timepoint: int, purpose: Optional[str]) -> str:
    return BeadRoisFilenameSpecification(fov=fov_idx, timepoint=timepoint, purpose=purpose).get_filename


def _read_bead_rois_file(fp: ExtantFile) -> Iterable[tuple[int, Point3D]]:
    with open(fp, "r") as fh:
        for obj in json.load(fh):
            i = obj["index"]
            centroid = obj[CENTROID_KEY]
            p = Point3D(
                x=centroid[X_CENTER_COLNAME], 
                y=centroid[Y_CENTER_COLNAME], 
                z=centroid[Z_CENTER_COLNAME],
            )
            yield i, p


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
        if set(self.config.keys()) & set(rounds.keys()):
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
    def background_subtraction_timepoint(self) -> Optional[int]:
        return self.config.get(_BACKGROUND_SUBTRACTION_TIMEPOINT_KEY)
    
    @property
    def bead_rois_path(self) -> Path:
        return Path(self.analysis_path) / "bead_rois"

    @property
    def bead_timepoint_for_spot_filtration(self) -> TimepointFrom0:
        match determine_bead_timepoint_for_spot_filtration(
            params_config=self.config, 
            image_rounds=self.iter_imaging_rounds(),
        ):
            case result.Result(tag="ok", ok=t):
                return TimepointFrom0(t)
            case result.Result(tag="error", error=err):
                raise err
            case outcome:
                raise Exception(f"Unexpected outcome of determination of bead timepoint for spot filtration: {outcome}")

    def get_bead_rois_file(self, fov_idx: int, timepoint: int, purpose: Optional[str]) -> ExtantFile:
        filename = bead_rois_filename(fov_idx=fov_idx, timepoint=timepoint, purpose=purpose)
        folder = self.bead_rois_path if purpose is None else self.bead_rois_path / purpose
        return ExtantFile(folder / filename)

    def read_bead_rois_file_accuracy(self, fov_idx: int, timepoint: int) -> Iterable[tuple[int, Point3D]]:
        fp = self.get_bead_rois_file(fov_idx=fov_idx, timepoint=timepoint, purpose="accuracy")
        return _read_bead_rois_file(fp.path)

    def read_bead_rois_file_shifting(self, fov_idx: int, timepoint: int) -> Iterable[tuple[int, Point3D]]:
        fp = self.get_bead_rois_file(fov_idx=fov_idx, timepoint=timepoint, purpose="shifting")
        return _read_bead_rois_file(fp.path)
    
    @property
    def _severe_bead_roi_partition_problems_file(self) -> Optional[ExtantFile]:
        fp = self.bead_rois_path / "roi_partition_warnings.severe.json"
        return ExtantFile(fp) if fp.exists() else None

    @property
    def fov_timepoint_pairs_with_severe_problems(self) -> Set[Tuple[int, int]]:
        fp = self._severe_bead_roi_partition_problems_file
        if fp is None:
            return set()
        with open(fp.path, 'r') as fh:
            data = json.load(fh)
        return {(obj[FIELD_OF_VIEW_COLUMN], obj["time"]) for obj in data}

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
    def drift_correction_fov_names(self) -> List[str]:
        return self.image_lists[self.reg_input_moving]

    @property
    def drift_correction_reference_channel(self) -> int:
        return self.config["reg_ch_template"]

    @property
    def drift_correction_reference_timepoint(self) -> int:
        return self.config["reg_ref_timepoint"]

    @property
    def drift_correction_reference_images(self) -> Sequence[np.ndarray]:
        return self.images[self.reg_input_template]

    @property
    def fish_spots_folder(self) -> Path:
        return Path(self.analysis_path) / "fish_spots"

    def get_dc_filepath(self, prefix: str, suffix: str) -> Path:
        return Path(self.out_path(prefix + "_drift_correction" + suffix))

    def get_locus_timepoints_for_regional_timepoint(self, regional_timepoint: TimepointFrom0) -> Times:
        if not isinstance(regional_timepoint, TimepointFrom0):
            raise TypeError(f"Illegal type ({type(regional_timepoint).__name__}) for regional timepoint for which to lookup locus timepoints!")
        grouping: Optional[LocusGroupingData] = self.locus_grouping
        if grouping is None:
            raise ConfigurationValueError("No locus grouping present!")
        return self.locus_grouping.get(regional_timepoint, set())

    def iter_imaging_rounds(self) -> Iterable[ImageRound]:
        return sorted(self.config[IMAGING_ROUNDS_KEY], key=lambda r: r["time"])

    def list_all_regional_timepoints(self) -> list[TimepointFrom0]:
        return list(sorted(TimepointFrom0(r["time"]) for r in self.iter_imaging_rounds() if r.get("isRegional", False)))

    def list_locus_specific_imaging_timepoints_eligible_for_extraction(self) -> list[TimepointFrom0]:
        match Option.of_optional(self.locus_grouping):
            case option.Option(tag="none", none=_):
                return [TimepointFrom0(r["time"]) for r in self.iter_imaging_rounds() if not r.get("isRegional", False)]
            case option.Option(tag="some", some=grouping):
                return list(sorted(t for ts in grouping.values() for t in ts))

    def list_regional_imaging_timepoints_eligible_for_extraction(self) -> list[TimepointFrom0]:
        lg = self.locus_grouping
        if lg is None:
            logging.info("Null locus grouping, all regional timepoints are eligible")
            return self.list_all_regional_timepoints()
        else:
            logging.info("Using locus grouping to determine eligible regional timepoints")
            return list(sorted(lg.keys()))

    @property
    def locus_grouping(self) -> Optional[LocusGroupingData]:
        section_key = "locusGrouping"
        result: Optional[LocusGroupingData] = None
        try:
            data = self.config[section_key]
        except KeyError:
            logging.warning("Did not find locus grouping section key ('%s') in config data", section_key)
        else:
            result = {}
            for reg_time, locus_times in data.items():
                # Put the raw timepoint values into their semantic (and domain-narrowing) wrapper.
                curr: Times = {TimepointFrom0(t) for t in locus_times}
                if len(curr) != len(locus_times):
                    raise ValueError(f"Repetition is present in locus times for regional time {reg_time}: {locus_times}")
                try:
                    # The key ("regional timepoint") coming from the JSON parse should be string, but 
                    # we want a domain-specific type.
                    reg_time = TimepointFrom0(int(reg_time))
                except (TypeError, ValueError) as e:
                    logger.exception("Cannot lift alleged regional time into its wrapper type; error: %s", e)
                    raise
                else:
                    result[reg_time] = curr
        match result:
            case None:
                return None
            case _:
                return dict(sorted(result.items(), key=itemgetter(0)))
    
    @property
    def locus_spots_visualisation_folder(self) -> Path:
        return Path(self.analysis_path) / "locus_spots_visualisation"

    @property
    def minimum_spot_separation(self) -> Union[int, float]:
        return get_minimum_regional_spot_separation(self.config)

    @property
    def nanometers_per_pixel_xy(self) -> NumberLike:
        return self.config["xy_nm"]
    
    @property
    def nanometers_per_pixel_z(self) -> NumberLike:
        return self.config["z_nm"]

    @property
    def nuclear_masks_visualisation_data_path(self) -> Path:
        # Where to save data relevant to visualising nuclear masks.
        return Path(self.analysis_path) / "nuclear_masks_visualisation"

    @property
    def nuclei_channel(self) -> int:
        return self.config["nuc_channel"]

    @property
    def nuclei_coarse_drift_correction_file(self) -> Path:
        return self.get_dc_filepath(prefix="nuclei", suffix="_coarse.csv")

    @property
    def nuclei_filtered_spots_file_path(self) -> Path:
        return self.proximity_accepted_spots_file_path.with_suffix(".nuclei_filtered.csv")

    @property
    def nuclei_labeled_spots_file_path(self) -> Path:
        return self.proximity_accepted_spots_file_path.with_suffix(".nuclei_labeled.csv")

    @property
    def num_bead_rois_for_drift_correction(self) -> int:
        return self.config["num_bead_rois_for_drift_correction"]

    @property
    def num_bead_rois_for_drift_correction_accuracy(self) -> int:
        return self.config["num_bead_rois_for_drift_correction_accuracy"]

    @property
    def num_rounds(self) -> int:
        n1 = self.config.get("num_rounds")
        if n1 is None: # no timepoint count defined, try counting names
            return len(self.timepoint_names)
        try:
            n2 = len(self.timepoint_names)
        except KeyError: # no timepoint names defined, use parsed count
            return n1
        if n1 == n2:
            return n1
        raise Exception(f"Declared timepoint count ({n1}) from config disagrees with timepoint name count ({n2})")

    @property
    def num_timepoints(self) -> int:
        return self.num_rounds

    def out_path(self, fn_extra: str) -> str:
        return os.path.join(self.analysis_path, self.analysis_filename_prefix + fn_extra)

    @property
    def pre_merge_spots_file(self) -> Path:
        """Annotated with how/what will be merged, but not executed"""
        return self.raw_spots_file.with_suffix(".merge_determined.csv")

    @property
    def proximity_accepted_spots_file_path(self) -> Path:
        return self.raw_spots_file.with_suffix(".proximity_accepted.csv")

    @property
    def proximity_rejected_spots_file_path(self) -> Path:
        return self.raw_spots_file.with_suffix(".proximity_rejected.csv")

    @property
    def raw_spots_file(self) -> Path:
        return Path(self.out_path(self.spot_input_name + "_rois" + ".csv"))

    @property
    def regional_spots_visualisation_data_path(self) -> Path:
        return Path(self.analysis_path) / "regional_spots_visualisation"

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
    def rois_with_trace_ids_file(self) -> Path:
        return self.raw_spots_file.with_suffix(".with_trace_ids.csv")

    @property
    def spots_prefiltered_through_nuclei_file(self) -> Path:
        return Path(str(self.raw_spots_file).replace(".csv", ".prefiltered_through_nuclei.csv"))

    @property
    def spot_in_nuc(self) -> bool:
        return self.config.get("spot_in_nuc", False)

    @property
    def spots_fine_drift_correction_table(self) -> pd.DataFrame:
        return self.tables[self.spot_input_name + "_drift_correction_fine"]

    @property
    def spot_fits_file(self) -> Path:
        """Path to the file of the raw fits, before pairing back to ROIs with pair_rois_with_fits"""
        return Path(self.out_path("spot_fits.csv"))

    @property
    def spot_image_extraction_skip_reasons_json_file(self) -> Path:
        return Path(self.out_path("_spot_image_extraction_skip_reasons.json"))

    @property
    def spot_images(self) -> Iterable[da.Array]:
        return self.images[self.spot_input_name]

    @property
    def spot_input_name(self) -> str:
        return self.config["spot_input_name"]

    @property
    def spot_merge_contributors_file(self) -> Path:
        """Annotated with how/what will be merged, but not executed"""
        return self.raw_spots_file.with_suffix(".merge_contributors.csv")

    @property
    def spot_merge_results_file(self) -> Path:
        """Annotated with how/what will be merged, but not executed"""
        return self.raw_spots_file.with_suffix(".post_merge.csv")

    @property
    def spots_for_voxels_definition_file(self) -> Path:
        """Path to the file to use for defining the voxels for tracing"""
        return self.nuclei_filtered_spots_file_path if self.spot_in_nuc else self.rois_with_trace_ids_file

    @property
    def timepoint_names(self) -> List[str]:
        """The sequence of names corresponding to the imaging rounds used in the experiment"""
        names = []
        for round in self.iter_imaging_rounds():
            try:
                names.append(round["name"])
            except KeyError:
                probe = round["probe"]
                rep = round.get("repeat")
                n = probe + ("" if rep is None else f"_repeat{rep}")
                names.append(n)
        return names

    @property
    def trace_id_assignment_skipped_rois_file(self) -> Path:
        """File to write records of ROIs skipped for trace ID assignment (and not, then, present in downstream analysis)"""
        return self.rois_with_trace_ids_file.with_suffix(".skips.json")

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
    def tracing_exclusions(self) -> set[TimepointFrom0]:
        return set(map(TimepointFrom0, self.config.get(TRACING_SUPPORT_EXCLUSIONS_KEY, [])))

    @property
    def zarr_conversions(self) -> Mapping[str, str]:
        return self.config.get(ZARR_CONVERSIONS_KEY, dict())

    def load_tables(self):
        # TODO: the CSV parse needs to depend on whether the first column really is the index or not.
        # See: https://github.com/gerlichlab/looptrace/issues/261
        parsers = {".csv": read_csv_maybe_empty, ".pkl": pd.read_pickle}
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
                logger.debug(f"Cannot parse table name from filename: {fp.name}")
                continue
            fp = fp.path
            if ignore_path(fp):
                logger.debug(f"Not eligible as table: {fp}")
                continue
            try:
                parse = parsers[os.path.splitext(fp)[1]]
            except KeyError:
                logger.debug(f"Cannot load as table: {fp}")
                continue
            logger.info(f"Loading table '{tn}': {fp}")
            self.tables[tn] = parse(fp)
            logger.info(f"Loaded: {tn}")
            self.table_paths[tn] = fp
    
    def read_images(self, is_eligible: PathFilter = lambda p: p.name != SPOT_IMAGES_SUBFOLDER and not ignore_path(p)):
        '''
        Function to load existing images from the input folder, and then into a dictionary (self.images{}),
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
            elif sample_ext in (".tif", ".tiff"):
                raise NotImplementedError(
                    f"Parsing TIFF-like isn't supported! Found extension '{sample_ext}' in folder {images_folder}"
                    )
            else:
                from .image_io import multi_ome_zarr_to_dask
                parse = multi_ome_zarr_to_dask
            arrays, pos_names = parse(images_folder)
            # Now parsed, sort the parallel collections by the FOV names, then store them.
            try:
                arrays, pos_names = zip(*sorted(zip(arrays, pos_names), key=itemgetter(1)))
            except ValueError:
                logging.warning(f"Empty images folder ({images_folder}) perhaps? Contents: {', '.join(os.listdir(images_folder))}")
            else:
                images[image_name] = arrays
                image_lists[image_name] = pos_names
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
