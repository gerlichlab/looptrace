# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from collections import OrderedDict, defaultdict
import copy
import dataclasses
from enum import Enum
import json
import logging
from math import ceil, floor
import os
from pathlib import Path
from typing import *

import dask.array as da
from expression import Option
from joblib import Parallel, delayed
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import tqdm

from gertils import ExtantFolder, NonExtantPath
from gertils.types import TimepointFrom0
import spotfishing
from spotfishing_looptrace import DifferenceOfGaussiansSpecificationForLooptrace, ORIGINAL_LOOPTRACE_DOG_SPECIFICATION

from looptrace import FIELD_OF_VIEW_COLUMN, ArrayDimensionalityError, RoiImageSize, image_processing_functions as ip
from looptrace.filepaths import SPOT_BACKGROUND_SUBFOLDER, SPOT_IMAGES_SUBFOLDER, simplify_path
from looptrace.numeric_types import NumberLike

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

CROSSTALK_SUBTRACTION_KEY = "subtract_crosstalk"
DETECTION_METHOD_KEY = "detection_method"
DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC = "dog"
NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION = ".nuclei_labeled"
NUCLEI_LABELED_SPOTS_FILE_EXTENSION = NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION + ".csv"
NUMBER_OF_DIGITS_FOR_ROI_ID = 5
SPOT_CHANNEL_COLUMN_NAME = "spotChannel"
SPOT_IMAGE_PIXEL_VALUE_TYPE = np.uint16

logger = logging.getLogger()

SkipReasonsMapping = Mapping[int, Mapping[int, Mapping[int, str]]]


def detect_spots_dog(
    input_image, 
    *, 
    threshold: NumberLike, 
    expand_px: Optional[int] = 10,
    transform_specification: DifferenceOfGaussiansSpecificationForLooptrace = ORIGINAL_LOOPTRACE_DOG_SPECIFICATION, 
    ) -> spotfishing.DetectionResult:
    """Spot detection by difference of Gaussians filter"""
    return spotfishing.detect_spots_dog(
        input_image, 
        spot_threshold=threshold, 
        expand_px=expand_px, 
        transform=transform_specification.transformation,
        )


def detect_spots_int(input_image, *, threshold: NumberLike, expand_px: Optional[int] = 1) -> spotfishing.DetectionResult:
    """Spot detection by intensity filter"""
    return spotfishing.detect_spots_int(input_image, spot_threshold=threshold, expand_px=expand_px)


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoxelStackSpecification:
    """
    Bundle of the essential provenance related to a particular stack of voxels.

    More specifically, we extract, for each detected regional spot / ROI, a voxel for each of a 
    number of timepoints from the course of the imaging experiment. For when these voxels stacks 
    are then used for tracing and for locus spot visualisation, we need to know these things:
    1. Field of view
    2. ROI ID
    3. Regional/reference timepoint (i.e., in which timepoint the spot was produced/detected)
    4. Trace group ID (if applicable, i.e. merging ROIs into a bigger tracing structure)
    5. Trace ID
    
    These data items can also be used to sort CSV-like records of locus spots, from which these 
    data may be extracted, as well as the voxels themseleves (e.g., when merging ROIs  for tracing, 
    and needing to put the voxels into a structure by aggregate trace ID for the visualisation).

    These data components also function as a key on which to join trace fits (i.e., parameters of 
    a 3D Gaussian) to the original locus spot ROI record, with things like bounding box information.

    Finally, note that while the trace ID should uniquely determine the value of the trace group, 
    we include this additional piece of data here so that we don't need to parse or otherwise 
    maintain a lookup table between trace ID and trace group name, or between the regional timepoints 
    (which comprise a trace group) and the trace group ID/name.
    """

    field_of_view: str
    roiId: int
    ref_timepoint: int
    traceGroup: Option[str]
    traceId: int

    @property
    def file_name_base(self) -> str:
        return self._get_key_delimiter().join([
            self.field_of_view, 
            str(self.roiId).zfill(NUMBER_OF_DIGITS_FOR_ROI_ID), 
            str(self.ref_timepoint),
            self.traceGroup.default_value(""),
            str(self.traceId),
        ])

    @classmethod
    def from_roi(cls, roi: Union[pd.Series, Mapping[str, Any]]) -> "VoxelStackSpecification":
        return cls(
            field_of_view=roi[FIELD_OF_VIEW_COLUMN], 
            roiId=roi["roiId"], 
            ref_timepoint=roi["ref_timepoint"],
            traceGroup=cls._build_trace_group(roi["traceGroup"]),
            traceId=roi["traceId"],
        )
    
    @classmethod
    def from_file_name_base(cls, file_key: str) -> "VoxelStackSpecification":
        try:
            fov, roi, ref, raw_trace_group, trace = file_key.split(cls._get_key_delimiter())
        except ValueError:
            print(f"Failed to get key for file key: {file_key}")
            raise
        return cls(
            field_of_view=fov, 
            roiId=int(roi), 
            ref_timepoint=int(ref),
            traceGroup=cls._build_trace_group(raw_trace_group),
            traceId=int(trace),
        )

    @property
    def name_roi_file(self) -> str:
        return self.file_name_base + ".npy"
    
    @staticmethod
    def row_order_columns() -> list[str]:
        """What's used to sort the rows of the all-voxel-specifications file, and the traces file."""
        # NB: we don't include the trace group ID here, as it's entirely determined by the ID. 
        return [FIELD_OF_VIEW_COLUMN, "roiId", "ref_timepoint", "traceId"]
    
    @property
    def to_tuple(self) -> Tuple[str, int, int, int]:
        return dataclasses.astuple(self)

    @staticmethod
    def _build_trace_group(raw_value: str) -> Option[str]:
        return Option.Nothing() if raw_value == "" else Option.Some(raw_value)

    @staticmethod
    def _get_key_delimiter() -> str:
        return "_"


def get_locus_spot_row_order_columns() -> list[str]:
    return VoxelStackSpecification.row_order_columns() + ["timepoint"]


def finalise_single_spot_props_table(spot_props: pd.DataFrame, field_of_view: str, timepoint: int, channel: int) -> pd.DataFrame:
    """
    Perform the addition of several context-relevant fields to the table of detected spot properties for a particular image and channel.

    The arguments to this function specify the table to update, as well as the field of view, the hybridisation round / timepoint, 
    and the imaging channel from which the data came that were used to detect spots to which the given properties table corresponds.

    Parameters
    ----------
    spot_props : pd.DataFrame
        Data table with properties (location, intensity, etc.) of detected spots
    field_of_view : str
        Field of view in which spots were detected
    timepoint : int
        Hybridisation round / timepoint for which spots were detected
    channel : int
        Imaging channel in which spots were detected

    Returns
    -------
    pd.DataFrame
        A table annotated with the fields for context (field of view, hybridisation timepoint / round, and imaging channel)
    """
    old_cols = list(spot_props.columns)
    new_cols = [FIELD_OF_VIEW_COLUMN, "timepoint", SPOT_CHANNEL_COLUMN_NAME]
    spot_props[new_cols] = [field_of_view, timepoint, channel]
    return spot_props[new_cols + old_cols]


@dataclasses.dataclass(kw_only=True, frozen=True)
class SpotDetectionParameters:
    """Bundle the parameters which are relevant for spot detection."""
    detection_function: callable
    downsampling: int
    minimum_distance_between: NumberLike
    # TODO: non-nullity requirement for crosstalk_channel is coupled to this condition, and this should be reflected in the types.
    subtract_beads: bool
    crosstalk_channel: Optional[int]
    crosstalk_timepoint: Optional[int]
    roi_image_size: Optional[RoiImageSize]

    def try_centering_spot_box_coordinates(self, spots_table: pd.DataFrame) -> pd.DataFrame:
        if self.roi_image_size is None:
            return spots_table
        dims = self.roi_image_size.div_by(self.downsampling)
        return roi_center_to_bbox(spots_table, roi_size=dims)


def compute_downsampled_image(full_image: da.core.Array, *, timepoint: int, channel: int, downsampling: int) -> np.ndarray:
    """
    Standardise the way to pull--for a single FOV--the data for a particular (time, channel) combo, with downsampling.

    Parameters
    ----------
    full_image : np.ndarray
        The full dask array of image data, with all timepoints, channels, and spatial dimensions for a particular FOV
    timepoint : int
        The imaging timepoint for which to pull data
    channel : int
        The imaging channel for which to pull data
    downsampling : int
        The step size to take when pulling individual pixels from the underlying image; should be a natural number

    Returns
    -------
    np.ndarray
        The pixel data for the particular timepoint and channel requested, with the given downsampling factor
    """
    return full_image[timepoint, channel, ::downsampling, ::downsampling, ::downsampling].compute()


def detect_spot_single_fov_single_timepoint(
        single_fov_img: np.ndarray, 
        timepoint: int, 
        fish_channel: int, 
        spot_threshold: NumberLike, 
        detection_parameters: SpotDetectionParameters
        ) -> pd.DataFrame:
    print(f"Computing image for spot detection based on downsampling ({detection_parameters.downsampling})")
    img = compute_downsampled_image(single_fov_img, timepoint=timepoint, channel=fish_channel, downsampling=detection_parameters.downsampling)
    crosstalk_timepoint = timepoint if detection_parameters.crosstalk_timepoint is None else detection_parameters.crosstalk_timepoint
    if detection_parameters.subtract_beads:
        # TODO: non-nullity requirement for crosstalk_channel is coupled to this condition, and this should be reflected in the types.
        bead_img = compute_downsampled_image(single_fov_img, timepoint=crosstalk_timepoint, channel=detection_parameters.crosstalk_channel, downsampling=detection_parameters.downsampling)
        img, _ = ip.subtract_crosstalk(source=img, bleed=bead_img, threshold=0)
    spot_detection_result: spotfishing.DetectionResult = detection_parameters.detection_function(img, threshold=spot_threshold)
    spot_props: pd.DataFrame = spot_detection_result.table
    spot_props = detection_parameters.try_centering_spot_box_coordinates(spots_table=spot_props)
    columns_to_scale = ["zMin", "yMin", "xMin", "zMax", "yMax", "xMax", "zc", "yc", "xc"]
    spot_props[columns_to_scale] = spot_props[columns_to_scale] * detection_parameters.downsampling
    return spot_props


def build_spot_prop_table(
    img: np.ndarray, 
    field_of_view: str, 
    channel: int, 
    timepoint_spec: "SingleTimepointDetectionSpec", 
    detection_parameters: "SpotDetectionParameters"
) -> pd.DataFrame:
    timepoint = timepoint_spec.timepoint
    print(f"Building spot properties table; field_of_view={field_of_view}, timepoint={timepoint}, channel={channel}")
    spot_props = detect_spot_single_fov_single_timepoint(
        single_fov_img=img, 
        timepoint=timepoint, 
        fish_channel=channel, 
        spot_threshold=timepoint_spec.threshold, 
        detection_parameters=detection_parameters,
        )
    return finalise_single_spot_props_table(spot_props=spot_props, field_of_view=field_of_view, timepoint=timepoint, channel=channel)


def detect_spots_multiple(
    fov_img_pairs: Iterable[Tuple[str, np.ndarray]], 
    timepoint_specs: Iterable["SingleTimepointDetectionSpec"], 
    channels: Iterable[int], 
    spot_detection_parameters: "SpotDetectionParameters", 
    parallelise: bool = False,
    **joblib_kwargs
) -> pd.DataFrame:
    """Detect spots in each relevant channel and for each given timepoint for the given whole-FOV images."""
    kwargs = copy.copy(joblib_kwargs)
    kwargs.setdefault("n_jobs", -1)
    if parallelise:
        subframes = Parallel(**kwargs)(
            delayed(build_spot_prop_table)(
                img=img, field_of_view=fov, channel=ch, timepoint_spec=spec, detection_parameters=spot_detection_parameters
                )
            for fov, img in tqdm.tqdm(fov_img_pairs) for spec in timepoint_specs for ch in channels
            )
    else:
        subframes = []
        for fov, img in tqdm.tqdm(fov_img_pairs):
            for spec in tqdm.tqdm(timepoint_specs):
                for ch in channels:
                    spots = build_spot_prop_table(
                        img=img, field_of_view=fov, channel=ch, timepoint_spec=spec, detection_parameters=spot_detection_parameters
                        )
                    print(f"Spot count: {len(spots)}")
                    subframes.append(spots)
    return pd.concat(subframes).reset_index(drop=True)


def get_spot_images_zipfile(folder: Union[str, Path, ExtantFolder, NonExtantPath], *, is_background: bool) -> Path:
    """Return fixed-name path to zipfile for spot images, relative to the given folder."""
    folder = folder.path if isinstance(folder, NonExtantPath) else simplify_path(folder)
    fn_base = "spot_background" if is_background else "spot_images"
    return folder / f"{fn_base}.npz"


class DetectionMethod(Enum):
    """Enumerate the spot detection methods available"""
    INTENSITY = 'intensity'
    DIFFERENCE_OF_GAUSSIANS = 'dog'

    @classmethod
    def parse(cls, name: str) -> "DetectionMethod":
        try:
            return next(m for m in cls if m.value.lower() == name.lower())
        except StopIteration:
            raise ValueError(f"Unknown detection method: {name}")


@dataclasses.dataclass(kw_only=True, frozen=True)
class SingleTimepointDetectionSpec:
    # TODO: refine these values as nonnegative.
    timepoint: int # specifies the index of the hybridisation round/timepoint
    threshold: int # specifies a threshold value for intensity-based detection or detection with difference of Gaussians


def get_one_dim_drift_and_bound_and_pad(roi_min: NumberLike, roi_max: NumberLike, dim_limit: int, timepoint_drift: NumberLike, ref_drift: NumberLike) -> Tuple[int, NumberLike, NumberLike, NumberLike, NumberLike]:
    """
    Get the coarse drift, interval, and padding for a single dimension (z, y, or x) for a single ROI.

    Parameters
    ----------
    roi_min : NumberLike
        The lower bound in the current dimension for this ROI
    roi_max : NumberLike
        The upper bound in the current dimension for this ROI
    dim_limit : int
        The number of "pixels" value in this dimension (e.g., 2044/2048 for xy, ~30-40 for z)
    timepoint_drift : NumberLike
        Coarse drift in the current dimension, of the current timepoint 
    ref_drift : NumberLike
        Coarse drift in the current dimension, of the reference timepoint 

    Returns
    -------
    int, NumberLike, NumberLike, NumberLike, NumberLike
        Coarse drift for current dimension, ROI min in dimension, ROI max in dimension, lower padding in dimension, upper padding in dimension
    """
    try:
        coarse_drift = int(timepoint_drift) - int(ref_drift)
    except TypeError:
        logger.error(f"Debugging info -- type(timepoint_drift): {type(timepoint_drift).__name__}. Value: {timepoint_drift}")
        logger.error(f"Debugging info -- type(ref_drift): {type(ref_drift).__name__}. Value: {ref_drift}")
        raise
    target_min = roi_min - coarse_drift
    target_max = roi_max - coarse_drift
    if target_min < dim_limit and target_max > 0: # At least one bound within image
        if target_min < 0: # Lower out-of-bounds, upper in-bounds
            if target_max > dim_limit:
                raise SpotImageSliceOOB(f"Interval for a dimension encompasses entire dimension: ({target_min} < 0, {target_max} > {dim_limit})")
            new_min = 0
            new_max = target_max
            pad_min = 0 - target_min
            pad_max = 0
        elif target_max > dim_limit: # Lower in-bounds, upper out-of-bounds
            new_min = target_min
            new_max = dim_limit
            pad_min = 0
            pad_max = target_max - dim_limit
        else: # Lower and upper in-bounds
            new_min = target_min
            new_max = target_max
            pad_min = 0
            pad_max = 0
    elif target_min >= dim_limit: # Interval "above" image
        new_min, new_max = dim_limit, dim_limit
        pad_min = round(target_max - target_min)
        pad_max = 0
    else: # Interval "below" image
        new_min, new_max = 0, 0
        pad_min = 0
        pad_max = round(target_max - target_min)
    return coarse_drift, new_min, new_max, pad_min, pad_max


class SpotPicker:
    """Encapsulation of data and roles for detection of fluorescent spots in imaging data"""
    def __init__(self, image_handler):
        self.image_handler = image_handler
    
    @property
    def analysis_filename_prefix(self) -> str:
        """Prefix to use for output files in experiment's analysis subfolder, set by config file value"""
        return self.image_handler.analysis_filename_prefix

    @property
    def config(self) -> dict[str, object]:
        return self.image_handler.config

    @property
    def detection_function(self) -> Callable:
        try:
            return {DetectionMethod.INTENSITY.value: detect_spots_int, DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC: detect_spots_dog}[self.detection_method_name]
        except KeyError as e:
            raise ValueError(f"Illegal value for spot detection method in config: {self.detection_method_name}") from e

    @property
    def detection_method_name(self) -> str:
        return self.config.get(DETECTION_METHOD_KEY, DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC)

    @property
    def detection_parameters(self) -> "SpotDetectionParameters":
        try:
            subtract_beads = self.config[CROSSTALK_SUBTRACTION_KEY]
            crosstalk_ch = self.config["crosstalk_ch"]
        except KeyError: #Legacy config.
            subtract_beads = False
            crosstalk_ch = None # dummy that should cause errors; never accessed if subtract_beads is False
        return SpotDetectionParameters(
            detection_function=self.detection_function, 
            downsampling=self.downsampling, 
            minimum_distance_between=self.image_handler.minimum_spot_separation, 
            subtract_beads=subtract_beads, 
            crosstalk_channel=crosstalk_ch, 
            crosstalk_timepoint=None, 
            roi_image_size=self.roi_image_size, 
            )

    @property
    def downsampling(self) -> int:
        return self.config["spot_downsample"]

    @property
    def extraction_skip_reasons_json_file(self) -> Path:
        return self.image_handler.spot_image_extraction_skip_reasons_json_file
    
    @property
    def images(self) -> List[da.core.Array]:
        return self.image_handler.images[self.input_name]

    @property
    def input_name(self):
        """Name of the input to the spot detection phase of the pipeline; in particular, a subfolder of the 'all images' folder typically passed to looptrace"""
        return self.image_handler.spot_input_name

    def iter_timepoint_threshold_pairs(self) -> Iterable[SingleTimepointDetectionSpec]:
        """Iterate over the timepoints in which to detect spots, and the corresponding threshold for each (typically uniform across all timepoints)."""
        for i, t in self._iter_timepoints():
            yield SingleTimepointDetectionSpec(timepoint=t, threshold=self.spot_threshold[i])

    def iter_fov_img_pairs(self) -> Iterable[Tuple[str, np.ndarray]]:
        """Iterate over pairs of FOV name, and corresponding 5-tensor (t, c, z, y, x) of images."""
        for i, fov in enumerate(self.fov_list):
            yield fov, self.images[i]

    def _iter_timepoints(self) -> Iterable[tuple[int, int]]:
        for i, t in enumerate(self._spot_times):
            yield i, t.get

    @property
    def padding_method(self) -> str:
        return self.config.get("padding_method", "edge")

    @property
    def parallelise(self) -> bool:
        return self.config.get("parallelise_spot_detection", False)

    @property
    def path_to_detected_spot_images_folder(self) -> Path:
        return Path(self.image_handler.analysis_path) / "detected_spot_images"

    @property
    def fov_list(self) -> List[str]:
        return self.image_handler.image_lists[self.input_name]

    @property
    def roi_image_size(self) -> Optional[RoiImageSize]:
        """The dimensions (in pixels, as (z, y, x)) for the ROI bounding box around the center of each spot"""
        return self.image_handler.roi_image_size if self.detection_method_name == DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC else None

    @property
    def spot_background_path(self) -> str:
        return os.path.join(self.image_handler.image_save_path, SPOT_BACKGROUND_SUBFOLDER)

    @property
    def spot_background_zipfile(self) -> str:
        return get_spot_images_zipfile(self.image_handler.image_save_path, is_background=True)

    @property
    def spot_channel(self) -> List[int]:
        """The imaging channel in which spot detection is to be done"""
        spot_ch = self.config["spot_ch"]
        return spot_ch if isinstance(spot_ch, list) else [spot_ch]

    @property
    def _spot_times(self) -> list[TimepointFrom0]:
        """The imaging timepoints in which spot detection is to be done, generally those in which regional barcodes were imaged"""
        return self.image_handler.list_all_regional_timepoints()

    @property
    def spot_images_path(self) -> str:
        return os.path.join(self.image_handler.image_save_path, SPOT_IMAGES_SUBFOLDER)

    @property
    def spot_images_zipfile(self):
        return get_spot_images_zipfile(self.image_handler.image_save_path, is_background=False)
    
    @property
    def spot_threshold(self) -> list[int]:
        threshold: object = self.config.get('spot_threshold', 1000)
        if isinstance(threshold, list):
            return threshold
        elif isinstance(threshold, int):
            return [threshold] * len(self._spot_times)
        raise TypeError(f"Spot detection threshold is neither int nor list, but {type(threshold).__name__}")

    def rois_from_spots(self, outfile: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Detect regions of interest (ROIs) from regional barcode FISH spots.

        Parameters
        ----------
        outfile : str or Path, optional
            Path to output file to which to write ROIs; will use default path from 
            underlying image handler if no path is passed to this function
        
        Returns
        ---------
        Path to file containing ROI centers
        """
        params = self.detection_parameters
        logger.info(f"Using '{self.detection_method_name}' for spot detection, threshold = {self.spot_threshold}, downsampling = {params.downsampling}")
        output = detect_spots_multiple(
            fov_img_pairs=self.iter_fov_img_pairs(), 
            timepoint_specs=list(self.iter_timepoint_threshold_pairs()), 
            channels=list(self.spot_channel), 
            spot_detection_parameters=params, 
            parallelise=self.parallelise,
            )
        outfile = outfile or self.image_handler.raw_spots_file
        n_spots = output.shape[0]
        (logger.warning if n_spots == 0 else logger.info)(f"Writing initial spot ROIs with {n_spots} spot(s): {outfile}")
        
        # Here we introduce the RoiIndexColumnName field, giving each ROI an ID.
        # TODO: update with https://github.com/gerlichlab/looptrace/issues/354
        output.to_csv(outfile, index_label="index")
        
        return outfile

    def make_dc_rois_all_timepoints(self) -> str:
        #Precalculate all ROIs for extracting spot images, based on identified ROIs and precalculated drifts between timepoints.
        print("Generating list of all ROIs for tracing...")

        spotfile = self.image_handler.spots_for_voxels_definition_file

        print(f"Reading spots table: {spotfile}")
        # Set index_col=False so that first column (fieldOfView) doesn't become index.
        rois_table: pd.DataFrame = pd.read_csv(spotfile, index_col=False)

        get_fov_idx = lambda fov: self.image_handler.image_lists[self.input_name].index(fov)
        get_locus_timepoints = None if not self.image_handler.locus_grouping else self.image_handler.get_locus_timepoints_for_regional_timepoint
        get_zyx = lambda fov_idx, ch: self.images[fov_idx][0, ch].shape[-3:]
        def get_dc_table(fov_idx: int):
            dc_fov_name = self.image_handler.image_lists[self.config["reg_input_moving"]][fov_idx] # not unused; used for table query
            return self.image_handler.spots_fine_drift_correction_table.query('fieldOfView == @dc_fov_name')

        self.all_rois = build_locus_spot_data_extraction_table(
            rois_table=rois_table,
            get_fov_idx=get_fov_idx,
            get_dc_table=get_dc_table,
            get_locus_timepoints=get_locus_timepoints,
            get_zyx=get_zyx,
            background_timepoint=self.image_handler.background_subtraction_timepoint,
        )
        self.all_rois = self.all_rois.sort_values(get_locus_spot_row_order_columns()).reset_index(drop=True)
        
        print(self.all_rois)
        outfile = self.image_handler.drift_corrected_all_timepoints_rois_file
        print(f"Writing all ROIs file: {outfile}")
        self.all_rois.to_csv(outfile, index=False)
        self.image_handler.load_tables()
        return outfile

    def _write_single_fov_data(self, fov_group_name: str, fov_group_data: pd.DataFrame) -> SkipReasonsMapping:
        """
        Write all timepoints' 3D image arrays (1 for each hybridisation round) for each (region, trace ID) pair in the FOV.

        Parameters
        ----------
        fov_group_name : str
            The name of the field of view to which the given data corresponds
        fov_group_data : pd.DataFrame
            The data from the all-ROIs (_dc_rois.csv) file (1 row per hybridisation for each regional spot, all FOVs) 
            for a particular fieldOfView
        
        Returns
        -------
        SkipReasonsMapping
            Mapping from ref timepoint to mapping from ROI ID to mapping from timepoint to skip reason
        """
        get_num_timepoints: Callable[[int], int]
        if not self.image_handler.locus_grouping:
            print("No locus grouping is present, so all timepoints will be used.")
            total_num_times = len(fov_group_data.timepoint.unique())
            get_num_timepoints = lambda _: total_num_times
        else:
            num_loc_times_by_reg_time_raw = {rt.get: len(lts) for rt, lts in self.image_handler.locus_grouping.items()}
            print(f"Locus time counts by regional time (before +1): {num_loc_times_by_reg_time_raw}")
            # +1 to account for regional timepoint itself.
            get_num_timepoints = lambda reg_time_raw: 1 + num_loc_times_by_reg_time_raw[reg_time_raw]

        num_timepoints_processed: dict[str, int] = {}
        skip_spot_image_reasons = defaultdict(lambda: defaultdict(dict))
        fov_index = self.image_handler.image_lists[self.input_name].index(fov_group_name)
        for timepoint, timepoint_group in tqdm.tqdm(fov_group_data.groupby("timepoint")):
            for ch, ch_group in timepoint_group.groupby(SPOT_CHANNEL_COLUMN_NAME):
                image_stack = np.array(self.images[fov_index][int(timepoint), int(ch)])
                for _, roi in ch_group.iterrows():
                    fn_key = VoxelStackSpecification.from_roi(roi)
                    roi_img, error = extract_single_roi_img_inmem(
                        single_roi=roi, 
                        image_stack=image_stack, 
                        pad_mode=self.padding_method,
                        background_timepoint=self.image_handler.background_subtraction_timepoint, 
                        )
                    if len(roi_img.shape) != 3:
                        raise ArrayDimensionalityError(f"Got not 3, but {len(roi_img.shape)} dimension(s) for ROI image: {roi_img.shape}. fn_key: {fn_key}")
                    roi_img = roi_img.astype(SPOT_IMAGE_PIXEL_VALUE_TYPE)
                    if error is not None:
                        skip_spot_image_reasons[fn_key.ref_timepoint][fn_key.roiId][timepoint] = str(error)
                    # Determine where to write output and how many timepoints are associated with the current regional spot.
                    if timepoint == self.image_handler.background_subtraction_timepoint:
                        n_timepoints = 1
                        array_file_dir = self.spot_background_path
                    else:
                        n_timepoints = get_num_timepoints(fn_key.ref_timepoint)
                        array_file_dir = self.spot_images_path
                    fp = os.path.join(array_file_dir, fn_key.name_roi_file)
                    try:
                        f_id = num_timepoints_processed[fp]
                    except KeyError:
                        # New data stack (for this new regional spot)
                        f_id = 0
                        arr = open_memmap(fp, mode='w+', dtype=roi_img.dtype, shape=(n_timepoints, ) + roi_img.shape)
                    else:
                        # Some processing is done already for this data stack.
                        arr = open_memmap(fp, mode='r+')
                    try:
                        arr[f_id] = roi_img
                    except (IndexError, ValueError):
                        print(f"\n\nERROR adding ROI spot image to stack! Context follows below.")
                        print(f"Current filename key: {fn_key}")
                        print(f"Current file: {fp}")
                        print(f"Current regional time: {fn_key.ref_timepoint}")
                        print(f"Current locus time: {timepoint}")
                        print(f"Current ROI: {roi}")
                        raise
                    arr.flush() # Update what's on disk with what's in memory.
                    num_timepoints_processed[fp] = f_id + 1 # Increment the number of timepoints which have been processed for the current key.
        return skip_spot_image_reasons

    def gen_roi_imgs_inmem(self) -> str:
        rois_file: Path = self.image_handler.drift_corrected_all_timepoints_rois_file
        logger.info(f"Reading ROIs file: {rois_file}")
        rois = pd.read_csv(rois_file, index_col=False)

        if not os.path.isdir(self.spot_images_path):
            os.mkdir(self.spot_images_path)
        if self.image_handler.background_subtraction_timepoint is not None and not os.path.isdir(self.spot_background_path):
            os.mkdir(self.spot_background_path)

        skip_spot_image_reasons = OrderedDict()
        for fov, fov_group in tqdm.tqdm(rois.groupby(FIELD_OF_VIEW_COLUMN)):
            skip_reasons = self._write_single_fov_data(fov_group_name=fov, fov_group_data=fov_group)
            skip_spot_image_reasons[fov] = skip_reasons
        
        print(f"Writing spot image extraction skip reasons file: {self.extraction_skip_reasons_json_file}")
        with open(self.extraction_skip_reasons_json_file, 'w') as fh:
            json.dump(skip_spot_image_reasons, fh, indent=2)

        return self.spot_images_path


def build_locus_spot_data_extraction_table(
    rois_table: pd.DataFrame, 
    *, 
    get_fov_idx: Callable[[str], int], 
    get_dc_table: Callable[[int], pd.DataFrame], 
    get_locus_timepoints: Optional[Callable[[TimepointFrom0], set[TimepointFrom0]]],
    get_zyx: Callable[[int, int], tuple[int, int, int]], # Provide FOV index + channel, get (Z, Y, X).
    background_timepoint: Optional[int] = None,
) -> pd.DataFrame:
    
    all_rois = []

    for _, roi in tqdm.tqdm(rois_table.iterrows(), total=len(rois_table)):
        ref_timepoint: int = roi["timepoint"]
        if not isinstance(ref_timepoint, int):
            raise TypeError(f"Non-integer ({type(ref_timepoint).__name__}) timepoint: {ref_timepoint}")
        
        is_locus_time: Callable[[int], bool]
        if get_locus_timepoints is None:
            is_locus_time = lambda _: True
        else:
            locus_times: set[int] = {lt.get for lt in get_locus_timepoints(TimepointFrom0(ref_timepoint))}
            if not locus_times:
                logging.debug("No locus timepoints for regional timepoint %d, skipping ROI", ref_timepoint)
                continue
            is_locus_time = lambda t: t in locus_times
        
        is_background_time: Callable[[int], bool]
        if background_timepoint is None:
            is_background_time = lambda _: False
        elif not isinstance(background_timepoint, int):
            raise TypeError(f"background_timepoint is not int, but {type(background_timepoint).__name__}")
        else:
            is_background_time = lambda t: t == background_timepoint
        
        fov = roi[FIELD_OF_VIEW_COLUMN]
        fov_index = get_fov_idx(fov)
        sel_dc = get_dc_table(fov_index)
        ch = roi[SPOT_CHANNEL_COLUMN_NAME]
        ref_offset = sel_dc.query('timepoint == @ref_timepoint')
        # TODO: here we can update to iterate over channels for doing multi-channel extraction.
        # https://github.com/gerlichlab/looptrace/issues/138
        Z, Y, X = get_zyx(fov_index, ch)
        for _, dc_row in sel_dc.iterrows():
            timepoint: int = dc_row["timepoint"]
            if not isinstance(timepoint, int):
                raise TypeError(f"Non-integer ({type(timepoint).__name__}) timepoint: {timepoint}")
            if not (is_locus_time(timepoint) or timepoint == ref_timepoint or is_background_time(timepoint)):
                logging.debug("Timepoint %d isn't eligible for tracing in a spot from timepoint %d; skipping", timepoint, ref_timepoint)
                continue
            # min/max ensure that the slicing of the image array to make the small image for tracing doesn't go out of bounds.
            # Padding ensures homogeneity of size of spot images to be used for tracing.
            (
                (z_drift_coarse, z_min, z_max, pad_z_min, pad_z_max), 
                (y_drift_coarse, y_min, y_max, pad_y_min, pad_y_max), 
                (x_drift_coarse, x_min, x_max, pad_x_min, pad_x_max)
            ) = (get_one_dim_drift_and_bound_and_pad(
                roi_min=roi[f"{dim}Min"], 
                roi_max=roi[f"{dim}Max"], 
                dim_limit=lim, 
                timepoint_drift=dc_row[f"{dim}DriftCoarsePixels"], 
                ref_drift=ref_offset[f"{dim}DriftCoarsePixels"]
                ) for dim, lim in (("z", Z), ("y", Y), ("x", X))
                )

            # roi.name is the index value.
            all_rois.append([
                fov, roi["index"], timepoint, ref_timepoint, ch, 
                roi["traceGroup"], roi["traceId"], roi["tracePartners"],
                z_min, z_max, y_min, y_max, x_min, x_max, 
                pad_z_min, pad_z_max, pad_y_min, pad_y_max, pad_x_min, pad_x_max,
                z_drift_coarse, y_drift_coarse, x_drift_coarse, 
                dc_row["zDriftFinePixels"], dc_row["yDriftFinePixels"], dc_row["xDriftFinePixels"]
            ])

    return pd.DataFrame(all_rois, columns=[
        FIELD_OF_VIEW_COLUMN, "roiId", "timepoint", "ref_timepoint", SPOT_CHANNEL_COLUMN_NAME, 
        "traceGroup", "traceId", "tracePartners",
        "zMin", "zMax", "yMin", "yMax", "xMin", "xMax",
        "pad_z_min", "pad_z_max", "pad_y_min", "pad_y_max", "pad_x_min", "pad_x_max", 
        "zDriftCoarsePixels", "yDriftCoarsePixels", "xDriftCoarsePixels",
        "zDriftFinePixels", "yDriftFinePixels", "xDriftFinePixels"
    ])


def extract_single_roi_img_inmem(
    single_roi: pd.Series, 
    image_stack: np.ndarray, 
    pad_mode: str, 
    background_timepoint: Optional[int],
    ) -> Tuple[np.ndarray, Union[None, "SpotImagePaddingError", "SpotImageSliceOOB", "SpotImageSliceEmpty"]]:
    """Function for extracting a single cropped region defined by ROI from a larger 3D image
    
    Parameters
    ----------
    single_roi : pd.Series
        A single row iteration over the drift corrected ROIs file (1 row per timepoint per regional ROI)
    image_stack : np.ndarray
        The image for the current ROI, time, channel combination (corresponding to the single_roi)
    background_timepoint : int or None
        Optinally, the timepoint to be used for background subtraction and therefore excepted from the 
        prohibition on padding in x and y
    pad_mode : str
        Argument for numpy.pad
    
    Returns
    -------
    np.ndarray, Optional[Union[SpotImagePaddingError, SpotImageSliceOOB, SpotImageSliceEmpty]]
        The subimage specified by the ROI's bounding box, possibly with some padding applied; 
        also, an optional exception, indicating to ignore this dummy subimage if this value is not None
    """
    Z, Y, X = image_stack.shape
    # Compute padding for each dimension.
    z_pad, y_pad, x_pad = ((single_roi[f"pad_{dim}_min"], single_roi[f"pad_{dim}_max"]) for dim in ("z", "y", "x"))
    # Compute bounds for extracting the unpadded image.
    # Because of inclusiveness of lower and exclusiveness of upper bound, truncate decimals here.
    z = slice(_down_to_int(single_roi["zMin"]), _down_to_int(single_roi["zMax"]))
    y = slice(_down_to_int(single_roi["yMin"]), _down_to_int(single_roi["yMax"]))
    x = slice(_down_to_int(single_roi["xMin"]), _down_to_int(single_roi["xMax"]))
    # Determine any error.
    error = None
    if z.stop > Z or y.stop > Y or x.stop > X:
        error = SpotImageSliceOOB(f"Slice index OOB for image size {(Z, Y, X)}: {(z, y, x)}")
    elif z.start == z.stop or y.start == y.stop or x.start == x.stop:
        error = SpotImageSliceEmpty(f"Slice would result in at least one empty dimension: {(z, y, x)}")
    elif x_pad != (0, 0) or y_pad != (0, 0):
        if background_timepoint is None or single_roi["timepoint"] != background_timepoint:
            error = SpotImagePaddingError(f"x or y has nonzero padding: x={x_pad}, y={y_pad}")
    # Determine the final ROI (sub)image of a spot for tracing.
    roi_img = np.array(image_stack[z, y, x]) if error is None else np.zeros((z.stop - z.start, y.stop - y.start, x.stop - x.start))
    # If microscope drifted, ROI could be outside image; correct for this if needed.
    pad = (z_pad, y_pad, x_pad)
    if pad != ((0, 0), (0, 0), (0, 0)):
        # Because of truncation of decimals for the axis slicing intervals, round up on lower and down on upper here.
        # Specifically, at most one side of the padding interval should be nonzero.
        # If the nonzero padding is on the upper side, the upper slice bound was maxed and we've already gained the 
        # decimal part of the interval by rounding down the lower interval bound.
        # If the nonzero padding is on the lower side, the lower slice bound was 0 and we've lost gained the 
        # decimal part of the interval by rounding down the lower interval bound; gain back by rounding up lower padding.
        pad = tuple((_up_to_int(lo), _down_to_int(hi)) for lo, hi in pad) 
        kwargs = {"mode": "constant", "constant_values": 0} if any(x == 0 for x in roi_img.shape) else {"mode": pad_mode}
        try:
            roi_img = np.pad(roi_img, pad, **kwargs)
        except ValueError:
            print(f"Cannot pad spot image!\nroi={single_roi}\nshape={roi_img.shape}\n(z, y, x)={(z, y, x)}\npad={pad}\nmode={pad_mode}\n")
            raise
    return roi_img, error


def roi_center_to_bbox(rois: pd.DataFrame, roi_size: RoiImageSize):
    """Make bounding box coordinates around centers of regions of interest, based on box dimensions."""
    halved = roi_size.div_by(2)
    rois["zMin"] = rois["zc"] - halved.z
    rois["zMax"] = rois["zc"] + halved.z
    rois["yMin"] = rois["yc"] - halved.y
    rois["yMax"] = rois["yc"] + halved.y
    rois["xMin"] = rois["xc"] - halved.x
    rois["xMax"] = rois["xc"] + halved.x
    return rois


_down_to_int = lambda x: int(floor(x))
_up_to_int = lambda x: int(ceil(x))


class SpotImageExtractionError(Exception):
    """Represent case in which something's wrong with spot image extraction."""


class SpotImagePaddingError(SpotImageExtractionError):
    """Represent case in which something's wrong with spot image padding."""


class SpotImageSliceOOB(Exception):
    """Represent case in which something's wrong with spot image slicing."""


class SpotImageSliceEmpty(Exception):
    """Represent case in which something's wrong with spot image slicing."""


class SpotImageDimensionalityError(Exception):
    """Represent case in which something's wrong with the dimensionality of an image for spot detection."""
