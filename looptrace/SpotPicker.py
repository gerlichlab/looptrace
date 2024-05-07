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
from math import ceil, floor, pow
import os
from pathlib import Path
from typing import *

import dask.array as da
from joblib import Parallel, delayed
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from scipy import ndimage as ndi
import tqdm

from gertils import ExtantFolder, NonExtantPath
import spotfishing
from spotfishing_looptrace import DifferenceOfGaussiansSpecificationForLooptrace, ORIGINAL_LOOPTRACE_DOG_SPECIFICATION

from looptrace import RoiImageSize, image_processing_functions as ip
from looptrace.exceptions import MissingRoisTableException
from looptrace.filepaths import get_spot_images_path
from looptrace.numeric_types import NumberLike

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

CROSSTALK_SUBTRACTION_KEY = "subtract_crosstalk"
DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC = 'dog'
NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION = ".nuclei_labeled"
NUCLEI_LABELED_SPOTS_FILE_EXTENSION = NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION + ".csv"
DETECTION_METHOD_KEY = "detection_method"

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


class RoiOrderingSpecification:
    """
    Bundle of column/field names to match sorting of rows to sorting of filenames.
    
    In particular, the table of all ROIs (from regional spots, and that region per hybridisation round) 
    should be sorted so that iteration over sorted filenames corresponding to each regional spot will yield 
    an iteration to match row-by-row. This is important for fitting functional forms to the 
    individual spots, as fit parameters must match rows from the all-ROIs table.
    """

    @dataclasses.dataclass
    class FilenameKey:
        position: str
        roi_id: int
        ref_frame: int
        
        @classmethod
        def from_roi(cls, roi: Union[pd.Series, Mapping[str, Any]]) -> "FilenameKey":
            return cls(position=roi["position"], roi_id=roi["roi_id"], ref_frame=roi["ref_frame"])

        @property
        def name_roi_file(self):
            return "_".join([self.position, str(self.roi_id).zfill(5), str(self.ref_frame)]) + ".npy"
        
        @property
        def to_tuple(self) -> Tuple[str, int, int]:
            return self.position, self.roi_id, self.ref_frame

    @staticmethod
    def row_order_columns() -> List[str]:
        return ['position', 'roi_id', 'ref_frame', 'frame']
    
    @classmethod
    def get_file_sort_key(cls, file_key: str) -> FilenameKey:
        try:
            pos, roi, ref = file_key.split("_")
        except ValueError:
            print(f"Failed to get key for file key: {file_key}")
            raise
        return cls.FilenameKey(pos, int(roi), int(ref))


def finalise_single_spot_props_table(spot_props: pd.DataFrame, position: str, frame: int, channel: int) -> pd.DataFrame:
    """
    Perform the addition of several context-relevant fields to the table of detected spot properties for a particular image and channel.

    The arguments to this function specify the table to update, as well as the field of view, the hybridisation round / timepoint, 
    and the imaging channel from which the data came that were used to detect spots to which the given properties table corresponds.

    Parameters
    ----------
    spot_props : pd.DataFrame
        Data table with properties (location, intensity, etc.) of detected spots
    position : str
        Hybridisation round / timepoint in which spots were detected
    frame : int
        Hybridisation round / timepoint for which spots were detected
    channel : int
        Imaging channel in which spots were detected

    Returns
    -------
    pd.DataFrame
        A table annotated with the fields for context (field of view, hybridisation timepoint / round, and imaging channel)
    """
    old_cols = list(spot_props.columns)
    new_cols = ["position", "frame", "ch"]
    spot_props[new_cols] = [position, frame, channel]
    return spot_props[new_cols + old_cols]


@dataclasses.dataclass
class SpotDetectionParameters:
    """Bundle the parameters which are relevant for spot detection."""
    detection_function: callable
    downsampling: int
    minimum_distance_between: NumberLike
    # TODO: non-nullity requirement for crosstalk_channel is coupled to this condition, and this should be reflected in the types.
    subtract_beads: bool
    crosstalk_channel: Optional[int]
    crosstalk_frame: Optional[int]
    roi_image_size: Optional[RoiImageSize]

    def try_centering_spot_box_coordinates(self, spots_table: pd.DataFrame) -> pd.DataFrame:
        if self.roi_image_size is None:
            return spots_table
        dims = self.roi_image_size.div_by(self.downsampling)
        return roi_center_to_bbox(spots_table, roi_size=dims)


def compute_downsampled_image(full_image: da.core.Array, *, frame: int, channel: int, downsampling: int) -> np.ndarray:
    """
    Standardise the way to pull--for a single FOV--the data for a particular (time, channel) combo, with downsampling.

    Parameters
    ----------
    full_image : np.ndarray
        The full dask array of image data, with all timepoints, channels, and spatial dimensions for a particular FOV
    frame : int
        The imaging timepoint for which to pull data
    channel : int
        The imaging channel for which to pull data
    downsampling : int
        The step size to take when pulling individual pixels from the underlying image; should be a natural number

    Returns
    -------
    np.ndarray
        The pixel data for the particular frame and channel requested, with the given downsampling factor
    """
    return full_image[frame, channel, ::downsampling, ::downsampling, ::downsampling].compute()


def detect_spot_single_fov_single_frame(
        single_fov_img: np.ndarray, 
        frame: int, 
        fish_channel: int, 
        spot_threshold: NumberLike, 
        detection_parameters: SpotDetectionParameters
        ) -> pd.DataFrame:
    print(f"Computing image for spot detection based on downsampling ({detection_parameters.downsampling})")
    img = compute_downsampled_image(single_fov_img, frame=frame, channel=fish_channel, downsampling=detection_parameters.downsampling)
    crosstalk_frame = frame if detection_parameters.crosstalk_frame is None else detection_parameters.crosstalk_frame
    if detection_parameters.subtract_beads:
        # TODO: non-nullity requirement for crosstalk_channel is coupled to this condition, and this should be reflected in the types.
        bead_img = compute_downsampled_image(single_fov_img, frame=crosstalk_frame, channel=detection_parameters.crosstalk_channel, downsampling=detection_parameters.downsampling)
        img, _ = ip.subtract_crosstalk(source=img, bleed=bead_img, threshold=0)
    spot_detection_result: spotfishing.DetectionResult = detection_parameters.detection_function(img, threshold=spot_threshold)
    spot_props: pd.DataFrame = spot_detection_result.table
    spot_props = detection_parameters.try_centering_spot_box_coordinates(spots_table=spot_props)
    columns_to_scale = ["z_min", "y_min", "x_min", "z_max", "y_max", "x_max", "zc", "yc", "xc"]
    spot_props[columns_to_scale] = spot_props[columns_to_scale] * detection_parameters.downsampling
    return spot_props


def build_spot_prop_table(
        img: np.ndarray, 
        position: str, 
        channel: int, 
        frame_spec: "SingleFrameDetectionSpec", 
        detection_parameters: "SpotDetectionParameters"
        ) -> pd.DataFrame:
    frame = frame_spec.frame
    print(f"Building spot properties table; position={position}, frame={frame}, channel={channel}")
    spot_props = detect_spot_single_fov_single_frame(
        single_fov_img=img, 
        frame=frame, 
        fish_channel=channel, 
        spot_threshold=frame_spec.threshold, 
        detection_parameters=detection_parameters,
        )
    return finalise_single_spot_props_table(spot_props=spot_props, position=position, frame=frame, channel=channel)


def detect_spots_multiple(
        pos_img_pairs: Iterable[Tuple[str, np.ndarray]], 
        frame_specs: Iterable["SingleFrameDetectionSpec"], 
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
                img=img, position=pos, channel=ch, frame_spec=spec, detection_parameters=spot_detection_parameters
                )
            for pos, img in tqdm.tqdm(pos_img_pairs) for spec in frame_specs for ch in channels
            )
    else:
        subframes = []
        for pos, img in tqdm.tqdm(pos_img_pairs):
            for spec in tqdm.tqdm(frame_specs):
                for ch in channels:
                    spots = build_spot_prop_table(
                        img=img, position=pos, channel=ch, frame_spec=spec, detection_parameters=spot_detection_parameters
                        )
                    print(f"Spot count: {len(spots)}")
                    subframes.append(spots)
    return pd.concat(subframes).reset_index(drop=True)


def get_spot_images_zipfile(folder: Union[Path, ExtantFolder, NonExtantPath]) -> Path:
    """Return fixed-name path to zipfile for spot images, relative to the given folder."""
    if isinstance(folder, (ExtantFolder, NonExtantPath)):
        folder = folder.path
    return folder / "spot_images.npz"


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


@dataclasses.dataclass
class FieldOfViewRepresentation:
    # TODO: refine index as nonnegative
    name: str
    index: int # This specifies the 0-based index of the position name in a list of position names.


@dataclasses.dataclass
class SingleFrameDetectionSpec:
    # TODO: refine these values as nonnegative.
    frame: int # specifies the index of the hybridisation round/timepoint
    threshold: int # specifies a threshold value for intensity-based detection or detection with difference of Gaussians


@dataclasses.dataclass
class DetectionSpec3D:
    """Three values that, together, should constitute an index that retrieves a '3D' image (z-stack of 2D images)"""
    position: "FieldOfViewRepresentation" # specifies the field of view (FOV / "position")
    frame: "SingleFrameDetectionSpec" # specifies the hybridisation round / timepoint
    channel: int # specifies the imaging channel in which signal was captured


def generate_detection_specifications(positions: Iterable["FieldOfViewRepresentation"], single_frame_specs: Iterable["SingleFrameDetectionSpec"], channels: Iterable[int]) -> Iterable["DetectionSpec3D"]:
    """
    Build individual specifications for spot detection, with each specification bundling field of view, hybridisation round, and imaging channel.

    Parameters
    ----------
    positions : Iterable of str
        Collection of the names of the fields of view
    single_frame_specs : Iterable of SingleFrameDetectionSpec
        Collection of the hybridisation round and corresponding detection threshold
    channels : Iterable of int
        Collection of the imaging channels to process for each hybridisation timepoint/round

    Returns
    -------
    Iterable of DetectionSpec3D
        Collection of the specifications for where and how to detect spots
    """
    for position_definition in tqdm.tqdm(positions):
        print("Position: ", position_definition)
        for frame_spec in tqdm.tqdm(single_frame_specs):
            print("Frame spec: ", frame_spec)
            for ch in tqdm.tqdm(channels):
                yield DetectionSpec3D(position=position_definition, frame=frame_spec, channel=ch)


def get_one_dim_drift_and_bound_and_pad(roi_min: NumberLike, roi_max: NumberLike, dim_limit: int, frame_drift: NumberLike, ref_drift: NumberLike) -> Tuple[int, NumberLike, NumberLike, NumberLike, NumberLike]:
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
    frame_drift : NumberLike
        Coarse drift in the current dimension, of the current timepoint 
    ref_drift : NumberLike
        Coarse drift in the current dimension, of the reference timepoint 

    Returns
    -------
    int, NumberLike, NumberLike, NumberLike, NumberLike
        Coarse drift for current dimension, ROI min in dimension, ROI max in dimension, lower padding in dimension, upper padding in dimension
    """
    coarse_drift = int(frame_drift) - int(ref_drift)
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
        self.config = image_handler.config
    
    @property
    def analysis_filename_prefix(self) -> str:
        """Prefix to use for output files in experiment's analysis subfolder, set by config file value"""
        return self.image_handler.analysis_filename_prefix

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
            crosstalk_frame=None, 
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

    def iter_frames_and_channels(self) -> Iterable[Tuple[Tuple[int, int], int]]:
        for i, frame in enumerate(self.spot_frame):
            for channel in self.spot_channel:
                yield (i, frame), channel

    def iter_frame_threshold_pairs(self) -> Iterable[SingleFrameDetectionSpec]:
        """Iterate over the frames in which to detect spots, and the corresponding threshold for each (typically uniform across all frames)."""
        for i, frame in enumerate(self.spot_frame):
            yield SingleFrameDetectionSpec(frame=frame, threshold=self.spot_threshold[i])

    def iter_pos_img_pairs(self) -> Iterable[Tuple[str, np.ndarray]]:
        """Iterate over pairs of position (FOV) name, and corresponding 5-tensor (t, c, z, y, x ) of images."""
        for i, pos in enumerate(self.pos_list):
            yield pos, self.images[i]

    @property
    def padding_method(self) -> str:
        return self.config.get("padding_method", "edge")

    @property
    def parallelise(self) -> bool:
        return self.config.get("parallelise_spot_detection", False)

    @property
    def path_to_detected_spot_images_folder(self) -> Path:
        return Path(self.image_handler.analysis_path) / "detected_spot_images"

    def path_to_detected_spot_image_file(self, position: int, time: int, channel: int) -> Path:
        """Get the path to the detected spot image file for given FOV (position), timepoint (hybridisation round), and imaging channel."""
        fn = get_name_for_detected_spot_image_file(
            fn_prefix=self.analysis_filename_prefix, 
            position=position, 
            time=time, 
            channel=channel,
            )
        return self.path_to_detected_spot_images_folder / fn

    @property
    def pos_list(self) -> List[str]:
        return self.image_handler.image_lists[self.input_name]

    @property
    def roi_image_size(self) -> Optional[RoiImageSize]:
        """The dimensions (in pixels, as (z, y, x)) for the ROI bounding box around the center of each spot"""
        return self.image_handler.roi_image_size if self.detection_method_name == DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC else None

    @property
    def spot_channel(self) -> List[int]:
        """The imaging channel in which spot detection is to be done"""
        spot_ch = self.config["spot_ch"]
        return spot_ch if isinstance(spot_ch, list) else [spot_ch]

    @property
    def spot_frame(self) -> List[int]:
        """The imaging timepoints in which spot detection is to be done, generally those in which regional barcodes were imaged"""
        spot_frame = self.config["spot_frame"]
        return spot_frame if isinstance(spot_frame, list) else [spot_frame]

    @property
    def spot_images_path(self):
        return get_spot_images_path(self.image_handler.image_save_path)

    @property
    def spot_images_zipfile(self):
        # TODO: what to do if this path is nested under self.spot_images_path, and will be deleted upon zip?
        # See: https://github.com/gerlichlab/looptrace/issues/19
        # See: https://github.com/gerlichlab/looptrace/issues/20
        return get_spot_images_zipfile(self.image_handler.image_save_path)

    @property
    def spot_in_nuc(self) -> bool:
        return self.config.get('spot_in_nuc', False)
    
    @property
    def spot_threshold(self) -> List[int]:
        spot_threshold = self.config.get('spot_threshold', 1000)
        return spot_threshold if isinstance(spot_threshold, list) else [spot_threshold] * len(self.spot_frame)

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
            pos_img_pairs=self.iter_pos_img_pairs(), 
            frame_specs=list(self.iter_frame_threshold_pairs()), 
            channels=list(self.spot_channel), 
            spot_detection_parameters=params, 
            parallelise=self.parallelise,
            )
        outfile = outfile or self.image_handler.raw_spots_file
        n_spots = output.shape[0]
        (logger.warning if n_spots == 0 else logger.info)(f"Writing initial spot ROIs with {n_spots} spot(s): {outfile}")
        output.to_csv(outfile)
        return outfile
    
    def make_dc_rois_all_frames(self) -> str:
        #Precalculate all ROIs for extracting spot images, based on identified ROIs and precalculated drifts between time frames.
        print('Generating list of all ROIs for tracing:')

        all_rois = []
        
        spotfile = self.image_handler.nuclei_filtered_spots_file_path if self.config.get('spot_in_nuc', False) \
            else self.image_handler.proximity_filtered_spots_file_path
        key_rois_table, _ = os.path.splitext(spotfile.name)
        key_rois_table = key_rois_table\
            .lstrip(self.analysis_filename_prefix)\
            .lstrip(os.path.expanduser(os.path.expandvars(self.analysis_filename_prefix)))
        
        try:
            rois_table = self.image_handler.tables[key_rois_table]
        except KeyError as e:
            raise MissingRoisTableException(key_rois_table) from e
        
        for idx, (_, roi) in tqdm.tqdm(enumerate(rois_table.iterrows()), total=len(rois_table)):
            pos = roi['position']
            pos_index = self.image_handler.image_lists[self.input_name].index(pos)
            dc_pos_name = self.image_handler.image_lists[self.config['reg_input_moving']][pos_index] # not unused; used for table query
            sel_dc = self.image_handler.tables[self.input_name + '_drift_correction_fine'].query('position == @dc_pos_name')
            ref_frame = roi['frame']
            ch = roi['ch']
            ref_offset = sel_dc.query('frame == @ref_frame')
            # TODO: here we can update to iterate over channels for doing multi-channel extraction.
            # https://github.com/gerlichlab/looptrace/issues/138
            Z, Y, X = self.images[pos_index][0, ch].shape[-3:]
            for _, dc_frame in sel_dc.iterrows():
                # min/max ensure that the slicing of the image array to make the small image for tracing doesn't go out of bounds.
                # Padding ensures homogeneity of size of spot images to be used for tracing.
                (
                    (z_drift_coarse, z_min, z_max, pad_z_min, pad_z_max), 
                    (y_drift_coarse, y_min, y_max, pad_y_min, pad_y_max), 
                    (x_drift_coarse, x_min, x_max, pad_x_min, pad_x_max)
                ) = (get_one_dim_drift_and_bound_and_pad(
                    roi_min=roi[f"{dim}_min"], 
                    roi_max=roi[f"{dim}_max"], 
                    dim_limit=lim, 
                    frame_drift=dc_frame[f"{dim}_px_coarse"], 
                    ref_drift=ref_offset[f"{dim}_px_coarse"]
                    ) for dim, lim in (("z", Z), ("y", Y), ("x", X))
                    )

                # roi.name is the index value.
                all_rois.append([pos, pos_index, idx, roi.name, dc_frame['frame'], ref_frame, ch, 
                                z_min, z_max, y_min, y_max, x_min, x_max, 
                                pad_z_min, pad_z_max, pad_y_min, pad_y_max, pad_x_min, pad_x_max,
                                z_drift_coarse, y_drift_coarse, x_drift_coarse, 
                                dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])

        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_number', 'roi_id', 'frame', 'ref_frame', 'ch', 
                                "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
                                'pad_z_min', 'pad_z_max', 'pad_y_min', 'pad_y_max', 'pad_x_min', 'pad_x_max', 
                                'z_px_coarse', 'y_px_coarse', 'x_px_coarse',
                                'z_px_fine', 'y_px_fine', 'x_px_fine'])
        self.all_rois = self.all_rois.sort_values(RoiOrderingSpecification.row_order_columns()).reset_index(drop=True)
        print(self.all_rois)
        outfile = self.image_handler.drift_corrected_all_timepoints_rois_file
        print(f"Writing all ROIs file: {outfile}")
        self.all_rois.to_csv(outfile)
        self.image_handler.load_tables()
        return outfile

    def write_single_fov_data(self, pos_group_name: str, pos_group_data: pd.DataFrame) -> Tuple[List[str], SkipReasonsMapping]:
        """
        Write all timepoints' 3D image arrays (1 for each hybridisation round) for each (region, trace ID) pair in the FOV.

        Parameters
        ----------
        pos_group_name : str
            The name of the position (FOV) to which the given data corresponds
        pos_group_data : pd.DataFrame
            The data from the all-ROIs (_dc_rois.csv) file (1 row per hybridisation for each regional spot, all FOVs) 
            for a particular FOV / position
        
        Returns
        -------
        List[str], SkipReasonsMapping
            Paths of the files written, and mapping from ref frame to mapping from ROI ID to mapping from frame to skip reason
        """
        pos_index = self.image_handler.image_lists[self.input_name].index(pos_group_name)
        f_id = 0
        n_frames = len(pos_group_data.frame.unique())
        array_files = OrderedDict() # Maintain insertion order.
        skip_spot_image_reasons = defaultdict(lambda: defaultdict(dict))
        for frame, frame_group in tqdm.tqdm(pos_group_data.groupby('frame')):
            for ch, ch_group in frame_group.groupby('ch'):
                image_stack = np.array(self.images[pos_index][int(frame), int(ch)])
                for _, roi in ch_group.iterrows():
                    roi_img, error = extract_single_roi_img_inmem(
                        single_roi=roi, 
                        image_stack=image_stack, 
                        pad_mode=self.padding_method,
                        background_frame=self.image_handler.background_subtraction_frame, 
                        )
                    roi_img = roi_img.astype(np.uint16)
                    fn_key = RoiOrderingSpecification.FilenameKey.from_roi(roi)
                    if error is not None:
                        skip_spot_image_reasons[fn_key.ref_frame][fn_key.roi_id][frame] = str(error)
                    fp = os.path.join(self.spot_images_path, fn_key.name_roi_file)
                    if fp in array_files:
                        arr = open_memmap(fp, mode='r+')
                    else:
                        arr = open_memmap(fp, mode='w+', dtype = roi_img.dtype, shape=(n_frames,) + roi_img.shape)
                        array_files[fp] = None # Values don't matter, just for key order.
                    try:
                        arr[f_id] = roi_img
                    except ValueError:
                        print(f"ERROR adding ROI spot image to stack! ROI: {roi}")
                        raise
                    arr.flush()
            f_id += 1
        return list(array_files.keys()), skip_spot_image_reasons


    def gen_roi_imgs_inmem(self) -> str:
        # Load full stacks into memory to extract spots.
        # Not the most elegant, but depending on the chunking of the original data it is often more performant than loading subsegments.

        rois = self.image_handler.tables[self.input_name+'_dc_rois']

        if not os.path.isdir(self.spot_images_path):
            os.mkdir(self.spot_images_path)

        skip_spot_image_reasons = OrderedDict()
        for pos, pos_group in tqdm.tqdm(rois.groupby('position')):
            _, skip_reasons = self.write_single_fov_data(pos_group_name=pos, pos_group_data=pos_group)
            skip_spot_image_reasons[pos] = skip_reasons
        
        print(f"Writing spot image extraction skip reasons file: {self.extraction_skip_reasons_json_file}")
        with open(self.extraction_skip_reasons_json_file, 'w') as fh:
            json.dump(skip_spot_image_reasons, fh, indent=2)

        return self.spot_images_path

    def gen_roi_imgs_inmem_coarsedc(self) -> str:
        # Use this simplified function if the images that the spots are gathered from are already coarsely drift corrected!
        print('Generating single spot image stacks from coarsely drift corrected images.')
        rois = self.image_handler.tables[self.input_name+'_dc_rois']
        for pos, group in tqdm.tqdm(rois.groupby('position')):
            pos_index = self.image_handler.image_lists[self.input_name].index(pos)
            full_image = np.array(self.images[pos_index])
            for roi in group.to_dict('records'):
                spot_stack = full_image[:, 
                                roi['ch'], 
                                roi["z_min"]:roi["z_max"], 
                                roi["y_min"]:roi["y_max"],
                                roi["x_min"]:roi["x_max"]].copy()
                fn = pos+'_'+str(roi['frame'])+'_'+str(roi['roi_id_pos']).zfill(4)
                arr_out = os.path.join(self.spot_images_path, fn + '.npy')
                np.save(arr_out, spot_stack)
        return self.spot_images_path

    def fine_dc_single_roi_img(self, roi_img, roi):
        #Shift a single image according to precalculated drifts.
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        roi_img = ndi.shift(roi_img, (dz, dy, dx)).astype(np.uint16)
        return roi_img

    def gen_fine_dc_roi_imgs(self):
        #Apply fine scale drift correction to spot images, used mainly for visualizing fits (these images are not used for fitting)
        print('Making fine drift-corrected spot images.')

        imgs = self.image_handler.images['spot_images']
        
        rois = self.all_rois

        i = 0
        roi_array_fine = []
        for j, frame_stack in tqdm.tqdm(enumerate(imgs)):
            roi_stack_fine = []
            for roi_stack in frame_stack:
                roi_stack_fine.append(self.fine_dc_single_roi_img(roi_stack, rois.iloc[i]))
                i += 1
            roi_array_fine.append(np.stack(roi_stack_fine))

        self.image_handler.images['spot_images_fine'] = roi_array_fine
        np.savez_compressed(self.image_handler.image_save_path+os.sep+'spot_images_fine.npz', *roi_array_fine)


def extract_single_roi_img_inmem(
    single_roi: pd.Series, 
    image_stack: np.ndarray, 
    pad_mode: str, 
    background_frame: Optional[int],
    ) -> Tuple[np.ndarray, Union[None, "SpotImagePaddingError", "SpotImageSliceOOB", "SpotImageSliceEmpty"]]:
    """Function for extracting a single cropped region defined by ROI from a larger 3D image
    
    Parameters
    ----------
    single_roi : pd.Series
        A single row iteration over the drift corrected ROIs file (1 row per timepoint per regional ROI)
    image_stack : np.ndarray
        The image for the current ROI, time, channel combination (corresponding to the single_roi)
    background_frame : int or None
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
    z = slice(_down_to_int(single_roi["z_min"]), _down_to_int(single_roi["z_max"]))
    y = slice(_down_to_int(single_roi["y_min"]), _down_to_int(single_roi["y_max"]))
    x = slice(_down_to_int(single_roi["x_min"]), _down_to_int(single_roi["x_max"]))
    # Determine any error.
    error = None
    if z.stop > Z or y.stop > Y or x.stop > X:
        error = SpotImageSliceOOB(f"Slice index OOB for image size {(Z, Y, X)}: {(z, y, x)}")
    elif z.start == z.stop or y.start == y.stop or x.start == x.stop:
        error = SpotImageSliceEmpty(f"Slice would result in at least one empty dimension: {(z, y, x)}")
    elif x_pad != (0, 0) or y_pad != (0, 0):
        if background_frame is None or single_roi["frame"] != background_frame:
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
    rois["z_min"] = rois["zc"] - halved.z
    rois["z_max"] = rois["zc"] + halved.z
    rois["y_min"] = rois["yc"] - halved.y
    rois["y_max"] = rois["yc"] + halved.y
    rois["x_min"] = rois["xc"] - halved.x
    rois["x_max"] = rois["xc"] + halved.x
    return rois


def get_name_for_detected_spot_image_file(*, fn_prefix: str, position: int, time: int, channel: int, filetype: str = "png") -> Path:
    """
    Return the path to the detected spot images file for the given (FOV, time, channel) triplet.

    Each value must be in [0, 999] since each represents the value of an entity which should be nonnegative, 
    and each will be represented by 3 base-10 digits.

    Parameters
    ----------
    fn_prefix : str
        Prefix for the filename, e.g. name of experiment; use empty string to have no prefix
    position : int
        Imaging field of view (0-based index)
    time : int
        Imaging timepoint (0-based index)
    channel : int
        Imaging channel
    filetype : str
        Name of extension (without prefix dot) to save as

    Raises
    ------
    TypeError: if position is given as a value other than integer
    ValueError: if any value is negative or too big to be represented by 3 base-10 digits

    Returns
    -------
    Path: path to the detected spot images file for the given (FOV, time, channel) triplet.
    """
    num_digits = 3
    keyed_values = [("P", position), ("T", time), ("C", channel)]
    too_big = [(k, v) for k, v in keyed_values if v > int(pow(10, num_digits)) - 1] # Check that we have enough digits to represent number.
    if too_big:
        raise ValueError(f"{len(too_big)} values cannot be represented with {num_digits} digits: {too_big}")
    negatives = [(k, v) for k, v in keyed_values if v < 0] # Check that each value is nonnegative.
    if negatives:
        raise ValueError(f"{len(negatives)} values representing nonnegative fields are negative: {negatives}")
    if not isinstance(position, int):
        raise TypeError(f"For detected spot image file path, position should be integer, not {type(position).__name__}: {position}")
    fn_chunks = [pre + str(n).zfill(3) for pre, n in keyed_values]
    fn_base = "_".join([fn_prefix] + fn_chunks if fn_prefix else fn_chunks)
    return f"{fn_base}.regional_spots.{filetype}"


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
