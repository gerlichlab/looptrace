# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import copy
import dataclasses
from enum import Enum
from joblib import Parallel, delayed
import logging
import os
from pathlib import Path
import random
from typing import *
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
import tqdm

from gertils import ExtantFolder, NonExtantPath

from looptrace.exceptions import MissingRoisTableException
from looptrace.filepaths import get_spot_images_path
from looptrace import image_processing_functions as ip
from looptrace.numeric_types import NumberLike

CROSSTALK_SUBTRACTION_KEY = "subtract_crosstalk"
DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC = 'dog'
NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION = ".nuclei_labeled"
DETECTION_METHOD_KEY = "detection_method"

logger = logging.getLogger()


class RoiOrderingSpecification():
    """
    Bundle of column/field names to match sorting of rows to sorting of filenames.
    
    In particular, the table of all ROIs (from regional spots, and that region per hybridisation round) 
    should be sorted so that iteration over sorted filenames corresponding to each regional spot will yield 
    an iteration to match row-by-row. This is important for fitting functional forms to the 
    individual spots, as fit parameters must match rows from the all-ROIs table.
    """
    
    @staticmethod
    def row_order_columns() -> List[str]:
        return ['position', 'roi_id', 'ref_frame', 'frame']
    
    @staticmethod
    def name_roi_file(pos_name, roi) -> str:
        """
        Create a name for .npy file for particular ROI; ROI must support __getitem__.
        
        Parameters
        ----------
        pos_name : str
            The name of a position (FOV) group, e.g. P0001.zarr
        roi : Union[pd.Series, Mapping[str, Any]]
            Single row record from all ROIs table
        
        Returns
        -------
        str
            Name for file corresponding to this spot (regional)'s data in a single hybridisation round
        """
        return "_".join([pos_name, str(roi['roi_id']).zfill(5), str(roi['ref_frame'])]) + ".npy"
    
    @staticmethod
    def get_file_key(file_key: str) -> Tuple[str, int, int]:
        try:
            pos, roi, ref = file_key.split("_")
        except ValueError:
            print(f"Failed to get key for file key: {file_key}")
            raise
        return pos, int(roi), int(ref)


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
    subtract_beads: bool
    crosstalk_channel: Optional[int]
    crosstalk_frame: Optional[int]
    roi_image_size: Optional[Tuple[int, int, int]]

    def center_spots(self, spots_table: pd.DataFrame) -> pd.DataFrame:
        if self.roi_image_size is None:
            return spots_table
        dims = tuple(map(lambda x: x // self.downsampling, self.roi_image_size))
        return ip.roi_center_to_bbox(spots_table, roi_size=dims)


def detect_spot_single(
        full_image: np.ndarray, 
        frame: int, 
        fish_channel: int, 
        spot_threshold: NumberLike, 
        detection_parameters: SpotDetectionParameters
        ) -> pd.DataFrame:
    img = full_image[frame, fish_channel, ::detection_parameters.downsampling, ::detection_parameters.downsampling, ::detection_parameters.downsampling].compute()
    crosstalk_frame = frame if detection_parameters.crosstalk_frame is None else detection_parameters.crosstalk_frame
    if detection_parameters.subtract_beads:
        # TODO: non-nullity requirement for crosstalk_channel is coupled to this condition, and this should be reflected in the types.
        bead_img = full_image[
            crosstalk_frame, 
            detection_parameters.crosstalk_channel, 
            ::detection_parameters.downsampling, 
            ::detection_parameters.downsampling, 
            ::detection_parameters.downsampling
            ].compute()
        img, _ = ip.subtract_crosstalk(source=img, bleed=bead_img, threshold=0)
    spot_props, _ = detection_parameters.detection_function(img, spot_threshold, min_dist=detection_parameters.minimum_distance_between)
    spot_props = detection_parameters.center_spots(spots_table=spot_props)
    spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = \
        spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] * detection_parameters.downsampling
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
    spot_props = detect_spot_single(full_image=img, frame=frame, fish_channel=channel, spot_threshold=frame_spec.threshold, detection_parameters=detection_parameters)
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
    # return Parallel(**kwargs)(
    #     delayed(lambda img, t, c, threshold, position: finalise_single_spot_props_table(
    #         spot_props=spot_detection_parameters.detect_spot_single(full_image=img, frame=t, fish_channel=c, spot_threshold=threshold), 
    #         position=position, 
    #         frame=t, 
    #         channel=c))(img=img, t=spec.frame, c=ch, threshold=spec.threshold, position=pos)
    #     for pos, img in pos_img_pairs for spec in frame_specs for ch in channels
    #     )


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


def center_spots(spots_table: pd.DataFrame, spot_ds: int, roi_image_size: Optional[Tuple[int, int, int]]) -> pd.DataFrame:
    if roi_image_size is None:
        return spots_table
    dims = tuple(map(lambda x: x // spot_ds, roi_image_size))
    return ip.roi_center_to_bbox(spots_table, roi_size=dims)


class SpotPicker:
    """Encapsulation of data and roles for detection of fluorescent spots in imaging data"""
    def __init__(self, image_handler, array_id = None):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images = self.image_handler.images[self.input_name]
        self.pos_list = self.image_handler.image_lists[self.input_name]
        roi_file_ext = ".csv"
        self.dc_roi_path = self.image_handler.out_path(self.input_name + '_dc_rois' + roi_file_ext)
        self.array_id = array_id
        if self.array_id is not None:
            self.pos_list = [self.pos_list[int(self.array_id)]]
            roi_filename_differentiator = '_rois_' + str(self.array_id).zfill(4)
            #center_filename_differentiator = '_centers_' + str(self.array_id).zfill(4)
        else:
            roi_filename_differentiator = '_rois'
            #center_filename_differentiator = '_centers'
        #self.roi_centers_filepath = self.image_handler.out_path(self.input_name + center_filename_differentiator + roi_file_ext)
        self.roi_path = self.image_handler.out_path(self.input_name + roi_filename_differentiator + roi_file_ext)
        #self.roi_path_filtered = self.roi_path.replace(roi_file_ext, ".filtered" + roi_file_ext)
        #self.roi_path_unfiltered = self.roi_path.replace(roi_file_ext, ".unfiltered" + roi_file_ext)

    @property
    def crosstalk_frame(self) -> Optional[int]:
        return self.config.get('crosstalk_frame')

    @property
    def detection_method_name(self) -> str:
        return self.config.get(DETECTION_METHOD_KEY, DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC)
    
    @property
    def detection_function(self) -> Callable:
        try:
            return {DetectionMethod.INTENSITY.value: ip.detect_spots_int, DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC: ip.detect_spots}[self.detection_method_name]
        except KeyError as e:
            raise ValueError(f"Illegal value for spot detection method in config: {self.detection_method_name}") from e
        
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
        pos_names = self.image_handler.image_lists[self.input_name]
        for pos in self.pos_list:
            idx = pos_names.index(pos)
            yield pos, self.images[idx]

    @property
    def _raw_roi_image_size(self) -> Tuple[int, int, int]:
        return tuple(self.config['roi_image_size'])

    @property
    def roi_image_size(self) -> Optional[Tuple[int, int, int]]:
        return self._raw_roi_image_size if self.detection_method_name == DIFFERENCE_OF_GAUSSIANS_CONFIG_VALUE_SPEC else None

    @property
    def spot_channel(self) -> List[int]:
        spot_ch = self.config['spot_ch']
        return spot_ch if isinstance(spot_ch, list) else [spot_ch]

    @property
    def spot_frame(self) -> List[int]:
        spot_frame = self.config['spot_frame']
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

    def rois_from_spots(self, preview_pos=None, outfile: Optional[Union[str, Path]] = None) -> Optional[Union[str, Path]]:
        '''
        Autodetect ROIs from spot images using a manual threshold defined in config.
        
        Returns
        ---------
        Path to file containing ROI centers
        '''

        # Fetch some settings.        
        try:
            subtract_beads = self.config['subtract_crosstalk']
            crosstalk_ch = self.config['crosstalk_ch']
        except KeyError: #Legacy config.
            subtract_beads = False
            crosstalk_ch = None # dummy that should cause errors; never accessed if subtract_beads is False

        min_dist = self.config.get('min_spot_dist')

        # Determine the detection method and parameters threshold.
        spot_threshold = self.spot_threshold
        detect_func = self.detection_function
        logger.info(f"Using '{self.detection_method_name}' for spot detection, threshold : {spot_threshold}")
        spot_ds = self.config['spot_downsample']
        logger.info(f"Spot downsampling setting: {spot_ds}")
        
        params = SpotDetectionParameters(
            detection_function=detect_func, 
            downsampling=spot_ds, 
            minimum_distance_between=min_dist, 
            subtract_beads=subtract_beads, 
            crosstalk_channel=crosstalk_ch, 
            crosstalk_frame=None, 
            roi_image_size=self.roi_image_size, 
            )

        # previewing
        if preview_pos is not None:
            for (i, frame), ch in self.iter_frames_and_channels():
                logger.info(f'Preview spot detection in position {preview_pos}, frame {frame} with threshold {spot_threshold[i]}.')
                pos_index = self.image_handler.image_lists[self.input_name].index(preview_pos)
                img = self.images[pos_index][frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()

                if subtract_beads:
                    bead_img = self.images[pos_index][frame, crosstalk_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                    img, orig = ip.subtract_crosstalk(source=img, bleed=bead_img, threshold=0)

                spot_props, filt_img = detect_func(img, spot_threshold[i], min_dist = min_dist)
                spot_props['position'] = preview_pos
                spot_props = spot_props.reset_index().rename(columns={'index':'roi_id_pos'})

                spot_props = params.center_spots(spots_table=spot_props)
                
                roi_points, _ = ip.roi_to_napari_points(spot_props, position=preview_pos)
                try:
                    ip.napari_view(np.stack([filt_img, img, orig]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Subtracted', 'Original'])
                except NameError:
                    ip.napari_view(np.stack([filt_img, img]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Original'])

            return

        # Not previewing, but actually computing all ROIs
        output = detect_spots_multiple(
            pos_img_pairs=self.iter_pos_img_pairs(), 
            frame_specs=list(self.iter_frame_threshold_pairs()), 
            channels=list(self.spot_channel), 
            spot_detection_parameters=params
            )
        
        logger.info(f"Writing initial spot ROIs: {self.roi_path}")
        n_spots = len(output)
        (logger.warning if n_spots == 0 else logger.info)(f'Found {n_spots} spots.')
        outfile = outfile or self.roi_path
        print(f"Writing ROIs: {outfile}")
        output.to_csv(outfile)

        return outfile


    def rois_from_beads(self):
        print('Detecting bead ROIs for tracing.')
        all_rois = []
        n_fields = self.config['bead_trace_fields']
        n_beads = self.config['bead_trace_number']
        for pos in tqdm.tqdm(random.sample(self.pos_list, k=n_fields)):
            pos_index = self.image_handler.image_lists[self.input_name].index(pos)
            ref_frame = self.config['bead_reference_frame']
            ref_ch = self.config['bead_ch']
            threshold = self.config['bead_threshold']
            min_bead_int = self.config['min_bead_intensity']

            t_img = self.images[pos_index][ref_frame, ref_ch].compute()
            t_img_label, num_labels = ndi.label(t_img>threshold)
            
            spot_props = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
            spot_props = spot_props.query('max_intensity > @min_bead_int').sample(n=n_beads, random_state=1)
            
            spot_props.drop(['label'], axis=1, inplace=True)
            spot_props.rename(columns={'centroid-0': 'zc',
                                        'centroid-1': 'yc',
                                        'centroid-2': 'xc',
                                        'index':'roi_id_pos'},
                                        inplace = True)

            spot_props['position'] = pos
            spot_props['frame'] = ref_frame
            spot_props['ch'] = ref_ch
            print('Detected beads in position', pos, spot_props)
            all_rois.append(spot_props)
        
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'index':'roi_id'})
        rois = ip.roi_center_to_bbox(output, roi_size=self._raw_roi_image_size)

        self.image_handler.bead_rois = rois
        rois.to_csv(self.roi_path + '_beads.csv')
        self.image_handler.load_tables()
        return rois
    '''
    def refilter_rois(self):
        #TODO needs updating to new table syntax.
        if self.detection_method_name == 'dog':
            self.image_handler.roi_table = ip.roi_center_to_bbox(self.image_handler.roi_table.copy(), roi_size = tuple(self.config['roi_image_size']))
        
        self.image_handler.roi_table[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = self.image_handler.roi_table[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']].round().astype(int)

        if 'nuc_images' not in self.image_handler.images:
            print('Please generate nuclei images first.')
        if 'nuc_masks' in self.image_handler.images:
            self.image_handler.roi_table = ip.filter_rois_in_nucs(self.image_handler.roi_table.copy(), np.array(self.image_handler.images['nuc_masks']), self.pos_list, new_col='nuc_label', drifts = self.image_handler.tables['nuc_images_drift_correction'], target_frame=self.config['nuc_ref_frame'])
            
        if 'nuc_classes' in self.image_handler.images:
            self.image_handler.roi_table = ip.filter_rois_in_nucs(self.image_handler.roi_table.copy(), np.array(self.image_handler.images['nuc_classes']), self.pos_list, new_col='nuc_class', drifts = self.image_handler.tables['nuc_images_drift_correction'], target_frame=self.config['nuc_ref_frame'])

        print('ROIs (re)assigned to nuclei.')
        self.image_handler.roi_table.to_csv(self.roi_path)
        self.image_handler.load_tables()
    '''
    def make_dc_rois_all_frames(self) -> str:
        #Precalculate all ROIs for extracting spot images, based on identified ROIs and precalculated drifts between time frames.
        print('Generating list of all ROIs for tracing:')

        all_rois = []
        
        if self.config.get('spot_in_nuc', False):
            key_rois_table = self.input_name + '_rois' + NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION
            filter_rois_table = lambda t: t.loc[t['nuc_label'] != 0]
        else:
            key_rois_table = self.input_name + '_rois'
            filter_rois_table = lambda t: t
        
        try:
            rois_table = self.image_handler.tables[key_rois_table]
        except KeyError as e:
            raise MissingRoisTableException(key_rois_table) from e
        rois_table = filter_rois_table(rois_table)
        
        for _, roi in tqdm.tqdm(rois_table.iterrows(), total=len(rois_table)):
            pos = roi['position']
            pos_index = self.image_handler.image_lists[self.input_name].index(pos)
            dc_pos_name = self.image_handler.image_lists[self.config['reg_input_moving']][pos_index] # not unused; used for table query
            sel_dc = self.image_handler.tables[self.input_name + '_drift_correction_fine'].query('position == @dc_pos_name')
            ref_frame = roi['frame']
            ch = roi['ch']
            ref_offset = sel_dc.query('frame == @ref_frame')
            Z, Y, X = self.images[pos_index][0, ch].shape[-3:]
            for j, dc_frame in sel_dc.iterrows():
                z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])

                z_min = max(roi['z_min'] - z_drift_course, 0)
                z_max = min(roi['z_max'] - z_drift_course, Z)
                y_min = max(roi['y_min'] - y_drift_course, 0)
                y_max = min(roi['y_max'] - y_drift_course, Y)
                x_min = max(roi['x_min'] - x_drift_course, 0)
                x_max = min(roi['x_max'] - x_drift_course, X)

                pad_z_min = abs(min(0,z_min))
                pad_z_max = abs(max(0,z_max-Z))
                pad_y_min = abs(min(0,y_min))
                pad_y_max = abs(max(0,y_max-Y))
                pad_x_min = abs(min(0,x_min))
                pad_x_max = abs(max(0,x_max-X))

                all_rois.append([pos, pos_index, roi.name, dc_frame['frame'], ref_frame, ch, 
                                z_min, z_max, y_min, y_max, x_min, x_max, 
                                pad_z_min, pad_z_max, pad_y_min, pad_y_max, pad_x_min, pad_x_max,
                                z_drift_course, y_drift_course, x_drift_course, 
                                dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])

        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 
                                'z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max',
                                'pad_z_min', 'pad_z_max', 'pad_y_min', 'pad_y_max', 'pad_x_min', 'pad_x_max', 
                                'z_px_course', 'y_px_course', 'x_px_course',
                                'z_px_fine', 'y_px_fine', 'x_px_fine'])
        self.all_rois = self.all_rois.sort_values(RoiOrderingSpecification.row_order_columns()).reset_index(drop=True)
        print(self.all_rois)
        outfile = self.dc_roi_path
        self.all_rois.to_csv(outfile)
        self.image_handler.load_tables()
        return outfile

    def extract_single_roi_img(self, single_roi):
        #Function to extract single ROI lazily without loading entire stack in RAM.
        #Depending on chunking of original data can be more or less performant.

        p = single_roi['pos_index']
        t = single_roi['frame']
        c = single_roi['ch']
        z = slice(single_roi['z_min'], single_roi['z_max'])
        y = slice(single_roi['y_min'], single_roi['y_max'])
        x = slice(single_roi['x_min'], single_roi['x_max'])
        pad = ( (single_roi['pad_z_min'], single_roi['pad_z_max']),
                (single_roi['pad_y_min'], single_roi['pad_y_max']),
                (single_roi['pad_x_min'], single_roi['pad_x_max']))

        try:
            roi_img = np.array(self.images[p][t, c, z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros(self._raw_roi_image_size, dtype=np.float32)

        return roi_img

    def extract_single_roi_img_inmem(self, single_roi, images):
        # Function for extracting a single cropped region defined by ROI from a larger 3D image.
        from math import ceil, floor
        down = lambda x: int(floor(x))
        up = lambda x: int(ceil(x))
        z = slice(down(single_roi['z_min']), up(single_roi['z_max']))
        y = slice(down(single_roi['y_min']), up(single_roi['y_max']))
        x = slice(down(single_roi['x_min']), up(single_roi['x_max']))
        pad = ( (single_roi['pad_z_min'], single_roi['pad_z_max']),
                (single_roi['pad_y_min'], single_roi['pad_y_max']),
                (single_roi['pad_x_min'], single_roi['pad_x_max']))

        try:
            roi_img = np.array(images[z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros((np.abs(z.stop-z.start), np.abs(y.stop-y.start), np.abs(x.stop-x.start)), dtype=np.float32)

        return roi_img
    
    def write_single_fov_data(self, pos_group_name: str, pos_group_data: pd.DataFrame) -> List[str]:
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
        list of str
            Names of the files written
        """
        pos_index = self.image_handler.image_lists[self.input_name].index(pos_group_name)
        f_id = 0
        n_frames = len(pos_group_data.frame.unique())
        array_files = []
        for frame, frame_group in tqdm.tqdm(pos_group_data.groupby('frame')):
            for ch, ch_group in frame_group.groupby('ch'):
                image_stack = np.array(self.images[pos_index][int(frame), int(ch)])
                for i, roi in ch_group.iterrows():
                    roi_img = self.extract_single_roi_img_inmem(roi, image_stack).astype(np.uint16)
                    fp = os.path.join(self.spot_images_path, RoiOrderingSpecification.name_roi_file(pos_name=pos_group_name, roi=roi))
                    if f_id == 0:
                        array_files.append(fp)
                        arr = open_memmap(fp, mode='w+', dtype = roi_img.dtype, shape=(n_frames,) + roi_img.shape)
                        arr[f_id] = roi_img
                        arr.flush()
                    else:
                        arr = open_memmap(fp, mode='r+')
                        try:
                            arr[f_id] = roi_img
                            arr.flush()
                            #arr[f_id] = np.append(arr[f_id], np.expand_dims(roi_img,0).copy(), axis=0)
                        except ValueError: #Edge case: ROI fetching has failed giving strange shaped ROI, just leave the zeros as is.
                            pass
                            # roi_stack = np.append(roi_stack, np.expand_dims(np.zeros_like(roi_stack[0]), 0), axis=0)
            f_id += 1
        return array_files


    def gen_roi_imgs_inmem(self) -> str:
        # Load full stacks into memory to extract spots.
        # Not the most elegant, but depending on the chunking of the original data it is often more performant than loading subsegments.

        rois = self.image_handler.tables[self.input_name+'_dc_rois']
        if self.array_id is not None:
            pos_name = self.image_handler.image_lists[self.input_name][self.array_id]
            rois = rois[rois.position == pos_name]

        if not os.path.isdir(self.spot_images_path):
            os.mkdir(self.spot_images_path)

        for pos, pos_group in tqdm.tqdm(rois.groupby('position')):
            self.write_single_fov_data(pos_group_name=pos, pos_group_data=pos_group)
            
        return self.spot_images_path

    def gen_roi_imgs_inmem_coursedc(self) -> str:
        # Use this simplified function if the images that the spots are gathered from are already coursely drift corrected!
        print('Generating single spot image stacks from coursely drift corrected images.')
        rois = self.image_handler.tables[self.input_name+'_dc_rois']
        for pos, group in tqdm.tqdm(rois.groupby('position')):
            pos_index = self.image_handler.image_lists[self.input_name].index(pos)
            full_image = np.array(self.image_handler.images[self.input_name][pos_index])
            for roi in group.to_dict('records'):
                spot_stack = full_image[:, 
                                roi['ch'], 
                                roi['z_min']:roi['z_max'], 
                                roi['y_min']:roi['y_max'],
                                roi['x_min']:roi['x_max']].copy()
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
