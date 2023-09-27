# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from enum import Enum
from functools import partial
import os
from typing import *

import numpy as np
import pandas as pd
import dask.array as da
from dask.delayed import delayed
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.stats import trim_mean
import tqdm

from looptrace import image_processing_functions as ip
from looptrace import image_io
from looptrace.gaussfit import fitSymmetricGaussian3D
from looptrace.numeric_types import FloatLike, NumberLike
from looptrace.wrappers import phase_cross_correlation


DUMMY_SHIFT = [0, 0, 0]

DriftTableRow = List[Union[str, NumberLike]]


def get_method_name(config: Mapping[str, Any]) -> Optional[str]:
    """Get the name of the drift correction method from configuration data."""
    return config.get("dc_method")


class Methods(Enum):
    COARSE_NAME = "course"
    CROSS_CORRELATION_NAME = "cc"
    FIT_NAME = "fit"

    @classmethod
    def is_valid_name(cls, name: str) -> bool:
        """Determine if the given value is a valid drift correction method name."""
        return name in cls.__members__
    
    @classmethod
    def values(cls) -> Iterable[str]:
        """Iterator over the drift correction method names"""
        return (m.value for m in cls)
    
    @classmethod
    def get_func_and_args_getter(cls, method_name: str) -> Tuple[callable, Callable[[Tuple[np.ndarray, np.ndarray]], Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, int]]]]:
        """
        Attempt to look up the drift correction fnuction and arguments getter, based on the given method name.

        Parameters
        ----------
        method_name : str
            The name of the alleged drift correction method
        
        Returns
        -------
        Pair in which the first element is the function with which to compute drift correction shift, and the 
        second element is the function with which to obtain arguments from the pair of images also being used 
        as arguments to the first function in this returned pair
        """
        method_lookup = {
            Methods.CROSS_CORRELATION_NAME.value: (correlate_single_bead, lambda img_pair: img_pair + (100, )), 
            Methods.FIT_NAME.value: (fit_shift_single_bead, lambda img_pair: img_pair)
        }
        try:
            return method_lookup[method_name]
        except KeyError as e:
            raise NotImplementedError(f"Unknown drift correction method ({method_name}); choose from: {', '.join(method_lookup.keys())}") from e


def correlate_single_bead(t_bead, o_bead, upsampling):
    """
    Use scikit-image's phase_cross_correlation funnction to compute the best shift to make two beads coincide.

    Parameters
    ----------
    t_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for one bead image; "reference_image" in scikit-image terms
    o_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for other bead image; "moving_image" in scikit-image terms
    upsampling : int, optional

    Returns
    -------
    """
    try:
        shift, _, _ = phase_cross_correlation(t_bead, o_bead, upsample_factor=upsampling)
    except (ValueError, AttributeError):
        shift = np.array(DUMMY_SHIFT)
    return shift


def fit_shift_single_bead(t_bead, o_bead, strict: bool = False):
    """
    Fit the center of two beads using 3D gaussian fit, and fit the shift between the centers.

    Parameters
    ----------
    t_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for one bead image; "reference_image" in scikit-image terms
    o_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for other bead image; "moving_image" in scikit-image terms
    strict : bool, default False
        Whether the function must succeed; if an error occurs, throw it if this is True, 
        but if it's False, just print the error and return all-0s for the shift.

    Returns
    -------
    np.ndarray
        1D array with 3 values, representing the shift (in pixels) necessary to align (as 
        much as possible) the centers of the Gaussian distributions fit to each of the bead images
    
    Raises
    ------
    AttributeError or ValueError
        If something goes wrong with fitting the beads or taking the fits' coordinates' difference, 
        and strict = True
    """
    try:
        t_fit = np.array(fitSymmetricGaussian3D(t_bead, sigma=1, center=None)[0])
        o_fit = np.array(fitSymmetricGaussian3D(o_bead, sigma=1, center=None)[0])
        shift = t_fit[2:5] - o_fit[2:5]
    except (ValueError, AttributeError) as e: # TODO: when can this happen? -- narrow exceptions / provide better insight.
        if strict:
            print("Error finding shift between beads!")
            print(f"Bead 1 (below):")
            print(t_bead)
            print(f"Bead 2 (below):")
            print(o_bead)
            raise
        else:
            print(f"Error: {e}")
            shift = np.array(DUMMY_SHIFT)
    return shift


class ArrayLikeLengthMismatchError(Exception):
    """Exception subtype for when array-like objects of expected-equal length don't match on length"""


def generate_drift_function_arguments__coarse_drift_only(
        full_pos_list: List[str], 
        pos_list: Iterable[str], 
        reference_images: List[np.ndarray], 
        reference_frame: int, 
        reference_channel: int, 
        moving_images: List[np.ndarray], 
        moving_channel: int, 
        downsampling: int,
        ) -> Iterable[Tuple[str, int, np.ndarray, np.ndarray]]:
    """
    Generate the coarse and fine drift correction shifts, when just doing coarse correction (dummy values for fine DC).

    Parameters
    ----------
    full_pos_list : list of str
        Names of all known positions / fields of view
    pos_list : Iterable of str
        Names over which to iterate
    reference_images : list of np.ndarray
        The image arrays on which to base the shifts
    reference_frame : int
        The index of the frame (hybridisation round) on which the shifts are to be based
    reference_channel : int
        The index of the channel in which the reference signal is imaged (e.g., beads channel)
    moving_images : list of np.ndarray
        The images to shift to align to reference
    moving_channel : int
        The channel in signal to shift was imaged
    downsampling : int
        A factor by which to downsample the reference and moving signal, by taking every nth entry of an arrayt
    
    Raises
    ------
    ArrayLikeLengthMismatchError
        If the full list of positions doesn't match in length to the list of reference images, 
        or if the full list of positions doesn't match in length to the list of moving images
    """
    if len(full_pos_list) != len(reference_images) or len(full_pos_list) != len(moving_images):
        raise ArrayLikeLengthMismatchError(f"Full pos: {len(full_pos_list)}, ref imgs: {len(reference_images)}, mov imgs: {len(moving_images)}")
    for pos in pos_list:
        print(f'Running only course drift correction for position: {pos}.')
        i = full_pos_list.index(pos)
        t_img = np.array(reference_images[i][reference_frame, reference_channel, ::downsampling, ::downsampling, ::downsampling])
        for t in tqdm.tqdm(range(moving_images[i].shape[0])):
            o_img = np.array(moving_images[i][t, moving_channel, ::downsampling, ::downsampling, ::downsampling])
            yield pos, t, t_img, o_img
        print(f'Finished drift correction for position: {pos}')


def process_single_fov_single_frame__coarse_only(pos: str, frame: int, t_img: np.ndarray, o_img: np.ndarray) -> DriftTableRow:
    """
    Compute coarse drift for a single (FOV, frame) combination, passing through those values and addind dummy values for fine DC.

    Parameters
    ----------
    pos : str
        Name of the FOV from which the imaging data comes
    frame : int
        Index of the hybridisation timepoint for which the coarse drift/shift is to be computed
    t_img : np.ndarray
        Reference image
    o_img : np.ndarray
        Image for which drift/shift relative to reference is to be computed
    
    Returns
    -------
    DriftTableRow
        A list which will become a row/record in a data frame, with first value representing the index of the 
        hybridisation round / timepoint, the second representing the name of the field of view, and the rest 
        representing 6 values for coarse and fine drift correction (first three coarse, second three fine).
    """
    shift = ip.drift_corr_course(t_img=t_img, o_img=o_img, downsample=1)
    return _create_drift_table_record(pos=pos, frame=frame, coarse_drift=shift, fine_drift=DUMMY_SHIFT)


def generate_drift_function_arguments__coarse_and_fine(
    full_pos_list: List[str], 
    pos_list: Iterable[str], 
    reference_images: List[np.ndarray], 
    reference_frame: int, 
    reference_channel: int, 
    moving_images: List[np.ndarray], 
    moving_channel: int, 
    get_bead_rois: Callable[[np.ndarray], Iterable[Iterable[int]]], 
    get_ref_bead_img: Callable[[Iterable[int], np.ndarray], np.ndarray],
    ) -> Iterable[Tuple[str, int, np.ndarray, np.ndarray]]:
    for pos in pos_list:
        i = full_pos_list.index(pos)
        print(f'Running drift correction for position: {pos}')
        t_img = np.array(reference_images[i][reference_frame, reference_channel])
        bead_rois = get_bead_rois(t_img)
        t_bead_imgs = [get_ref_bead_img(point, t_img) for point in bead_rois]
        for t in tqdm.tqdm(range(moving_images[i].shape[0])):
            o_img = np.array(moving_images[i][t, moving_channel])
            yield pos, t, t_img, o_img, bead_rois, t_bead_imgs


def process_single_fov_single_frame__coarse_and_fine(
        pos: str, 
        frame: int, 
        t_img: np.ndarray, 
        o_img: np.ndarray, 
        bead_rois: Iterable[np.ndarray], 
        t_bead_imgs: Iterable[np.ndarray], 
        ds: int, 
        corr_func: callable,
        get_args: callable,
        ) -> DriftTableRow:
    coarse = ip.drift_corr_course(t_img=t_img, o_img=o_img, downsample=ds)
    o_bead_imgs = Parallel(n_jobs=-1, prefer='threads')(delayed(ip.extract_single_bead)(point, o_img, drift_course=coarse) for point in bead_rois)
    if len(bead_rois) > 0:
        print("Computing fine drift")
        fine = Parallel(n_jobs=-1, prefer='threads')(delayed(corr_func)(*get_args(img_pair)) for img_pair in zip(t_bead_imgs, o_bead_imgs))
        fine = np.array(fine)
        fine = trim_mean(fine, proportiontocut=0.2, axis=0)
    else:
        print("No bead ROIs, setting fine drift to all-0s")
        fine = DUMMY_SHIFT
    return _create_drift_table_record(pos=pos, frame=frame, coarse_drift=coarse, fine_drift=fine)


def _create_drift_table_record(pos: str, frame: int, coarse_drift: Iterable[FloatLike], fine_drift: Iterable[NumberLike]) -> DriftTableRow:
    return [frame, pos] + list(coarse_drift) + list(fine_drift)


class Drifter():

    def __init__(self, image_handler, array_id = None):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.image_handler = image_handler
        self.config = self.image_handler.config
        self.images_template = self.image_handler.images[self.image_handler.reg_input_template]
        self.images_moving = self.image_handler.images[self.image_handler.reg_input_moving]
        self.full_pos_list = self.image_handler.image_lists[self.image_handler.reg_input_moving]
        self.pos_list = self.full_pos_list
        
        self.bead_roi_px = self.config.get('bead_roi_size', 15)
        
        if array_id is not None:
            self.dc_file_path = self.image_handler.out_path(self.image_handler.reg_input_moving + '_drift_correction.csv'[:-4]+'_' + str(array_id).zfill(4) + '.csv')
            self.pos_list = [self.pos_list[int(array_id)]]
        else:
            self.dc_file_path = self.image_handler.out_path(self.image_handler.reg_input_moving + '_drift_correction.csv')

    @property
    def method_name(self):
        return get_method_name(self.config) or Methods.CROSS_CORRELATION_NAME.value

    def drift_corr(self) -> Optional[str]:
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images/arrays.
        Settings set in config file.

        '''
        reference_frame = self.config['reg_ref_frame']
        reference_channel = self.config['reg_ch_template']
        moving_channel = self.config['reg_ch_moving']
        downsampling = self.config['course_drift_downsample']

        dc_method = self.method_name
        print(f"Drift correction method: {dc_method}")

        if dc_method == Methods.COARSE_NAME.value:
            # NB: as in upstream looptrace, we pass downsampling = 1 for the actual call to the 
            #     coarse drift computation function; we do the downsampling in the determination 
            #     of the arguments, stepping through the image arrays according to the downsampling factor.
            all_drifts = Parallel(n_jobs=-1, prefer='threads')(
                delayed(process_single_fov_single_frame__coarse_only)(pos, frame, t_img, o_img) 
                    for pos, frame, t_img, o_img in 
                    generate_drift_function_arguments__coarse_drift_only(
                        full_pos_list=self.full_pos_list, 
                        pos_list=self.pos_list, 
                        reference_images=self.images_template, 
                        reference_frame=reference_frame, 
                        reference_channel=reference_channel,
                        moving_images=self.images_moving, 
                        moving_channel=moving_channel, 
                        downsampling=downsampling,
                    )
                )
        else:
            corr_func, get_args = Methods.get_func_and_args_getter(dc_method)
            get_bead_rois = partial(
                ip.generate_bead_rois, 
                threshold=self.config['bead_threshold'], 
                min_bead_int=self.config['min_bead_intensity'], 
                bead_roi_px=self.bead_roi_px, 
                n_points=self.config['bead_points'],
                )
            get_ref_bead_img = partial(ip.extract_single_bead, bead_roi_px=self.bead_roi_px)
            # Run drift correction for each position and save results in table.
            all_drifts = Parallel(n_jobs=-1, prefer='threads')(
                delayed(process_single_fov_single_frame__coarse_and_fine)(pos, t, t_img, o_img, bead_rois, t_bead_imgs, downsampling, corr_func, get_args)
                    for pos, t, t_img, o_img, bead_rois, t_bead_imgs in
                    generate_drift_function_arguments__coarse_and_fine(
                        full_pos_list=self.full_pos_list,
                        pos_list=self.pos_list, 
                        reference_images=self.images_template, 
                        reference_frame=reference_frame, 
                        reference_channel=reference_channel,
                        moving_images=self.images_moving, 
                        moving_channel=moving_channel, 
                        get_bead_rois=get_bead_rois,
                        get_ref_bead_img=get_ref_bead_img,
                    )
                )                    
        
        all_drifts=pd.DataFrame(all_drifts, columns=['frame',
                                                    'position',
                                                    'z_px_course',
                                                    'y_px_course',
                                                    'x_px_course',
                                                    'z_px_fine',
                                                    'y_px_fine',
                                                    'x_px_fine',
                                                    ])
        
        outfile = self.dc_file_path
        all_drifts.to_csv(outfile)
        print('Drift correction complete.')
        self.image_handler.drift_table = all_drifts
        return outfile

    def gen_dc_images(self, pos):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''
        n_t = self.images[0].shape[0]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
            pos_img.append(da.roll(self.images[pos_index][t], shift = shift, axis = (1,2,3)))
        self.dc_images = da.stack(pos_img)

        print('DC images generated.')

    def save_proj_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)
        '''

        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.full_pos_list.index(pos)
            pos_img = self.images_moving[pos_index]
            proj_img = da.max(pos_img, axis=2)
            zarr_out_path = os.path.join(self.image_handler.image_save_path, self.image_handler.reg_input_moving + '_max_proj_dc')
            z = image_io.create_zarr_store(
                path=zarr_out_path,
                name = self.image_handler.reg_input_moving + '_max_proj_dc', 
                pos_name = pos,
                shape = proj_img.shape, 
                dtype = np.uint16,  
                chunks = (1,1,proj_img.shape[-2], proj_img.shape[-1]),
                )

            n_t = proj_img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.image_handler.reg_input_moving + '_drift_correction'].query('position == @pos').iloc[t][['y_px_course', 'x_px_course', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                z[t] = ndi.shift(proj_img[t].compute(), shift=(0,)+shift, order = 2)
        
        print('DC images generated.')
    
    def save_course_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)
        '''

        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.image_handler.image_lists['seq_images'].index(pos)
            pos_img = self.images[pos_index]
            #proj_img = da.max(pos_img, axis=2)
            zarr_out_path = os.path.join(self.image_handler.image_save_path, self.image_handler.reg_input_moving + '_course_dc')
            z = image_io.create_zarr_store(path=zarr_out_path,
                                            name = self.image_handler.reg_input_moving + '_dc_images', 
                                            pos_name = pos,
                                            shape = pos_img.shape, 
                                            dtype = np.uint16,  
                                            chunks = (1,1,1,pos_img.shape[-2], pos_img.shape[-1]))

            n_t = pos_img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.image_handler.reg_input_moving + '_drift_correction'].query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']]
                shift = (shift[0], shift[1], shift[2])
                z[t] = ndi.shift(pos_img[t].compute(), shift=(0,)+shift, order = 0)
        
        print('DC images generated.')