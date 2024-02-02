# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import dataclasses
from enum import Enum
from itertools import takewhile
import os
from pathlib import Path
import sys
from typing import *

import numpy as np
import pandas as pd
import dask.array as da
from dask.delayed import delayed
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.stats import trim_mean
import tqdm

from gertils import ExtantFile, ExtantFolder

from looptrace import image_io, read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.bead_roi_generation import BeadRoiParameters, extract_single_bead
from looptrace.gaussfit import fitSymmetricGaussian3D
from looptrace.numeric_types import FloatLike, NumberLike
from looptrace.wrappers import phase_xcor


FRAME_COLUMN = "frame"
POSITION_COLUMN = "position"
CoarseDriftTableRow = Tuple[int, str, NumberLike, NumberLike, NumberLike]
Z_PX_COARSE = 'z_px_coarse'
Y_PX_COARSE = 'y_px_coarse'
X_PX_COARSE = 'x_px_coarse'
COARSE_DRIFT_COLUMNS = [Z_PX_COARSE, Y_PX_COARSE, X_PX_COARSE]
COARSE_DRIFT_TABLE_COLUMNS = [FRAME_COLUMN, POSITION_COLUMN] + COARSE_DRIFT_COLUMNS
FullDriftTableRow = Tuple[int, str, NumberLike, NumberLike, NumberLike, NumberLike, NumberLike, NumberLike]
FULL_DRIFT_TABLE_COLUMNS = COARSE_DRIFT_TABLE_COLUMNS + ['z_px_fine', 'y_px_fine', 'x_px_fine']
DUMMY_SHIFT = [0, 0, 0]


def get_method_name(config: Mapping[str, Any]) -> Optional[str]:
    """Get the name of the drift correction method from configuration data."""
    return config.get("dc_method")


class Methods(Enum):
    CROSS_CORRELATION_NAME = "cc"
    FIT_NAME = "fit"

    @classmethod
    def is_valid_name(cls, name: str) -> bool:
        """Determine if the given value is a valid drift correction method name."""
        return name in (e.value for e in cls)
    
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
    Use scikit-image's phase_cross_correlation function to compute the best shift to make two beads coincide.

    Parameters
    ----------
    t_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for one bead image; "reference_image" in scikit-image terms
    o_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for other bead image; "moving_image" in scikit-image terms
    upsampling : int, optional

    Returns
    -------
    np.ndarray
        3-element vector representing the computed offset between the reference/"template" image and an image 
        that should align with that one; dummy values of 0s if something goes wrong with the computation
    """
    try:
        shift = phase_xcor(t_bead, o_bead, upsample_factor=upsampling)
    except (ValueError, AttributeError):
        # TODO: log if all-0s is used here?
        shift = np.array(DUMMY_SHIFT)
    return shift


def fit_bead_coordinates(bead_img: np.ndarray) -> Iterable[FloatLike]:
    try:
        return np.array(fitSymmetricGaussian3D(bead_img, sigma=1, center=None)[0])[2:5]
    except (AttributeError, ValueError) as e: # TODO: when can this happen? -- narrow exceptions / provide better insight.
        print(f"Error finding shift between beads: {e}")
        return None


def subtract_point_fits(ref_fit: np.ndarray, mov_fit: np.ndarray) -> np.ndarray:
    # This should accord with the relation of reference image, offset image, and offset (drift) in phase_cross_correlation.
    # https://github.com/gerlichlab/looptrace/issues/194
    # https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html#image-registration
    try:
        diff = ref_fit - mov_fit
    except TypeError: # One or both of the fits is null.
        diff = np.array(DUMMY_SHIFT)
    return diff


def fit_shift_single_bead(t_bead: np.ndarray, o_bead: np.ndarray):
    """
    Fit the center of two beads using 3D gaussian fit, and fit the shift between the centers.

    Parameters
    ----------
    t_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for one bead image; "reference_image" in scikit-image terms
    o_bead : np.ndarray
        3D array (z, y, x) of pixel intensities for other bead image; "moving_image" in scikit-image terms

    Returns
    -------
    np.ndarray
        1D array with 3 values, representing the shift (in pixels) necessary to align (as 
        much as possible) the centers of the Gaussian distributions fit to each of the bead images
    """
    t_fit = fit_bead_coordinates(t_bead)
    o_fit = fit_bead_coordinates(o_bead)
    return subtract_point_fits(ref_fit=t_fit, mov_fit=o_fit)


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
    nuclei_mode: bool,
    stop_after: int = sys.maxsize,
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
        A factor by which to downsample the reference and moving signal, by taking every nth entry of an array
    nuclei_mode : bool
        Whether the arguments are being generated for nuclei, changing expected dimensionality of moving images
    stop_after : int, default infinite
        Point (position) after which to stop the generation
    
    Returns
    -------
    Iterable of (str, int, np.ndarray, np.ndarray)
        Bundle of position (FOV), frame (hybdridisation timepoint), reference image, and shifted image

    Raises
    ------
    ArrayLikeLengthMismatchError
        If the full list of positions doesn't match in length to the list of reference images, 
        or if the full list of positions doesn't match in length to the list of moving images
    """
    if len(full_pos_list) != len(reference_images) or len(full_pos_list) != len(moving_images):
        raise ArrayLikeLengthMismatchError(f"Full pos: {len(full_pos_list)}, ref imgs: {len(reference_images)}, mov imgs: {len(moving_images)}")
    for i, pos in takewhile(lambda i_and_p: i_and_p[0] <= stop_after, map(lambda p: (full_pos_list.index(p), p), pos_list)):
        print(f'Running coarse drift correction for position: {pos}.')
        t_img = np.array(reference_images[i][reference_frame, reference_channel, ::downsampling, ::downsampling, ::downsampling])
        mov_img = moving_images[i]
        if nuclei_mode:
            o_img = np.array(mov_img[moving_channel, ::downsampling, ::downsampling, ::downsampling])
            yield pos, 0, t_img, o_img
        else:
            for t in tqdm.tqdm(range(mov_img.shape[0])):
                o_img = np.array(mov_img[t, moving_channel, ::downsampling, ::downsampling, ::downsampling])
                yield pos, t, t_img, o_img
        print(f"Finished drift correction for position: {pos}")


@dataclasses.dataclass
class JoblibParallelSpecification:
    """Bundle of the parameters to pass to joblib.Parallel"""
    n_jobs: int
    prefer: str


@dataclasses.dataclass
class MultiprocessingPoolSpecification:
    """Bundle of parameters to pass to multiprocessing.Pool"""
    n_workers: int    


def coarse_correction_workflow(config_file: ExtantFile, images_folder: ExtantFolder):
    """The workflow for the initial (and sometimes only), coarse, drift correction."""
    D = Drifter(image_handler=ImageHandler(config_file, images_folder))
    try:
        pos_halt_point = D.config["dc_pos_limit"]
    except KeyError:
        pos_halt_point = sys.maxsize
        update_outfile = lambda fp: fp
    else:
        update_outfile = lambda fp: fp.with_suffix(f".halt_after_{pos_halt_point}.csv")
    all_args = generate_drift_function_arguments__coarse_drift_only(
        full_pos_list=D.full_pos_list, 
        pos_list=D.pos_list, 
        reference_images=D.images_template, 
        reference_frame=D.reference_frame, 
        reference_channel=D.reference_channel,
        moving_images=D.images_moving, 
        moving_channel=D.moving_channel, 
        downsampling=D.downsampling,
        nuclei_mode=False,
        stop_after=pos_halt_point,
    )
    print("Computing coarse drifts...")
    records = Parallel(n_jobs=-1, prefer='threads')(
        delayed(lambda p, t, ref_ds, mov_ds: (t, p) + tuple(phase_xcor(ref_ds, mov_ds) * D.downsampling))(*args) 
        for args in all_args
        )
    try:
        coarse_drifts = pd.DataFrame(records, columns=COARSE_DRIFT_TABLE_COLUMNS)
    except ValueError: # most likely if element count of one or more rows doesn't match column count
        print(f"Example record (below):\n{records[0]}")
        raise
    outfile = update_outfile(D.dc_file_path__coarse)
    print(f"Writing coarse drifts: {outfile}")
    coarse_drifts.to_csv(outfile)
    return outfile


def fine_correction_workflow(config_file: ExtantFile, images_folder: ExtantFolder) -> str:
    """The workflow for the second, optional, fine drift correction"""
    D = Drifter(image_handler=ImageHandler(config_file, images_folder))
    print("Computing fine drifts")
    compute_fine_drifts(D)
    all_drifts = D.read_all_fine_drifts() # Read each per-FOV file, aggregating to single table
    all_drifts.columns = FULL_DRIFT_TABLE_COLUMNS
    outfile = D.dc_file_path__fine
    print(f"Writing fine drifts: {outfile}")
    all_drifts.to_csv(outfile) # Write the single full drift correction table.
    return outfile


def iter_coarse_drifts_by_position(filepath: Union[str, Path, ExtantFile]) -> Iterable[Tuple[str, pd.DataFrame]]:
    print(f"Reading coarse drift table: {filepath}")
    coarse_table = read_table_pandas(filepath)
    coarse_table = coarse_table.sort_values([POSITION_COLUMN, FRAME_COLUMN]) # Sort so that grouping by position then frame doesn't alter order.
    return coarse_table.groupby(POSITION_COLUMN)


def _get_frame_and_coarse(row) -> Tuple[int, Tuple[int, int, int]]:
    return row[FRAME_COLUMN], tuple(row[COARSE_DRIFT_COLUMNS])


def compute_fine_drifts(drifter: "Drifter") -> None:
    """
    Compute the fine drifts, using what's already been done for coarse drifts.

    Parameters
    ----------
    drifter : The drift correction abstraction, managing paths for relevant files

    Returns
    -------
    Iterable of tuple
        Data for each row of the fine/full drift correction table
    """
    roi_px = drifter.bead_roi_px
    beads_exp_shape = (drifter.num_bead_rois_for_drift_correction, 3)
    pos_time_problems = drifter.image_handler.position_frame_pairs_with_severe_problems
    for position, position_group in iter_coarse_drifts_by_position(filepath=drifter.dc_file_path__coarse):
        pos_idx = drifter.full_pos_list.index(position)
        if not drifter.overwrite and drifter.checkpoint_filepath(pos_idx=pos_idx).is_file():
            print(f"Fine DC checkpoint exists, skipping FOV: {pos_idx}")
            continue
        print(f"Running fine drift correction for position {position} (index {pos_idx})")
        ref_img = drifter.get_reference_image(pos_idx)
        get_no_partition_message = lambda t: f"No bead ROIs partition for (pos={pos_idx}, frame={t})"
        
        bead_rois = drifter.image_handler.read_bead_rois_file_shifting(pos_idx=pos_idx, frame=drifter.reference_frame)
        
        if bead_rois.shape != beads_exp_shape:
            msg_base = f"Unexpected bead ROIs shape for reference (pos={pos_idx}, frame={drifter.reference_frame})! ({bead_rois.shape}), expecting {beads_exp_shape}"
            if len(bead_rois.shape) == 2 and bead_rois.shape[1] == beads_exp_shape[1]:
                print(f"WARNING: {msg_base}")
            else:
                raise Exception(msg_base)
        
        curr_position_rows = []
        if drifter.method_name == Methods.FIT_NAME.value:
            print("Computing reference bead fits")
            ref_bead_subimgs = Parallel(n_jobs=-1, prefer='threads')(delayed(extract_single_bead)(pt, ref_img, bead_roi_px=roi_px) for pt in tqdm.tqdm(bead_rois))
            ref_bead_fits = Parallel(n_jobs=-1, prefer='threads')(delayed(fit_bead_coordinates)(rbi) for rbi in tqdm.tqdm(ref_bead_subimgs))
            print("Iterating over timepoints/hybridisations")
            # Exactly 1 row per FOV per timepoint; here, we iterate over timepoints for current FOV.
            for _, row in position_group.iterrows():
                # This should be unique now in frame, since we're iterating within a single FOV.
                frame, coarse = _get_frame_and_coarse(row)
                print(f"Current frame: {frame}")
                
                # Use current (FOV, time) pair if and only if it has a beads partition defined.
                try:
                    _ = drifter.image_handler.get_bead_rois_file(pos_idx=pos_idx, frame=frame, purpose="shifting")
                except TypeError: # arises iff the file doesn't exist for current (FOV, time), because ExtantFile wrapping fails
                    print(f"WARNING: {get_no_partition_message(frame)}")
                    assert (pos_idx, frame) in pos_time_problems, \
                        f"No bead ROIs for fine DC for (FOV={pos_idx}, time={frame}), but no evidence of a problem there"
                    fine = (0.0, 0.0, 0.0)
                else:
                    mov_img = drifter.get_moving_image(pos_idx=pos_idx, frame_idx=frame)
                    print(f"Computing fine drifts: ({position}, {frame})")
                    mov_bead_subimgs = Parallel(n_jobs=-1, prefer='threads')(delayed(extract_single_bead)(pt, mov_img, bead_roi_px=roi_px, drift_coarse=coarse) for pt in tqdm.tqdm(bead_rois))
                    mov_bead_fits = Parallel(n_jobs=-1, prefer='threads')(delayed(fit_bead_coordinates)(mbi) for mbi in tqdm.tqdm(mov_bead_subimgs))
                    fine_drifts = [subtract_point_fits(ref, mov) for ref, mov in zip(ref_bead_fits, mov_bead_fits)]
                    fine = finalise_fine_drift(fine_drifts)
                curr_position_rows.append((frame, position) + coarse + fine)
        elif drifter.method_name == Methods.CROSS_CORRELATION_NAME.value:
            print("Extracting reference bead images")
            ref_bead_subimgs = [extract_single_bead(point, ref_img, bead_roi_px=roi_px) for point in bead_rois]
            print("Iterating over frames/timepoints/hybridisations")
            for _, row in position_group.iterrows():
                # This should be unique now in frame, since we're iterating within a single FOV.
                frame, coarse = _get_frame_and_coarse(row)
                print(f"Current frame: {frame}")
                
                # Use current (FOV, time) pair if and only if it has a beads partition defined.
                try:
                    _ = drifter.image_handler.get_bead_rois_file(pos_idx=pos_idx, frame=frame, purpose="shifting")
                except TypeError: # Current (FOV, time) pair doesn't have a beads partition defined.
                    print(f"WARNING: {get_no_partition_message(frame)}")
                    assert (pos_idx, frame) in pos_time_problems, \
                        f"No bead ROIs for fine DC for (FOV={pos_idx}, time={frame}), but no evidence of a problem there"
                    fine = (0.0, 0.0, 0.0)
                else:
                    mov_img = drifter.get_moving_image(pos_idx=pos_idx, frame_idx=frame)
                    print(f"Computing fine drifts: ({position}, {frame})")
                    fine_drifts = Parallel(n_jobs=-1, prefer='threads')(
                        delayed(lambda point, ref_bead_img: correlate_single_bead(ref_bead_img, extract_single_bead(point, mov_img, bead_roi_px=roi_px, drift_coarse=coarse), 100))(*args)
                        for args in tqdm.tqdm(zip(bead_rois, ref_bead_subimgs))
                        )
                    fine = finalise_fine_drift(fine_drifts)
                curr_position_rows.append((frame, position) + coarse + fine)
        else:
            raise Exception(f"Unknown drift correction method: {drifter.method_name}")
        
        drifter.fine_correction_subfolder.mkdir(exist_ok=True)
        curr_position_temp = drifter.fine_correction_tempfile(pos_idx=pos_idx)
        print(f"Writing drift correction tempfile for FOV {pos_idx}: {curr_position_temp}")
        pd.DataFrame(curr_position_rows).to_csv(curr_position_temp)
        checkpoint_file = drifter.fine_correction_subfolder / f"{pos_idx}.checkpoint"
        print(f"Touching checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'w'): # Just create the empty file.
            pass


def finalise_fine_drift(drift: Iterable[np.ndarray]) -> Tuple[FloatLike, FloatLike, FloatLike]:
    return tuple(trim_mean(np.array(drift), proportiontocut=0.2, axis=0))


class Drifter():

    def __init__(self, image_handler: ImageHandler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.image_handler = image_handler
        self.config = self.image_handler.config
    
    @property
    def bead_detection_max_intensity(self) -> Optional[int]:
        return self.config.get("max_bead_intensity")

    @property
    def bead_roi_max_size(self) -> int:
        return self.config.get("max_bead_roi_size", 500)

    @property
    def bead_roi_px(self) -> int:
        return self.config.get('bead_roi_size', 12)

    @property
    def bead_threshold(self) -> int:
        return self.config['bead_threshold']

    @property
    def bead_roi_parameters(self) -> BeadRoiParameters:
        return BeadRoiParameters(
            min_intensity_for_segmentation=self.bead_threshold, 
            min_intensity_for_detection=self.min_bead_intensity, 
            roi_pixels=self.bead_roi_px, 
            max_region_size=self.bead_roi_max_size, 
            max_intensity_for_detection=self.bead_detection_max_intensity,
            )

    @staticmethod
    def _checkpoint_filename(pos_idx: int) -> str:
        return f"{pos_idx}.checkpoint"

    def checkpoint_filepath(self, pos_idx: int) -> Path:
        return self.fine_correction_subfolder / self._checkpoint_filename(pos_idx)

    @property
    def dc_file_path__coarse(self) -> Path:
        return self.image_handler.drift_correction_file__coarse

    @property
    def dc_file_path__fine(self) -> Path:
        return self.image_handler.drift_correction_file__fine

    @property
    def downsampling(self) -> int:
        return self.config["coarse_drift_downsample"]

    @property
    def fine_correction_subfolder(self) -> Path:
        return Path(self.image_handler.analysis_path) / "fine_drift_temp"

    def fine_correction_tempfile(self, pos_idx: int) -> Path:
        return self.fine_correction_subfolder / f"{pos_idx}.dc_fine.tmp.csv"

    @property
    def full_pos_list(self) -> List[str]:
        return self.image_handler.drift_correction_position_names

    @property
    def images_moving(self) -> Sequence[np.ndarray]:
        return self.image_handler.drift_correction_moving_images

    @property
    def images_template(self) -> Sequence[np.ndarray]:
        return self.image_handler.drift_correction_reference_images

    @property
    def method_name(self) -> str:
        return get_method_name(self.config) or Methods.CROSS_CORRELATION_NAME.value

    @property
    def min_bead_intensity(self) -> int:
        return self.config['min_bead_intensity']

    @property
    def moving_channel(self) -> int:
        return self.image_handler.drift_correction_moving_channel

    @property
    def num_bead_rois_for_drift_correction(self) -> int:
        return self.image_handler.num_bead_rois_for_drift_correction

    @property
    def num_positions(self) -> int:
        return len(self.full_pos_list)

    @property
    def overwrite(self) -> bool:
        return self.config.get("overwrite_fine_dc", False)

    @property
    def pos_list(self) -> List[str]:
        return self.full_pos_list

    def read_all_fine_drifts(self) -> pd.DataFrame:
        files = [
            self.fine_correction_subfolder / fn 
            for fn in os.listdir(self.fine_correction_subfolder) 
            if os.path.splitext(fn)[1] == ".csv"
        ]
        files = sorted(files, key=lambda fp: int(fp.name.split(".")[0]))
        print(f"Reading {len(files)} fine drift correction file(s):\n" + "\n".join(p.name for p in files))
        return pd.concat((read_table_pandas(fp) for fp in files), axis=0, ignore_index=True)

    @property
    def reference_channel(self) -> int:
        return self.image_handler.drift_correction_reference_channel

    @property
    def reference_frame(self) -> int:
        return self.image_handler.drift_correction_reference_frame

    def get_moving_image(self, pos_idx: int, frame_idx: int) -> np.ndarray:
        return np.array(self.images_moving[pos_idx][frame_idx, self.moving_channel])

    def get_reference_image(self, pos_idx: int) -> np.ndarray:
        return np.array(self.images_template[pos_idx][self.reference_frame, self.reference_channel])

    def gen_dc_images(self, pos):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''
        n_t = self.images[0].shape[0]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('position == @pos').iloc[t][['z_px_coarse', 'y_px_coarse', 'x_px_coarse']])
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
                shift = self.image_handler.tables[self.image_handler.reg_input_moving + '_drift_correction'].query('position == @pos').iloc[t][['y_px_coarse', 'x_px_coarse', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                z[t] = ndi.shift(proj_img[t].compute(), shift=(0,)+shift, order = 2)
        
        print('DC images generated.')
    
    def save_coarse_dc_images(self):
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
            zarr_out_path = os.path.join(self.image_handler.image_save_path, self.image_handler.reg_input_moving + '_coarse_dc')
            z = image_io.create_zarr_store(path=zarr_out_path,
                                            name = self.image_handler.reg_input_moving + '_dc_images', 
                                            pos_name = pos,
                                            shape = pos_img.shape, 
                                            dtype = np.uint16,  
                                            chunks = (1,1,1,pos_img.shape[-2], pos_img.shape[-1]))

            n_t = pos_img.shape[0]
            
            for t in tqdm.tqdm(range(n_t)):
                shift = self.image_handler.tables[self.image_handler.reg_input_moving + '_drift_correction'].query('position == @pos').iloc[t][['z_px_coarse', 'y_px_coarse', 'x_px_coarse']]
                shift = (shift[0], shift[1], shift[2])
                z[t] = ndi.shift(pos_img[t].compute(), shift=(0,)+shift, order = 0)
        
        print('DC images generated.')
