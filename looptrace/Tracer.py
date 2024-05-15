# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import dataclasses
import itertools
import logging
from pathlib import Path
from typing import *

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from tqdm import tqdm

from gertils import ExtantFile, ExtantFolder
from gertils.types import TimepointFrom0
from numpydoc_decorator import doc

from looptrace import *
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SPOT_IMAGE_PIXEL_VALUE_TYPE, RoiOrderingSpecification
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from looptrace.image_io import NPZ_wrapper, write_jvm_compatible_zarr_store
from looptrace.numeric_types import NumberLike
from looptrace.tracing_qc_support import apply_frame_names_and_spatial_information

BOX_Z_COL = "spot_box_z"
BOX_Y_COL = "spot_box_y"
BOX_X_COL = "spot_box_x"
IMG_SIDE_LEN_COLS = [BOX_Z_COL, BOX_Y_COL, BOX_X_COL]
ROI_FIT_COLUMNS = ["BG", "A", "z_px", "y_px", "x_px", "sigma_z", "sigma_xy"]
MASK_FITS_ERROR_MESSAGE = "Masking fits for tracing currently isn't supported!"

logger = logging.getLogger()


@dataclasses.dataclass
class BackgroundSpecification:
    frame_index: int
    drifts: Iterable[np.ndarray]


@dataclasses.dataclass
class FunctionalForm:
    function: callable
    dimensionality: int

    def __post_init__(self) -> None:
        if self.dimensionality != 3:
            raise NotImplementedError("Only currently supporting dimensionality = 3 for functional form fit")


def run_frame_name_and_distance_application(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder,
    ) -> Tuple[pd.DataFrame, Path]:
    H = ImageHandler(
        rounds_config=rounds_config, 
        params_config=params_config, 
        images_folder=images_folder,
        )
    T = Tracer(H)
    traces = apply_frame_names_and_spatial_information(traces_file=T.traces_path, frame_names=H.frame_names)
    outfile = T.traces_path_enriched
    print(f"Writing enriched traces file: {outfile}")
    traces.to_csv(outfile)
    return traces, outfile


class Tracer:
    def __init__(self, image_handler: ImageHandler, trace_beads: bool = False):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.pos_list = self.image_handler.image_lists[image_handler.spot_input_name]
        if trace_beads:
            self.roi_table = image_handler.tables[image_handler.spot_input_name + '_bead_rois']
            finalise_suffix = lambda p: Path(str(p).replace(".csv", "_beads.csv"))
        else:
            self.roi_table = image_handler.tables[image_handler.spot_input_name + '_rois']
            finalise_suffix = lambda p: p
        self.all_rois = image_handler.tables[image_handler.spot_input_name + '_dc_rois']

        fit_func_specs = {
            'LS': FunctionalForm(function=fitSymmetricGaussian3D, dimensionality=3), 
            'MLE': FunctionalForm(function=fitSymmetricGaussian3DMLE, dimensionality=3)
            }
        fit_func_value = self.config['fit_func']
        try:
            self.fit_func_spec = fit_func_specs[fit_func_value]
        except KeyError as e:
            raise Exception(f"Unknown fitting function ('{fit_func_value}'); choose from: {', '.join(fit_func_specs.keys())}") from e
        
        # will be concatenated horizontally with fits; idempotent if already effectively unindexed
        self.all_rois = self.all_rois.reset_index(drop=True)
        
        self.traces_path = finalise_suffix(image_handler.traces_path)
        self.traces_path_enriched = finalise_suffix(image_handler.traces_path_enriched)

    @property
    def background_specification(self) -> BackgroundSpecification:
        bg_frame_idx = self.image_handler.background_subtraction_frame
        if bg_frame_idx is None:
            return None
        # TODO: note -- the drifts part of this is currently (2023-12-01) irrelevant, as the 'drifts' component of the background spec is unused.
        pos_drifts = self.drift_table[self.drift_table.position.isin(self.pos_list)][["z_px_fine", "y_px_fine", "x_px_fine"]].to_numpy()
        return BackgroundSpecification(frame_index=bg_frame_idx, drifts=pos_drifts - pos_drifts[bg_frame_idx])

    @property
    def drift_table(self) -> pd.DataFrame:
        return self.image_handler.spots_fine_drift_correction_table

    def write_all_spot_images_to_one_per_fov_zarr(self, overwrite: bool = False) -> List[Path]:
        name_data_pairs = compute_spot_images_multiarray_per_fov(
            npz=self._images_wrapper,
            # Add 1 to account for tracing the regional timepoint itself.
            locus_grouping=self.image_handler.locus_grouping,
        )
        return write_jvm_compatible_zarr_store(
            name_data_pairs, 
            root_path=self.locus_spots_visualisation_folder, 
            dtype=SPOT_IMAGE_PIXEL_VALUE_TYPE, 
            overwrite=overwrite,
        )

    @property
    def images(self) -> Iterable[np.ndarray]:
        """Iterate over the small, single spot images for tracing (1 per timepoint per ROI)."""
        _, keyed_filenames = _prep_npz_to_zarr(self._images_wrapper)
        for _, fn in keyed_filenames:
            yield self._images_wrapper[fn]

    @property
    def _input_name(self) -> str:
        return self.config['trace_input_name']

    @property
    def _images_wrapper(self) -> NPZ_wrapper:
        return self.image_handler.images[self._input_name]

    @property
    def locus_spots_visualisation_folder(self) -> Path:
        """Where to write the locus-specific spot images visualisation data"""
        return self.image_handler.locus_spots_visualisation_folder

    @property
    def nanometers_per_pixel_xy(self) -> NumberLike:
        return self.config["xy_nm"]

    @property
    def nanometers_per_pixel_z(self) -> NumberLike:
        return self.config["z_nm"]

    def trace_all_rois(self) -> str:
        """Fits 3D gaussian to previously detected ROIs across positions and timeframes"""
        spot_fits = find_trace_fits(
            fit_func_spec=self.fit_func_spec,
            # TODO: fix this brittle / fragile / incredibly error-prone thing; #84
            # TODO: in essence, we need a better mapping between these images and the ordering of the index of the ROIs table.
            images=self.images, 
            mask_ref_frames=self.roi_table['frame'].to_list() if self.image_handler.config.get('mask_fits', False) else None, 
            background_specification=self.background_specification, 
            cores=self.config.get("tracing_cores")
            )
        
        traces = finalise_traces(rois=self.all_rois, fits=spot_fits, z_nm=self.nanometers_per_pixel_z, xy_nm=self.nanometers_per_pixel_xy)
        
        print(f"Writing traces: {self.traces_path}")
        traces.to_csv(self.traces_path)

        return self.traces_path


# For parallelisation (multiprocessing) in the case of mask_fits being False.
def _iter_fit_args(
        fit_func_spec: FunctionalForm, 
        images: Iterable[np.ndarray], 
        bg_spec: Optional[BackgroundSpecification]
        ) -> Iterable[Tuple[FunctionalForm, np.ndarray]]:
    if bg_spec is None:
        # Iterating here over regional spots (single_roi_timecourse)
        for single_roi_timecourse in images:
            # Iterating here over individal timepoints / hybridisation rounds for each regional 
            for spot_img in single_roi_timecourse:
                yield fit_func_spec, spot_img
    else:
        # Iterating here over regional spots (single_roi_timecourse)
        for single_roi_timecourse in images:
            bg_img = single_roi_timecourse[bg_spec.frame_index].astype(np.int16)
            # Iterating here over individal timepoints / hybridisation rounds for each regiona
            for spot_img in single_roi_timecourse:
                yield fit_func_spec, spot_img.astype(np.int16) - bg_img


def finalise_traces(rois: pd.DataFrame, fits: pd.DataFrame, z_nm: NumberLike, xy_nm: NumberLike) -> pd.DataFrame:
    """
    Pair ROIs (single spots) table with row-by-row fits, apply drift correction, convert to nanometers, sort, and name columns.

    Parameters
    ----------
    rois : pd.DataFrame
        The table of data for each spot in each hybridisation frame
    fits : pd.DataFrame
        The table of functional form fits for each row in the ROIs frame
    z_nm : NumberLike
        Number of nanometers per pixel in the z-direction
    xy_nm : NumberLike
        Number of nanometers per pixel in the x- and y-directions

    Returns
    -------
    pd.DataFrame
        The result of joining (horizontally) the frames, applying drift correction, sorting, and applying units
    """
    # First, combine the original ROI data with the newly obtained Gaussian fits data.
    traces = pair_rois_with_fits(rois=rois, fits=fits)
    #Then, apply fine scale drift to fits, and map pixels to physcial units.
    traces = apply_fine_scale_drift_correction(traces)
    traces = apply_pixels_to_nanometers(traces, z_nm_per_px=z_nm, xy_nm_per_px=xy_nm)
    traces = traces.sort_values(RoiOrderingSpecification.row_order_columns())
    # Finally, rename columns and yield the result.
    traces.rename(columns={"roi_id": "trace_id"}, inplace=True)
    return traces


def pair_rois_with_fits(rois: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge (horizontally) the data from the individual spots / ROIs (1 per frame per regional spot) and the Gaussian fits.

    Parameters
    ----------
    rois : pd.DataFrame
        Individual spot data (1 per frame per regional spot)
    fits : pd.DataFrame
        Parameters for function fit to each individual spot
    
    Returns
    -------
    pd.DataFrame
        A frame combining the individual spot data with parameters of functional form fit to that data
    
    Raises
    ------
    ValueError
        If the indexes of the frames to combine don't match
    """
    if rois.shape[0] != fits.shape[0]:
        raise ValueError(f"ROIs table has {rois.shape[0]} rows, but fits table has {fits.shape[0]}; these should match.")
    if any(rois.index != fits.index):
        raise ValueError("Indexes of spots table and fits table don't match!")
    # TODO: fix this brittle / fragile / incredibly error-prone thing; #84
    traces = pd.concat([rois, fits], axis=1)
    return traces


def find_trace_fits(
    fit_func_spec: FunctionalForm, 
    images: Iterable[np.ndarray], 
    mask_ref_frames: Optional[List[int]], 
    background_specification: Optional[BackgroundSpecification], 
    cores: Optional[int] = None
    ) -> pd.DataFrame:
    """
    Fit distributions to each of the regional spots, but over all hybridisation rounds.

    Parameters
    ----------
    fit_func_spec : FunctionalForm
        Pair of function to fit to each spot, and dimensionality of the data for that fit (e.g. 3)
    images : Iterable of np.ndarray
        The collection of 4D arrays from the spot_images.npz (1 for each FOV + ROI combo)
    mask_ref_frames : list of int, optional
        Frames to use for masking when fitting, indexed by FOV
    background_specification : BackgroundSpecification, optional
        Bundle of index of hybridisation round (e.g. 0) to define as background, and associated 
        positional drifts
    cores : int, optional
        How many CPUs to use
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with one row per hybridisation round per FOV, with columns named to 
        denote the parameters of the optimised functional form for each one
    """
    # NB: For these iterations, each is expected to be a 4D array (first dimension being hybridisation round, and (z, y, x) for each).
    if mask_ref_frames:
        raise NotImplementedError(MASK_FITS_ERROR_MESSAGE)
        if background_specification is None:
            def finalise_spot_img(img, _):
                return img
        else:
            def finalise_spot_img(img, fov_imgs):
                return img.astype(np.int16) - fov_imgs[background_specification.frame_index].astype(np.int16)
        fits = []
        for p, single_roi_timecourse in tqdm(enumerate(images), total=len(images)):
            ref_img = single_roi_timecourse[mask_ref_frames[p]]
            #print(ref_img.shape)
            for t, spot_img in enumerate(single_roi_timecourse):
                #if background_specification is not None:
                    #shift = ndi.shift(single_roi_timecourse[background_specification.frame_index], shift=background_specification.drifts[t])
                    #spot_img = np.clip(spot_img.astype(np.int16) - shift, a_min = 0, a_max = None)
                spot_img = finalise_spot_img(spot_img, single_roi_timecourse)
                fits.append(fit_single_roi(fit_func_spec=fit_func_spec, roi_img=spot_img, mask=ref_img))
    else:
        fits = Parallel(n_jobs=cores or -1)(
            delayed(fit_single_roi)(fit_func_spec=ff_spec, roi_img=spot_img) 
            for ff_spec, spot_img in _iter_fit_args(fit_func_spec=fit_func_spec, images=images, bg_spec=background_specification)
            )
    
    full_cols = ROI_FIT_COLUMNS + IMG_SIDE_LEN_COLS
    bads = [(i, row) for i, row in enumerate(fits) if len(row) != len(full_cols)]
    if bads:
        raise Exception(f"{len(bads)} row(s) with field count different than column count ({len(full_cols)} == len({full_cols}))")
    return pd.DataFrame(fits, columns=full_cols)


def fit_single_roi(
        fit_func_spec: FunctionalForm, 
        roi_img: np.ndarray, 
        mask: Optional[np.ndarray] = None, 
        background: Optional[np.ndarray] = None,
        ) -> np.ndarray:
    """
    Fit a single roi with 3D gaussian (MLE or LS as defined in config).
    
    Masking by intensity or label image can be used to improve fitting correct spot (set in config).

    Parameters
    ----------
    fit_func : FunctionalForm
        The 3D functional form to fit, e.g. a 3D Gaussian -- bundle of function and dimensionality
    roi_img : np.ndarray
        The data to which the given functional form should be fit; namely, usually a 3D array of 
        signal intensity values, with each entry corresponding to a pixel
    mask : np.ndarray, optional
        Array of values which, after transformation, multiplies the ROI image, allegedly to perhaps 
        provide better tracing performance; if provided, the dimensions should match that of ROI image.
    background : np.ndarray, optional
        Data of values to subtract from main image, representing background fluorescence signal, optional; 
        if not provided, no background subtraction will be performed

    Returns
    -------
    np.ndarray
        Array-/vector-Like of values representing the optimised parameter values of the function to fit, 
        and the lengths of the box defining the ROI (z, y, x)
    """
    try:
        len_z, len_y, len_x = roi_img.shape
    except ValueError as e:
        raise Exception(f"ROI image for tracing isn't 3D: {roi_img.shape}") from e
    if len(roi_img.shape) != fit_func_spec.dimensionality:
        raise ValueError(f"ROI image to trace isn't correct dimensionality ({fit_func_spec.dimensionality}); shape: {roi_img.shape}")
    if background:
        # TODO: check that dimension of background image matches that of main ROI image.
        roi_img = roi_img - background
    if not np.any(roi_img) or any(d < 3 for d in roi_img.shape): # Check if empty or too small for fitting.
        fit = [-1] * len(ROI_FIT_COLUMNS)
    else:
        center = 'max' if mask is None \
            else list(np.unravel_index(np.argmax(roi_img * (mask/np.max(mask))**2, axis=None), roi_img.shape))
        fit = list(fit_func_spec.function(roi_img, sigma=1, center=center)[0])
    return np.array(fit + [len_z, len_y, len_x])


def apply_fine_scale_drift_correction(traces: pd.DataFrame) -> pd.DataFrame:
    """Shift pixel coordinates by the amount of fine-scale drift correction."""
    traces['z_px_dc'] = traces['z_px'] + traces['z_px_fine']
    traces['y_px_dc'] = traces['y_px'] + traces['y_px_fine']
    traces['x_px_dc'] = traces['x_px'] + traces['x_px_fine']
    return traces


def apply_pixels_to_nanometers(traces: pd.DataFrame, z_nm_per_px: float, xy_nm_per_px: float) -> pd.DataFrame:
    """Add columns for distance in nanometers, based on pixel-to-nanometer conversions."""
    simple_z = [BOX_Z_COL, "sigma_z"]
    simple_xy = [BOX_Y_COL, BOX_X_COL, "sigma_xy"]
    traces[simple_z] = traces[simple_z] * z_nm_per_px
    traces[simple_xy] = traces[simple_xy] * xy_nm_per_px
    traces["z"] = traces["z_px_dc"] * z_nm_per_px
    traces["y"] = traces["y_px_dc"] * xy_nm_per_px
    traces["x"] = traces["x_px_dc"] * xy_nm_per_px
    return traces


@doc(
    summary="Compute a list of multiarrays, grouping and stacking by 'position' (FOV).",
    extended_summary="""
        The expectation is that the data underlying the NPZ input will be a list of 4D arrays, each corresponding to 
        one of the retained (after filtering) regional barcode spots. The 4 dimensions: (time, z, y, x). 
        The time dimension represents the extraction of the bounding box corresponding to the spot ROI, extracting 
        pixel intensity data in each of the pertinent imaging timepoints.

        The expectation is that this data is flattened over the hypothetical FOV, regional barcode, and channel dimensions. 
        That is, the underlying arrays may come from any field of view imaged during the experiment, any channel, and 
        any of the regional barcode imaging timepoints. The names of the underlying arrays in the NPZ must encode 
        this information (about FOV, regional barcode timepoint, and ROI ID).
    """,
    parameters=dict(
        npz="Path to the NPZ file containing pixel volume stacks (across timepoints) for each ROI (detected regional spot)",
        locus_grouping="Mapping from regional timepoint to associated locus timepoints",
    ),
    raises=dict(ArrayDimensionalityError="If spot image volumes from the same regional barcode have different numbers of timepoints"),
    returns="""
        List of pairs, where first pair element is the name for the FOV, and the second element is the stacking of all ROI stacks for that FOV, 
        each ROI stack consisting of a pixel volume for multiple timepoints
    """,
)
def compute_spot_images_multiarray_per_fov(npz: str | Path | NPZ_wrapper, locus_grouping: Optional[dict[TimepointFrom0, set[TimepointFrom0]]]) -> List[Tuple[str, np.ndarray]]:
    full_data_file: str | Path = npz.filepath if isinstance(npz, NPZ_wrapper) else npz
    npz, keyed = _prep_npz_to_zarr(npz)
    
    num_loc_times_by_reg_time: dict[TimepointFrom0, int] = {}
    for rt, lts in (locus_grouping or {}).items():
        n_lt = len(lts)
        if n_lt == 0:
            raise ValueError(f"Empty locus times collection for regional time {rt}")
        num_loc_times_by_reg_time[rt] = n_lt

    # Facilitate assurance of same number of timepoints for each regional spot, to create non-ragged array.
    if len(num_loc_times_by_reg_time) == 0:
        max_num_times = max(arr.shape[0] for arr in npz)
    else:
        max_num_times = max(num_loc_times_by_reg_time.values())
    
    result: list[tuple[str, np.ndarray]] = []
    for pos, pos_group in itertools.groupby(keyed, lambda k_: k_[0].position):
        current_stack: list[np.ndarray] = []
        for filename_key, filename in pos_group:
            pixel_array = npz[filename]
            obs_num_times: int = pixel_array.shape[0]
            reg_time: TimepointFrom0 = TimepointFrom0(filename_key.ref_frame)
            
            if locus_grouping:
                # For nonempty locus grouping case, try to validate the time dimension.
                num_loc_times: int = num_loc_times_by_reg_time.get(reg_time, 0)
                if num_loc_times == 0:
                    raise RuntimeError(f"No expected locus time count for regional time {reg_time}, despite iterating over spot image file {filename}")
                exp_num_times: int = num_loc_times + 1 # Add 1 to account for the regional timepoint itself.
                if obs_num_times != exp_num_times:
                    raise ArrayDimensionalityError(
                        f"Expected {exp_num_times} timepoints for regional time {reg_time} but got {obs_num_times} from filename {filename} in archive {full_data_file}"
                    )
            
            if obs_num_times < max_num_times:
                num_to_fill = max_num_times - obs_num_times
                logger.debug("Backfilling array of %d timepoints with %d empty timepoints", obs_num_times, num_to_fill)
                pixel_array = backfill_array(pixel_array, num_places=num_to_fill)
            if pixel_array.shape[0] != max_num_times:
                raise ArrayDimensionalityError(
                    f"Need {max_num_times} timepoints but have {pixel_array.shape[0]} for pixel array from file {filename} in archive {full_data_file}"
                )
            current_stack.append(pixel_array)
        result.append(((pos, np.stack(current_stack))))
    return result


def backfill_array(array: np.ndarray, *, num_places: int, **kwargs) -> np.ndarray:
    if not isinstance(num_places, int):
        raise TypeError(f"Number of places to backfill must be int; got {type(num_places).__name__}")
    if num_places < 0:
        raise ValueError(f"Number of places to backfill must be nonnegative; got {num_places}")
    pad_width = [(0, num_places)] + ([(0, 0)] * max(0, len(array.shape) - 1))
    return np.pad(array, pad_width=pad_width **kwargs)


def _prep_npz_to_zarr(npz: Union[str, Path, NPZ_wrapper]) -> Tuple[NPZ_wrapper, Iterable[Tuple[RoiOrderingSpecification.FilenameKey, str]]]:
    if isinstance(npz, (str, Path)):
        npz = NPZ_wrapper(npz)
    keyed = sorted(map(lambda fn: (RoiOrderingSpecification.get_file_sort_key(fn), fn), npz.files), key=lambda k_: k_[0].to_tuple)
    return npz, keyed
