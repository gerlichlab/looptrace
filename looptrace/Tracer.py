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
from looptrace.ImageHandler import ImageHandler, LocusGroupingData
from looptrace.SpotPicker import SPOT_IMAGE_PIXEL_VALUE_TYPE, RoiOrderingSpecification, get_spot_images_zipfile
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
    """Fitting 3D Gaussians to pixel values in 3D subvolumes"""
    def __init__(self, image_handler: ImageHandler, trace_beads: bool = False):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.pos_list = self.image_handler.image_lists[image_handler.spot_input_name]
        self._trace_beads = trace_beads
        fit_func_specs = {
            'LS': FunctionalForm(function=fitSymmetricGaussian3D, dimensionality=3), 
            'MLE': FunctionalForm(function=fitSymmetricGaussian3DMLE, dimensionality=3)
            }
        fit_func_value = self.config['fit_func']
        try:
            self.fit_func_spec = fit_func_specs[fit_func_value]
        except KeyError as e:
            raise Exception(f"Unknown fitting function ('{fit_func_value}'); choose from: {', '.join(fit_func_specs.keys())}") from e
    
    @property
    def all_rois(self) -> pd.DataFrame:
        # will be concatenated horizontally with fits; idempotent if already effectively unindexed
        return self.image_handler.tables[self.image_handler.spot_input_name + "_dc_rois"].reset_index(drop=True)

    def finalise_suffix(self, p: Path) -> Path:
        return Path(str(p).replace(".csv", "_beads.csv")) if self._trace_beads else p

    @property
    def roi_table(self) -> pd.DataFrame:
        return self.image_handler.tables[self.image_handler.spot_input_name + ("_bead_rois" if self._trace_beads else "_rois")]

    @property
    def traces_path(self) -> Path:
        return self.finalise_suffix(self.image_handler.traces_path)
    
    @property
    def traces_path_enriched(self) -> Path:
        return self.finalise_suffix(self.image_handler.traces_path_enriched)

    @property
    def drift_table(self) -> pd.DataFrame:
        return self.image_handler.spots_fine_drift_correction_table

    @property
    def images(self) -> Iterable[np.ndarray]:
        """Iterate over the small, single spot images for tracing (1 per timepoint per ROI)."""
        for fn in self._iter_filenames():
            yield self._images_wrapper[fn]

    @property
    def _background_wrapper(self) -> Optional[NPZ_wrapper]:
        bg_time: Optional[int] = self.image_handler.background_subtraction_frame
        if bg_time is None:
            return None
        try:
            return self.image_handler.images[self._input_name]
        except KeyError as e:
            sure_message = f"Background subtraction frame ({bg_time}) is non-null, but no spot image background was found."
            zip_path = get_spot_images_zipfile(self.image_handler.image_save_path, is_background=True)
            best_guess = f"Has {zip_path} been generated?"
            raise RuntimeError(f"{sure_message} {best_guess}") from e

    @property
    def _input_name(self) -> str:
        return self.config['trace_input_name']

    @property
    def _images_wrapper(self) -> NPZ_wrapper:
        return self.image_handler.images[self._input_name]

    def _iter_filenames(self) -> Iterable[str]:
        _, keyed_filenames = _prep_npz_to_zarr(self._images_wrapper)
        for _, fn in keyed_filenames:
            yield fn

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
            filenames=self._iter_filenames(),
            image_data=self._images_wrapper, 
            background_data=self._background_wrapper, 
            mask_ref_frames=self.roi_table["frame"].to_list() if self.image_handler.config.get("mask_fits", False) else None, 
            cores=self.config.get("tracing_cores"),
        )

        if self._background_wrapper is None:
            logging.info("No background subtraction; will pair fits with full ROIs table")
            rois_table = self.all_rois
        else:
            logging.info("Subsetting ROIs table to exclude background timepoint records before pairing ROIs with fits...")
            bg_time = self.image_handler.background_subtraction_frame
            assert isinstance(bg_time, int) and bg_time >= 0, f"Background subtraction timepoint isn't nonnegative int: {bg_time}"
            rois_table = self.all_rois[self.all_rois.frame != bg_time]
        
        logging.info("Finalising traces table...")
        traces = finalise_traces(rois=rois_table, fits=spot_fits, z_nm=self.nanometers_per_pixel_z, xy_nm=self.nanometers_per_pixel_xy)
        
        logging.info("Writing traces: %s", self.traces_path)
        traces.to_csv(self.traces_path)

        return self.traces_path

    def write_all_spot_images_to_one_per_fov_zarr(self, overwrite: bool = False) -> list[Path]:
        name_data_pairs: list[tuple[str, np.ndarray]] = (
            compute_spot_images_multiarray_per_fov(self._images_wrapper, locus_grouping=self.image_handler.locus_grouping) \
                if self.image_handler.locus_grouping else \
            compute_spot_images_multiarray_per_fov(self._images_wrapper, num_timepoints=sum(1 for _ in self.image_handler.iter_imaging_rounds()))
        )
        assert (
            isinstance(name_data_pairs, list), 
            f"Result of computation of per-FOV spot images arrays isn't list, but {type(name_data_pairs).__name__}"
        )
        if len(name_data_pairs) == 0:
            return []
        _, a0 = name_data_pairs[0]
        return write_jvm_compatible_zarr_store(
            name_data_pairs, 
            root_path=self.locus_spots_visualisation_folder, 
            dtype=a0.dtype, 
            overwrite=overwrite,
        )


# For parallelisation (multiprocessing) in the case of mask_fits being False.
def _iter_fit_args(
    filenames: Iterable[str],
    image_data: NPZ_wrapper,
    background_data: Optional[NPZ_wrapper],
) -> Iterable[Tuple[FunctionalForm, np.ndarray]]:
    get_data: Callable[[str], np.ndarray] = (
        (lambda fn: image_data[fn]) 
        if background_data is None else 
        # Here we need np.int16 rather than np.uint16 to properly handle negatives.
        (lambda fn: image_data[fn].astype(np.int32) - background_data[fn].astype(np.int32))
    )
    for fn in filenames:
        time_stack_of_volumes: np.ndarray = get_data(fn)
        n_times = time_stack_of_volumes.shape[0]
        for t in range(n_times):
            yield time_stack_of_volumes[t]


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
    filenames: Iterable[str],
    image_data: NPZ_wrapper, 
    *,
    background_data: Optional[NPZ_wrapper], 
    mask_ref_frames: Optional[List[int]], 
    cores: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit distributions to each of the regional spots, but over all hybridisation rounds.

    Parameters
    ----------
    fit_func_spec : FunctionalForm
        Pair of function to fit to each spot, and dimensionality of the data for that fit (e.g. 3)
    filenames : Iterable of str
        Names of the single-spot time stacks as keys in a NPZ
    image_data : NPZ_wrapper
        Single-spot time stacks in NPZ
    background_data : NPZ_wrapper, optional
        Wrapper around NPZ stack of per-spot background data to subtract, optional
    mask_ref_frames : list of int, optional
        Frames to use for masking when fitting, indexed by FOV
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
        if background_data is None:
            def finalise_spot_img(img, _):
                return img
        else:
            def finalise_spot_img(img, fov_imgs):
                return img.astype(np.int16) - fov_imgs[background_data.frame_index].astype(np.int16)
        fits = []
        for p, single_roi_timecourse in tqdm(enumerate(images), total=len(images)):
            ref_img = single_roi_timecourse[mask_ref_frames[p]]
            #print(ref_img.shape)
            for t, spot_img in enumerate(single_roi_timecourse):
                #if background_data is not None:
                    #shift = ndi.shift(single_roi_timecourse[background_data.frame_index], shift=background_data.drifts[t])
                    #spot_img = np.clip(spot_img.astype(np.int16) - shift, a_min = 0, a_max = None)
                spot_img = finalise_spot_img(spot_img, single_roi_timecourse)
                fits.append(fit_single_roi(fit_func_spec=fit_func_spec, roi_img=spot_img, mask=ref_img))
    else:
        fits = Parallel(n_jobs=cores or -1)(
            delayed(fit_single_roi)(fit_func_spec=fit_func_spec, roi_img=spot_img) 
            for spot_img in tqdm.tqdm(_iter_fit_args(filenames=filenames, image_data=image_data, background_data=background_data))
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
        bg_npz="Optionally, NPZ-stored data with the background to subtract from each image array",
        locus_grouping="Mapping from regional timepoint to associated locus timepoints",
        num_timepoints="Number of imaging timepoints in the experiment",
    ),
    raises=dict(
        ArrayDimensionalityError="If spot image volumes from the same regional barcode have different numbers of timepoints",
        TypeError="If locus_grouping and num_timepoints are provided, or if neither is provided",
    ),
    returns="""
        List of pairs, where first pair element is the name for the FOV, and the second element is the stacking of all ROI stacks for that FOV, 
        each ROI stack consisting of a pixel volume for multiple timepoints
    """,
)
def compute_spot_images_multiarray_per_fov(
    npz: str | Path | NPZ_wrapper, 
    *, 
    bg_npz: Optional[str | Path | NPZ_wrapper] = None,
    locus_grouping: Optional[LocusGroupingData] = None, 
    num_timepoints: Optional[int] = None,
) -> list[tuple[str, np.ndarray]]:
    full_data_file: str | Path = npz.filepath if isinstance(npz, NPZ_wrapper) else npz
    npz, keyed = _prep_npz_to_zarr(npz)
    if len(npz) == 0:
        logging.warning(f"Empty spot images file! {full_data_file}")
        return []
    
    get_pixel_array: Callable[[str], np.ndarray]
    if bg_npz is None:
        logging.warning("No background to subtract for preparation of locus spots visualisation data!")
        get_pixel_array = lambda fn: npz[fn]
    else:
        logging.info("Will subtract background during preparation of locus spots visualisation data.")
        # If background is present, prepare it in the same way as the image arrays.
        # We don't care about the keyed filenames; we'll let the iteration proceed from the main images NPZ.
        bg_npz, _ = _prep_npz_to_zarr(bg_npz)
        # The background array will be a single 3D image volume (null time dimension), 
        # but numpy will properly broadcast this such that this volume is subtracted 
        # from the spot image volume for every timepoint.
        get_pixel_array = lambda fn: npz[fn].astype(np.int32) - bg_npz[fn].astype(np.int32)

    # Facilitate assurance of same number of timepoints for each regional spot, to create non-ragged array.
    # Compute the max independent of FOV so that if some regional barcode has no spots at all in a particular FOV, 
    # the renumbering of the timepoints won't be messed up.
    max_num_times: int
    if locus_grouping and num_timepoints is not None:
        raise TypeError("Provided locus_grouping and num_timepoints for spot images arrays computation; provide just one of those!")
    elif not locus_grouping and num_timepoints is None:
        raise TypeError("Provided neither nonempty locus_grouping nor num_timepoints for spot images arrays computation; provide exactly one of those!")
    elif locus_grouping:
        # +1 to account for the regional timepoint itself
        max_num_times = 1 + max(len(ts) for ts in locus_grouping.values())
    elif num_timepoints is not None:
        max_num_times = num_timepoints
    else:
        raise RuntimeError("Should never happen! Did not successfully check all cases of locus_grouping and num_timepoints for spot images arrays construction!")

    num_loc_times_by_reg_time: dict[TimepointFrom0, int] = {rt: len(lts) for rt, lts in (locus_grouping or {}).items()}

    result: list[tuple[str, np.ndarray]] = []

    for pos, pos_group in itertools.groupby(keyed, lambda k_: k_[0].position):
        logging.info("Computing spot image arrays stack for position '%s'...", pos)
        current_stack: list[np.ndarray] = []
        for filename_key, filename in sorted(pos_group, key=lambda fk_fn: (fk_fn[0].ref_frame, fk_fn[0].roi_id)):
            pixel_array = get_pixel_array(filename)
            reg_time: TimepointFrom0 = TimepointFrom0(filename_key.ref_frame)
            obs_num_times: int = pixel_array.shape[0]
            if locus_grouping:
                # For nonempty locus grouping case, try to validate the time dimension.
                num_loc_times: int = num_loc_times_by_reg_time.get(reg_time, 0)
                if num_loc_times == 0:
                    raise RuntimeError(f"No expected locus time count for regional time {reg_time}, despite iterating over spot image file {filename}")
                # Add 1 to account for the regional timepoint itself.
                exp_num_times: int = 1 + num_loc_times
            else:
                exp_num_times: int = num_timepoints
            if obs_num_times != exp_num_times:
                raise ArrayDimensionalityError(
                    f"Locus times count doesn't match expectation: {obs_num_times} != {exp_num_times}, for regional time {reg_time} from filename {filename} in archive {full_data_file}"
                )
            current_stack.append(pixel_array)
        
        # For each regional spot, 1 of 2 things will be true:
        # This is assured by the logic of the spot extraction table construction, and the fact that a dummy volume is obtained if a real one's not possible.
        # 1. An image volume will be present from every timepoint in the imaging sequence.
        # 2. An image volume will be present only from those timepoints to which the regional timepoint was mapped (and the regional timepoint itself).
        tempstack: list[np.ndarray] = []
        for img in current_stack:
            if img.shape[0] < max_num_times:
                num_to_fill = max_num_times - img.shape[0]
                logging.debug("Backfilling array of %d timepoints with %d empty timepoints", img.shape[0], num_to_fill)
                img = backfill_array(img, num_places=num_to_fill)
            if img.shape[0] != max_num_times:
                raise ArrayDimensionalityError(
                    f"Need {max_num_times} timepoints but have {img.shape[0]} for pixel array from file {filename} in archive {full_data_file}"
                )
            tempstack.append(img)
        
        result.append(((pos, np.stack(tempstack))))
        
    return result


def backfill_array(array: np.ndarray, *, num_places: int) -> np.ndarray:
    if not isinstance(num_places, int):
        raise TypeError(f"Number of places to backfill must be int; got {type(num_places).__name__}")
    if num_places < 0:
        raise ValueError(f"Number of places to backfill must be nonnegative; got {num_places}")
    pad_width = [(0, num_places)] + ([(0, 0)] * max(0, len(array.shape) - 1))
    return np.pad(array, pad_width=pad_width, mode="constant", constant_values=0)


def _prep_npz_to_zarr(npz: Union[str, Path, NPZ_wrapper]) -> Tuple[NPZ_wrapper, Iterable[Tuple[RoiOrderingSpecification.FilenameKey, str]]]:
    if isinstance(npz, (str, Path)):
        npz = NPZ_wrapper(npz)
    keyed = sorted(map(lambda fn: (RoiOrderingSpecification.get_file_sort_key(fn), fn), npz.files), key=lambda k_: k_[0].to_tuple)
    return npz, keyed
