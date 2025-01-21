# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import dataclasses
import itertools
import json
import logging
from operator import itemgetter
from pathlib import Path
from typing import *

import attrs
from expression import Option, Result, compose, fst, option, result, snd
from expression.collections import Seq, seq
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from tqdm import tqdm

from gertils import ExtantFile, ExtantFolder
from gertils.types import TimepointFrom0
from numpydoc_decorator import doc

from looptrace import *

from looptrace.ImageHandler import ImageHandler, LocusGroupingData, Times
from looptrace.SpotPicker import get_locus_spot_row_order_columns, get_spot_images_zipfile
from looptrace.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE, symmetricGaussian3D
from looptrace.image_io import NPZ_wrapper, write_jvm_compatible_zarr_store
from looptrace.numeric_types import FloatLike, NumberLike
from looptrace.trace_metadata import LocusSpotViewingKey, LocusSpotViewingReindexingDetermination, PotentialTraceMetadata, TraceGroupName, trace_group_option_to_string
from looptrace.tracing_qc_support import apply_timepoint_names_and_spatial_information
from looptrace.utilities import traverse_through_either
from looptrace.voxel_stack import VoxelStackSpecification

BOX_Z_COL = "spot_box_z"
BOX_Y_COL = "spot_box_y"
BOX_X_COL = "spot_box_x"
FIT_COLUMNS = list(symmetricGaussian3D.__annotations__.keys())
IMG_SIDE_LEN_COLS = [BOX_Z_COL, BOX_Y_COL, BOX_X_COL]
MASK_FITS_ERROR_MESSAGE = "Masking fits for tracing currently isn't supported!"

FitResult: TypeAlias = Result["FitValues", str]
FitValues: TypeAlias = list[FloatLike]
VoxelProcessingError: TypeAlias = ArrayDimensionalityError | TypeError | ValueError


@dataclasses.dataclass
class FunctionalForm:
    function: callable
    dimensionality: int

    def __post_init__(self) -> None:
        if self.dimensionality != 3:
            raise NotImplementedError("Only currently supporting dimensionality = 3 for functional form fit")


def run_timepoint_name_and_distance_application(
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
    traces = apply_timepoint_names_and_spatial_information(traces_file=T.traces_path, timepoint_names=H.timepoint_names)
    outfile = T.traces_path_enriched
    print(f"Writing enriched traces file: {outfile}")
    traces.to_csv(outfile, index=False)
    return traces, outfile


class Tracer:
    """Fitting 3D Gaussians to pixel values in 3D subvolumes"""
    def __init__(self, image_handler: ImageHandler, trace_beads: bool = False):
        self.image_handler = image_handler
        self.config = image_handler.config
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
        return pd.read_csv(self.image_handler.drift_corrected_all_timepoints_rois_file, index_col=False)
        #return self.image_handler.tables[self.image_handler.spot_input_name + "_dc_rois"].reset_index(drop=True)

    @property
    def drift_table(self) -> pd.DataFrame:
        return self.image_handler.spots_fine_drift_correction_table

    def finalise_suffix(self, p: Path) -> Path:
        return Path(str(p).replace(".csv", "_beads.csv")) if self._trace_beads else p

    @property
    def nanometers_per_pixel_xy(self) -> NumberLike:
        return self.image_handler.nanometers_per_pixel_xy

    @property
    def nanometers_per_pixel_z(self) -> NumberLike:
        return self.image_handler.nanometers_per_pixel_z

    @property
    def potential_trace_metadata(self) -> PotentialTraceMetadata:
        pass

    @property
    def roi_table(self) -> pd.DataFrame:
        return self.image_handler.tables[self.image_handler.spot_input_name + ("_bead_rois" if self._trace_beads else "_rois")]

    @property
    def spot_fits_file(self) -> Path:
        """Path to the file of the raw fits, before pairing back to ROIs with pair_rois_with_fits"""
        return self.finalise_suffix(self.image_handler.spot_fits_file)

    def trace_all_rois(self) -> Path:
        """Fits 3D gaussian to previously detected ROIs across fields of view and timepoints"""
        spot_fits = find_trace_fits(
            fit_func_spec=self.fit_func_spec,
            filenames=self._iter_filenames(),
            image_data=self._images_wrapper, 
            background_data=self._background_wrapper, 
            mask_ref_timepoints=self.roi_table["timepoint"].to_list() if self.image_handler.config.get("mask_fits", False) else None, 
            cores=self.config.get("tracing_cores"),
        )
        
        logging.info("Writing spot fits: %s", self.spot_fits_file)
        spot_fits.to_csv(self.spot_fits_file, index=False)
        return self.spot_fits_file

    @property
    def traces_path(self) -> Path:
        return self.finalise_suffix(self.image_handler.traces_path)
    
    @property
    def traces_path_enriched(self) -> Path:
        return self.finalise_suffix(self.image_handler.traces_path_enriched)

    def write_traces_file(self) -> str:
        logging.info("Reading spot fits file: %s", self.spot_fits_file)
        spot_fits = pd.read_csv(self.spot_fits_file, index_col=False)
        
        rois_table = self.all_rois
        if self._background_wrapper is None:
            logging.info("No background subtraction; will pair fits with full ROIs table")
        else:
            logging.info("Subsetting ROIs table to exclude background timepoint records before pairing ROIs with fits...")
            bg_time = self.image_handler.background_subtraction_timepoint
            assert isinstance(bg_time, int) and bg_time >= 0, f"Background subtraction timepoint isn't nonnegative int: {bg_time}"
            rois_table = rois_table[rois_table.timepoint != bg_time].reset_index(drop=True)
            spot_fits = spot_fits.reset_index(drop=True)
        
        logging.info("Finalising traces table...")
        traces = finalise_traces(
            rois=rois_table, 
            fits=spot_fits, 
            z_nm=self.nanometers_per_pixel_z, 
            xy_nm=self.nanometers_per_pixel_xy, 
        )
        
        logging.info("Writing traces: %s", self.traces_path)
        traces.to_csv(self.traces_path, index=False)

        return self.traces_path

    def write_all_spot_images_to_viewable_stacks(self, *, metadata: Optional[PotentialTraceMetadata], overwrite: bool = False) -> list[Path]:
        args = (self._images_wrapper, )
        kwargs = {
            "bg_npz": self._background_wrapper, 
            "num_timepoints": sum(1 for _ in self.image_handler.iter_imaging_rounds()), 
            "potential_trace_metadata": metadata,
        }
        if self.image_handler.locus_grouping:
            # This will be used to determine the max numbers of timepoints.
            kwargs["locus_grouping"] = self.image_handler.locus_grouping
        key_data_pairs, trace_group_metadata = compute_locus_spot_voxel_stacks_for_visualisation(*args, **kwargs)
        assert isinstance(key_data_pairs, list), f"Result of computation of per-FOV spot images arrays isn't list, but {type(key_data_pairs).__name__}"
        name_data_pairs: list[(str, np.ndarray)] = [(fst(pair).to_string, snd(pair)) for pair in key_data_pairs]
        if len(name_data_pairs) == 0:
            return []
        _, first_array = name_data_pairs[0]
        metadata_file: Path = self.image_handler.trace_group_metadata_file
        logging.info("Writing trace group metadata to file: %s", metadata_file)
        with metadata_file.open(mode="w") as fh:
            json.dump(
                obj={trace_group_option_to_string(k): v.to_json for k, v in trace_group_metadata.items()}, 
                fp=fh, 
                indent=2
            )
        return write_jvm_compatible_zarr_store(
            name_data_pairs, 
            root_path=self.image_handler.locus_spots_visualisation_folder, 
            dtype=first_array.dtype, 
            overwrite=overwrite,
        )

    @property
    def _background_wrapper(self) -> Optional[NPZ_wrapper]:
        bg_time: Optional[int] = self.image_handler.background_subtraction_timepoint
        if bg_time is None:
            return None
        try:
            return self.image_handler.images["spot_background"]
        except KeyError as e:
            sure_message = f"Background subtraction timepoint ({bg_time}) is non-null, but no spot image background was found."
            zip_path = get_spot_images_zipfile(self.image_handler.image_save_path, is_background=True)
            best_guess = f"Has {zip_path} been generated?"
            raise RuntimeError(f"{sure_message} {best_guess}") from e

    @property
    def _images_wrapper(self) -> NPZ_wrapper:
        return self.image_handler.images["spot_images"]

    def _iter_filenames(self) -> Iterable[str]:
        _, keyed_filenames = _prep_npz_to_zarr(self._images_wrapper)
        for _, fn in keyed_filenames:
            yield fn


# For parallelisation (multiprocessing) in the case of mask_fits being False.
def _iter_fit_args(
    filenames: Iterable[str],
    image_data: NPZ_wrapper,
    background_data: Optional[NPZ_wrapper],
) -> Iterable[tuple[VoxelStackSpecification, TimepointFrom0, np.ndarray]]:
    get_data: Callable[[str], np.ndarray] = (
        (lambda fn: image_data[fn]) 
        if background_data is None else 
        # Here we need signed rather than unsigned integer types to handle negatives.
        (lambda fn: image_data[fn].astype(np.int32) - background_data[fn].astype(np.int32))
    )
    for fn in filenames:
        stack_key: VoxelStackSpecification = VoxelStackSpecification.from_file_name_base(fn)
        time_stack_of_volumes: np.ndarray = get_data(fn)
        n_times = time_stack_of_volumes.shape[0]
        for t in range(n_times):
            yield stack_key, TimepointFrom0(t), time_stack_of_volumes[t]


def finalise_traces(*, rois: pd.DataFrame, fits: pd.DataFrame, z_nm: NumberLike, xy_nm: NumberLike) -> pd.DataFrame:
    """
    Pair ROIs (single spots) table with row-by-row fits, apply drift correction, convert to nanometers, sort, and name columns.

    Parameters
    ----------
    rois : pd.DataFrame
        The table of data for each spot in each hybridisation timepoint
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
    traces = traces.sort_values(get_locus_spot_row_order_columns())
    return traces


def pair_rois_with_fits(rois: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """
    Merge (horizontally) the data from the individual spots / ROIs (1 per timepoint per regional spot) and the Gaussian fits.

    Parameters
    ----------
    rois : pd.DataFrame
        Individual spot data (1 per timepoint per regional spot)
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
    num_rois: int = rois.shape[0]
    num_fits: int = fits.shape[0]
    if num_rois != num_fits:
        raise ValueError(f"{num_rois} ROIs, but {num_fits} fits; these should match.")
    traces = pd.merge(
        left=rois, 
        right=fits, 
        how="inner", 
        on=get_locus_spot_row_order_columns(),
        validate="1:1"
    )
    return traces


def find_trace_fits(
    fit_func_spec: FunctionalForm, 
    filenames: Iterable[str],
    image_data: NPZ_wrapper, 
    *,
    background_data: Optional[NPZ_wrapper], 
    mask_ref_timepoints: Optional[List[int]], 
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
    mask_ref_timepoints : list of int, optional
        Timepoints to use for masking when fitting, indexed by FOV
    cores : int, optional
        How many CPUs to use
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with one row per hybridisation round per FOV, with columns named to 
        denote the parameters of the optimised functional form for each one
    """
    # NB: For these iterations, each is expected to be a 4D array (first dimension being hybridisation round, and (z, y, x) for each).
    if mask_ref_timepoints:
        raise NotImplementedError(MASK_FITS_ERROR_MESSAGE)
        if background_data is None:
            def finalise_spot_img(img, _):
                return img
        else:
            def finalise_spot_img(img, fov_imgs):
                return img.astype(np.int16) - fov_imgs[background_data.timepoint_index].astype(np.int16)
        fits = []
        for p, single_roi_timecourse in tqdm(enumerate(images), total=len(images)):
            ref_img = single_roi_timecourse[mask_ref_timepoints[p]]
            #print(ref_img.shape)
            for t, spot_img in enumerate(single_roi_timecourse):
                #if background_data is not None:
                    #shift = ndi.shift(single_roi_timecourse[background_data.timepoint_index], shift=background_data.drifts[t])
                    #spot_img = np.clip(spot_img.astype(np.int16) - shift, a_min = 0, a_max = None)
                spot_img = finalise_spot_img(spot_img, single_roi_timecourse)
                fits.append(fit_single_roi(fit_func_spec=fit_func_spec, roi_img=spot_img, mask=ref_img))
    else:
        raw_fits: list[tuple[str, int, int, int, Result[tuple["VoxelSize", FitResult], VoxelProcessingError]]] = \
            Parallel(n_jobs=cores or -1)(
                # NB: don't include regional timepoint and trace group here, since they're not necessary 
                # to uniquely determine row match between the tables to join, and they'll already be present 
                # in the table to which this fit parameters table under construction will be joined.
                delayed(lambda key, time, fun, img: (key.field_of_view, key.traceId, key.roiId, time.get, process_voxel(fit_fun_spec=fun, roi_img=img)))(
                    key=stack_key, 
                    time=timepoint, 
                    fun=fit_func_spec, 
                    img=spot_img,
                ) 
                for stack_key, timepoint, spot_img in tqdm(_iter_fit_args(filenames=filenames, image_data=image_data, background_data=background_data))
            )
    
    full_cols = [FIELD_OF_VIEW_COLUMN, "traceId", "roiId", "timepoint"] + FIT_COLUMNS + IMG_SIDE_LEN_COLS
    rows = []
    for record in raw_fits:
        fov, trace_id, roi_id, time, outcome = record
        match outcome:
            case result.Result(tag="ok", ok=(size, maybe_fit)):
                parameter_values: FitValues = _merge_useable_voxel_cases(maybe_fit)
                rows.append([fov, trace_id, roi_id, time] + parameter_values + list(size.to_tuple))
            case result.Result(tag="error", error=Exception(e)):
                raise e
            case unknown:
                raise Exception(f"Pattern match failed on result of ROI fit: {unknown}")
    bads = sum(1 for r in rows if len(r) != len(full_cols))
    if bads:
        raise Exception(f"{len(bads)} row(s) (of {len(raw_fits)}) with field count different than column count ({len(full_cols)} == len({full_cols}))")
    return pd.DataFrame(rows, columns=full_cols)


def _merge_useable_voxel_cases(fit_result: FitResult) -> FitValues:
    """Create a vector of parameter fit values, either using the good result (identity), or dummy values."""
    match fit_result:
        case result.Result(tag="ok", ok=values):
            return values
        case result.Error(tag="error", error=msg):
            logging.debug(msg)
            return [-1.0] * len(FIT_COLUMNS)


def process_voxel(
    roi_img: np.ndarray, 
    fit_fun_spec: FunctionalForm, 
    mask: Optional[np.ndarray] = None,
) -> Result[tuple["VoxelSize", FitResult], VoxelProcessingError]:
    return _get_voxel_size(roi_img).map(
        lambda size: (
            size, 
            fit_single_roi(
                fit_func_spec=fit_fun_spec, 
                roi_img=roi_img, 
                mask=mask
            ),
        )
    )


def _is_pos_int(_, attribute: attrs.Attribute, value: Any) -> None:
    if not isinstance(value, int):
        raise TypeError(f"Value for attribute {attribute.name} isn't int, but {type(value).__name__}")
    if value < 1:
        raise ValueError(f"Value for attribute {attribute.name} isn't positive: {value}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoxelSize:
    z = attrs.field(validator=_is_pos_int) # type: int
    y = attrs.field(validator=_is_pos_int) # type: int
    x = attrs.field(validator=_is_pos_int) # type: int

    def __post_init__(self) -> None:
        bad_values = {}
        for attr, in (f.name for f in dataclasses.fields(self)):
            value = getattr(self, attr)
            if not isinstance(value, int):
                bad_values[attr] = TypeError(f"Alleged '{attr}' attribute for voxel size isn't integer, but {type(value).__name__}")
            elif value < 1:
                bad_values[attr] = ValueError(f"Value for 'z' attribute for voxek size isn't positive: {value}")
        if bad_values:
            raise Exception(f"{len(bad_values)} problem(s) building voxel size instance: {bad_values}")
        
    @property
    def to_tuple(self) -> tuple[int, int, int]:
        return attrs.astuple(self)


def _get_voxel_size(roi_img: np.ndarray) -> Result["VoxelSize", VoxelProcessingError]:
    try:
        len_z, len_y, len_x = roi_img.shape
    except ValueError as broad_error:
        narrow_error = ArrayDimensionalityError(f"Expected a 3D ROI image but got an array with shape of length {len(roi_img.shape)}")
        narrow_error.__cause__ = broad_error
        return Result.Error(narrow_error)
    return Result.Ok(VoxelSize(z=len_z, y=len_y, x=len_x))


def fit_single_roi(
    fit_func_spec: FunctionalForm, 
    roi_img: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
) -> FitResult:
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
    Either an Error-wrapped message or an Ok-wrapped list of optimized parameter values
    """
    if len(roi_img.shape) != fit_func_spec.dimensionality:
        raise ValueError(f"ROI image to trace isn't correct dimensionality ({fit_func_spec.dimensionality}); shape: {roi_img.shape}")
    if not np.any(roi_img) or any(d < 3 for d in roi_img.shape): # Check if empty or too small for fitting.
        return Result.Error("No element of the ROI image evaluates to true as Boolean (possibly all-zeros?)")
    if any(d < 3 for d in roi_img.shape):
        return Result.Error("At least one dimension of the given voxel is too short for fitting")
    center = "max" if mask is None \
        else list(np.unravel_index(np.argmax(roi_img * (mask/np.max(mask))**2, axis=None), roi_img.shape))
    # We take the first element since, regardless of the optimization method at work, the optimized parameters' result is in the first element of the return value.
    return Result.Ok(list(fit_func_spec.function(roi_img, sigma=1, center=center)[0]))


def apply_fine_scale_drift_correction(traces: pd.DataFrame) -> pd.DataFrame:
    """Shift pixel coordinates by the amount of fine-scale drift correction."""
    try:
        traces["z_px_dc"] = traces["z_px"] + traces["zDriftFinePixels"]
        traces["y_px_dc"] = traces["y_px"] + traces["yDriftFinePixels"]
        traces["x_px_dc"] = traces["x_px"] + traces["xDriftFinePixels"]
    except KeyError as e:
        logging.exception(f"Error ({e}) during application of drift correction. Available columns in traces table: {', '.join(map(str, traces.columns))}")
        raise
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


def explode_voxel_stack(
    spec: VoxelStackSpecification, 
    stack: np.ndarray, 
    locus_times_by_regional_time: Mapping[TimepointFrom0, Times],
) -> Result[Iterable[tuple[TimepointFrom0, np.ndarray]], str]:
    def get_locus_times(rt: TimepointFrom0) -> Result[Times, str]:
        return Option.of_optional(locus_times_by_regional_time.get(rt)).to_result(f"No locus times for regional time: {rt}")
    num_voxels: int = stack.shape[0]
    rt: TimepointFrom0 = TimepointFrom0(spec.ref_timepoint)
    get_locus_times(rt).bind(
        lambda lts: (
            zip(sorted(list(lts + rt)), iter(stack)) 
            if num_voxels == len(lts) + 1 
            else Result.Error(f"Unexpected voxel count ({num_voxels}) for ROI from regional timepoint with {len(lts)} locus timepoints")
        )
    )


def explode_voxels_within_trace(
    specs_and_stacks: list[tuple[VoxelStackSpecification, np.ndarray]], 
    locus_times_by_regional_time: Mapping[TimepointFrom0, Times],
) -> Result[Mapping[TimepointFrom0, np.ndarray], Seq[str]]:
    traverse_through_either(
        lambda tup: explode_voxel_stack(spec=fst(tup), stack=snd(tup), locus_times_by_regional_time=locus_times_by_regional_time)
    )(specs_and_stacks)\
        .map(lambda exploded_substacks: seq.concat(*exploded_substacks))\
        .map(dict)


@doc(
    summary="Compute a list of multiarrays, grouping and stacking by visualisation unit (drag-and-drop for Napari).",
    extended_summary="""
        The expectation is that the data underlying the NPZ input will be a list of 4D arrays, each corresponding to 
        one of the retained (after filtering) regional barcode spots. The 4 dimensions: (time, z, y, x). 
        The time dimension represents the extraction of the bounding box corresponding to the spot ROI, extracting 
        pixel intensity data in each of the pertinent imaging timepoints.

        The expectation is that these data have been flattened over the hypothetical FOV, regional barcode, and channel dimensions. 
        That is, the underlying arrays may come from any field of view imaged during the experiment, any channel, and 
        any of the regional barcode imaging timepoints. The names of the underlying arrays in the NPZ must encode 
        this information (about FOV, trace group, trace ID, ROI ID, and regional timepoint).
    """,
    parameters=dict(
        npz="Path to the NPZ file containing pixel volume stacks (across timepoints) for each ROI (detected regional spot)",
        bg_npz="Optionally, NPZ-stored data with the background to subtract from each image array",
        num_timepoints="Number of imaging timepoints in the experiment",
        potential_trace_metadata="If merging ROIs for tracing, the result of parsing relevant metadata from the rounds config",
        locus_grouping="Mapping from regional timepoint to associated locus timepoints",
    ),
    raises=dict(
        ArrayDimensionalityError="If spot image volumes from the same regional barcode have different numbers of timepoints",
    ),
    returns="""
        List of pairs, where first pair element is itself a pair of name of field of view and (option-wrapped) trace group name, 
        and the second element is the stacking of all ROI stacks for that FOV, each ROI stack consisting of a pixel volume for 
        multiple timepoints
    """,
)
def compute_locus_spot_voxel_stacks_for_visualisation(
    npz: str | Path | NPZ_wrapper, 
    *, 
    bg_npz: Optional[str | Path | NPZ_wrapper],
    num_timepoints: int,
    potential_trace_metadata: Optional[PotentialTraceMetadata],
    locus_grouping: Optional[LocusGroupingData] = None, 
) -> tuple[list[tuple[LocusSpotViewingKey, np.ndarray]], Mapping[Option[TraceGroupName], LocusSpotViewingReindexingDetermination]]:
    
    # We first get the path, in case we need to message about it, as we'll lose it overwriting the variable.
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

    num_loc_times_by_reg_time: dict[TimepointFrom0, int] = {rt: len(lts) for rt, lts in (locus_grouping or {}).items()}
    def get_locus_times(t: TimepointFrom0) -> Result[Times, TimepointFrom0]:
        return Option.of_optional(num_loc_times_by_reg_time.get(t)).to_result(t)
    
    arrays: list[tuple[LocusSpotViewingKey, np.ndarray]] = []
    lookup_maximal_regional_time_locus_times_pairs: Mapping[TraceGroupName, tuple[Mapping[TimepointFrom0, Times], list[TimepointFrom0]]] = {}
    trace_group_metadata: Mapping[Option[TraceGroupName], LocusSpotViewingReindexingDetermination] = {}
    sorted_maximal_voxel_times: Optional[list[TimepointFrom0]]

    for (fov, trace_group_key), vis_group in itertools.groupby(keyed, compose(fst, lambda voxel_spec: (voxel_spec.field_of_view, voxel_spec.traceGroup))):
        logging.info("Computing spot image arrays stack for FOV '%s', trace group %s...", fov, trace_group_key)
        vis_group = list(vis_group) # Avoid iterator exhaustion.
        current_stack: list[tuple[int, np.ndarray]] = []
        for trace_id, spec_key_pairs in itertools.groupby(vis_group, compose(fst, lambda voxel_spec: voxel_spec.traceId)): # We'll defer sorting by trace ID.
            spec_key_pairs = list(spec_key_pairs) # Avoid iterator exhaustion.
            match trace_group_key:
                case option.Option(tag="none", none=_):
                    # Here we are in the case where the current trace ID is not for a larger group/structure of ROIs.
                    sorted_maximal_voxel_times = None
                    match spec_key_pairs:
                        case []:
                            raise ValueError(f"Somehow, there are no pairs of voxel specification and key for trace ID {trace_id}!")
                        case [(filename_key, filename)]:
                            pixel_array = get_pixel_array(filename)
                            reg_time: TimepointFrom0 = TimepointFrom0(filename_key.ref_timepoint)
                            obs_num_times: int = pixel_array.shape[0]
                            exp_num_times: int
                            if locus_grouping:
                                # For nonempty locus grouping case, try to validate the time dimension.
                                try:
                                    # Add 1 to account for the regional timepoint itself.
                                    exp_num_times: int = num_loc_times_by_reg_time[reg_time] + 1
                                except KeyError as e:
                                    raise RuntimeError(f"No expected locus time count for regional time {reg_time}, despite iterating over spot image file {filename}") from e
                            else:
                                exp_num_times: int = num_timepoints
                            if obs_num_times != exp_num_times:
                                raise ArrayDimensionalityError(
                                    f"Timepoint count doesn't match expectation ({obs_num_times} != {exp_num_times}), for regional time {reg_time} from filename {filename} in archive {full_data_file}"
                                )
                            current_stack.append((trace_id, pixel_array))
                        case stacks:
                            raise ValueError(f"{len(stacks)} voxel stacks (not just 1), but the trace group/structure is null")
                case option.Option(tag="some", some=trace_group):
                    maximal_regional_locus_times_pairs: Mapping[TimepointFrom0, Times]
                    sorted_maximal_voxel_times: list[TimepointFrom0]
                    try:
                        maximal_regional_locus_times_pairs, sorted_maximal_voxel_times = lookup_maximal_regional_time_locus_times_pairs[trace_group]
                    except KeyError:
                        # This is the non-empty side of the option, so we have the implication that the trace metadata must be non-null
                        if potential_trace_metadata is None:
                            raise ValueError(f"Processing data from trace group {trace_group}, but there's no trace metadata to reference")
                        # Here we have the possibility of a multi-ROI trace, so we're combining voxel stacks which correspond to different ROIs, 
                        # and which are therefore composed of different timepoints.
                        # Here we extract the arrays for the relevant voxel stacks, meld them together in appropriate order, and 
                        # determine how to index the timepoints. We also take care of dimensionality concerns here.
                        if locus_grouping is None:
                            raise NotImplementedError("At the moment, merging ROIs for tracing is only supported locus_grouping")
                        match potential_trace_metadata.get_group_times(trace_group):
                            case option.Option(tag="none", none=_):
                                raise ValueError(f"Failed to lookup group times for trace group {trace_group}")
                            case option.Option(tag="some", some=group_regional_times):
                                match traverse_through_either(lambda rt: get_locus_times(rt).map(lambda lts: (rt, lts)))(group_regional_times):
                                    case result.Result(tag="error", error=unfound_regional_times):
                                        raise ValueError(f"Failed to find locus times for {len(unfound_regional_times)} regional timepoint(s) in trace group {trace_group}: {unfound_regional_times}")
                                    case result.Result(tag="ok", ok=pairs):
                                        maximal_regional_locus_times_pairs = dict(pairs.to_list())
                                        sorted_maximal_voxel_times = list(sorted(seq.concat(*(lts + rt for rt, lts in maximal_regional_locus_times_pairs.items()))))
                                        lookup_maximal_regional_time_locus_times_pairs[trace_group] = (maximal_regional_locus_times_pairs, sorted_maximal_voxel_times)
                    
                    # For each (this) spec...
                    # 1. Get the locus times (through the regional times)
                    # 2. Get the voxel stack
                    # 3. Check that the length of the voxel stack is 1 more than the number of locus timepoints
                    # Then, interweave the locus (and regional) timepoints for the entire ensemble of voxel stacks.
                    match explode_voxels_within_trace(
                        specs_and_stacks=[(spec, get_pixel_array(key)) for spec, key in spec_key_pairs],
                        locus_times_by_regional_time=maximal_regional_locus_times_pairs
                    ):
                        case result.Result(tag="error", error=messages):
                            msg_pfx = f"{len(messages)} error(s) processing trace {trace_id} (group {trace_group})"
                            logging.error(msg_pfx)
                            for msg in messages:
                                logging.error(msg)
                            raise RuntimeError(f"{len(messages)} error(s) processing trace {trace_id} (group: {trace_group}): {messages}")
                        case result.Result(tag="ok", ok=exploded_group):
                            try:
                                first_voxel = next(exploded_group.values())
                            except StopIteration as e:
                                msg = f"No voxels! Trace = {trace_id} (group: {trace_group})"
                                logging.error(msg)
                                raise RuntimeError(msg) from e
                            voxel_shape: tuple[int, int, int] = first_voxel.shape
                            voxel_dtype: type = first_voxel.dtype
                            if len(voxel_shape) != 3:
                                raise ArrayDimensionalityError(f"For trace {trace_id} (group: {trace_group}), first voxel has not 3 dimensions, but {len(voxel_shape)}")                            
                            # Stack up the voxels (1 per timepoint in the trace), creating a time dimension. 3D arrays --> single 4D array
                            current_stack.append((trace_id, np.stack([exploded_group.get(t, np.zeros(shape=voxel_shape, dtype=voxel_dtype)) for t in sorted_maximal_voxel_times])))
        
        # Order by trace ID.
        current_stack.sort(key=fst)

        # TODO: store the reindexed the trace IDs, so that we can map mentally back-and-forth when viewing in Napari.
        # Stack up each trace's voxel stack, creating a 5D array from a list of 4D arrays. The new dimension represents the trace ID.
        arrays.append((LocusSpotViewingKey(fov, trace_group_key), np.stack([arr for _, arr in current_stack])))
        trace_group_metadata[trace_group_key] = \
            LocusSpotViewingReindexingDetermination(
                timepoints=list(map(TimepointFrom0, range(num_timepoints))) if sorted_maximal_voxel_times is None else sorted_maximal_voxel_times,
                traces=list(map(fst, current_stack)),
            )
        
    return arrays, trace_group_metadata


def backfill_array(array: np.ndarray, *, num_places: int) -> np.ndarray:
    if not isinstance(num_places, int):
        raise TypeError(f"Number of places to backfill the array must be int; got {type(num_places).__name__}")
    if num_places < 0:
        raise ValueError(f"Number of places to backfill the array must be nonnegative; got {num_places}")
    pad_width = [(0, num_places)] + ([(0, 0)] * max(0, len(array.shape) - 1))
    return np.pad(array, pad_width=pad_width, mode="constant", constant_values=0)


def _prep_npz_to_zarr(npz: Union[str, Path, NPZ_wrapper]) -> Tuple[NPZ_wrapper, Iterable[Tuple[VoxelStackSpecification, str]]]:
    if isinstance(npz, (str, Path)):
        npz = NPZ_wrapper(npz)
    keyed = sorted(
        map(lambda fn: (VoxelStackSpecification.from_file_name_base(fn), fn), npz.files), 
        key=compose(fst, lambda spec: (spec.field_of_view, spec.traceGroup, spec.roiId)),
    )
    return npz, keyed
