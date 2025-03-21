"""Filter detected spots for overlap with detected nuclei."""

import argparse
import logging
from pathlib import Path
from typing import *

import dask.array as da
from expression import Option, identity
import numpy as np
import pandas as pd
import tqdm

from gertils import ExtantFile, ExtantFolder
from looptrace import (
    FIELD_OF_VIEW_COLUMN,
    X_CENTER_COLNAME, 
    Y_CENTER_COLNAME, 
    Z_CENTER_COLNAME, 
    DimensionalityError,
)
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


NUC_LABEL_COL = "nucleusNumber"

FieldOfViewName: TypeAlias = str


def _determine_labels(
    *, 
    field_of_view: FieldOfViewName,
    rois: pd.DataFrame, 
    nuc_label_img: np.ndarray, 
    new_col: str, 
    nuc_drift: pd.DataFrame, 
    spot_drifts: pd.DataFrame,
    timepoint: Option[int],
    remove_zarr_suffix_on_fov_name: bool, 
) -> pd.DataFrame:
    """
    Check if a spot is in inside a segmented nucleus.

    Arguments
    ---------
    field_of_fiew: FieldOfViewName
        Name of the field of view from which the passed ROIs table comes
    rois : pd.DataFrame
        ROI table to check, usually FISH spots (from regional barcodes)
    nuc_label_img : np.ndarray
        2D/3D label images, where 0 is outside any nuclear region and >0 inside a nuclear region
    new_col : str
        The name of the new column in the ROI table
    nuc_drift : pd.DataFrame
        Data table with information on drift correction for nuclei, for the given FOV; 
        since we only image nuclei at one time point, this should be a table with a 
        single row
    spot_drifts : pd.DataFrame
        Data table with information on drift correction for FISH spots
    timepoint : Option[int]
        Optionally, a single timepoint from which the given ROIs table comes; 
        this will be the case (single timepoint) when it's a bead ROIs subtable 
        which is being passed to this function
    remove_zarr_suffix_on_fov_name : bool
        Whether to tolerate mismatch on FOV name between what's given and what's observed, 
        modulo the presence or absence of a .zarr suffix

    Returns
    -------
    pd.DataFrame: Updated ROI table, with column indicating if ROI is inside nucleus (or other labeled region)
    """

    # We're only interested here in adding a nucelus (or other region) ID column, so keep all other data.
    new_rois = rois.copy()
    
    def spot_in_nuc(row: Union[pd.Series, dict], nuc_label_img: np.ndarray) -> int:
        base_idx = (int(row[Y_CENTER_COLNAME]), int(row[X_CENTER_COLNAME]))
        num_dim: int = len(nuc_label_img.shape)
        if num_dim == 2:
            idx = base_idx
        else:
            try:
                idx_px_z = 0 if nuc_label_img.shape[-3] == 1 else int(row[Z_CENTER_COLNAME]) # Flat in z dimension?
            except IndexError as e:
                logging.error(f"IndexError ({e}) trying to get z-axis length from images with shape {nuc_label_img}")
                raise
            if num_dim == 3:
                extra_index = (idx_px_z, )
            elif num_dim == 5:
                # TODO: should confirm that the first 2 dimensions (time and channel) are trivial, length 1.
                extra_index = (0, 0, idx_px_z)
            else:
                raise DimensionalityError(
                    f"Image for nucleus-based filtration of spots has {num_dim} dimensions; shape: {nuc_label_img.shape}"
                )
            idx = extra_index + base_idx
        try:
            spot_label = nuc_label_img[idx]
        except IndexError as e: # If, due to drift spot, is outside timepoint?
            print(f"IndexError ({e}) extracting {idx} from image of shape {nuc_label_img.shape}. Spot label set to 0.")
            spot_label = 0
        return int(spot_label)

    # Check the type and content of the spots table.
    if not isinstance(rois, pd.DataFrame):
        raise TypeError(f"Spots table is not a data frame, but {type(rois).__name__}")
    if FIELD_OF_VIEW_COLUMN in rois.columns:
        logging.debug("Checking field of view column (%s) against given value: %s", FIELD_OF_VIEW_COLUMN, field_of_view)
        get_obs_fov_to_match: Callable[[FieldOfViewName], FieldOfViewName] = \
            (lambda s: s.removesuffix(".zarr")) if remove_zarr_suffix_on_fov_name else identity
        match list(rois[FIELD_OF_VIEW_COLUMN].unique()):
            case [obs_spot_fov]:
                # NB: here we do NOT .removesuffix(".zarr"), beacuse this should already have been done if this field is present for this table.
                obs_spot_fov = get_obs_fov_to_match(obs_spot_fov)
                if obs_spot_fov != field_of_view:
                    raise ValueError(f"Given FOV is {field_of_view}, but FOV (from column {FIELD_OF_VIEW_COLUMN}) in ROIs table is different: {obs_spot_fov}")
            case obs_fovs:
                raise ValueError(f"Expected exactly 1 unique FOV for nucleus label assignment, but got {len(obs_fovs)} in ROIs table: {obs_fovs}")
    else:
        logging.debug("Field of view column (%s) is absent, so no FOV validation will be done for ROIs", FIELD_OF_VIEW_COLUMN)

    # Check the type and content of the spots drifts table.
    if not isinstance(spot_drifts, pd.DataFrame):
        raise TypeError(f"Nuclear drift for FOV {field_of_view} is not a data frame, but {type(spot_drifts).__name__}")
    match list(spot_drifts[FIELD_OF_VIEW_COLUMN].unique()):
        case [raw_obs_spot_drift_fov]:
            obs_spot_drift_fov: FieldOfViewName = raw_obs_spot_drift_fov.removesuffix(".zarr")
            if obs_spot_drift_fov != field_of_view:
                raise ValueError(f"Given FOV is {field_of_view}, but FOV (from column {FIELD_OF_VIEW_COLUMN}) in spot drifts table is different: {obs_spot_drift_fov}")
        case obs_fovs:
            raise ValueError(f"Expected exactly 1 unique FOV for nucleus label assignment, but got {len(obs_fovs)} in spot drifts table: {obs_fovs}")

    # Check the type, shape, and content of the nuclei drift table.
    if not isinstance(nuc_drift, pd.DataFrame):
        raise TypeError(f"Nuclear drift for FOV {field_of_view} is not a data frame, but {type(nuc_drift).__name__}")
    if nuc_drift.shape[0] != 1:
        raise DimensionalityError(f"Nuclear drift for FOV {field_of_view} is not exactly 1 row, but {nuc_drift.shape[0]} rows!")
    else:
        obs_nuc_fov: FieldOfViewName = list(nuc_drift[FIELD_OF_VIEW_COLUMN])[0].removesuffix(".zarr")
        if obs_nuc_fov != field_of_view:
            raise ValueError(f"Given FOV is {field_of_view}, but FOV (from column {FIELD_OF_VIEW_COLUMN}) in nuclear drift table is different: {obs_nuc_fov}")
    
    # Handle either a single, fixed timepoint passed as an argument, or to extract this value from each row (ROI).
    get_roi_time: Callable[[pd.Series], int] = timepoint.map(lambda t: (lambda _: t)).default_value(lambda r: r["timepoint"])

    # Remove the labels column if it already exists.
    new_rois.drop(columns=[new_col], inplace=True, errors="ignore")
    rois_shifted = new_rois.copy()
    shifts = []
    shift_column_names = ["z", "y", "x"]
    center_column_names = [Z_CENTER_COLNAME, Y_CENTER_COLNAME, X_CENTER_COLNAME]
    drift_column_names = ["zDriftCoarsePixels", "yDriftCoarsePixels", "xDriftCoarsePixels"]

    for _, row in tqdm.tqdm(rois_shifted.iterrows()):
        drift_target = nuc_drift[drift_column_names].to_numpy()
        drift_roi = spot_drifts[spot_drifts["timepoint"] == get_roi_time(row)][drift_column_names].to_numpy()
        shift = drift_target - drift_roi
        shifts.append(shift[0])
    shifts = pd.DataFrame(shifts, columns=shift_column_names)
    rois_shifted[center_column_names] = rois_shifted[center_column_names].to_numpy() - shifts[shift_column_names].to_numpy()

    # Store the vector of nucleus IDs in a new column on the original ROI table.
    new_rois.loc[:, new_col] = rois_shifted.apply(spot_in_nuc, nuc_label_img=nuc_label_img, axis=1)
    
    return new_rois


def add_nucleus_labels(
    *, 
    rois_table: pd.DataFrame, 
    mask_images: list[tuple[FieldOfViewName, da.Array]], 
    nuclei_drift_file: Path, 
    spots_drift_file: Path, 
    timepoint: Option[int], 
    remove_zarr_suffix: bool,
) -> pd.DataFrame:
    
    def query_table_for_fov(table: pd.DataFrame) -> Callable[[FieldOfViewName], pd.DataFrame]:
        return (lambda fov: table.query('fieldOfView == @fov'))

    get_rois: Callable[[FieldOfViewName], pd.DataFrame]
    combine_subtables: Callable[[list[pd.DataFrame]], pd.DataFrame]
    if timepoint.is_none():
        get_rois = query_table_for_fov(rois_table)
        combine_subtables = lambda ts: pd.concat(ts).sort_values([FIELD_OF_VIEW_COLUMN, "timepoint"])
    else:
        get_rois = lambda _: rois_table
        def combine_subtables(ts: list[pd.DataFrame]) -> pd.DataFrame:
            match ts:
                case [t]:
                    return t
                case list():
                    raise ValueError(f"Expected exactly one subtable but got {len(ts)}")
                case _:
                    raise TypeError(f"Expected to be combining a list of subtables, but got {type(ts).__name__}")

    logging.info("Reading drift file for nuclei: %s", nuclei_drift_file)
    drift_table_nuclei = pd.read_csv(nuclei_drift_file, index_col=False)
    get_nuc_drift: Callable[[FieldOfViewName], pd.DataFrame] = query_table_for_fov(drift_table_nuclei)
    
    logging.info("Reading coarse-drift file for spots: %s", spots_drift_file)
    drift_table_spots = pd.read_csv(spots_drift_file, index_col=False)
    get_spot_drifts: Callable[[FieldOfViewName], pd.DataFrame] = query_table_for_fov(drift_table_spots)    
    
    subtables: list[pd.DataFrame] = []

    for pos, nuc_mask_image in tqdm.tqdm(mask_images):
        fov_name: FieldOfViewName = pos.removesuffix(".zarr")
        rois = get_rois(fov_name) # Here we use the refined FOV name.
        if len(rois) == 0:
            logging.warning("No ROIs for FOV: %s", fov_name)
            continue

        logging.info("Assigning nuclei labels for spots from FOV: %s", fov_name)
        rois = _determine_labels(
            field_of_view=fov_name, 
            rois=rois, 
            nuc_label_img=nuc_mask_image, 
            new_col=NUC_LABEL_COL, 
            nuc_drift=get_nuc_drift(pos), # Here we use the raw, unrefined FOV name.
            spot_drifts=get_spot_drifts(pos), # Here we use the raw, unrefined FOV name.
            timepoint=timepoint,
            remove_zarr_suffix_on_fov_name=remove_zarr_suffix,
        )
        subtables.append(rois.copy())
    
    return combine_subtables(subtables)


def run_labeling(
    *, 
    rois: pd.DataFrame, 
    image_handler: ImageHandler, 
    timepoint: Option[int],
    nuc_detector: Optional[NucDetector] = None,
    remove_zarr_suffix: bool,
) -> pd.DataFrame:
    if nuc_detector is None:
        nuc_detector = NucDetector(image_handler)
    fov_names: Iterable[str] = image_handler.image_lists[image_handler.spot_input_name]
    return add_nucleus_labels(
        rois_table=rois, 
        mask_images=[(pos, nuc_detector.mask_images[i]) for i, pos in enumerate(fov_names)], 
        nuclei_drift_file=nuc_detector.drift_correction_file__coarse, 
        spots_drift_file=image_handler.drift_correction_file__coarse,
        timepoint=timepoint,
        remove_zarr_suffix=remove_zarr_suffix,
    )


def workflow(
    *,
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    remove_zarr_suffix: bool,
    image_save_path: Optional[ExtantFolder] = None,
    ) -> None:
    
    # Set up the spot picker and the nuclei detector instances, to manage paths and settings.
    H = ImageHandler(
        rounds_config=rounds_config, 
        params_config=params_config, 
        images_folder=images_folder, 
        image_save_path=image_save_path,
        )
    N = NucDetector(H)
    if N.class_images is not None:
            raise NotImplementedError("Nuclear classification isn't supported.")
    
    input_file: Path = H.proximity_accepted_spots_file_path
    output_file: Path = H.nuclei_labeled_spots_file_path

    logging.info("Assigning spots to nuclei labels...")
    all_rois: pd.DataFrame = run_labeling(
        rois=pd.read_csv(input_file, index_col=False), 
        image_handler=H, 
        nuc_detector=N,
        timepoint=Option.Nothing(),
        remove_zarr_suffix=remove_zarr_suffix,
    )

    if all_rois.shape[0] == 0:
        logging.warning("No ROIs! Cannot write output file: %s", output_file)
    else:
        logging.info("Writing output file file: %s", output_file)
        all_rois.to_csv(output_file, index=False)

    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign detected spots to nuclei (or not).")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--remove-zarr-suffix", action="store_true", help="Remove the .zarr suffix on ROI FOV names for matching")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        remove_zarr_suffix=args.remove_zarr_suffix,
        image_save_path=args.image_save_path,
        )
