"""Filter detected spots for overlap with detected nuclei."""

import argparse
import logging
from typing import *

import numpy as np
import pandas as pd
import tqdm

from gertils import ExtantFile, ExtantFolder
from looptrace import DimensionalityError
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

logger = logging.getLogger()


NUC_LABEL_COL = "nucleusNumber"


def _filter_rois_in_nucs(
    rois: pd.DataFrame, 
    nuc_label_img: np.ndarray, 
    new_col: str, 
    *,
    nuc_drifts: pd.DataFrame, 
    spot_drifts: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Check if a spot is in inside a segmented nucleus.

    Arguments
    ---------
    rois : pd.DataFrame
        ROI table to check, usually FISH spots (from regional barcodes)
    nuc_label_img : np.ndarray
        2D/3D label images, where 0 is outside any nuclear region and >0 inside a nuclear region
    new_col : str
        The name of the new column in the ROI table
    nuc_drifts : pd.DataFrame
        Data table with information on drift correction for nuclei
    spot_drifts : pd.DataFrame
        Data table with information on drift correction for FISH spots

    Returns
    -------
    pd.DataFrame: Updated ROI table, with column indicating if ROI is inside nucleus (or other labeled region)
    """

    # We're only interested here in adding a nucelus (or other region) ID column, so keep all other data.
    new_rois = rois.copy()
    
    def spot_in_nuc(row: Union[pd.Series, dict], nuc_label_img: np.ndarray):
        base_idx = (int(row["yc"]), int(row["xc"]))
        if len(nuc_label_img.shape) == 2:
            idx = base_idx
        else:
            try:
                idx_px_z = 1 if nuc_label_img.shape[-3] == 1 else int(row["zc"]) # Flat in z dimension?
            except IndexError as e:
                print(f"IndexError ({e}) trying to get z-axis length from images with shape {nuc_label_img}")
                raise
            idx = (idx_px_z, ) + base_idx
        try:
            spot_label = nuc_label_img[idx]
        except IndexError as e: # If, due to drift spot, is outside timepoint?
            print(f"IndexError ({e}) extracting {idx} from image of shape {nuc_label_img.shape}. Spot label set to 0.")
            spot_label = 0
        return int(spot_label)

    # Remove the labels column if it already exists.
    new_rois.drop(columns=[new_col], inplace=True, errors="ignore")

    rois_shifted = new_rois.copy()
    shifts = []
    for _, row in rois_shifted.iterrows():
        curr_pos_name = row["fieldOfView"]
        raw_nuc_drift_match = nuc_drifts[nuc_drifts["fieldOfView"] == curr_pos_name]
        if not isinstance(raw_nuc_drift_match, pd.DataFrame):
            raise TypeError(f"Nuclear drift for FOV {curr_pos_name} is not a data frame, but {type(raw_nuc_drift_match).__name__}")
        if not raw_nuc_drift_match.shape[0] == 1:
            raise DimensionalityError(f"Nuclear drift for FOV {curr_pos_name} is not exactly 1 row, but {raw_nuc_drift_match.shape[0]} rows!")
        drift_target = raw_nuc_drift_match[["zDriftCoarsePixels", "yDriftCoarsePixels", "xDriftCoarsePixels"]].to_numpy()
        drift_roi = spot_drifts[(spot_drifts["fieldOfView"] == curr_pos_name) & (spot_drifts["timepoint"] == row["timepoint"])][["zDriftCoarsePixels", "yDriftCoarsePixels", "xDriftCoarsePixels"]].to_numpy()
        shift = drift_target - drift_roi
        shifts.append(shift[0])
    shifts = pd.DataFrame(shifts, columns=["z", "y", "x"])
    rois_shifted[["zc", "yc", "xc"]] = rois_shifted[["zc", "yc", "xc"]].to_numpy() - shifts[["z","y","x"]].to_numpy()

    # Store the vector of nucleus IDs in a new column on the original ROI table.
    new_rois.loc[:, new_col] = rois_shifted.apply(spot_in_nuc, nuc_label_img=nuc_label_img, axis=1)
    
    return new_rois


def workflow(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
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

    def query_table_for_fov(table: pd.DataFrame) -> Callable[[str], pd.DataFrame]:
        return (lambda fov: table.query('fieldOfView == @fov'))

    logger.info(f"Reading coarse-drift file for nuclei: {N.drift_correction_file__coarse}")
    drift_table_nuclei = pd.read_csv(N.drift_correction_file__coarse, index_col=False)
    get_nuc_drifts: Callable[[str], pd.DataFrame] = query_table_for_fov(drift_table_nuclei)
    
    logger.info(f"Reading coarse-drift file for spots: {H.drift_correction_file__coarse}")
    drift_table_spots = pd.read_csv(H.drift_correction_file__coarse, index_col=False)
    get_spot_drifts = query_table_for_fov(drift_table_spots)
    
    rois_table = pd.read_csv(H.proximity_accepted_spots_file_path, index_col=False)
    get_rois = query_table_for_fov(rois_table)

    logger.info("Assigning spots to nuclei labels...")
    all_rois = []
    for i, pos in tqdm.tqdm(enumerate(H.image_lists[H.spot_input_name])):
        rois = get_rois(pos)
        if len(rois) == 0:
            logger.warning(f"No ROIs for FOV: {pos}")
            continue

        nuc_drifts: pd.DataFrame = get_nuc_drifts(pos)
        spot_drifts = get_spot_drifts(pos)
        
        filter_kwargs = {"nuc_drifts": nuc_drifts, "spot_drifts": spot_drifts}
        # TODO: this array indexing is sensitive to whether the mask and class images have the dummy time and channel dimensions or not.
        # See: https://github.com/gerlichlab/looptrace/issues/247
        logger.info(f"Assigning nuclei labels for spots from FOV: {pos}")
        rois = _filter_rois_in_nucs(rois, nuc_label_img=N.mask_images[i].compute(), new_col=NUC_LABEL_COL, **filter_kwargs)
        if N.class_images is not None:
            raise NotImplementedError("Nuclear classification isn't supported.")
            logger.info(f"Assigning nuclei classes for spots from FOV: {pos}")
            rois = _filter_rois_in_nucs(rois, nuc_label_img=N.class_images[i].compute(), new_col="nuc_class", **filter_kwargs)
        all_rois.append(rois.copy())
    
    if len(all_rois) == 0:
        logger.warning(f"No ROIs! Cannot write nuclei-labeled spots file")
    else:
        all_rois = pd.concat(all_rois).sort_values(["fieldOfView", "timepoint"])
        logger.info(f"Writing nuclei-labeled spots file: {H.nuclei_labeled_spots_file_path}")
        all_rois.to_csv(H.nuclei_labeled_spots_file_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign detected spots to nuclei (or not).")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        image_save_path=args.image_save_path,
        )
