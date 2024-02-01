"""Filter detected spots for overlap with detected nuclei"""

import argparse
import logging
from typing import *

from gertils import ExtantFile, ExtantFolder
import pandas as pd
import tqdm

from looptrace import IllegalSequenceOfOperationsError
from looptrace.ImageHandler import handler_from_cli
from looptrace.NucDetector import NucDetector
import looptrace.image_processing_functions as ip

logger = logging.getLogger()


NUC_LABEL_COL = "nuc_label"


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        image_save_path: Optional[ExtantFolder] = None,
        ) -> None:
    
    # Set up the spot picker and the nuclei detector instances, to manage paths and settings.
    H = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    N = NucDetector(H)
    if N.mask_images is None:
        raise IllegalSequenceOfOperationsError("Nuclei need to be detected/segmented before assigning spots to nuclei.")

    read_table = lambda f: pd.read_csv(f, index_col=0)
    
    def query_table_for_pos(table: pd.DataFrame) -> Callable[[str], pd.DataFrame]:
        return (lambda pos: table.query('position == @pos'))

    logger.info(f"Reading coarse-drift file for nuclei: {N.drift_correction_file__coarse}")
    drift_table_nuclei = read_table(N.drift_correction_file__coarse)
    get_nuc_drifts = query_table_for_pos(drift_table_nuclei)
    
    logger.info(f"Reading coarse-drift file for spots: {H.drift_correction_file__coarse}")
    drift_table_spots = read_table(H.drift_correction_file__coarse)
    get_spot_drifts = query_table_for_pos(drift_table_spots)
    
    rois_table = read_table(H.proximity_filtered_spots_file_path)
    get_rois = query_table_for_pos(rois_table)

    logger.info("Assigning spots to nuclei labels...")
    all_rois = []
    for i, pos in tqdm.tqdm(enumerate(H.image_lists[H.spot_input_name])):
        rois = get_rois(pos)
        if len(rois) == 0:
            logger.warn(f"No ROIs for position: {pos}")
            continue

        nuc_drifts = get_nuc_drifts(pos)
        spot_drifts = get_spot_drifts(pos)
        
        filter_kwargs = {"nuc_drifts": nuc_drifts, "nuc_target_frame": H.config['nuc_ref_frame'], "spot_drifts": spot_drifts}
        # TODO: this array indexing is sensitive to whether the mask and class images have the dummy time and channel dimensions or not.
        # See: https://github.com/gerlichlab/looptrace/issues/247
        logger.info(f"Assigning nuclei labels for sports from position: {pos}")
        rois = ip.filter_rois_in_nucs(rois, nuc_label_img=N.mask_images[i].compute(), new_col=NUC_LABEL_COL, **filter_kwargs)
        if N.class_images is not None:
            logger.info(f"Assigning nuclei classes for spots from position: {pos}")
            rois = ip.filter_rois_in_nucs(rois, nuc_label_img=N.class_images[i].compute(), new_col="nuc_class", **filter_kwargs)
        all_rois.append(rois.copy())
    
    outfile = H.nuclei_labeled_spots_file_path
    if len(all_rois) == 0:
        logger.warn(f"No ROIs! Cannot write nuclei-labeled spots file: {outfile}")
    else:
        all_rois = pd.concat(all_rois).sort_values(["position", "frame"])
        logger.info(f"Writing nuclei-labeled spots file: {outfile}")
        all_rois.to_csv(outfile)
        logger.info(f"Writing nuclei-filtered spots file: {H.nuclei_labeled_spots_file_path}")
        all_rois.to_csv(all_rois[all_rois[NUC_LABEL_COL] != 0].drop(NUC_LABEL_COL, axis=1))

    logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign detected spots to nuclei (or not).')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
