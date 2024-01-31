"""Filter detected spots for overlap with detected nuclei"""

import argparse
import logging
from typing import *

from gertils import ExtantFile, ExtantFolder
import pandas as pd
import tqdm

from looptrace.ImageHandler import handler_from_cli
from looptrace.NucDetector import NucDetector
import looptrace.image_processing_functions as ip

logger = logging.getLogger()


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        image_save_path: Optional[ExtantFolder] = None,
        ) -> None:
    
    # Set up the spot picker and the nuclei detector instances, to manage paths and settings.
    H = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    N = NucDetector(H)

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
        if NucDetector.MASKS_KEY in H.images:
            logger.info(f"Assigning nuclei labels for sports from position: {pos}")
            rois = ip.filter_rois_in_nucs(rois, nuc_label_img=H.images[NucDetector.MASKS_KEY][i][0, 0], new_col='nuc_label', **filter_kwargs)
        if NucDetector.CLASSES_KEY in H.images:
            logger.info(f"Assigning nuclei classes for spots from position: {pos}")
            rois = ip.filter_rois_in_nucs(rois, nuc_label_img=H.images[NucDetector.CLASSES_KEY][i][0, 0], new_col='nuc_class', **filter_kwargs)
        all_rois.append(rois.copy())
    
    outfile = H.nuclei_labeled_spots_file_path
    if len(all_rois) == 0:
        logger.warn(f"No ROIs! Cannot write output file: {outfile}")
    else:
        all_rois = pd.concat(all_rois).sort_values(['position', 'frame'])
        logger.info(f"Writing output file: {outfile}")
        all_rois.to_csv(outfile)
        # if H.spot_input_name + '_traces' in H.tables:
        #     logger.info('Assigning ids to traces.')
        #     traces = H.tables[H.spot_input_name + '_traces'].copy()
        #     if 'nuc_masks' in H.images:
        #         traces.loc[:, 'nuc_label'] = traces['trace_id'].map(all_rois['nuc_label'].to_dict())
        #     if 'nuc_classes' in H.images:
        #         traces.loc[:, 'nuc_class'] = traces['trace_id'].map(all_rois['nuc_class'].to_dict())
        #     traces.sort_values(['trace_id', 'frame']).to_csv(H.out_path(H.spot_input_name + '_traces.csv'))

    logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign detected spots to nuclei (or not).')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
