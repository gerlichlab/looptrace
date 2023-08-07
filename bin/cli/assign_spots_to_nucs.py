"""Filter detected spots for overlap with detected nuclei"""

import argparse
import logging
import os
from typing import *

from gertils.pathtools import ExtantFile, ExtantFolder
import pandas as pd
import tqdm

from looptrace.ImageHandler import handler_from_cli
from looptrace.SpotPicker import NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION
import looptrace.image_processing_functions as ip

logger = logging.getLogger()


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        image_save_path: Optional[ExtantFolder] = None,
        ) -> pd.DataFrame:
    
    # Set up the spot picker, to manage paths and settings.
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is not None:
        raise NotImplementedError(f"Array ID effect not yet ready for spot-in-nuclei determination!")
    H = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=image_save_path)
    pos_list = H.image_lists[H.spot_input_name]
    
    def query_table_for_pos(input_name_key: str, table_suffix: str):
        table_name = H.config[input_name_key] + table_suffix
        try:
            table = H.tables[table_name]
        except KeyError:
            logger.warning(f"Drift-corrected table unavailable for filtration: {table_name}")
            fun = lambda _: None
        else:
            logger.info(f"Using drift-corrected table for filtration: {table_name}")
            fun = lambda pos: table.query('position == @pos')
        return fun

    logger.info('Assigning spots to nuclei labels.')
    all_rois = []
    get_nuc_drifts = query_table_for_pos('nuc_input_name', '_drift_correction')
    get_spot_drifts = query_table_for_pos('spot_input_name', '_drift_correction')
    get_rois = query_table_for_pos('spot_input_name', '_rois')
    for i, pos in tqdm.tqdm(enumerate(H.image_lists[H.spot_input_name])):
        rois = get_rois(pos)
        if len(rois) == 0:
            continue

        nuc_drifts = get_nuc_drifts(pos)
        spot_drifts = get_spot_drifts(pos)
        
        filter_kwargs = {"nuc_drifts": nuc_drifts, "nuc_target_frame": H.config['nuc_ref_frame'], "spot_drifts": spot_drifts}
        if 'nuc_masks' in H.images:
            logger.info(f"Assigning nuclei labels for sports from position: {pos}")
            rois = ip.filter_rois_in_nucs(rois, nuc_label_img=H.images['nuc_masks'][i][0,0], new_col='nuc_label', **filter_kwargs)
            #rois = ip.filter_rois_in_nucs(rois, H.images['nuc_masks'][i][0,0], pos_list, new_col='nuc_label', **filter_kwargs)
        if 'nuc_classes' in H.images:
            logger.info(f"Assigning nuclei classes for spots from position: {pos}")
            rois = ip.filter_rois_in_nucs(rois, nuc_label_img=H.images['nuc_classes'][i][0,0], new_col='nuc_class', **filter_kwargs)
            #rois = ip.filter_rois_in_nucs(rois, H.images['nuc_classes'][i][0,0], pos_list, new_col='nuc_class', **filter_kwargs)
        all_rois.append(rois.copy())

    all_rois = pd.concat(all_rois).sort_values(['position', 'frame'])
    out_file_ext = f"{NUCLEI_LABELED_SPOTS_FILE_SUBEXTENSION}.csv"
    if array_id is not None:
        all_rois.to_csv(H.out_path(H.spot_input_name + '_rois_' + str(array_id).zfill(4) + out_file_ext))
    else:
        all_rois.to_csv(H.out_path(H.spot_input_name + '_rois' + out_file_ext))
        if H.spot_input_name + '_traces' in H.tables:
            logger.info('Assigning ids to traces.')
            traces = H.tables[H.spot_input_name + '_traces'].copy()
            if 'nuc_masks' in H.images:
                traces.loc[:, 'nuc_label'] = traces['trace_id'].map(all_rois['nuc_label'].to_dict())
            if 'nuc_classes' in H.images:
                traces.loc[:, 'nuc_class'] = traces['trace_id'].map(all_rois['nuc_class'].to_dict())
            traces.sort_values(['trace_id', 'frame']).to_csv(H.out_path(H.spot_input_name + '_traces.csv'))

    logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign detected spots to nuclei (or not).')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
