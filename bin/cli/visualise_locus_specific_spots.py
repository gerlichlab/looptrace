"""Visualisation of retained (after QC filters) locus-specific FISH spots."""

import argparse
from typing import *

import napari
import pandas as pd

from gertils import ExtantFile, ExtantFolder
from looptrace import read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.image_io import multi_ome_zarr_to_dask
from looptrace.napari_helpers import \
    SIGNAL_TO_QUIT, add_points_to_viewer, prompt_continue_napari, shutdown_napari

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

# TODO: feature ideas -- see https://github.com/gerlichlab/looptrace/issues/259
# Include annotation and/or coloring based on the skipped reason(s).

POSITION_COLUMN = "position"
TRACE_ID_COLUMN = "trace_id"
FRAME_COLUMN = "frame"
QC_PASS_COLUMN = "qcPass"


def workflow(config_file: ExtantFile, images_folder: ExtantFolder):
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    T = Tracer(H)
    coordinate_columns = ["z_px", "y_px", "x_px"]
    extra_columns = [POSITION_COLUMN, TRACE_ID_COLUMN, FRAME_COLUMN, QC_PASS_COLUMN]
    print(f"Reading ROIs file: {H.traces_file_qc_unfiltered}")
    # NB: we do NOT use the drift-corrected pixel values here, since we're interested 
    #     in placing each point within its own ROI, not relative to some other ROI.
    # TODO: we need to add the columns for frame and ROI ID / ROI number to the 
    #       list of what we pull, because we need these to add to the points coordinates.
    point_table = read_table_pandas(H.traces_file_qc_unfiltered)
    if POSITION_COLUMN not in point_table:
        # TODO -- See: https://github.com/gerlichlab/looptrace/issues/261
        print(f"DEBUG -- Column '{POSITION_COLUMN}' is not in the spots table parsed in package-standard pandas fashion")
        print(f"DEBUG -- Retrying the spots table parse while assuming no index: {H.traces_file_qc_unfiltered}")
        point_table = pd.read_csv(H.traces_file_qc_unfiltered, index_col=None)[coordinate_columns + extra_columns]
    data_path = T.all_spot_images_zarr_root_path
    print(f"INFO -- Reading image data: {data_path}")
    images, positions = multi_ome_zarr_to_dask(data_path)
    for img, pos in zip(images, positions):
        cur_pts_tab = point_table[point_table.position == pos]
        unique_traces = cur_pts_tab[TRACE_ID_COLUMN].nunique()
        unique_frames = cur_pts_tab[FRAME_COLUMN].nunique()
        points_array_long = cur_pts_tab[coordinate_columns].to_numpy()
        print(f"DEBUG -- points array long shape: {points_array_long.shape}")
        print(f"DEBUG -- nesting {points_array_long.shape[0]} into {(unique_traces, unique_frames)}")
        points_array_nested = points_array_long.reshape(unique_traces, unique_frames)
        viewer = napari.view_image(img)
        add_points_to_viewer(
            viewer=viewer, 
            # TODO: use info about frame and ROI ID / ROI number to put each point in the proper 3D array (z, y, x).
            points=points_array_nested, 
            properties={QC_PASS_COLUMN: cur_pts_tab[QC_PASS_COLUMN].values.astype(bool)},
            size=1,
            symbol=cur_pts_tab.apply(lambda row: "star" if bool(row[QC_PASS_COLUMN]) else "hbar", axis=1).values,
            edge_color="transparent",
            face_color=QC_PASS_COLUMN,
            face_color_cycle=["blue", "red"], 
            )
        napari.run()
        if prompt_continue_napari() == SIGNAL_TO_QUIT:
            break
        # Here we don't call viewer.close() programmatically since it's expected that the user closes the window.
        print("DEBUG: Continuing...")
    shutdown_napari()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation of retained (after QC filters) locus-specific FISH spots", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path)
