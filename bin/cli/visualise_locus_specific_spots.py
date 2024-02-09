"""Visualisation of retained (after QC filters) locus-specific FISH spots."""

import argparse
from typing import *

import napari

from gertils import ExtantFile, ExtantFolder
from looptrace import read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.image_io import multi_ome_zarr_to_dask
from looptrace.napari_helpers import \
    SIGNAL_TO_QUIT, add_points_to_viewer, prompt_continue_napari, shutdown_napari

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

# TODO: feature ideas:
# 2. Include annotation and/or coloring based on the skipped reason(s).

QC_PASS_COLUMN = "qcPass"

def workflow(config_file: ExtantFile, images_folder: ExtantFolder):
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    T = Tracer(H)
    print(f"Reading ROIs file: {H.drift_corrected_all_timepoints_rois_file}")
    # NB: we do NOT use the drift-corrected pixel values here, since we're interested 
    #     in placing each point within its own ROI, not relative to some other ROI.
    coordinate_columns = ["z_px", "y_px", "x_px"]
    extra_columns = ["position", QC_PASS_COLUMN]
    point_table = read_table_pandas(H.drift_corrected_all_timepoints_rois_file)[coordinate_columns + extra_columns]
    data_path = T.all_spot_images_zarr_root_path
    print(f"INFO -- Reading image data: {data_path}")
    images, positions = multi_ome_zarr_to_dask(data_path)
    for img, pos in zip(images, positions):
        cur_pts_tab = point_table[point_table.position == pos]
        viewer = napari.view_image(img)
        add_points_to_viewer(
            viewer=viewer, 
            points=cur_pts_tab[coordinate_columns].drop(extra_columns), 
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
