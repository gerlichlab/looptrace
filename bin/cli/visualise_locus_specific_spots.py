"""Visualisation of retained (after QC filters) locus-specific FISH spots."""

import argparse
from typing import *

import napari
import numpy as np
import pandas as pd

from gertils import ExtantFile, ExtantFolder
from looptrace import RoiImageSize, read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.image_io import multi_ome_zarr_to_dask
from looptrace.napari_helpers import SIGNAL_TO_QUIT, prompt_continue_napari, shutdown_napari

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

# TODO: feature ideas -- see https://github.com/gerlichlab/looptrace/issues/259
# Include annotation and/or coloring based on the skipped reason(s).

POSITION_COLUMN = "position"
ROI_NUMBER_COLUMN = "roi_number"
FRAME_COLUMN = "frame"
QC_PASS_COLUMN = "qcPass"
COORDINATE_COLUMNS = ["z_px", "y_px", "x_px"]


def workflow(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder):
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    extra_columns = [POSITION_COLUMN, ROI_NUMBER_COLUMN, FRAME_COLUMN, QC_PASS_COLUMN]
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
        point_table = pd.read_csv(H.traces_file_qc_unfiltered, index_col=None)[COORDINATE_COLUMNS + extra_columns]
    data_path = H.locus_spot_images_root_path
    print(f"INFO -- Reading image data: {data_path}")
    images, positions = multi_ome_zarr_to_dask(data_path)
    step_through_positions(zip(positions, images), point_table=point_table, roi_size=H.roi_image_size)
    shutdown_napari()


def step_through_positions(pos_img_pairs: Iterable[Tuple[str, np.ndarray]], point_table: pd.DataFrame, roi_size: RoiImageSize):
    for pos, img in pos_img_pairs:
        _, num_times, _, _, _ = img.shape
        cur_pts_tab = point_table[point_table.position == pos]
        points_data = compute_points(cur_pts_tab, num_times=num_times, roi_size=roi_size)
        visibilities, point_symbols, qc_passes, points = zip(*points_data)
        viewer = napari.view_image(img)
        viewer.add_points(
            points, 
            properties={QC_PASS_COLUMN: qc_passes},
            size=0.5,
            shown=visibilities,
            symbol=point_symbols,
            edge_width=0.1,
            edge_width_is_relative=True,
            edge_color=QC_PASS_COLUMN,
            edge_color_cycle=["blue", "red"],
            face_color=QC_PASS_COLUMN,
            face_color_cycle=["blue", "red"], 
            n_dimensional=False,
            )
        napari.run()
        if prompt_continue_napari() == SIGNAL_TO_QUIT:
            break
        # Here we don't call viewer.close() programmatically since it's expected that the user closes the window.
        print("DEBUG: Continuing...")


def compute_points(cur_pts_tab, *, num_times: int, roi_size: RoiImageSize):
    bad_shape = "o"
    points_data = []
    for roi_idx, roi_group in cur_pts_tab.groupby(ROI_NUMBER_COLUMN):
        lookup = {row[FRAME_COLUMN]: (row[QC_PASS_COLUMN], row[COORDINATE_COLUMNS].to_numpy()) for _, row in roi_group.iterrows()}
        for t in range(num_times):
            try:
                qc_pass, coords = lookup[t]
            except KeyError:
                # TODO: this is when the unfiltered traces file misses the exclusions, which should no longer continue to be the case.
                coords = np.zeros(3)
                visible = False
                point_shape = bad_shape
                qc_pass = False
            else:
                visible = True
                qc_pass = bool(qc_pass)
                point_shape = "*" if qc_pass else "o"
            if coords[0] < 0 or coords[0] > roi_size.z or coords[1] < 0 or coords[1] > roi_size.y or coords[2] < 0 or coords[2] > roi_size.x:
                if qc_pass:
                    print(f"WARN -- spot point passed QC! {coords}")
                coords = np.array([0, 0, 0])
                visible = False
                point_shape = bad_shape
            point = np.concatenate(([roi_idx, t], coords)).astype(np.float32)
            points_data.append((visible, point_shape, qc_pass, point))
    return points_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation of retained (after QC filters) locus-specific FISH spots", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--interactive", action="store_true", help="Run the program interactively")
    parser.add_argument("--save-points", action="store_true", help="Save points")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.image_path,
        )
