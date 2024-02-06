"""Visualisation of detected FISH spots"""

import argparse
from typing import *

from gertils import ExtantFile, ExtantFolder
import napari

from looptrace import read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker, compute_downsampled_image
from looptrace.napari_helpers import SIGNAL_TO_QUIT, add_points_to_viewer, extract_roi_centers, shutdown_napari

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

# TODO: ideas...
# See: https://github.com/gerlichlab/looptrace/issues/255

def workflow(
    config_file: ExtantFile, 
    images_folder: ExtantFolder, 
    image_save_path: Optional[ExtantFolder] = None,
    ):
    H = ImageHandler(config_path=config_file, image_path=images_folder, image_save_path=image_save_path)
    S = SpotPicker(H)
    print(f"Reading ROIs file: {H.nuclei_filtered_spots_file_path}")
    rois = read_table_pandas(H.nuclei_filtered_spots_file_path)
    get_points = lambda p, t, c: extract_roi_centers(rois[(rois.position == p) & (rois.frame == t) & (rois.ch == c)])
    _, roi_size, _ = S.roi_image_size

    print("Iterating over images...")
    for pos_name, full_img in S.iter_pos_img_pairs():
        for (i, frame), ch in S.iter_frames_and_channels():
            print(f"Visualising spot detection in position {pos_name}, frame {frame}, channel {ch}, method '{S.detection_method_name}', threshold {S.spot_threshold[i]}...")
            img = compute_downsampled_image(full_image=full_img, frame=frame, channel=ch, downsampling=S.downsampling)
            points = get_points(p=pos_name, t=frame, c=ch)
            viewer = napari.view_image(img)
            add_points_to_viewer(viewer=viewer, points=points/S.downsampling, size=roi_size/S.downsampling)
            napari.run()
            user_input = input(f"Press enter to continue to next image, or {SIGNAL_TO_QUIT} to quit.")
            if user_input == SIGNAL_TO_QUIT:
                break
            print("Continuing...")
    shutdown_napari()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the detected spot ROIs.")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, image_save_path=args.image_save_path)
