"""Visualisation of detected FISH spots"""

import argparse
from typing import *

import napari
import numpy as np

from gertils import ExtantFile, ExtantFolder
from looptrace import read_table_pandas
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker, compute_downsampled_image as compute_subimage
from looptrace.napari_helpers import SIGNAL_TO_QUIT, add_points_to_viewer, prompt_continue_napari, save_screenshot, shutdown_napari

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

# TODO: ideas...
# See: https://github.com/gerlichlab/looptrace/issues/255
# A natural implementation of the saving approach would be to have a max-projection in z, 
# and then to reduce the points/spots/ROIs to 2D rather than 3D.

def iterate_over_select_pos_subset(S: SpotPicker, positions: Optional[Set[int]] = None) -> Iterable[Tuple[int, int, int, np.ndarray]]:
    """
    Iterate over a subset of spot detection images.

    The subsetting is done by field of view (FOV), or "position", optionally. 
    If no position names are specified, all are used; otherwise, only those specified are used.

    Parameters
    ----------
    S : looptrace.SpotPicker
        The spot picker instance with paths to images and settings used for spot detection
    positions : set of int, optional
        The of the fields of view (FOVs), or "positions" to use; if unspecified, all are used

    Returns
    -------
    Iterable of (int, int, int, np.ndarray)
        An iterable in which each element is a bundle of position name, timepoint, channel, and subimage; 
        "subimage" in the sense that the image is from one particular (FOV, timepoint, channel) combination.
    """
    for pos, full_img in enumerate(S.images):
        if positions and pos not in positions:
            print(f"DEBUG: Skipping position {pos}")
            continue
        for (_, frame), ch in S.iter_frames_and_channels():
            subimg = compute_subimage(full_image=full_img, frame=frame, channel=ch, downsampling=1)
            yield pos, frame, ch, subimg


def workflow(
    config_file: ExtantFile, 
    images_folder: ExtantFolder, 
    image_save_path: Optional[ExtantFolder] = None,
    *,
    interactive : bool = False,
    save_projections : bool = False,
    positions_to_use: Optional[Set[int]] = None,
    ):
    if not interactive and not save_projections:
        raise ValueError("Detected spot visualisation must be for interactivity or for image saving; set to True at least interactive or save_projections.")

    H = ImageHandler(config_path=config_file, image_path=images_folder, image_save_path=image_save_path)
    S = SpotPicker(H)
    print(f"Reading ROIs file: {H.nuclei_filtered_spots_file_path}")
    rois = read_table_pandas(H.nuclei_filtered_spots_file_path)
    get_sub_rois = lambda p, t, c: rois[(rois.position == S.pos_list[p]) & (rois.frame == t) & (rois.ch == c)]
    if H.roi_image_size.y != H.roi_image_size.x:
        roi_size = (H.roi_image_size.y + H.roi_image_size.x) / 2
        print(f"WARN -- ROI size differs in y ({H.roi_image_size.y}) and x ({H.roi_image_size.x}). Will use average: {roi_size}")
    else:
        roi_size = H.roi_image_size.y

    print("INFO: Iterating over images...")
    for pos, frame, ch, img in iterate_over_select_pos_subset(S, positions=positions_to_use):
        print(f"INFO: Visualising spot detection in position {pos}, frame {frame}, channel {ch}...")
        sub_rois = get_sub_rois(p=pos, t=frame, c=ch)
        if save_projections:
            viewer = napari.view_image(np.amax(img, axis=0))
            add_points_to_viewer(
                viewer=viewer, 
                points=sub_rois[["yc", "xc"]], 
                properties={"zc": sub_rois["zc"].values},
                size=roi_size,
                edge_color="zc",
                edge_colormap="turbo",
                face_color="transparent", 
                )
            outfile = S.path_to_detected_spot_image_file(position=pos, time=frame, channel=ch)
            print(f"DEBUG: saving image for ({outfile})")
            save_screenshot(viewer=viewer, outfile=outfile, scale=2)
            print(f"DEBUG: saved {outfile}")
            viewer.close()
        if interactive:
            viewer = napari.view_image(img)
            add_points_to_viewer(
                viewer=viewer, 
                points=sub_rois[["zc", "yc", "xc"]], 
                size=roi_size,
                edge_color="red",
                face_color="transparent", 
                )
            napari.run()
            if prompt_continue_napari() == SIGNAL_TO_QUIT:
                break
            # Here we don't call viewer.close() programmatically since it's expected that the user closes the window.
        print("DEBUG: Continuing...")
    if interactive:
        shutdown_napari()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise the detected spot ROIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", type=ExtantFolder.from_string, help="(Optional): Path to folder to save images to.")
    position_subset_spec = parser.add_mutually_exclusive_group()
    position_subset_spec.add_argument("--positions", nargs="*", type=int, help="Indices (0-based) of positions to use, otherwise use all.")
    position_subset_spec.add_argument("--num-positions", type=int, help="Number of positions to use")
    parser.add_argument("--interactive", action="store_true", help="Do interactive image viewing.")
    parser.add_argument("--save-projections", action="store_true", help="Save (max-z-projected) 2D images with bounding boxes.")
    args = parser.parse_args()
    if args.positions:
        positions = args.positions
    elif args.num_positions:
        positions = set(range(args.num_positions))
    else:
        positions = None
    pos_use_msg = f"Positions to use for spot detection visualisation: {', '.join(map(str, positions))}" \
        if positions else "All positions will be used for spot detection visualisation."
    print(pos_use_msg)
    workflow(
        config_file=args.config_path, 
        images_folder=args.image_path, 
        image_save_path=args.image_save_path, 
        interactive=args.interactive,
        save_projections=args.save_projections, 
        positions_to_use=positions,
        )
