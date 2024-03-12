
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from itertools import dropwhile, takewhile
import sys
from typing import *

import numpy as np
import napari
import pandas as pd
from skimage.measure import regionprops_table

from gertils import ExtantFile, ExtantFolder
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace.napari_helpers import SIGNAL_TO_QUIT, prompt_continue_napari, save_screenshot, shutdown_napari

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


def workflow(
    rounds_config: ExtantFile,
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    *, 
    save_images: bool = True, 
    do_qc: bool = False, 
    ):
    """
    Either save detected nuclei / mask images, or examine them interactively with napari.

    Parameters
    ----------
    rounds_config : gertils.ExtantFile
        Path to the configuration file for declaration of imaging rounds
    params_config : gertils.ExtantFile
        Path to the main looptrace processing configuration file
    images_folder : gertils.ExtantFolder
        Path to an experiment's main images folder
    save_images : bool
        Indicate that nuclei images should simply be saved, rather than examined interactivel. 
        This is mutually exclusive with do_qc.
    do_qc : bool
        Indicate that you'd like to interactively do QC with the nuclei images. 
        This is mutually exclusive with save_images.
    start_from : int, optional
        0-based index (inclusive) of the position (FOV) with which to start the 
        image saving or interactive examination
    stop_after : int, optional
        0-based index (inclusive) of the position (FOV) at which to stop the 
        image saving or interactive examination

    Raises
    ------
    ValueError : if image saving and interactive QC are both selected
    
    """
    if save_images and do_qc:
        raise ValueError("Cannot do interactive QC and image saving at the same time!")
    
    prep_image_to_add = np.array if do_qc else (lambda img: img)

    H = ImageHandler(rounds_config, params_config, images_folder)
    N = NucDetector(H)
    
    # Gather the images to use and determine what to do for each FOV.
    mask_imgs = N.mask_images
    class_imgs = N.class_images
    get_class_layer = (lambda *_: None) if class_imgs is None else (lambda view, pos_idx: view.add_labels(prep_image_to_add(class_imgs[pos_idx])))

    for i, nuc_img in enumerate(N.images_for_segmentation):
        viewer = napari.view_image(nuc_img)
        mask = mask_imgs[i]
        point_table = pd.DataFrame(regionprops_table(mask, properties=("label", "centroid")))
        masks_layer = viewer.add_labels(prep_image_to_add(mask))
        class_layer = get_class_layer(viewer, i)
        label_layer = viewer.add_points(
            point_table[["centroid-0", "centroid-1"]].values, 
            properties={"label": point_table["label"].values},
            text={
                "string": "{label}",
                "size": 10,
                "color": "black",
                'translation': np.array([-30, 0]),
                },
            size=1,
            face_color="transparent", 
            edge_color="transparent"
            )
        if save_images:
            print(f"DEBUG -- saving nuclei image for position: {i}")
            outfile = H.nuclear_mask_screenshots_folder / f"nuc_maks.{i}.png"
            save_screenshot(viewer=viewer, outfile=outfile)
            print(f"DEBUG -- saved nuclei image: {outfile}")
        else:
            napari.run()
            if prompt_continue_napari() == SIGNAL_TO_QUIT:
                break
            if do_qc:
                N.update_masks_after_qc(
                    new_mask=masks_layer.data.astype(np.uint16), 
                    old_mask=np.array(mask), 
                    mask_name=NucDetector.MASKS_KEY, 
                    position=H.image_lists[NucDetector.MASKS_KEY][i],
                    )
                if class_layer is not None:
                    N.update_masks_after_qc(
                        new_mask=class_layer.data.astype(np.uint16), 
                        old_mask=np.array(class_imgs[i]), 
                        mask_name=NucDetector.CLASSES_KEY, 
                        position=H.image_lists[NucDetector.CLASSES_KEY][i],
                        )
        print("Removing layers and closing current viewer...")
        del masks_layer
        del class_layer
        del label_layer
        viewer.close()
    
    shutdown_napari()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label detected nuclei with masks / mask images, optionally performing QC.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    exec_flow = parser.add_mutually_exclusive_group(required=True)
    exec_flow.add_argument("--save-images", action="store_true", help="Save comuted mask images.")
    exec_flow.add_argument("--qc", action="store_true", help="Additionally run QC (allows edits).")
    
    args = parser.parse_args()
    
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        save_images=args.save_images, 
        do_qc=args.qc, 
        )
    