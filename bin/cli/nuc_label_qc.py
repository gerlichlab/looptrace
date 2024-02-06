
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
from itertools import dropwhile, takewhile
import os
import sys
from typing import *

import numpy as np
import napari

from gertils import ExtantFile, ExtantFolder
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace.napari_helpers import SIGNAL_TO_QUIT, prompt_continue_napari, save_screenshot, shutdown_napari

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


def workflow(
    config_file: ExtantFile, 
    images_folder: ExtantFolder, 
    *, 
    save_images: bool = True, 
    do_qc: bool = False, 
    start_from: Optional[int] = None, 
    stop_after: Optional[int] = None,
    ):
    """
    Either save detected nuclei / mask images, or examine them interactively with napari.

    Parameters
    ----------
    config_file : gertils.ExtantFile
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

    H = ImageHandler(config_file, images_folder)
    N = NucDetector(H)
    
    # Gather the images to use and determine what to do for each FOV.
    seg_imgs = N.images_for_segmentation
    mask_imgs = N.mask_images
    class_imgs = N.class_images
    get_class_layer = (lambda *_: None) if class_imgs is None else (lambda view, pos_idx: view.add_labels(prep_image_to_add(class_imgs[pos_idx])))

    for i, nuc_img in takewhile(lambda i_: i_[0] <= stop_after, dropwhile(lambda i_: i_[0] < start_from, enumerate(seg_imgs))):
        viewer = napari.view_image(nuc_img)
        masks_layer = viewer.add_labels(prep_image_to_add(mask_imgs[i]))
        class_layer = get_class_layer(viewer, i)
        if save_images:
            print(f"DEBUG: saving image for position: {i}")
            outfile = H.nuclear_mask_screenshots_folder / f"nuc_maks.{i}.png"
            save_screenshot(viewer=viewer, outfile=outfile)
            print(f"DEBUG: saved image {outfile}")
        else:
            napari.run()
            if prompt_continue_napari() == SIGNAL_TO_QUIT:
                break
            if do_qc:
                N.update_masks_after_qc(
                    new_mask=masks_layer.data.astype(np.uint16), 
                    old_mask=np.array(mask_imgs[i]), 
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
        viewer.close()
    
    shutdown_napari()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract experimental PSF from bead images.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("config_file", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    exec_flow = parser.add_mutually_exclusive_group(required=True)
    exec_flow.add_argument("--save-images", action="store_true", help="Save comuted mask images.")
    exec_flow.add_argument("--qc", action="store_true", help="Additionally run QC (allows edits).")
    parser.add_argument("--start-from", type=int, default=0, help="Minimum (inclusive) position index (0-based) to use")
    parser.add_argument("--stop-after", type=int, default=sys.maxsize, help="Maximum (inclusive) position index (0-based) to use")
    
    args = parser.parse_args()
    
    workflow(
        config_file=args.config_file, 
        images_folder=args.images_folder, 
        save_images=args.save_images, 
        do_qc=args.qc, 
        start_from=args.start_from, 
        stop_after=args.stop_after,
        )
    