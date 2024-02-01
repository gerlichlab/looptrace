
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import os
import sys
import numpy as np

from skimage.io import imsave
import napari

from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract experimental PSF from bead images.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    exec_flow = parser.add_mutually_exclusive_group(required=True)
    exec_flow.add_argument("--save-images", action="store_true", help="Save comuted mask images.")
    exec_flow.add_argument("--qc", action="store_true", help="Additionally run QC (allows edits).")
    args = parser.parse_args()
    
    prep_image_to_add = np.array if args.qc else (lambda img: img)

    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    N = NucDetector(H)
    
    # Gather the images to use and determine what to do for each FOV.
    seg_imgs = N.images_for_segmentation
    mask_imgs = N.mask_images
    if mask_imgs is None:
        print("Nuclei need to be segmented first.")
        sys.exit()
    
    class_imgs = N.class_images
    if class_imgs is None:
        get_class_layer = lambda _1, _2: None
    else:
        get_class_layer = lambda view, pos_idx: view.add_labels(prep_image_to_add(class_imgs[pos_idx]))

    for i, nuc_img in enumerate(seg_imgs):
        viewer = napari.view_image(nuc_img)
        masks_layer = viewer.add_labels(prep_image_to_add(mask_imgs[i]))
        class_layer = get_class_layer(viewer, i)
        
        if args.save_images:
            screenshot = viewer.screenshot()
            viewer.add_image(screenshot)
            outfile = H.nuclear_mask_images_folder / f"nuc_maks.{i}.png"
            print(f"Saving image for position {i}: {outfile}")
            os.makedirs(outfile.parent, exist_ok=True)
            imsave(outfile, screenshot)
        else:
            napari.run()

            sentinel = "q"
            user_input = input(f"Press enter to continue to next position, or {sentinel} to quit.")
            if user_input == sentinel:
                break

            if args.qc:
                N.update_masks_after_qc(masks_layer.data.astype(np.uint16), np.array(mask_imgs[i]), NucDetector.MASKS_KEY, H.image_lists[NucDetector.MASKS_KEY][i])
                if class_layer is not None:
                    N.update_masks_after_qc(class_layer.data.astype(np.uint16), np.array(class_imgs[i]), NucDetector.CLASSES_KEY, H.image_lists[NucDetector.CLASSES_KEY][i])
                    del class_layer