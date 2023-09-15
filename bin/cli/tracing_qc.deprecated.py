"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import napari
from looptrace.ImageHandler import ImageHandler
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spot detection on all frames and channels listed in config.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument("trace_id", help="Trace_id to visualize.", default=0)
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
        
    points = H.tables['traces'][H.tables['traces']['trace_id'] == int(args.trace_id)][['frame', 'z_px', 'y_px', 'x_px']].to_numpy()
    print(points)
    imgs=np.max(H.images['spot_images'][int(args.trace_id)], axis=1)
    print(imgs.shape)
    viewer = napari.view_image(imgs, contrast_limits=(200, 2000))
    viewer.add_points(points[:,(0,2,3)], size=[0,1,1], face_color='blue', edge_color='blue', symbol='cross', n_dimensional=False)
    napari.run()