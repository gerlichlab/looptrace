
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
import napari
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nucleus detection on images.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument('image_name', help="Name of images to show (folder name)")
    parser.add_argument('pos_index', help='Index of position to view.', default='0')
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    
    napari.view_image(H.images[args.image_name][int(args.pos_index)], channel_axis=1, contrast_limits=(100,2000))
    napari.run()
    _ = input('Press enter once images viewed.')