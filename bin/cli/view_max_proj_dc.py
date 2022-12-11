"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.Drifter import Drifter
import dask.array as da
import napari
import sys
import os

if __name__ == '__main__':
    H = ImageHandler(sys.argv[1])
    if 'max_proj_dc' in H.images:
        try:
            imgs = da.stack(H.images['max_proj_dc'], axis=0)
            ch_axis = 2
        except ValueError:
            pos_idx = int(input('Enter position index: '))
            imgs = H.images['max_proj_dc'][pos_idx]
            ch_axis = 1
        napari.view_image(imgs, channel_axis=ch_axis)
        napari.run()
        _ = input('Press enter once images viewed.')
    else:
        print('Please save max proj DC images first.')