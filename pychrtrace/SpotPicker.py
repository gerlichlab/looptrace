"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
from pychrtrace import image_processing_functions as ip
import pandas as pd

class SpotPicker:
    def __init__(self, image_handler):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images, self.pos_list = image_handler.images, image_handler.pos_list
        
    def rois_from_spots(self):
        '''
        Autodetect ROIs from spot images using a manual threshold defined in config.
        
        Returns
        ---------
        A pandas DataFrame with the bounding boxes and identifyer of the detected ROIs.
        '''

        #Read parameters from config.
        trace_ch = self.config['trace_ch']
        ref_slice = self.config['ref_slice']
        spot_threshold = self.config['spot_threshold']
        spot_ds = self.config['spot_downsample']

        #Loop through the imaging positions.
        all_rois = []
        for position in self.pos_list:
            #Read correct image
            print(f'Detecting spots in position {position}.')
            pos_index = self.pos_list.index(position)
            img = self.images[pos_index, ref_slice, trace_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
            spot_props, _ = ip.detect_spots(img, spot_threshold)
            spot_props[['zc', 'yc', 'xc']] = spot_props[['zc', 'yc', 'xc']]*spot_ds
            spot_props['position'] = position
            all_rois.append(spot_props)
        output = pd.concat(all_rois)
        output=output.reset_index()
        print(f'Found {len(output)} spots.')

        self.image_handler.roi_table = output
        self.image_handler.save_data(rois=output)

        return output