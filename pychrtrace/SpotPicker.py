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
        
    def rois_from_spots(self, preview_pos=False, filter_nucs = False):
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
        if preview_pos:
            print(f'Preview spot detection in position {preview_pos} with threshold {spot_threshold}.')
            pos_index = self.pos_list.index(preview_pos)
            img = self.images[pos_index, ref_slice, trace_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
            spot_props, filt_img = ip.detect_spots(img, spot_threshold)
            spot_props['position'] = preview_pos
            spot_props = spot_props.reset_index().rename(columns={'index':'roi_id'})
            return spot_props, img, filt_img

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
        output=output.reset_index().rename(columns={'index':'roi_id'})
        

        self.image_handler.roi_table = output
        self.image_handler.save_data(rois=output)

        if filter_nucs:
            self.image_handler.load_nucs()
            if not self.image_handler.nuc_masks:
                print('No nuclei mask images found, cannot filter.')
            else:
                print('Filtering in nuclei.')
                filt_rois = ip.filter_rois_in_nucs(output, self.image_handler.nuc_masks, self.image_handler.pos_list)
                filt_rois = filt_rois[filt_rois['nuc_label'] > 0 ]
                self.image_handler.roi_table = filt_rois
                self.image_handler.save_data(rois=filt_rois)

        print(f'Found {len(output)} spots.')
        return output