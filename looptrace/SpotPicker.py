# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from scipy import ndimage as ndi
from looptrace import image_processing_functions as ip
import pandas as pd
import numpy as np
import random
from skimage.measure import regionprops_table
import tqdm

class SpotPicker:
    def __init__(self, image_handler):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images, self.pos_list = image_handler.images, image_handler.pos_list
        
    def rois_from_spots(self, preview_pos=None, filter_nucs = False):
        '''
        Autodetect ROIs from spot images using a manual threshold defined in config.
        
        Returns
        ---------
        A pandas DataFrame with the bounding boxes and identifyer of the detected ROIs.
        '''

        #Read parameters from config.
        
        ch = self.config['trace_ch']
        spot_frame = self.config['spot_frame']
        if type(spot_frame) == int:
            spot_frame = [spot_frame]

        bead_ch = self.config['bead_ch']
        try:
            subtract_beads = self.config['subtract_beads']
        except KeyError: #Legacy config.
            subtract_beads = False

        try:
            min_dist = self.config['min_spot_dist']
        except KeyError: #Legacy config.
            min_dist = None

        spot_threshold = self.config['spot_threshold']
        if not isinstance(spot_threshold, list):
            spot_threshold = [spot_threshold]*len(spot_frame)
        spot_ds = self.config['spot_downsample']

        #Loop through the imaging positions.
        all_rois = []
        if preview_pos is not None:
            for i, frame in enumerate(spot_frame):
                print(f'Preview spot detection in position {preview_pos}, frame {frame} with threshold {spot_threshold[i]}.')
                pos_index = self.pos_list.index(preview_pos) 
                img = self.images[pos_index, frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()

                if subtract_beads:
                    bead_img = self.images[pos_index, frame, bead_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                    img, orig = ip.subtract_crosstalk(bead_img, img, threshold=self.config['bead_threshold'])

                spot_props, filt_img = ip.detect_spots(img, spot_threshold[i], min_dist = min_dist)
                spot_props['position'] = preview_pos
                spot_props = spot_props.reset_index().rename(columns={'index':'roi_id_pos'})
            
                roi_points, _ = ip.roi_to_napari_points(spot_props, position=preview_pos)
                try:
                    ip.napari_view(np.stack([filt_img, img, orig]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Subtracted', 'Original'])
                except NameError:
                    ip.napari_view(np.stack([filt_img, img]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Original'])

            return
        
        for position in self.pos_list:
            for i, frame in enumerate(spot_frame):
            
                print(f'Detecting spots in position {position}, frame {frame}, ch {ch}.')
                
                pos_index = self.pos_list.index(position)
                img = self.images[pos_index, frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                if subtract_beads:
                    bead_img = self.images[pos_index, frame, bead_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                    img, _ = ip.subtract_crosstalk(bead_img, img, threshold=self.config['bead_threshold'])
                spot_props, _ = ip.detect_spots(img, spot_threshold[i], min_dist = min_dist)
                spot_props[['zc', 'yc', 'xc']] = spot_props[['zc', 'yc', 'xc']]*spot_ds
                
                spot_props['position'] = position
                spot_props['frame'] = frame
                spot_props['ch'] = ch
                all_rois.append(spot_props)
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'index':'roi_id_pos'})
        

        self.image_handler.roi_table = output
        self.image_handler.save_data(rois=output)
        print(f'Found {len(output)} spots.')

        if filter_nucs:
            self.image_handler.load_nucs()
            if not self.image_handler.nuc_masks:
                print('No nuclei mask images found, cannot filter.')
            else:
                print('Filtering in nuclei.')
                filt_rois = ip.filter_rois_in_nucs(output, self.image_handler.nuc_masks, self.image_handler.pos_list, drifts=self.image_handler.drift_table, target_frame=self.config['nuc_ref_frame'])
                filt_rois = filt_rois[filt_rois['nuc_label'] > 0 ]
                self.image_handler.roi_table = filt_rois
                self.image_handler.save_data(rois=filt_rois)
                print(f'Filtering complete, {len(filt_rois)} ROIs after filtering.')
        
        return output

    def rois_from_beads(self):
        print('Detecting bead ROIs for tracing.')
        all_rois = []
        n_fields = self.config['bead_trace_fields']
        n_beads = self.config['bead_trace_number']
        for pos in tqdm.tqdm(random.sample(self.pos_list, k=n_fields)):
            pos_index = self.pos_list.index(pos)
            ref_frame = self.config['bead_reference_frame']
            ref_ch = self.config['bead_ch']
            threshold = self.config['bead_threshold']
            min_bead_int = self.config['min_bead_intensity']

            t_img = self.images[pos_index, ref_frame, ref_ch].compute()
            t_img_label,num_labels = ndi.label(t_img>threshold)
            #t_img_maxima=np.array(ndi.measurements.maximum_position(t_img, 
            #                                            labels=t_img_label, 
            #                                            index=np.random.choice(np.arange(1,num_labels), size=n_points*2)))
            
            spot_props = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
            spot_props = spot_props.query('max_intensity > @min_bead_int').sample(n=n_beads, random_state=1)
            
            spot_props.drop(['label'], axis=1, inplace=True)
            spot_props.rename(columns={'centroid-0': 'zc',
                                        'centroid-1': 'yc',
                                        'centroid-2': 'xc',
                                        'index':'roi_id_pos'},
                                        inplace = True)
                
            spot_props['position'] = pos
            spot_props['frame'] = ref_frame
            spot_props['ch'] = ref_ch
            all_rois.append(spot_props)
        
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'index':'roi_id'})
        self.image_handler.bead_rois = output
        self.image_handler.save_data(rois=output, suffix = '_beads')
        return output