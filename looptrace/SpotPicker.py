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
        self.images = self.image_handler.images['seq_images']
        self.pos_list = self.image_handler.image_lists['seq_images']
        
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

        try:
            detect_method = self.config['detection_method']
            if detect_method == 'intensity':
                detect_func = ip.detect_spots_int
                print('Using intensity based spot detection with threshold ', spot_threshold)
            else:
                detect_func = ip.detect_spots
                print('Using DoG based spot detection with threshold ',  spot_threshold)
        except KeyError: #Legacy config.
            detect_func = ip.detect_spots
            detect_method = 'dog'
            print('Using DoG based spot detection.')

        #Loop through the imaging positions.
        all_rois = []
        if preview_pos is not None:
            for i, frame in enumerate(spot_frame):
                print(f'Preview spot detection in position {preview_pos}, frame {frame} with threshold {spot_threshold[i]}.')
                pos_index = self.pos_list.index(preview_pos) 
                img = self.images[pos_index][frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()

                if subtract_beads:
                    bead_img = self.images[pos_index][frame, bead_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                    img, orig = ip.subtract_crosstalk(bead_img, img, threshold=self.config['bead_threshold'])

                spot_props, filt_img = detect_func(img, spot_threshold[i], min_dist = min_dist)
                spot_props['position'] = preview_pos
                spot_props = spot_props.reset_index().rename(columns={'index':'roi_id_pos'})
            
                roi_points, _ = ip.roi_to_napari_points(spot_props, position=preview_pos)
                try:
                    ip.napari_view(np.stack([filt_img, img, orig]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Subtracted', 'Original'])
                except NameError:
                    ip.napari_view(np.stack([filt_img, img]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Original'])

            return
        
        for position in tqdm.tqdm(self.pos_list):
            
            for i, frame in enumerate(spot_frame):
            
                #print(f'Detecting spots in position {position}, frame {frame}, ch {ch}.',  end=' ')
                
                pos_index = self.pos_list.index(position)
                img = self.images[pos_index][frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                if subtract_beads:
                    bead_img = self.images[pos_index][frame, bead_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                    img, _ = ip.subtract_crosstalk(bead_img, img, threshold=self.config['bead_threshold'])
                spot_props, _ = detect_func(img, spot_threshold[i], min_dist = min_dist)

                if self.config['detection_method'] != 'intensity':
                    spot_props = ip.roi_center_to_bbox(spot_props, np.array(self.config['roi_image_size'])//spot_ds)

                spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']]*spot_ds
                
                spot_props['position'] = position
                spot_props['frame'] = frame
                spot_props['ch'] = ch
                all_rois.append(spot_props)
        output = pd.concat(all_rois)

        print(f'Found {len(output)} spots.')

        if filter_nucs:
            if 'nuc_masks' not in self.image_handler.images:
                print('No nuclei mask images found, cannot filter.')
            else:
                print('Filtering in nuclei.')
                output = ip.filter_rois_in_nucs(output, np.array(self.image_handler.images['nuc_masks']), self.pos_list, drifts=self.image_handler.tables['drift_correction'], target_frame=self.config['nuc_ref_frame'])
                output = output[output['nuc_label'] > 0 ]
        
        output=output.reset_index().rename(columns={'index':'roi_id_pos'})
        if detect_method == 'dog':
            rois = ip.roi_center_to_bbox(output, roi_size = tuple(self.config['roi_image_size']))
        else:
            rois = output

        rois = rois.sort_values(['position', 'frame'])
        rois.to_csv(self.image_handler.roi_file_path)
        self.image_handler.load_tables()
        print(f'Filtering complete, {len(rois)} ROIs after filtering.')
        
        return rois

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

            t_img = self.images[pos_index][ref_frame, ref_ch].compute()
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
            print('Detected beads in position', pos, spot_props)
            all_rois.append(spot_props)
        
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'index':'roi_id'})
        rois = ip.roi_center_to_bbox(output, roi_size = tuple(self.config['roi_image_size']))

        self.image_handler.bead_rois = rois
        rois.to_csv(self.image_handler.roi_file_path+'_beads.csv')
        return rois

    def refilter_rois(self):
        #TODO needs updating to new table syntax.
        if self.config['detection_method'] == 'dog':
            self.image_handler.roi_table = ip.roi_center_to_bbox(self.image_handler.roi_table.copy(), roi_size = tuple(self.config['roi_image_size']))
        
        self.image_handler.roi_table[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = self.image_handler.roi_table[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']].round().astype(int)

        if 'nuc_images' not in self.image_handler.images:
            print('Please generate nuclei images first.')
        if 'nuc_masks' in self.image_handler.images:
            self.image_handler.roi_table = ip.filter_rois_in_nucs(self.image_handler.roi_table.copy(), np.array(self.image_handler.images['nuc_masks']), self.pos_list, new_col='nuc_label', drifts = self.image_handler.tables['drift_correction'], target_frame=self.config['nuc_ref_frame'])
            
        if 'nuc_classes' in self.image_handler.images:
            self.image_handler.roi_table = ip.filter_rois_in_nucs(self.image_handler.roi_table.copy(), np.array(self.image_handler.images['nuc_classes']), self.pos_list, new_col='nuc_class', drifts = self.image_handler.tables['drift_correction'], target_frame=self.config['nuc_ref_frame'])

        print('ROIs (re)assigned to nuclei.')
        self.image_handler.roi_table.to_csv(self.image_handler.roi_file_path)