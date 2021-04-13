# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import numpy as np
import pandas as pd
from looptrace import image_processing_functions as ip

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.config = image_handler.config
        self.dc_file_path = image_handler.dc_file_path
        self.images, self.pos_list = image_handler.images, image_handler.pos_list

    def drift_corr(self):
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images.
        '''

        images = self.images
        pos_list = self.pos_list
        t_slice = self.config['bead_reference_frame']
        t_all = range(images[0].shape[0])
        ch = self.config['bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']

        #Run drift correction for each position and save results in table.
        all_drifts=[]
        for i, pos in enumerate(pos_list):
            print(f'Running drift correction for position {pos}')
            drifts_course = []
            drifts_fine = []
            for t in t_all:
                print('Drift correcting frame', t)
                t_img = np.array(images[i, t_slice, ch])
                o_img = np.array(images[i, t, ch])
                drift_course = ip.drift_corr_course(t_img, o_img, downsample=1)
                drifts_course.append(drift_course)
                drifts_fine.append(ip.drift_corr_multipoint_cc(t_img, 
                                                                o_img,
                                                                drift_course, 
                                                                threshold, 
                                                                min_bead_int, 
                                                                n_points))
            print('Drift correction complete in position.')
            drifts = pd.concat([pd.DataFrame(drifts_course), pd.DataFrame(drifts_fine)], axis = 1)
            drifts['position'] = pos
            drifts.index.name = 'frame'
            all_drifts.append(drifts)
            print('Finished drift correction for position ', pos)
        
        all_drifts=pd.concat(all_drifts).reset_index()
        
        all_drifts.columns=['frame',
                            'z_px_course',
                            'y_px_course',
                            'x_px_course',
                            'z_px_fine',
                            'y_px_fine',
                            'x_px_fine',
                            'position']
        all_drifts.to_csv(self.dc_file_path)
        print('Drift correction complete.')
        return all_drifts