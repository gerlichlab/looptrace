"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""

import os
import yaml
import chromatin_tracing_python.image_processing_functions as ip
import pandas as pd

class Handler:
    def __init__(self, config_path):
        '''
        Initialize Handler class with config read in from YAML file.
    '''
        
        self.config_path = config_path
        self.config = ip.load_config(config_path)
        
        self.drift_table = None
        if os.path.exists(default_dc_path):
            self.drift_table = pd.read_csv(dc_path)

        self.images, self.pos_list = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        self.images_shape = self.images.shape
        

        print('Loaded images of shape ', self.images_shape)
        print('Found positions ', self.pos_list)