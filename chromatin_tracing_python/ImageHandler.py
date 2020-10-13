"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
from chromatin_tracing_python import image_processing_functions as ip
import os
import pandas as pd
import numpy as np
import yaml
import tifffile as tiff

class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize Tracer class with config read in from YAML file.
        '''
        self.config_path = config_path
        self.config = ip.load_config(config_path)
        self.images, self.pos_list = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        self.images_shape = self.images.shape
        self.dc_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv'

    def reload_config(self):
        self.config = ip.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')
    
    def set_drift_table(self, path=None):
        if not path:
            path = self.dc_file_path
        self.drift_table = pd.read_csv(path)
    
    def save_data(self, traces=None, imgs=None, rois=None, pwds=None, pairs=None, config=None, suffix=''):
        output_folder=self.config['output_folder']
        output_filename=self.config['output_file_prefix']
        output_file=output_folder+os.sep+output_filename
        
        if traces is not None:
            traces.to_csv(output_file+'traces.csv')
        if pwds is not None:
            np.save(output_file+'pwds.npy',pwds)
        if rois is not None:
            rois.to_csv(output_file+'rois.csv', index=False)
        if imgs is not None:
            imgs=np.moveaxis(imgs,0,2)
            tiff.imsave(output_file+'imgs.tif', imgs, imagej=True)
        if pairs is not None:
            pairs.to_csv(output_file+'pairs.csv')
        if config is not None:
            with open(output_file+'config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')