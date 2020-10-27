"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
from chromatin_tracing_python import image_processing_functions as ip
import os
import pandas as pd
import numpy as np
import yaml
from dask.diagnostics import ProgressBar
import dask.array as da
import tifffile as tiff
import czifile

class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize Tracer class with config read in from YAML file.
        '''
        self.config_path = config_path
        self.config = ip.load_config(config_path)
        self.zarr_path = self.config['input_folder']+os.sep+self.config['output_file_prefix']+'zarr.zarr'
        if os.path.isdir(self.zarr_path):
            self.images = da.from_zarr(self.zarr_path)
            self.pos_list = pd.read_csv(self.zarr_path+'_positions.txt', sep='\n', header=None)[0].to_list()
        else:
            self.images, self.pos_list = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        self.images_shape = self.images.shape
        self.dc_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv'
        self.dc_images = None

    def reload_config(self):
        self.config = ip.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')
    
    def set_drift_table(self, path=None):
        if not path:
            path = self.dc_file_path
        self.drift_table = pd.read_csv(path, index_col=0)
    
    def images_to_zarr(self):
        pbar = ProgressBar()
        pbar.register()
        self.images.to_zarr(self.zarr_path, compression='blosc', compression_opts=dict(cname='zstd', clevel=5, shuffle=2))
        pd.DataFrame(self.pos_list).to_csv(self.zarr_path+'_positions.txt', index=None, sep='\n')
        self.images = da.from_zarr(self.zarr_path)
        print('Images saved as zarr.')
    
    def save_metadata(self):
        first_path = ip.all_matching_files_in_subfolders(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        first_img = czifile.CziFile(first_path)
        out_path = self.config['input_folder']+os.sep+self.config['output_file_prefix']

        meta = first_img.metadata()
        with open(out_path+'metadata.xml', 'w') as file:
            file.writelines(meta)
        
        metadict = first_img.metadata(raw=False)
        with open(out_path+'metadata.yaml', 'w') as file:
            yaml.safe_dump(metadict, file)
        
        print('Metadata saved.')

    def gen_dc_images(self):
        chunk_size = self.images.chunksize
        img_dc = []
        for pos in self.pos_list:
            pos_index = self.pos_list.index(pos)
            pos_img = []
            for t in range(self.images_shape[1]):
                shift = tuple(self.drift_table.query('pos_id == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
                pos_img.append(da.roll(self.images[pos_index,t], shift = shift, axis = (1,2,3)))
            img_dc.append(da.stack(pos_img).rechunk(chunks=chunk_size[1:]))
        self.dc_images = img_dc

        print('DC images generated.')

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