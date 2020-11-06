"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
from pychrtrace import image_processing_functions as ip
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from dask.diagnostics import ProgressBar
import dask.array as da
import tifffile
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
            print('Images loaded from ZARR file, shape is ', self.images.shape)
            print('Positions found: ', self.pos_list)
        else:
            self.images, self.pos_list = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        self.images_shape = self.images.shape
        self.dc_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv'
        self.dc_images = None
        self.nucs = None
        self.nuc_masks = None
        self.nuc_class = None
        
        self.nuc_folder = self.config['output_folder']+os.sep+'nucs'

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
        zarr_img = da.rechunk(self.images, chunks=(1,1,1,1,-1,-1))
        zarr_img.to_zarr(self.zarr_path, compression='blosc', compression_opts=dict(cname='zstd', clevel=5, shuffle=2))
        pd.DataFrame(self.pos_list).to_csv(self.zarr_path+'_positions.txt', index=None, header=None, sep='\n')
        self.images = da.from_zarr(self.zarr_path)
        print('Images saved as zarr.')
    
    def save_metadata(self):
        first_path = ip.all_matching_files_in_subfolders(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])[0]
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

    def gen_nuc_images(self):
        imgs = []
        for pos in self.pos_list:
            pos_index = self.pos_list.index(pos)
            img = da.max(self.images[pos_index,self.config['ref_slice'], self.config['nuc_channel']], axis=0).compute()
            imgs.append(img)
        self.nucs = imgs
        self.save_nucs(img_type='raw')
    
    def load_nucs(self):
        print('Loading nucleus images from ', self.nuc_folder)
        self.nucs = [tifffile.imread(img) for img in Path(self.nuc_folder).glob('nuc_raw_*.tiff')]
        self.nuc_masks = [tifffile.imread(img) for img in Path(self.nuc_folder).glob('nuc_labels_*.tiff')]
        self.nuc_class = [np.load(img) for img in Path(self.nuc_folder).glob('nuc_raw_*_Object*.npy')]

        if self.nucs == []:
            print('No nuclei images found here, please run detect nuclei first.')
            self.nucs = None
        if self.nuc_masks == []:
            print('No nuclei masks found here, please run detect nuclei first.')
            self.nuc_masks = None
        if self.nuc_class == []:
            print('No classification images found, please run classification first if desired.')
            self.nuc_class = None

    def save_nucs(self, img_type):
        Path(self.nuc_folder).mkdir(parents=True, exist_ok=True)
        imgs = []
        for pos in self.pos_list:
            pos_index = self.pos_list.index(pos)
            if img_type=='raw':
                img = self.nucs[pos_index]
                tifffile.imsave(self.nuc_folder+os.sep+'nuc_raw_'+pos+'.tiff', data=img)
            elif img_type=='mask':
                img = self.nuc_masks[pos_index]
                tifffile.imsave(self.nuc_folder+os.sep+'nuc_labels_'+pos+'.tiff', data=img)
            elif img_type=='class':
                img = self.nuc_class[pos_index]
                np.save(self.nuc_folder+os.sep+'nuc_raw_'+pos+'_Object Predictions.npy', img)

    def save_data(self, traces=None, imgs=None, rois=None, pwds=None, pairs=None, config=None, suffix=''):
        output_folder=self.config['output_folder']
        output_filename=self.config['output_file_prefix']
        output_file=output_folder+os.sep+output_filename
        
        if traces is not None:
            traces.to_csv(output_file+'traces.csv')
        if pwds is not None:
            np.save(output_file+'pwds.npy',pwds)
        if rois is not None:
            rois.reset_index(drop=True, inplace=True)
            rois.to_csv(output_file+'rois.csv')
        if imgs is not None:
            imgs=np.moveaxis(imgs,0,2)
            tifffile.imsave(output_file+'imgs.tif', imgs, imagej=True)
        if pairs is not None:
            pairs.to_csv(output_file+'pairs.csv')
        if config is not None:
            with open(output_file+'config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')