# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from czifile.czifile import CziFile
from looptrace import image_processing_functions as ip
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import dask.array as da
from numcodecs import Blosc
import zarr
import tifffile
import czifile
import joblib

class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        
        self.config_path = config_path
        self.config = ip.load_config(config_path)

        if self.config['input_zarr_file']:
            self.zarr_path = self.config['input_zarr_file']
        else:
            self.zarr_path = self.config['input_folder']+os.sep+self.config['output_file_prefix']+'zarr'

        try:
            self.pos_list = pd.read_csv(self.zarr_path+'_positions.txt', sep='\n', header=None)[0].to_list()
            print('Position list found: ', self.pos_list)
            self.images_from_zarr()
        except FileNotFoundError:
            try:
                self.images, self.pos_list, self.all_image_files = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
            except ValueError:
                print('No images found, check configuration.')
        self.dc_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv'
        self.roi_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'rois.csv'
        self.roi_table = None
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
    
    def images_to_zarr(self, tif = False):
        '''
        Function to save images loaded as a position-list of 5D TCZYX dask array into zarr format.
        Will chuck into two last dimensions.
        Also saves a position list of the named positions.
        '''

        template =  self.config['image_filetype']+self.config['image_template']
        if tif:
            for i, pos in enumerate(self.pos_list):
                #if not os.path.isdir(self.zarr_path):
                #    os.mkdir(self.zarr_path)
                print('Saving position ', pos)
                ip.czi_to_tif(self.config['input_folder'], template, self.zarr_path+os.sep, self.config['output_file_prefix']+'_'+pos)
        else:
            with joblib.parallel_backend("threading"):  
                joblib.Parallel(n_jobs=-3)(joblib.delayed(self.single_pos_to_zarr)(pos) for pos in self.pos_list)
        
        self.images_from_zarr()
        
        pd.DataFrame(self.pos_list).to_csv(self.zarr_path+'_positions.txt', index=None, header=None, sep='\n')
        print('Images saved as a zarr store.')
    
    def single_pos_to_zarr(self, pos):
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        s = self.images[0].shape
        chunks = (1,1,1,s[-2],s[-1])
        
        print('Saving position ', pos)
        
        z = zarr.open(self.zarr_path+'_'+pos, mode='w', compressor=compressor, shape=s, chunks=chunks)
        i=0
        for img_path in self.all_image_files:
            if pos in img_path:
                img = ip.read_czi_image(img_path)
                z[i] = img
                i+=1

    def images_from_zarr(self):
        try:
            self.images = [da.from_zarr(self.zarr_path+'_'+pos) for pos in self.pos_list]
        except (FileNotFoundError, zarr.errors.ArrayNotFoundError): #Legacy, not split into positions
            try:
                if '.zip' in self.zarr_path:
                    store = zarr.ZipStore(self.zarr_path, mode='r')
                else:
                    store = zarr.open(self.zarr_path, mode='r')
                imgs = da.from_zarr(store)
                self.images = [imgs[i] for i in range(imgs.shape[0])]
            except (FileNotFoundError, zarr.errors.ArrayNotFoundError):
                self.images = ip.tif_store_to_dask(self.zarr_path, self.pos_list[0][0]+'[0-9]{4}')
            
        print(f'Images loaded from zarr store, {len(self.images)} positions of shape {self.images[0].shape} found.')

    def save_metadata(self):
        '''
        Saves czi metadata from czi input files as part of conversion to zarr.
        '''

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

    def gen_dc_images(self, pos):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''

        chunk_size = self.images[0].chunksize
        img_dc = []
        n_t = self.images[0].shape[0]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('pos_id == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
            pos_img.append(da.roll(self.images[pos_index][t], shift = shift, axis = (1,2,3)))
        self.dc_images = da.stack(pos_img)#.rechunk(chunks=chunk_size)

        print('DC images generated.')

    def gen_nuc_images(self):
        '''
        Saves 2D max projected images of the nuclear channel into analysis folder for later analysis.
        '''

        imgs = []
        for pos in self.pos_list:
            pos_index = self.pos_list.index(pos)
            img = da.max(self.images[pos_index][self.config['nuc_ref_frame'], self.config['nuc_channel']], axis=0).compute()
            imgs.append(img)
        self.nucs = imgs
        self.save_nucs(img_type='raw')
    
    def load_nucs(self):
        '''
        Function to load existing nuclear images from nucs folder in analysis folder.
        '''

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
        '''
        Function to save nuclear images, either raw or the masks, as tiff files in nucs folder.

        Args:
            img_type ([str]): Type of images to save, can be 'raw', 'mask' or 'class'.
        '''
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
        '''
        Helper function to save output data from the various modules into the output folder.
        '''
        
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