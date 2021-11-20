# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

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
import tqdm
import json
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from multiprocessing.pool import ThreadPool
import dask
dask.config.set(pool=ThreadPool(20))

class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        
        self.config_path = config_path
        self.config = ip.load_config(config_path)

        self.input_parser()
        self.out_path = self.config['output_path']+os.sep+self.config['output_prefix']
        self.dc_file_path = self.out_path+'drift_correction.csv'
        self.roi_file_path = self.out_path+'rois.csv'
        self.roi_table = None
        self.dc_images = None
        self.nucs = None
        self.nuc_masks = None
        self.nuc_class = None
        
        self.nuc_folder = self.config['output_path']+os.sep+'nucs'
        self.maxz_dc_folder = self.config['output_path']+os.sep+'maxz_dc'

    def input_parser(self):
        ft = self.config['image_filetype']
        if ft in ['czi']:
            self.images, self.pos_list, self.all_image_files = ip.images_to_dask(self.config['input_path'], self.config['image_filetype']+self.config['image_template'])
       
        elif ft in ['zip', 'zarr']:
            try:
                self.pos_list = pd.read_csv(self.config['input_path']+os.sep+self.config['output_prefix']+'positions.txt', sep='\n', header=None)[0].to_list()
                print('Position list found: ', self.pos_list)
                self.images_from_zarr()
            except FileNotFoundError:
                self.images_from_zarr()
                self.pos_list = ['P'+str(i).zfill(4) for i in range(1,self.images.shape[0]+1)]
        elif ft == 'ome-zarr':
            self.pos_list = [p.name for p in os.scandir(self.config['input_path']) if os.path.isdir(p)]
            self.images_from_zarr()

        elif ft in ['nikon_tiff', 'nikon_tiff_multifolder']:
            self.images_from_zarr()
            self.pos_list = ['P'+str(i).zfill(4) for i in range(1,self.images.shape[0]+1)]
        
        else:
            print('Unknown file format, please check config file.')

        try:
            crop = self.config['crop_xy']
            Y = self.images.shape[-2]
            X = self.images.shape[-1]
            newY = int(Y  * (1 - crop) / 2)
            newX = int(X  * (1 - crop) / 2)
            self.images = self.images[..., newY:(Y-newY), newX:(X-newX)]
            print('Images cropped to ', self.images.shape)
        except KeyError: #Crop attribute not set.
            pass

    def reload_config(self):
        self.config = ip.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')
    
    def set_drift_table(self, path=None):
        if not path:
            path = self.dc_file_path
        self.drift_table = pd.read_csv(path, index_col=0)

    def images_from_zarr(self):
        in_path = self.config['input_path']
        filetype = self.config['image_filetype']

        if filetype == 'ome-zarr':
            self.images = da.stack([da.from_zarr(in_path+os.sep+pos+os.sep+'0') for pos in self.pos_list])
            print(f'Images loaded from zarr store: ', self.images)

        elif filetype == 'zip':
            store = zarr.ZipStore(in_path, mode='r')
            self.images = da.from_zarr(zarr.Array(store))
            #self.images = [imgs[i] for i in range(imgs.shape[0])]
            print(f'Images loaded from zarr store: ', self.images)

        elif filetype == 'zarr':
            self.images = da.from_zarr(in_path)
            #self.images = [imgs[i] for i in range(imgs.shape[0])]
            print(f'Images loaded from zarr store: ', self.images)
        
        elif filetype == 'nikon_tiff':
            self.images = ip.nikon_tiff_to_dask(in_path)
            print(f'Images loaded from tiff folder: ', self.images)

        elif filetype == 'nikon_tiff_multifolder':
            images = []
            for path in os.listdir(in_path):
                try:
                    images.append(ip.nikon_tiff_to_dask(in_path+os.sep+path))
                except ValueError:
                    continue
            self.images = da.concatenate(images, axis = 1)
            print(f'Images loaded from multiple tiff folders: ', self.images)

            
        
    def gen_dc_images(self, pos):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''
        n_t = self.images.shape[1]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
            pos_img.append(da.roll(self.images[pos_index, t], shift = shift, axis = (1,2,3)))
        self.dc_images = da.stack(pos_img)

        print('DC images generated.')

    def dask_to_ome_zarr(self):
        '''
        Saves a Dask array to ome-zarr format (kinda, metadata still needs work)
        '''

        def single_image_to_zarr(z, idx, img):
            z[idx] = img
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y//2,X//2)
        
        image_path = self.config['input_path']+os.sep+'zarr_images'

        if not os.path.isdir(image_path):
            os.mkdir(image_path)

        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.pos_list.index(pos)

            store = zarr.DirectoryStore(image_path+os.sep+pos)
            root = zarr.group(store=store, overwrite=True)

            root.attrs['multiscale'] = {'multiscales': [{'version': '0.2', 'name': 'dataset', 'datasets': [{'path': '0'}]}]}

            multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=(T, C, Z, Y, X), chunks=chunks)

            Parallel(n_jobs=-1, prefer='threads', verbose=10)(delayed(single_image_to_zarr)(multiscale_level, t, self.images[pos_index, t]) for t in range(T))

        print('OME ZARR images generated.')

    def save_proj_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        '''
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)

        for pos in tqdm.tqdm(self.pos_list):
            pos_img = []
            pos_index = self.pos_list.index(pos)
            for t in tqdm.tqdm(range(T)):
                shift = self.drift_table.query('position == @pos').iloc[t][['y_px_course', 'x_px_course', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                proj_img = da.max(self.images[pos_index, t], axis=1).compute()
                proj_img = ndi.shift(proj_img, shift=(0,)+shift, order = 3)
                pos_img.append(proj_img)
            pos_img = np.stack(pos_img)
            pos_img = pos_img[:,:,np.newaxis, :, :]

            store = zarr.DirectoryStore(self.maxz_dc_folder+os.sep+pos)
            root = zarr.group(store=store, overwrite=True)

            root.attrs['multiscale'] = {'multiscales': [{'version': '0.2', 'name': 'dataset', 'datasets': [{'path': '0'}]}]}

            multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=pos_img.shape, chunks=chunks)
            multiscale_level[:] = pos_img
        
        print('DC images generated.')

    def load_proj_dc_images(self):
        try:
            self.maxz_dc_images = ip.multi_ome_zarr_to_dask(self.maxz_dc_folder)
        except FileNotFoundError:
            print('Could not find maxz_dc images, generate first.')

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
        
        output_path=self.config['output_path']
        output_filename=self.config['output_prefix']
        output_file=output_path+os.sep+output_filename
        
        if traces is not None:
            traces.to_csv(output_file+'traces'+suffix+'.csv', index=None)
        if pwds is not None:
            np.save(output_file+'pwds.npy',pwds)
        if rois is not None:
            rois.reset_index(drop=True, inplace=True)
            rois.to_csv(output_file+'rois'+suffix+'.csv')
        if imgs is not None:
            #imgs=np.moveaxis(imgs,0,2)
            print(imgs.shape)
            tifffile.imsave(output_file+'imgs'+suffix+'.tif', imgs, imagej=True)
        if pairs is not None:
            pairs.to_csv(output_file+'pairs.csv')
        if config is not None:
            with open(output_file+'config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')