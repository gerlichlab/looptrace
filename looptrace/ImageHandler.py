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
from joblib import Parallel, delayed
from scipy import ndimage as ndi

class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        
        self.config_path = config_path
        self.config = ip.load_config(config_path)
        self.images_path = self.config['input_path']+os.sep+'seq_images'
        self.input_parser()
        self.out_path = self.config['output_path']+os.sep+self.config['output_prefix']
        self.dc_file_path = self.out_path+'drift_correction.csv'
        self.roi_file_path = self.out_path+'rois.csv'
        
        try:
            self.load_drift_table()
        except FileNotFoundError:
            self.drift_table = None
        
        try:
            self.load_roi_table()
        except FileNotFoundError:
            self.roi_table = None

        self.dc_images = None
        self.cell_images_path = self.config['input_path'] + os.sep + 'cell_images' 
        self.load_cell_images()
        self.spot_images_path = self.config['input_path'] + os.sep + 'spot_images' 
        self.load_spot_images()
        self.maxz_dc_folder = self.config['input_path']+os.sep+'maxz_dc'
        

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
            self.pos_list = sorted([p.name for p in os.scandir(self.config['input_path']) if os.path.isdir(p)])
            self.images_from_zarr()

        elif ft in ['nikon_tiff', 'nikon_tiff_multifolder']:
            self.images_from_zarr()
            self.pos_list = ['P'+str(i).zfill(4)+'.zarr' for i in range(1,self.images.shape[0]+1)]
        
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
    
    def load_drift_table(self, path=None):
        if not path:
            path = self.dc_file_path
        self.drift_table = pd.read_csv(path, index_col=0)
    
    def load_roi_table(self, path = None):
        if not path:
            path = self.roi_file_path
        self.roi_table = ip.rois_from_csv(path)

    def images_from_zarr(self):
        in_path = self.config['input_path']
        filetype = self.config['image_filetype']

        if filetype == 'ome-zarr':
            if os.path.isdir(self.images_path):
                self.images, self.pos_list = ip.multi_ome_zarr_to_dask(self.images_path)
                print(f'Images loaded from zarr store: ', self.images)
            else:
                print('No seq images found.')

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
            for path in sorted(os.listdir(in_path)):
                try:
                    print('Reading TIFF files from ', in_path+os.sep+path)
                    images.append(ip.nikon_tiff_to_dask(in_path+os.sep+path))
                except ValueError:
                    continue
            self.images = da.concatenate(images, axis = 1)
            print(f'Images loaded from multiple tiff folders: ', self.images)
        try:
            self.images = self.images.astype(np.uint16)
        except AttributeError:
            pass

    def load_cell_images(self):
        '''
        Function to load existing nuclear images from nucs folder in analysis folder.
        '''
        self.cell_images = {}
        try:
            image_folders = sorted([(p.name, p.path) for p in os.scandir(self.cell_images_path) if os.path.isdir(p)])
            for name, path in image_folders:

                self.cell_images[name], image_names = ip.multi_ome_zarr_to_dask(path)
                self.cell_images[name] = self.cell_images[name][:,0,0,0,:,:]
                print('Loaded cell images: ', name)
                
        except FileNotFoundError:
            print('Could not find any cell images.')

    def load_spot_images(self):
        '''
        Function to load existing nuclear images from nucs folder in analysis folder.
        '''
        self.spot_images = {}

        if os.path.isdir(self.spot_images_path):
            try:
                image_folders = sorted([(p.name, p.path) for p in os.scandir(self.spot_images_path)])
                for name, path in image_folders:
                    name = os.path.splitext(name)[0]
                    try:
                        self.spot_images[name] = np.load(path, mmap_mode='r')
                    except ValueError:
                        self.spot_images[name] = np.load(path, allow_pickle=True)
                    print('Loaded spot images: ', name)
                    
            except FileNotFoundError:
                print('Could not find any spot images.')
        else:
            os.makedirs(self.spot_images_path)
    
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


    def save_proj_dc_images(self):
        '''
        Makes internal coursly drift corrected images based on precalculated drift
        correction (see Drifter class for details).
        
        P, T, C, Z, Y, X = self.images.shape
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y,X)
        
        if not os.path.isdir(self.maxz_dc_folder):
            os.mkdir(self.maxz_dc_folder)
        '''

        imgs = []
        for pos in tqdm.tqdm(self.pos_list):
            pos_img = []
            pos_index = self.pos_list.index(pos)
            for t in tqdm.tqdm(range(self.images.shape[1])):
                shift = self.drift_table.query('position == @pos').iloc[t][['y_px_course', 'x_px_course', 'y_px_fine', 'x_px_fine']]
                shift = (shift[0]+shift[2], shift[1]+shift[3])
                proj_img = da.max(self.images[pos_index, t], axis=1).compute()
                proj_img = ndi.shift(proj_img, shift=(0,)+shift, order = 2)
                pos_img.append(proj_img)
            pos_img = np.stack(pos_img)
            pos_img = pos_img[:,:,np.newaxis, :, :]
            imgs.append(pos_img)
        
        imgs = np.stack(imgs)

        ip.imgs_to_ome_zarr(images=imgs, path=self.maxz_dc_folder, name='max_proj_dc', axes=['p','t','c','z','y','x'])
        
        print('DC images generated.')

    def load_proj_dc_images(self):
        try:
            self.maxz_dc_images, _ = ip.multi_ome_zarr_to_dask(self.maxz_dc_folder)
        except FileNotFoundError:
            print('Could not find maxz_dc images, generate first.')

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
            tifffile.imsave(output_file+'spot_imgs'+suffix+'.tif', imgs, imagej=True)
        if pairs is not None:
            pairs.to_csv(output_file+'pairs.csv')
        if config is not None:
            with open(output_file+'config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')