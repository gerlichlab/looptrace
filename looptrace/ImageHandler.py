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
import czifile
import joblib

class ImageHandler:
    def __init__(self, config_path, image_path = None, image_save_path = None):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.
        '''
        
        self.config_path = config_path
<<<<<<< Updated upstream
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

    def input_parser(self):
        if self.config['image_filetype'] in ['czi', 'tif']:
            self.images, self.pos_list, self.all_image_files = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
       
        elif self.config['image_filetype'] in ['zip', 'zarr', 'zarrpos']:
            self.pos_list = pd.read_csv(self.config['input_path']+os.sep+self.config['output_prefix']+'positions.txt', sep='\n', header=None)[0].to_list()
            print('Position list found: ', self.pos_list)
            self.images_from_zarr()
        
        else:
            print('Unknown file format, please check config file.')

    def reload_config(self):
        self.config = ip.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')
    
    def set_drift_table(self, path=None):
        if not path:
            path = self.dc_file_path
        self.drift_table = pd.read_csv(path, index_col=0)
    
    def images_to_zarr(self):
        '''
        Function to save images loaded as a position-list of 5D TCZYX dask array into zarr format.
        Will chuck into two last dimensions.
        Also saves a position list of the named positions.
        '''

        with joblib.parallel_backend("threading"):  
            joblib.Parallel(n_jobs=-2)(joblib.delayed(self.single_pos_to_zarr)(pos) for pos in self.pos_list)

        self.images_from_zarr()
        
        pd.DataFrame(self.pos_list).to_csv(self.config['input_path']+os.sep+self.config['output_prefix']+'positions.txt', index=None, header=None, sep='\n')
        print('Images saved as a zarr store.')
    
    def single_pos_to_zarr(self, pos):
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        s = self.images[0].shape
        chunks = (1,1,1,s[-2],s[-1])
        
        print('Saving position ', pos)
        
        z = zarr.open(self.config['input_path']+os.sep+self.config['output_prefix']+'_zarr_'+pos, mode='w', compressor=compressor, shape=s, chunks=chunks)
        i=0
        for img_path in self.all_image_files:
            if pos in img_path:
                img = ip.read_czi_image(img_path)
                z[i] = img
                i+=1

    def images_from_zarr(self):
        in_path = self.config['input_path']+os.sep+self.config['output_prefix']
        filetype = self.config['image_filetype']

        if filetype == 'zarrpos':
            self.images = [da.from_zarr(in_path+'_zarr_'+pos) for pos in self.pos_list]
            print(f'Images loaded from zarr store, {len(self.images)} positions of shape {self.images[0].shape} found.')

        elif filetype == 'zip':
            store = zarr.ZipStore(in_path+'.zip', mode='r')
            self.images = da.from_zarr(zarr.Array(store))
            #self.images = [imgs[i] for i in range(imgs.shape[0])]
            print(f'Images loaded from zarr store: ', self.images)

        elif self.config['image_filetype'] == 'zarr':
            self.images = da.from_zarr(in_path+'.zarr')
            #self.images = [imgs[i] for i in range(imgs.shape[0])]
            print(f'Images loaded from zarr store: ', self.images)
            
        

    def save_metadata(self):
        '''
        Saves czi metadata from czi input files as part of conversion to zarr.
        '''

        first_path = ip.all_matching_files_in_subfolders(self.config['input_path'], self.config['image_filetype']+self.config['image_template'])[0]
        first_img = czifile.CziFile(first_path)
        out_path = self.config['input_path']+os.sep+self.config['output_prefix']

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
        n_t = self.images.shape[1]
        pos_index = self.pos_list.index(pos)
        pos_img = []
        for t in range(n_t):
            shift = tuple(self.drift_table.query('position == @pos').iloc[t][['z_px_course', 'y_px_course', 'x_px_course']])
            pos_img.append(da.roll(self.images[pos_index, t], shift = shift, axis = (1,2,3)))
        self.dc_images = da.stack(pos_img)

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
        
        output_path=self.config['output_path']
        output_filename=self.config['output_prefix']
        output_file=output_path+os.sep+output_filename
        
        if traces is not None:
            traces.to_csv(output_file+'traces.csv', index=None)
        if pwds is not None:
            np.save(output_file+'pwds.npy',pwds)
        if rois is not None:
            rois.reset_index(drop=True, inplace=True)
            rois.to_csv(output_file+'rois.csv')
        if imgs is not None:
            #imgs=np.moveaxis(imgs,0,2)
            tifffile.imsave(output_file+'imgs.tif', imgs, imagej=True)
        if pairs is not None:
            pairs.to_csv(output_file+'pairs.csv')
        if config is not None:
            with open(output_file+'config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')
=======
        self.reload_config()

        self.image_path = image_path

        if self.image_path is not None:
            self.read_images()

        if image_save_path is not None:
            self.image_save_path = image_save_path
        else:
            self.image_save_path = self.image_path
        
        self.out_path = self.config['analysis_path']+os.sep+self.config['analysis_prefix']

        self.load_tables()

    def reload_config(self):
        self.config = image_io.load_config(self.config_path)

    def load_tables(self):
        self.tables = {}
        self.table_paths = {}
        for f in os.scandir(self.config['analysis_path']):
            if f.name.endswith('.csv') and not f.name.startswith('_'):
                table_name = os.path.splitext(f.name)[0].split(self.config['analysis_prefix'])[1]
                print('Loading table ', table_name)
                table = pd.read_csv(f.path, index_col = 0)
                self.tables[table_name] = table
                self.table_paths[table_name] = f.path
            elif f.name.endswith('.pkl') and not f.name.startswith('_'):
                table_name = os.path.splitext(f.name)[0].split(self.config['analysis_prefix'])[1]
                print('Loading table ', table_name)
                table = pd.read_pickle(f.path)
                self.tables[table_name] = table
                self.table_paths[table_name] = f.path

    def read_images(self):
        '''
        Function to load existing images from the input folder, and them into a dictionary (self.images{}),
        with folder name or image name (without extensions) as keys, images as values.
        Standardized to either folders with OME-ZARR, single NPY files or NPZ collections.
        More can be added as needed.
        '''
        self.images = {}
        self.image_lists = {}
        image_paths = [(p.name, p.path) for p in os.scandir(self.image_path)]
        for image_name, image_path in image_paths:
            if image_name.startswith('_') or image_name == 'spot_images_dir':
                continue
            else:
                if os.path.isdir(image_path):
                    if len(os.listdir(image_path)) == 0:
                        continue
                    else:
                        sample_file = os.listdir(image_path)[0]
                        print(image_path)
                        if sample_file.endswith('.nd2'):
                            self.images[image_name], self.image_lists[image_name] = image_io.stack_nd2_to_dask(image_path)
                            print('Loaded images: ', image_name)
                        elif sample_file.endswith('.tiff') or sample_file.endswith('.tif'):
                            self.images[image_name], self.image_lists[image_name] = image_io.stack_tif_to_dask(image_path)
                            print('Loaded images: ', image_name)
                        else:
                            self.images[image_name], self.image_lists[image_name] = image_io.multi_ome_zarr_to_dask(image_path, remove_unused_dims = False)
                            print('Loaded images: ', image_name)

                elif image_name.endswith('.npz'):
                    self.images[os.path.splitext(image_name)[0]] = image_io.NPZ_wrapper(image_path)
                    print('Loaded images: ', image_name)
                elif image_name.endswith('.npy'):
                    try:
                        self.images[os.path.splitext(image_name)[0]] = np.load(image_path, mmap_mode = 'r')
                    except ValueError: #This is for legacy datasets, will be removed after dataset cleanup!
                        self.images[os.path.splitext(image_name)[0]] = np.load(image_path, allow_pickle = True)
                    print('Loaded images: ', image_name)
>>>>>>> Stashed changes
