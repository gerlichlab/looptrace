# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:30:43 2020

@author: ellenberg
"""
import os
import numpy as np
import pandas as pd
from chromatin_tracing_python import image_processing_functions as ip
#from joblib import Parallel, delayed
from dask import delayed, compute
import scipy.ndimage as ndi
import tifffile as tiff
#import dask

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.config = image_handler.config
        self.dc_file_path = image_handler.dc_file_path
        self.images, self.pos_list = image_handler.images, image_handler.pos_list

    def drift_corr_mypic(self):
        '''
        Running function for drift correction of a whole deconvolved myPIC experiment.

        Parameters
        ----------
        toplevel_folder : Path to folder with all images.
        output_folder : Path to save drift correction results.
        t_index : Int, timepoint to use as template for drift correction. The default is 0.
        ch : Int, channel to use for drift correction. The default is 0.
        filetype : String, type of file. Only works for h5 files for now.
        template : String, template for filenames.

        Returns
        -------
        all_drifts : Table of all drifts. Also saves this as csv in output folder.

        '''
        #List all files in top folder and group according to WXXXX position assuming format
        # *_WXXXX_PXXXX_TXXXX_*.h5

        images = self.images
        pos_list = self.pos_list
        t_slice = self.config['bead_reference_timepoint']
        t_all = range(images.shape[1])
        ch = self.config['bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']

        #Run drift correction based on drift_sv5 for each group and save results in table.
        all_drifts=[]
        for i, group in enumerate(pos_list):
            print(f'Running drift correction for position {group}')
            #t_img = images[i, t_slice, ch]
            #o_imgs = [images[i, o_slice, ch] for o_slice in t_all]
            print('Images loaded.')
            drifts_course = []
            drifts_fine = []
            for t in t_all:
                print('Drift correcting frame', t)
                t_img = np.array(images[i, t_slice, ch])
                o_img = np.array(images[i, t, ch])
                drift_course = ip.drift_corr_course(t_img, o_img, downsample=2)
                drifts_course.append(drift_course)
                drifts_fine.append(ip.drift_corr_multipoint_cc(t_img, 
                                                                o_img,
                                                                drift_course, 
                                                                threshold, 
                                                                min_bead_int, 
                                                                n_points))
            print('Drift correction complete in position.')
            drifts = pd.concat([pd.DataFrame(drifts_course), pd.DataFrame(drifts_fine)], axis = 1)
            drifts['pos_id'] = group
            drifts.index.name = 'frame'
            all_drifts.append(drifts)
            print('Finished drift correction for position ', group)
        
        all_drifts=pd.concat(all_drifts).reset_index()
        
        all_drifts.columns=['frame',
                            'z_px_course',
                            'y_px_course',
                            'x_px_course',
                            'z_px_fine',
                            'y_px_fine',
                            'x_px_fine',
                            'pos_id']
        all_drifts.to_csv(self.dc_file_path)
        print('Drift correction complete.')
        return all_drifts

    def apply_drift_corr_mypic(self):
        '''
        Running function to apply course drifts from a drift table to a set of images,
        so these images can be used for e.g. spot picking.

        Parameters
        ----------
        toplevel_folder : Path to folder with all images.
        output_folder : Path to save drift corrected images.
        dc_file_path : Path to drift correction csv file.
        filetype : String, type of file. Only works for h5 files atm. The default is '.h5'.
        template : String, template for files. The default is 'DE_2'.
        scale : Float, scale parameter to downscale drift corrected images. The default is 0.5.

        Returns
        -------
        None, just saves drift corrected images.

        '''
        downscale = self.config['image_view_downscaling']
        dc_image_folder = self.config['dc_image_folder']
        input_folder = self.config['input_folder']
        output_folder = self.config['output_folder']
        output_prefix = self.config['output_file_prefix']
        filetypes = self.config['image_filetype']
        template = self.config['image_template']

        template_list = filetypes + template

        drifts=pd.read_csv(self.dc_file_path)
        #Group images by position in drift file and iterate per position.
        for pos, pos_group in drifts.groupby('pos_id'):
            pos_img=[]
            print('Running DC for group', pos)
            pos_index = self.pos_list.index(pos)

            for i, row in pos_group.iterrows():
                
                img = self.images[pos_index, i, :, ::downscale, ::downscale, ::downscale].compute()
                #img=resize(img,(img.shape[0], img.shape[1],  img.shape[2]*scale, img.shape[3]*scale))
                #print('Resized image to ', img.shape)

                #Read course drifts from file.
                dz=row['z_px_course']/downscale
                dy=row['y_px_course']/downscale
                dx=row['x_px_course']/downscale
                
                #Apply course drifts using linear (no) interpolation.
                img=ndi.shift(img,(0,dz,dy,dx), order=0)
                print('Applied shift: ',dz,dy,dx)

                #Rescale each channel and convert to 8-bit.            
                for i in range(img.shape[0]):
                    img[i]=img[i]/np.max(img[i])*255
                img=img.astype(np.uint8)
                
                pos_img.append(img)
            
            #Stack images per position together and save drift corrected image as tiff.
            pos_img=np.stack(pos_img, axis=0)
            
            pos_img=np.moveaxis(pos_img,1,2)
            tiff.imsave(dc_image_folder+os.sep+output_prefix+pos+'__dc.tif', pos_img, imagej=True)
            print('Saved image '+dc_image_folder+os.sep+output_prefix+pos+'__dc.tif')

            # In case H5 files want to be used could replace above with this:
            #with h5py.File(output_folder+os.sep+group_ind+'__dc.h5', 'w') as file:
            #    dset=file.create_dataset('Image', data=pos_img)
            #    dset.attrs['element_size_um'] = (0.2, 0.1/scale, 0.1/scale)
            #    dset.attrs['scaling_xy'] = scale
            #    print('Saved file', file)