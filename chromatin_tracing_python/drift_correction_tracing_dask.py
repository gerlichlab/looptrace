# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:30:43 2020

@author: ellenberg
"""
import h5py
import os
import itertools
import scipy.ndimage as ndi
import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from skimage.transform import resize
from scipy.stats import trim_mean
import tifffile as tiff
from chromatin_tracing_python import image_processing_functions as ip
from joblib import Parallel, delayed
#import dask

class Drifter():

    def __init__(self, config_path):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.config = ip.load_config(config_path)
        self.dc_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv'

    def drift_svih5(self, t_img, o_img):
        '''
        Calculates course and fine 
        drift between two svih5 images by phase cross correlation.

        Parameters
        ----------
        t_path : Path to template image in svih5 format.
        o_path : Path to offset image in svih5 format.
        ch : Which channel to use for drift correction.

        Returns
        -------
        A list of zyx course drifts and fine drifts (compared to course)

        '''        
        print('Starting DC')
        #Calculate course drift
        course_drift=phase_cross_correlation(t_img, o_img, return_error=False)
        print('Course DC done')
        #Shift image for fine drift correction
        o_img=ndi.shift(o_img,course_drift,order=0)
        print('Shift done')
        #Calculate fine drift
        fine_drift, fine_drift_std = self.drift_corr_multipoint_cc(t_img, o_img)
        print('Fine DC done')
        return [*course_drift, *fine_drift, *fine_drift_std]

    def drift_corr_multipoint_cc(self, t_img, o_img, upsampling=100):
        '''
        Function for fine scale drift correction. 

        Parameters
        ----------
        t_img : Template image, 2D or 3D ndarray.
        o_img : Offset image, 2D or 3D ndarray.
        threshold : Int, threshold to segment fiducials.
        min_bead_int : Int, minimal value for maxima of fiducials 
        points : Int, number of fiducials to use for drift correction. The default is 5.
        upsampling : Int, upsampling grid for subpixel correlation. The default is 100.

        Returns
        -------
        A trimmed mean (default 20% on each side) of the drift for each fiducial.

        '''
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        points = self.config['bead_points']

        #Label fiducial candidates and find maxima.
        t_img_label,num_labels=ndi.label(t_img>threshold)
        t_img_maxima=np.array(ndi.measurements.maximum_position(t_img, 
                                                    labels=t_img_label, 
                                                    index=range(num_labels)))
        
        #Filter maxima so not too close to edge and bright enough.
        t_img_maxima=np.array([m for m in t_img_maxima 
                                if min_bead_int<t_img[tuple(m)]
                                and all(m>8)])
        
        #Select random fiducial candidates. Seeded for reproducibility.
        np.random.seed(1)
        rand_points = t_img_maxima[np.random.choice(t_img_maxima.shape[0], size=points), :]
        
        #Initialize array to store shifts for all selected fiducials.
        shifts=np.empty_like(rand_points, dtype=np.float32)
        
        #Calculate fine scale drift for all selected fiducials.
        for i, point in enumerate(rand_points):
            s=tuple([slice(ind-8, ind+8) for ind in point])
            shift = phase_cross_correlation(t_img[s], 
                                        o_img[s], 
                                        upsample_factor=upsampling,
                                        return_error=False)
            shifts[i] = shift
            
        #Return the 60% central mean to avoid outliers.
        return trim_mean(shifts, proportiontocut=0.2, axis=0), np.std(shifts, axis=0)

        
    def drift_corr_mypic_h5(self):
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
        images, pos_list = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        #List all files in top folder and group according to WXXXX position assuming format
        # *_WXXXX_PXXXX_TXXXX_*.h5
        t_slice = self.config['bead_reference_timepoint']
        t_all = range(images.shape[1])
        ch = self.config['bead_ch']
        
        #Run drift correction based on drift_sv5 for each group and save results in table.
        all_drifts=[]
        for i, group in enumerate(pos_list):
            print(f'Running drift correction for position {group}')
            #drifts_delayed=Parallel(n_jobs=-2)(delayed(self.drift_svih5)(i, template_slice, offset_slice) for 
            #                                                offset_slice in t_all) 
            drifts_delayed = []
            t_img=np.array(images[i, t_slice, ch])
            print(type(t_img))
            o_imgs = [np.array(images[i, o_slice, ch]) for o_slice in t_all]
            print(type(o_imgs[0]))
            drifts_delayed = Parallel(n_jobs=-2)(delayed(self.drift_svih5)(t_img, o_img) for o_img in o_imgs)
            '''
            for offset_slice in t_all:
                #Load single channel images from paths
                o_img=np.array(self.images[i, offset_slice, ch])
                print('Offset image loaded: ', o_img.shape, o_img.dtype)
                drifts_delayed.append(self.drift_svih5(t_img, o_img))
            '''
            drifts=pd.DataFrame(drifts_delayed)
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
                            'z_px_fine_std',
                            'y_px_fine_std',
                            'x_px_fine_std',
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
        scale = self.config['dc_image_scaling']
        dc_image_folder = self.config['dc_image_folder']
        input_folder = self.config['input_folder']
        output_folder = self.config['output_folder']
        output_prefix = self.config['output_file_prefix']
        filetypes = self.config['image_filetype']
        template = self.config['image_template']

        template_list = filetypes + template
        all_files = ip.all_matching_files_in_subfolders(input_folder, 
                                                        template_list)

        drifts=pd.read_csv(self.dc_file_path)
        
        #Group images by position in drift file and iterate per position.
        for group_ind, group in drifts.groupby('pos_id'):
            pos_img=[]
            print('Running DC for group', group_ind)
            for i, row in group.iterrows():
                
                #Load and rescale image.
                img=ip.image_from_svih5(all_files[i])
                print('Loaded image ', all_files[i])
                
                #img=np.moveaxis(img,0,1)
                #tiff.imsave(output_folder+os.sep+group_ind+'__dc_test.tiff', img, imagej=True)
                #return
                img=resize(img,(img.shape[0], 
                                img.shape[1], 
                                img.shape[2]*scale, 
                                img.shape[3]*scale))
                print('Resized image to ', img.shape)
                #Read course drifts from file.
                dz=row['z_px_course']
                dy=row['y_px_course']*scale
                dx=row['x_px_course']*scale
                
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
            tiff.imsave(dc_image_folder+os.sep+output_prefix+group_ind+'__dc.tif', pos_img, imagej=True)
            print('Saved image '+output_prefix+group_ind+'__dc.tif')

            # In case H5 files want to be used could replace above with this:
            #with h5py.File(output_folder+os.sep+group_ind+'__dc.h5', 'w') as file:
            #    dset=file.create_dataset('Image', data=pos_img)
            #    dset.attrs['element_size_um'] = (0.2, 0.1/scale, 0.1/scale)
            #    dset.attrs['scaling_xy'] = scale
            #    print('Saved file', file)