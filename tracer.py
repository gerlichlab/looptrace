# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
import os
import yaml
import numpy as np
import pandas as pd
from image_processing_functions.gaussfit import fitSymmetricGaussian3D,fitSymmetricGaussian3DMLE
from read_roi import read_roi_zip, read_roi_file
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
from scipy.stats import mode
import scipy.ndimage as ndi
from image_processing_functions import image_processing_functions as ip
from image_processing_functions import tracing_functions as tr
import h5py
import tifffile as tiff

class Tracer:
    def __init__(self, config_path):
        '''
        Initialize Tracer class with config read in from YAML file.
        '''
    
        self.config = ip.load_config(config_path)
        self.config_path = config_path
        self.trace_id = 0
    
    def tracing_multi(self):
        '''
        Perfors maxima tracing of all images at matching ROIs in folders.
    
        Parameters
        ----------
        image_folder : Folder path to image folder.
        roi_folder : Folder path to ROI folder.
    
        Returns
        -------
        traces : List of pandas dataframes with trace data.
        imgs : Single hyperstack with all trace images.
        '''
        
        #Find lists of images and ROIs, and match them so that only images
        #with matching ROIs are included.
        image_folder = self.config['image_folder']
        roi_folder = self.config['roi_folder']
        match_idx = self.config['match_idx']
        image_template = ['.tiff']
        roi_template = ['.zip']
        image_list = ip.all_matching_files_in_subfolders(image_folder,image_template)
        roi_list = ip.all_matching_files_in_subfolders(image_folder,roi_template)
        image_list = ip.match_file_lists(roi_list, image_list, match_idx)        
        
        # res is a list of tuples, each with a list of lists for each image/roi set
        res = [self.tracing_3d(image_path, roi_path) for image_path, roi_path in zip(image_list, roi_list)]
        
        #Unpack res into traces and images
        traces, imgs, pwds = list(zip(*res))
        
        #Flatten list of traces, imgs and pwds.
        traces = pd.concat(traces)     
        imgs = np.concatenate(imgs)
        pwds = np.concatenate(pwds)
        
        self.save_data(traces=traces,imgs=imgs,pwds=pwds,config=self.config)
        return traces, imgs, pwds
    
    def tracing_3d(self, image_path, roi_path):
        '''
        Fits gaussian maxima over time in list of ROIs in a given image.
    
        Parameters
        ----------
        image_path : File path to image (expects a imageJ format tiff hyperstack)
        roi_path : Path to ROIs file saved from imageJ ROI manager.
    
        Returns
        -------
        res : List of pandas dataframes containing trace data for each ROI
        imgs : Hyperstack image with raw image data of each ROI.
    
        '''
        min_nuc_size=self.config['min_nuc_size']
        nuc_ch=self.config['nuc_ch']
        nuc_frame=self.config['nuc_frame']
        trace_ch=self.config['trace_ch']
        roi_image_size=self.config['roi_image_size']
        man_qc=self.config['man_qc']
        # Read in images and ROIs.
        image=ip.read_tif_image(image_path)
        print('Loaded image ', image_path)
        
        image_name=image_path.split('\\')[-1].split('.')[0]
        
        nuc_labels, nuc_props = ip.detect_nuclei(image[nuc_frame,:,nuc_ch,:,:], min_nuc_size)
        
        
        if roi_path[-3:] == 'roi':
            rois=read_roi_file(roi_path)
            rois=[rois[k] for k in list(rois)]
        else:
            rois=read_roi_zip(roi_path)
            rois=[rois[k] for k in list(rois)]
        all_coords=[]
        all_imgs=[]
        roi_id=0
        for roi in rois:
            # Generate cropped image from ROI specification
            crop_pos=(0, roi['position']['slice']-8, roi['top'], roi['left'])
            size=(image.shape[0], 16, roi['height'], roi['width'])
            crop_img=ip.crop_at_pos(image[:,:,trace_ch,:,:], crop_pos, size)
            nuc_roi=ip.crop_at_pos(nuc_labels, crop_pos[1:], size[1:])
            try:
                nuc_id=mode(nuc_roi[nuc_roi>0], axis=None)[0][0]
            except IndexError:
                nuc_id = 0
            
            roi_coords=[]
            trace_id=self.trace_id
             # LS fit maxima in each timepoint of ROI by 3d gaussian.
            for t in range(image.shape[0]):
                img=crop_img[t]
                if self.config['pre_filter']==1:
                    img=ndi.median_filter(img,2)
                dog=gaussian(img,1)-gaussian(img,10)
                max_ind=list(np.unravel_index(np.argmax(dog, axis=None), dog.shape))
                roi_coords.append([trace_id, image_name, roi_id, nuc_id, t]+[*fitSymmetricGaussian3D(img,3,max_ind)[0]])
            
            all_coords.append(pd.DataFrame(roi_coords, columns=["trace_ID",
                                                                "img_name",
                                                                "roi_ID",
                                                                "cell_ID",
                                                                "frame",
                                                                "BG", 
                                                                "A", 
                                                                "z",
                                                                "y",
                                                                "x",
                                                                "sigma_w",
                                                                "sigma_z"]))
            all_imgs.append(crop_img)
            roi_id+=1
            self.trace_id+=1
        
        #Apply quality control and calibration metrics to all traces.
        
        traces=pd.concat(all_coords)
        traces['z']=traces['z']*self.config['z_nm']
        traces['y']=traces['y']*self.config['xy_nm']
        traces['x']=traces['x']*self.config['xy_nm']
        traces['sigma_w']=traces['sigma_w']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces=traces.set_index(['trace_ID', 'img_name', 'roi_ID', 'cell_ID', 'frame'])
        traces['QC']=traces.apply(self.tracing_qc,axis=1)

        # Calculate pairwise distances for all traces.
        points = [tr.points_from_df_nan(df)[0] for _, df in traces.groupby(level=0)]
        pwds = [cdist(p, p) for p in points]
        pwds = np.stack(pwds)
        
        # Pad ROI images to standard shape to allow hyperstack generation.
        exp_shape=[image.shape[0]]+roi_image_size
        all_imgs=[ip.pad_to_shape(img, exp_shape) for img in all_imgs]
        imgs=np.stack(all_imgs)
        print('Processed image ', image_path)
        return traces, imgs, pwds
    
    def tracing_qc(self, row):
        A_to_BG=self.config['A_to_BG']
        sigma_w_max=self.config['sigma_w_max']
        sigma_z_max=self.config['sigma_z_max']
        man_qc=self.config['man_qc']
                
        if row['BG']>(A_to_BG*row['A']):
            return 0
        elif row['sigma_w'] > sigma_w_max or row['sigma_z'] > sigma_z_max:
            return 0
        elif row.name[4] in man_qc:
            return 0
        else:
            return 1
        
    def reapply_QC(self,traces, pwds):
        traces['QC']=traces.apply(self.tracing_qc,axis=1)
        # Recalculate pairwise distances for all traces.
        points = [tr.points_from_df_nan(df)[0] for _, df in traces.groupby(level=0)]
        pwds = [cdist(p, p) for p in points]
        pwds = np.stack(pwds)
        return traces, pwds
    
    def save_data(self, traces=None, imgs=None, pwds=None, pairs=None, config=None, suffix=''):
        output_folder=self.config['output_folder']
        output_filename=self.config['output_name']
        output_file=output_folder+os.sep+output_filename
        
        if traces is not None:
            traces.to_hdf(output_file+'_traces'+suffix+'.h5', key='traces', mode='w')
        if pwds is not None:
            np.save(output_file+'_pwds'+suffix+'.npy',pwds)
        if imgs is not None:
            imgs=np.moveaxis(imgs,1,2)
            tiff.imsave(output_file+'_imgs'+suffix+'.tiff', imgs, imagej=True)
        if pairs is not None:
            pairs.to_hdf(output_file+'_pairs'+suffix+'.h5', key='pairs', mode='w')
        if config is not None:
            with open(output_file+'_config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')
        