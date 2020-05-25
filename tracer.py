# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
import os
import yaml
import numpy as np
import pandas as pd
from gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from read_roi import read_roi_zip, read_roi_file
from skimage.filters import gaussian
from scipy.stats import mode
import scipy.ndimage as ndi
import image_processing_functions as ip
import tracing_functions as tr
import h5py
import tifffile as tiff
from joblib import Parallel, delayed

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
        self.trace_id = 0
        
        image_folder = self.config['image_folder']
        roi_folder = self.config['roi_folder']
        image_template = self.config['image_template']
        roi_template = self.config['roi_template']
        image_list = ip.all_matching_files_in_subfolders(image_folder,image_template)
        roi_list = ip.all_matching_files_in_subfolders(roi_folder,roi_template)
        image_list = ip.match_file_lists(roi_list, image_list)        
        # res is a list of tuples, each with a list of lists for each image/roi set
        res = [self.tracing_3d(image_path, roi_path) for image_path, roi_path in zip(image_list, roi_list)]
        
        #Unpack res into traces and images
        traces, imgs = list(zip(*res))
        
        #Flatten list of traces, imgs and pwds.
        traces = pd.concat(traces)     
        imgs = np.concatenate(imgs)
        
        return traces, imgs

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
        #Read parameters from config.
        min_nuc_size=self.config['min_nuc_size']
        nuc_ch=self.config['nuc_ch']
        nuc_frame=self.config['nuc_frame']
        
        bead_ch = self.config['bead_ch']
        bead_thresh = self.config['bead_threshold']       
        bead_int = self.config['bead_intensity']
        bead_points = self.config['bead_points']
        
        trace_ch=self.config['trace_ch']
        crop_z = self.config['crop_z']
        roi_image_size = self.config['roi_image_size']
        man_qc = self.config['man_qc']
        
        # Read in tif image from with course drift correction, and
        # reorganize from imageJ TZCYX to Python TCZYX format.
        image=ip.read_tif_image(image_path)
        image=np.moveaxis(image, 1, 2)
        print('Loaded image ', image_path)
        
        #Define image name and final image size for stacking.
        image_name=image_path.split('\\')[-1].split('.')[0]
        exp_shape=[image.shape[0]]+roi_image_size
        
        #Detect nuclei and data about nuclei.
        nuc_labels, nuc_props = ip.detect_nuclei(image[nuc_frame,nuc_ch], min_nuc_size)
        print('Nuclei detection completed.')
        #Detect subpixel drift correction from fiducials.
        drift = Parallel(n_jobs=-2)(delayed(ip.drift_corr_multipoint_cc)(image[0],
                                             image[i],
                                             bead_ch,
                                             bead_thresh,
                                             bead_int,
                                             points=bead_points)
                 for i in range(image.shape[0]))
        print('Drift correction completed.')
        #Read ImageJ ROIs from .roi or .zip file.
        if roi_path[-3:] == 'roi':
            rois=read_roi_file(roi_path)
            rois=[rois[k] for k in list(rois)]
        else:
            rois=read_roi_zip(roi_path)
            rois=[rois[k] for k in list(rois)]
            
        #Initialize loop over ROIs.
        num_rois=len(rois)
        all_coords=[]
        all_imgs=[]
        roi_id=0
        for roi in rois:
            #Expand roi to even numbers for consistency.
            roi['height']=roi['height']+roi['height']%2
            roi['width']=roi['width']+roi['width']%2
            
            # Generate cropped image and expand with zeros from ROI and config specification
            crop_pos=(0, max(0,roi['position']['slice']-8), roi['top'], roi['left'])
            crop_size=(image.shape[0], crop_z, roi['height'], roi['width'])
            crop_img=ip.crop_at_pos(image[:,trace_ch,:,:,:], crop_pos, crop_size)
            
            # Find transposition coordinates to transpose fit to 
            # correct place in ROI.
            transp_z=(exp_shape[1]-crop_z)/2
            transp_y=(exp_shape[2]-roi['height'])/2
            transp_x=(exp_shape[3]-roi['width'])/2
            
            #Find out which nucleus the ROI belongs to.
            nuc_roi=ip.crop_at_pos(nuc_labels, crop_pos[1:], crop_size[1:])
            try:
                nuc_id=mode(nuc_roi[nuc_roi>0], axis=None)[0][0]
            except IndexError:
                nuc_id = 0
            
            #Prepare loop over each timepoint/exchange.
            roi_coords=[]
            trace_id=self.trace_id
            
            for t in range(image.shape[0]):
                # Select image and optionally filter, 
                # find max index of DoG of image to initialize LS gaussian.
                img=crop_img[t]
                if self.config['pre_filter']==1:
                    img=ndi.median_filter(img,2)
                dog=gaussian(img,1)-gaussian(img,10)
                max_ind=list(np.unravel_index(np.argmax(dog, axis=None), dog.shape))
                roi_coords.append([trace_id, image_name, roi_id, nuc_id, t]+
                                  [*fitSymmetricGaussian3D(img,1,max_ind)[0]]+
                                  [transp_z, transp_y, transp_x]+
                                  [*drift[t]])
            
            # Add parameters to a list of pd DataFrames and images to list of images.
            all_coords.append(pd.DataFrame(roi_coords, columns=["trace_ID",
                                                                "img_name",
                                                                "roi_ID",
                                                                "cell_ID",
                                                                "frame",
                                                                "BG", 
                                                                "A", 
                                                                "z_px",
                                                                "y_px",
                                                                "x_px",
                                                                "sigma_z",
                                                                "sigma_xy",
                                                                'transp_z',
                                                                'transp_y',
                                                                'transp_x',
                                                                'drift_z',
                                                                'drift_y',
                                                                'drift_x']))
            
            # Pad and drift correct the spot image.
            
            crop_img=ip.pad_to_shape(crop_img, exp_shape)
            crop_img=np.stack([ndi.shift(crop_img[i], s) for 
                               i, s in enumerate(drift)])
            
            all_imgs.append(crop_img)
            roi_id+=1
            self.trace_id+=1
            print('ROI {} of {} finished tracing.'.format(roi_id,num_rois))
        #Apply quality control and calibration metrics to all traces.
        traces=pd.concat(all_coords)
        traces['z_px']=traces['z_px']+traces['transp_z']+traces['drift_z']
        traces['y_px']=traces['y_px']+traces['transp_y']+traces['drift_y']
        traces['x_px']=traces['x_px']+traces['transp_x']+traces['drift_x']
        traces=traces.drop(columns=['transp_z','transp_y','transp_x', 
                                    'drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_px']*self.config['z_nm']
        traces['y']=traces['y_px']*self.config['xy_nm']
        traces['x']=traces['x_px']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces['sigma_xy']=traces['sigma_xy']*self.config['xy_nm']
        traces=traces.set_index(['trace_ID', 'img_name', 'roi_ID', 'cell_ID', 'frame'])
        traces=self.reapply_QC(traces)
        
        #Stack list of trace images to single array.
        imgs=np.stack(all_imgs)
        print('Processed image ', image_path)
        return traces, all_imgs
    
    def tracing_qc(self, row):
        '''
        Function to set QC value of each fit based on 
        settings from config file.
        '''
        
        A_to_BG=self.config['A_to_BG']
        sigma_xy_max=self.config['sigma_xy_max']
        sigma_z_max=self.config['sigma_z_max']
        man_qc=self.config['man_qc']
        if row['A']<(A_to_BG*row['BG']):
            return 0
        elif row['sigma_xy'] > sigma_xy_max or row['sigma_z'] > sigma_z_max:
            return 0
        elif row.name[4] in man_qc:
            return 0
        elif row['x_px']<0 or row['y_px'] < 0 or row['z_px']<0:
            return 0
        elif row['x_px']>100 or row['y_px'] > 100 or row['z_px'] > 100:
            return 0
        else:
            return 1
        
    def group_mean_qc(self, row, groups):
        '''
        Function to set QC value of each row
        based on group calculation, in this case 
        number of nm away from group mean each point can be.
        Preserves original QC, can only change 1 to 0.
        '''
        
        min_groups=groups-1000
        max_groups=groups+1000
        if max_groups.iloc[row.name[0]]['z']>row['z']<min_groups.iloc[row.name[0]]['z']:
            return 0
        if max_groups.iloc[row.name[0]]['y']>row['y']<min_groups.iloc[row.name[0]]['y']:
            return 0
        if max_groups.iloc[row.name[0]]['x']>row['x']<min_groups.iloc[row.name[0]]['x']:
            return 0
        if row['QC'] == 0:
            return 0
        else:
            return 1
        
    def reapply_QC(self,traces):
        traces['QC']=traces.apply(self.tracing_qc,axis=1)
        group_means=traces.query('QC==1').groupby(level=0).mean()
        traces['QC']=traces.apply(self.group_mean_qc, args=(group_means,), axis=1)
        return traces
    
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
        