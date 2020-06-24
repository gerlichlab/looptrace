"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
import os
import yaml
import numpy as np
import pandas as pd
from chromatin_tracing_python.gaussfit import fitSymmetricGaussian3D
from read_roi import read_roi_zip, read_roi_file
#from skimage.filters import gaussian
#from scipy.stats import mode
import scipy.ndimage as ndi
from scipy import stats
import chromatin_tracing_python.image_processing_functions as ip
#import tracing_functions as tr
#import h5py
import tifffile as tiff
#from joblib import Parallel, delayed

class Tracer_decon:
    def __init__(self, config_path):
        '''
        Initialize Tracer class with config read in from YAML file.
        '''
        self.config = ip.load_config(config_path)
        self.drift_table = pd.read_csv(self.config['drift_table'])
        self.trace_id = 0
    
    def tracing_multi_decon(self):
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
        image_list = [ip.match_file_lists_decon([roi], image_list) for
                      roi in roi_list]
        
        # res is a list of tuples, each with a list of lists for each image/roi set
        res = [self.tracing_3d_decon(image_paths, roi_path) for 
               image_paths, roi_path in zip(image_list, roi_list)]
        
        #Unpack res into traces and images
        traces, imgs = list(zip(*res))
        
        #Flatten list of traces, imgs and pwds.
        traces = pd.concat(traces)     
        imgs = np.concatenate(imgs)
        
        return traces, imgs

    def tracing_3d_decon(self, image_paths, roi_path):
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
                
        trace_ch=self.config['trace_ch']
        crop_z = self.config['crop_z']
        roi_image_size = self.config['roi_image_size']
        roi_scale = self.config['dc_scale_factor']
        drift_table = self.drift_table
        image_name=roi_path.split('\\')[-1].split('__')[0]
        print('Tracing image ', image_name)

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
                        
            # Find transposition coordinates to transpose fit to 
            # correct place in ROI.
            transp_z=(roi_image_size[0]-crop_z)/2
            transp_y=(roi_image_size[1]-roi['height']//roi_scale)/2
            transp_x=(roi_image_size[2]-roi['width']//roi_scale)/2
            
            #Prepare loop over each timepoint/exchange.
            roi_coords=[]
            trace_id=self.trace_id
            t_img=[]
            for image_path in image_paths:
                # Determine image name.
                image_name=image_path.split('\\')[-1]
                
                #Read drift from drift table.
                drift_table_row = drift_table.loc[drift_table['filename'] == image_name]
                t = int(drift_table_row['frame'])
                z_drift_course = int(drift_table_row['z_px_course'])
                y_drift_course = int(drift_table_row['y_px_course'])
                x_drift_course = int(drift_table_row['x_px_course'])
                z_drift_fine = float(drift_table_row['z_px_fine'])
                y_drift_fine = float(drift_table_row['y_px_fine'])
                x_drift_fine = float(drift_table_row['x_px_fine'])
                
                #Determine ROI box from ImageJ ROI, set to 0-index, account for drift.
                z_min = int(roi['position']['slice']-1-crop_z//2-z_drift_course)
                z_max = int(roi['position']['slice']-1-z_drift_course+crop_z//2)
                y_min = int((roi['top']-1)/roi_scale-y_drift_course)
                y_max = int(y_min + roi['height']/roi_scale)
                x_min = int((roi['left']-1)/roi_scale-x_drift_course)
                x_max = int(x_min + roi['width']/roi_scale)
                
                sz=slice(max(0,z_min), z_max)
                sy=slice(y_min, y_max)
                sx=slice(x_min, x_max)
                print('Tracing at ', sz, sy, sx)
                img=ip.image_from_svih5(image_path, ch=trace_ch, index=(sz,sy,sx))
                
                if z_min < 0:
                    print('Found low image, padding from shape ', img.shape)
                    img=np.pad(img,((-z_min,0),(0,0),(0,0)), mode='edge')
                    print('Padded image to', img.shape)
                
                max_ind=list(np.unravel_index(np.argmax(img, axis=None), img.shape))
                roi_coords.append([trace_id, image_name, roi_id, t]+
                                  [*fitSymmetricGaussian3D(img,1,max_ind)[0]]+
                                  [transp_z, transp_y, transp_x]+
                                  [z_drift_fine, y_drift_fine, x_drift_fine])
                
                img = ip.pad_to_shape(img, roi_image_size)

                img = ndi.shift(img, (z_drift_fine, y_drift_fine, x_drift_fine))
                t_img.append(img)
            # Add parameters to a list of pd DataFrames and images to list of images.
            all_coords.append(pd.DataFrame(roi_coords, columns=["trace_ID",
                                                                "img_name",
                                                                "roi_ID",
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
            
           
            all_imgs.append(np.stack(t_img, axis=0))
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
        traces=traces.set_index(['trace_ID'])
        
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
        elif row['frame'] in man_qc:
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
        #print(groups.iloc[row.name]['z'])
        #min_groups=groups-self.config['max_dist_qc']
        #max_groups=groups+self.config['max_dist_qc']
        max_dist = self.config['max_dist_qc']
        z_mean=groups.iloc[row.name]['z']
        y_mean=groups.iloc[row.name]['y']
        x_mean=groups.iloc[row.name]['x']

        if row['z']>(z_mean+max_dist) or row['z']<(z_mean-max_dist):
            return 0
        if row['y']>(y_mean+max_dist) or row['y']<(y_mean-max_dist):
            return 0
        if row['x']>(x_mean+max_dist) or row['x']<(x_mean-max_dist):
            return 0
        if row['QC'] == 0:
            return 0
        else:
            return 1
        
    def reapply_QC(self,traces):
        traces['QC']=traces.apply(self.tracing_qc,axis=1)
        group_means=traces[traces['frame']==self.config['search_frame']]
        print(group_means)
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
            imgs=np.moveaxis(imgs,0,2)
            tiff.imsave(output_file+'_imgs'+suffix+'.tiff', imgs, imagej=True)
        if pairs is not None:
            pairs.to_hdf(output_file+'_pairs'+suffix+'.h5', key='pairs', mode='w')
        if config is not None:
            with open(output_file+'_config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')
        
