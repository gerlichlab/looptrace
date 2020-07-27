"""
Created on Thu Apr 23 09:26:44 2020

@author: ellenberg
"""
import os
import yaml
import numpy as np
import pandas as pd
from read_roi import read_roi_zip, read_roi_file
import scipy.ndimage as ndi
import chromatin_tracing_python.image_processing_functions as ip
from chromatin_tracing_python.gaussfit import fitSymmetricGaussian3D, fitSymmetricGaussian3DMLE
from skimage.measure import regionprops_table
import h5py
import tifffile as tiff
import dask
from dask import delayed


class Tracer:
    def __init__(self, config_path, dc_path):
        '''
        Initialize Tracer class with config read in from YAML file.
    '''
        self.config_path = config_path
        self.config = ip.load_config(config_path)
        self.drift_table = pd.read_csv(dc_path)
        self.images, self.pos_list = ip.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        self.images_shape = self.images.shape
        

        print('Loaded images of shape ', self.images_shape)
        print('Found positions ', self.pos_list)

        self.fit_funcs = {'LS': fitSymmetricGaussian3D, 'MLE': fitSymmetricGaussian3DMLE}
        self.fit_func = self.fit_funcs[self.config['fit_func']]

    def reload_config(self):
        self.config = ip.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')

    def rois_from_spots(self):
        '''
        Autodetect ROIs from spot images using a manual threshold defined in config.
        
        Returns
        ---------
        A pandas DataFrame with the bounding boxes and identifyer of the detected ROIs.
        '''

        #Read parameters from config.
        trace_ch = self.config['trace_ch']
        ref_slice = self.config['ref_slice']
        spot_threshold = self.config['spot_threshold']

        #Loop through the imaging positions.
        all_rois = []
        for position in self.pos_list:
            #Read correct image
            print(f'Detecting spots in position {position}.')
            pos_index = self.pos_list.index(position)
            img = self.images[pos_index, ref_slice, trace_ch]

            #Threshold, dilate and label image
            spot_img, num_spots = ndi.label(ndi.binary_dilation(img>spot_threshold, iterations = 5))
            
            #Make a DataFrame with the ROI info
            spot_props=pd.DataFrame(regionprops_table(spot_img, 
                                                properties=('label',
                                                            'bbox',
                                                            'centroid'
                                                            )))
            spot_props['position'] = position
            all_rois.append(spot_props)

            print(f'Found {num_spots} spots.')
            
        #Cleanup and saving of the DataFrame
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'bbox-0':'z_min', 
                                                'bbox-1':'y_min', 
                                                'bbox-2':'x_min', 
                                                'bbox-3':'z_max', 
                                                'bbox-4':'y_max', 
                                                'bbox-5':'x_max',
                                                'index':'roi_id'})
        self.roi_table = output
        self.save_data(rois = self.roi_table)
        return output

    def man_qc_rois(self):
        '''
        Function to visualize detected ROIs on the image data using napari to verify accuracy.
        Can modify ROIs if necessary.
        TODO: Only deleting ROIs implemented so far, moving and adding should be implemented.
        '''
        
        # only visualize on position at the time
        for position in self.pos_list:
            pos_index = self.pos_list.index(position)
            
            # Plot the roi as a napari shape and add to napari viewer with image:
            roi_shapes, roi_props = ip.roi_to_napari_shape(rois, position)
            with napari.gui_qt():
                viewer = napari.view_image(self.images[pos_index], contrast_limits=(0,10000))
                shape_layer = viewer.add_shapes(roi_shapes, face_color = [0]*4, edge_color='red', edge_width=2, properties=roi_props)
            
            # Update the roi_table with any changes made in the viewer
            self.roi_table = update_roi_shapes(shape_layer, self.roi_table, position)
        return self.roi_table


    def slice_for_roi(self, roi, drift_table_row):
        '''
        Calculate the correct slice object based on a given ROI and drift table
        '''

        #Find size of image
        Z, Y, X = self.images_shape[-3:]
        
        #Read values from drift table
        z_drift_course = int(drift_table_row['z_px_course'])
        y_drift_course = int(drift_table_row['y_px_course'])
        x_drift_course = int(drift_table_row['x_px_course'])
        
        #Course drift correct of the ROI: 
        z_min = int(roi['z_min'])-z_drift_course
        z_max = int(roi['z_max'])-z_drift_course
        y_min = int(roi['y_min'])-y_drift_course
        y_max = int(roi['y_max'])-y_drift_course
        x_min = int(roi['x_min'])-x_drift_course
        x_max = int(roi['x_max'])-x_drift_course

        #Handling case of ROI extending beyond image edge after drift correction:
        pad = ((abs(min(0,z_min)),abs(max(0,z_max-Z))),
               (abs(min(0,y_min)),abs(max(0,y_max-Y))),
               (abs(min(0,x_min)),abs(max(0,x_max-X))))

        sz = (max(0,z_min),min(Z,z_max))
        sy = (max(0,y_min),min(Y,y_max))
        sx = (max(0,x_min),min(X,x_max))

        #If drift correction has failed completely this checks if empty ROIs are generated:
        good = True
        if any([a == b for (a,b) in (sz, sy, sx)]):
            good = False

        #Create slice object after above corrections:
        s = (slice(sz[0],sz[1]), 
             slice(sy[0],sy[1]), 
             slice(sx[0],sx[1]))

        return s, pad, good

    def trace_single_roi_frame(self, img):
        '''
        Fit an image, typically with a 3D gaussian function, but any function defined in 
        fit_func will do that takes similar parameters. Initialized fit at brightest point of image.
        '''

        max_ind=list(np.unravel_index(np.argmax(img, axis=None), img.shape))
        fit_results=delayed(self.fit_func)(img,1,max_ind)
        return fit_results

    def tracing_3d(self):
        '''
        Fits 3D gaussian to previously detected ROIs in all timeframes.
       
        Returns
        -------
        res : Pandas dataframe containing trace data
        imgs : Hyperstack image with raw image data of each ROI.
    
        '''
        #Extract parameters from config and predefined roi table.
        num_frames = self.images.shape[1]
        trace_ch = self.config['trace_ch']
        roi_image_size = self.config['roi_image_size']
        roi_table = self.roi_table[self.roi_table['position'].isin(self.pos_list)]

        #Setup loop over each timepoint.
        trace_index = []
        trace_res = []
        all_images = []
        for frame in range(num_frames):
            print(f'Tracing frame {frame}.')
            
            #Setup loop over each ROI.
            frame_result = []
            frame_index = []
            roi_images = []
            for index, roi in roi_table.iterrows():
                #Select matching row from drift table.
                drift_table_row = self.drift_table.loc[(self.drift_table['frame'] == frame) & 
                                                       (self.drift_table['pos_id'] == roi['position'])]
                #Extract position and ROI coordinates, and extract roi image from 
                #slicable image object (typically a dask array)
                pos_index = self.pos_list.index(roi['position'])
                roi_slice, pad, good = self.slice_for_roi(roi, drift_table_row)
                roi_image = np.array(self.images[pos_index, 
                                                frame, 
                                                trace_ch,
                                                roi_slice[0], 
                                                roi_slice[1],
                                                roi_slice[2]])
                                                
                #If microscope drifted, ROI could be outside image. Correct for this:
                if not good:
                    print('Bad image.')
                    roi_image=np.zeros((10,10,10), dtype=np.float32)
                elif pad != ((0,0),(0,0),(0,0)):
                    print('Padding ', pad)
                    try:
                        roi_image = np.pad(roi_image, pad, mode='edge')
                    except ValueError:
                        roi_image = np.zeros((10,10,10), dtype=np.float32)
                
                #Perform 3D gaussian fit
                frame_result.append(self.trace_single_roi_frame(roi_image)[0])
                #Expand the image to a standard size for hyperstack.
                roi_image_exp = delayed(ip.pad_to_shape)(roi_image, roi_image_size)
                #Extract fine drift from drift table and shift image for display.
                dz = float(drift_table_row['z_px_fine'])
                dy = float(drift_table_row['y_px_fine'])
                dx = float(drift_table_row['x_px_fine'])
                roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
                roi_images.append(roi_image_shifted)
                #Add some parameters for tracing table
                frame_index.append([roi.name, frame, roi['position'], roi['roi_id'], dz, dy, dx])
            #Add all the results per timepoint, compute on delayed dask objects.
            trace_res.append(dask.compute(*frame_result))
            trace_index.append(frame_index)
            all_images.append(np.stack(dask.compute(*roi_images)))
        
        #Cleanup of results into dataframe format
        trace_res = pd.DataFrame([i for t in trace_res for i in t], columns=["BG", 
                                                                    "A", 
                                                                    "z_px",
                                                                    "y_px",
                                                                    "x_px",
                                                                    "sigma_z",
                                                                    "sigma_xy"
                                                                    ])
        trace_index = pd.DataFrame([i for t in trace_index for i in t], columns=[
                                                                    "trace_ID",
                                                                    "frame",
                                                                    "position",
                                                                    "roi_ID",
                                                                    "drift_z",
                                                                    "drift_y",
                                                                    "drift_x"])
        traces = pd.concat([trace_index, trace_res], axis=1)

        #Apply fine scale drift to fits, and physcial units.
        traces['z_px']=traces['z_px']+traces['drift_z']
        traces['y_px']=traces['y_px']+traces['drift_y']
        traces['x_px']=traces['x_px']+traces['drift_x']
        traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_px']*self.config['z_nm']
        traces['y']=traces['y_px']*self.config['xy_nm']
        traces['x']=traces['x_px']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces['sigma_xy']=traces['sigma_xy']*self.config['xy_nm']
        traces = traces.sort_values(['trace_ID', 'frame'])
        traces = traces.set_index(['trace_ID'])
        #Make final hyperstack of images, will typically be in TPZYX order.
        all_images = np.stack(all_images)
        self.save_data(traces=traces, imgs=all_images)
        return traces, all_images
    
    def save_data(self, traces=None, imgs=None, rois=None, pwds=None, pairs=None, config=None, suffix=''):
        output_folder=self.config['output_folder']
        output_filename=self.config['output_file_prefix']
        output_file=output_folder+os.sep+output_filename
        
        if traces is not None:
            traces.to_csv(output_file+'traces.csv')
        if pwds is not None:
            np.save(output_file+'pwds.npy',pwds)
        if rois is not None:
            rois.to_csv(output_file+'rois.csv')
        if imgs is not None:
            imgs=np.moveaxis(imgs,0,2)
            tiff.imsave(output_file+'imgs.tif', imgs, imagej=True)
        if pairs is not None:
            pairs.to_csv(output_file+'pairs.csv')
        if config is not None:
            with open(output_file+'config.yaml', 'w') as myfile:
                yaml.safe_dump(config, myfile)
        
        print('Data saved')
        
