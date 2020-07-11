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
from chromatin_tracing_python.gaussfit import fitSymmetricGaussian3D
from skimage.measure import regionprops_table
import h5py
import tifffile as tiff
import dask
from dask import delayed
from dask.distributed import Client

class Tracer:
    def __init__(self, config_path):
        '''
        Initialize Tracer class with config read in from YAML file.
    '''
        self.config_path = config_path
        self.config = ip.load_config(config_path)
        self.drift_table = pd.read_csv(self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv')
        self.images, self.pos_list = ip.svih5_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
        self.images_shape = self.images.shape
        self.client = Client()

    '''
    def extract_nuc_ilastik(self):
        spot_labels, num_spots = ndi.label(ndi.binary_dilation(img*(img == 3)))
        spot_props=pd.DataFrame(regionprops_table(spot_labels, img, 
                                                properties=('label',
                                                            'bbox',
                                                            'centroid'
                                                            )))
        nuc_labels, num_nucs = ndi.label(ndi.binary_dilation(img*(img == 2), iterations = 5))
        nuc_props=pd.DataFrame(regionprops_table(nuc_labels, img, 
                                                properties=('label',
                                                            'bbox',
                                                            'centroid'
                                                            )))

        print(f'Found {num_spots} spots and {num_nucs} nuclei.')
        return spot_props, nuc_props
    '''
    def reload_config(self):
        self.config = ip.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')

    def rois_from_spots(self):
        trace_ch = self.config['trace_ch']
        ref_slice = self.config['ref_slice']
        spot_threshold = self.config['spot_threshold']
        all_rois = []

        for position in self.pos_list:
            print(f'Detecting spots in position {position}.')
            pos_index = self.pos_list.index(position)
            img = self.images[pos_index, ref_slice, trace_ch, 0]
            spot_img, num_spots = ndi.label(ndi.binary_dilation(img>spot_threshold, iterations = 5))
            spot_props=pd.DataFrame(regionprops_table(spot_img, 
                                                properties=('label',
                                                            'bbox',
                                                            'centroid'
                                                            )))
            print(f'Found {num_spots} spots.')
            spot_props['position'] = position
            all_rois.append(spot_props)
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'bbox-0':'z_min', 
                                                'bbox-1':'y_min', 
                                                'bbox-2':'x_min', 
                                                'bbox-3':'z_max', 
                                                'bbox-4':'y_max', 
                                                'bbox-5':'x_max',
                                                'index':'roi_id'})
        #spot_props[['bbox-0','bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5']].to_numpy().tolist()
        self.roi_table = output
        self.save_data(rois = self.roi_table)
        return output

    def man_qc_rois(self):
        for position in self.pos_list:
            pos_index = self.pos_list.index(position)
            roi_shapes, roi_props = ip.roi_to_napari_shape(rois, position)
            with napari.gui_qt():
                viewer = napari.view_image(self.images[pos_index], contrast_limits=(0,10000))
                shape_layer = viewer.add_shapes(roi_shapes, face_color = [0]*4, edge_color='red', edge_width=2, properties=roi_props)
            self.roi_table = update_roi_shapes(shape_layer, self.roi_table, position)
        return self.roi_table


    def slice_for_roi(self, roi, drift_table_row):
        Z, Y, X = self.images_shape[-3:]

        z_drift_course = int(drift_table_row['z_px_course'])
        y_drift_course = int(drift_table_row['y_px_course'])
        x_drift_course = int(drift_table_row['x_px_course'])
        
        z_min = int(roi['z_min'])-z_drift_course
        z_max = int(roi['z_max'])-z_drift_course
        y_min = int(roi['y_min'])-y_drift_course
        y_max = int(roi['y_max'])-y_drift_course
        x_min = int(roi['x_min'])-x_drift_course
        x_max = int(roi['x_max'])-x_drift_course

        pad = ((abs(min(0,z_min)),abs(max(0,z_max-Z))),
               (abs(min(0,y_min)),abs(max(0,y_max-Y))),
               (abs(min(0,x_min)),abs(max(0,x_max-X))))

        sz = (max(0,z_min),min(Z,z_max))
        sy = (max(0,y_min),min(Y,y_max))
        sx = (max(0,x_min),min(X,x_max))

        good = True
        if any([a == b for (a,b) in (sz, sy, sx)]):
            good = False

        s = (slice(sz[0],sz[1]), 
             slice(sy[0],sy[1]), 
             slice(sx[0],sx[1]))

        return s, pad, good

    def trace_single_roi_frame(self, img):
        max_ind=list(np.unravel_index(np.argmax(img, axis=None), img.shape))
        fit_results=delayed(fitSymmetricGaussian3D)(img,1,max_ind)
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
                                                0, 
                                                roi_slice[0], 
                                                roi_slice[1],
                                                roi_slice[2]])
                #If microscope drifted, ROI could be outside image. Correct for this:
                #TODO: Implement for all possibilities.
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
                frame_index.append([index, frame, roi['position'], roi['roi_id'], dz, dy, dx])
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
        #group_means=traces[traces['frame']==self.config['search_frame']]
        #traces['QC']=traces.apply(self.group_mean_qc, args=(group_means,), axis=1)
        return traces
    
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
        
