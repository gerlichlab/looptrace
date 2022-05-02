# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import PySimpleGUI as sg
import numpy as np
import pandas as pd
import os
import itertools
import yaml
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
import dask.array as da
import czifile
from skimage.registration import phase_cross_correlation
from scipy.stats import trim_mean
import scipy.ndimage as ndi
import re
import napari

def main():
    layout = [
        [sg.Text('Choose config file:')],
        [sg.InputText('YAML_config_file', key='-CONFIG_PATH-'), sg.FileBrowse()],
        [sg.Button('Initialize', key='-INIT-'),
        sg.Button('Reload config', key='-RELOAD-'),
        sg.Button('View images for tracing', key='-VIEW_IMAGES-')
        ],
        [sg.Text('_'*50)],
        [sg.Button('Save to zarr', key='-ZARR-'),
        sg.Button('Run drift correction', key='-RUN_DC-'),
        sg.Button('Save to zarr and drift correction', key='-ZARR_DC-')],
        [sg.Text('_'*50)],
        [sg.Text('Choose drift correction file:')],
        [sg.InputText('Drift correction file', key='-DC_PATH-'), sg.FileBrowse()],
        [sg.Combo(['Position'], default_value='Position', key='-DC_POSITION-', auto_size_text=True),
        sg.Button('View DC images', key='-VIEW_DC-')]
        ]

    window = sg.Window('Tracing GUI', layout)

    while True:
        # Wake every 100ms and look for work
        event, values = window.read(timeout=100)
        if event == '-INIT-':
            H = ImageHandler(values['-CONFIG_PATH-'])
            window['-DC_POSITION-'].update(values=H.pos_list, set_to_index=0)
            window['-ROI_POSITION-'].update(values=H.pos_list, set_to_index=0)
            window['-DC_PATH-'].update(H.dc_file_path)
        elif event == '-VIEW_IMAGES-':
            napari_view(H.images, downscale=H.config['image_view_downscaling'])
        elif event == '-RELOAD-':
            H.reload_config()
        elif event == '-ZARR-':
            if os.path.isdir(H.zarr_path):
                sg.popup('ZARR file already exis.')
            else:
                H.images_to_zarr()
                H.save_metadata()
        elif event == '-RUN_DC-':
            D = Drifter(H)
            if os.path.exists(D.dc_file_path):
                dc_exists_choice = sg.popup_yes_no('Existing drift correction file found, recalculate?')
                if dc_exists_choice == 'Yes':
                    D.drift_corr()
            else:
                D.drift_corr()
        elif event == '-ZARR_DC-':
            if os.path.isdir(H.zarr_path):
                sg.popup('ZARR file already exis.')
            else:
                H.images_to_zarr()
                H.save_metadata()
            D = Drifter(H)
            if os.path.exists(D.dc_file_path):
                dc_exists_choice = sg.popup_yes_no('Existing drift correction file found, recalculate?')
                if dc_exists_choice == 'Yes':
                    D.drift_corr()
            else:
                D.drift_corr()

        elif event == '-VIEW_DC-':
            pos_index = H.pos_list.index(values['-DC_POSITION-'])
            if H.dc_images is not None:
                napari_view(H.dc_images[pos_index], downscale=H.config['image_view_downscaling']) 
            else:
                H.set_drift_table(path=values['-DC_PATH-'])
                H.gen_dc_images()
                napari_view(H.dc_images[pos_index], downscale=H.config['image_view_downscaling'])

        elif event in  (None, 'Exit'):
            client.close()
            break
    window.close()

def napari_view(img, points=None, downscale=2, trace_ch=0, ref_slice=0, contrast_limits=(100,1000)):
    with napari.gui_qt():
        if not isinstance(img, list):
            viewer = napari.view_image(img[...,::downscale,::downscale,::downscale], contrast_limits=contrast_limits)
        else:
            viewer = napari.view_image(img[0][...,::downscale,::downscale,::downscale], contrast_limits=None)
            for i in img[1:]:
                viewer.add_image(i[...,::downscale,::downscale,::downscale], contrast_limits=None)
        if points is not None:
            point_layer = viewer.add_points(points/downscale, 
                                                    size=12,
                                                    edge_width=3,
                                                    edge_color='red',
                                                    face_color='transparent',
                                                    n_dimensional=True)
            sel_dim = [ref_slice, trace_ch] + list(points[0,:]/downscale)
            for dim in range(len(sel_dim)):
                viewer.dims.set_current_step(dim, sel_dim[dim])

    if points is not None:
        return point_layer

if __name__ == '__main__':
    try:
        client = Client('127.0.0.1:8787') #Check for existing local dask client
    except  OSError:
        client = Client() #If no local client exists, start a new one.
    
    main()

class ImageHandler:
    def __init__(self, config_path):
        '''
        Initialize ImageHandler class with config read in from YAML file.
        See config file for details on parameters.
        Will try to use zarr file if present.

        Assumes filenames of format, will import anything AICSimageIO can read:
        *_WXXXX_PXXXX_TXXXX_*.ext
        '''
        
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.zarr_path = self.config['input_folder']+os.sep+self.config['output_file_prefix']+'zarr.zarr'
        if os.path.isdir(self.zarr_path):
            self.images = da.from_zarr(self.zarr_path)
            self.pos_list = pd.read_csv(self.zarr_path+'_positions.txt', sep='\n', header=None)[0].to_list()
            print('Images loaded from ZARR file, shape is ', self.images.shape)
            print('Positions found: ', self.pos_list)
        else:
            try:
                self.images, self.pos_list = self.images_to_dask(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])
            except ValueError:
                print('No images found, check configuration.')
        self.images_shape = self.images.shape
        self.dc_file_path = self.config['output_folder']+os.sep+self.config['output_file_prefix']+'drift_correction.csv'
        self.dc_images = None
        self.nucs = None
        self.nuc_masks = None
        self.nuc_class = None
        
        self.nuc_folder = self.config['output_folder']+os.sep+'nucs'

    def load_config(self, config_file):
        '''
        Open config file and return config variable form yaml file.
        '''
        with open(config_file, 'r') as stream:
            try:
                config=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    def images_to_dask(self, folder, template):
        all_files = self.all_matching_files_in_subfolders(folder, template)
        grouped_files, groups = self.group_filelist(all_files, re_phrase='W[0-9]{4}')
        #print(groups, pos_list)
        
        if '.czi' in template:
            sample = self.read_czi_image(all_files[0])
        else:
            raise TypeError('Input filetype not yet implemented.')

        pos_stack=[]
        for g in grouped_files:
            dask_arrays = []
            for fn in g:
                if '.czi' in template:
                    d = dask.delayed(self.read_czi_image)(fn)
                array = da.from_delayed(d, shape=sample.shape, dtype=sample.dtype)
                dask_arrays.append(array)
            pos_stack.append(da.stack(dask_arrays, axis=0))
        x = da.stack(pos_stack, axis=0)
        print('\n Loaded images of shape: ', x.shape)
        print('Found positions ', groups)
        return x, groups

    def all_matching_files_in_subfolders(self, path, template):
        '''
        Generates a sorted list of all files with the template in the 
        filename in directory and subdirectories.
        '''

        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if all([s in file for s in template]):
                    files.append(os.path.join(r, file))
        return sorted(files)

    def group_filelist(self, input_list, re_phrase):
        '''
        Takes a list of strings (typically filepaths) and groups them according
        to a given element given by its position after splitting the string at split_char.
        E.g.for '..._WXXXX_PXXXX_TXXXX.ext' format this will by split_char='_' and element = -3.
        Returns a list of the , and 
        '''
        grouped_list = []
        groups=[]
        for k, g in itertools.groupby(sorted(input_list),
                                    lambda x: re.search(re_phrase, x).group(0)):
            grouped_list.append(list(g))
            groups.append(k)
        return grouped_list, groups

    def reload_config(self):
        self.config = self.load_config(self.config_path)
        print('Config reloaded. Note images are not reloaded.')
    
    def set_drift_table(self, path=None):
        if not path:
            path = self.dc_file_path
        self.drift_table = pd.read_csv(path, index_col=0)

    def read_czi_image(self, image_path):
        '''
        Reads czi files as arrays using czifile package. Returns only CZYX image.
        '''
        with czifile.CziFile(image_path) as czi:
            image=czi.asarray()[0,0,:,0,:,:,:,0]
        return image

    
    def images_to_zarr(self):
        '''
        Function to save images loaded as a 6D PTCZYX dask array into zarr format.
        Will chuck into two last dimensions.
        Also saves a position list of the named positions.
        '''
        pbar = ProgressBar()
        pbar.register()
        zarr_img = da.rechunk(self.images, chunks=(1,1,1,1,-1,-1))
        zarr_img.to_zarr(self.zarr_path, compression='blosc', compression_opts=dict(cname='zstd', clevel=5, shuffle=2))
        pd.DataFrame(self.pos_list).to_csv(self.zarr_path+'_positions.txt', index=None, header=None, sep='\n')
        self.images = da.from_zarr(self.zarr_path)
        print('Images saved as zarr.')
    
    def save_metadata(self):
        '''
        Saves czi metadata from czi input files as part of conversion to zarr.
        '''

        first_path = self.all_matching_files_in_subfolders(self.config['input_folder'], self.config['image_filetype']+self.config['image_template'])[0]
        first_img = czifile.CziFile(first_path)
        out_path = self.config['input_folder']+os.sep+self.config['output_file_prefix']

        meta = first_img.metadata()
        with open(out_path+'metadata.xml', 'w') as file:
            file.writelines(meta)
        
        metadict = first_img.metadata(raw=False)
        with open(out_path+'metadata.yaml', 'w') as file:
            yaml.safe_dump(metadict, file)
        
        print('Metadata saved.')

class Drifter():

    def __init__(self, image_handler):
        '''
        Initialize Drifter class with config read in from YAML file.
        '''
        self.config = image_handler.config
        self.dc_file_path = image_handler.dc_file_path
        self.images, self.pos_list = image_handler.images, image_handler.pos_list

    def drift_corr_course(self, t_img, o_img, downsample=1):
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
        #Calculate course drift
        s = tuple(slice(None, None, downsample) for i in t_img.shape)
        course_drift=phase_cross_correlation(np.array(t_img[s]), np.array(o_img[s]), return_error=False) * downsample
        #Shift image for fine drift correction
        #o_img=ndi.shift(o_img,course_drift,order=0)
        print('Course drift:', course_drift)
        return course_drift.tolist()

    def drift_corr_multipoint_cc(self, t_img, o_img, course_drift, threshold, min_bead_int, n_points=50, upsampling=100):
        '''
        Function for fine scale drift correction. 

        Parameters
        ----------
        t_img : Template image, 2D or 3D ndarray.
        o_img : Offset image, 2D or 3D ndarray.
        threshold : Int, threshold to segment fiducials.
        min_bead_int : Int, minimal value for maxima of fiducials 
        n_points : Int, number of fiducials to use for drift correction. The default is 5.
        upsampling : Int, upsampling grid for subpixel correlation. The default is 100.

        Returns
        -------
        A trimmed mean (default 20% on each side) of the drift for each fiducial.

        '''

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
        try:
            rand_points = t_img_maxima[np.random.choice(t_img_maxima.shape[0], size=n_points), :]
        except ValueError: #If no maxima are found just choose one random point:
            rand_points = [[10,10,10]]
        
        #Initialize array to store shifts for all selected fiducials.
        shifts=np.empty_like(rand_points, dtype=np.float32)
        
        #Calculate fine scale drift for all selected fiducials.
        sub_imgs_t = []
        sub_imgs_o = []
        for i, point in enumerate(rand_points):
            s_t = tuple([slice(ind-8, ind+8) for ind in point])
            s_o = tuple([slice(ind-int(shift)-8, ind-int(shift)+8) for (ind, shift) in zip(point, course_drift)])
            t = t_img[s_t]
            o = o_img[s_o]
            if (t.shape == (16, 16, 16)) and (o.shape == (16,16,16)):
                sub_imgs_t.append(t)
                sub_imgs_o.append(o)
            else:
                img = np.zeros((16, 16, 16))
                img[8,8,8] = 1000
                sub_imgs_t.append(img)
                sub_imgs_o.append(img)

        shifts = dask.compute([dask.delayed(phase_cross_correlation)(t, 
                                            o, 
                                            upsample_factor=upsampling,
                                            return_error=False)
                                for (t,o) in zip(sub_imgs_t, sub_imgs_o)])[0]
        fine_drift = trim_mean(shifts, proportiontocut=0.2, axis=0)
        print('Fine drift:', fine_drift)
        #Return the 60% central mean to avoid outliers.
        return fine_drift#, np.std(shifts, axis=0)

    def drift_corr(self):
        '''
        Running function for drift correction along T-axis of 6D (PTCZYX) images.
        '''

        images = self.images
        pos_list = self.pos_list
        t_slice = self.config['bead_reference_frame']
        t_all = range(images.shape[1])
        ch = self.config['bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_points= self.config['bead_points']

        #Run drift correction for each position and save results in table.
        all_drifts=[]
        for i, group in enumerate(pos_list):
            print(f'Running drift correction for position {group}')
            drifts_course = []
            drifts_fine = []
            for t in t_all:
                print('Drift correcting frame', t)
                t_img = np.array(images[i, t_slice, ch])
                o_img = np.array(images[i, t, ch])
                drift_course = self.drift_corr_course(t_img, o_img, downsample=2)
                drifts_course.append(drift_course)
                drifts_fine.append(self.drift_corr_multipoint_cc(t_img, 
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