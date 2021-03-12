# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import PySimpleGUI as sg
import os
import threading
import numpy as np

from pychrtrace import Tracer
from pychrtrace import Drifter
from pychrtrace import ImageHandler
from pychrtrace import SpotPicker
from pychrtrace import NucDetector
import pychrtrace.image_processing_functions as ip
from dask.distributed import Client
import logging
import napari

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
sg.theme('DarkTeal6')  # please make your windows colorful

def main():
    layout = [
        [sg.Text('Choose config file:')],
        [sg.InputText('YAML_config_file', key='-CONFIG_PATH-'), sg.FileBrowse()],
        [sg.Button('Initialize', key='-INIT-'),
        sg.Button('Save to zarr', key='-ZARR-'),
        sg.Button('View images for tracing', key='-VIEW_IMAGES-'),
        sg.Button('Reload config', key='-RELOAD-')],
        [sg.Text('_'*50)],
        [sg.Button('Run drift correction', key='-RUN_DC-')],
        [sg.Text('_'*50)],
        [sg.Text('Choose drift correction file:')],
        [sg.InputText('Drift correction file', key='-DC_PATH-'), sg.FileBrowse()],
        [sg.Combo(['Position'], default_value='Position', key='-DC_POSITION-', auto_size_text=True),
        sg.Button('View DC images', key='-VIEW_DC-')],
        [sg.Text('_'*50)],
        [sg.Button('Detect nuclei', key='-DET_NUC-'),
        sg.Button('Classify nuclei', key='-CLASS_NUC-'),
        sg.Button('View nuclei', key='-VIEW_NUC-')],
        [sg.Text('_'*50)],
        [sg.Radio('Use IJ ROIs', "ROI_SEL", key='-IJ_ROI-'),
        sg.InputText('Folder with ImageJ ROIs', key='-IJ_ROI_PATH-'), sg.FolderBrowse()],
        [sg.Radio('Use ROI file', "ROI_SEL", key='-EXISTING_ROI-'),
        sg.InputText('ROI file', key='-ROI_FILE_PATH-'), sg.FileBrowse()],
        [sg.Button('Load existing ROIs', key='-LOAD_ROI-')],
        [sg.Text('_'*50)],
        [sg.Text('Generate new ROIs:')],
        [sg.Button('Preview ROIs', key='-PREVIEW_ROI-'),
        sg.Button('Detect new ROIs', key='-DET_ROI-'),
        sg.Button('Refilter ROIs', key='-SPOT_IN_NUC-')],
        [sg.Text('_'*50)],
        [sg.Combo(['Position'], default_value='Choose position', key='-ROI_POSITION-', auto_size_text=True),
        sg.Checkbox('Show DC?', key='-DC_IMAGE_ROIS-'), 
        sg.Button('View ROIs', key='-VIEW_ROI-')],
        [sg.Text('_'*50)],
        [sg.Button('Run tracing', key='-RUN_TRACING-')]
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
            ip.napari_view(H.images, downscale=H.config['image_view_downscaling'])
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
        elif event == '-VIEW_DC-':
            pos_index = H.pos_list.index(values['-DC_POSITION-'])
            if H.dc_images is not None:
                ip.napari_view(H.dc_images[pos_index], downscale=H.config['image_view_downscaling']) 
            else:
                H.set_drift_table(path=values['-DC_PATH-'])
                H.gen_dc_images()
                ip.napari_view(H.dc_images[pos_index], downscale=H.config['image_view_downscaling'])
        elif event == '-DET_NUC-':
            N = NucDetector(H)
            N.segment_nuclei()
        elif event == '-CLASS_NUC-':
            try:
                N.classify_nuclei()
            except NameError:
                N = NucDetector(H)
                N.classify_nuclei()
        elif event == '-VIEW_NUC-':
            try:
                N
            except NameError:
                N = NucDetector(H)
            with napari.gui_qt():
                if H.nucs:
                    nuc_imgs = np.stack(H.nucs)
                    viewer = napari.view_image(nuc_imgs.copy())
                if H.nuc_masks:
                    nuc_labels = np.stack(H.nuc_masks)
                    labels_layer = viewer.add_labels(nuc_labels.copy())
                if H.nuc_class:
                    nuc_class = np.stack(H.nuc_class)
                    classes_layer = viewer.add_labels(nuc_class.copy())
            try:             
                if not np.allclose(labels_layer.data, nuc_labels):
                    print('Labels changed, resaving.')
                    nuc_masks = [labels_layer.data[i] for i in range(labels_layer.data.shape[0])] 
                    H.nuc_masks = nuc_masks
                    H.save_nucs(img_type='mask')
                    N.mask_to_binary(nuc_masks)
    
                if not np.allclose(classes_layer.data, nuc_class):
                    print('Class labels changed, resaving.')
                    nuc_class = [classes_layer.data[i] for i in range(classes_layer.data.shape[0])]
                    H.nuc_class = nuc_class
                    H.save_nucs(img_type='class')
            except UnboundLocalError: #In case labels or classes do not exist.
                continue

        elif event == '-PREVIEW_ROI-':
            print('Previewing spot detection.')
            S = SpotPicker(H)
            spots, img, filt_img = S.rois_from_spots(preview_pos=values['-ROI_POSITION-'])
            roi_points, roi_props = ip.roi_to_napari_points(spots, values['-ROI_POSITION-'])
            ip.napari_view([img, filt_img], points=roi_points, downscale=1)
        
        elif event == '-DET_ROI-':
            print('Detecting spots in all positions.')
            S = SpotPicker(H)
            S.rois_from_spots(filter_nucs=H.config['spot_in_nuc'])

        elif event == '-SPOT_IN_NUC-':
            if not H.nuc_masks:
                H.load_nucs()
            H.roi_table = ip.filter_rois_in_nucs(H.roi_table, H.nuc_masks, H.pos_list, new_col='nuc_label')
            if H.nuc_class:
                H.roi_table = ip.filter_rois_in_nucs(H.roi_table, H.nuc_class, H.pos_list, new_col='nuc_class')
            H.save_data(rois=H.roi_table)

        elif event == '-LOAD_ROI-':
            if values['-IJ_ROI-']:
                H.roi_table = ip.rois_from_imagej(values['-IJ_ROI_PATH-'], '.zip', 16 , float(T.config['dc_image_scaling']))
                H.save_data(rois=H.roi_table)
            elif values['-EXISTING_ROI-']:
                H.roi_table = ip.rois_from_csv(values['-ROI_FILE_PATH-'])

        elif event == '-VIEW_ROI-':
            if values['-DC_IMAGE_ROIS-']:
                if H.dc_images is not None:
                    print('Found DC_images.')
                    img = H.dc_images
                else:
                    print('Generating DC images.')
                    H.set_drift_table(path=values['-DC_PATH-'])
                    H.gen_dc_images()
                    img = H.dc_images
            else:
                img = H.images
            position = values['-ROI_POSITION-']
            print('Checking ROIs in position ', position)
            pos_index = H.pos_list.index(position)
            #roi_shapes, roi_props = ip.roi_to_napari_shape(T.roi_table, position = position)
            roi_points, roi_props = ip.roi_to_napari_points(H.roi_table, position = position)
            if roi_points.size == 0:
                roi_points = np.empty((0, 3))
            point_layer = ip.napari_view(img[pos_index], 
                                            points = roi_points,
                                            downscale= H.config['image_view_downscaling'],
                                            trace_ch = H.config['trace_ch'],
                                            ref_slice= H.config['bead_reference_frame'],
                                            contrast_limits=(100,5000))
            new_roi_table = ip.update_roi_points(point_layer, H.roi_table, 
                                                position=position, 
                                                downscale= H.config['image_view_downscaling'])
            H.roi_table = new_roi_table.copy()
            H.save_data(rois=H.roi_table)
            
        elif event == '-RUN_TRACING-':
            H.set_drift_table(values['-DC_PATH-'])
            T = Tracer(H)
            T.tracing_3d()

        elif event in  (None, 'Exit'):
            client.close()
            break
    window.close()

if __name__ == '__main__':
    try:
        client = Client('127.0.0.1:8787') #Check for existing local dask client
    except  OSError:
        client = Client() #If no local client exists, start a new one.
    
    main()