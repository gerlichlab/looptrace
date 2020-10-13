import PySimpleGUI as sg
import os
import threading

from chromatin_tracing_python.Tracer import Tracer
from chromatin_tracing_python.Drifter import Drifter
from chromatin_tracing_python.ImageHandler import ImageHandler
from chromatin_tracing_python.SpotPicker import SpotPicker
import chromatin_tracing_python.image_processing_functions as ip
from dask.distributed import Client
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
sg.theme('Dark Blue 3')  # please make your windows colorful


    
def main():

    layout = [
        [sg.Text('Choose config file:')],
        [sg.InputText('YAML_config_file', key='-CONFIG_PATH-'), sg.FileBrowse()],
        [sg.Button('Initialize', key='-INIT-'),
        sg.Button('View images for tracing', key='-VIEW_IMAGES-'),
        sg.Button('Reload config', key='-RELOAD-')],
        [sg.Text('_'*50)],
        [sg.Button('Run drift correction', key='-RUN_DC-'),
        sg.Button('Apply drift correction', key='-APPLY_DC-')],
        [sg.Text('_'*50)],
        [sg.Text('Choose drift correction file:')],
        [sg.InputText('Drift correction file', key='-DC_PATH-'), sg.FileBrowse()],
        [sg.Text('_'*50)],
        [sg.Radio('Detect new ROIs', "ROI_SEL", default=True, key='-NEW_ROI-')],
        [sg.Radio('Use IJ ROIs', "ROI_SEL", key='-IJ_ROI-'),
        sg.InputText('Folder with ImageJ ROIs', key='-IJ_ROI_PATH-'), sg.FolderBrowse()],
        [sg.Radio('Use ROI file', "ROI_SEL", key='-EXISTING_ROI-'),
        sg.InputText('ROI file', key='-ROI_FILE_PATH-'), sg.FileBrowse()],
        [sg.Button('Load/detect ROIs', key='-RUN_ROI-'), sg.Button('View ROIs', key='-VIEW_ROI-')],
        [sg.Text('_'*50)],
        [sg.Button('Run tracing', key='-RUN_TRACING-')]
        ]

    window = sg.Window('Tracing GUI', layout)

    while True:
        # Wake every 100ms and look for work
        event, values = window.read(timeout=100)
        if event == '-INIT-':
            H = ImageHandler(values['-CONFIG_PATH-'])
            window['-DC_PATH-'].update(H.dc_file_path)
        elif event == '-VIEW_IMAGES-':
            ip.napari_view(H.images)
        elif event == '-RELOAD-':
            H.reload_config()
        elif event == '-RUN_DC-':
            D = Drifter(H)
            if os.path.exists(D.dc_file_path):
                dc_exists_choice = sg.popup_yes_no('Existing drift correction file found, recalculate?')
                if dc_exists_choice == 'Yes':
                    dc_thread = threading.Thread(target=D.drift_corr_mypic)
                    dc_thread.start()
            else:
                dc_thread = threading.Thread(target=D.drift_corr_mypic)
                dc_thread.start()
        elif event == '-APPLY_DC-':
            D = Drifter(H)
            dc_thread = threading.Thread(target = D.apply_drift_corr_mypic)
            dc_thread.start()
        
        elif event == '-RUN_ROI-':
            if values['-NEW_ROI-']:
                S = SpotPicker(H)
                roi_thread = threading.Thread(target=S.rois_from_spots)
                roi_thread.start()
            elif values['-IJ_ROI-']:
                H.roi_table = ip.rois_from_imagej(values['-IJ_ROI_PATH-'], '.zip', 16 , float(T.config['dc_image_scaling']))
                H.save_data(rois=H.roi_table)
            elif values['-EXISTING_ROI-']:
                H.roi_table = ip.rois_from_csv(values['-ROI_FILE_PATH-'])

        elif event == '-VIEW_ROI-':
            for position in H.pos_list:
                print('Checking ROIs in position ', position)
                pos_index = H.pos_list.index(position)
                #roi_shapes, roi_props = ip.roi_to_napari_shape(T.roi_table, position = position)
                roi_points, roi_props = ip.roi_to_napari_points(H.roi_table, position = position)
                print(roi_points)
                if roi_points == []:
                    print('No ROIs found, skipping position.')
                    continue
                point_layer = ip.napari_view(H.images[pos_index], 
                                             points = roi_points,
                                             downscale= H.config['image_view_downscaling'],
                                             trace_ch = H.config['trace_ch'],
                                             ref_slice= H.config['ref_slice'])
                new_roi_table = ip.update_roi_points(point_layer, H.roi_table, 
                                                    position=position, 
                                                    downscale= H.config['image_view_downscaling'])
                H.roi_table = new_roi_table.copy()
                H.save_data(rois=H.roi_table)
            
        elif event == '-RUN_TRACING-':
            H.set_drift_table(values['-DC_PATH-'])
            T = Tracer(H)
            trace_thread = threading.Thread(target=T.tracing_3d)
            trace_thread.start()

        elif event in  (None, 'Exit'):
            client.close()
            break
    window.close()

if __name__ == '__main__':
    client = Client()
    main()