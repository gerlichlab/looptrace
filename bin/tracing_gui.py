import PySimpleGUI as sg
import os
import threading
import napari
from chromatin_tracing_python.tracer import Tracer
from chromatin_tracing_python.drift_correction_tracing_dask import Drifter
import chromatin_tracing_python.image_processing_functions as ip
from dask.distributed import Client

sg.theme('Dark Blue 3')  # please make your windows colorful

def napari_view(img, shape=None, shape_props = None):
    with napari.gui_qt():
        viewer = napari.view_image(img, contrast_limits=(0,20000))
        if shape is not None:
            shape_layer = viewer.add_shapes(shape, properties=shape_props, 
                                                   edge_color='red', 
                                                   edge_width=2,
                                                   face_color='transparent')
    if shape is not None:
        return shape_layer
    
def main():

    layout = [
        [sg.Text('Choose config file:')],
        [sg.InputText('YAML_config_file', key='-CONFIG_PATH-'), sg.FileBrowse()],
        [sg.Text('_'*50)],
        [sg.Button('Run drift correction', key='-RUN_DC-'),
        sg.Button('Apply drift correction', key='-APPLY_DC-')],
        [sg.Text('_'*50)],
        [sg.Text('Choose drift correction file:')],
        [sg.InputText('Drift correction file', key='-DC_PATH-'), sg.FileBrowse()],
        [sg.Button('Initialize images for tracing', key='-TRACER-'),
        sg.Button('View images for tracing', key='-VIEW_IMAGES-', disabled=True),
        sg.Button('Reload config', key='-RELOAD-', disabled=True)],
        [sg.Text('_'*50)],
        [sg.Radio('Detect new ROIs', "ROI_SEL", default=True, key='-NEW_ROI-')],
        [sg.Radio('Use IJ ROIs', "ROI_SEL", key='-IJ_ROI-'),
        sg.InputText('Folder with ImageJ ROIs', key='-IJ_ROI_PATH-'), sg.FolderBrowse()],
        [sg.Radio('Use ROI file', "ROI_SEL", key='-EXISTING_ROI-'),
        sg.InputText('ROI file', key='-ROI_FILE_PATH-'), sg.FileBrowse()],
        [sg.Button('Load/detect ROIs', key='-RUN_ROI-'), sg.Button('View ROIs', key='-VIEW_ROI-', disabled=True)],
        [sg.Text('_'*50)],
        [sg.Button('Run tracing', key='-RUN_TRACING-')]
        ]

    window = sg.Window('Tracing GUI', layout)

    while True:
        # Wake every 100ms and look for work
        event, values = window.read(timeout=100)

        if event == '-RUN_DC-':
            D = Drifter(values['-CONFIG_PATH-'])
            print('Drifter created')
            if os.path.exists(D.dc_file_path):
                dc_exists_choice = sg.popup_yes_no('Existing drift correction file found, recalculate?')
                if dc_exists_choice == 'Yes':
                    dc_thread = threading.Thread(target=D.drift_corr_mypic_h5)
                    dc_thread.start()
            else:
                dc_thread = threading.Thread(target=D.drift_corr_mypic_h5)
                dc_thread.start()
        elif event == '-APPLY_DC-':
            D = Drifter(values['-CONFIG_PATH-'])
            dc_thread = threading.Thread(target = D.apply_drift_corr_mypic)
            dc_thread.start()
        elif event == '-TRACER-':
            T = Tracer(values['-CONFIG_PATH-'], values['-DC_PATH-'])
            window['-VIEW_IMAGES-'].update(disabled = False)
            window['-RELOAD-'].update(disabled = False)
        elif event == '-VIEW_IMAGES-':
            napari_view(T.images)
        elif event == '-RELOAD-':
            T.reload_config()
        elif event == '-RUN_ROI-':
            if values['-NEW_ROI-']:
                roi_thread = threading.Thread(target=T.rois_from_spots)
                roi_thread.start()
            elif values['-IJ_ROI-']:
                T.roi_table = ip.rois_from_imagej(values['-IJ_ROI_PATH-']) # Oeyvind added ", '.zip', 0.5" to read scaled images but it did not work
            elif values['-EXISTING_ROI-']:
                T.roi_table = ip.rois_from_csv(values['-ROI_FILE_PATH-'])
            window['-VIEW_ROI-'].update(disabled = False)
        elif event == '-VIEW_ROI-':
            for position in T.pos_list:
                print('Checking ROIs in position ', position)
                pos_index = T.pos_list.index(position)
                roi_shapes, roi_props = ip.roi_to_napari_shape(T.roi_table, position = position)
                shape_layer = napari_view(T.images[pos_index], shape = roi_shapes, shape_props = roi_props)
                new_roi_table = ip.update_roi_shapes(shape_layer, T.roi_table, position=position)
                T.roi_table = new_roi_table.copy()
                T.save_data(rois=T.roi_table)
            
        elif event == '-RUN_TRACING-':
            trace_thread = threading.Thread(target=T.tracing_3d)
            trace_thread.start()
        elif event in  (None, 'Exit'):
            client.close()
            break
    window.close()

if __name__ == '__main__':
    client = Client()
    main()