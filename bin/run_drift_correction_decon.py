# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:32:52 2020

@author: ellenberg

Running script for drift correction of deconvoluted images.

"""
from chromatin_tracing_python import drift_correction_decon as dc
from chromatin_tracing_python import image_processing_functions as ip
from tkinter.filedialog import askopenfilename
import os

def main():
    
    config_path = askopenfilename()
    config = ip.load_config(config_path)
    print('Read config: ', config)
    
    input_folder = config['input_folder']
    output_folder = config['output_folder']
    output_filename = config['drift_output_filename']
    filetypes = config['image_filetype']
    template = config['image_template']
    threshold = config['bead_threshold']
    min_bead_int = config['min_bead_intensity']
    ch = config['bead_ch']
    t_index = config['bead_reference_timepoint']
    points = config['bead_points']
    dc_image_scale = config['dc_image_scaling']
    
    if os.path.exists(output_folder+os.sep+output_filename):
        skip=input('Existing drift correction file found, skip calculating drift? (y/n)')
        if skip != 'y':
            drift = dc.drift_corr_mypic_h5(input_folder,
                                output_folder,
                                output_filename,
                                threshold, 
                                min_bead_int, 
                                points, 
                                t_index,
                                ch,
                                filetypes, 
                                template)
        else:
            pass
    else:
        drift = dc.drift_corr_mypic_h5(input_folder,
                    output_folder,
                    output_filename,
                    threshold, 
                    min_bead_int, 
                    points, 
                    t_index,
                    ch,
                    filetypes, 
                    template)
    
    apply=input('Apply drift correction? (y/n)')
    if apply == 'y':
    
        dc.apply_drift_corr_mypic(input_folder,
                               output_folder,
                               dc_file,
                               filetype,
                               template,
                               dc_image_scale)
    else:
        pass
    
    print('Drift correction done.')
    
if __name__ == '__main__':
    main()