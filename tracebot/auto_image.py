# -*- coding: utf-8 -*-
"""
On the fly drift correction interfacing with MyPIC
"""

import winreg
import time
import os
import io
import sys
from skimage.registration import phase_cross_correlation
import czifile as cz
from xml.etree import ElementTree
import yaml
from datetime import date
import logging
import pyautogui
pyautogui.FAILSAFE = False
import json
import threading

def update_status(new_data, filename='status.json'):
    '''
    Update selected field(s) in status.json file.
    Input:
        new_data: Dictionary with status commands.
        filename: Path to status json file.
    '''
    with open(filename,'r') as f:
        data=json.load(f)
    for k,v in new_data.items():
        data[k]=new_data[k]
    with open(filename, 'w') as f:
        json.dump(data, f)
    logging.info('Status updated to: '+str(data))

def read_status(filename='status.json'):
    '''
    Read out the status.json file.
    '''

    with open(filename,'r') as f:
        status=json.load(f)
    return status

def set_command(command, filename='status.json'):
    '''
    Convenience function for setting current command in status json file.
    Input:
        command: string with text to go in command key dict in status json.
    '''
    update_status({'command':command}, filename)

def imaging_loop():
    '''
    Loop for automated imaging based on json input. Runs as follows:
        - Check if status json reports imaging.
        - Try to click resume button.
        - Starts loop to wait for completion of imaging.
        - Once complete, pause imaging again.
        - Send command for fluidics sequence to start to json.
    '''
    # Set original file modification time, only check status if file changed.
    status_mod_time_old = os.stat('status.json').st_mtime
    while True:
        status_mod_time_new = os.stat('status.json').st_mtime
        if status_mod_time_new != status_mod_time_old:
            time.sleep(2)
            status = read_status()
            if status['command'] == 'image':
                logging.info('Image command found, starting imaging.')
                #zen_window=gw.getWindowsWithTitle('Zen 2.3 SP1')[0]
                #zen_window.activate()
                #time.sleep(1)
                try:
                    pyautogui.click('pause_button_paused.png', button='left')
                except TypeError:
                    try:
                        pyautogui.click('pause_button_paused2.png', button='left')
                    except TypeError:
                        logging.error('Pause button not found.')
                        sys.exit(0)
                    
                pyautogui.moveTo(500, 500, duration=0.25)
                logging.info('Imaging started.')
                set_command('imaging')
                time.sleep(5)
                
                while True:
                    check=pyautogui.locateOnScreen('imaging_cycle_done.png')
                    if check != None:
                        logging.info('Imaging cycle done.')
                        break
                    else:    
                        time.sleep(2)
                    
                try:
                    pyautogui.click('pause_button.png', button='left')
                except TypeError:
                    logging.error('Pause button not found.')
                    sys.exit(0)
                
                pyautogui.moveTo(500, 500, duration=0.25)
                
                logging.info('Imaging complete.')
                set_command('robot')
                status_mod_time_old = os.stat('status.json').st_mtime
        else:
            time.sleep(3)

def read_czi_image(image_path):
    tags = {'Title',
            'SizeX',
            'SizeY',
            'SizeZ',
            'SizeC',
            'SizeT',
            'SizeS',
            'Model',
            'System',
            'ScalingX',
            'ScalingY',
            'ScalingZ'}
    
    with cz.CziFile(image_path) as czi:
        image=czi.asarray()
        
    metadict = read_czi_meta(image_path, tags)
    
    return image, metadict

def read_czi_meta(image_path, tags, save_meta=False):
    '''
    Function to read metadata and image data for CZI files.
    Define the information to be extracted from the xml tags dict in config file.
    Optionally a YAML file with the metadata can be saved in the same path as the image.
    Return a dictionary with the extracted metadata.
    '''
    def parser(data, tags):
        tree = ElementTree.iterparse(data, events=('start',))
        _, root = next(tree)
    
        for event, node in tree:
            if node.tag in tags:
                yield node.tag, node.text
            root.clear()
    
    with cz.CziFile(image_path) as czi:
        meta=czi.metadata()
    
    with io.StringIO(meta) as f:
        results = parser(f, tags)
        metadict={} 
        for tag, text in results:
            metadict[tag]=text
    if save_meta:
        with open(image_path[:-4]+'_meta.yaml', 'w') as myfile:
            yaml.safe_dump(metadict, myfile)
    return metadict

def drift_corr_cc(offset_image, template_image, upsampling=1, downsampling=1):
    '''
    Performs 3d drift correction by cross-correlation.
    Image can be upsampled for sub-pixel accuracy,
    or downsampled (by simple slicing) for speed based on sampling
    '''
    
    s = slice(None,None,downsampling)
    shift = phase_cross_correlation(template_image[:,s,s], offset_image[:,s,s], upsample_factor=upsampling, return_error=False)
    shift = shift * [1, downsampling, downsampling] 
    return shift

def read_regkey(root, path):
    '''
    Reads an whole key from the windows registry.
    Input: 
        root: a winreg.HKEY_ constant
        path: String with subpath to the key
    Output: List of tuples containing all the values (i.e. name, value, type) of the key.
    '''
    
    with winreg.OpenKey(root, path) as key:
        value_list = []
        num_values=winreg.QueryInfoKey(key)[1]
        for num in range(num_values):
            value_list.append(winreg.EnumValue(key, num))
        
        return value_list

def value_in_regkey(root, path, value_name):
    '''
    Searches for a given value in a registry key.
    
    Input: 
        root:       a winreg.HKEY_ constant
        path:       String with subpath to the key
        value_name: The name of the value to search for (a string)
    Output: First tuple containing the indicated value.
    '''
    
    value_list=read_regkey(root, path)
    for entry in value_list:
        if entry[0]==value_name:
            return entry
        
def write_regvalue(root, path, value):
    '''
    Input:
        root:       a winreg.HKEY_ constant
        path:       String with subpath to the key
        value:      A tuple of the form (value_name, value_content, type), type usually is 1.
    Output: Returns True if the value was succesfully written and matches the input value.
    '''
    
    with winreg.OpenKey(root, path, access=winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value[0], 0, value[2], value[1])
    return value_in_regkey(root, path, value[0]) == value

def register_new_image(image_path, template_folder=False):
    '''    
    Finds the offset between two images using drift_corr_cc, 
    where the template image is defined having 'template' in the name.
    NB! Assumes alignment channel is 0.
    Input:
        image_path: path to image to analyze, should be a czi image.
        template_folder: top level folder of MyPIC experiment
    Output:
        shift: a tuple, zyx shift in pixels between template and test image
        image_meta: a dict containing metadata of the test image
    '''    

    # Finding file paths for template and new image.
    image_dir=os.path.dirname(image_path)+os.sep
    
    if template_folder == False:
        files=[file for file in os.listdir(image_dir) if '.czi' in file]
        template_path=image_dir+files[0]
    else:
        subfolder = [f.path for f in os.scandir(template_folder) if f.is_dir() and image_path[-21:-10] in f.path]
        files=[file for file in os.listdir(subfolder[0]) if '.czi' in file]
        template_path=subfolder[0]+os.sep+files[0]
        
    logging.info('Template image: '+template_path)
    logging.info('New image: '+image_path)
    
    # Read image data and extract only 
    image_data, image_meta = read_czi_image(image_path)
    template_data, template_meta = read_czi_image(template_path)
    
    image_data = image_data[0,0,0,0,:,:,:,0]
    template_data = template_data[0,0,0,0,:,:,:,0]
    
    #Special case for first image on confocal to center on max int plane.
    if image_path == template_path and 'LSM' in image_meta['System']:
        z_max=np.argmax(np.sum(image_data,axis=(1,2)))
        z_center = image_data.shape[0]/2
        shift = np.array([z_max-z_center,0.,0.])
    
    # Regular rapid drift correction to template.
    else:
        shift = drift_corr_cc(image_data, template_data, upsampling=8, downsampling=8)
    
    return shift, image_meta

def dc_loop(root, path, template_folder=False):
    '''
    Runs drift correction loop as follows:
        - Look for newImage value in myPIC registry.
        - Run crosscorr drift correction based on template image choice.
        - Calculate new coordinates for stage.
        - Write new coordinates to registry and update registry values to start imaging again.
    
    Input:
        Root, path: Strings with location in registry of myPIC entries.
        Template_folder: Optional, if False uses first image in same folder as newImage, otherwise first image in given folder.
    '''
    
    while True:
        if value_in_regkey(root, path, 'codeOia')[1] == 'newImage':
            image_path = value_in_regkey(root, path, 'filePath')[1]
            shift, image_meta = register_new_image(image_path, template_folder)
            size_x, size_y, size_z = int(image_meta['SizeX']), int(image_meta['SizeY']), int(image_meta['SizeZ'])
            newX, newY, newZ =  str(size_x/2.0-shift[2]), str(size_y/2.0-shift[1]), str(size_z/2.0-shift[0])
            logging.info('Finished registering, newX, newY, newZ: '+newX+' '+newY+' '+newZ)
            write_regvalue(root, path, ('X', newX, 1))
            write_regvalue(root, path, ('Y', newY, 1))
            write_regvalue(root, path, ('Z', newZ, 1))
            write_regvalue(root, path, ('codeMic', 'focus', 1))
            write_regvalue(root, path, ('codeOia', 'nothing', 1))
            time.sleep(5)
        time.sleep(2)
    
def error_check(root, path):
    '''
    Checks if myPIC registry entries and status json are available.
    Logs error and exits if not.

    Returns
    -------
    None.

    '''        
    try: 
        value_in_regkey(root, path, 'codeOia')[1]
    except (TypeError, FileNotFoundError):
        logging.error('myPIC macro not initialized correctly.')
        sys.exit(0)
    
    try: 
        value_in_regkey(root, path, 'filePath')[1]
    except (TypeError, FileNotFoundError):
        logging.error('myPIC macro not initialized correctly.')
        sys.exit(0)
        
    try:
        read_status()
    except (FileNotFoundError, PermissionError):
        logging.error('Cannot find or access status.json file')
        sys.exit(0)
    

def main_loop(template_folder=False):
    
    '''
    Running function. Performs the following tasks:
    Checks if myPIC registry entries and status json files are available.    
    Initiate logging function. (Small modification for PIL in pyautogui to prevent internal DEBUG logging.)
    Launches automated imaging loop in seperate thread.
    Checks if new image available for analysis from myPIC. Perform drift correction on the fly, updating the stage position if so.
    '''
    logging.basicConfig(level=logging.DEBUG, filename='logging'+os.sep+str(date.today())+'imaging_log.log',
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(logging.getLogger().handlers) < 2:
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger('PIL').setLevel(logging.INFO)
    
    root=winreg.HKEY_CURRENT_USER
    path="SOFTWARE\\VB and VBA Program Settings\\OnlineImageAnalysis\\macro"
    
    error_check(root, path)    
    
    logging.info('AutoImage script initialized in '+str(template_folder))
            
    t_img = threading.Thread(target=imaging_loop)
    t_img.start()

    t_dc = threading.Thread(target=dc_loop, args=(root, path, template_folder))
    t_dc.start()


if __name__ == '__main__':
    if len(sys.argv)>1:
        main_loop(sys.argv[1])
    else:
        main_loop()