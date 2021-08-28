# -*- coding: utf-8 -*-
"""
On the fly drift correction interfacing with MyPIC
"""

import time
import os
from datetime import date
import logging
import pyautogui
pyautogui.FAILSAFE = False
import json

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
                while True:
                    try:
                        pyautogui.click('move_zero.png', button='left')
                        time.sleep(1)
                        pyautogui.click('find_surface.png', button='left')
                        time.sleep(5)
                        pyautogui.click('zero_z.png', button='left')
                        time.sleep(1)
                        pyautogui.click('load.png', button='left')
                        time.sleep(1)
                        pyautogui.click('filename.png', clicks=2, button='left')
                        time.sleep(1)
                        pyautogui.click('start_exp.png', button='left')
                        break
                    except TypeError:
                        logging.error('A button was not found.')
                        continue
                    
                pyautogui.moveTo(500, 500, duration=0.25)
                logging.info('Imaging started.')
                set_command('imaging')
                time.sleep(5)
                
                while True:
                    check=pyautogui.locateOnScreen('start_exp_button.png')
                    if check != None:
                        logging.info('Imaging cycle done.')
                        break
                    else:    
                        time.sleep(3)
                     
                logging.info('Imaging complete.')
                set_command('robot')
                status_mod_time_old = os.stat('status.json').st_mtime
        else:
            time.sleep(3)
 

def main_loop():
    
    '''
    Running function. Performs the following tasks:
    Checks if myPIC registry entries and status json files are available.    
    Initiate logging function. (Small modification for PIL in pyautogui to prevent internal DEBUG logging.)
    Launches automated imaging loop in seperate thread.
    Checks if new image available for analysis from myPIC. Perform drift correction on the fly, updating the stage position if so.
    '''
    if not os.path.isdir('logging'):
        os.mkdir('logging')
    logging.basicConfig(level=logging.DEBUG, filename='logging'+os.sep+str(date.today())+'_imaging.log',
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(logging.getLogger().handlers) < 2:
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger('PIL').setLevel(logging.INFO)
    
    logging.info('AutoImage script initialized')
            
    imaging_loop()

if __name__ == '__main__':
    main_loop()