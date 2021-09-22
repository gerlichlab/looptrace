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

import socket
import time

def read_command():
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b'read_command')
        data = s.recv(1024).decode('utf-8')
    return data

def set_command(command):
    '''
    Convenience function for setting current command in status json file.
    Input:
        command: string with text to go in command key dict in status json.
    '''
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        if command == 'imaging':
            s.sendall(b'set_imaging')
        elif command == 'robot':
            s.sendall(b'set_robot')
        data = s.recv(1024).decode('utf-8')
    return data

def imaging_loop():
    '''
    Loop for automated imaging based on json input. Runs as follows:
        - Check if status json reports imaging.
        - Try to click resume button.
        - Starts loop to wait for completion of imaging.
        - Once complete, pause imaging again.
        - Send command for fluidics sequence to start to json.
    '''
    while True:
        command = read_command()
        if command == 'image':
            time.sleep(1)
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
    #logging.getLogger('PIL').setLevel(logging.INFO)
    
    logging.info('AutoImage script initialized')
            
    imaging_loop()

if __name__ == '__main__':)
    main_loop()