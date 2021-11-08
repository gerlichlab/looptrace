'''
GUI for running GRBL stage and connected pump.

'''

import PySimpleGUI as sg
import os
import logging
import threading
import time
from mw_pump_functions import Robot
from datetime import date
from io import StringIO

sg.theme('Dark Blue 3')  # please make your windows colorful

def main():

    layout = [
        [sg.Text('Choose config file:')],
        [sg.InputText('robot_config.yaml', key='-CONFIG_PATH-'), sg.FileBrowse()],
        [sg.Button('Initialize robot', key='-INIT-'), 
        sg.Button('Refresh config', key='-REFRESH-'),
        sg.Button('Reconnect pump', key='-REPUMP-'),
        sg.Button('Reconnect stage', key='-RESTAGE-')],
        [sg.Text('_'*30)],
        [sg.Text('Pump time [s]'), sg.InputText(size=(4, None), key='-PUMP_TIME-', default_text='23'),
        sg.Button('Pump cycle', key='-PUMP-')],
        [sg.Text('_'*30)],
        [sg.Text('X'), sg.InputText(size=(3, None), key='-X-'), 
        sg.Text('Y'), sg.InputText(size=(3, None), key='-Y-'),
        sg.Text('Z'), sg.InputText(size=(3, None), key='-Z-'),
        sg.Button('Move to coordinates', key='-MOVE-'),
        sg.Button('Move to zero', key='-MOVE_ZERO-'), 
        sg.Button('Zero stage', key='-ZERO-')],
        [sg.InputCombo('None', size=(20, 1), key='-POSITION-'), 
        sg.Button('Move to position', key='-MOVE_POS-')],
        [sg.Text('_'*30)],
        [sg.Button('Start robot', key='-START-'),
        sg.Checkbox('Restart?', key='-RESTART-', default=True),
        sg.Button('Stop robot', key='-STOP-')],
        [sg.Text('_'*30)],
        [sg.Text('Output:')],
        [sg.Multiline(size=(50, 15), key='-LOG-', autoscroll=True)]
            ]

    window = sg.Window('Robot control', layout)

    #Initialize logging system, and print messages to console as well
    if not os.path.isdir('logging'):
        os.mkdir('logging')
    logging.basicConfig(level=logging.DEBUG, filename='logging'+os.sep+str(date.today())+'_robot.log', 
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(logging.getLogger().handlers) < 2:
        logging.getLogger().addHandler(logging.StreamHandler())
        log_stream = StringIO()
        old_log = 0
        logging.getLogger().addHandler(logging.StreamHandler(stream=log_stream))
    # Loop taking in user input and querying queue
    while True:
        # Wake every 100ms and look for work
        event, values = window.read(timeout=100)

        if event == '-INIT-':
            try:
                R = Robot(values['-CONFIG_PATH-'])
                window['-POSITION-'].update(values=tuple(R.config['positions']))
            except (FileNotFoundError, KeyError) as e:
                logging.error(e)
                logging.info('Please provide a valid config file.')

        elif event == '-REFRESH-':
            try:
                R.refresh_config()
                window['-POSITION-'].update(values=tuple(R.config['positions']))
            except UnboundLocalError:
                logging.info('Robot not initialized.')

        elif event == '-REPUMP-':
            try:
                R.close_pump()
                time.sleep(2)
                R.start_pump()
            except (UnboundLocalError, AttributeError):
                logging.info('Robot, stage or pump not initialized.')

        elif event == '-RESTAGE-':
            try:
                R.close_stage()
                time.sleep(2)
                R.start_stage()
            except (UnboundLocalError, AttributeError):
                logging.info('Robot, stage or pump not initialized.')

        elif event == '-PUMP-':
            try:
                R.pump.pump_cycle(int(values['-PUMP_TIME-']))
            except (UnboundLocalError, AttributeError):
                logging.info('Robot or pump not initialized.')

        elif event == '-MOVE-':
            pos = {'x': values['-X-'], 'y': values['-Y-'], 'z': values['-Z-']}
            pos = {key:val for key, val in pos.items() if val != ''}
            logging.info('Moving to ' + str(pos))
            try:
                R.stage.move_stage(pos)
            except (UnboundLocalError, AttributeError):
                logging.info('Robot or stage not initialized.')

        elif event == '-MOVE_ZERO-':
            pos = {'x': 0, 'y': 0, 'z': 0}
            logging.info('Moving to ' + str(pos))
            try:
                R.stage.move_stage(pos)
            except (UnboundLocalError, AttributeError):
                logging.info('Robot or stage not initialized.')

        elif event == '-ZERO-':
            window['-X-'].update('0')
            window['-Y-'].update('0')
            window['-Z-'].update('0')
            try:
                R.stage.zero_stage()
            except (UnboundLocalError, AttributeError):
                logging.info('Robot or stage not initialized.')

        elif event == '-MOVE_POS-':
            pos=R.config['positions'][values['-POSITION-']]
            logging.info('Moving to '+values['-POSITION-'] + ' '+str(pos))
            try:
                R.stage.move_stage(pos)
            except (UnboundLocalError, AttributeError):
                logging.info('Robot or stage not initialized.')

        elif event == '-START-':
            try:
                R.stop.clear()
                print(values['-RESTART-'])
                run_wp_cycle = threading.Thread(target=R.wp_cycle, kwargs={'restart' : values['-RESTART-']})
                run_wp_cycle.start()
                window['-START-'].update(disabled=True)
            except (UnboundLocalError, AttributeError):
                logging.info('Robot or stage not initialized.')

        elif event == '-STOP-':
            window['-START-'].update(disabled=False)
            try:
                R.stop.set()
                if not run_wp_cycle.is_alive():
                    logging.info('Robot stopped.')
            except (UnboundLocalError):
                logging.info('Robot not initialized.')

        elif event in  (None, 'Exit'):
            try:
                R.close_connections()
                R.stop_socket_server()
            except (UnboundLocalError, AttributeError):
                pass

            break

        log = log_stream.getvalue().splitlines()
        if old_log != len(log):
            msg = str(log[old_log:])
            now = time.strftime("%x, %X", time.localtime())
            window['-LOG-'].update(now+msg+'\n', append=True)
            old_log=len(log)

    window.close()


if __name__ == '__main__':
    main()