import serial
import json
import PySimpleGUI as sg
import os
import logging
import threading
import time
from datetime import date
from io import StringIO
from serial.tools.list_ports import comports

class MPX():

    def __init__(self, com_port):
        
        #Load configuration file
        self.stop = threading.Event()
        self.pump = self.initialize_pump(com_port)
        logging.info('Pump initialized.')

    def initialize_pump(self, port): #Pump communication initialize
        try:
            pump=serial.Serial(port=port,timeout=3)  # open serial port
            logging.info('Pump connection established on '+pump.name)
        except serial.SerialException:
            logging.error('No pump found on ' + self.config['pump_port'])
        return pump

    def close(self):
        logging.info('Disconnecting pump.')
        self.pump.close()

    def bartels_set_freq(self, freq):
        pump = self.pump
        pump.write(('F'+str(freq)+'\r').encode('utf-8'))  
        logging.info('Set frequency '+str(freq))

    def bartels_set_voltage(self, voltage):
        pump = self.pump
        pump.write(('A'+str(voltage)+'\r').encode('utf-8')) 
        logging.info('Set voltage '+str(voltage))
    
    def bartels_set_waveform(self, waveform):
        pump = self.pump
        pump.write((waveform+'\r').encode('utf-8')) 
        logging.info('Set waveform to '+waveform) 
    
    def bartels_start(self):
        pump = self.pump
        pump.write(b'bon\r') 
        logging.info('Pump ON')

    def bartels_stop(self):
        pump = self.pump
        pump.reset_output_buffer()
        pump.write(b'boff\r')
        logging.info('Pump OFF')


    def bartels_cycle(self, run_time, interval):
        while True:
            self.bartels_start()
            for i in range(int(run_time)):
                time.sleep(1)
                if self.stop.is_set():
                    logging.info('Stopping pump.')
                    raise SystemExit
            self.bartels_stop()
            for i in range(int(interval)):
                time.sleep(1)
                if self.stop.is_set():
                    logging.info('Stopping pump.')
                    raise SystemExit
    
    def bartels_status_loop(self):
            pump=self.pump
            pump.write(b'\r')
            while True:
                pump_out=pump.readline().decode('utf-8')
                #logging.info('Pump status: '+pump_out)
                time.sleep(0.1)
                if pump_out == '':
                    break

    def bartels_status(self):
        
        #pump.flushOutput()
        #pump.write(b'\n\r')
        
        status = threading.Thread(target=self.bartels_status_loop)
        status.start()

def available_ports():
    ports = list(comports())
    coms = [p.device for p in ports]

    return coms

def read_defaults():
    #Read out the status.json file.
    with open('bartels_defaults.json','r') as f:
        defaults=json.load(f)
    return defaults
    
def set_defaults(new_defaults):
    #Update selected field(s) in status.json file. new_defaults should be dictionary.
    with open('bartels_defaults.json','r') as f:
        data=json.load(f)
    for k,v in new_defaults.items():
        data[k]=new_defaults[k]
    with open('bartels_defaults.json','w') as f:
        json.dump(data, f)
    logging.info('Defaults updated to: '+str(data))

sg.theme('Dark Blue 3')  # please make your windows colorful

def main():

    layout = [
        [sg.Button('Refresh COM ports', key='-REFRESH_COM-'),
        sg.InputCombo('None', size=(6, 1), key='-COM-'),
        sg.Button('Connect pump', key='-CONNECT-')],
        [sg.Text('_'*30)],
        [sg.Button('Load defaults', key='-LOAD-'), 
        sg.Button('Save defaults', key='-SAVE-')],
        [sg.Text('Frequency: '), sg.InputText(size=(4, None), key='-FREQ-'), 
        sg.Text('Voltage: '), sg.InputText(size=(4, None), key='-VOLT-'),
        sg.Text('Waveform: '),
        sg.InputCombo(['MC', 'MS', 'MR'], size=(4, 1), key='-WAVEFORM-', default_value='MC')],
        [sg.Text('On time [s]: '), sg.InputText(size=(3, None), key='-RUN_TIME-'),
        sg.Text('Off interval [s]: '), sg.InputText(size=(3, None), key='-PAUSE_TIME-')],
        [sg.Button('Set values', key='-SET-'),
        sg.Button('Start pump', key='-START-'),
        sg.Button('Stop pump', key='-STOP-')]
        #sg.Button('Pump status', key='-STATUS-')
        ]

    window = sg.Window('Bartels MP-X control', layout)

    #Initialize logging system, and print messages to console as well
    logging.basicConfig(level=logging.DEBUG, filename='logging'+os.sep+str(date.today())+'_mpx_log.log', 
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(logging.getLogger().handlers) < 2:
        logging.getLogger().addHandler(logging.StreamHandler())
    # Loop taking in user input and querying queue
    while True:
        # Wake every 100ms and look for work
        event, values = window.read(timeout=100)

        if event == '-LOAD-':
            defaults=read_defaults()
            logging.info('Defaults loaded: ' + str(defaults))
            for key, value in defaults.items():
                window[key].update(value)

        elif event == '-SAVE-':
            defaults = {key:value for key, value in values.items() if key in ['-FREQ-', '-VOLT-', '-RUN_TIME-', '-PAUSE_TIME-']}
            set_defaults(defaults)

        elif event == '-REFRESH_COM-':
            ports=available_ports()
            window['-COM-'].update(values=ports)

        elif event == '-CONNECT-':
            logging.info('Connecting to ' + str(values['-COM-']))
            P = MPX(values['-COM-'])

        elif event == '-SET-':
            freq = values['-FREQ-']
            volt = values['-VOLT-']
            waveform = values['-WAVEFORM-']
            logging.info('Setting values.')
            P.bartels_set_freq(freq)
            time.sleep(0.2)
            P.bartels_set_voltage(volt)
            time.sleep(0.2)
            P.bartels_set_waveform(waveform)

        elif event == '-START-':
            run_pump_cycle = threading.Thread(target=P.bartels_cycle, kwargs={'run_time' : values['-RUN_TIME-'],
                                                                              'interval' : values['-PAUSE_TIME-']})
            run_pump_cycle.start()
            window['-START-'].update(disabled=True)

        elif event == '-STOP-':
            window['-START-'].update(disabled=False)
            P.bartels_stop()
            P.stop.set()
            time.sleep(3)
            if not run_pump_cycle.is_alive():
                logging.info('Pump stopped.')
                
        elif event == '-STATUS-':
            logging.info('Checking status.')
            P.bartels_status()

        elif event in  (None, 'Exit'):
            P.close()
            break

    window.close()


if __name__ == '__main__':
    main()

