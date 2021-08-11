import serial
import yaml
import logging
import time
import json
import os
from threading import Thread, Event

'''
# Before running, ensure correct COM ports are set in yaml file.
# Use GUI in bin folder to run. If running without GUI:
# To run first execute: p, s , config = initialize() to start pump and stage.
# Check that pump is OK, and make sure to reset 0 position correctly.
# Finally, run wp_cycle() to start automated run according to config file.


pyserial: Copyright (c) 2001-2017 Chris Liechti <cliechti@gmx.net> All Rights Reserved.

Created by:
Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg

'''

class Robot():

    def __init__(self, config_path):
        
        #Load configuration file
        self.stop = Event() 
        self.config_path = config_path
        self.config = self.load_config()
        self.status_file = 'status.json'
        self.all_coords, self.sel_coords = self.wp_coord_list()

        self.stage = Stage(self.config)
        
        if self.config['pump_type'] == 'bartels':
            self.pump = Bartels(self.config)
        elif self.config['pump_type'] == 'CPP':
            self.pump = CPP_pump(self.config)
        else:
            logging.error('Unknown pump type.')

    def close_connections(self):
        try:
            self.stage.close()
            logging.info('Closed stage connection.')
        except AttributeError:
            logging.info('Stage already closed')
        try:
            self.pump.close()
            logging.info('Closed pump connection.')
        except AttributeError:
            logging.info('Pump already closed')

    def load_config(self):
        #Open config file and return config variable form yaml file
        with open(self.config_path, 'r') as file:
            try:
                config=yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    def refresh_config(self):
        self.config = self.load_config()
        if hasattr(self, 'stage'):
            self.stage.config = self.config
        if hasattr(self, 'pump'):
            self.pump.config = self.config

        self.all_coords, self.sel_coords = self.wp_coord_list()
        logging.info('Config refreshed.')

    def update_status(self, new_data):
        #Update selected field(s) in status.json file. new_data should be dictionary.
        with open(self.status_file,'r') as f:
            data=json.load(f)
        for k,v in new_data.items():
            data[k]=new_data[k]
        with open(self.status_file, 'w') as f:
            json.dump(data, f)
        logging.info('Status updated to: '+str(data))

    def read_status(self):
        #Read out the status.json file.
        with open(self.status_file,'r') as f:
            status=json.load(f)
        return status

    def set_well(self, well):
        #Convenience function for setting current well status with a string.
        self.update_status({'current_well': well})

    def set_command(self, command):
        #Convenience function for setting current command status with a string.
        self.update_status({'command':command})
            
    def pause(self, sleep_time):
        #Set and run a pause step, and log.
        time.sleep(int(sleep_time))
        logging.info('Paused for '+str(sleep_time))

    def wp_coord_list(self):
    # Generate coordinate list for whole and selected wells of a 96-well plate.
    # Also adjusts for a rotated plate by using measured top left and bottom right positions from
    # config file.

        config=self.config['well_plate']
        rows=config['rows']
        columns=config['columns']
        z_base=config['z_base']
        well_spacing=config['well_spacing']
        tl_x=config['top_left']['x']
        tl_y=config['top_left']['y']
        br_x=config['bottom_right']['x']
        br_y=config['bottom_right']['y']
        first_probe=config['first_probe']
        last_probe=config['last_probe']

        # Make list of well names selected in config file
        all_wells=[]
        for i in range(rows):
            for j in range(columns):
                all_wells.append(chr(65+i)+str(j+1))
        
        # Calculate adjustment based on measured top left and bottom right well positions 
        well_adjust_x=(1-(br_x-tl_x)/(12*well_spacing))
        well_adjust_y=(1-(br_y-tl_y)/(8*well_spacing))
        
        # Generate list of all well coordinates adjusted for rotation.
        all_coords={} 
        for i in range(rows):
            for j in range(columns):
                all_coords[chr(65+i)+str(1+j)]={'x': tl_x+j*well_spacing+i*well_adjust_x, 'y': tl_y+i*well_spacing+j*well_adjust_y, 'z':config['z_base']}
        
        # Make coordinate list for only the selected wells
        first_well=all_wells.index(first_probe)
        last_well=all_wells.index(last_probe)
        sel_wells=all_wells[first_well:(last_well+1)]
        sel_coords={}
        for well in sel_wells:
            sel_coords[well]=all_coords[well]
        
        return all_coords, sel_coords

    '''
    # Make json files that store all wellplate coordinates and selected wellplate coordinates.
    with open('all_coords.json', 'w') as file:
        json.dump(all_coords, file)
    with open('sel_coords.json', 'w') as file:
        json.dump(sel_coords, file)
    '''

    def single_cycle(self):
        '''
        Runs a single sequence as defined in the config file.
        '''
        config=self.config
        all_coords = self.all_coords
        sel_coords = self.sel_coords
        sel_coord_list=list(sel_coords.keys())
        
        for a in config['sequence']:                # Sequence is defined as list in config file.
            action=list(a.keys())[0]
            param=list(a.values())[0]

            if action == 'probe' and param == 'wp': # Run for wellplate position defined in status json file.
                if self.stop.is_set():
                    logging.info('Stopping robot.')
                    raise SystemExit

                current_pos=self.read_status()['current_well']
                coords=all_coords[current_pos]
                self.stage.move_stage(coords)
                while self.stage.check_stage() != 'Idl': #Wait until move is done before proceeding.
                    time.sleep(2)
                logging.info('Probe at '+str(current_pos))
                
                i=sel_coord_list.index(current_pos)
                try:
                    next_well=sel_coord_list[i+1]
                except IndexError:
                    logging.info('Last well reached')
                    next_well='Last'
                self.set_well(next_well)
                
            elif action == 'probe' and param != 'wp':   # For all position outside well plates, such as reservoirs.
                if self.stop.is_set():
                    logging.info('Stopping robot.')
                    raise SystemExit
                try: 
                    coords=config['positions'][param]
                    self.stage.move_stage(coords)
                    while self.stage.check_stage() != 'Idl': #Wait until move is done before proceeding.
                        time.sleep(2)
                    logging.info('probe at '+str(param))
                except KeyError:
                    logging.error('Invalid probe position: '+str(param))

            elif action == 'pump':
                if self.stop.is_set():
                    logging.info('Stopping robot.')
                    raise SystemExit
                self.pump.pump_cycle(param)
                time.sleep(1)
                logging.info('Pump cycle completed for '+str(param))

            elif action == 'pause':
                if self.stop.is_set():
                    logging.info('Stopping robot.')
                    raise SystemExit
                self.pause(param)

            elif action == 'image':
                if self.stop.is_set():
                    logging.info('Stopping robot.')
                    raise SystemExit
                self.set_command('image')
                logging.info('Starting imaging.')
                status_mod_time_old = os.stat('status.json').st_mtime
                time.sleep(2)
                while True:
                    status_mod_time_new = os.stat('status.json').st_mtime
                    if status_mod_time_new == status_mod_time_old:
                        time.sleep(5)
                    else:
                        command_status=self.read_status()['command']
                        if command_status=='image' or command_status=='imaging':
                            time.sleep(5)
                        else:
                            logging.info('Imaging complete.')
                            break
            else:
                logging.error('Unrecognized sequence command')       

    def calc_num_cycles(self):
        '''
        Helper function to check number of cycles based config setting.
        If an int, runs that number of cycles. If 'all' counts how many cycles is 
        needed to probe all wellplate positions.
        
        Output:
            num_cycles: Int with number of cycles.

        '''
        config=self.config
        sel_coords = self.sel_coords
        wells_seq = 0
        for seq in config['sequence']:
            for val in seq.values():
                if val == 'wp':
                    wells_seq += 1
        wells=len(sel_coords.keys())
        num_cycles=wells//wells_seq
        print('Calculated number of cycles is ', num_cycles)
        return num_cycles

    def wp_cycle(self, restart=True):
        '''
        Runs a full cycle of sequences for all selected well positions.
        Refreshes coordinate list based on config file.
        Runs number of cycles determined by det_num_cycles helper function.
        Restart flag determines if start from current well in status or first well.
        '''
        
        config=self.config
        if self.config['n_cycles'] == 'all':
            num_cycles=self.calc_num_cycles()
        else:
            num_cycles = self.config['n_cycles']
        self.set_command('robot')
        if restart:
            self.set_well(config['well_plate']['first_probe'])
        for cycle in range(num_cycles):
            logging.info('Starting cycle ' + str(cycle) + ' of ' +str(num_cycles)) 
            self.single_cycle()

    
    def socket_server(self):
        import socket
        import sys

        host = 'localhost'
        port = 65432
        address = (host, port)

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(address)
        server_socket.listen(5)

        while True:
            print("Listening for client . . .")
            conn, address = server_socket.accept()
            print("Connected to client at ", address)
            #pick a large output buffer size because i dont necessarily know how big the incoming packet is                                                    
            while True:
                data = conn.recv(2048);
                if data.strip() == b"disconnect":
                    conn.close()
                    print('Closing connection.')
                    break
                    #sys.exit("Received disconnect message.  Shutting down.")
                    #conn.send(b"dack")
                elif data.strip() == b"shutdown":
                    conn.close()
                    print('Received shutdown message.  Shutting down.')
                    sys.exit()

                elif data.strip() == b'read_command':
                    conn.sendall(bytes(self.command))

                elif data.strip() == b'set_imaging':
                    status = self.set_command('imaging')
                    conn.sendall(bytes('Imaging started.'))

                elif data.strip() == b'set_robot':
                    status = self.set_command('robot')
                    conn.sendall(bytes('Robot set.'))

                    
    def start_socket_server(self):
        self.socket_thread = Thread(target=self.socket_server)
        self.socket_thread.start()
    
    def stop_socket_server(self):
        self.socket_thread.join()

class Bartels():

    def __init__(self, config):
        self.config = config
        try:
            self.pump = self.initialize_pump()
            time.sleep(0.2)
            self.bartels_set_freq(self.config['bartels_freq'])
            time.sleep(0.5)
            self.bartels_set_voltage(self.config['bartels_voltage'])
        except serial.SerialException:
            logging.error('No pump found on ' + self.config['pump_port'])

    def initialize_pump(self): #Pump communication initialize
        try:
            pump=serial.Serial(port=self.config['pump_port'],timeout=3)  # open serial port
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

    def pump_cycle(self, run_time):
        self.bartels_start()
        time.sleep(run_time)
        self.bartels_stop()

'''
Not used:    
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
'''

class CPP_pump():

    def __init__(self, config):
        self.config = config
        try:
            self.pump = self.initialize_pump()
            time.sleep(0.2)
        except serial.SerialException:
            logging.error('No pump found on ' + self.config['pump_port'])

    def initialize_pump(self): #Pump communication initialize
        try:
            pump=serial.Serial(port=self.config['pump_port'],timeout=3)  # open serial port
            logging.info('Pump connection established on '+pump.name)
        except serial.SerialException:
            logging.error('No pump found on ' + self.config['pump_port'])
        return pump

    def close(self):
        logging.info('Disconnecting pump.')
        self.pump.close()

    def pump_cycle(self, run_time):
        self.pump.write(('/0S1'+str(self.config['CPP_speed'])+'D0I1M'+str(run_time*1000)+'I0R\n').encode('utf-8'))
        time.sleep(run_time)

class Stage(Robot):

    def __init__(self, config):
        self.config = config
        try:
            self.stage = self.initialize_grbl()
            time.sleep(0.2)
        except serial.SerialException:
            logging.error('No pump found on ' + self.config['pump_port'])

    def initialize_grbl(self):
        s = serial.Serial(port=self.config['stage_port'],baudrate=115200) # open grbl serial port
        s.write(("\r\n\r\n").encode('utf-8')) # Wake up grbl
        time.sleep(2)   # Wait for grbl to initialize
        s.flushInput()  # Flush startup text in serial input
        s.write(('? \n').encode('utf-8')) # Request machine status
        grbl_out = s.readline().decode('utf-8') # Wait for grbl response with carriage return
        logging.info('GRBL initialized:' +grbl_out)
        return s

    def zero_stage(self):
        stage = self.stage
        stage.write(('G10 L20 P0 X0 Y0 Z0 \n').encode('utf-8'))
        grbl_out = stage.readline().decode('utf-8')
        logging.info('Current position set to zero: '+grbl_out)

    def check_stage(self):
        stage = self.stage
        stage.flushInput()
        stage.write(('?\n\r').encode('utf-8'))
        time.sleep(0.2)
        grbl_out = stage.readline().decode('utf-8')
        return grbl_out[1:4]

    def move_stage(self, pos):
        stage = self.stage
        #######
        # Define positions from dictionary position input of the dict format {'x':5, etc},
        # and send GRBL code to move to give position
        # Movements will be executed in order received (use Pyton3.6 or later dicts).
        ######
        for axis in pos:
            stage.write(('G0 Z0 \n').encode('utf-8')) # Always move to Z=0 first.
            time.sleep(0.2)
            stage.write(('G0 '+axis.upper()+str(pos[axis])+' \n').encode('utf-8')) # Move code to GRBL, xy first
            time.sleep(0.2)
            grbl_out = stage.readline().decode('utf-8') # Wait for grbl response with carriage return
            logging.info('GRBL out:'+grbl_out)
            logging.info('Moved to '+axis.upper()+'='+str(pos[axis]))