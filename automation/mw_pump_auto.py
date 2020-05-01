import serial
import yaml
import logging
import time
import json
#import pyautogui
#import pygetwindow as gw

######
# Before running, ensure correct COM ports are set in yaml file.
# To run first execute: p, s = initialize() to start pump and stage.
# 
#
######

def initialize():
    
    #Initialize logging system, and print messages to console as well.
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(logging.getLogger().handlers) < 2:
        logging.getLogger().addHandler(logging.StreamHandler())
    
    config=load_config()
    pump=initialize_pump(config['pump_COM'])
    stage=initialize_grbl(config['stage_COM'])
    
    return pump, stage, config

def load_config(config_file='config.yml'):
    #Open config file and return config variable form yaml file
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def update_status(new_data, filename='status.json'):
    #Update selected field(s) in status.json file. new_data should be dictionary.
    with open(filename,'r') as f:
        data=json.load(f)
    for k,v in new_data.items():
        data[k]=new_data[k]
    with open(filename, 'w') as f:
        json.dump(data, f)
    logging.info('Status updated to: '+str(data))

def read_status(filename='status.json'):
    #Read out the status.json file.
    with open(filename,'r') as f:
        status=json.load(f)
    return status

def set_well(well):
    #Convenience function for setting current well with a string.
    update_status({'current_well': well})

def set_command(command):
    #Convenience function for setting current command with a string.
    update_status({'command':command})
        
def pause(sleep_time):
    #Set and run a pause step, and log.
    time.sleep(int(sleep_time))
    logging.info('Paused for '+str(sleep_time))

def initialize_pump(pump_port): #Pump communication initialize
    try:
        pump=serial.Serial(port=pump_port,timeout=3)  # open serial port
        logging.info('Pump connection established on '+pump.name)
        return pump
    except:
        logging.error('Pump connection failed or already initialized.')
    # # check which port was really used

def pump_cycle(pump): #Pump cycle
    pump.write(b'start\r')     # Start pump cycle
    pump_out=pump.readline().decode('utf-8')          # Read pump output
    logging.info('Started pump cycle: '+pump_out) 

def initialize_grbl(stage_port):
    s = serial.Serial(port=stage_port,baudrate=115200) # open grbl serial port
    s.write(("\r\n\r\n").encode('utf-8')) # Wake up grbl
    time.sleep(2)   # Wait for grbl to initialize
    s.flushInput()  # Flush startup text in serial input
    #s.write(('$21=1 \n').encode('utf-8')) # enable hard limits
    #s.write(('$H \n').encode('utf-8')) # tell grbl to find zero 
    #grbl_out = s.readline() # Wait for grbl response with carriage return
    s.write(('? \n').encode('utf-8')) # Request machine status
    grbl_out = s.readline().decode('utf-8') # Wait for grbl response with carriage return
    logging.info('GRBL initialized:' +grbl_out)
    #response=response.replace(":",","); response=response.replace(">",""); response=response.replace("<","")
    #a_list=response.split(",")
    #print(a_list)
    return s

def zero_stage(stage):
    stage.write(('G10 L20 P0 X0 Y0 Z0 \n').encode('utf-8'))
    grbl_out = stage.readline().decode('utf-8')
    logging.info('Current position set to zero: '+grbl_out)

def check_stage(stage):
    stage.flushInput()
    stage.write(('?\n\r').encode('utf-8'))
    time.sleep(0.2)
    grbl_out = stage.readline().decode('utf-8')
    return grbl_out[1:4]

def move_stage(pos, stage):
    #######
    # Define positions from dictionary position input of the format {'x':5, etc},
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
        
def wp_coord_list():
    # Generate coordinate list for whole and selected wells of a 96-well plate.
    # Also adjusts for a rotated plate by using measured top left and bottom right positions from
    # config file.

    config=load_config()
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
    
    # Make json files that store all wellplate coordinates and selected wellplate coordinates.
    with open('all_coords.json', 'w') as file:
        json.dump(all_coords, file)
    with open('sel_coords.json', 'w') as file:
        json.dump(sel_coords, file)


def single_cycle(s, p):
    '''
    Runs a single sequence as defined in the config file.
    '''
    
    config=load_config()
    wp_coord_list()
    with open('all_coords.json', 'r') as file:  # Use file generated by wp_coord_list() here.
        all_coords=json.load(file)
    with open('sel_coords.json', 'r') as file:
        sel_coords=json.load(file)
        
    sel_coord_list=list(sel_coords.keys())
    
    for a in config['sequence']:                # Sequence is defined as list in config file.
        action=list(a.keys())[0]
        param=list(a.values())[0]
        if action == 'probe' and param == 'wp': # Run for wellplate position defined in status json file.
            current_pos=read_status()['current_well']
            coords=all_coords[current_pos]
            move_stage(coords, s)
            while check_stage(s) != 'Idl': #Wait until move is done before proceeding.
                time.sleep(2)
            logging.info('Probe at '+str(current_pos))
            
            i=sel_coord_list.index(current_pos)
            try:
                next_well=sel_coord_list[i+1]
            except IndexError:
                logging.info('Last well reached')
                next_well='Last'
            update_status({'current_well': next_well})
            
        elif action == 'probe' and param != 'wp':   # For all position outside well plates, such as reservoirs.
            try: 
                coords=config[param]
                move_stage(coords, s)
                while check_stage(s) != 'Idl': #Wait until move is done before proceeding.
                    time.sleep(2)
                logging.info('probe at '+str(param))
            except KeyError:
                logging.error('Invalid probe position: '+str(param))
        elif action == 'pump':
            pump_cycle(p)
            time.sleep(int(param))
            logging.info('Pump cycle completed for '+str(param))
        elif action == 'pause':
            pause(param)
        elif action == 'image':
            set_command('image')
            time.sleep(2)
            while read_status()['command']=='image' or read_status()['command']=='imaging':
                time.sleep(5)
        else:
            logging.error('Unrecognized sequence command')       

def det_num_cycles():
    '''
    Helper function to check number of cycles based config setting.
    If an int, runs that number of cycles. If 'all' counts how many cycles is 
    needed to probe all wellplate positions.
    
    Output:
        num_cycles: Int with number of cycles.

    '''
    
    config=load_config()
    
    if config['num_cycles'] == 'all':
        wells_seq = 0
        for seq in config['sequence']:
            for val in seq.values():
                if val == 'wp':
                    wells_seq += 1
        with open('sel_coords.json', 'r') as file:
            sel_coords=json.load(file)
        wells=len(sel_coords.keys())
        num_cycles=wells//wells_seq
        
    else:
        num_cycles=int(config['num_cycles'])
    return num_cycles

def wp_cycle(s, p, restart=True):
    '''
    Runs a full cycle of sequences for all selected well positions.
    Refreshes coordinate list based on config file.
    Runs number of cycles determined by det_num_cycles helper function.
    Restart flag determines if start from current well in status or first well.
    '''
    
    config=load_config()
    wp_coord_list()
    num_cycles=det_num_cycles()
    
    if restart:
        update_status({'current_well': config['first_probe']})
    for cycle in range(num_cycles): 
        single_cycle(s, p)
