#from tracemalloc import start
import numpy as np
from numba import njit, prange, types, typed, float64
from numba.experimental import jitclass
import os


@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

spec_ctcf = [
            ('sites',  types.int32[:]),
            ('probabilities', types.float32[:]),
            ('directions', types.int32[:]),
            ('avail_sites', types.boolean[:]),
            ('bound_lifetime', types.int_),
            ('unbound_lifetime', types.int_),
            ('smc_bound_lifetime', types.int_),
            ('bound', types.boolean),
            ('i', types.int_),
            ('random_numbers', types.float32[:]),
            ("position", types.int_),
            ('direction', types.int_),
            ('lifetime', float64),
            ('ctcf_id', types.int_),
            ('site_id', types.int_),
            ('bound_smc_id', types.int_)
            ]
@jitclass(spec_ctcf)
class CTCF():
    '''Class of chromatin binding protein, that can halt an SMC when encountering.
    Will bind and unbind to random sites drawn with weighted probabilites and associated directions according to an exponential decay with the bound/unbound lifetime.
    '''
    def __init__(self, sites: np.ndarray, probabilities: np.ndarray, directions: np.ndarray, avail_sites: np.ndarray, bound_lifetime: int, unbound_lifetime: int, smc_bound_lifetime: int, ctcf_id: int = 0):
        self.sites = sites #Array of site coordinates CTCF can occupy
        self.probabilities = probabilities #List of probabilities of binding each site (probabilities of all sites sum to 1)
        self.directions = directions #List of direction associated with each site.
        self.avail_sites = avail_sites
        self.bound_lifetime = bound_lifetime #seconds, From e.g.Hansen et al, 2017
        self.unbound_lifetime = unbound_lifetime #seconds
        self.smc_bound_lifetime = smc_bound_lifetime
        self.ctcf_id = ctcf_id
        self.i = 0
        self.random_numbers = np.random.random(10000).astype(np.float32)
        self.unbind() #Initialize in unbound state.
        self.unlink_from_site()

    def draw_random(self):
        self.i += 1
        if self.i == len(self.random_numbers)-1:
            self.i = 0
        return self.random_numbers[self.i]
    
    def update(self):
        #self.age += 1
        if self.bound_lifetime == 0: #only bind if lifetime > 0 (for CTCF-free simulation purposes)
            pass
        elif self.draw_random() < 1/self.lifetime:# self.age > self.lifetime: #(Un)binding event
            if self.bound:
                self.unbind()
            else:
                self.bind()
    
    def bind(self):
        if not np.any(self.avail_sites): #No available sites to bind to.
            pass
        else:
            self.site_id = int(rand_choice_nb(np.arange(self.sites.shape[0])[self.avail_sites], self.probabilities[self.avail_sites]/np.sum(self.probabilities[self.avail_sites]))) #Draw random (or weighted random) site.
            #if self.avail_sites[chosen_site] == True: #Check if chosen site is available
            # = chosen_site
        self.bound = True
        #self.age = 0
        self.position = self.sites[self.site_id]#int(np.random.choice(self.sites, p = self.probabilities)) #Draw a site weighted by the given probabilites.
        self.direction = self.directions[self.site_id] #Set the associated direction
        self.lifetime = self.bound_lifetime#float(np.random.exponential(self.bound_lifetime)) #Draw a bound lifetime from the lifetime distribution.
                    
    
    def unbind(self):
        self.bound = False
        #self.age = 0
        self.lifetime = self.unbound_lifetime#float(np.random.exponential(self.unbound_lifetime)) #Draw an unbound lifetime from the lifetime distribution.

    def unlink_from_site(self):
        self.bound_smc_id = -1
        self.site_id = -1
        self.position = -1
        self.direction = 0

    def bind_smc(self, bound_smc_id):
        self.bound_smc_id = bound_smc_id
        #self.age = 0
        self.lifetime = self.smc_bound_lifetime#float(np.random.exponential(self.smc_bound_lifetime)) #Draw a SMC bound lifetime from the lifetime distribution.


spec_smc = [
            ('N_beads',  types.int_),
            ('bound_lifetime', types.int_),
            ('unbound_lifetime', types.int_),
            ('CTCF_bound_lifetime', types.int_),
            ('loop_lifetime', types.int_),
            ('ext_rate', types.float64),
            ('bound', types.boolean),
            #('age', types.int_),
            ('i', types.int_),
            ('random_numbers', types.float32[:]),
            ('start_pos', types.int_),
            ('l_pos', types.int_),
            ('r_pos', types.int_),
            ('CTCF_bound', types.boolean),
            ('CTCF_bound_l', types.boolean),
            ('CTCF_bound_r', types.boolean),
            ('SMC_bound_l', types.boolean),
            ('SMC_bound_r', types.boolean),
            ('current_lifetime', float64),
            ('smc_id', types.int_)
            ]
            
@jitclass(spec_smc)
class SMC():
    def __init__(self, N_beads: int, bound_lifetime: int, unbound_lifetime: int, CTCF_bound_lifetime: int, extrusion_rate: float, loop_lifetime: int, smc_id: int = 0):
        self.ext_rate = extrusion_rate#int(np.round(1/extrusion_rate)) #ext_rate must be one (kb/s) or lower in this setup, gives interval between simulation steps that walk steps are made.
        self.N_beads = N_beads #The length of the simulated polymer.
        self.bound_lifetime = bound_lifetime
        self.unbound_lifetime = unbound_lifetime
        self.CTCF_bound_lifetime = CTCF_bound_lifetime
        self.loop_lifetime = loop_lifetime
        self.smc_id = smc_id
        self.unbind() #Initialize in unbound state.
        self.i = 0
        self.random_numbers = np.random.random(10000).astype(np.float32)
        #print(self.current_lifetime)

    def draw_random(self):
        self.i += 1
        if self.i == len(self.random_numbers)-1:
            self.i = 0
        return self.random_numbers[self.i]

    def update(self):
        #self.age += 1
        if not self.bound:# and (self.age > self.current_lifetime): #Binding event
            if self.bound_lifetime == 0: #only bind if lifetime > 0 (for SMC-free simulation purposes)
                pass
            elif self.draw_random() < 1/self.current_lifetime:
                self.bind()
        elif self.bound:# and (self.age > self.current_lifetime): #Unbinding event
            if self.draw_random() < 1/self.current_lifetime:
                self.unbind()
            else:
                self.walk()


    def bind(self):
        #Resets most parameters.
        self.start_pos = int(np.random.randint(0,self.N_beads-3)) #Starting position is random with both arms on polymer.
        self.l_pos = self.start_pos
        self.r_pos = self.start_pos+2
        self.bound = True
        #self.CTCF_bound = False
        self.CTCF_bound_l = False
        self.CTCF_bound_r = False
        self.SMC_bound_l = False
        self.SMC_bound_r = False
        self.current_lifetime = self.bound_lifetime#Draw a new bound lifetime upon binding.
    
    def unbind(self):
        #Resets most parameters.
        self.current_lifetime = self.unbound_lifetime#Draw a new unbound lifetime upon unbinding.
        self.bound = False 
        self.CTCF_bound_l = False
        self.CTCF_bound_r = False
        self.SMC_bound_l = False
        self.SMC_bound_r = False
    
    def bind_ctcf(self, side):
        self.current_lifetime = self.CTCF_bound_lifetime

        if side == 'left':
            self.CTCF_bound_l = True

        elif side == 'right':
            self.CTCF_bound_r = True


    def unbind_ctcf(self, side):
        if side == 'left':
            self.CTCF_bound_l = False
        elif side == 'right':
            self.CTCF_bound_r = False

        if not (self.CTCF_bound_l or self.CTCF_bound_r): #No longer CTCF bound.
            self.current_lifetime = self.bound_lifetime
    
    def walk(self):
        
        if not (self.CTCF_bound_l or self.SMC_bound_l):
            if self.draw_random() < self.ext_rate: #Only step if not CTCF/SMC bound on this side.
                self.l_pos -= 1
            
        if not (self.CTCF_bound_r or self.SMC_bound_r):
            if self.draw_random() < self.ext_rate: #Only step if not CTCF/SMC bound on this side. 
                self.r_pos += 1

        # if not self.CTCF_bound_r and self.CTCF_bound_l and (self.draw_random() < 1/self.loop_lifetime): #Loop release without unbinding
        #     self.r_pos = self.l_pos + 2

        # if not self.CTCF_bound_l and self.CTCF_bound_r and (self.draw_random() < 1/self.loop_lifetime): #Loop release without unbinding
        #     self.l_pos = self.r_pos - 2

        if (self.l_pos < 0) or (self.r_pos >= self.N_beads): #Unbind if SMC has stepped outside of polymer.
            self.unbind()

def init_SMC_sim(params):
    # Initialize the CTCF and SMC classes for the simulation with the chosen parameters.
    CTCFs = typed.List()#[]
    SMCs = typed.List()#[]
    for i in range(params['n_CTCF']):
        CTCFs.append(CTCF(sites = np.array(params['CTCF_sites']).astype(np.int32), 
                            probabilities = np.array(params['CTCF_site_probability']).astype(np.float32), 
                            directions = np.array(params['CTCF_site_direction']).astype(np.int32),
                            avail_sites = np.ones(len(params['CTCF_sites'])).astype(np.bool),
                            bound_lifetime = params['CTCF_bound_lifetime'], 
                            unbound_lifetime = params['CTCF_unbound_lifetime'],
                            smc_bound_lifetime= params['CTCF_SMC_bound_lifetime'],
                            ctcf_id=i))
    if params['SMC_types'] > 1:
        for i in range(params['n_SMC']):
            SMC_type = int(np.random.choice(params['SMC_types'], p=params['SMC_distribution']))
            SMCs.append(SMC(params['size_smc_sim'], 
                            bound_lifetime=params['SMC_bound_lifetime'][SMC_type],
                            unbound_lifetime=params['SMC_unbound_lifetime'][SMC_type],
                            CTCF_bound_lifetime=params['SMC_CTCF_bound_lifetime'][SMC_type],
                            loop_lifetime=params['loop_lifetime'][SMC_type],
                            extrusion_rate=params['extrusion_rate'][SMC_type],
                            smc_id = i))
    else:
        for i in range(params['n_SMC']):
            SMCs.append(SMC(params['size_smc_sim'], 
                            bound_lifetime=params['SMC_bound_lifetime'],
                            unbound_lifetime=params['SMC_unbound_lifetime'],
                            CTCF_bound_lifetime=params['SMC_CTCF_bound_lifetime'],
                            loop_lifetime=params['loop_lifetime'],
                            extrusion_rate=params['extrusion_rate'],
                            smc_id = i))
        
    return CTCFs, SMCs

@njit(parallel = False) #Sometimes segfaults if run in paralell...      
def update_SMC_sim(CTCFs: list, SMCs: list, SMC_crash: float = 0.0):
    #Runs every SMC simulation step.
    
    loop_pos = -np.ones((len(SMCs), 2)) #types.ListType(types.Tuple(types.uint, types.uint)) #Initialize empty list for gathering looping positions based on SMC positions.
    for i in range(len(SMCs)):
        s = SMCs[i]
        s.update() #Update all SMCs to let them bind/unbind or take a step.
        if s.bound: 

            if SMC_crash > 0: #If probability for SMCs crashing is >0, check for this.
                for j, other_s in enumerate(SMCs):
                    if other_s.bound:
                        if j == i:
                            pass
                        else:
                            if not s.SMC_bound_r and (s.r_pos == other_s.l_pos):
                                if np.random.random() < SMC_crash:
                                    s.SMC_bound_r = True
                                    other_s.SMC_bound_l = True
                            if not s.SMC_bound_l and (s.l_pos == other_s.r_pos):
                                if np.random.random() < SMC_crash:
                                    s.SMC_bound_l = True
                                    other_s.SMC_bound_r = True

            loop_pos[i, 0] = s.l_pos
            loop_pos[i, 1] = s.r_pos #Appends the left and right arm positions of each bound SMC as a loop.
        
    avail_sites = CTCFs[0].avail_sites
    for i in range(len(CTCFs)):
        c = CTCFs[i]
        c.avail_sites = avail_sites #Update available sites in each CTCF
        c.update() #Update CTCFs to let them bind/unbind.
        if c.bound:
            avail_sites[c.site_id] = False #If CTCF was bound after update step, make this site unavailable.
            if c.bound_smc_id < 0: #If CTCF not already bound to an SMC, check for this.
                for j in range(len(SMCs)):
                    s = SMCs[j]
                    if s.bound:
                        if (np.abs(s.r_pos - c.position) < 6) and (c.direction == -1):# and not s.CTCF_bound_l: #Check if it is bound at the SMC arm and has the correct orientation. Can bind to +/- 5 position (10kb capture)
                            c.bind_smc(bound_smc_id = s.smc_id)
                            s.bind_ctcf(side = 'right')
                            s.r_pos = c.position
                        elif (np.abs(s.l_pos - c.position) < 6) and (c.direction == 1):# and not s.CTCF_bound_r: # Same as above for other SMC arm.
                            c.bind_smc(bound_smc_id = s.smc_id)
                            s.bind_ctcf(side = 'left')
                            s.l_pos = c.position

        elif (c.site_id > -1) and not c.bound: #This means CTCF was unbound during the current update step.
            avail_sites[c.site_id] = True #Make this site available.
            if c.bound_smc_id > -1: #If CTCF was bound to an SMC.
                for j in range(len(SMCs)):
                    s = SMCs[j]
                    if (c.bound_smc_id == s.smc_id) and (s.bound):
                        if c.direction == -1:
                            s.unbind_ctcf(side='right') #Unbind CTCF from right side
                        elif c.direction == 1:
                            s.unbind_ctcf(side='left') #Unbind CTCF from left side

            c.unlink_from_site() #Unlink the CTCF from the site, resets all site ids.

    return loop_pos

def gen_random_steps(steps, step_size, N=None, start_pos = None, cumulative=True):
    if start_pos is None:
        start_pos = np.zeros((N, 3), dtype=np.float32)
    else:
        N = start_pos.shape[0]

    phi = 2*np.pi*np.random.rand(steps, N) #Get the polar angles between. Picks random angle between 0-deg 
    theta = 2*np.pi*np.random.rand(steps, N) #Get the azimuthal angels. Picks random angle between 0-deg
    r = step_size*np.random.randn(steps, N)
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    step_arr = np.stack([z,y,x])
    step_arr = np.moveaxis(step_arr,0,2)
    step_arr[0,:,:] = start_pos
    #print(step_arr.shape)

    if cumulative:
        cum_steps = np.cumsum(step_arr, axis=0)
        return cum_steps.astype(np.float32)
    else:
        return step_arr.astype(np.float32)

@njit
def gen_rouse_drift(steps, force_dict):
    old_pos = steps[0]
    T = steps.shape[0]
    for i in range(T-1):
        new_pos = apply_forces(old_pos, force_dict) + steps[int(i)+1]
        old_pos = new_pos.copy()
    return new_pos

@njit(fastmath=True, parallel = True)
def apply_forces(pos, force_dict):
    forces = np.zeros_like(pos, dtype=np.float32)
    for i in prange(pos.shape[0]):
        for k, v in force_dict[i].items():
            r = pos[i]-pos[k]
            r_euc = np.sqrt(np.sum(r**2))
            if r_euc < 0.025:
                forces[i]+= (0.025-r_euc) * r/r_euc
            else:
                forces[i]+= -v*r
    #print(forces)
    return pos + forces

@njit
def gen_forces_dict(N, k=1, loop_pos = None, loop_force = 1):
    
    force_dict = {i:{i-1:k, i+1:k} for i in range(N)}
    force_dict[0] = {1:k}
    force_dict[N-1] = {N-2:k}

    if loop_pos is not None:
        #if isinstance(loop_force, int) or isinstance(loop_force, float):
        #    loop_force = [loop_force]*len(loop_pos)
        for (a, b) in loop_pos:
            force_dict[a][b] = loop_force*k
            force_dict[b][a] = loop_force*k

    return force_dict

def run_SMC_sim(params, run_id = 0):
    '''Function for running SMC simulation with boundary factors (e.g. CTCF).
    Parameters for simulation set in param dict (see notebook/docs on defining the parameters)
    Saves all SMC positions.

    Args:
        params (dict): Dict of all parameters for running simulation
        run_id (int, optional): If running function in parelell, define which run this is (e.g. on HTC cluster). Defaults to 0.
    '''
    CTCF_props = []
    SMC_props = []
    res_loop_pos = np.zeros((params['smc_steps'], params['n_SMC'], 2), dtype=np.int64)
    CTCFs, SMCs = init_SMC_sim(params)
    for i in range(params['smc_init_steps']):
        update_SMC_sim(CTCFs, SMCs, params['SMC_crash'])
    for j in range(params['smc_steps']):
        loop_pos = update_SMC_sim(CTCFs, SMCs, params['SMC_crash'])
        res_loop_pos[j] = loop_pos.copy()
        CTCF_props.append([(int(c.bound), c.position) for c in CTCFs])
        SMC_props.append([(int(s.bound), s.bound_lifetime, s.current_lifetime) for s in SMCs])
    np.save(params['out_path']+os.sep+'CTCF_props_'+str(run_id)+'.npy', np.array(CTCF_props))
    np.save(params['out_path']+os.sep+'SMC_props_'+str(run_id)+'.npy', np.array(SMC_props))
    np.save(params['out_path']+os.sep+'SMC_pos_'+str(run_id)+'.npy', res_loop_pos)
    print('SMC simulation done.')

def run_SMC_sim_random_halt(params: dict, run_id: int = 0):
    '''Function for running SMC simulation with boundary factors (e.g. CTCF), that will run until a random timepoint and save the final state.
    Parameters for simulation set in param dict (see notebook/docs on defining the parameters)
    A second set of parameters can optionally be provided if a parameter change after the initialization run is desired.
    Saves certain properties of the simulated molecules and the SMC positions in the final state.

    Args:
        params (dict): Dict of all parameters for running simulation
        run_id (int, optional): If running function in parelell, define which run this is (e.g. on HTC cluster). Defaults to 0.
        params_updated (dict, optional): Optional paramater dict if change is needed for running step. Defaults to None.
    '''

    CTCF_props = []
    SMC_props = []
    res_loop_pos = np.zeros((params['smc_steps'], params['n_SMC'], 2), dtype=np.int64)
    for i in range(params['smc_steps']):
        CTCFs, SMCs = init_SMC_sim(params)
        for j in range(params['smc_init_steps']):
            update_SMC_sim(CTCFs, SMCs, params['SMC_crash'])
        
        for k in range(np.random.randint(0,params['smc_init_steps'])):
            loop_pos = update_SMC_sim(CTCFs, SMCs, params['SMC_crash'])

        CTCF_props.append([(int(c.bound), c.position) for c in CTCFs])
        SMC_props.append([(int(s.bound), s.bound_lifetime, s.current_lifetime) for s in SMCs])
        res_loop_pos[i] = loop_pos.copy()
    np.save(params['out_path']+os.sep+'CTCF_props_'+str(run_id)+'.npy', np.array(CTCF_props))
    np.save(params['out_path']+os.sep+'SMC_props_'+str(run_id)+'.npy', np.array(SMC_props))
    np.save(params['out_path']+os.sep+'SMC_pos_'+str(run_id)+'.npy', res_loop_pos)
    print('SMC simulation done.')

def run_rouse_SMC_sim(params, run_id = 0):
    res = np.lib.format.open_memmap(params['out_path']+os.sep+'sim_res_'+str(run_id)+'.npy', dtype = np.float32, shape = (params['smc_steps'],params['size_polymer_sim'],3), mode='w+')

    start_pos = gen_random_steps(steps = params['size_polymer_sim'], step_size=np.sqrt(6*params['D']), N = 1, cumulative= True)[:,0,:]
    all_loops = (np.load(params['out_path']+os.sep+'SMC_pos_'+str(run_id)+'.npy') - params['smc_pad_size']//1000).astype(np.int64)
    loop_list = [all_loops[i][np.all(all_loops[i]>0, axis = 1) & np.all(all_loops[i]<params['size_polymer_sim'], axis=1)] for i in range(all_loops.shape[0])]
    for i in range(params['smc_steps']+1):
        if i == 0:
            force_dict = gen_forces_dict(N = params['size_polymer_sim'], k = params['k'], loop_pos=None, loop_force=params['loop_force'])
            random_steps = gen_random_steps(steps=params['polymer_init_steps'], step_size= np.sqrt(6*params['D']), N=None, start_pos = start_pos, cumulative=False)
            rw = gen_rouse_drift(random_steps, force_dict)
            
            force_dict = gen_forces_dict(N = params['size_polymer_sim'], k = params['k'], loop_pos=loop_list[i], loop_force=params['loop_force'])
            random_steps = gen_random_steps(steps=params['polymer_init_steps'], step_size= np.sqrt(6*params['D']), N=None, start_pos = rw, cumulative=False)
            rw = gen_rouse_drift(random_steps, force_dict)

        else:
            force_dict = gen_forces_dict(N = params['size_polymer_sim'], k = params['k'], loop_pos=loop_list[i-1], loop_force=params['loop_force'])
            random_steps = gen_random_steps(steps=params['steps_per_smc_step'], step_size= np.sqrt(6*params['D']), N=None, start_pos = rw, cumulative=False)
            rw = gen_rouse_drift(random_steps, force_dict)
            res[i-1] = rw
            res.flush()

    print('Rouse SMC simulation done.')

def run_single_rouse_sim(params, run_id=0):
    res = np.lib.format.open_memmap(params['out_path']+os.sep+'sim_res_'+str(run_id)+'.npy', dtype = np.float32, shape = (params['smc_steps'],params['N_beads'],3), mode='w+')
    res_loop_pos = []
    start_pos = gen_random_steps(steps = params['N_beads'], step_size=np.sqrt(6*params['D']), N = 1, cumulative= True)[:,0,:]
    CTCFs, SMCs = init_SMC_sim(params)
    loop_pos = None
    for j in range(params['smc_steps']+1):
        force_dict = gen_forces_dict(N = params['N_beads'], k = params['k'], loop_pos=loop_pos, loop_force=params['loop_force'])#((1,11),) loop_force=10)
        if j == 0:
            random_steps = gen_random_steps(steps=int(1e4), step_size= np.sqrt(6*params['D']), N=None, start_pos = start_pos, cumulative=False)
            rw = gen_rouse_drift(random_steps, force_dict)
        else:
            random_steps = gen_random_steps(steps=params['steps_per_smc_step'], step_size= np.sqrt(6*params['D']), N=None, start_pos = rw, cumulative=False)
            rw = gen_rouse_drift(random_steps, force_dict)
            res[j-1] = rw
            loop_pos = update_SMC_sim(CTCFs, SMCs, params['SMC_crash'], params['loop_override'], params['loop_override_freq'])
            res_loop_pos.append(loop_pos)
            #start_pos = rw[-1].copy()
    #np.save(params['out_path']+os.sep+'sim_res_'+str(id)+'.npy',np.stack(res))
            res.flush()
    arr = np.zeros((len(res_loop_pos), params['N_beads']))
    for row, i in enumerate(res_loop_pos):
        try:
            for j in i:
                for k in j:
                    arr[row,k] = 1
        except TypeError:
            pass
    np.save(params['out_path']+os.sep+'SMC_pos_'+str(run_id)+'.npy',arr)
    

def run_multiple_rouse_sim(params, repeats = 1):
    for i in range(repeats):
        run_single_rouse_sim(params, id = i) 
    print("Simulation complete.")