'''
This module contains the classes necessary to run a Part lifecycle simulation. These are
attached to the simpy environment and evolve parts through the warehouse, machines, repair
and preprocessing steps (as well as creation by the Factory).
'''

import logging
import itertools
import random

import numpy as np
import scipy

class Machine:

    '''
    The Machine object drives the simulation forward. It requests Parts, evolves through time,
    sends them back, requests new parts, and continues the process indefinitely.

    Args:
        env (simpy.Environment): The environment in which the simulation is run.
        warehouse (simpy.Store): The warehouse storing all Parts ready for use.
        repair_storage (simpy.Store): The warehouse storing all Parts that need to be evaluated 
            for repair or scrap, and repaired if necessary.
        mach_type (str): The number of times a part has gone through a machine.
        init_partset (PartSet): The initial PartSet generated during setup.
        init_time (int): The remaining time that the initial PartSet can be used.
        pred_scrap_params (dict): Parameters describing actual scrap rate.
        pred_scrap_params (dict): Parameters describing predicted scrap rate distribution.

    Attributes:
        id (int): Machine ID, for debugging purposes only.
        env (simpy.Environment): stores env arg.
        warehouse (simpy.Store): stores warehouse arg.
        repair_storage (simpy.Store): stores repair_storage arg.
        mach_type (str): stores mach_type arg.
        n_parts (int): stores n_parts arg.
        parts (PartSet): stores current PartSet in machine.
        curr_down_start (int): Stores the time at which the current downtime started. Needed if the
            downtime continues until the end of the simulation.
        downtime (int): records accumulated downtime due to lack of available Parts.
        run_duration (int): stores the number of times steps that a set of Parts is used 
            in the machine.
        maint_duration (int): stores the downtime for a machine while its Parts are changed.
        part_inv_wait_time (int): stores the waiting interval if the warehouse does not have enough Parts.
        init_time (int): stores init_time arg.
        partset_change_date (int): simulation step number at which PartSet will be replaced.
        act_scrap_params (dict): Parameters describing actual scrap rate Beta distribution.
        pred_scrap_params (dict): Parameters describing predicted scrap rate Beta distribution.
        act_scrap_rate (float): Scrap rate for current PartSet
        pred_scrap_params (dict): Parameters describing predicted Beta distribution scrap rate for current PartSet.
        
    '''
    id_iter = itertools.count()
    def __init__(self, env, warehouse, repair_storage, part_run_dur, init_partset = None, init_time = None, act_scrap_params=None, pred_scrap_params=None):

        self.mach_id = next(Machine.id_iter)
        self.env = env
        self.warehouse = warehouse
        self.repair_storage = repair_storage
        if init_partset is None:
            self.partset = None
        else:
            self.partset = init_partset
        self.init_time = init_time
        
        self.curr_down_start = None
        self.downtime = 0 # record downtime due to lack of available parts

        self.run_duration = part_run_dur # days
        self.part_inv_wait_time = 7 # days
        
        self.act_scrap_params = act_scrap_params
        self.pred_scrap_params = pred_scrap_params
        
        self.act_scrap_rate = None
        self.scrap_rate_detr = None
        self.partset_change_date =  None

        # Start machine
        self.action = self.env.process(self.run())

    def run(self):

        # Infinite loop to run machine
        while True:
            # Run the first set of parts according to the initial conditions
            if self.init_time is not None:
                # Run with the first parts
                logging.debug('%s: Machine %s running with initial PartSet %s',self.env.now,self.mach_id,self.partset)
                self.calc_scrap()
                self.partset_change_date = self.env.now + (self.run_duration-self.init_time)
                yield self.env.timeout(self.run_duration-self.init_time)

                # Return parts to warehouse
                self.partset.inc_runs()
                logging.debug('%s: Machine %s sending initial PartSet %s to repair',self.env.now,self.mach_id,self.partset)
                
                yield self.repair_storage.put(self.partset)
                self.partset = None

                # Cancel out init time to continue in main loop.
                self.init_time = None

            # Get new PartSet
            self.curr_down_start = self.env.now
            self.partset = yield self.warehouse.get()
            self.downtime += (self.env.now-self.curr_down_start) # Record sim waiting time as Machine downtime
            self.curr_down_start = None
            
            logging.debug('%s: Machine %s getting PartSet %s',self.env.now,self.mach_id,self.partset)
                                                           
            # Set class variables related to scrap rate prediction and time.
            self.calc_scrap()
            self.partset_change_date = self.env.now + self.run_duration
            # Run the machine
            yield self.env.timeout(self.run_duration)

            # Set scrap rate and send PartSet to repair storage
            self.partset.inc_runs()
            yield self.repair_storage.put(self.partset)
            logging.debug('%s: Machine %s sending PartSet %s to repair',self.env.now,self.mach_id,self.partset)

            self.partset = None
   
    def calc_scrap(self):
        '''
        Evaluates the PartSet's scrap rate. Called at the start of one run through a Machine, because
        the predicted scrap rate is used for planning orders. Sets the to_scrap boolean flag in each 
        Part, which is one of the criteria used by the RepairShop to scrap parts.

        Args:
            None

        Returns:
            pred_scrap_num (int): The predicted number of Parts to scrap according to assumed distribution
                (rather than actual distribution), or None if no scrap rate parameters are provided.
        '''
        
        # Update relevant scrap parameters in case anything changed
        self.set_scrap_params()
        self.calc_scrap_detr()
        
        # Set subset of parts to be scrapped if act_scrap_rate is non-None
        if self.act_scrap_rate is not None:
            scrapped_parts = int(self.act_scrap_rate*len(self.partset.parts))
            to_scrap =  random.sample(self.partset.parts, scrapped_parts)
            for part in to_scrap:
                part.to_scrap = True
                
    def set_scrap_params(self):
        '''
        Sets the actual scrap rate given the input parameters at initialization. This function helps modify the
        scrap rate on a per-set basis, can be modified over time by changing the parameters, and makes performing
        the selected studies easier.

        Args:
            None

        Returns:
            None
        '''
            
        # Set scrap parameters of PartSet depending variables passed
        if isinstance(self.act_scrap_params, dict):
            if 'const' in self.act_scrap_params.keys():
                self.act_scrap_rate = self.act_scrap_params['const']
            elif 'a' in self.act_scrap_params.keys() and 'b' in self.act_scrap_params.keys():
                self.act_scrap_rate = scipy.stats.beta.rvs(self.act_scrap_params['a'], self.act_scrap_params['b'])
            else:
                raise ValueError(f'act_scrap_rate does not contain valid keys: {self.act_scrap_rate.keys()}')
        else:
            self.act_scrap_rate = None     
            
        if isinstance(self.pred_scrap_params, dict):
            # If std dev is passed to parameterize the predicted distribution,
            # generate a dist based on the drawn actual mean.
            if 'const' in self.act_scrap_params.keys():
                self.pred_scrap_params['const'] = self.pred_scrap_params['const']
            elif 's' in self.pred_scrap_params.keys():
                m = scipy.stats.uniform.rvs()*0.9+0.1
                n = m*(1-m)/self.pred_scrap_params['s']

                a = m*n
                b = (1-m)*n

                self.act_scrap_rate = m                                   
                self.pred_scrap_params = {'a':a,'b':b}
            # 
            elif 'a' in self.pred_scrap_params.keys() and 'b' in self.pred_scrap_params.keys():
                self.pred_scrap_params = {'a':self.pred_scrap_params['a'],'b':self.pred_scrap_params['b']}
            else:
                raise ValueError(f'pred_scrap_params does not contain set of valid keys: {self.pred_scrap_params.keys()}')                        
        else: 
            self.pred_scrap_params = None

    def calc_scrap_detr(self):
        '''
        Evaluates the PartSet's deterministic scrap rate. This is the number of Parts that are at
        max_runs-1 of their lifetime limit before starting this machine's run.

        Args:
            None

        Returns:
            detr_scrap_num (int): The number of Parts that will be scrapped deterministically because
                they have reached their maximum number of runs through a Machine.
        '''
        
        scrap_rate_detr = 0
        for part in self.partset.parts:
            if part.n_runs == part.max_runs-1:
                scrap_rate_detr += 1
        self.scrap_rate_detr = scrap_rate_detr
                    

class Factory:
    '''
    The Factory object contains variables to ensure all parts have a unique identifier, and
    store information regarding default part number and lifetime parameters.


    Args:
        env (simpy.Environment): The environment in which the simulation is run.
        prepro_storage (simpy.Store): The warehouse storing all Parts before they go to preprocessing.
        lead_time (int): The number of times a part has gone through a machine
        max_runs (int): The maximum number of times this part can go through a machine.
        early_fail_dist (numpy.random.Generator): parameterized random number generator for early failures (max_runs = 1) 

    Attributes:
        n_parts_made (int): Tracks the number of parts made. 
        Rest same as args.
    '''
    def __init__(self, env, prepro_storage,  lead_time, max_runs, sim_length, early_fail_dist=None):
        
        self.env = env
        self.prepro_storage = prepro_storage
        self.lead_time = lead_time
        
        # Order array: add a big offset for orders made near the end of the sim time. Exact extra time isn't known
        # a priori in case the stochastic lead time, so add a big buffer.
        self.orders = np.zeros(sim_length+1000) 
        self.n_parts_made = 0
        self.max_runs = max_runs   
        self.early_fail_dist = early_fail_dist
        
        self.env.process(self.run())
            
    def order_parts(self, n_parts, max_runs=None):

        '''
        Order new Parts given number of parts and their lifetime. Parts
        are sent to the preprocessing warehouse to await a processing step before being
        service-ready.
        
        Args:
            n_parts (int): Number of requested parts.
            max_runs (int): Lifetime of the parts

        Returns:
            True
        '''

        # Set default value as class variable. Can't do this in the declaration due to self scope.
        if max_runs is None:
            max_runs = self.max_runs

        self.n_parts_made += n_parts # Part count incremented before delivery as it's used for cost calculation.

        # Create parts according to lead time, random or deterministic
        if callable(self.lead_time):    
            lead_time_inst = int(self.lead_time())
        else:
            lead_time_inst = self.lead_time
        self.orders[int(self.env.now)+lead_time_inst] += n_parts
            
    def run(self):
        
        while True:  
            # Send parts to the preprocessing warehouse.
            for _ in range(int(self.orders[int(self.env.now)])):

                if self.early_fail_dist is not None:
                    if self.early_fail_dist() == 1:
                        new_part = Part(1)
                    else:
                        new_part = Part(self.max_runs)
                else:
                    new_part = Part(self.max_runs)

                logging.debug('%s: Factory delivering part %s',self.env.now,new_part)
                yield self.prepro_storage.put(new_part)
            
            yield self.env.timeout(1) # deliver orders every day
            

            
class RepairShop:
    '''
    A repair shop runs in an infinite loop, waiting for PartSets from the repair shop storage
    and starting as soon as one becomes available. If the part has exhausted its useful life,
    it goes to scrap; otherwise, it is repaired and sent to preprocessing storage.
    
    Args:
        env (simpy.Environment): The environment in which the simulation is run.
        preproc_storage (simpy.Store): The warehouse storing all Parts waiting to be preprocessed.
        repair_storage (simpy.Store): The warehouse storing all Parts that need to 
            be evaluated for repair or scrap, and repaired if necessary.
        scrap (simpy.Store): The object storing all Parts that have been scrapped.
        throughput (int): The number of Parts that can be processed per timestep. Assumed >=1.
    Attributes:
        env (simpy.Environment): Stores argument.
        preproc_storage (simpy.Store): Stores argument.
        rep_storage (simpy.Store): Stores argument.
        scrap (simpy.Store): Stores argument.
        throughput (int): Stores argument.
        rep_backlog (list): The number of Parts that has been split out of a PartSet and are waiting to be repaired.
        repair_time (int): Hardcoded to 1, needs to be changed if repair time for one Part is >1 timestep.
    '''

    def __init__(self, env, preproc_storage, repair_storage, scrap, throughput):

        self.env = env
        self.prepro_storage = preproc_storage
        self.rep_storage = repair_storage
        self.scrap = scrap
        self.rep_backlog = []

        self.throughput = throughput
        self.repair_time = 1

        # Start repair shop
        self.action = self.env.process(self.run())

    def run(self):
        '''
        The main method running in the simpy environment. Evolves the behaviour of the RepairShop in time.

        Args:
            None

        Returns:
            None
        '''
        while True:
            yield self.env.timeout(self.repair_time) # Wait one step before processing again.

            # If all parts have been processed, split another PartSet
            if len(self.rep_backlog) == 0:
                partset_to_rep = yield self.rep_storage.get()
                # Put in scrap rate application here
                # Split partset into parts
                logging.debug('%s: Repair shop split PartSet %s',self.env.now,partset_to_rep)

                self.rep_backlog.extend(partset_to_rep.parts)

            n_to_repair = len(self.rep_backlog)
            for _ in range(min(self.throughput, n_to_repair)):                 
                to_rep = self.rep_backlog.pop()
                logging.debug('%s: Repair checking part %s with %s runs',self.env.now,to_rep,to_rep.n_runs)

                # Scrap the part if it has run its life
                if to_rep.n_runs >= to_rep.max_runs or to_rep.to_scrap:
                    yield self.scrap.put(to_rep)
                    logging.debug('%s: Repair shop scrapped Part %s',self.env.now,to_rep)

                # Otherwise repair it
                else:
                    logging.debug(f'%s: Repair shop repairing Part %s',self.env.now,to_rep)

                    yield self.prepro_storage.put(to_rep)


class PreprocessShop:
    '''
    A preprocess shop runs in an infinite loop, waiting for parts from the preprocessing shop storage
    and starting as soon as one becomes available. Once a full PartSet is complete, it is set to the
    warehouse.

    Args:
        env (simpy.Environment): The environment in which the simulation is run.
        prepro_storage (simpy.Store): The warehouse storing all Parts waiting to be preprocessed.
        repair_storage (simpy.Store): The warehouse storing all Parts that need to be evaluated
            for repair or scrap, and repaired if necessary.
        scrap (simpy.Store): The object storing all Parts that have been scrapped.
        throughput (int): The number of Parts that can be processed per timestep. Assumed >=1.
    Attributes:
        env (simpy.Environment): Stores argument.
        preproc_storage (simpy.Store): Stores argument.
        rep_storage (simpy.Store): Stores argument.
        scrap (simpy.Store): Stores argument.
        throughput (int): Stores argument.
        waiting_for_set (list): List of Parts waiting for enough Parts to be available to make 
            PartSet to send to the warehouse.
        preprocess_time (int): Hardcoded to 1, needs to be changed if repair time for one Part is >1 timestep.

    '''
    def __init__(self, env, warehouse, prepro_storage, throughput, n_parts_per_mach):

        self.env = env
        self.warehouse = warehouse
        self.prepro_storage = prepro_storage
        self.throughput = throughput
        self.n_parts_per_mach = n_parts_per_mach

        self.active_parts = []
        self.waiting_for_set = []

        self.preprocess_time = 1

        # Start repair shop
        self.action = self.env.process(self.run())

    def run(self):
        '''
        The main method running in the simpy environment. Evolves the behaviour of the PreprocessShop in time.

        Args:
            None

        Returns:
            None

        '''
        while True:
            # Get the list of parts in preprocessing storage
            pp_backlog = self.prepro_storage.items

            # Get the maximum parts per step or all backlogged parts, whichever is smaller.
            for _ in range(min(self.throughput, len(pp_backlog))):  
                to_prepro = yield self.prepro_storage.get()
                self.active_parts.append(to_prepro)
            logging.debug('%s: Prepro got %s parts',self.env.now,min(self.throughput, len(pp_backlog)))
            
            # Preprocessing time
            yield self.env.timeout(self.preprocess_time)

            # Check if enough parts are available to create a set; if so, send it to the warehouse.
            while len(self.active_parts)>=self.n_parts_per_mach:
                # Create PartSet from list of preprocessed parts, remove from internal storage
                to_send = PartSet(self.active_parts[:self.n_parts_per_mach])
                del self.active_parts[:self.n_parts_per_mach]

                yield self.warehouse.put(to_send)
                logging.debug('%s: Prepro created partset %s',self.env.now,to_send)
                        
# Define an object for each part.
class Part:
    '''
    The Part object contains variables to uniquely identify a part and to track its life.

    Args:

        n_runs (int): The number of times a part has gone through a machine
        max_runs (int): The maximum number of times this part can go through a machine.

    Attributes:
        serial_num (str): The part's serial number (unique identifier)
        to_scrap (bool): Flag set by machine to determine whether part must be scrapped after 
            current usage (regardless of life).
        Rest same as args.
    '''
    id_iter = itertools.count()
    def __init__(self, max_runs):

        self.serial_num = str(next(Part.id_iter)).zfill(6)
        self.n_runs = 0
        self.max_runs = max_runs   
        self.to_scrap = False

    def __repr__(self):
        return f"<Part: {self.serial_num}>"

# Define an object to gather individual parts into a set.
class PartSet:
    '''
    The PartSet object organizes a set of parts that goes into a Machine. It is used to enforce
    that a machine can only request a whole set (rather than accumulating individual valid parts
    until it has enough to run), and to perform scrap rate calculations.

    Args:
        parts (list of Parts): Parts that compose the PartSet

    Attributes:
        id (int): PartSet ID, for debugging purposes only
        parts (list): Holds the Parts that compose the PartSet.
    '''
    id_iter = itertools.count()
    def __init__(self, parts):
        self.partset_id = next(PartSet.id_iter)
        self.parts = parts

    # Increment number of runs for all parts
    def inc_runs(self):
        '''
        Increments every Part in the PartSet after it has been run through a Machine.

        Args:
            None

        Returns:
            None
        '''
        for part in self.parts:
            part.n_runs += 1
                     
    def __repr__(self):
        return f"<PartSet {self.partset_id} containing: {[x.serial_num for x in self.parts]}>"
    
    
def setup(env, warehouse, preproc_storage, repair_storage, scrap, params):
    '''
    Setup classes for simulation. Required to populate machines with parts and track them, and
    adjust the Factory so that the unique identifiers are created accordingly.

    Args:
        env (simpy.Environment): The environment in which the simulation is run.
        warehouse (simpy.Store): The warehouse storing all Parts ready for use.
        preproc_storage (simpy.Store): The warehouse storing all Parts that need to be preprocessed to be made ready for use.
        repair_storage (simpy.Store): The warehouse storing all Parts awaiting repairs.
        scrap (simpy.Store): The warehouse storing all scrapped Parts.
        params (dict): parameters to specific the simulation initialization.

    Returns:
        machines (list): list of Machines initialized with list of Parts init_parts.
        factory (Factory): Factory initialized with correct starting Part count.
    '''
    rng = np.random.default_rng()

    # Initialize parts and then place them into a Machine
    machines = []
    for i in range(params['n_machine']):
        init_mach_partset = PartSet([Part(params['part_n_runs_life']) for j in range(params['n_parts_per_mach'])])

        if params['rand_start']:
            init_time = rng.integers(1,params['part_run_dur']+1) # Generate uniformly distributed start time for the parts in the machine
        else:
            init_time = 0

        machines.append(Machine(env, warehouse, repair_storage,params['part_run_dur'], init_mach_partset, init_time))
        logging.debug('setup: send initial PartSet %s to Machine %s',init_mach_partset,machines[-1].mach_id)

    # Initialize spare parts for the warehouse    
    init_ware_partsets = []
    for i in range(params['n_partsets_warehouse']):
        init_ware_partsets.append(PartSet([Part(params['part_n_runs_life']) for i in range(params['n_parts_per_mach'])]))
        logging.debug(f'setup: create initial PartSet {init_ware_partsets[-1]} for warehouse')

    # Add them to the warehouse
    warehouse.items = init_ware_partsets

    # Create the factory, initialize the shops
    factory = Factory(env, preproc_storage, params['fact_lead_time'], params['part_n_runs_life'], params['sim_length'])
    prepro_shop = PreprocessShop(env, warehouse, preproc_storage, params['prepro_through'], params['n_parts_per_mach'])
    rep_shop = RepairShop(env, preproc_storage, repair_storage, scrap, params['rep_through'])

    return machines, factory


def monitor_warehouse(env, warehouse, warehouse_data):
    '''
    Monitor the contents at the warehouse at every step in time. Used to
    quantify the opportunity loss of the idle parts.

    Args:
        env (simpy.Environment): The environment in which the simulation is run.
        warehouse (simpy.Store): The warehouse storing all Parts.
        warehouse_data  (list): list to which warehouse contents information is appended.
    '''
    while True:
        warehouse_data.append(len(warehouse.items))
        yield env.timeout(1) # Check every step