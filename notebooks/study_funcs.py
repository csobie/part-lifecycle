import logging
import pandas as pd
import simpy
import part_lifecycle as plc
import numpy as np
import scipy

    
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
        init_mach_partset = plc.PartSet([plc.Part(params['part_n_runs_life']) for j in range(params['n_parts_per_mach'])])

        if params['rand_start']:
            init_time = rng.integers(1,params['part_run_dur']+1) # Generate uniformly distributed start time for the parts in the machine
        else:
            init_time = 0

        machines.append(plc.Machine(env, warehouse, repair_storage,params['part_run_dur'], init_mach_partset, init_time,params['act_scrap_params'],params['pred_scrap_params']))
        logging.debug('setup: send initial PartSet %s to Machine %s',init_mach_partset,machines[-1].mach_id)

    # Initialize spare parts for the warehouse    
    init_ware_partsets = []
    for i in range(params['n_partsets_warehouse']):
        init_ware_partsets.append(plc.PartSet([plc.Part(params['part_n_runs_life']) for i in range(params['n_parts_per_mach'])]))
        logging.debug(f'setup: create initial PartSet {init_ware_partsets[-1]} for warehouse')

    # Add them to the warehouse
    warehouse.items = init_ware_partsets

    # Create the factory, initialize the shops
    factory = plc.Factory(env, preproc_storage, params['fact_lead_time'], params['part_n_runs_life'], params['sim_length'])
    prepro_shop = plc.PreprocessShop(env, warehouse, preproc_storage, params['prepro_through'], params['n_parts_per_mach'])
    rep_shop = plc.RepairShop(env, preproc_storage, repair_storage, scrap, params['rep_through'])

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
        
def inventory_control(env, warehouse, factory, machines, params, inv_ctrl_method, **kwargs):
    
    if inv_ctrl_method == 'r_q':
        try:
            r = kwargs['r']
            q = kwargs['q']
        except KeyError as e:
            print('kwargs r, q must be passed for inv_ctrl_method r_q')
            raise e
        while True:
            # Consider the warehouse contents plus the number of parts arriving within one lead time.
            total_items = len(warehouse.items)*params['n_parts_per_mach'] + np.sum(factory.orders[env.now:env.now+params['fact_lead_time']])
            if total_items < r:
                logging.debug(f'{env.now}: Inventory control requesting parts from factory, warehouse has {len(warehouse.items)} PartSets')
                factory.order_parts(q, max_runs=params['part_n_runs_life'])
            yield env.timeout(30)  # Check every month
    
    
    elif inv_ctrl_method == 's_s':
        try:
            small_s = kwargs['small_s']
            big_s = kwargs['big_s']
        except KeyError as e:
            print('kwargs small_s, big_s must be passed for inv_ctrl_method s_s')
            raise e
        while True:
            total_items = len(warehouse.items)*params['n_parts_per_mach'] + np.sum(factory.orders[env.now:env.now+params['fact_lead_time']])
            if total_items < small_s:
                logging.debug(f'{env.now}: Inventory control requesting parts from factory, warehouse has {len(warehouse.items)} PartSets')
                factory.order_parts(big_s-len(warehouse.items), max_runs=params['part_n_runs_life'])
            yield env.timeout(30)  # Check every month
                
    # Need to deal with mix between stochastic and deterministic scrappage here
    elif inv_ctrl_method == 'fcast':

        try:
            days_lookahead = kwargs['days_lookahead']
            target_level = kwargs['target_level']
        except KeyError as e:
            print('kwargs days_lookahead, target_level must be passed for inv_ctrl_method fcast')
            raise e
        while True:
            # Calculate the number of availabile parts within one full lead time, minus the expected scrapped parts.
            expected_scrap = np.zeros(params['sim_length']+1000)
            for m in machines:
                # Set scrap rate: (all Parts)*predicted scrap rate, plus (1-predicted scrap rate)*(number of parts being scrapped deterministically)
                if 'const' in m.pred_scrap_params:
                    scrap_rate = m.pred_scrap_params['const']
                elif all (x in m.pred_scrap_params for x in ("a","b")):
                    try:
                        scrap_rate_ppf = kwargs['scrap_rate_ppf']
                    except KeyError as e:
                        print('kwargs scrap_rate_ppf must be passed inv_ctrl_method fcast and pred_scrap_params (a,b)')
                        raise e
                    # Get nth percentile of dist
                    scrap_rate = scipy.stats.beta.ppf(scrap_rate_ppf,m.pred_scrap_params['a'], m.pred_scrap_params['b'])
                else:
                    raise ValueError(f'Machine {m} does not have required pred_scrap_params keys ("const" or ["a","b"]): {m.pred_scrap_params}')
                expected_scrap[m.partset_change_date] += params['n_parts_per_mach']*scrap_rate+(1-scrap_rate)*m.scrap_num_detr

            # Subtract the expected scrap during one lead time from the effective warehoused quantity (on-hand parts + ordered parts)
            n_scrap_pred = int(sum(expected_scrap[env.now:env.now+int(params['fact_lead_time']*1.25)]))
            total_items = len(warehouse.items)*params['n_parts_per_mach'] + np.sum(factory.orders[env.now:env.now+params['fact_lead_time']])-n_scrap_pred
            if total_items < target_level:
                logging.debug(f'{env.now}: Inventory control requesting parts from factory, warehouse has {len(warehouse.items)} PartSets')
                factory.order_parts(target_level-total_items, max_runs=params['part_n_runs_life'])
            yield env.timeout(30)  # Check every month
    else:
        raise ValueError(f'Invalid control type passed to inventory_control: {inv_ctrl_method}')
    yield

def run_sim(inv_ctrl,act_scrap_params,pred_scrap_params,**kwargs):

    env = simpy.Environment()
    warehouse = simpy.Store(env, capacity=10000)
    preproc_storage = simpy.Store(env, capacity=100000)
    repair_storage = simpy.Store(env, capacity=100000)
    scrap = simpy.Store(env, capacity=1000000)

    leadtime_func = None


    sim_years = 10
    sim_days = int(365*sim_years)

    params = {'sim_length':sim_days,
          'fact_lead_time':180,
          'part_n_runs_life':2,
          'part_run_dur':300,
          'prepro_through':kwargs['prepro_through'],
          'rep_through':kwargs['rep_through'],
          'n_parts_per_mach':75,
          'n_partsets_warehouse':kwargs['q_init'],
          'n_machine':75,
          'rand_start':True,
          'leadtime_func':leadtime_func,
          'act_scrap_params':act_scrap_params,
          'pred_scrap_params':pred_scrap_params
         }

    # Monitor the contents of the part storage facilities
    warehouse_data = []
    preproc_storage_data = []
    repair_storage_data = []
    env.process(monitor_warehouse(env, warehouse, warehouse_data))
    env.process(monitor_warehouse(env, preproc_storage, preproc_storage_data))
    env.process(monitor_warehouse(env, repair_storage, repair_storage_data))

    machines, factory = setup(env, warehouse, preproc_storage, repair_storage, scrap, params)
    # Start inventory management process
    env.process(inventory_control(env, warehouse, factory, machines, params,inv_ctrl, **kwargs))

    env.run(until=sim_days)

    # Cost calculation
    irr = 0.04
    part_value = 1e4

    rep_shop_cost = 15e6+params['rep_through']*1e6
    prepro_shop_cost = 15e6+params['prepro_through']*1e6

    # Calculate total downtime: downtime that started and ended in the sim time, as well as those running to the end of the sim.
    total_downtime = (sum([mach.downtime for mach in machines]) + 
                      sum([sim_days-mach.curr_down_start for mach in machines if mach.curr_down_start is not None]))

    downtime_cost = total_downtime*2e4

    # Opportunity loss calculation
    df = pd.DataFrame(data={'warehouse':warehouse_data,'preproc':preproc_storage_data,'repair':repair_storage_data})
    # Warehouse and repair storage contains PartSets, convert to number of Parts
    df['warehouse'] *= params['n_parts_per_mach']
    df['repair'] *= params['n_parts_per_mach']

    df['idle_parts'] = df.sum(axis=1)
    df['interest']=df['idle_parts']*part_value*((1+irr)**(1/365)-1)
    op_loss_cost = df['interest'].sum()
    
    # Cost of accumulated stock
    start_stock = (params['n_machine']+params['n_partsets_warehouse'])*params['n_parts_per_mach']
    part_cost = (start_stock+factory.n_parts_made)*part_value

    final_cost = downtime_cost+op_loss_cost+rep_shop_cost+prepro_shop_cost+part_cost
#     print(f'Cost breakdown: downtime_cost:{downtime_cost} op_loss_cost:{op_loss_cost} rep_shop_cost:{rep_shop_cost} prepro_shop_cost:{prepro_shop_cost} part_cost:{part_cost}')
#     print(inv_opt_p1,inv_opt_p2,prepro_through,rep_through,q_init,final_cost/1e9)
    return final_cost