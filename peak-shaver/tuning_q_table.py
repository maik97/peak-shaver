'''
Parameter tuning: RL-Agent that uses the basic Q-Table.
'''
import numpy as np
from datetime import datetime
from collections import deque

from common_settings import dataset_and_logger
from main.common_func import max_seq, mean_seq, training, testing
from main.reward_maker import reward_maker
from main.common_env import common_env

# Import the Q-Table agent: 
from main.agent_q_table import Q_Learner


def run_agent(name='',gamma=.9, lr=0.1, update_num=100, load_table=None,
              epsilon_decay='linear', input_list=['norm_total_power','normal','seq_max']):
    
    # Naming the agent:
    now    = datetime.now()
    NAME   = 'agent_Q-Table_'+name+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")

    # Import dataset and logger based on the common settings
    df, power_dem_df, logger, period_min = dataset_and_logger(NAME)

    # Number of warm-up steps:
    num_warmup_steps = 100
    # Number of epochs and steps:
    epochs           = 100


    # Setup reward_maker
    r_maker = reward_maker(
        LOGGER                  = logger,
        # Settings:
        COST_TYPE               = 'exact_costs',
        R_TYPE                  = 'savings_focus',
        R_HORIZON               = 'single_step',
        # Parameter to calculate costs:
        cost_per_kwh            = 0.2255,
        LION_Anschaffungs_Preis = 34100,
        LION_max_Ladezyklen     = 6000,
        SMS_Anschaffungs_Preis  = 55000,#115000/3,
        SMS_max_Nutzungsjahre   = 25,
        Leistungspreis          = 102,)

    # Setup common_env
    env = common_env(
        reward_maker   = r_maker,
        df             = df,
        power_dem_df   = power_dem_df,
        # Datset Inputs for the states:
        input_list     = input_list,
        # Batters stats:
        max_SMS_SoC        = 25,
        max_LION_SoC       = 54,
        LION_max_entladung = 50,
        SMS_max_entladung  = 100,
        SMS_entladerate    = 0.72,
        LION_entladerate   = 0.00008,
        # Period length in minutes:
        PERIODEN_DAUER = period_min,
        # Q-Table can only take discrete inputs and make discrete outputs
        ACTION_TYPE    = 'discrete',
        OBS_TYPE       = 'discrete',
        # Set number of discrete values:
        discrete_space = 22,
        # Size of validation data:
        val_split      = 0.1)

    # Setup agent:
    agent = Q_Learner(
        env            = env,
        memory_len     = update_num,
        # Training parameter:
        gamma          = gamma,
        epsilon        = 0.99,
        epsilon_min    = 0.1,
        epsilon_decay  = epsilon_decay,
        lr             = lr,
        load_table     = load_table)

    # Train:
    training(agent, epochs, update_num, num_warmup_steps)

    # Test with dataset that includes val-data:
    env.use_all_data()
    testing(agent)

#run_agent(name='Compare_Agents')

def parameter_tuning(num_runs=3):

    for i in range(num_runs):
        
        run_agent(name='Compare_Agents')
        
        # Learning rate:
        lr_list = [0.001,0.01,0.25]
        for lr in lr_list:
            run_agent(name='learning_rate_{}'.format(lr), lr=lr)
        
        # Gamma:
        gamma_list = [0.8,0.99]
        for gamma in gamma_list:
            run_agent(name='gamma_{}'.format(gamma), gamma=gamma)
        
        # update_num:
        update_num_list = [250,500,1000]
        for update_num in update_num_list:
            run_agent(name='update_num_{}'.format(update_num), update_num=update_num)
        
        # epsilon_decay:
        epsilon_decay_list = [0.999996]
        for epsilon_decay in epsilon_decay_list:
            run_agent(name='epsilon_decay_{}'.format(epsilon_decay), epsilon_decay=epsilon_decay)
        
        # input_list:
        lstm_inputs_list = [['norm_total_power'],['norm_total_power','normal'],
                            ['norm_total_power','seq_max'], ['norm_total_power','normal','seq_max'],
                            ['norm_total_power','mean']]
        i = 1
        for lstm_inputs in lstm_inputs_list:
            run_agent(name='lstm_inputs_test-{}'.format(i), input_list=lstm_inputs)
            i += 1
        

parameter_tuning()
