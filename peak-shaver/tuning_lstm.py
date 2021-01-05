'''
Example of an RL-Agent that uses Dualing Deep Q-Networks.
'''
from datetime import datetime

from common_settings import dataset_and_logger
from main.common_func import max_seq, mean_seq, training, testing
from main.reward_maker import reward_maker
from main.common_env import common_env

# Import the DQN agent: 
from main.agent_deep_q import DQN


def run_agent(name='', input_sequence=12):
    
    # Naming the agent:
    now    = datetime.now()
    NAME   = 'DQN+LSTM'+now.strftime("_%d-%m-%Y_%H-%M-%S")

    # Import dataset and logger based on the common settings
    df, power_dem_df, logger = dataset_and_logger(NAME)


    # Number of warm-up steps:
    num_warmup_steps = 100
    # Train every x number of steps:
    update_num       = 50
    # Number of epochs and steps:
    epochs           = 1000
    # Horizon for Multi-Step-Rewards and/or LSTM-Implementation:
    # horizon = 0
    # input_sequence = 1


    # Setup reward_maker
    r_maker = reward_maker(
        LOGGER                  = logger,
        COST_TYPE               = 'exact_costs',
        R_TYPE                  = 'savings_focus',
        R_HORIZON               = 'single_step',
        cost_per_kwh            = 0.2255,
        LION_Anschaffungs_Preis = 34100,
        LION_max_Ladezyklen     = 1000,
        SMS_Anschaffungs_Preis  = 115000/3,
        SMS_max_Nutzungsjahre   = 20,
        Leistungspreis          = 102)

    # Setup common_env
    env = common_env(
        reward_maker   = r_maker,
        df             = df,
        power_dem_df   = power_dem_df,
        input_list     = ['norm_total_power','normal','seq_max'],
        max_SMS_SoC    = 12/3,
        max_LION_SoC   = 54,
        PERIODEN_DAUER = 15,
        ACTION_TYPE    = 'discrete',
        OBS_TYPE       = 'contin',
        discrete_space = 22)


    # Setup Agent:
    agent = DQN(
        env            = env,
        memory_len     = update_num,
        input_sequence = 12,
        gamma          = 0.85,
        epsilon        = 0.8,
        epsilon_min    = 0.1,
        epsilon_decay  = 0.999996,
        lr             = 0.5,
        tau            = 0.125,
        activation     = 'relu',
        loss           = 'mean_squared_error',
        model_type     = 'lstm')


    training(agent, epochs, update_num, num_warmup_steps)
    testing(agent)


def parameter_tuning():

    # horizon:
    input_sequence_list = []
    for input_sequence in input_sequence_list:
        run_agent(name='input_sequence_{}'.format(input_sequence), input_sequence=input_sequence)

