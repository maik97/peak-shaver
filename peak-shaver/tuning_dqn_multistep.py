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


def run_agent(name='', horizon=24, pre_trained_model=None):

    # Naming the agent:
    now    = datetime.now()
    NAME   = 'agent_DQN+MS_'+name+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")

    # Import dataset and logger based on the common settings
    df, power_dem_df, logger, period_min = dataset_and_logger(NAME)


    # Number of warm-up steps:
    num_warmup_steps = 100
    # Train every x number of steps:
    update_num       = 500
    # Number of epochs and steps:
    epochs           = 100


    # Setup reward_maker
    r_maker = reward_maker(
        LOGGER                  = logger,
        # Settings:
        COST_TYPE               = 'exact_costs',
        R_TYPE                  = 'savings_focus',
        # R_HORIZON is now an int for the periodes of the reward horizon:
        R_HORIZON               = horizon,
        # Additional the multi-step strategy must be set:
        M_STRATEGY              = 'sum_to_terminal',
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
        input_list     = ['norm_total_power','normal','seq_max'],
        # Batters stats:
        max_SMS_SoC        = 25,
        max_LION_SoC       = 54,
        LION_max_entladung = 50,
        SMS_max_entladung  = 100,
        SMS_entladerate    = 0.72,
        LION_entladerate   = 0.00008,
        # Period length in minutes:
        PERIODEN_DAUER = period_min,
        # DQN inputs can be conti and must be discrete:
        ACTION_TYPE    = 'discrete',
        OBS_TYPE       = 'contin',
        # Set number of discrete values:
        discrete_space = 22,
        # Size of validation data:
        val_split      = 0.1)


    # Setup Agent:
    agent = DQN(
        env            = env,
        memory_len     = update_num+horizon,
        gamma          = 0.9,#0.85,
        epsilon        = 0.99,
        epsilon_min    = 0.1,
        epsilon_decay  = 'linear',#0.999996,
        lr             = 0.1,#0.5,
        tau            = 0.15,#0.125,
        # Training parameter:
        activation     = 'relu',
        loss           = 'mean_squared_error',
        model_type     = 'dense',
        hidden_size    = 256,
        pre_trained_model = pre_trained_model,
        target_update_num = None)


    # Train:
    training(agent, epochs, update_num, num_warmup_steps)

    # Test with dataset that includes val-data:
    env.use_all_data()
    testing(agent)

#run_agent(name='Compare_Agents')

def parameter_tuning(num_runs=3):
    
    for i in range(num_runs):
        # horizon:
        horizon_list = [6,12,24]
        for horizon in horizon_list:
            run_agent(name='horizon_{}'.format(horizon), horizon=horizon)


parameter_tuning()
