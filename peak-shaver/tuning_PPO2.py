'''
Example of an Agent that uses PPO2 provided by stable baselines
'''
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2

from datetime import datetime

import common_settings as cms
from main.common_func import max_seq, mean_seq, training, testing
from main.reward_maker import reward_maker
from main.common_env import common_env


def run_agent(name='', n_steps=2500):
    # Naming the agent:
    now    = datetime.now()
    NAME   = 'PPO2'+name+now.strftime("_%d-%m-%Y_%H-%M-%S")

    # Import dataset and logger based on the common settings
    df, power_dem_df, logger, period_min = cms.dataset_and_logger(NAME)

    epochs = 1


    # Setup reward_maker
    r_maker = reward_maker(
        LOGGER                  = logger,
        # Settings:
        COST_TYPE               = 'exact_costs',
        R_TYPE                  = 'savings_focus',
        # Agents from stable base-lines cant use multi-step rewards from our code
        # So R_HOTIZON can only be 'single-step'
        R_HORIZON               = 'single_step',
        # Parameter to calculate costs:
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
        # Datset Inputs for the states:
        input_list     = ['norm_total_power','normal','seq_max'],
        # Batters stats:
        max_SMS_SoC    = 12/3,
        max_LION_SoC   = 54,
        # Period length in minutes:
        PERIODEN_DAUER = period_min,
        # PPO can use conti values:
        ACTION_TYPE    = 'contin',
        OBS_TYPE       = 'contin',
        # Tells the environment to make standart GYM outputs, 
        # so agents from stable-baselines (or any other standart module) can be used
        AGENT_TYPE     = 'standart_gym')

    # Create vectorised environment:
    dummy_env = DummyVecEnv([lambda: env])

    # Callback:
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=cms.__dict__['D_PATH']+'agent-models/',
                                             name_prefix=NAME)

    # Setup Model:
    model = PPO2(MlpPolicy, dummy_env, verbose=1, tensorboard_log=cms.__dict__['D_PATH']+'agent-logs/', n_steps=2500)
    #model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=DATENSATZ_PATH+'LOGS/agent_logging',callback=checkpoint_callback, n_steps=2500)
    
    # Train:
    model.learn(total_timesteps=epochs*len(env.__dict__['df']), tb_log_name=NAME)
    model.save(cms.__dict__['D_PATH']+"agent-models/"+NAME)
    
    # Test with dataset that includes val-data:
    env.use_all_data()
    testing(model, use_stable_b=True, env=env)


run_agent()


def parameter_tuning():

    # Learning rate:
    n_steps_list = []
    for n_steps in n_steps_list:
        run_agent(name='n_steps_{}'.format(n_steps), n_steps=n_steps)

    # Gamma:
    gamma_list = []
    for gamma in gamma_list:
        run_agent(name='gamma_{}'.format(gamma), gamma=gamma)

    # Tau:
    tau_list = []
    for tau in tau_list:
        run_agent(name='tau_{}'.format(tau), tau=tau)

    # update_num:
    update_num_list = []
    for update_num in update_num_list:
        run_agent(name='update_num_{}'.format(update_num), update_num=update_num)

    # epsilon_decay:
    epsilon_decay_list = []
    for epsilon_decay in epsilon_decay_list:
        run_agent(name='epsilon_decay_{}'.format(epsilon_decay), epsilon_decay=epsilon_decay)

    # input_list:
    input_list_list = []
    for input_list in input_list_list:
        run_agent(name='input_list_{}'.format(input_list), input_list=input_list)

    # hidden_size:
    hidden_size_list = []
    for hidden_size in hidden_size_list:
        run_agent(name='hidden_size_{}'.format(hidden_size), hidden_size=hidden_size)