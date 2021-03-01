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


def run_agent(name='', learning_rate=0.00025, gamma=0.99, n_steps=2500, ent_coef=0.01, vf_coef=0.5, cliprange=0.2):
    # Naming the agent:
    now    = datetime.now()
    NAME   = 'agent_PPO2_'+name+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")

    # Import dataset and logger based on the common settings
    df, power_dem_df, logger, period_min = cms.dataset_and_logger(NAME)

    epochs = 100


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
        input_list     = ['norm_total_power','seq_max'],
        # Batters stats:
        max_SMS_SoC        = 25,
        max_LION_SoC       = 54,
        LION_max_entladung = 50,
        SMS_max_entladung  = 100,
        SMS_entladerate    = 0.72,
        LION_entladerate   = 0.00008,
        # Period length in minutes:
        PERIODEN_DAUER = period_min,
        # PPO can use conti values:
        ACTION_TYPE    = 'contin',
        OBS_TYPE       = 'contin',
        # Tells the environment to make standart GYM outputs, 
        # so agents from stable-baselines (or any other RL-library that uses gym) can be used
        AGENT_TYPE     = 'standart_gym',
        val_split      = 0.1)

    # Create vectorised environment:
    dummy_env = DummyVecEnv([lambda: env])

    # Callback:
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=cms.__dict__['D_PATH']+'agent-models/',
                                             name_prefix=NAME)

    # Setup Model:
    model = PPO2(MlpPolicy, dummy_env, verbose=1, tensorboard_log=cms.__dict__['D_PATH']+'agent-logs/',
                learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef, cliprange=cliprange)
    #model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=DATENSATZ_PATH+'LOGS/agent_logging',callback=checkpoint_callback, n_steps=2500)
    
    # Train:
    model.learn(total_timesteps=epochs*len(env.__dict__['df']), tb_log_name=NAME)
    model.save(cms.__dict__['D_PATH']+"agent-models/"+NAME)
    
    # Test with dataset that includes val-data:
    env.use_all_data()
    testing(model, use_stable_b=True, env=env)


#run_agent()


def parameter_tuning():

    for i in range(3):
        '''
        run_agent(name='standard')
        
        # Learning rate:
        learning_rate_list = [0.003,0.00001]
        for learning_rate in learning_rate_list:
            run_agent(name='learning_rate_{}'.format(learning_rate), learning_rate=learning_rate)
        
        # gamma:
        gamma_list = [0.8,0.9997]
        for gamma in gamma_list:
            run_agent(name='gamma_{}'.format(gamma), gamma=gamma)
        
        # n_steps:
        n_steps_list = [1000,5000]
        for n_steps in n_steps_list:
            run_agent(name='n_steps_{}'.format(n_steps), n_steps=n_steps)
        
        # ent_coef:
        ent_coef_list = [0,0.1]
        for ent_coef in ent_coef_list:
            run_agent(name='ent_coef_{}'.format(ent_coef), ent_coef=ent_coef)
        
        # vf_coef:
        vf_coef_list = [0.75,1]
        for vf_coef in vf_coef_list:
            run_agent(name='vf_coef_{}'.format(vf_coef), vf_coef=vf_coef)
        '''
        # cliprange:
        cliprange_list = [0.1,0.3]
        for cliprange in cliprange_list:
            run_agent(name='cliprange_{}'.format(cliprange), cliprange=cliprange)
        '''
        
        # lstm_inputs:
        lstm_inputs_list = []
        for lstm_inputs in lstm_inputs_list:
            run_agent(name='lstm_inputs_{}'.format(lstm_inputs), input_list=lstm_inputs)
        '''

parameter_tuning()