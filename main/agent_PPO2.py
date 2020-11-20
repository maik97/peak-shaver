import numpy as np
import gym
import pandas as pd
'''
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2

from datetime import datetime

#import schaffer
from wahrsager import wahrsager, max_seq, mean_seq
from common_env import common_env
from reward_maker import reward_maker
'''

def main():
    '''
    Example of an Agent that uses PPO2 provided by stable baselines
    '''
    # Logging-Namen:
    now            = datetime.now()
    NAME           = 'PPO2'+now.strftime("_%d-%m-%Y_%H:%M:%S")
    DATENSATZ_PATH = '_BIG_D/'

    # Lade Dataframe:
    df = schaffer.alle_inputs_neu()[24:-12]

    #df['pred_mean']       = wahrsager(TYPE='MEAN').pred()[:-12]
    #df['pred_max']        = wahrsager(TYPE='MAX').pred()[:-12]
    #df['pred_normal']     = wahrsager(TYPE='NORMAL').pred()[:-12]
    #df['pred_max_labe']   = wahrsager(TYPE='MAX_LABEL_SEQ').pred()
    #df['pred_mean_label'] = wahrsager(TYPE='MEAN_LABEL_SEQ').pred()

    prediction_seq        = wahrsager(TYPE='SEQ', num_outputs=12).pred()
    df['max_pred_seq']    = max_seq(prediction_seq)
    #df['mean_pred_seq']   = mean_seq(prediction_seq)

    power_dem_arr  = schaffer.load_total_power()[24:-12]

    # Lade Reward-Maker:
    R_HORIZON = 0
    r_maker        = reward_maker(
                            COST_TYPE               = 'exact_costs',     # 'yearly_costs', 'max_peak_focus'
                            R_TYPE                  = 'costs_focus',   #'costs_focus', 'savings_focus'
                            M_STRATEGY              = None,              # None, 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
                            R_HORIZON               = 'single_step',     # 'episode', 'single_step', integer for multi-step
                            cost_per_kwh            = 0.05,#0.2255,  # in €
                            LION_Anschaffungs_Preis = 34100,   # in €
                            LION_max_Ladezyklen     = 1000,
                            SMS_Anschaffungs_Preis  = 115000/3,# in €
                            SMS_max_Nutzungsjahre   = 20,      # in Jahre
                            Leistungspreis          = 102,     # in €
                            focus_peak_multiplier   = 4        # multiplier for max_peak costs
                            )

    # Lade Environment:
    env            = common_env(
                        df                   = df,
                        power_dem_arr        = power_dem_arr,
                        input_list           = ['norm_total_power','max_pred_seq'],
                        DATENSATZ_PATH       = DATENSATZ_PATH,
                        NAME                 = NAME,
                        max_SMS_SoC          = 12,
                        max_LION_SoC         = 54,
                        PERIODEN_DAUER       = 5,
                        ACTION_TYPE          = 'contin',
                        num_discrete_obs     = 21,
                        num_discrete_actions = 22,
                        reward_maker         = r_maker)

    # Lade vektorisierte Environment
    env = DummyVecEnv([lambda: env])

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=DATENSATZ_PATH+'models/',
                                             name_prefix='AGENT_'+NAME)

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=DATENSATZ_PATH+'LOGS/agent_logging', n_steps=2500)
    #model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=DATENSATZ_PATH+'LOGS/agent_logging',callback=checkpoint_callback, n_steps=2500)
    model.learn(total_timesteps=26000000, tb_log_name='agent_'+NAME)
    model.save(DATENSATZ_PATH+"models/AGENT_"+NAME)
    #obs = env.reset()
    #for i in range(MAX_STEPS*3):
    #  action, _states = model.predict(obs)
    #  obs, rewards, done, info = env.step(action)
    #  env.render()

    '''

    TENSORBOARD:
    tensorboard --logdir=_BIG_D/LOGS_ENV/


    '''

