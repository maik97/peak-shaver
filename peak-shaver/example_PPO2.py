'''
Example of an Agent that uses PPO2 provided by stable baselines
'''
import numpy as np
import gym
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines import PPO2

from datetime import datetime

from main.common_func import max_seq, mean_seq, training, testing
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.logger import Logger
from main.reward_maker import reward_maker
from main.common_env import common_env

# Naming the agent and setting up the directory path:
now    = datetime.now()
NAME   = 'PPO2'+now.strftime("_%d-%m-%Y_%H-%M-%S")
D_PATH = '_BIG_D/'

# Load the dataset:
main_dataset = mainDataset(
    D_PATH=D_PATH,
    period_string_min='5min',
    full_dataset=True)

# Normalized dataframe:
df = main_dataset.make_input_df(
    drop_main_terminal=False,
    use_time_diff=True,
    day_diff='holiday-weekend')

# Sum of the power demand dataframe (nor normalized):
power_dem_df = main_dataset.load_total_power()

# Load the LSTM input dataset:
lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

# Making predictions:
normal_predictions = wahrsager(lstm_dataset, power_dem_df, TYPE='NORMAL').pred()[:-12]
seq_predictions    = wahrsager(lstm_dataset, power_dem_df, TYPE='SEQ', num_outputs=12).pred()

# Adding the predictions to the dataset:
df            = df[24:-12]
df['normal']  = normal_predictions
df['seq_max'] = max_seq(seq_predictions)

# Set up tensorboard logging:
logger = Logger(NAME,D_PATH)

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
    PERIODEN_DAUER = 5,
    ACTION_TYPE    = 'contin',
    OBS_TYPE       = 'contin')

# Lade vektorisierte Environment
env = DummyVecEnv([lambda: env])

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=D_PATH+'agent-models/',
                                         name_prefix=NAME)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=D_PATH+'agent-logs/', n_steps=2500)
#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=DATENSATZ_PATH+'LOGS/agent_logging',callback=checkpoint_callback, n_steps=2500)
model.learn(total_timesteps=26000000, tb_log_name=NAME)
model.save(D_PATH+"agent-models/"+NAME)
#obs = env.reset()
#for i in range(MAX_STEPS*3):
#  action, _states = model.predict(obs)
#  obs, rewards, done, info = env.step(action)
#  env.render()

'''

TENSORBOARD:
tensorboard --logdir=_BIG_D/LOGS_ENV/


'''

