'''
Example of an Agent that uses heuristics.
'''
import sys
import numpy as np
import pandas as pd
import random
import h5py

import gym
from gym import spaces

from datetime import datetime
from collections import deque
from tqdm import tqdm

import main.schaffer as schaffer
from main.wahrsager import wahrsager, max_seq, mean_seq
from main.common_env import common_env
from main.reward_maker import reward_maker
from main.agent_heuristic import heurisitc_agents

'''
### 'Single-Value-Heuristic' ###
Bestimmt einen einzelnen Zielnetzetzverbrauch, der für alle Steps benutzt wird.
-> selbe nochmal aber mit Reward-Focus!

### 'Perfekt-Pred-Heuristic' ###
Alle zukünfitigen Werte sind bekannt.

### 'LSTM-Pred-Heuristic' ###
Heuristik mit realistischen Inputs, sprich höchstens Vorhersehungen mit LSTM möglich. 
'''

# Logging-Namen:
HEURISTIC_TYPE = 'Perfekt-Pred-Heuristic'

now            = datetime.now()
NAME           = HEURISTIC_TYPE+now.strftime("_%d-%m-%Y_%H:%M:%S")
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

power_dem_arr  = schaffer.load_total_power().to_numpy()[24:-12]

# Lade Reward-Maker:
R_HORIZON = 0
r_maker        = reward_maker(
                    COST_TYPE               = 'exact_costs',     # 'yearly_costs', 'max_peak_focus'
                    R_TYPE                  = 'costs_focus',     # 'savings_focus'
                    M_STRATEGY              = None, # None, 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
                    R_HORIZON               = 'single_step',         # 'episode', '...', integer for multi-step
                    cost_per_kwh            = 0.07,  # in €
                    LION_Anschaffungs_Preis = 32000, # in €
                    LION_max_Ladezyklen     = 1000,
                    SMS_Anschaffungs_Preis  = 10000, # in €
                    SMS_max_Nutzungsjahre   = 15,    # in Jahre
                    Leistungspreis          = 90,    # in €
                    focus_peak_multiplier   = 4      # multiplier for max_peak costs
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
                    #action_space         = spaces.Discrete(22), # A ∈ [0,1]
                    #observation_space    = spaces.Box(low=0, high=21, shape=(4,1), dtype=np.float16),
                    reward_maker         = r_maker,
                    AGENT_TYPE           = 'heuristic'
                    )


# Inititialsiere Epoch-Parameter:
epochs           = 20
epochs_len       = len(df)

neu_global_zielverbrauch = 30
# Init Agent:
Agent          = heurisitc_agents(
                    NAME, DATENSATZ_PATH, HEURISTIC_TYPE, df,
                    env = env)

if HEURISTIC_TYPE == 'Perfekt-Pred-Heuristic':
	Agent.find_optimum_for_perfect_pred(power_dem_arr)

print('Testing the Heuristic for',epochs,'Epochs')
for e in range(epochs):

    cur_state = env.reset()
    env.set_soc_and_current_state()

    bar = tqdm(range(epochs_len))
    bar.set_description(Agent.bar_printer())

    for step in bar:

        action                               = Agent.act(SMS_PRIO=0)# 0 heißt Prio ist aktiv
        new_state, reward, done, max_peak, _ = env.step(action)         

        if done:
            break
    
    if HEURISTIC_TYPE == 'Single-Value-Heuristic':
        neu_global_zielverbrauch = Agent.global_single_value_for_max_peak(max_peak)
    elif HEURISTIC_TYPE == 'Single-Value-Heuristic-Reward':
        neu_global_zielverbrauch = Agent.global_single_value_for_reward(r_maker.get_sum_reward())





