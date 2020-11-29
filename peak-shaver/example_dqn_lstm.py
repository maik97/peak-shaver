'''
Example of an RL-Agent that uses Dualing Deep Q-Networks with LSTM implementation
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
from main.agent_lstm_q import DQN_LSTM

# Variablen:
gamma   = 0.9
epsilon = .95

# Logging-Namen:
now            = datetime.now()
NAME           = 'GDQN'+now.strftime("_%d-%m-%Y_%H:%M:%S")
DATENSATZ_PATH = '_BIG_D/'

# Lade Dataframe:
df        = wahrsager.predictions_and_inputs()
MAX_STEPS = len(df)

# Lade Environment
env       = LSTM_DQN_env(df, DATENSATZ_PATH, NAME)

# Initilisiere Varaiblen für Target-Network
update_TN = 1000
u         = 0

# Inititialsiere Epoch-Variablen:
epochs     = 1000
epochs_len = len(df)
min_steps  = 100

dqn_agent = DQN_LSTM(env=env)
steps = []
for e in range(epochs):
    print('Epoch:', e)
    cur_state = env.reset().reshape(1,9)
    
    for step in tqdm(range(epochs_len)):
        u += 1

        if step > min_steps:
            action, state_inputs = dqn_agent.act(cur_state)
        else:
            action, state_inputs = dqn_agent.act(cur_state, random_mode = True)
            u = 0

        new_state, reward, done, new_peak, _ = env.step(action)

        # reward = reward if not done else -20
        new_state = new_state.reshape(1,9)
        dqn_agent.remember(cur_state, action, reward, new_state, done, new_peak, state_inputs)
        #print(cur_state)
        
        if u == update_TN:
            
            dqn_agent.replay()     					 # internally iterates default (prediction) model,  heurisitc_replay=False
            dqn_agent.target_train() 				 # iterates target model
            
            #dqn_agent.heuristic_rewards()
            #dqn_agent.replay()     					 # internally iterates default (prediction) model,  heurisitc_replay=False
            #dqn_agent.replay(heurisitc_replay=True)    
            #dqn_agent.target_train() 				 # iterates target model

            u = 0

        cur_state = new_state
        if done:
            break

    if e % 10 == 0:
        Agent.save_agent(NAME, DATENSATZ_PATH, e)


