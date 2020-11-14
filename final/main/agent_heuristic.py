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
'''
import schaffer
from wahrsager import wahrsager, max_seq, mean_seq
from common_env import common_env
from reward_maker import reward_maker

### 'Single-Value-Heuristic' ###
Bestimmt einen einzelnen Zielnetzetzverbrauch, der für alle Steps benutzt wird.
-> selbe nochmal aber mit Reward-Focus!

### 'Perfekt-Pred-Heuristic' ###
Alle zukünfitigen Werte sind bekannt.

### 'LSTM-Pred-Heuristic' ###
Heuristik mit realistischen Inputs, sprich höchstens Vorhersehungen mit LSTM möglich. 
'''


class heurisitc_agents:
    '''
    There are three main heuristic approaches, with the goal to minimize the maximum energy peak:
    
    1. 'Single-Value-Heuristic' approximates the best global value (for all steps), that is used to determine the should energy consumption from the grid.
    
    2. 'Perfekt-Pred-Heuristic' finds the best should energy consumptions for each steps, under the assumption, that the future energy-need is perfectly predicted.
    
    3. 'LSTM-Pred-Heuristic' approximates the best should energy consumptions for each step, with LSTM-predicted future energy-need.
    
    These three aproaches can also have the goal to minimize the sum of cost instead of the maximum peak.
    
    Args:
        NAME
            Name of the model
        DATENSATZ_PATH
            Path to save the model
        HEURISTIC_TYPE
            Determines the type of heuristic to be used: 'Single-Value-Heuristic', 'Perfekt-Pred-Heuristic', 'LSTM-Pred-Heuristic'
        df
            Takes in the HIPE-dataset as pandas dataframe
        env
            Takes in a GYM environment, use the common_env module to simulate the HIPE-Dataset.
        global_zielverbrauch
            Used to initilize the start value to approximate a global should energy consumption from the grid, relevant for 'Single-Value-Heuristic'
    '''
    def __init__(self, NAME, DATENSATZ_PATH, HEURISTIC_TYPE, df, env, global_zielverbrauch=50):

        self.HEURISTIC_TYPE              = HEURISTIC_TYPE
        self.df                          = df
        self.env                         = env
        #self.memory                      = memory
        self.vorher_global_zielverbrauch = 0
        self.prev_sum_reward             = 0
        self.global_zielverbrauch        = global_zielverbrauch

        # Init Logging
        # self.LOGGER                    = logger.Logger(DATENSATZ_PATH+'LOGS/agent_logging/'+NAME)
    
    def global_single_value_for_max_peak(self, max_peak):
        '''
        Uses the mean-value theorem to calculate at the end of each episode a new global SECG (should energy-consumption from the grid)

        Args:
            max_peak
                Takes in max_peak at the end of each episode

        Returns:
            neu_global_zielverbrauch (float): New global SECG, after calculation with the mean-value theorem
        '''
        print(max_peak)
        # wenn max_peak größer ist als benchmark, muss aktueller benchmark größer werden:
        if max_peak > self.global_zielverbrauch:
            neu_global_zielverbrauch = self.global_zielverbrauch + (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        # wenn max-peak kleiner/gleich benchmark ist als, kann aktueller benchmark kleiner werden
        if max_peak <= self.global_zielverbrauch:
            neu_global_zielverbrauch = self.global_zielverbrauch - (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        #print('Neuer globaler Zielnetzetzverbrauch:',neu_global_zielverbrauch)

        self.vorher_global_zielverbrauch = self.global_zielverbrauch
        self.global_zielverbrauch = neu_global_zielverbrauch
        return neu_global_zielverbrauch

    def global_single_value_for_reward(self, sum_reward, positivity_value=10000):
        '''Uses the mean-value theorem to calculate a new SECG at the at of each episode, instead of minimizing the maximum peak, this function maximizes the sum of rewards.
        
        Args:
            sum_reward 
                Takes in the sum of rewards after each episode
            positivity_value
                Should be a large number, which is added to the sum of rewards to make sure that negative rewards will be transformed into positive ones
            
        Returns:
            neu_global_zielverbrauch (float): New global SECG, after calculation with the mean-value theorem
        '''
        sum_reward += positivity_value
        # wenn max_peak größer ist als benchmark, muss aktueller benchmark größer werden:
        if sum_reward < self.prev_sum_reward:
            neu_global_zielverbrauch = self.global_zielverbrauch + (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        # wenn max-peak kleiner/gleich benchmark ist als, kann aktueller benchmark kleiner werden
        if sum_reward >= self.prev_sum_reward:
            neu_global_zielverbrauch = self.global_zielverbrauch - (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        #print('Neuer globaler Zielnetzetzverbrauch:',neu_global_zielverbrauch)

        self.vorher_global_zielverbrauch = self.global_zielverbrauch
        self.global_zielverbrauch = neu_global_zielverbrauch
        self.prev_sum_reward = sum_reward
        return neu_global_zielverbrauch

    def act(self, SMS_PRIO = 0):
        '''Function, in which the heuristic decides an action, based on HEURISTIC_TYPE previously set.
        
        Args:
            SMS_PRIO (float): Is either set to 0 or 1:
                
                0 sets the priority of SMS to true
                
                1 sets the priority of SMS to false

        Returns:
            SECG, SMS_PRIO (tuple):
            
            SECG (float): should energy-consumption from the grid at step
            
            SMS_PRIO (float): priority of SMS at step
        '''
        if self.HEURISTIC_TYPE == 'Single-Value-Heuristic' or self.HEURISTIC_TYPE == 'Single-Value-Heuristic-Reward':
            return [self.global_zielverbrauch, SMS_PRIO] # 0 bedeutet dass SMS-Priority aktiviert wird.
        
        elif self.HEURISTIC_TYPE == 'Perfekt-Pred-Heuristic' or self.HEURISTIC_TYPE =='LSTM-Pred-Heuristic':
            action = [self.heuristic_zielnetzverbrauch[self.current_step], SMS_PRIO]
            self.current_step += 1
            return action
        else:
            print("ERROR: HEURISTIC_TYPE not understood. HEURISTIC_TYPE must be: 'Single-Value-Heuristic', 'Single-Value-Heuristic-Reward', 'Perfekt-Pred-Heuristic', 'LSTM-Pred-Heuristic'")
            exit()

    def bar_printer(self):
        '''Helper function to print helpful information about the mean-value process at the process bar'''
        if self.HEURISTIC_TYPE == 'Single-Value-Heuristic':
            return "Global Value - {}".format(self.global_zielverbrauch)
        elif self.HEURISTIC_TYPE == 'Single-Value-Heuristic-Reward':
            return "Global Value - {}, Sum Reward - {}".format(self.global_zielverbrauch, self.prev_sum_reward )


    def find_optimum_for_perfect_pred(self,power_dem_arr,global_value=50):
        '''Funtion to prepare the SECG for each step, used when HEURISTIC_TYPE is set to 'Perfekt-Pred-Heuristic'. Used before iterating through each step.

        Args:
            power_dem_arr (array): Array that represents the local power consumption at each step
            global_value (float): Minimal peak, that the battery arrangement is able to shave

        '''
        print('Calculating optimal actions for perfect predictions...')
        power_to_shave = 0
        zielnetzverbrauch = []
        for power_dem_step in power_dem_arr[::-1]:
        	
        	if power_dem_step > global_value:
        		power_to_shave += power_dem_step - global_value
        		zielnetzverbrauch.append(global_value)
        	
        	else:# power_dem_step <= global_value:
        		if (power_to_shave + power_dem_step) > global_value:
        			power_to_shave -= (global_value - global_value)
        			zielnetzverbrauch.append(global_value)
        		else:
        			zielnetzverbrauch.append(power_to_shave + power_dem_step)
        			power_to_shave = 0
        self.heuristic_zielnetzverbrauch = zielnetzverbrauch[::-1]
        self.current_step = 0


def main():
    '''
    Example of an Agent that uses heuristics.
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


if __name__ == "__main__":
    main()




