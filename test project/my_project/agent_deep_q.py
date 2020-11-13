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

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import schaffer
from wahrsager import wahrsager, max_seq, mean_seq
from common_env import common_env
from reward_maker import reward_maker

class DQN:
    def __init__(self, env, memory, gamma, epsilon, epsilon_min, epsilon_decay, lr, tau, Q_table):

        self.env            = env
        self.memory         = memory
        
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.lr             = lr
        self.tau            = tau

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(518, input_dim=88, activation="relu"))
        #model.add(Dense(48, input_dim=state_shape[0], activation="relu"))
        #model.add(Dense(1000, activation="relu"))
        #model.add(Dense(1000, activation="relu"))
        model.add(Dense(518, activation="relu"))
        #model.add(Dense(128, activation="relu"))
        #model.add(Dense(128, activation="relu"))
        #model.add(Dense(64, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.lr))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), self.epsilon
        return np.argmax(self.model.predict(state)[0]), self.epsilon

    def remember(self, state, action, reward, new_state, done, step_counter_episode):
        self.memory.append([state, action, reward, new_state, done, step_counter_episode])


    def replay(self, index_len):
        for i in range(index_len):
            # Lade Step von Simulation
            state, action, reward, new_state, done, step_counter_episode = self.memory[i] 
            
            target = self.target_model.predict(state)
            #print(target[0][action])
            
            if reward == None:
                reward = self.env.get_multi_step_reward(step_counter_episode)

            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            #print(target[0][action])
            self.model.fit(state, target, epochs=1, verbose=0)
            

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_agent(self, NAME, DATENSATZ_PATH, e):
        self.model.save(DATENSATZ_PATH+'models/'+NAME+'_{}'.format(e))


def discrete_input_space(state_dim_size, state):
    state_placeholder = []
    for dim in state:
        dim_append = np.zeros((state_dim_size))
        dim_append[dim] = 1
        state_placeholder = np.append(state_placeholder,dim_append)
    state_placeholder = state_placeholder.reshape(1,len(state_placeholder))
    #print(state_placeholder)
    return state_placeholder




def main():
    # Logging-Namen:
    now            = datetime.now()
    NAME           = 'Deep-Q'+now.strftime("_%d-%m-%Y_%H:%M:%S")
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
                        R_TYPE                  = 'savings_focus',   #'costs_focus', 'savings_focus'
                        M_STRATEGY              = None,              # None, 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
                        R_HORIZON               = 'single_step',     # 'episode', 'single_step', integer for multi-step
                        cost_per_kwh            = 0.2255,  # in €
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
                        ACTION_TYPE          = 'discrete',
                        num_discrete_obs     = 21,
                        num_discrete_actions = 22,
                        reward_maker         = r_maker
                        )

    # Initilisiere Parameter für Target-Network
    update_num       = 500
    update_counter   = 0

    # Inititialsiere Epoch-Parameter:
    epochs           = 1000
    epochs_len       = len(df)
    
    num_warmup_steps = 100
    warmup_counter   = 0

    # Init Agent Parameter

    
    # Init Agent:
    Agent          = DQN(
                        env            = env,
                        memory         = deque(maxlen=(R_HORIZON+update_num)),

                        gamma          = 0.85,
                        epsilon        = 0.8,
                        epsilon_min    = 0.1,
                        epsilon_decay  = 0.999996,
                        lr             = 0.5,
                        tau            = 0.125,

                        Q_table        = np.zeros((22,22,22,22,22)), # jede Dimension jeweils ∈ [0,0.05,...,1]
                        )

    print('Warmup-Steps per Episode:', num_warmup_steps)
    print('Training for',epochs,'Epochs')
    state_placeholder = np.zeros((22,22,22,22))

    for e in range(epochs):
        #print('Epoch:', e)
        #tqdm.write('Starting Epoch: {}'.format(e))
        cur_state = env.reset()
        #cur_state = cur_state.reshape(1,len(cur_state))[0]
        cur_state = discrete_input_space(22, cur_state)

        #tqdm.write('Warm-Up for {} Steps...'.format(num_warmup_steps))
        while warmup_counter < num_warmup_steps:
            action, epsilon            = Agent.act(cur_state)
            new_state, reward, done, step_counter_episode, _ = env.step(action, epsilon)
            new_state                  = discrete_input_space(22, new_state)
            Agent.remember(cur_state, action, reward, new_state, done, step_counter_episode)

            cur_state                  = new_state
            warmup_counter            += 1

        bar = tqdm(range(epochs_len))#, leave=True, file=sys.stdout)
        bar.set_description("Training - Epoch {}".format(e))
        for step in bar:

            action, epsilon            = Agent.act(cur_state)
            new_state, reward, done, step_counter_episode, _ = env.step(action, epsilon)
            new_state                  = discrete_input_space(22, new_state)         
            Agent.remember(cur_state, action, reward, new_state, done, step_counter_episode)
            
            cur_state                  = new_state
            
            if done == False:
                index_len = update_num
            else:
                index_len = update_num + R_HORIZON

            update_counter += 1
            if update_counter == update_num or done == True:
                Agent.replay(index_len)
                Agent.target_train()                 # iterates target model
                update_counter = 0

            if done:
                break
        #bar.clear()

        if e % 10 == 0:
            Agent.save_agent(NAME, DATENSATZ_PATH, e)


if __name__ == "__main__":
    main()




