import numpy as np
import gym
import pandas as pd
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam

from datetime import datetime
from collections import deque
from tqdm import tqdm

import schaffer
from wahrsager import wahrsager
from common_env import common_env
from reward_maker import reward_maker
from common_func import try_training_on_gpu, max_seq, mean_seq

class DQN:
    """Deep Q Network with LSTM Implementation

    Args:
        env (object): Takes in a GYM environment, use the common_env to simulate the HIPE-Dataset.
        memory (object): Takes in degue object: deque(maxlen=x)
        gamma (float): Factor that determines the importance of futue Q-values, value between 0 and 1
        epsilon (float): Initial percent of random actions, value between 0 and 1
        epsilon_min (float): Minimal percent of random actions, value between 0 and 1
        epsilon_decay (float): Factor by which epsilon decays, value between 0 and 1
        lr (float): Sets the learning rate of the RL-Agent
        tau (float): Factor for copying weights from model network to target network
        model_lr (float): Sets the learning rate for the Neural Network
        activation (string): Defines Keras activation function for each Dense layer (except the ouput layer) for the DQN
        loss (string): Defines Keras loss function to comile the DQN model
    """

    def __init__(self, env):
        
        self.env            = env
        self.memory         = deque(maxlen=1000)
        self.heurisitic_mem = deque(maxlen=1000)
        
        self.gamma = 0.85
        self.epsilon = 0.8
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999996
        self.learning_rate = 0.0001
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        '''Creates a Deep Neural Network with LSTM implementation which predicts Q-Values, when the class is initilized.

        Args:
            activation (string): Previously defined Keras activation
            loss (string): Previously defined Keras loss
        '''
        model   = Sequential()
        #state_shape  = self.env.observation_space.shape
        model.add(LSTM(512, input_shape=(30, 9), dropout=0.02, recurrent_dropout=0.02))
        #model.add(LSTM(256))
        model.add(Dense(512, activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(512, activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))

        #model = load_model('epoch-16.model')
        return model

    def past_states(self, heuristic_num = -1):
        '''
        ahdfbv
        '''
        if heuristic_num == -1:
        	past_num = -30
        else:
        	past_num = heuristic_num - 30

        if len(self.memory) < 30:
            past_num = -len(self.memory)
            num =len(self.memory)
        else:
        	num = 30

        update_state_inputs = np.zeros((1,30,9))
        for i in range(num):
            state, action, reward, new_state, done, new_peak, state_inputs = self.memory[past_num+i]
            update_state_inputs[0][-num+i] = new_state
            i =+ 1
        #print(state_inputs)
        return update_state_inputs

    def act(self, state, random_mode = False):
        '''
        Function, in which the agent decides an action, either from greedy-policy or from prediction. Use this function when iterating through each step.
        
        Args:
            state (array): Current state at the step

        Returns:
            action, epsilon (tuple):
            
            action (integer): Action that was chosen by the agent
            
            epsilon (float): Current (decayed) epsilon
        '''
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        state_inputs = self.past_states()
        
        if np.random.random() < self.epsilon or random_mode == True:
            return self.env.action_space.sample(), state_inputs
        return np.argmax(self.model.predict(state_inputs)[0]), state_inputs

    def remember(self, state, action, reward, new_state, done, new_peak, state_inputs):
        '''
        Takes in all necessery variables for the learning process and appends those to the memory. Use this function when iterating through each step.

        Args:
            state (array): State at which the agent chose the action
            action (integer): Chosen action
            reward (float): Real reward for the action
            new_state (array): State after action is performed
            done (bool): If True the episode ends
            new_peak (float): Passes the new maximum peak
            state_inputs (array): Passes the current inputs
        '''
        self.memory.append([state, action, reward, new_state, done, new_peak, state_inputs])

    def heuristic_rewards(self):
        '''
        function needs to be updated!
        '''

        power_dem = []
        SMS_SOC = []
        LION_SOC = []
        ziel_netzverbrauch = []
        max_peak = []
        new_peak_list = []
        soll_ist_diff = []
        action_liste = []

        for sample in self.memory:
            state, action, reward, new_state, done, new_peak, state_inputs = sample

            power_dem.append(state[0][3])
            SMS_SOC.append(new_state[0][7])
            LION_SOC.append(new_state[0][8])
            #ziel_netzverbrauch.append(state[0][9])
            #max_peak.append(state[0][10])
            #soll_ist_diff.append(state[0][12])
            #new_peak_list.append(new_peak)
            action_liste.append(action)

        action_list,SMS_SOC,LION_SOC = self.env.reverse_heuristic(power_dem,SMS_SOC,LION_SOC,action_liste)
        #action_list,SMS_SOC,LION_SOC,ziel_netz_dem,max_peak,soll_ist_diff = self.env.reverse_heuristic(power_dem,SMS_SOC,LION_SOC,ziel_netzverbrauch,max_peak,soll_ist_diff,new_peak_list,action_liste)

        i = 0
        for sample in self.memory:
            state, action, reward, new_state, done, new_peak, state_inputs = sample
            state[0][3] = power_dem[i]
            new_state[0][7] = SMS_SOC[i]
            new_state[0][8] = LION_SOC[i]
            #state[0][9] = ziel_netzverbrauch[i]
            #state[0][10] = max_peak[i]
            #state[0][12] = soll_ist_diff[i]
            #new_peak = new_peak_list[i]
            action = action_liste[i]
            reward = 100

            if i > 30:
            	state_inputs = self.past_states(heuristic_num = i)
            i += 1
            sample = [state, action, reward, new_state, done, new_peak, state_inputs]


    def replay(self, heurisitc_replay=False):
        '''
        Training-Process for the DQN from past steps. Use this function after a few iteration-steps (best use is the number of index_len). Alternatively use this function at each step.

        function needs to be updated!
        '''
        batch_size = 32



        if len(self.memory) < batch_size: 
        	return
        samples = random.sample(self.memory, batch_size)
        
        #else:
        #	if len(self.heurisitic_mem) < batch_size: 
        #		return
        #	samples = random.sample(self.heurisitic_mem, batch_size)


        for sample in samples:
            
            state, action, reward, new_state, done, new_peak, state_inputs = sample

            new_state_inputs = np.zeros((1,30,9))
            for i in range(30):
                if i < 29:
                    new_state_inputs[0][i] = state_inputs[0][i+1]
                else:
                    new_state_inputs[0][i] = new_state
            
            target = self.target_model.predict(state_inputs)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state_inputs)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state_inputs, target, epochs=1, verbose=0)

    def target_train(self):
        '''Updates the Target-Weights. Use this function after replay(index_len)'''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_agent(self, NAME, DATENSATZ_PATH, e):
        '''For saving the agents model at specific epoch. Make sure to not use this function at each epoch, since this will take up your memory space.

        Args:
            NAME (string): Name of the model
            DATENSATZ_PATH (string): Path to save the model
            e (integer): Takes the epoch-number in which the model is saved
        '''
        self.model.save(DATENSATZ_PATH+'models/'+NAME+'_{}'.format(e))


def main():
    '''
    Example of an RL-Agent that uses Dualing Deep Q-Networks with LSTM implementation
    '''
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
    
    dqn_agent = DQN(env=env)
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

if __name__ == "__main__":
    main()




