import sys

import numpy as np
import pandas as pd
import random
import h5py

import gym
from gym import spaces

from datetime import datetime
from collections import deque

try:
    from main.common_func import try_training_on_gpu, AgentStatus
except:
    from common_func import try_training_on_gpu, AgentStatus


class Q_Learner:
    """
    Basic Q-Agent that uses a Q-Table
    
    Args:
        env (object): Takes in a GYM environment, use the common_env to simulate the HIPE-Dataset.
        memory_len (int): Sets the memory length of a deque object. The deque object is used to store past steps, so they can be learned all at once. Use this based on your update number and possible on the multi-step-, lstm-hotizon.
        gamma (float): Factor that determines the importance of futue Q-values, value between 0 and 1
        epsilon (float): Initial percent of random actions, value between 0 and 1
        epsilon_min (float): Minimal percent of random actions, value between 0 and 1
        epsilon_decay (float): Factor by which epsilon decays, value between 0 and 1. Can also be set to `linear` when you want epsilon to decrease over all steps (In this case ``epsilon_min`` will not be used).
        lr (float): Sets the learning rate of the RL-Agent
        tau (float): Factor for copying weights from model network to target network
        Q_table (array or list): Will be intepreted as a Q-Table when passing an array (all values should be initially set to zero) or will create a Q-Table when passing a list (list must be the shape for the Q-Table).
        load_table (string): Name of an existing Q-Table in `[D_PATH]/agent-models`. Loads a pre-trained table (Note that the table must be in .h5 format).
    """
    def __init__(self, env, memory_len, gamma=0.85, epsilon=0.8, epsilon_min=0.1, epsilon_decay=0.999996, lr=0.5, tau=0.125, Q_table=[22,22,22,22,22], load_table=None):


        self.env            = env

        self.D_PATH         = self.env.__dict__['D_PATH']
        self.NAME           = self.env.__dict__['NAME']

        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay # string:'linear' or float
        self.epsilon_og     = epsilon
        self.lr             = lr
        self.tau            = tau

        # Create or loads Q-Table:
        if isinstance(Q_table, list) and load_table == None:
            self.Q_table    = np.zeros(Q_table)
        elif load_table == None:
            self.Q_table        = Q_table # each dimension ∈ [0,0.05,...,1] with standart settings
        else:
        	with h5py.File(D_PATH+'agent-models/'+load_table, 'r') as hf:
                self.Q_table = hf[:][:]

        # Pass logger object:
        self.LOGGER         = self.env.__dict__['LOGGER']
        
        # Check horizon lenght and create memory deque object:
        self.horizon = self.env.__dict__['reward_maker'].__dict__['R_HORIZON']
        if isinstance(self.horizon, int) == True:
            memory_len += self.horizon
        else:
            self.horizon = 0
        self.memory = deque(maxlen=memory_len)

        # Check action type:
        if self.env.__dict__['ACTION_TYPE'] != 'discrete':
            raise Exception("Q-Table-Agent can not use continious values for the actions. Change 'ACTION_TYPE' to 'discrete'!")

        # Check observation type:
        if self.env.__dict__['OBS_TYPE'] != 'discrete':
            raise Exception("Q-Table-Agent can not use continious values for the observations. Change 'OBS_TYPE' to 'discrete'!")


    def init_agent_status(self, epochs, epoch_len):
        self.agent_status = AgentStatus(epochs*epoch_len)


    def act(self, state, random_mode=False):
        '''
        Function, in which the agent decides an action, either from greedy-policy or from prediction. Use this function when iterating through each step.
        
        Args:
            state (array): Current state at the step

        Returns:
            action, epsilon (tuple):
            
            action (integer): Action that was chosen by the agent
            
            epsilon (float): Current (decayed) epsilon
        '''
        # Calculate new epsilon:
        if self.epsilon_decay == 'linear':
            self.epsilon = self.epsilon_og*(self.env.__dict__['sum_steps']/self.agent_status.__dict__['max_steps'])
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        # Random decision if action will be random:
        if np.random.random() < self.epsilon or random_mode==True:
            return self.env.action_space.sample()  # = random action
        
        # If action is not random:
        # state[0] : Power_demand
        # state[1] : SoC_SMS
        # state[2] : SoC_LiON
        state = state.reshape(len(state),1).tolist()
        return np.argmax(self.Q_table[state][0]) # = action


    def remember(self, state, action, reward, new_state, done, step_counter_episode):
        '''
        Takes in all necessery variables for the learning process and appends those to the memory. Use this function when iterating through each step.

        Args:
            state (array): State at which the agent chose the action
            action (integer): Chosen action
            reward (float): Real reward for the action
            new_state (array): State after action is performed
            done (bool): If True the episode ends
            step_counter_episode (integer): Episode step at which the action was performed
        '''
        state = state.reshape(len(state),1).tolist()
        new_state = new_state.reshape(len(new_state),1).tolist()
        self.memory.append([state, action, reward, new_state, done, step_counter_episode])


    def replay(self, batch_size):
        '''
        Training-Process for the DQN from past steps. Use this function after a few iteration-steps (best use is the update_num, alternatively use this function at each step).

        Args:
            batch_size (integer): Number of past states to learn from
        '''
        # Training:
        loss_list = []
        for i in range(batch_size):
            # Load single state:
            state, action, reward, new_state, done, step_counter_episode = self.memory[i] 
            
            # Check for multi-step rewards:
            if reward == None:
                reward = self.env.get_multi_step_reward(step_counter_episode)

            # Check if epoch is finished:
            if done:
                Q_future = reward
            else:
                Q_future = max(self.Q_table[new_state][0])
            
            state_and_action = np.append(state,action)
            state_and_action = state_and_action.reshape(len(state_and_action),1).tolist()
            
            # Update table with q-function:
            Q_target = self.Q_table [state_and_action]
            loss = (self.lr * (reward + (Q_future * self.gamma) - Q_target))
            self.Q_table[state_and_action] += loss
            loss_list.append(loss) 

        # Prepare agent status to log and print:
        name         = self.env.__dict__['NAME']
        epoch        = self.env.__dict__['episode_counter']
        total_steps  = self.env.__dict__['sum_steps']

        # Print status
        self.agent_status.print_agent_status(name, epoch, total_steps, batch_size, self.epsilon, np.mean(loss_list))         
        
        # Log status:
        self.LOGGER.log_scalar('Loss:', loss, total_steps, done)
        self.LOGGER.log_scalar('Epsilon:', self.epsilon, total_steps, done)


    def save_agent(self, e):
        '''For saving the agents model at specific epoch. Make sure to not use this function at each epoch, since this will take up your memory space.

        Args:
            e (integer): Takes the epoch-number in which the model is saved
        '''
        with h5py.File(self.D_PATH+'agent-models/'+self.NAME+'_{}.h5'.format(e), 'w') as hf:
            hf.create_dataset(self.NAME, data=self.Q_table)
