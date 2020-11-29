'''DQN implementation'''
import numpy as np
import random

from collections import deque

from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

from main.common_func import try_training_on_gpu, AgentStatus

class DQN:
    """
    Deep Q Network
    
    Args:
        env (object): Takes in a ``gym`` environment, use the common_env to simulate the HIPE-Dataset.
        memory (object): Takes in degue object: deque(maxlen=x)
        gamma (float): Factor that determines the importance of futue Q-values, value between 0 and 1
        epsilon (float): Initial percent of random actions, value between 0 and 1
        epsilon_min (float): Minimal percent of random actions, value between 0 and 1
        epsilon_decay (float): Factor by which ``epsilon`` decays, value between 0 and 1
        lr (float): Sets the learning rate of the RL-Agent
        tau (float): Factor for copying weights from model network to target network
        activation (string): Defines Keras activation function for each Dense layer (except the ouput layer) for the DQN
        loss (string): Defines Keras loss function to comile the DQN model
    """
    def __init__(self, env, memory_len, input_sequence=1, gamma=0.85, epsilon=0.8, epsilon_min=0.1, epsilon_decay=0.999996, lr=0.5, tau=0.125,
                 activation='relu', loss='mean_squared_error', model_type='dense', use_model=None, pre_trained_model=None):

        self.env            = env
        
        self.D_PATH         = self.env.__dict__['D_PATH']
        self.NAME           = self.env.__dict__['NAME']
        self.model_type     = model_type

        self.input_sequence = input_sequence
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay # string:'linear' oder float
        self.epsilon_og     = epsilon
        self.tau            = tau

        self.horizon = self.env.__dict__['reward_maker'].__dict__['R_HORIZON']
        if isinstance(self.horizon, int) == True:
            memory_len += self.horizon
        else:
            self.horizon = 0
        self.memory = deque(maxlen=memory_len+self.input_sequence-1)
        
        if pre_trained_model == None:
            if model_type == 'dense' and use_model == None:
                self.model        = self.create_normal_model(lr, activation, loss)
                self.target_model = self.create_normal_model(lr, activation, loss)

            elif model_type == 'lstm' and use_model == None:
                if self.input_sequence <= 1:
                    raise Exception("input_sequence was set to {} but must be > 1, when model_type='lstm'".format(self.input_sequence))
                self.model        = self.create_lstm_model(lr, activation, loss)
                self.target_model = self.create_lstm_model(lr, activation, loss)

            elif use_model != None:
                self.model        = use_model
                self.target_model = clone_model(self.model)
                self.model_type   = 'costum'
            else:
                raise Exception("model_type='"+model_type+"' not understood. Use 'dense', 'lstm' or pass your own compiled keras model with own_model.")
        else:
            self.model = load_model(self.D_PATH+'agent-models/'+pre_trained_model)

        self.LOGGER = self.env.__dict__['LOGGER']

        try_training_on_gpu()

        if self.env.__dict__['ACTION_TYPE'] != 'discrete':
            raise Exception("DQN-Agent can not use continious values for the actions. Change 'ACTION_TYPE' to 'discrete'!")

        if self.input_sequence == 0:
            print("Attention: input_sequence can not be set to 0, changing now to input_sequence=1")
            self.input_sequence = 1 


    def init_agent_status(self, epochs, epoch_len):
        self.agent_status = AgentStatus(epochs*epoch_len)


    def create_normal_model(self, lr, activation, loss):
        '''Creates a Deep Neural Network which predicts Q-Values, when the class is initilized.

        Args:
            activation (string): Previously defined Keras activation
            loss (string): Previously defined Keras loss
        '''
        input_dim = self.env.observation_space.shape[0]*self.input_sequence
        model = Sequential()
        model.add(Dense(518, input_dim=input_dim, activation=activation))
        model.add(Dense(518, activation=activation))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss=loss, optimizer=Adam(lr=lr))
        return model

    def create_lstm_model(self, lr, activation, loss):
        '''Creates an LSTM which predicts Q-Values, when the class is initilized.

        Args:
            activation (string): Previously defined Keras activation
            loss (string): Previously defined Keras loss
        '''
        input_dim = (self.input_sequence, self.env.observation_space.shape[0])
        model = Sequential()
        model.add(LSTM(518, input_dim=input_dim, activation=activation))
        model.add(Dense(518, activation=activation))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss=loss, optimizer=Adam(lr=model_lr))
        return model

    def sequence_states_to_act(self, cur_state):

        state_array = []
        for mem in memory[-self.input_sequence+1:]:
            state, action, reward, new_state, done, step_counter_episode = mem
            state_array = np.append(state_array, state)
        state_array = np.append(state_array, cur_state)

        if self.model_type == 'lstm':
            state_array = state_array.reshape(-1,self.input_sequence,len(cur_state[0]))

        return state_array

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
        if self.epsilon_decay == 'linear':
            self.epsilon = self.epsilon_og*(self.env.__dict__['sum_steps']/self.agent_status.__dict__['max_steps'])
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        if np.random.random() < self.epsilon or random_mode == True:
            return self.env.action_space.sample()
        
        if self.env.__dict__['num_discrete_obs'] > 1:
            shape = self.env.__dict__['num_discrete_obs']
            state = self.discrete_input_space(shape,state)
        else:
            state = state.reshape(1,len(state))

        if self.input_sequence > 1:
            state = self.sequence_states_to_act(state)
        return np.argmax(self.model.predict(state)[0])

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
        if self.env.__dict__['num_discrete_obs'] > 1:
            shape     = self.env.__dict__['num_discrete_obs']
            state     = self.discrete_input_space(shape,state)
            new_state = self.discrete_input_space(shape,state)
        else:
            state     = state.reshape(1,len(state))
            new_state = new_state.reshape(1,len(new_state))
        self.memory.append([state, action, reward, new_state, done, step_counter_episode])

    def sequence_states_for_replay(self,i):

        state_array     = []
        new_state_array = []
        for mem in memory[i:i+self.input_sequence]:
            state, action, reward, new_state, done, step_counter_episode = mem
            state_array     = np.append(state_array, state)
            new_state_array = np.append(new_state_array, new_state)

        if self.model_type == 'lstm':
            state_array     = state_array.reshape(-1,self.input_sequence,len(cur_state[0]))
            new_state_array = new_state_array.reshape(-1,self.input_sequence,len(cur_state[0]))
        
        return state_array, action, reward, new_state_array, done, step_counter_episode


    def replay(self, batch_size):
        '''
        Training-Process for the DQN from past steps. Use this function after a few iteration-steps (best use is the number of index_len). Alternatively use this function at each step.

        Args:
            index_len (integer): Number of past states to learn from
        '''
        state_batch  = []
        target_batch = []
        for i in range(batch_size):
            # Lade Step von Simulation
            if self.input_sequence == 1:
                state, action, reward, new_state, done, step_counter_episode = self.memory[i]
            else:
                state, action, reward, new_state, done, step_counter_episode = self.sequence_states_for_replay(i)

            
            target = self.target_model.predict(state)
            #print(target[0][action])
            
            if reward == None:
                reward = self.env.get_multi_step_reward(step_counter_episode)

            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            state_batch = np.append(state_batch, state)
            target_batch = np.append(target_batch, target)

        state_batch  = state_batch.reshape(-1,len(state[0]))
        target_batch = target_batch.reshape(-1,len(target[0]))

        name         = self.env.__dict__['NAME']
        loss         = self.model.train_on_batch(state_batch,target_batch)
        epoch        = self.env.__dict__['episode_counter']
        total_steps  = self.env.__dict__['sum_steps']
        self.agent_status.print_agent_status(name, epoch, total_steps, batch_size, self.epsilon, loss) 
        self.LOGGER.log_scalar('Loss:', loss, total_steps, done)
        self.LOGGER.log_scalar('Epsilon:', self.epsilon, total_steps, done)


    def target_train(self):
        '''Updates the Target-Weights. Use this function after replay(index_len)'''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_agent(self, e):
        '''For saving the agents model at specific epoch. Make sure to not use this function at each epoch, since this will take up your memory space.

        Args:
            NAME (string): Name of the model
            D_PATH (string): Path to save the model
            e (integer): Takes the epoch-number in which the model is saved
        '''
        self.model.save(self.D_PATH+'agent-models/'+self.model_type+self.NAME+'_{}'.format(e))


    def discrete_input_space(state_dim_size, state):
        '''
        Transform the state to a discrete input (one-hot-encoding)

        Args:
            state_dim_size (shape): Size to which the state will be transformed
            state (array): Takes in a state

        Returns:
            discrete_state (array): Transformed state for discrete inputs
        '''
        discrete_state = []
        for dim in state:
            dim_append = np.zeros((state_dim_size))
            dim_append[dim] = 1
            discrete_state = np.append(discrete_state,dim_append)
        discrete_state = discrete_state.reshape(1,len(discrete_state))
        #print(discrete_state)
        return discrete_state