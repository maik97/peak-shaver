import os
import time
import numpy as np

import tensorflow as tf
#import keras


'''
TENSORBOARD:
Kommando in neue Konsole eingeben um TensorBoard zu benutzen:
tensorboard --logdir=_small_d/logs/
tensorboard --logdir=_BIG_D/logs/
'''


def make_dir(new_dir):
    '''
    Creates a new directory, if the directory does not already exist.

    Args:
        new_dir (string): Name of the directory
    '''
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def try_training_on_gpu():
    '''
    Checks if Keras can use a GPU and switch to GPU usage if possible
    '''
    try:
        config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56} ) 
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)
    except:
        print('Keras could not setup a GPU session. CPU will be used instead.')

def temp_line_print(text):
    '''
    Prints a temporary line

    Args:
        text (string): Text that will be printed temporarly
    '''
    print(text, end='')
    print('\b' * len(text), end='', flush=True)

def print_progress(process_name, i, max_i):
    '''
    Prints the progress of an iteration

    Args:
        process_name (string): Name of the process
        i (int): Current iteration step
        max_i (int): The number of all iterations
    '''
    if i < max_i:
        percentage = round(i/max_i*100)
        temp_line_print(process_name+': {}%'.format(percentage))
    else:
        print(process_name+': 100%')

def max_seq(seq_data):
    '''
    Calculates the (rolling) maximum of a 2 dimensional array. 

    Args:
        seq_data (array): 2-dimensianal array

    Returns:
        array: One dimensional array (index 1 is transformed to max)
    '''
    max_data = []
    for i in range(len(seq_data)):
        max_data.append(np.max(seq_data[i]))
    return np.asarray(max_data, dtype=np.float32)

def mean_seq(seq_data):
    '''
    Calculates the (rolling) mean of a 2 dimensional array. 

    Args:
        seq_data (array): 2-dimensianal array

    Returns:
        array: One dimensional array (index 1 is transformed to mean)
    '''
    mean_data = []
    for i in range(len(seq_data)):
        mean_data.append(np.mean(seq_data[i]))
    return np.asarray(mean_data, dtype=np.float32)

def wait_to_continue(text='Press Enter to continue.'):
    '''
    Simple helper function that pauses the process till you press enter
    '''
    wait_for = input(text)


class Timer:
    '''
    Used to record the time of a process and make estimations about the time till completion
    '''

    def start(self):
        '''Starting point for time recording'''
        self.tic = time.perf_counter()

    def stop(self):
        '''Calculates the time difference to the starting time'''
        self.toc = time.perf_counter()
        self.elapsed_time = self.toc - self.tic

    def time_format(self, time_float):
        '''
        Formats the time to a readable format

        Args:
            time_float (float): Time in float format

        Returns:
            strings: Time in readable format
        '''
        hours, rem = divmod(time_float, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

    def elapsed_time_string(self):
        '''
        Used to pass the elapsed time as a string

        Returns:
            string: The elapsed time in a readable format
        '''

        return self.time_format(self.elapsed_time)
        #return time.strftime('{%H:%M:%S}'.format(self.elapsed_time))

    def rest_time_string(self, i, max_i):
        '''
        Calculates the estimated time from now till completion

        Args:
            i (int): Current iteration step
            i_max (int): 
        '''
        if i != 0:
            rest_time = self.elapsed_time * ((max_i - i)/i)
            return self.time_format(rest_time)
        else:
            return '-'
    
    def print_time_progress(self, process_name, i, max_i, temp_line=True):
        '''
        Prints the percentage of the progress and the estimated time left

        Args:
            process_name (string): Name of the process
            i (int): Current iteration step
            max_i (int): The number of all iterations
            temp_line (bool): If true the line printed will be temporary
        '''
        if i < max_i:
            percentage = round(i/max_i*100)
            self.stop()
            if temp_line==True:
                temp_line_print(process_name+': {}%, estimated time remaining: {}'.format(percentage, self.rest_time_string(i,max_i)))
            else:
                print(process_name+': {}%, estimated time remaining: {}'.format(percentage, self.rest_time_string(i,max_i)))
        else:
            print(process_name+': 100%')

    def print_progress(self, text, i, max_i):
        '''
        Prints the progress of an iteration

        Args:
            process_name (string): Name of the process
            i (int): Current iteration step
            max_i (int): The number of all iterations
        '''
        if i < max_i:
            percentage = round(i/max_i*100)
            print(text+': {}%'.format(percentage))
        else:
            print(text+': 100%')

    def print_eta(self, text, i, max_i):
        ''' Prints the estimated time left
        
        Args: 
            text (string): For example `Estimated Time:`
            i (int): Current iteration step
            max_i (int): The number of all iterations
        '''
        if i != 0:
            rest_time = self.elapsed_time * ((max_i - i)/i)
            print(text,self.time_format(rest_time))
        else:
            print(text,'-')

    def print_el_time(self, text):
        ''' Prints the estimated time left
        
        Args: 
            text (string): For example the `Elapsed Time:`
        '''
        print(text,self.elapsed_time_string())



class AgentStatus:

    def __init__(self,max_steps):
        '''
        Can print the stats of a learning process

        Args:
            max_steps (int): Sum of all steps over all epochs
        '''
        self.timer     = Timer()
        self.max_steps = max_steps
        self.timer.start()
    
    def print_agent_status(self, name, epoch, total_steps, batch_size, epsilon, loss=None):
        '''
        Prints the current stats of a learning process

        Args:
            name (string): Name of the agent
            epoch (int): Current epoch
            total_steps (int): Steps passed so far
            batch_size (int): Batch size used in the learning process
            epsilon (float): Current epsilon
            loss (float): Current loss
        '''
        self.timer.stop()
        print('')
        print('# Name:',name)
        print('# Epoch:',epoch)
        print('# Total steps:',total_steps)
        print('# Batch size:',batch_size)
        print('# Epsilon:', round(epsilon,5))
        if loss != None:
            print('# Loss:', round(loss,5))

        self.timer.print_progress('# Progress:',total_steps,self.max_steps)
        self.timer.print_el_time('# Elapsed time:')
        self.timer.print_eta('# Estimated time remaining:',total_steps,self.max_steps)


def training(agent, epochs=1000, update_num=1000, num_warmup_steps=100, epoch_save_intervall=1000, random_start=True, SoC_full=False):
    '''
    General function for the learning process, which can be used by all agents except those from ``stable-baselines``
    
    Args:
        epochs (int): Number of epochs
        update_num (int): Interval of steps after which the training process starts
        epoch_save_intervall (int): Interval of steps after which the agent will be saved
        random_start (bool): If set to `True` the epoch will start at a random point in the dataset
        SoC_full (bool): If set to `True` the batteries will be full at the beginning of an epoch
    '''
    horizon   = agent.__dict__['horizon']
    env       = agent.__dict__['env']
    epoch_len = len(env.__dict__['df'])

    agent.init_agent_status(epochs,epoch_len)

    print('Warmup-Steps per Episode:', num_warmup_steps)
    print('Training for',epochs,'Epochs')

    for e in range(epochs):

        if random_start == False:
            cur_state = env.set_soc_and_current_state(SoC_full)
        else:
            cur_state = env.reset()

        update_counter   = 0
        warmup_counter   = 0

        while warmup_counter < num_warmup_steps:
            action                          = agent.act(cur_state, random_mode=True )
            new_state, reward, done, sce, _ = env.step(action)
            new_state                       = new_state
            agent.remember(cur_state, action, reward, new_state, done, sce)

            cur_state = new_state
            warmup_counter += 1

        for step in range(epoch_len):

            action                          = agent.act(cur_state)
            new_state, reward, done, sce, _ = env.step(action)
            new_state                       = new_state
            agent.remember(cur_state, action, reward, new_state, done, sce)
            
            cur_state = new_state
            
            if done == False:
                training_num = update_num
            else:
                training_num = update_num + horizon

            update_counter += 1
            if update_counter == update_num or done == True:
                agent.replay(training_num)
                update_counter = 0

            if done:
                break

        if e % epoch_save_intervall == 0:
            agent.save_agent(e)


def testing(agent, epochs=1, random_start=False, SoC_full=True, use_stable_b=False, env=None):
    '''
    General function for the learning purposes, which can be used by all agents except those from ``stable-baselines``
    
    Args:
        random_start (bool): If set to `True` the epoch will start at a random point in the dataset
        SoC_full (bool): If set to `True` the batteries will be full at the beginning of an epoch
        use_stable_b (bool): Set this to `True` if you are using a model from ``stable-baselines``
        df (dataframe): Needs to be passed if xou are using ``stable-baselines``
    '''
    if use_stable_b == False:
        env       = agent.__dict__['env']
        epoch_len = len(env.__dict__['df'])
    else:
        epoch_len = len(env.__dict__['df'])

    print('Testing for {} epochs...'.format(epochs))

    if random_start == False:
        cur_state = env.set_soc_and_current_state(SoC_full)
    else:
        cur_state = env.reset()

    for e in range(epochs):
        for step in range(epoch_len):

            if use_stable_b == False:
                action                          = agent.act(cur_state, test_mode=True)
                new_state, reward, done, sce, _ = env.step(action)
                new_state                       = new_state
                agent.remember(cur_state, action, reward, new_state, done, sce)

            else:
                action, _states = agent.predict(cur_state)
                new_state, reward, done, _ = env.step(action)


            cur_state = new_state

            if done:
                break