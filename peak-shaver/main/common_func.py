import os
import time
import pandas as pd
import numpy as np

import tensorflow as tf
#import keras


'''
TENSORBOARD:
Kommando in neue Konsole eingeben um TensorBoard zu benutzen:
tensorboard --logdir=_small_d/logs/
tensorboard --logdir=_BIG_D/logs/
'''

# TO-DO:
# Printer Function
# tray training on gpu für jedes modul implementieren

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def try_training_on_gpu():
    try:
        # Für Keras GPU machen:
        config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)
    except:
        print('Keras could not setup a GPU session. CPU will be used instead.')

def temp_line_print(text):
    print(text, end='')
    print('\b' * len(text), end='', flush=True)

def print_progress(process_name, i, max_i):
    if i < max_i:
        percentage = round(i/max_i*100)
        temp_line_print(process_name+': {}%'.format(percentage))
    else:
        print(process_name+': 100%')

def max_seq(seq_data):
    max_data = []
    for i in range(len(seq_data)):
        max_data.append(np.max(seq_data[i]))
    return np.asarray(max_data, dtype=np.float32)

def mean_seq(seq_data):
    mean_data = []
    for i in range(len(seq_data)):
        mean_data.append(np.mean(seq_data[i]))
    return np.asarray(mean_data, dtype=np.float32)

def wait_to_continue(text='Press Enter to continue.'):
    wait_for = input(text)


class Timer:

    def start(self):
        self.tic = time.perf_counter()

    def stop(self):
        self.toc = time.perf_counter()
        self.elapsed_time = self.toc - self.tic

    def time_format(self, time_float):
        hours, rem = divmod(time_float, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

    def elapsed_time_string(self):
        return self.time_format(self.elapsed_time)
        #return time.strftime('{%H:%M:%S}'.format(self.elapsed_time))

    def rest_time_string(self, i, max_i):
        if i != 0:
            rest_time = self.elapsed_time * ((max_i - i)/i)
            return self.time_format(rest_time)
        else:
            return '-'
    
    def print_time_progress(self, process_name, i, max_i):
        if i < max_i:
            percentage = round(i/max_i*100)
            self.stop()
            temp_line_print(process_name+': {}%, estimated time left: {}'.format(percentage, self.rest_time_string(i,max_i)))
        else:
            print(process_name+': 100%')


