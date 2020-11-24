import pandas as pd
import numpy as np

import tensorflow as tf
import keras


'''
TENSORBOARD:
Kommando in neue Konsole eingeben um TensorBoard zu benutzen:
tensorboard --logdir=_small_d/logs/
tensorboard --logdir=_BIG_D/logs/
'''

# TO-DO:
# Printer Function
# tray training on gpu für jedes modul implementieren

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

def print_progress(proess_name, max_i, i):
    if i <= max_i:
        percentage = i/max_i*100
        temp_line_print(proess_name+': {}%'.format(percentage))
    else:
        print(proess_name+': 100%')

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








