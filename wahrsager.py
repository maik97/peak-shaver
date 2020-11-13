# VERSION 5

# Allgemein Module
import pandas as pd
import numpy as np
import datetime
import tqdm
import tensorflow as tf
import keras
import time
import matplotlib as mpl
import matplotlib.pyplot  as plt

from keras import backend as K
from keras.models import load_model
from matplotlib.pyplot import *

from keras.optimizers import RMSprop, adam
from keras.models import Sequential
from keras.layers import Dense, InputLayer, LSTM, Dropout
from keras.callbacks import TensorBoard


# Module aus selben Ordner
import schaffer

'''
TENSORBOARD:
Kommando in neue Konsole eingeben um TensorBoard zu benutzen:
tensorboard --logdir=_small_d/logs/
tensorboard --logdir=_BIG_D/logs/
'''

def try_training_on_gpu():
    try:
        # Für Keras GPU machen:
        config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)
    except:
        print('Keras konnte GPU Session nicht einrichten')

def teste_ordung_training_label_data(training_data,label_data):
    test_data = []
    for i in range(len(training_data)):
        test_data.append(training_data[i][-1][10])

    test_df = pd.DataFrame({
                'training': test_data,
                'label': label_data,
                'Nicht NULL!': abs(test_data - label_data),
                })
    print(test_df)
    print(np.shape(training_data))
    print(np.shape(label_data))

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

class wahrsager:

    def __init__(
                self,
                # Allgemeine Parameter
                TYPE              = 'MEAN',
                NAME              = '_wahrsager_v5',
                PLOT_MODE         = False,
                
                # Model-Parameter
                num_past_periods  = 12,
                num_inputs        = 24,
                num_outputs       = 1,
                dropout           = 0.2,
                recurrent_dropout = 0.2,
                activation_hidden = 'sigmoid',
                activation_end    = 'sigmoid',
                num_hidden        = 3,
                lstm_size         = 128,
                first_hidden_size = 128,
                neuron_num_change = 0.5,
                
                # Trainings-Parameter
                großer_datensatz  = True,
                val_data_size     = 2000,
                num_epochs        = 1,
                ):

        # Allgemeine Parameter
        self.TYPE              = TYPE
        self.NAME              = NAME
        self.PLOT_MODE         = PLOT_MODE   
        self.großer_datensatz  = großer_datensatz
        
        # Model-Parameter
        self.num_past_periods  = num_past_periods
        self.num_inputs        = num_inputs
        self.num_outputs       = num_outputs
        self.dropout           = dropout
        self.recurrent_dropout = recurrent_dropout
        self.activation_hidden = activation_hidden
        self.activation_end    = activation_end
        self.num_hidden        = num_hidden
        self.lstm_size         = lstm_size
        self.first_hidden_size = first_hidden_size
        self.neuron_num_change = neuron_num_change

        # Trainings-Parameter:
        self.val_data_size     = val_data_size
        self.num_epochs        = num_epochs

        # Überprüfe ob num_outputs bei seq passt:
        if self.num_outputs == 1 and TYPE == 'SEQ':
            print('\nAchtung: Parameter num_outputs fehlt!')
            print('Setzte num_outputs standartmäßig auf 12\n')
            self.num_outputs = 12

        # Entscheidung für Wahl des Datensatzes
        if self.großer_datensatz == False:
            self.DATENSATZ_PATH = '_small_d/'
            self.VERSION        = 'small-{}'.format(int(time.time()))
        else:
            self.DATENSATZ_PATH = '_BIG_D/'
            self.VERSION        = 'big-{}'.format(int(time.time()))

        # Einstellungen Pandas:
        pd.set_option('min_rows', 50)

        # Einstellungen Schaffer
        schaffer.global_var(self.NAME, self.VERSION, self.DATENSATZ_PATH, self.großer_datensatz)

        # Init GPU Training
        try_training_on_gpu()

        # Importiere Power Demand
        self.total_power = schaffer.load_total_power()

        # Maximum, um später normalisierte Daten wieder zu ent-normalisieren:
        self.max_total_power = np.max(self.total_power.to_numpy())
        #print('MAXIMUM:',self.max_total_power)

    def save_parameter(self,path_and_name):
        f = open(path_and_name, 'w')

        f.write('Allgemeine Parameter\n')
        f.write('TYPE:              %s\n' % self.TYPE)
        f.write('NAME:              %s\n' % self.NAME)
        f.write('PLOT_MODE:         %s\n' % self.PLOT_MODE)
        f.write('großer_datensatz:  %s\n' % self.großer_datensatz)
        
        f.write('\nModel-Parameter\n')
        f.write('num_past_periods:  %s\n' % self.num_past_periods)
        f.write('num_inputs:        %s\n' % self.num_inputs)
        f.write('num_outputs:       %s\n' % self.num_outputs)
        f.write('dropout:           %s\n' % self.dropout)
        f.write('recurrent_dropout: %s\n' % self.recurrent_dropout)
        f.write('activation_hidden: %s\n' % self.activation_hidden)
        f.write('activation_end:    %s\n' % self.activation_end)
        f.write('num_hidden:        %s\n' % self.num_hidden)
        f.write('lstm_size:         %s\n' % self.lstm_size)
        f.write('first_hidden_size: %s\n' % self.first_hidden_size)
        f.write('neuron_num_change: %s\n' % self.neuron_num_change)

        f.write('\nTrainings-Parameter\n')
        f.write('val_data_size:     %s\n' % self.val_data_size)
        f.write('num_epochs:        %s\n' % self.num_epochs)

        f.close()

    def import_data(self):

        if self.TYPE == 'MEAN':
                training_data, label_data = schaffer.rolling_mean_training_data(self.num_past_periods)
        elif self.TYPE == 'MAX':
                training_data, label_data = schaffer.rolling_max_training_data(self.num_past_periods)
        elif self.TYPE == 'NORMAL':
                training_data, label_data = schaffer.normal_training_data(self.num_past_periods)
        elif self.TYPE == 'SEQ':
                training_data, label_data = schaffer.sequence_training_data(self.num_past_periods)
        elif self.TYPE == 'MAX_LABEL_SEQ':
                training_data, label_data_seq = schaffer.sequence_training_data(self.num_past_periods)
                label_data = max_seq(label_data_seq)    
        elif self.TYPE == 'MEAN_LABEL_SEQ':
                training_data, label_data_seq = schaffer.sequence_training_data(self.num_past_periods)
                label_data = mean_seq(label_data_seq)
        else:
            print("Error: Data-Import TYPE not understood! - Supported TYPES:'MEAN','MAX','NORMAL','SEQ','MAX_LABEL_SEQ','MEAN_LABEL_SEQ'")
            exit()

        return training_data, label_data


    def lstm_model(self):
        '''
        self.num_past_periods  = 12
        self.num_inputs        = 24
        self.num_outputs       = 1
        self.dropout           = 0.2
        self.recurrent_dropout = 0.2
        self.activation_hidden = 'sigmoid'
        self.activation_end    = 'sigmoid'
        self.num_hidden        = 3
        self.lstm_size         = 128
        self.first_hidden_size = 128
        self.neuron_num_change = 0.5
        '''
    
        model = Sequential()
        model.add(LSTM(self.lstm_size, 
                       input_shape       = (self.num_past_periods,self.num_inputs),
                       dropout           = self.dropout,
                       recurrent_dropout = self.recurrent_dropout))
        
        neuron_size = self.first_hidden_size
        for i in range(self.num_hidden):
            model.add(Dense(int(neuron_size), activation = self.activation_hidden))
            model.add(Dropout(self.dropout))
            neuron_size *= self.neuron_num_change

        model.add(Dense(self.num_outputs, activation = self.activation_end))  
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        #model.compile(loss='mse', optimizer=adam(lr=0.0001, decay=1e-6), metrics=['mse'])
        return model


    def train_model(self, training_data, label_data, model_name):

        training_data = np.nan_to_num(training_data)
        label_data    = np.nan_to_num(label_data)
        
        # Tensorboard:
        tensorboard = TensorBoard(log_dir=self.DATENSATZ_PATH+'logs/'+model_name+'_val-size-{}_epochs-{}_'.format(self.val_data_size,self.num_epochs)+self.VERSION)

        # Erstelle Keras Model:
        model = self.lstm_model()

        # Trainiere:
        model.fit(training_data[:-self.val_data_size], label_data[:-self.val_data_size],
                  epochs          = self.num_epochs,
                  verbose         = 1,
                  validation_data = (training_data[-self.val_data_size:], label_data[-self.val_data_size:]),
                  callbacks       = [tensorboard]
                  )
        
        # Speichern:
        model.save(self.DATENSATZ_PATH+'LSTM-models/'+model_name+'_val-size-{}_epochs-{}_'.format(self.val_data_size,self.num_epochs)+self.VERSION+'.h5')
        self.save_parameter(self.DATENSATZ_PATH+'LSTM-models/'+model_name+'_val-size-{}_epochs-{}_'.format(self.val_data_size,self.num_epochs)+self.VERSION+'.txt')

        # Vorhersehen:
        return model.predict(training_data).reshape(np.shape(label_data))

    def predict_model(self, training_data, label_data, model_name):
    
        training_data = np.nan_to_num(training_data)
        label_data    = np.nan_to_num(label_data)
        
        model = load_model(self.DATENSATZ_PATH+'LSTM-models/'+model_name+'.h5')
        return model.predict(training_data).reshape(np.shape(label_data))


    def plot_pred_single(self,prediction,label_data):
        # Ausgabe Dataframe Vorhersage und richtige Werte:
        prediction_df = pd.DataFrame({
                'Vorhersage'        : prediction * self.max_total_power,
                'Richtige Werte'    : label_data * self.max_total_power,
                'Absoluter Fehler'  : abs(prediction - label_data) * self.max_total_power,
                'Quadrierter Fehler': (prediction - label_data)*(prediction - label_data) * self.max_total_power
                })

        print(prediction_df)

        print('Maximaler Absoluter Fehler:',  np.max(prediction_df['Absoluter Fehler'].to_numpy()))
        print('Mittelwert Absoluter Fehler:', np.mean(prediction_df['Absoluter Fehler'].to_numpy()))

        # Plotte Dataframe
        prediction_df.plot()
        plt.show()

    
    def plot_pred_multiple(self,prediction,label_data):
        prediction = np.transpose(prediction)
        label_data = np.transpose(label_data)

        for i in range(self.num_outputs):
            prediction_df = pd.DataFrame({
                'Vorhersage'        : prediction[i] * self.max_total_power,
                'Richtige Werte'    : label_data[i] * self.max_total_power,
                'Absoluter Fehler'  : abs(prediction[i] - label_data[i]) * self.max_total_power,
                'Quadrierter Fehler': (prediction[i] - label_data[i])*(prediction[i] - label_data[i]) * self.max_total_power
                })

            print(prediction_df)

            print('Maximaler Absoluter Fehler:',  np.max(prediction_df['Absoluter Fehler'].to_numpy()))
            print('Mittelwert Absoluter Fehler:', np.mean(prediction_df['Absoluter Fehler'].to_numpy()))

            # Plotte Dataframe
            prediction_df.plot()
            plt.show()



    def train(self):
        training_data, label_data = self.import_data()

        prediction = self.train_model(training_data, label_data, self.TYPE+self.NAME)

        if self.PLOT_MODE == True:
            if self.num_outputs > 1:
                self.plot_pred_multiple(prediction,label_data)
            else:
                self.plot_pred_single(prediction,label_data)
        else:
            print('\nWahrsager: Plotting is disabled.')

        return prediction

    def pred(self):
        training_data, label_data = self.import_data()

        try:
            prediction = self.predict_model(training_data, label_data, self.TYPE+'_MODEL')
        except:
            print('Please name a model: '+self.TYPE+'_MODEL.h5! (Path: '+self.DATENSATZ_PATH+'LSTM-models/)')
            exit()

        if self.PLOT_MODE == True:
            if self.num_outputs > 1:
                self.plot_pred_multiple(prediction,label_data)
            else:
                self.plot_pred_single(prediction,label_data)
        else:
            print('\nWahrsager: Plotting is disabled.')
        return prediction


def predictions_and_inputs():
    df = schaffer.alle_inputs_neu()[24:-12]
    
    df['pred_mean']       = wahrsager(TYPE='MEAN').pred()[:-12]
    df['pred_max']        = wahrsager(TYPE='MAX').pred()[:-12]
    df['pred_normal']     = wahrsager(TYPE='NORMAL').pred()[:-12]
    df['pred_max_labe']   = wahrsager(TYPE='MAX_LABEL_SEQ').pred()
    df['pred_mean_label'] = wahrsager(TYPE='MEAN_LABEL_SEQ').pred()
    
    prediction_seq        = wahrsager(TYPE='SEQ', num_outputs=12).pred()
    df['max_pred_seq']    = max_seq(prediction_seq)
    df['mean_pred_seq']   = mean_seq(prediction_seq)
    return df


def main():
    '''
    #### Wahrsager: ####
    TYPE             = 'MEAN',
    NAME             = 'wahrsager_v5',
    num_past_periods = 12,
    num_outputs      = 1,
    PLOT_MODE        = False,
    großer_datensatz = True,
    val_data_size    = 2000,
    num_epochs       = 10,
    '''
    
    print('Teste alle Trainings-Möglichkeiten mit den Standart-Einstellungen:')

    prediction_mean           = wahrsager(PLOT_MODE=True, TYPE='MEAN').train()
    prediction_max            = wahrsager(PLOT_MODE=True, TYPE='MAX').train()
    prediction_normal         = wahrsager(PLOT_MODE=True, TYPE='NORMAL').train()
    prediction_max_label_seq  = wahrsager(PLOT_MODE=True, TYPE='MAX_LABEL_SEQ').train()
    prediction_mean_label_seq = wahrsager(PLOT_MODE=True, TYPE='MEAN_LABEL_SEQ').train()

    prediction_seq      = wahrsager(PLOT_MODE=True, TYPE='SEQ', num_outputs=12).train()
    max_prediction_seq  = max_seq(prediction_seq)
    mean_prediction_seq = mean_seq(prediction_seq)

    print('Teste alle Predictions-Möglichkeiten mit den Standart-Einstellungen:')
    print(predictions_and_inputs())

    print('Tests erfolgreich beendet')

if __name__ == "__main__":
    main()





