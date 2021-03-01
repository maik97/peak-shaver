'''Various tests for parameter tuningof the wahrsager LSTM-Network'''
from common_settings import basic_dataset
from main.wahrsager import wahrsager
from main.schaffer import lstmInputDataset

# Load the dataset:
df, power_dem_df, main_dataset = basic_dataset()

def standart_settings():
    '''Creates standart results to compare the tests'''

    print('Testing standart_settings()')
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = 'NORMAL', 
        NAME ='_test_standart',
        #PLOTTING = True,
        # Model-Parameter:
        num_outputs       = 1,
        dropout           = 0.1,
        recurrent_dropout = 0.1,
        num_hidden        = 3,
        lstm_size         = 128,
        first_hidden_size = 128,
        neuron_num_change = 0.5,
        activation_hidden = 'relu',
        activation_end    = 'relu',
        lr                = 0.001,
        # Trainings-Parameter
        val_data_size     = 2000,
        num_epochs        = 200,
        ).train()


def test_learning_rate():
    '''Tests different learning rates'''

    print('Testing test_learning_rate()')
    learning_rate_list = [0.00001,0.0001,0.001,0.01,0.1,0.5]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for learning_rate in learning_rate_list:
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE       = 'NORMAL', 
            NAME       = '_test_learning_rate_{}'.format(learning_rate),
            lr         = learning_rate,
            num_epochs = 1000,
            PLOTTING   = True,
           
        ).train()


def test_dropout():
    '''Tests overfitting by comparing training loss to validation loss'''

    print('Testing test_dropout()')
    dropout_list = [0.1,0.2,0.4,0.6]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for dropout in dropout_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE              = 'NORMAL', 
            NAME              ='_test_dropout_{}'.format(dropout),
            dropout           = dropout,
            recurrent_dropout = dropout,
            num_epochs        = 1000,
            PLOTTING          = True,
            ).train()



def test_sigmoid():
    ''' Tests difference between relu and sigmoid activation function'''
    
    print('Testing test_sigmoid()')
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = 'NORMAL', 
        NAME ='_test_sigmoid',
        activation_hidden = 'sigmoid',
        activation_end    = 'sigmoid',
        PLOTTING = True,
        ).train()


def test_lstm_layers():
    '''Tests difference between different numbers of lstm layers'''

    print('Testing test_lstm_layers()')
    lstm_layers_list = [16,32,64,256,512,1028]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for lstm_layers in lstm_layers_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_lstm_layers_{}'.format(lstm_layers),
            # Model-Parameter:
            lstm_size         = lstm_layers,
            first_hidden_size = lstm_layers,
            PLOTTING = True,
            ).train()


def test_hidden_layers():
    '''Tests difference between different numbers of hidden layers'''

    print('Testing test_hidden_layers()')
    hidden_layers_list = [1,2]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for hidden_layers in hidden_layers_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_hidden_layers_{}'.format(hidden_layers),
            num_hidden = hidden_layers,
            PLOTTING = True,
            ).train()


def test_past_periods():
    ''' Tests difference between different numbers of past periods'''
    
    print('Testing test_past_periods()')
    past_periods_list = [6,24]

    for past_periods in past_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=past_periods)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_past_periods_{}'.format(past_periods),
            PLOTTING = True,
            ).train()


def test_mean():
    ''' Tests difference between different numbers of past periods'''

    print('Testing test_mean()')
    past_periods_list = [6,12,24]

    for past_periods in past_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=past_periods)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MEAN', 
            NAME ='_test_mean_{}'.format(past_periods),
            #PLOTTING = True,
            ).train()


def test_max():
    ''' Tests difference between different numbers of past periods'''

    print('Testing test_max()')
    past_periods_list = [6,12,24]

    for past_periods in past_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=past_periods)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MAX', 
            NAME ='_test_max_{}'.format(past_periods),
            #PLOTTING = True,
            ).train()


def test_max_label_seq():
    ''' Tests difference between different numbers of past periods'''

    print('Testing test_max_label_seq()')
    output_periods_list = [6,12,24]

    for output_periods in output_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MAX_LABEL_SEQ', 
            NAME ='_test_max_label_seq_{}'.format(output_periods),
            num_outputs = output_periods,
            #PLOTTING = True,
            ).train()


def test_mean_label_seq():
    ''' Tests difference between different numbers of past periods'''

    print('Testing test_mean_label_seq()')
    output_periods_list = [6,12,24]

    for output_periods in output_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MEAN_LABEL_SEQ', 
            NAME ='_test_mean_label_seq_{}'.format(output_periods),
            num_outputs = output_periods,
            #PLOTTING = True,
            ).train()


def test_seq():
    ''' Tests difference between different numbers of past periods'''

    print('Testing test_seq()')
    output_periods_list = [6,12,24]

    for output_periods in output_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'SEQ', 
            NAME ='_test_seq_{}'.format(output_periods),
            num_outputs = output_periods,
            #PLOTTING = True,
            ).train()

'''
standart_settings()
#test_learning_rate()
#test_dropout()
test_sigmoid()
test_lstm_layers()
test_hidden_layers()
test_past_periods()

try:
    test_mean()
except Exception as e:
    print(e)

try:
    test_max()
except Exception as e:
    print(e)

try:
    test_max_label_seq()
except Exception as e:
    print(e)

try:
    test_mean_label_seq()
except Exception as e:
    print(e)

try:
    test_seq()
except Exception as e:
    print(e)
'''

def final_normal():
    '''Creates standart results to compare the tests'''

    print('Testing final_normal()')
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=24)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = 'NORMAL', 
        NAME ='_test_standart',
        #PLOTTING = True,
        # Model-Parameter:
        num_outputs       = 1,
        dropout           = 0.1,
        recurrent_dropout = 0.1,
        num_hidden        = 3,
        lstm_size         = 256,
        first_hidden_size = 256,
        neuron_num_change = 0.5,
        activation_hidden = 'relu',
        activation_end    = 'relu',
        lr                = 0.001,
        # Trainings-Parameter
        val_data_size     = 2000,
        num_epochs        = 1000,
        ).train()

def final_seq():
    '''Creates standart results to compare the tests'''

    print('Testing final_seq()')
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=24)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = 'MAX_LABEL_SEQ', 
        NAME ='_test_standart',
        #PLOTTING = True,
        # Model-Parameter:
        num_outputs       = 24,
        dropout           = 0.1,
        recurrent_dropout = 0.1,
        num_hidden        = 3,
        lstm_size         = 256,
        first_hidden_size = 256,
        neuron_num_change = 0.5,
        activation_hidden = 'relu',
        activation_end    = 'relu',
        lr                = 0.001,
        # Trainings-Parameter
        val_data_size     = 2000,
        num_epochs        = 1000,
        ).train()



for i in range(3):
	#final_normal()
	final_seq()