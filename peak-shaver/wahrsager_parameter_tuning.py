'''Various tests for parameter tuningof the wahrsager LSTM-Network'''
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager

# ----------- test für dropout?

# Load the dataset:
main_dataset = mainDataset(
    D_PATH='_BIG_D/',
    period_string_min='5min',
    full_dataset=True)

# Normalized dataframe:
df = main_dataset.make_input_df(
    drop_main_terminal=False,
    use_time_diff=True,
    day_diff='holiday-weekend')

# Sum of the power demand dataframe (not normalized):
power_dem_df = main_dataset.load_total_power()


def test_learning_rate():
    '''Tests overfitting by comparing training loss to validation loss'''

    learning_rate_list = [0.00001,0.0001,0.001,0.01,0.1,0.5]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for learning_rate in learning_rate_list:
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_num_epochs',
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 200,
            lr                = learning_rate,
        ).train()

def test_overfitting():
    '''Tests overfitting by comparing training loss to validation loss'''

    dropout_list = [0.1,0.2,0.4,0.6]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for dropout in dropout_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_dropout_{}'.format(dropout),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = dropout,
            recurrent_dropout = dropout,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 150,
            ).train()


def standart_settings():
    '''Creates standart results to compare the tests'''

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = 'NORMAL', 
        NAME ='_test_standart',
        #PLOTTING = True,
        # Model-Parameter:
        num_outputs       = 1,
        dropout           = 0.2,
        recurrent_dropout = 0.2,
        num_hidden        = 3,
        lstm_size         = 128,
        first_hidden_size = 128,
        neuron_num_change = 0.5,
        activation_hidden = 'relu',
        activation_end    = 'relu',
        # Trainings-Parameter
        val_data_size     = 2000,
        num_epochs        = 15,
        ).train()


def test_sigmoid():
    ''' Tests difference between relu and sigmoid activation function'''
    
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = 'NORMAL', 
        NAME ='_test_sigmoid',
        #PLOTTING = True,
        # Model-Parameter:
        num_outputs       = 1,
        dropout           = 0.2,
        recurrent_dropout = 0.2,
        num_hidden        = 3,
        lstm_size         = 128,
        first_hidden_size = 128,
        neuron_num_change = 0.5,
        activation_hidden = 'sigmoid',
        activation_end    = 'sigmoid',
        # Trainings-Parameter
        val_data_size     = 2000,
        num_epochs        = 15,
        ).train()

def test_lstm_layers():
    '''Tests difference between different numbers of lstm layers'''

    lstm_layers_list = [16,32,64,256,512,1028]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for lstm_layers in lstm_layers_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL_', 
            NAME ='_test_lstm_layers_{}'.format(lstm_layers),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = lstm_layers,
            first_hidden_size = lstm_layers,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_hidden_layers():
    '''Tests difference between different numbers of hidden layers'''

    hidden_layers_list = [1,2]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for hidden_layers in hidden_layers_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_hidden_layers_{}'.format(hidden_layers),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = hidden_layers,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_dropout():
    '''Tests difference between different numbers of hidden layers'''

    dropout_list = [0.1,0.4,0.6]

    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
    
    for dropout in dropout_list:
        
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_dropout_{}'.format(dropout),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = dropout,
            recurrent_dropout = dropout,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_past_periods():
    ''' Tests difference between different numbers of past periods'''

    past_periods_list = [6,24]

    for past_periods in past_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=past_periods)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'NORMAL', 
            NAME ='_test_past_periods_{}'.format(past_periods),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_mean():
    ''' Tests difference between different numbers of past periods'''

    past_periods_list = [6,12,24]

    for past_periods in past_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=past_periods)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MEAN', 
            NAME ='_test_mean_{}'.format(past_periods),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_max():
    ''' Tests difference between different numbers of past periods'''

    past_periods_list = [6,12,24]

    for past_periods in past_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=past_periods)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MAX', 
            NAME ='_test_max_{}'.format(past_periods),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = 1,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_max_label_seq():
    ''' Tests difference between different numbers of past periods'''

    output_periods_list = [6,12,24]

    for output_periods in output_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MAX_LABEL_SEQ', 
            NAME ='_test_max_label_seq_{}'.format(output_periods),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = output_periods,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_mean_label_seq():
    ''' Tests difference between different numbers of past periods'''

    output_periods_list = [6,12,24]

    for output_periods in output_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'MEAN_LABEL_SEQ', 
            NAME ='_test_mean_label_seq_{}'.format(output_periods),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = output_periods,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

def test_seq():
    ''' Tests difference between different numbers of past periods'''

    output_periods_list = [6,12,24]

    for output_periods in output_periods_list:
        
        lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)
        normal_predictions = wahrsager(lstm_dataset, power_dem_df,
            TYPE = 'SEQ', 
            NAME ='_test_seq_{}'.format(output_periods),
            #PLOTTING = True,
            # Model-Parameter:
            num_outputs       = output_periods,
            dropout           = 0.2,
            recurrent_dropout = 0.2,
            num_hidden        = 3,
            lstm_size         = 128,
            first_hidden_size = 128,
            neuron_num_change = 0.5,
            activation_hidden = 'relu',
            activation_end    = 'relu',
            # Trainings-Parameter
            val_data_size     = 2000,
            num_epochs        = 15,
            ).train()

test_learning_rate()()
#test_overfitting()
#standart_settings()
#test_sigmoid()
#test_lstm_layers()
#test_hidden_layers()
#test_dropout()
#test_past_periods()
#test_mean()
#test_max()
#test_max_label_seq()
#test_mean_label_seq()
#test_seq()

