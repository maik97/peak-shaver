'''Various tests for parameter tuningof the wahrsager LSTM-Network'''
from common_settings import basic_dataset
from main.wahrsager import wahrsager
from main.schaffer import lstmInputDataset

# Load the dataset:
df, power_dem_df, main_dataset = basic_dataset()


def run_wahrsager(NAME,TYPE='NORMAL',num_outputs=1, dropout=0.1, num_hidden=3,lstm_size=256, num_past_periods=24,
                  activation_hidden='relu',activation_end='relu',lr=0.001,num_epochs=1000):

    #try:
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=num_past_periods)
    normal_predictions = wahrsager(lstm_dataset, power_dem_df,
        TYPE = TYPE, 
        NAME = NAME,
        PLOTTING = True,
        # Model-Parameter:
        num_outputs       = num_outputs,
        dropout           = dropout,
        recurrent_dropout = dropout,
        num_hidden        = num_hidden,
        lstm_size         = lstm_size,
        first_hidden_size = lstm_size,
        neuron_num_change = 0.5,
        activation_hidden = activation_hidden,
        activation_end    = activation_end,
        lr                = lr,
        # Trainings-Parameter
        val_data_size     = 2000,
        num_epochs        = num_epochs,
        ).train()
    #except Exception as e:
    #   print(e)

def parameter_tuning(num_runs=3):

    for i in range(num_runs):
        
        run_wahrsager('_test_standart')

        learning_rate_list = [0.0001,0.001,0.01] 
        for learning_rate in learning_rate_list:
            run_wahrsager('_test_learning_rate_{}'.format(learning_rate), lr=learning_rate)

        dropout_list = [0,0.2,0.3]
        for dropout in dropout_list:
            run_wahrsager('_test_dropout_{}'.format(dropout), dropout=dropout)

        activation_list = [['relu','sigmoid'],['sigmoid','sigmoid'],['sigmoid','relu']] 
        for activation in activation_list:
            run_wahrsager('_test_activation_{}-{}'.format(activation[0],activation[1]), activation_hidden=activation[0], activation_end=activation[1])
        
        lstm_layers_list = [64,128,512]
        for lstm_layers in lstm_layers_list:
            run_wahrsager('_test_lstm_layers_{}'.format(lstm_layers), lstm_size=lstm_layers)

        hidden_layers_list = [1,2]
        for hidden_layers in hidden_layers_list:
            run_wahrsager('_test_hidden_layers_{}'.format(hidden_layers), num_hidden=hidden_layers)
        
        past_periods_list = [6,12,30] 
        for past_periods in past_periods_list:
            run_wahrsager('_test_past_periods_{}'.format(past_periods), num_past_periods=past_periods)

        mean_list = [6,12,24] 
        for mean_ in mean_list:
            run_wahrsager('_test_mean_{}'.format(mean_), num_past_periods=mean_)

        max_list = [6,12,24] 
        for max_ in max_list:
            run_wahrsager('_test_max_{}'.format(max_), num_past_periods=max_)
        
        max_label_seq_list = [6,12,24]
        for max_label_seq in max_label_seq_list:
            run_wahrsager('_test_max_label_seq_{}'.format(max_label_seq), TYPE='MAX_LABEL_SEQ', num_outputs=max_label_seq)
        
        mean_label_seq_list = [6,12,24]
        for mean_label_seq in mean_label_seq_list:
            run_wahrsager('_test_mean_label_seq_{}'.format(mean_label_seq), TYPE='MEAN_LABEL_SEQ', num_outputs=mean_label_seq)
        
        seq_list = [6,12,24]
        for seq in seq_list:
            run_wahrsager('_test_seq_{}'.format(seq), TYPE='SEQ', num_outputs=seq)



parameter_tuning()


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

'''
def final_normal():

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
'''