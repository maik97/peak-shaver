'''Common settings'''
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.logger import Logger

# Directory of the dataset:
D_PATH = '_small_d/'

# Parameter mainDataset:
period_string_min = '15min'
full_dataset      = True

# Parameter mainDataset.make_input_df:
drop_main_terminal = False
use_time_diff      = True
day_diff           = 'holiday-weekend'

# Other parameter:
num_past_periods = 12

def basic_dataset():

	# Load the dataset:
	main_dataset = mainDataset(
	    D_PATH=D_PATH,
	    period_string_min=period_string_min,
	    full_dataset=full_dataset)

	# Normalized dataframe:
	df = main_dataset.make_input_df(
	    drop_main_terminal=drop_main_terminal,
	    use_time_diff=use_time_diff,
	    day_diff=day_diff)

	# Sum of the power demand dataframe (not normalized):
	power_dem_df = main_dataset.load_total_power()

	return df, power_dem_df, main_dataset

def dataset_and_logger(NAME):

	df, power_dem_df, main_dataset = basic_dataset()

	# Load the LSTM input dataset:
	lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=num_past_periods)

	# Making predictions:
	normal_predictions = wahrsager(lstm_dataset, power_dem_df, TYPE='NORMAL').pred()[:-num_past_periods]
	seq_predictions    = wahrsager(lstm_dataset, power_dem_df, TYPE='SEQ', num_outputs=num_past_periods).pred()

	# Adding the predictions to the dataset:
	df            = df[num_past_periods*2:-num_past_periods]
	df['normal']  = normal_predictions
	df['seq_max'] = max_seq(seq_predictions)

	logger = Logger(NAME,D_PATH)

	return df, power_dem_df, logger