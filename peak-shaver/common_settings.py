'''Common settings'''
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.logger import Logger
from main.common_func import max_seq, mean_seq

# Directory of the dataset:
D_PATH = '_BIG_D/'

# Parameter mainDataset:
period_min        = 5
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
	    period_min=period_min,
	    full_dataset=full_dataset)

	# Normalized dataframe:
	df = main_dataset.make_input_df(
	    drop_main_terminal=drop_main_terminal,
	    use_time_diff=use_time_diff,
	    day_diff=day_diff)

	# Sum of the power demand dataframe (not normalized):
	power_dem_df = main_dataset.load_total_power()

	return df, power_dem_df, main_dataset


def load_specific_sets(num_past_periods=24,num_outputs=24,TYPE_LIST=['NORMAL','SEQ'],seq_transform=['MAX']):

	df, power_dem_df, main_dataset = basic_dataset()

	# Load the LSTM input dataset:
	lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=num_past_periods)

	df           = df[num_past_periods*2:-num_outputs]
	power_dem_df = power_dem_df[num_past_periods*2:-num_outputs]
	# Slice-Explanation:
	# first num_past_periods: possible rolling operation generates first values as nan
	# second num_past_periods: first input sequence can only be used for the last timestep of the sequence
	# num_outputs: the last label-steps cant be used as training data since there are no labels to train left

	input_list = ['norm_total_power']

	for TYPE in TYPE_LIST:

		if TYPE == 'SEQ':
			print(len(df))
			seq_predictions = wahrsager(lstm_dataset, power_dem_df, TYPE=TYPE, num_outputs=num_outputs).pred()

			for transform in seq_transform:
				if transform == 'MAX':
					df['SEQ_MAX'] = max_seq(seq_predictions)
					input_list.append('SEQ_MAX')

				elif transform == 'MEAN':
					df['SEQ_MEAN'] = mean_seq(seq_predictions)
					input_list.append('SEQ_MEAN')

				else:
					raise Exception("Unsupported sequence transformation: {}, use 'MAX' or 'MEAN'".format(transform))

		elif TYPE == 'MAX_LABEL_SEQ' or TYPE == 'MEAN_LABEL_SEQ':
			predictions = wahrsager(lstm_dataset, power_dem_df, TYPE=TYPE).pred()[:-num_outputs+1]
			print(len(df))
			print(len(predictions))
			df[TYPE]    = predictions
			input_list.append(TYPE)
		else:
			predictions = wahrsager(lstm_dataset, power_dem_df, TYPE=TYPE).pred()[:-num_outputs]
			print(len(df))
			print(len(predictions))
			df[TYPE]    = predictions
			input_list.append(TYPE)


		return df, power_dem_df, input_list


def load_logger(NAME='test',only_per_episode=True):
	# Initilize logging:
	logger = Logger(NAME,D_PATH,only_per_episode=only_per_episode)
	return logger, period_min


def dataset_and_logger(NAME='test'):

	df, power_dem_df, input_list = load_specific_sets()

	# Initilize logging:
	logger = Logger(NAME,D_PATH,only_per_episode=True)

	return df, power_dem_df, logger, period_min


def dataset_and_logger_old(NAME='test', preprocess_all=False):

	df, power_dem_df, main_dataset = basic_dataset()

	# Load the LSTM input dataset:
	lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=num_past_periods)

	# Making predictions:
	if  preprocess_all == False:
		normal_predictions = wahrsager(lstm_dataset, power_dem_df, TYPE='NORMAL').pred()[:-num_past_periods]
		seq_predictions    = wahrsager(lstm_dataset, power_dem_df, TYPE='SEQ', num_outputs=num_past_periods).pred()
	# Only preprocess:
	else:
		lstm_dataset_creator.rolling_mean_training_data()
		lstm_dataset_creator.rolling_max_training_data()
		lstm_dataset_creator.normal_training_data()
		lstm_dataset_creator.sequence_training_data(num_seq_periods=num_past_periods)

	if  preprocess_all == False:
		# Adding the predictions to the dataset:
		df            = df[num_past_periods*2:-num_past_periods]
		df['normal']  = normal_predictions
		df['seq_max'] = max_seq(seq_predictions)
		power_dem_df  = power_dem_df[num_past_periods*2:-num_past_periods]

		# Initilize logging:
		logger = Logger(NAME,D_PATH,only_per_episode=True)

		return df, power_dem_df, logger, period_min


def main():
	dataset_and_logger_old(preprocess_all=True)


if __name__ == '__main__':
	main()