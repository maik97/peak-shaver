''' Example of preprocessing the HIPE-Dataset '''
from main.schaffer import mainDataset, lstmInputDataset, DatasetStatistics
from main.common_func import wait_to_continue


''' Main Dataset '''

# Setup main dataset creater/loader:
main_dataset = mainDataset(
    D_PATH='_BIG_D/',
    period_string_min='5min',
    full_dataset=True)

# Run this first, since this can take up a lot of time:
main_dataset_creator.smoothed_df()
# wait_to_continue() # Pauses the execution until you press enter

# These don't take up a lot of time to run, 
# but you can run those beforhand to check if everything is setup properly:
main_dataset_creator.load_total_power()
main_dataset_creator.normalized_df()
main_dataset_creator.norm_activation_time_df()
# wait_to_continue()


''' LSTM Dataset: '''

# Import main dataset as dataframe:
df = main_dataset.make_input_df(
    drop_main_terminal=False,
    use_time_diff=True,
    day_diff='holiday-weekend')

# Setup lstm dataset creator/loader:
lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

# If you want to check that everything works fine, run those rather step by step:
lstm_dataset_creator.rolling_mean_training_data()
#wait_to_continue()

lstm_dataset_creator.rolling_max_training_data()
#wait_to_continue()

lstm_dataset_creator.normal_training_data()
#wait_to_continue()

lstm_dataset_creator.sequence_training_data(num_seq_periods=12)
#wait_to_continue()


#plotter = DatasetStatistics(D_PATH='_small_d/', period_string_min='15min', full_dataset=True)

#plotter.plot_dist_power('test1')
#plotter.plot_dist_power_change('test2')
#plotter.plot_compare_peak_lenghts('test3', show_plot=False)