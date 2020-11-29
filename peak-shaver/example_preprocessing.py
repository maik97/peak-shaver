'''
Example of preprocessing the HIPE-Dataset.
'''
from main.schaffer import mainDataset, lstmInputDataset, DatasetStatistics
from main.common_func import wait_to_continue

main_dataset_creator = mainDataset(D_PATH='_small_d/', period_string_min='15min', full_dataset=True)

# If you want to check that everything works fine, run those rather step by step:
'''
main_dataset_creator.smoothed_df()
wait_to_continue()

main_dataset_creator.load_total_power()
wait_to_continue()

main_dataset_creator.normalized_df()
wait_to_continue()

main_dataset_creator.norm_activation_time_df()
wait_to_continue()


lstm_dataset_creator = lstmInputDataset(D_PATH='_small_d/', period_string_min='15min',full_dataset=False,
                                        num_past_periods=12, drop_main_terminal=False, use_time_diff=True,
                                        day_diff='holiday-weekend')

# If you want to check that everything works fine, run those rather step by step:
lstm_dataset_creator.rolling_mean_training_data()
wait_to_continue()

lstm_dataset_creator.rolling_max_training_data()
wait_to_continue()

lstm_dataset_creator.normal_training_data()
wait_to_continue()

lstm_dataset_creator.sequence_training_data(num_seq_periods=12)
wait_to_continue()
'''

plotter = DatasetStatistics(D_PATH='_small_d/', period_string_min='15min', full_dataset=True)

#plotter.plot_dist_power('test1')
#plotter.plot_dist_power_change('test2')
plotter.plot_compare_peak_lenghts('test3', show_plot=False)