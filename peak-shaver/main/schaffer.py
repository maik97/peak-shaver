import os
import pandas as pd
import numpy as np
import datetime
import csv
import matplotlib as mpl
import matplotlib.pyplot  as plt
import seaborn as sns
import h5py

import time

try:
    from main.common_func import print_progress, make_dir, Timer
except:
    from common_func import print_progress, make_dir, Timer


# TO-DO:
# statt global var lieber class

class mainDataset:
    '''This class is used to create the main dataset from wich the inputs for ``wahrsager`` and any of the agents can be chosen from.

    Args:
        D_PATH (string): Path that indicates which dataset is used. Use `'_BIG_D/'` for the full dataset and `'_small_d' for the small dataset, if you followed the propesed folder structure.
        period_string_min (string): Sets the time-period for the dataset. The string should look like this: ``xmin`` where x are the minutes of one period.
        full_dataset (bool): Set this to ``True`` if you are using the full dataset and ``false`` otherwis
    '''
    def __init__(self, D_PATH='_BIG_D/', period_string_min='5min', full_dataset=False):

        self.D_PATH            = D_PATH
        self.period_string_min = period_string_min
        self.full_dataset      = full_dataset
        self.timer             = Timer()

        make_dir(self.D_PATH+'lstm-models/')
        make_dir(self.D_PATH+'lstm-logs/')
        make_dir(self.D_PATH+'lstm-outputs/')
        make_dir(self.D_PATH+'agent-models/')
        make_dir(self.D_PATH+'agent-logs/')
        make_dir(self.D_PATH+'agent-outputs/')

    def return_parameter(self):
        return self.D_PATH

    def coulmn_to_smoothed_period_df(self, dataset_name, coulmn_name, c_num=None, c_total=None): # returns Datafram: Strombedarf (nur noch eine Spalte, Index = SensorDateTime, neuerstellt)
        ''' Trys to open the dataset for a specific machine that is already smoothed to the time-period. Creates a new dataset if a dataset for the given time-period can not be opened.
        Used by :meth:`schaffer.mainDataset.smoothed_df`.

        Args:
            dataset_name (string): The name of the downloaded HIPE-dataset for a specific machine.
            coulmn_name (string): The name of the new dataset for a specif machine, that will be later used as a column name when all the machine-datasets are merged.
            c_num (int): Can be used to show the progress and has to be the count of current the machine-dataset 
            c_total (int): Can be used to show the progress and has to be the sum of all machine-dataset

        Returns:
            dataframe: The smoothed dataset for a given machine
        '''
        # Falls möglich Daten öffnen
        try: 
            dataframe = pd.read_csv(self.D_PATH+'tables/'+self.period_string_min+'/single-column-tables/'+coulmn_name+'.csv', index_col='SensorDateTime', parse_dates=True)
        
        # Sonst erstellen
        except:
            self.timer.start()
            print('Could not open:', self.D_PATH+'tables/'+self.period_string_min+'/single-column-tables/'+coulmn_name+'.csv')
            print('Creating new /single-column-tables/'+coulmn_name+'.csv...')

            make_dir(self.D_PATH+'tables/'+self.period_string_min+'/single-column-tables/')

            dataframe = pd.read_csv('dataset/'+dataset_name+'.csv', index_col='SensorDateTime', parse_dates=True)
            dataframe = dataframe["P_kW"].to_frame()

            # Converting the index as date
            dataframe.index = pd.to_datetime(dataframe.index, utc=True)

            dataframe = dataframe.resample(self.period_string_min).mean().rename(columns={"P_kW": coulmn_name})

            dataframe.to_csv(self.D_PATH+'tables/'+self.period_string_min+'/single-column-tables/'+coulmn_name+'.csv')
            self.timer.stop()

            if c_num == None and c_total == None:
                print('Created dataset for',coulmn_name,'from',dataset_name)
                print('Elapsed time:',self.timer.elapsed_time_string())
            else:
                print('Created dataset for',coulmn_name,'from',dataset_name, '({}/{})'.format(c_num,c_total))
                print('Elapsed time:',self.timer.elapsed_time_string())
                print('Estimated time to create the rest of /single-column-tables/:',self.timer.rest_time_string(c_num,c_total))
        
        print('Table '+self.D_PATH+'tables/'+self.period_string_min+'/single-column-tables/'+coulmn_name+'.csv loaded successfully')
        return dataframe


    def merge_columns_to_df(self, df, column_df, m_num=None, m_total=None):
        ''' Used by :meth:`schaffer.mainDataset.smoothed_df` to merge two dataframes into one

        Args:
            df (dataframe): Smoothed dataset of (multiple) machines
            column_df (dataframe): Smoothed dataset o a given machine that will be merged to ``df``
            m_num (int): Can be used to show the progress and has to be the count of the current the machine-dataset 
            m_total (int): Can be used to show the progress and has to be the sum of all amothed machine-dataset
        
        Returns:
            dataframe: The merged dataframe of ``df`` and ``column_df``
        '''
        df = pd.merge(df,column_df,left_index=True, right_index=True, how='outer')
        if m_num != None and m_total != None:
            print_progress('Merging columns to one dataframe',m_num,m_total)
        return df


    def smoothed_df(self): # returns geglätten Dataframe für alle Maschinen (komplette Tabelle, Index = SensorDateTime, neuerstellt)
        ''' Trys to open the merged and smoothed dataset that includes all machines. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.coulmn_to_smoothed_period_df` to smooth and :meth:`schaffer.mainDataset.merge_columns_to_df` to merge, when creating a new dataset.

        Returns:
            dataframe: The merged and smothed dataframe that includes all machines.
        '''
        try:
            df = pd.read_csv(self.D_PATH+'tables/'+self.period_string_min+'/smoothed_table.csv', index_col='SensorDateTime', parse_dates=True)

        except:
            timer_smooth = Timer()
            timer_smooth.start()
            print('Could not open:',self.D_PATH+'tables/'+self.period_string_min+'/smoothed_table.csv')
            print('Creating new smoothed_table.csv...')

            # Der Zeitraum im Namen der beiden Datensätzen ist unterschiedlich:
            if self.full_dataset  == True:
                zeitraum = '2017-10-01_lt_2018-01-01' # BIG_D
            else:
                zeitraum = '2017-10-23_lt_2017-10-30' # small_d

            original_path = 'hipe_cleaned_v1.0.1_geq_'+zeitraum+'/'

            # Lade, bzw erstelle über den vorg. Zeitraum geglättete CSVs:
            main_terminal         = self.coulmn_to_smoothed_period_df(original_path+'MainTerminal_PhaseCount_3_geq_'+zeitraum,'main_terminal',1,11)
            chip_press            = self.coulmn_to_smoothed_period_df(original_path+'ChipPress_PhaseCount_3_geq_'+zeitraum,'chip_press',2,11)
            chip_saw              = self.coulmn_to_smoothed_period_df(original_path+'ChipSaw_PhaseCount_3_geq_'+zeitraum,'chip_saw',3,11)
            high_temperature_oven = self.coulmn_to_smoothed_period_df(original_path+'HighTemperatureOven_PhaseCount_3_geq_'+zeitraum,'high_temperature_oven',4,11)
            pick_and_place_unit   = self.coulmn_to_smoothed_period_df(original_path+'PickAndPlaceUnit_PhaseCount_2_geq_'+zeitraum,'pick_and_place_unit',5,11)
            screen_printer        = self.coulmn_to_smoothed_period_df(original_path+'ScreenPrinter_PhaseCount_2_geq_'+zeitraum,'screen_printer',6,11)
            soldering_oven        = self.coulmn_to_smoothed_period_df(original_path+'SolderingOven_PhaseCount_3_geq_'+zeitraum,'soldering_oven',7,11)
            vacuum_oven           = self.coulmn_to_smoothed_period_df(original_path+'VacuumOven_PhaseCount_3_geq_'+zeitraum,'vacuum_oven',8,11)
            vacuum_pump_1         = self.coulmn_to_smoothed_period_df(original_path+'VacuumPump1_PhaseCount_3_geq_'+zeitraum,'vacuum_pump_1',9,11)
            vacuum_pump_2         = self.coulmn_to_smoothed_period_df(original_path+'VacuumPump2_PhaseCount_2_geq_'+zeitraum,'vacuum_pump_2',10,11)
            washing_machine       = self.coulmn_to_smoothed_period_df(original_path+'WashingMachine_PhaseCount_3_geq_'+zeitraum,'washing_machine',11,11)

            # Erstelle eine zusammengefügte Dataframe mit allen Maschinen:
            df = self.merge_columns_to_df(main_terminal,chip_press,1,10)
            df = self.merge_columns_to_df(df,chip_saw,2,10)
            df = self.merge_columns_to_df(df,high_temperature_oven,3,10)
            df = self.merge_columns_to_df(df,pick_and_place_unit,4,10)
            df = self.merge_columns_to_df(df,screen_printer,5,10)
            df = self.merge_columns_to_df(df,soldering_oven,6,10)
            df = self.merge_columns_to_df(df,vacuum_oven,7,10)
            df = self.merge_columns_to_df(df,vacuum_pump_1,8,10)
            df = self.merge_columns_to_df(df,vacuum_pump_2,9,10)
            df = self.merge_columns_to_df(df,washing_machine,10,10)
            
            # Rauschen um Null entfernen:
            df[df<0.01] = 0

            # Spechere zusammengefügtes Dataframe als csv:
            df.to_csv(self.D_PATH+'tables/'+self.period_string_min+'/smoothed_table.csv')
            timer_smooth.stop()
            print('Created smoothed_table.csv, elapsed time:', timer_smooth.elapsed_time_string())


        print('Table '+self.D_PATH+'tables/'+self.period_string_min+'/smoothed_table.csv loaded successfully')
        return df


    def load_total_power(self):
        ''' Trys to open the dataset for the sum of all power requirements which are not normalized. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.smoothed_df` when creating a new dataset.

        Returns:
            dataframe: The sum of all power reguieremts per period.
        '''
        # TO-DO: include option to drop main_terminal !!!!!!!
        try:
            total_power = pd.read_csv(self.D_PATH+'tables/'+self.period_string_min+'/total_power.csv', index_col='SensorDateTime', parse_dates=True)
        except:
            print('Could not open:', self.D_PATH+'tables/'+self.period_string_min+'/total_power.csv')
            print('Creating new total_power.csv...')

            df = self.smoothed_df()
            # Berechne Summe des insgesamt benötigten Stroms:
            total_power = pd.DataFrame({
                'total_power' : df.sum(axis = 1)
                }, index = df.index)
            # Speichere Summe des insgesamt benötigten Stroms als CSV:
            total_power.to_csv(self.D_PATH+'tables/'+self.period_string_min+'/total_power.csv')

        print('Table '+self.D_PATH+'tables/'+self.period_string_min+'/total_power.csv loaded successfully')
        return total_power

    def normalize(self, column): # returns array: Normalisierter Strombededarf einer Maschine (neuerstellt)
        ''' Is used by some class functions to normalize a dataframe-column.

        Args:
            column (series, array): The given dataframe-column

        Returns:
            array: A normalized array of the column that can be interpreted as a new (updated) column
        '''
        # Maximaler Wert im ganzen Array:
        max_wert = np.max(column)
        # Falls Werte exisitieren, d.h. kompletter Array ist nicht gleich Null:
        if max_wert != 0:
            normalisiert_array = column / max_wert 
        # Sonst ist Normalisierung auch gleich Null:
        else:
            normalisiert_array = column
        return normalisiert_array


    def normalized_df(self, drop_main_terminal=False):
        ''' Trys to open the normalized dataset that includes all machines. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.normalize` to normalize when creating a new dataset.

        Returns:
            dataframe: The normalized dataset that includes all machines.
        '''
        # main terminal könnte statt gedropped zu weren auch "berichtigt" werden indam man minus des rest rechnen
        try:
            if drop_main_terminal == False:
                norm_df = pd.read_csv(self.D_PATH+'tables/'+self.period_string_min+'/normalized_table.csv', index_col='SensorDateTime', parse_dates=True)
            else:
                norm_df = pd.read_csv(self.D_PATH+'tables/'+self.period_string_min+'/normalized_table_without_main_terminal.csv', index_col='SensorDateTime', parse_dates=True)

        except:
            if drop_main_terminal == False:
                print('Could not open:', self.D_PATH+'tables/'+self.period_string_min+'/normalized_table.csv')
                print('Creating new norm_activ_df.csv...')
            else:
                print('Could not open:', self.D_PATH+'tables/'+self.period_string_min+'/normalized_table_without_main_terminal.csv')
                print('Creating new normalized_df_without_main_terminal.csv...')

            df          = self.smoothed_df()
            total_power = self.load_total_power()
            # Normalisiere Maschinen-Dataframe:
            norm_df = pd.DataFrame({
                'norm_total_power'           : self.normalize(total_power['total_power']),
                'norm_main_terminal'         : self.normalize(df['main_terminal']), 
                'norm_chip_press'            : self.normalize(df['chip_press']),
                'norm_chip_saw'              : self.normalize(df['chip_saw']),
                'norm_high_temperature_oven' : self.normalize(df['high_temperature_oven']),
                'norm_pick_and_place_unit'   : self.normalize(df['pick_and_place_unit']),
                'norm_screen_printer'        : self.normalize(df['screen_printer']),
                'norm_soldering_oven'        : self.normalize(df['soldering_oven']),
                'norm_vacuum_oven'           : self.normalize(df['vacuum_oven']),
                'norm_vacuum_pump_1'         : self.normalize(df['vacuum_pump_1']),
                'norm_vacuum_pump_2'         : self.normalize(df['vacuum_pump_2']),
                'norm_washing_machine'       : self.normalize(df['washing_machine'])
                })

            if drop_main_terminal == False: 
                norm_df.to_csv(self.D_PATH+'tables/'+self.period_string_min+'/normalized_table.csv')
            else:
                 # main terminal könnte statt gedropped zu weren auch "berichtigt" werden indam man minus des rest rechnen
                norm_df = norm_df.drop(['norm_main_terminal'])
                norm_df.to_csv(self.D_PATH+'tables/'+self.period_string_min+'/normalized_table_without_main_terminal.csv')
        
        if drop_main_terminal == False: 
            print('Table '+self.D_PATH+'tables/'+self.period_string_min+'/normalized_table.csv loaded successfully')
        else:
            print('Table '+self.D_PATH+'tables/'+self.period_string_min+'/normalized_table_without_main_terminal.csv loaded successfully')

        return norm_df


    def aktiverungszeit_berechnen(self, column):
        ''' Is used by :meth:`schaffer.mainDataset.norm_activation_time_df` to calculate the activation time a of a given machine.

        Args:
            column (series, array): The given dataframe-column

        Returns:
            list: List that counts the periods since when the machine is active,can be interpreted as a new (updated) column
        '''
        # Initialisiere:
        aktivierungszeit = []
        aktivierung = 0
        # Iteration über jede Zeile der bestimmten Spalte:
        for zeile in column:
            # Wenn Zeile nicht Null, dann ist die Maschine aktiv:
            if zeile != 0:
                aktivierung += 1 # Desto länger die Maschine aktiv ist, desto höher die Summe
            # Wenn Zeile Null, dann ist Maschine inaktiv:
            else:
                aktivierung = 0 # Setze 'Summe' wieder auf Null
            # Füge den Summen-Wert der Aktivierung dem Array hinzu:
            aktivierungszeit.append(aktivierung)
        return aktivierungszeit


    def norm_activation_time_df(self):
        ''' Trys to open the normalized dataset that includes all activation times for the machines. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.aktiverungszeit_berechnen` to calculate the activation time and :meth:`schaffer.mainDataset.normalize` to normalize, when creating a new dataset.

        Returns:
            dataframe: The normalized dataset that includes the activation times for all machines, except `main_terminal` since this is always active.
        '''
        try:
            norm_aktiv_df = pd.read_csv(self.D_PATH+'tables/'+self.period_string_min+'/norm_activ_table.csv', index_col='SensorDateTime', parse_dates=True)
        except:
            print('Could not open:', self.D_PATH+'tables/'+self.period_string_min+'/norm_activ_table.csv')
            print('Creating new norm_activ_df.csv...')

            df = self.smoothed_df()
            # Normalisiere Aktivitätszeiten:
            norm_aktiv_df = pd.DataFrame({
                #'norm_aktiv_t_main_terminal'        : 1, # weil main terminal eh immer an ist
                'norm_aktiv_t_chip_press'            : self.normalize(self.aktiverungszeit_berechnen(df['chip_press'])),
                'norm_aktiv_t_chip_saw'              : self.normalize(self.aktiverungszeit_berechnen(df['chip_saw'])),
                'norm_aktiv_t_high_temperature_oven' : self.normalize(self.aktiverungszeit_berechnen(df['high_temperature_oven'])),
                'norm_aktiv_t_pick_and_place_unit'   : self.normalize(self.aktiverungszeit_berechnen(df['pick_and_place_unit'])),
                'norm_aktiv_t_screen_printer'        : self.normalize(self.aktiverungszeit_berechnen(df['screen_printer'])),
                'norm_aktiv_t_soldering_oven'        : self.normalize(self.aktiverungszeit_berechnen(df['soldering_oven'])),
                'norm_aktiv_t_vacuum_oven'           : self.normalize(self.aktiverungszeit_berechnen(df['vacuum_oven'])),
                'norm_aktiv_t_vacuum_pump_1'         : self.normalize(self.aktiverungszeit_berechnen(df['vacuum_pump_1'])),
                'norm_aktiv_t_vacuum_pump_2'         : self.normalize(self.aktiverungszeit_berechnen(df['vacuum_pump_2'])),
                'norm_aktiv_t_washing_machine'       : self.normalize(self.aktiverungszeit_berechnen(df['washing_machine']))
                }, index=df.index)
            norm_aktiv_df.to_csv(self.D_PATH+'tables/'+self.period_string_min+'/norm_activ_table.csv')
        print('Table '+self.D_PATH+'tables/'+self.period_string_min+'/norm_activ_table.csv loaded successfully')
        return norm_aktiv_df


    def add_day_time_difference(self, df):    
        ''' Can be used by :meth:`schaffer.mainDataset.make_input_df` to create a new (normalized) column that represents the time of the day.
        
        Args:
            df (datafrme): The dataframe to wich the new coulmn will be added.

        Returns:
            dataframe: The new dataframe with the added column
        '''
        # Tageszeit-Format in Zahl:
        df['time'] = df.index.time
        df['time'] = self.normalize(df['time'].index.hour * 60 + df['time'].index.minute + df['time'].index.second/60)
        return df

    def add_day_difference(self, df, day_diff='holiday-weekend'):
        ''' Can be used by :meth:`schaffer.mainDataset.make_input_df` to create a new column that represents the day-type.
        
        Args:
            df (datafrme): The dataframe to wich the new coulmn will be added.
            day_diff (string): The mode by which the day-type will be represented

        Returns:
            dataframe: The new dataframe with the added column(s)

        - ``day_diff='weekend-normal'`` will set each day of the week a value from 0.1 to 0.7
        - ``day_diff='weekend-binary'`` will set 1 for normal work-days and 0 for weekend-days
        - ``day_diff='holiday-weekend'`` will set 1 for normal work-days, 0.5 for holidays and 0 for weekend-days. This will also create a second column with those values for the next day.
        '''
        if day_diff == 'weekday-normal':
            df['day'] = df.index.weekday * 0.1
        elif day_diff == 'weekend-binary':
            # Arbeitstage = 1:
            df['day'][df['day'] < 5] = 1 
            # Wochenende = 0:
            df['day'][df['day'] >= 5] = 0
        elif day_diff == 'holiday-weekend':
            feiertag_liste = ['2017-10-02','2017-10-03','2017-10-30','2017-10-31',
                          '2017-11-01', '2017-12-25','2017-12-26','2017-12-27',
                          '2017-12-28','2017-12-29','2017-12-30']
            df['day-type'] = 1 # Arbeitstag
            df['day-type'][df.index.normalize().isin(feiertag_liste)] = 0.5 # Feiertag
            df['day-type'][df.index.weekday > 4] = 0 # Wochenende
            df['day-type-tomorrow'] = df['day-type'].shift(periods=-len(df['day-type'][df.index.normalize().isin(['2017-10-02'])]), fill_value=df['day-type'][-1])
        else:
            raise Exception("day_diff was set to {}, use 'weekday-normal',  'weekend-binary', 'holiday-weekend' or None!" )
        return df

    def make_input_df(self, drop_main_terminal=False, use_time_diff=True, day_diff='holiday-weekend'):
        ''' Returns an input-dataset using :meth:`schaffer.mainDataset.normalized_df` and :meth:`schaffer.mainDataset.norm_activation_time_df`, by merging those datasets.
        Optionally uses :meth:`schaffer.mainDataset.add_day_time_difference` when ``use_time_diff=True`` and :meth:`schaffer.mainDataset.add_day_difference` when ``day_diff`` is NOT `None`.
        
        Args:
            drop_main_terminal (bool): The column main_terminal will be removed from the dataset if set to `True`
            use_time_diff (bool): Uses :meth:`schaffer.mainDataset.add_day_time_difference` when set to `True`
            day_diff (string, null): Uses :meth:`schaffer.mainDataset.add_day_difference` when set to `'weekday-normal'`. `'weekend-binary'` or `'weekend-binary`. Does not use this function if set to `None`.

        Returns:
            dataframe: A dataframe that can be used as inputs for ``wahrsager`` or any of the agents.
        '''
        norm_df       = self.normalized_df(drop_main_terminal)
        norm_aktiv_df = self.norm_activation_time_df()
        input_df = pd.merge(norm_df, norm_aktiv_df, left_index=True, right_index=True, how='outer')

        if use_time_diff == True:
            input_df = self.add_day_time_difference(input_df)

        if day_diff != None:
            input_df = self.add_day_difference(input_df, day_diff)

        self.lstm_name = ''
        if use_time_diff == True:
            self.lstm_name += '_time-diff'
        if day_diff != None:
            self.lstm_name += '_'+day_diff
            # day_diff = 'holiday-weekend', 'weekend-normal', 'weekend-binary', None
        if drop_main_terminal == True:
            self.lstm_name += '_no-main-t'

        return input_df



class lstmInputDataset:
    '''This class is used to create the LSTM-dataset as the inputs for ``wahrsager``.

    Args:
        D_PATH (string): Path that indicates which dataset is used. Use `'_BIG_D/'` for the full dataset and `'_small_d' for the small dataset, if you followed the propesed folder structure.
        period_string_min (string): Sets the time-period for :class:`schaffer.mainDataset`. The string should look like this: ``xmin`` where x are the minutes of one period.
        full_dataset (bool): Set this to ``True`` if you are using the full dataset and ``false`` otherwise. Parameter that will be used by :class:`schaffer.mainDataset`.
        num_past_periods (int): The size of the input-sequence for the LSTM.
        drop_main_terminal (bool): Parameter will be used for :meth:`schaffer.mainDataset.make_input_df`: The column main_terminal will be removed from the dataset if set to `True`.
        use_time_diff (bool): Parameter will be used for :meth:`schaffer.mainDataset.make_input_df`:  Uses :meth:`schaffer.mainDataset.add_day_time_difference` when set to `True`.
        day_diff (string): Parameter will be used for :meth:`schaffer.mainDataset.make_input_df`: Uses :meth:`schaffer.mainDataset.add_day_difference` when set to `'weekday-normal'`. `'weekend-binary'` or `'weekend-binary`. Does not use this function if set to `None`.
    '''

    def __init__(self, main_dataset, df, num_past_periods=12):


        self.D_PATH            = main_dataset.__dict__['D_PATH']
        self.period_string_min = main_dataset.__dict__['period_string_min']
        self.name              = main_dataset.__dict__['lstm_name']
        self.alle_inputs_df    = df
        self.num_past_periods  = num_past_periods
        self.timer             = Timer()

        make_dir(self.D_PATH+'tables/'+self.period_string_min+'/training-data/')


    def rolling_mean_training_data(self):
        ''' Trys to open an LSTM-input-dataset that was transformed with a `rolling mean` operation with the time-frame ``num_past_periods``. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.make_input_df` to setup a dataset for the given paramerters.

        Returns:
            array, array: training-data, label-data
        '''
        try:
            
            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_rolling-mean_{}.h5'.format(self.num_past_periods), 'r') as hf:
                training_data = hf['training_data'][:]
                label_data = hf['label_data'][:]

        except:            
            rolling_mean_inputs_df = self.alle_inputs_df.rolling(self.num_past_periods).mean()
            rolling_mean_inputs = rolling_mean_inputs_df.to_numpy()

            self.timer.start()
            training_data = []
            for i in range(len(rolling_mean_inputs[:-self.num_past_periods])):
                training_data = np.append(training_data, rolling_mean_inputs[i:i+self.num_past_periods])
                self.timer.print_time_progress('Creating training data',i , len(rolling_mean_inputs[:-self.num_past_periods]))


            num_inputs_t = len(rolling_mean_inputs[0]) 
            num_t = len(rolling_mean_inputs[self.num_past_periods:])


            training_data = np.reshape(training_data, (num_t, self.num_past_periods, num_inputs_t))[self.num_past_periods:]
            label_data = rolling_mean_inputs_df['norm_total_power'].to_numpy()[self.num_past_periods:][self.num_past_periods:]

            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_rolling-mean_{}.h5'.format(self.num_past_periods), 'w') as hf:
                hf.create_dataset("training_data",  data=training_data)
                hf.create_dataset("label_data",  data=label_data)
        
        print('Dataset '+self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_rolling-mean_{}.h5 loaded successfully')       
        return training_data, label_data


    def rolling_max_training_data(self):
        ''' Trys to open an LSTM-input-dataset that was transformed with a `rolling max` operation with the time-frame ``num_past_periods``. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.make_input_df` to setup a dataset for the given paramerters.

        Returns:
            array, array: training-data, label-data
        '''
        try:
            
            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_rolling-max_{}.h5'.format(self.num_past_periods), 'r') as hf:
                training_data = hf['training_data'][:]
                label_data = hf['label_data'][:]

        except:            
            rolling_max_inputs_df = self.alle_inputs_df.rolling(self.num_past_periods).max()
            rolling_max_inputs = rolling_max_inputs_df.to_numpy()

            self.timer.start()
            training_data = []
            for i in range(len(rolling_max_inputs[:-self.num_past_periods])):
                training_data = np.append(training_data, rolling_max_inputs[i:i+self.num_past_periods])
                self.timer.print_time_progress('Creating training data', i, len(rolling_max_inputs[:-self.num_past_periods]))


            num_inputs_t = len(rolling_max_inputs[0]) 
            num_t = len(rolling_max_inputs[self.num_past_periods:])


            training_data = np.reshape(training_data, (num_t, self.num_past_periods, num_inputs_t))[self.num_past_periods:]
            label_data = rolling_max_inputs_df['norm_total_power'].to_numpy()[self.num_past_periods:][self.num_past_periods:]

            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_rolling-max_{}.h5'.format(self.num_past_periods), 'w') as hf:
                hf.create_dataset("training_data",  data=training_data)
                hf.create_dataset("label_data",  data=label_data)

        print('Dataset '+self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_rolling-max_{}.h5 loaded successfully')       
        return training_data, label_data


    def normal_training_data(self):
        ''' Trys to open an LSTM-input-dataset that was transformed with the time-frame ``num_past_periods``. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.make_input_df` to setup a dataset for the given paramerters.

        Returns:
            array, array: training-data, label-data
        '''
        try:
            
            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_normal_{}.h5'.format(self.num_past_periods), 'r') as hf:
                training_data = hf['training_data'][:]
                label_data = hf['label_data'][:]

        except:
            normal_inputs = self.alle_inputs_df.to_numpy()

            self.timer.start()
            training_data = []
            for i in range(len(normal_inputs[:-self.num_past_periods])):
                training_data = np.append(training_data, normal_inputs[i:i+self.num_past_periods])
                self.timer.print_time_progress('Creating training data',i, len(normal_inputs[:-self.num_past_periods]))

            num_inputs_t = len(normal_inputs[0]) 
            num_t = len(normal_inputs[self.num_past_periods:])


            training_data = np.reshape(training_data, (num_t, self.num_past_periods, num_inputs_t))[self.num_past_periods:]
            label_data = self.alle_inputs_df['norm_total_power'].to_numpy()[self.num_past_periods:][self.num_past_periods:]

            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_normal_{}.h5'.format(self.num_past_periods), 'w') as hf:
                hf.create_dataset("training_data",  data=training_data)
                hf.create_dataset("label_data",  data=label_data)

        print('Dataset '+self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_normal_{}.h5 loaded successfully')       
        return training_data, label_data


    def sequence_training_data(self, num_seq_periods=12):
        ''' Trys to open an LSTM-input-dataset has time-frame ``num_past_periods`` for the sequence-input and ``num_seq_periods`` for the label-sequence. Creates a new dataset if the dataset can not be opened.
        Uses :meth:`schaffer.mainDataset.make_input_df` to setup a dataset for them given paramerters.
        
        Returns:
            array, array: training-data, label-data
        '''
        try:
            
            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_sequence_{}-{}.h5'.format(self.num_past_periods,num_seq_periods), 'r') as hf:
                training_data = hf['training_data'][:]
                label_data = hf['label_data'][:]

        except:            
            sequence_inputs = self.alle_inputs_df.to_numpy()
            sequence_outputs = self.alle_inputs_df['norm_total_power'].to_numpy()

            self.timer.start()
            label_data = []
            for i in range(len(sequence_outputs[:-num_seq_periods])):
                label_data = np.append(label_data, sequence_outputs[i:i+num_seq_periods])
                self.timer.print_time_progress('Creating label data (sequence)', i, len(sequence_outputs[:-self.num_past_periods]))

            #num_inputs_t = len(sequence_outputs[0]) 
            num_t = len(sequence_outputs[num_seq_periods:])
            label_data = np.reshape(label_data, (num_t, num_seq_periods))[self.num_past_periods:][self.num_past_periods:]

            self.timer.start()
            training_data = []
            for i in range(len(sequence_inputs[:-self.num_past_periods])):
                training_data = np.append(training_data, sequence_inputs[i:i+self.num_past_periods])
                self.timer.print_time_progress('Creating training data (sequence)', i, len(sequence_inputs[:-self.num_past_periods]))

            num_inputs_t = len(sequence_inputs[0]) 
            num_t = len(sequence_inputs[self.num_past_periods:])
            training_data = np.reshape(training_data, (num_t, self.num_past_periods, num_inputs_t))[self.num_past_periods:-num_seq_periods]

            with h5py.File(self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_sequence_{}-{}.h5'.format(self.num_past_periods,num_seq_periods), 'w') as hf:
                hf.create_dataset("training_data",  data=training_data)
                hf.create_dataset("label_data",  data=label_data)
        print(np.shape(training_data))
        print(np.shape(label_data))
        print('Dataset '+self.D_PATH+'tables/'+self.period_string_min+'/training-data/'+self.name+'_sequence_{}-{}.h5 loaded successfully')       
        return training_data, label_data


class DatasetStatistics:

    def __init__(self, D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True):

        self.sum_total_power_df    = mainDataset(D_PATH, period_string_min, full_dataset).load_total_power()
        self.sum_total_power_array = self.sum_total_power_df.to_numpy()

        self.D_PATH = D_PATH+'statistics/dataset-statistics/'
        make_dir(self.D_PATH)


    def list_peak_len_above_1(self, peak_value=42):
        
        len_peak_list = []
        value_to_append = 0
        for v in self.sum_total_power_array:

            if v >= peak_value:
                value_to_append +=1
            elif value_to_append != 0:
                len_peak_list.append(value_to_append)
                value_to_append = 0

        return len_peak_list

    def list_peak_len_above(self, peak_value=42):
        
        len_peak_list = []
        value_to_append = 0
        for v in self.sum_total_power_array:

            if v >= peak_value:
                value_to_append +=1
            else:
                if value_to_append != 0:
                    len_peak_list.append(value_to_append)
                    value_to_append = 0

        return len_peak_list


    def plot_dist_power(self, save_plot_as=None, show_plot=True):
        # Distribution KW-Bedarf
        sns.distplot(self.sum_total_power_df, hist=False, kde_kws={"shade": True}, color="m")

        if save_plot_as != None:
            plt.savefig(self.D_PATH+save_plot_as)

        if show_plot == True:
            plt.show()

        
    def plot_dist_power_change(self, save_plot_as=None, show_plot=True):
        # Distribution KW-Bedarf
        sns.distplot(self.sum_total_power_df.diff().dropna(), hist=False, kde_kws={"shade": True}, color="m")

        if show_plot == True:
            plt.show()

        if save_plot_as != None:
            plt.savefig(self.D_PATH+save_plot_as)


    def plot_compare_peak_lenghts_1(self, save_plot_as=None, show_plot=True):

        plt.subplot(4,3,1)
        sns.displot(self.list_peak_len_above(40), kde=False)

        plt.subplot(4,3,2)
        sns.displot(self.list_peak_len_above(42), kde=False)

        plt.subplot(4,3,3)
        sns.displot(self.list_peak_len_above(45), kde=False)

        plt.subplot(4,3,4)
        sns.displot(self.list_peak_len_above(50), kde=False)

        plt.subplot(4,3,5)
        sns.displot(self.list_peak_len_above(55), kde=False)

        plt.subplot(4,3,6)
        sns.displot(self.list_peak_len_above(60), kde=False)

        plt.subplot(4,3,7)
        sns.displot(self.list_peak_len_above(65), kde=False)

        plt.subplot(4,3,8)
        sns.displot(self.list_peak_len_above(70), kde=False)

        plt.subplot(4,3,9)
        sns.displot(self.list_peak_len_above(75), kde=False)

        plt.subplot(4,3,10)
        sns.displot(self.list_peak_len_above(80), kde=False)

        plt.subplot(4,3,11)
        sns.displot(self.list_peak_len_above(85), kde=False)

        plt.subplot(4,3,12)
        sns.displot(self.list_peak_len_above(90), kde=False)

        plt.tight_layout()
        
        if show_plot == True:
            plt.show()

        if save_plot_as != None:
            plt.savefig(self.D_PATH+save_plot_as)


    def plot_compare_peak_lenghts(self, save_plot_as=None, show_plot=True):

        fig, axes = plt.subplots(4, 3)

        sns.histplot(ax=axes[0,0], data=self.list_peak_len_above(40), kde=False)
        sns.histplot(ax=axes[0,1], data=self.list_peak_len_above(42), kde=False)
        sns.histplot(ax=axes[0,2], data=self.list_peak_len_above(45), kde=False)
        sns.histplot(ax=axes[1,0], data=self.list_peak_len_above(50), kde=False)
        sns.histplot(ax=axes[1,1], data=self.list_peak_len_above(55), kde=False)
        sns.histplot(ax=axes[1,2], data=self.list_peak_len_above(60), kde=False)
        sns.histplot(ax=axes[2,0], data=self.list_peak_len_above(65), kde=False)
        sns.histplot(ax=axes[2,1], data=self.list_peak_len_above(70), kde=False)
        sns.histplot(ax=axes[2,2], data=self.list_peak_len_above(75), kde=False)
        sns.histplot(ax=axes[3,0], data=self.list_peak_len_above(80), kde=False)
        sns.histplot(ax=axes[3,1], data=self.list_peak_len_above(85), kde=False)
        sns.histplot(ax=axes[3,2], data=self.list_peak_len_above(90), kde=False)

        if save_plot_as != None:
            plt.savefig(self.D_PATH+save_plot_as)

        if show_plot == True:
            plt.show()

    def plot_compare_peak_lenght_3(self, save_plot_as=None, show_plot=True):

        data = []
        ranking = [40,42,45,50,55,60,65,70,75]
        for list_data in ranking:
            data.append(np.array(self.list_peak_len_above(list_data)))

        print(data)

        sns.boxenplot(palette="light:m_r",data=data)
        
        if save_plot_as != None:
            plt.savefig(self.D_PATH+save_plot_as)

        if show_plot == True:
            plt.show()       










