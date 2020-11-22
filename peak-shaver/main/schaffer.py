import pandas as pd
import numpy as np
import datetime
import tqdm
import csv
import matplotlib as mpl
import matplotlib.pyplot  as plt
import seaborn as sns
import h5py

from tqdm import tqdm

# TO-DO:
# statt global var lieber class

def global_var(_NAME='', _VERSION='', _DATENSATZ_PATH ='_BIG_D/', _großer_datensatz = True, _zeitintervall = '5min'):
    global NAME
    global VERSION
    global großer_datensatz
    global DATENSATZ_PATH
    global zeitintervall

    NAME = _NAME
    VERSION = _VERSION
    großer_datensatz = _großer_datensatz
    DATENSATZ_PATH = _DATENSATZ_PATH
    zeitintervall = _zeitintervall



def power_csv(dateiname, name): # returns Datafram: Strombedarf (nur noch eine Spalte, Index = SensorDateTime, neuerstellt)
    
    # Falls möglich Daten öffnen
    try: 
        dataframe = pd.read_csv(DATENSATZ_PATH+'datensatz/5min/'+name+'.csv', index_col='SensorDateTime', parse_dates=True)
    

    # Sonst erstellen
    except:

        dataframe = pd.read_csv(DATENSATZ_PATH+'datensatz/OG/'+dateiname+'.csv', index_col='SensorDateTime', parse_dates=True)
        dataframe = dataframe["P_kW"].to_frame()

        # Converting the index as date
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)

        dataframe = dataframe.resample(zeitintervall).mean().rename(columns={"P_kW": name})

        #dataframe.to_csv(DATENSATZ_PATH+'datensatz/5min/'+name+'.csv')


    return dataframe



def alle_csv(): # returns geglätten Dataframe für alle Maschinen (komplette Tabelle, Index = SensorDateTime, neuerstellt)
    
    # Initialisiere Progress Bar
    prog_bar_glätten = tqdm.tqdm(total=22, desc='Glätten der Daten mit dem Zeitintervall {}'.format(zeitintervall), position=1)
    
    # Der Zeitraum im Namen der beiden Datensätzen ist unterschiedlich:
    if großer_datensatz == True:
        zeitraum = '2017-10-01_lt_2018-01-01' # BIG_D
    else:
        zeitraum = '2017-10-23_lt_2017-10-30' # small_d

    # Lade, bzw erstelle über den vorg. Zeitraum geglättete CSVs:
    main_terminal = power_csv('MainTerminal_PhaseCount_3_geq_'+zeitraum,'main_terminal')
    prog_bar_glätten.update(1)

    chip_press = power_csv('ChipPress_PhaseCount_3_geq_'+zeitraum,'chip_press')
    prog_bar_glätten.update(1)

    chip_saw = power_csv('ChipSaw_PhaseCount_3_geq_'+zeitraum,'chip_saw')
    prog_bar_glätten.update(1)

    high_temperature_oven = power_csv('HighTemperatureOven_PhaseCount_3_geq_'+zeitraum,'high_temperature_oven')
    prog_bar_glätten.update(1)

    pick_and_place_unit = power_csv('PickAndPlaceUnit_PhaseCount_2_geq_'+zeitraum,'pick_and_place_unit')
    prog_bar_glätten.update(1)

    screen_printer = power_csv('ScreenPrinter_PhaseCount_2_geq_'+zeitraum,'screen_printer')
    prog_bar_glätten.update(1)

    soldering_oven = power_csv('SolderingOven_PhaseCount_3_geq_'+zeitraum,'soldering_oven')
    prog_bar_glätten.update(1)

    vacuum_oven = power_csv('VacuumOven_PhaseCount_3_geq_'+zeitraum,'vacuum_oven')
    prog_bar_glätten.update(1)

    vacuum_pump_1 = power_csv('VacuumPump1_PhaseCount_3_geq_'+zeitraum,'vacuum_pump_1')
    prog_bar_glätten.update(1)

    vacuum_pump_2 = power_csv('VacuumPump2_PhaseCount_2_geq_'+zeitraum,'vacuum_pump_2')
    prog_bar_glätten.update(1)

    washing_machine = power_csv('WashingMachine_PhaseCount_3_geq_'+zeitraum,'washing_machine')
    prog_bar_glätten.update(1)


    # Erstelle eine zusammengefügte Dataframe mit allen Maschinen:

    df = pd.merge(main_terminal,chip_press,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,chip_saw,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,high_temperature_oven,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,pick_and_place_unit,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,screen_printer,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,soldering_oven,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,vacuum_oven,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,vacuum_pump_1,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,vacuum_pump_2,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    df = pd.merge(df,washing_machine,left_index=True, right_index=True, how='outer')
    prog_bar_glätten.update(1)

    
    # Spechere zusammengefügtes Dataframe als csv:
    df.to_csv(DATENSATZ_PATH+'datensatz/5min/komplette_tabelle.csv')
    return df



def aktiverungszeit_berechnen(spalte): # returns array: aktivierungszeit einer bestimmten Maschine (neuerstellt)
    
    # Initialisiere:
    aktivierungszeit = []
    aktivierung = 0


    # Iteration über jede Zeile der bestimmten Spalte:
    for zeile in spalte:
        
        # Wenn Zeile nicht Null, dann ist die Maschine aktiv:
        if zeile != 0:
            aktivierung += 1 # Desto länger die Maschine aktiv ist, desto höher die Summe
        
        # Wenn Zeile Null, dann ist Maschine inaktiv:
        else:
            aktivierung = 0 # Setze 'Summe' wieder auf Null

        # Füge den Summen-Wert der Aktivierung dem Array hinzu:
        aktivierungszeit.append(aktivierung)

    return aktivierungszeit



def normalisieren(array): # returns array: Normalisierter Strombededarf einer Maschine (neuerstellt)

    # Maximaler Wert im ganzen Array:
    max_wert = np.max(array)
    
    # Falls Werte exisitieren, d.h. kompletter Array ist nicht gleich Null:
    if max_wert != 0:
        normalisiert_array = array / max_wert 

    # Sonst ist Normalisierung auch gleich Null:
    else:
        normalisiert_array = array

    return normalisiert_array



def load_geglättet_df(): # returns geglätten Dataframe für alle Maschinen (geladen falls möglich, sonst über alle_csv)

    # Falls möglich Daten öffnen:
    try:
        df = pd.read_csv(DATENSATZ_PATH+'datensatz/5min/komplette_tabelle.csv', index_col='SensorDateTime', parse_dates=True)
    
    # Sonst Daten erstellen:
    except:
        print('Geglättete Daten nicht gefunden')
        print('Erstelle zunächst geglättete Daten neu...')

        df = alle_csv()

    return df



def datensatz_bearbeiten(): # returns total_power, norm_aktiv_df, norm_df, norm_time (jeweils neuerstellte dataframes)

    # Geglätteten Dataframe lade:
    df = load_geglättet_df()


    # Berechne Summe des insgesamt benötigten Stroms:
    total_power = pd.DataFrame({
        'total_power' : df.sum(axis = 1)
        }, index = df.index)

    #df.sum(axis=1).rename('total_power')

    # Rauschen um Null entfernen:
    df[df<0.01] = 0
    

    # neue Spalte mit Wochentag:
    df['Weekday'] = df.index.weekday * 0.1
    df['time'] =  df.index.time

    # Ändere Tageszeit-Format in Zahl:
    df['time'] = df['time'].index.hour * 60 + df['time'].index.minute + df['time'].index.second/60
    
    # Arbeitstage = 1:
    #df['Weekday'][df['Weekday'] < 5] = 1 

    # Wochenende = 0:
    #df['Weekday'][df['Weekday'] >= 5] = 0

    # Erstelle neuen Dataframe für die Tageszeit und den Wochentag:
    df_time = pd.DataFrame({
        'time' : df['time'],
        'Weekday' : df['Weekday']
        })

    # Lösche Spalten Tageszeit und Wochentag aus dem Maschinen-Dataframe:
    df.drop(['time', 'Weekday'], axis=1)
    

    # Neues Dataframe, ohne Main_Terminal -> 'Rest':
    #df_rest = df
    #df_rest = df_rest.drop(['main_terminal'], axis=1)

    # Summiere den 'Rest':
    #rest_sum = df_rest.sum(axis=1)

    # Berichtige Main_Terminal, da in Main_terminal zuvor der 'Rest' enthalten war:
    #df['main_terminal'] -= rest_sum


    # Berechen für jede Spalte die Aktivitätszeiten
    aktiv_t_main_terminal = aktiverungszeit_berechnen(df['main_terminal'])
    aktiv_t_chip_press = aktiverungszeit_berechnen(df['chip_press'])
    aktiv_t_chip_saw = aktiverungszeit_berechnen(df['chip_saw'])
    aktiv_t_high_temperature_oven = aktiverungszeit_berechnen(df['high_temperature_oven'])
    aktiv_t_pick_and_place_unit = aktiverungszeit_berechnen(df['pick_and_place_unit'])
    aktiv_t_screen_printer = aktiverungszeit_berechnen(df['screen_printer'])
    aktiv_t_soldering_oven = aktiverungszeit_berechnen(df['soldering_oven'])
    aktiv_t_vacuum_oven = aktiverungszeit_berechnen(df['vacuum_oven'])
    aktiv_t_vacuum_pump_1 = aktiverungszeit_berechnen(df['vacuum_pump_1'])
    aktiv_t_vacuum_pump_2 = aktiverungszeit_berechnen(df['vacuum_pump_2'])
    aktiv_t_washing_machine = aktiverungszeit_berechnen(df['washing_machine'])



    # Normalisiere Aktivitätszeiten:
    norm_aktiv_df = pd.DataFrame({
        #'aktiv_t_main_terminal' : normalisieren(aktiv_t_main_terminal),
        #'norm_aktiv_t_main_terminal' : 1, # weil main terminal eh immer an ist
        'norm_aktiv_t_chip_press' : normalisieren(aktiv_t_chip_press),
        'norm_aktiv_t_chip_saw' : normalisieren(aktiv_t_chip_saw),
        'norm_aktiv_t_high_temperature_oven' : normalisieren(aktiv_t_high_temperature_oven),
        'norm_aktiv_t_pick_and_place_unit' : normalisieren(aktiv_t_pick_and_place_unit),
        'norm_aktiv_t_screen_printer' : normalisieren(aktiv_t_screen_printer),
        'norm_aktiv_t_soldering_oven' : normalisieren(aktiv_t_soldering_oven),
        'norm_aktiv_t_vacuum_oven' : normalisieren(aktiv_t_vacuum_oven),
        'norm_aktiv_t_vacuum_pump_1' : normalisieren(aktiv_t_vacuum_pump_1),
        'norm_aktiv_t_vacuum_pump_2' : normalisieren(aktiv_t_vacuum_pump_2),
        'norm_aktiv_t_washing_machine' : normalisieren(aktiv_t_washing_machine)
        }, index=df.index)

    # Normalisiere Maschinen-Dataframe:
    norm_df = pd.DataFrame({
        'norm_total_power': normalisieren(total_power['total_power']),
        'norm_main_terminal' : normalisieren(df['main_terminal']), 
        'norm_chip_press' : normalisieren(df['chip_press']),
        'norm_chip_saw' : normalisieren(df['chip_saw']),
        'norm_high_temperature_oven' : normalisieren(df['high_temperature_oven']),
        'norm_pick_and_place_unit' : normalisieren(df['pick_and_place_unit']),
        'norm_screen_printer' : normalisieren(df['screen_printer']),
        'norm_soldering_oven' : normalisieren(df['soldering_oven']),
        'norm_vacuum_oven' : normalisieren(df['vacuum_oven']),
        'norm_vacuum_pump_1' : normalisieren(df['vacuum_pump_1']),
        'norm_vacuum_pump_2' : normalisieren(df['vacuum_pump_2']),
        'norm_washing_machine' : normalisieren(df['washing_machine'])
        })

    # Normalisiere Tageszeit-Wochentag-Dataframe:
    norm_time = pd.DataFrame({
        'norm_wochentag' : df_time['Weekday'],
        'norm_tageszeit' : normalisieren(df_time['time'])
        })


    # Speichere Normalisierte Dataframes als CSVs:
    norm_aktiv_df.to_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_aktiv_df.csv')
    norm_df.to_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_df.csv')
    norm_time.to_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_time.csv')
    

    # Speichere Summe des insgesamt benötigten Stroms als CSV:
    total_power.to_csv(DATENSATZ_PATH+'datensatz/verarbeitet/total_power.csv')

    # Ausgabe:
    print('Normalisierte Daten und Total-Power-Daten erfolgreich neu erstellt')

    return total_power, norm_aktiv_df, norm_df, norm_time



def load_norm_data(): # returns total_power, norm_aktiv_df, norm_df, norm_time (jeweils geladen falls möglich, sonst über datensatz_bearbeiten)
    # Falls möglich Daten öffnen:
    try:
        total_power = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/total_power.csv', index_col='SensorDateTime', parse_dates=True)
        norm_aktiv_df = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_aktiv_df.csv', index_col='SensorDateTime', parse_dates=True)
        norm_df = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_df.csv', index_col='SensorDateTime', parse_dates=True)
        norm_time = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_time.csv', index_col='SensorDateTime', parse_dates=True)

    # Sonst erstellen:
    except:
        print('Normalisierte Daten oder Total-Power-Daten nicht gefunden')
        print('Erstelle normalisierte Daten, bzw Total-Power-Daten neu...')
        total_power, norm_aktiv_df, norm_df, norm_time = datensatz_bearbeiten()

    return total_power, norm_aktiv_df, norm_df, norm_time


def load_only_norm_data(): # returns total_power, norm_aktiv_df, norm_df, norm_time (jeweils geladen falls möglich, sonst über datensatz_bearbeiten)
    # Falls möglich Daten öffnen:
    try:
        norm_aktiv_df = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_aktiv_df.csv', index_col='SensorDateTime', parse_dates=True)
        norm_df = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_df.csv', index_col='SensorDateTime', parse_dates=True)
        norm_time = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/norm_time.csv', index_col='SensorDateTime', parse_dates=True)

    # Sonst erstellen:
    except:
        print('Normalisierte Daten nicht gefunden')
        print('Erstelle normalisierte Daten, bzw Total-Power-Daten neu...')
        total_power, norm_aktiv_df, norm_df, norm_time = datensatz_bearbeiten()

    return norm_aktiv_df, norm_df, norm_time


def load_total_power(): # returns total_power, norm_aktiv_df, norm_df, norm_time (jeweils geladen falls möglich, sonst über datensatz_bearbeiten)
    # Falls möglich Daten öffnen:
    try:
        total_power = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/total_power.csv', index_col='SensorDateTime', parse_dates=True)

    # Sonst erstellen:
    except:
        print('Total-Power-Daten nicht gefunden')
        print('Erstelle normalisierte Daten, bzw Total-Power-Daten neu...')
        total_power = datensatz_bearbeiten()

    return total_power


def alle_inputs():

    try:
        alle_inputs_df = pd.read_csv(DATENSATZ_PATH+'datensatz/verarbeitet/alle_inputs_df.csv', index_col='SensorDateTime', parse_dates=True)
    
    except:
        norm_aktiv_df, norm_df, norm_time  = load_only_norm_data()

        # norm_aktiv_df, norm_df und norm_time zusammenfügen:
        alle_inputs_df = pd.merge(norm_aktiv_df, norm_df, left_index=True, right_index=True, how='outer')
        alle_inputs_df = pd.merge(alle_inputs_df, norm_time, left_index=True, right_index=True, how='outer')

        alle_inputs_df.to_csv(DATENSATZ_PATH+'datensatz/verarbeitet/alle_inputs_df.csv')
    
    return alle_inputs_df


#day_2h = Day_List[day_iteration].between_time(start_hour, end_hour)
#pandas.DatetimeIndex.date
#df.loc['2015-08-12':'2015-08-10']
def alle_inputs_neu():
    
    df = alle_inputs()

    feiertag_liste = ['2017-10-02','2017-10-03','2017-10-30','2017-10-31',
                  '2017-11-01',
                  '2017-12-25','2017-12-26','2017-12-27','2017-12-28','2017-12-29','2017-12-30'   
                 ]

    df['tagestyp'] = 1 # Arbeitstag
    df['tagestyp'][df.index.normalize().isin(feiertag_liste)] = 0.5 # Feiertag
    df['tagestyp'][df.index.weekday > 4] = 0 # Wochenende

    df['tagestyp_morgen'] = df['tagestyp'].shift(periods=-len(df['tagestyp'][df.index.normalize().isin(['2017-10-02'])]), fill_value=df['tagestyp'][-1])

    return df





### FÜR LSTM:
def rolling_mean_training_data(num_past_periods=12):

    try:
        
        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_rolling_mean_{}.h5'.format(num_past_periods), 'r') as hf:
            training_data = hf['training_data'][:]
            label_data = hf['label_data'][:]

    except:
        alle_inputs_df = alle_inputs()
        
        rolling_mean_inputs_df = alle_inputs_df.rolling(num_past_periods).mean()
        rolling_mean_inputs = rolling_mean_inputs_df.to_numpy()

        training_data = []

        #steps = tqdm.tqdm(total=len(rolling_mean_inputs[:-num_past_periods]), desc='Training Data'.format(i+1), position=1)
        for i in tqdm(range(len(rolling_mean_inputs[:-num_past_periods]))):
            #steps.update(1)

            training_data = np.append(training_data, rolling_mean_inputs[i:i+num_past_periods])


        num_inputs_t = len(rolling_mean_inputs[0]) 
        num_t = len(rolling_mean_inputs[num_past_periods:])


        training_data = np.reshape(training_data, (num_t, num_past_periods, num_inputs_t))[num_past_periods:]
        label_data = rolling_mean_inputs_df['norm_total_power'].to_numpy()[num_past_periods:][num_past_periods:]

        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_rolling_mean_{}.h5'.format(num_past_periods), 'w') as hf:
            hf.create_dataset("training_data",  data=training_data)
            hf.create_dataset("label_data",  data=label_data)

    return training_data, label_data

def rolling_max_training_data(num_past_periods=12):

    try:
        
        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_rolling_max_{}.h5'.format(num_past_periods), 'r') as hf:
            training_data = hf['training_data'][:]
            label_data = hf['label_data'][:]

    except:
        alle_inputs_df = alle_inputs()
        
        rolling_max_inputs_df = alle_inputs_df.rolling(num_past_periods).max()
        rolling_max_inputs = rolling_max_inputs_df.to_numpy()

        training_data = []

        #steps = tqdm.tqdm(total=len(rolling_mean_inputs[:-num_past_periods]), desc='Training Data'.format(i+1), position=1)
        for i in tqdm(range(len(rolling_max_inputs[:-num_past_periods]))):
            #steps.update(1)

            training_data = np.append(training_data, rolling_max_inputs[i:i+num_past_periods])


        num_inputs_t = len(rolling_max_inputs[0]) 
        num_t = len(rolling_max_inputs[num_past_periods:])


        training_data = np.reshape(training_data, (num_t, num_past_periods, num_inputs_t))[num_past_periods:]
        label_data = rolling_max_inputs_df['norm_total_power'].to_numpy()[num_past_periods:][num_past_periods:]

        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_rolling_max_{}.h5'.format(num_past_periods), 'w') as hf:
            hf.create_dataset("training_data",  data=training_data)
            hf.create_dataset("label_data",  data=label_data)

    return training_data, label_data

def normal_training_data(num_past_periods=12):

    try:
        
        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_normal_{}.h5'.format(num_past_periods), 'r') as hf:
            training_data = hf['training_data'][:]
            label_data = hf['label_data'][:]

    except:
        alle_inputs_df = alle_inputs()
        
        normal_inputs = alle_inputs_df.to_numpy()

        training_data = []

        #steps = tqdm.tqdm(total=len(rolling_mean_inputs[:-num_past_periods]), desc='Training Data'.format(i+1), position=1)
        for i in tqdm(range(len(normal_inputs[:-num_past_periods]))):
            #steps.update(1)

            training_data = np.append(training_data, normal_inputs[i:i+num_past_periods])


        num_inputs_t = len(normal_inputs[0]) 
        num_t = len(normal_inputs[num_past_periods:])


        training_data = np.reshape(training_data, (num_t, num_past_periods, num_inputs_t))[num_past_periods:]
        label_data = alle_inputs_df['norm_total_power'].to_numpy()[num_past_periods:][num_past_periods:]

        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_normal_{}.h5'.format(num_past_periods), 'w') as hf:
            hf.create_dataset("training_data",  data=training_data)
            hf.create_dataset("label_data",  data=label_data)

    return training_data, label_data

def sequence_training_data(num_past_periods=12):

    try:
        
        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_sequence_{}.h5'.format(num_past_periods), 'r') as hf:
            training_data = hf['training_data'][:]
            label_data = hf['label_data'][:]

    except:
        alle_inputs_df = alle_inputs()
        
        sequence_inputs = alle_inputs_df.to_numpy()
        sequence_outputs = alle_inputs_df['norm_total_power'].to_numpy()

        label_data = []
        for i in tqdm(range(len(sequence_outputs[:-num_past_periods]))):
            label_data = np.append(label_data, sequence_outputs[i:i+num_past_periods])
        #num_inputs_t = len(sequence_outputs[0]) 
        num_t = len(sequence_outputs[num_past_periods:])
        label_data = np.reshape(label_data, (num_t, num_past_periods))[num_past_periods:][num_past_periods:]

        training_data = []
        for i in tqdm(range(len(sequence_inputs[:-num_past_periods]))):
            training_data = np.append(training_data, sequence_inputs[i:i+num_past_periods])
        num_inputs_t = len(sequence_inputs[0]) 
        num_t = len(sequence_inputs[num_past_periods:])
        training_data = np.reshape(training_data, (num_t, num_past_periods, num_inputs_t))[num_past_periods:-num_past_periods]

        with h5py.File(DATENSATZ_PATH+'datensatz/training_LSTM/training_und_label_data_sequence_{}.h5'.format(num_past_periods), 'w') as hf:
            hf.create_dataset("training_data",  data=training_data)
            hf.create_dataset("label_data",  data=label_data)

    return training_data, label_data




def peak_len(peak_value=42):

    sum_total_power = load_total_power().to_numpy()

    len_peak_list = []
    value_to_append = 0
    
    for v in sum_total_power:

        if v >= peak_value:
            value_to_append +=1
        elif value_to_append != 0:
            len_peak_list.append(value_to_append)
            value_to_append = 0

    #print(len_peak_list)
    #print(len(len_peak_list))
            

    return len_peak_list





def plot_leistungs_benutzung():

    global_var()

    sum_total_power = load_total_power()




    # Distribution KW-Bedarf
    #sns.distplot(sum_total_power, hist=False, kde_kws={"shade": True}, color="m")

    # Distribution KW-Bedarf-Steigung
    #sns.distplot(sum_total_power.diff().dropna(), hist=False, kde_kws={"shade": True}, color="m")
    '''
    plt.subplot(4,3,1)
    sns.distplot(peak_len(40), kde=False)

    plt.subplot(4,3,2)
    sns.distplot(peak_len(42), kde=False)

    plt.subplot(4,3,3)
    sns.distplot(peak_len(45), kde=False)

    plt.subplot(4,3,4)
    sns.distplot(peak_len(50), kde=False)

    plt.subplot(4,3,5)
    sns.distplot(peak_len(55), kde=False)

    plt.subplot(4,3,6)
    sns.distplot(peak_len(60), kde=False)

    plt.subplot(4,3,7)
    sns.distplot(peak_len(65), kde=False)

    plt.subplot(4,3,8)
    sns.distplot(peak_len(70), kde=False)

    plt.subplot(4,3,9)
    sns.distplot(peak_len(75), kde=False)

    plt.subplot(4,3,10)
    sns.distplot(peak_len(80), kde=False)

    plt.subplot(4,3,11)
    sns.distplot(peak_len(85), kde=False)

    plt.subplot(4,3,12)
    sns.distplot(peak_len(90), kde=False)


    plt.tight_layout()
    #plt.show()
    '''

#plot_leistungs_benutzung()



















