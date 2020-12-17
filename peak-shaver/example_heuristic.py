'''
Example of an Agent that uses heuristics.
'''
from datetime import datetime

from main.common_func import max_seq, mean_seq, training, testing
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.logger import Logger
from main.reward_maker import reward_maker
from main.common_env import common_env

#Import the heuristics:
from main.agent_heuristic import heurisitc

'''
### 'Single-Value-Heuristic' ###
Bestimmt einen einzelnen Zielnetzetzverbrauch, der für alle Steps benutzt wird.
-> selbe nochmal aber mit Reward-Focus!

### 'Perfekt-Pred-Heuristic' ###
Alle zukünfitigen Werte sind bekannt.

### 'LSTM-Pred-Heuristic' ###
Heuristik mit realistischen Inputs, sprich höchstens Vorhersehungen mit LSTM möglich. 
'''

def use_heuristic(HEURISTIC_TYPE='Perfekt-Pred-Heuristic', epochs=1,
                 threshold_dem=50, deactivate_SMS=False, deactivate_LION=False):

    # Naming the agent and setting up the directory path:
    now            = datetime.now()
    NAME           = str(round(threshold_dem))+'_TARGET_VALUE_'+HEURISTIC_TYPE+now.strftime("_%d-%m-%Y_%H-%M-%S")
    D_PATH         = '_small_d/'

    # Load the dataset:
    main_dataset = mainDataset(
        D_PATH=D_PATH,
        period_string_min='15min',
        full_dataset=True)

    # Normalized dataframe:
    df = main_dataset.make_input_df(
        drop_main_terminal=False,
        use_time_diff=True,
        day_diff='holiday-weekend')

    # Sum of the power demand dataframe (nor normalized):
    power_dem_df = main_dataset.load_total_power()[24:-12]

    # Load the LSTM input dataset:
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

    # Making predictions:
    normal_predictions = wahrsager(lstm_dataset, power_dem_df, TYPE='NORMAL').pred()[:-12]
    seq_predictions    = wahrsager(lstm_dataset, power_dem_df, TYPE='SEQ', num_outputs=12).pred()

    # Adding the predictions to the dataset:
    df            = df[24:-12]
    df['normal']  = normal_predictions
    df['seq_max'] = max_seq(seq_predictions)

    logger = Logger(NAME,D_PATH)

    # Setup reward_maker
    r_maker = reward_maker(
        LOGGER                  = logger,
        COST_TYPE               = 'exact_costs',
        R_TYPE                  = 'savings_focus',
        R_HORIZON               = 'single_step',
        cost_per_kwh            = 0.2255,
        LION_Anschaffungs_Preis = 34100,
        LION_max_Ladezyklen     = 1000,
        SMS_Anschaffungs_Preis  = 115000/3,
        SMS_max_Nutzungsjahre   = 20,
        Leistungspreis          = 102,
        logging_list            = ['cost_saving','exact_costs','sum_exact_costs','sum_cost_saving'],
        deactivate_SMS          = deactivate_SMS,
        deactivate_LION         = deactivate_LION)


    # Lade Environment:
    env = common_env(
        reward_maker   = r_maker,
        df             = df,
        power_dem_df   = power_dem_df,
        input_list     = ['norm_total_power','normal','seq_max'],
        max_SMS_SoC    = 12/3,
        max_LION_SoC   = 54,
        PERIODEN_DAUER = 15,
        ACTION_TYPE    = 'contin',
        OBS_TYPE       = 'contin',
        AGENT_TYPE     = 'heuristic')

    agent = heurisitc(
        env = env,
        HEURISTIC_TYPE = HEURISTIC_TYPE,
        threshold_dem  = threshold_dem)


    return agent.calculate(epochs=epochs)


def test_threshold_for_all_heuristics():

    threshold_dem = use_heuristic('Single-Value-Heuristic', epochs=15, threshold_dem=50)
    #use_heuristic('Perfekt-Pred-Heuristic', threshold_dem=threshold_dem)
    #use_heuristic('LSTM-Pred-Heuristic', threshold_dem=threshold_dem)
    #use_heuristic('Practical-Heuristic', threshold_dem=threshold_dem)

def test_for_different_thresholds(HEURISTIC_TYPE,threshold_list=[10,20,30,40,50,60,70,80,90]):
    for threshold in threshold_list:
        use_heuristic(HEURISTIC_TYPE, threshold_dem=threshold)

def test_battery_activations(HEURISTIC_TYPE,threshold_dem=60):

    use_heuristic(HEURISTIC_TYPE, threshold_dem=threshold_dem)
    use_heuristic(HEURISTIC_TYPE, threshold_dem=threshold_dem, deactivate_SMS=True,)
    use_heuristic(HEURISTIC_TYPE, threshold_dem=threshold_dem, deactivate_LION=True)
    use_heuristic(HEURISTIC_TYPE, threshold_dem=threshold_dem, deactivate_SMS=True, deactivate_LION=True)

def main():

    test_threshold_for_all_heuristics()
    #test_for_different_thresholds('Perfekt-Pred-Heuristic')
    #test_battery_activations('Perfekt-Pred-Heuristic')

    #use_heuristic('Perfekt-Pred-Heuristic', threshold_dem=47.7)
    #use_heuristic('Perfekt-Pred-Heuristic', threshold_dem=threshold_dem)

if __name__ == "__main__":
    main()