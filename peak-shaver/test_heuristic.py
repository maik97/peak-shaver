'''
There are four main heuristic approaches, with the goal to minimize the maximum energy peak. You can define those in the class with the parameter ``HEURISTIC_TYPE``.

1. `Single-Value-Heuristic` approximates the best global value (for all steps), that is used to determine the should energy consumption from the grid.

2. `Perfekt-Pred` finds the best should energy consumptions for each steps, under the assumption, that the future energy-need is perfectly predicted.

3. `LSTM-Pred` approximates the best should energy consumptions for each step, with LSTM-predicted future energy-need.

4. `Practical` tries to find a solution with LSTM-predictions of the next step (without knowledge about the predictions over all steps from the beginning like in approach 3).

The first approach can also have the goal to minimize the sum of cost instead of the maximum peak. Use `Single-Value-Heuristic-Reward` if you want to try this.
'''
from datetime import datetime

from common_settings import dataset_and_logger, load_specific_sets, load_logger
from main.common_func import max_seq, mean_seq, training, testing
from main.reward_maker import reward_maker
from main.common_env import common_env

#Import the heuristics:
from main.agent_heuristic import heurisitc


def use_heuristic(HEURISTIC_TYPE='Perfekt-Pred', test_name='', epochs=1,
                 threshold_dem=50, deactivate_SMS=False, deactivate_LION=False,
                 num_past_periods=12,num_outputs=12,TYPE_LIST=['NORMAL'],seq_transform=['MAX'],
                 max_SMS_SoC=25,max_LION_SoC=54):

    # Naming the agent:
    now = datetime.now()
    if test_name != 'Configurations':
        NAME = 'heuristic_'+test_name+'_'+HEURISTIC_TYPE+'_'+str(round(threshold_dem))+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")
    else:
        if deactivate_SMS == False:
            SMS_string = 'SMS'
        else:
            SMS_string = 'None'
        if deactivate_LION == False:
            LION_string = 'LION'
        else:
            LION_string = 'None'

        NAME = str(round(threshold_dem))+'-'+LION_string+'-'+SMS_string
        NAME = 'heuristic_'+test_name+'_'+HEURISTIC_TYPE+'_'+NAME+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")

    
    # Import dataset and logger based on the common settings
    df, power_dem_df, input_list = load_specific_sets(num_past_periods, num_outputs, TYPE_LIST, seq_transform)
    logger, period_min           = load_logger(NAME, only_per_episode=False)


    # Setup reward_maker
    r_maker = reward_maker(
        LOGGER                  = logger,
        # Settings:
        COST_TYPE               = 'exact_costs',
        R_TYPE                  = 'savings_focus',
        R_HORIZON               = 'single_step',
        # Parameter to calculate costs:
        cost_per_kwh            = 0.2255,
        LION_Anschaffungs_Preis = 34100,
        LION_max_Ladezyklen     = 6000,
        SMS_Anschaffungs_Preis  = 55000,#115000/3,
        SMS_max_Nutzungsjahre   = 25,
        Leistungspreis          = 102,
        # Setup logging tags:
        logging_list            = ['cost_saving','exact_costs','sum_exact_costs','sum_cost_saving'],
        # Deactivation options for the batteries:
        deactivate_SMS          = deactivate_SMS,
        deactivate_LION         = deactivate_LION)


    # Lade Environment:
    env = common_env(
        reward_maker   = r_maker,
        df             = df,
        power_dem_df   = power_dem_df,
        # Datset Inputs for the states:
        input_list     = input_list,
        # Batters stats:
        max_SMS_SoC        = max_SMS_SoC,#/1.2,
        max_LION_SoC       = max_LION_SoC,#/1.2,
        LION_max_entladung = 50,
        SMS_max_entladung  = 100,
        SMS_entladerate    = 0.72,
        LION_entladerate   = 0.00008,
        # Period length in minutes:
        PERIODEN_DAUER = period_min,
        # Heuristics can only use continious values:
        ACTION_TYPE    = 'contin',
        OBS_TYPE       = 'contin',
        # Define heuristic usage:
        AGENT_TYPE     = 'heuristic',
        val_split      = 0)

    # Use the complete dataset (no validation split):
    #env.use_all_data()

    # Setup Agent
    agent = heurisitc(
        env = env,
        HEURISTIC_TYPE = HEURISTIC_TYPE,
        threshold_dem  = threshold_dem)


    return agent.calculate(epochs=epochs,LSTM_column=input_list[-1])


def test_threshold_for_all_heuristics():
    # Find maximum performance of battery configuration:
    #threshold_dem = use_heuristic('Single-Value', test_name='find_threshold', epochs=15, threshold_dem=50)[0]
    threshold_dem = 40
    # Test all heuristic:
    use_heuristic('Perfekt-Pred', test_name='Compare_Approches', threshold_dem=threshold_dem)
    use_heuristic('LSTM-Pred', test_name='Compare_Approches', threshold_dem=threshold_dem)
    use_heuristic('Practical', test_name='Compare_Approches', threshold_dem=threshold_dem)
    #use_heuristic('LSTM-SEQ-Pred', test_name='Compare_Approches', threshold_dem=threshold_dem)


def test_for_different_thresholds(HEURISTIC_TYPE,threshold_list=[20,25,30,40,50,60]):
    # Test different threshold values
    for threshold in threshold_list:
        use_heuristic(HEURISTIC_TYPE, test_name='Tresholds', threshold_dem=threshold)


def test_battery_activations(HEURISTIC_TYPE,threshold_dem=40):
    # Test battery-configurations:
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem)
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem, deactivate_SMS=True,)
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem, deactivate_LION=True)
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem, deactivate_SMS=True, deactivate_LION=True)


def test_lstm_types(threshold_dem=40,num_past_periods=24,num_outputs=24):
    #seq_transform=['MAX']
    name = 'Pred-'+str(num_past_periods)+'-'+str(num_outputs)+'_'
    use_heuristic('LSTM-Pred', test_name=name+'NORMAL', TYPE_LIST=['NORMAL'], threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)
    use_heuristic('LSTM-Pred', test_name=name+'MAX', TYPE_LIST=['MAX'], threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)
    use_heuristic('LSTM-Pred', test_name=name+'MEAN', TYPE_LIST=['MEAN'], threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)
    use_heuristic('LSTM-Pred', test_name=name+'MAX-LABEL-SEQ', TYPE_LIST=['MAX_LABEL_SEQ'], threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)
    use_heuristic('LSTM-Pred', test_name=name+'MEAN-LABEL-SEQ', TYPE_LIST=['MEAN_LABEL_SEQ'], threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)
    use_heuristic('LSTM-Pred', test_name=name+'SEQ-MAX', TYPE_LIST=['SEQ'], seq_transform=['MAX'],threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)
    use_heuristic('LSTM-Pred', test_name=name+'SEQ-MEAN', TYPE_LIST=['SEQ'], seq_transform=['MEAN'], threshold_dem=threshold_dem, num_past_periods=num_past_periods,num_outputs=num_outputs)

def test_max_reward(threshold_dem=50,max_LION_SoC=54,max_SMS_SoC=25,deactivate_SMS=False,deactivate_LION=False):
    #threshold_dem = use_heuristic('Single-Value-Reward', test_name='find_reward_threshold', epochs=15, threshold_dem=threshold_dem)
    return use_heuristic(
    	'Single-Value', test_name='find_threshold', epochs=15, threshold_dem=threshold_dem,
    	max_LION_SoC=max_LION_SoC,max_SMS_SoC=max_SMS_SoC,
    	deactivate_SMS=deactivate_SMS,deactivate_LION=deactivate_LION)


def main():
    threshold_dem = use_heuristic('Single-Value', test_name='find_threshold', epochs=15, threshold_dem=50)
    threshold_dem = use_heuristic('Single-Value', test_name='final', epochs=1, threshold_dem=threshold_dem)


    #use_heuristic('LSTM-Pred', test_name='lstm-max', threshold_dem=40)
    #use_heuristic('Practical', test_name='lstm-max', threshold_dem=40)

    # General test with max performance threshhold:
    #test_threshold_for_all_heuristics()
    #test_battery_activations('Perfekt-Pred',threshold_dem=40)
    
    # Test the three main heuristics with different thresholds and with different battery activations:
    #for HEURISTIC_TYPE in ['Perfekt-Pred','LSTM-Pred','Practical']:
        #test_for_different_thresholds(HEURISTIC_TYPE)
        #test_battery_activations(HEURISTIC_TYPE)
        #use_heuristic(HEURISTIC_TYPE, test_name='test_rewards', threshold_dem=100, deactivate_SMS=True, deactivate_LION=True)



    list_lion = [test_max_reward(max_LION_SoC=value,deactivate_SMS=True)[1] for value in [10,15,20]]

    list_sms = [test_max_reward(max_SMS_SoC=value,deactivate_LION=True)[1] for value in [10,15,20]]

    '''
    for num in [6,12,24]:
        test_lstm_types(threshold_dem=40,num_past_periods=num,num_outputs=num)
    
    for num in [6,12]:
        test_lstm_types(threshold_dem=40,num_past_periods=24,num_outputs=num)
    '''


if __name__ == "__main__":
    main()
