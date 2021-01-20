'''
There are four main heuristic approaches, with the goal to minimize the maximum energy peak. You can define those in the class with the parameter ``HEURISTIC_TYPE``.

1. `Single-Value-Heuristic` approximates the best global value (for all steps), that is used to determine the should energy consumption from the grid.

2. `Perfekt-Pred-Heuristic` finds the best should energy consumptions for each steps, under the assumption, that the future energy-need is perfectly predicted.

3. `LSTM-Pred-Heuristic` approximates the best should energy consumptions for each step, with LSTM-predicted future energy-need.

4. `Practical-Heuristic` tries to find a solution with LSTM-predictions of the next step (without knowledge about the predictions over all steps from the beginning like in approach 3).

The first approach can also have the goal to minimize the sum of cost instead of the maximum peak. Use `Single-Value-Heuristic-Reward` if you want to try this.
'''
from datetime import datetime

from common_settings import dataset_and_logger
from main.common_func import max_seq, mean_seq, training, testing
from main.reward_maker import reward_maker
from main.common_env import common_env

#Import the heuristics:
from main.agent_heuristic import heurisitc


def use_heuristic(HEURISTIC_TYPE='Perfekt-Pred-Heuristic', test_name='', epochs=1,
                 threshold_dem=50, deactivate_SMS=False, deactivate_LION=False):

    # Naming the agent:
    now            = datetime.now()
    NAME           = 'heuristic_'+test_name+'_'+HEURISTIC_TYPE+'_'+str(round(threshold_dem))+'_t-stamp'+now.strftime("_%d-%m-%Y_%H-%M-%S")
    
    # Import dataset and logger based on the common settings
    df, power_dem_df, logger, period_min = dataset_and_logger(NAME)


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
        LION_max_Ladezyklen     = 1000,
        SMS_Anschaffungs_Preis  = 115000/3,
        SMS_max_Nutzungsjahre   = 20,
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
        input_list     = ['norm_total_power','normal','seq_max'],
        # Batters stats:
        max_SMS_SoC    = 12/3,
        max_LION_SoC   = 54,
        # Period length in minutes:
        PERIODEN_DAUER = period_min,
        # Heuristics can only use continious values:
        ACTION_TYPE    = 'contin',
        OBS_TYPE       = 'contin',
        # Define heuristic usage:
        AGENT_TYPE     = 'heuristic')
    # Use the complete dataset (no validation split):
    env.use_all_data()

    # Setup Agent
    agent = heurisitc(
        env = env,
        HEURISTIC_TYPE = HEURISTIC_TYPE,
        threshold_dem  = threshold_dem)


    return agent.calculate(epochs=epochs)


def test_threshold_for_all_heuristics():
    # Find maximum performance of battery configuration:
    threshold_dem = use_heuristic('Single-Value-Heuristic', test_name='find_threshold', epochs=15, threshold_dem=50)
    
    # Test all heuristic:
    use_heuristic('Perfekt-Pred-Heuristic', test_name='Compare_Approches', threshold_dem=threshold_dem)
    use_heuristic('LSTM-Pred-Heuristic', test_name='Compare_Approches', threshold_dem=threshold_dem)
    use_heuristic('Practical-Heuristic', test_name='Compare_Approches', threshold_dem=threshold_dem)


def test_for_different_thresholds(HEURISTIC_TYPE,threshold_list=[10,20,30,40,50,60,70,80,90]):
    # Test different threshold values
    for threshold in threshold_list:
        use_heuristic(HEURISTIC_TYPE, test_name='Tresholds', threshold_dem=threshold)


def test_battery_activations(HEURISTIC_TYPE,threshold_dem=60):
    # Test battery-configurations:
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem)
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem, deactivate_SMS=True,)
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem, deactivate_LION=True)
    use_heuristic(HEURISTIC_TYPE, test_name='Configurations', threshold_dem=threshold_dem, deactivate_SMS=True, deactivate_LION=True)


def main():

    # General test with max performance threshhold:
    test_threshold_for_all_heuristics()

    # Test the three main heuristics with different thresholds and with different battery activations:
    for HEURISTIC_TYPE in ['Perfekt-Pred-Heuristic','LSTM-Pred-Heuristic','Practical-Heuristic']:
        test_for_different_thresholds(HEURISTIC_TYPE)
        test_battery_activations(HEURISTIC_TYPE)


if __name__ == "__main__":
    main()
