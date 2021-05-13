'''
There are four main heuristic approaches, with the goal to minimize the maximum energy peak. You can define those in the class with the parameter ``HEURISTIC_TYPE``.

1. `Single-Value` approximates the best global value (for all steps), that is used to determine the should energy consumption from the grid.

2. `Perfekt-Pred` finds the best should energy consumptions for each steps, under the assumption, that the future energy-need is perfectly predicted.

3. `LSTM-Pred` approximates the best should energy consumptions for each step, with LSTM-predicted future energy-need.

4. `Practical` tries to find a solution with LSTM-predictions of the next step (without knowledge about the predictions over all steps from the beginning like in approach 3).

The first approach can also have the goal to minimize the sum of cost instead of the maximum peak. Use `Single-Value-Reward` if you want to try this.
'''
import numpy as np

try:
    import main.common_func as cm
except:
    import common_func as cm


class heurisitc:
    '''
    Class that includes all heuristic approaches

    Args:
        env (object): Takes in a GYM environment, use the common_env to simulate the HIPE-Dataset.
        HEURISTIC_TYPE (string): Determines the type of heuristic to be used: 'Single-Value', 'Perfekt-Pred', 'LSTM-Pred' or 'Practical'
        threshold_dem (float): Determines the global maximum power consumption that shpuld be used from the grid (global SECG).
        use_SMS (bool): Can be used to deactivate the flying wheel when set to `False`
        use_LION (bool): Can be used to deactivate the lithium-ion battery when set to `False`
    '''
    def __init__(self, env, HEURISTIC_TYPE, threshold_dem=50, use_SMS=True, use_LION=True):

        self.env                         = env
        self.HEURISTIC_TYPE              = HEURISTIC_TYPE
        self.df                          = self.env.__dict__['df']
        self.vorher_global_zielverbrauch = 0
        self.prev_sum_reward             = 0
        self.global_zielverbrauch        = threshold_dem
        self.use_SMS                     = use_SMS
        self.use_LION                    = use_LION


    def global_single_value_for_max_peak(self, max_peak):
        '''
        Class-Function: Uses the mean-value theorem to calculate at the end of each episode a new global SECG (should energy-consumption from the grid)

        Args:
            max_peak (float):Takes in max_peak at the end of each episode

        Returns:
            float: New global SECG, after calculation with the mean-value theorem
        ''' 
        print('')
        print('Max-Peak:', max_peak)
        
        # If max_peak is greater than current SECG, SECG must become greater:
        if max_peak > self.global_zielverbrauch:
            neu_global_zielverbrauch = self.global_zielverbrauch + (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        # If max_peak is smaller than current SECG, SECG must become smaller:
        if max_peak <= self.global_zielverbrauch:
            neu_global_zielverbrauch = self.global_zielverbrauch - (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        self.vorher_global_zielverbrauch = self.global_zielverbrauch
        self.global_zielverbrauch = neu_global_zielverbrauch
        return neu_global_zielverbrauch


    def global_single_value_for_reward(self, sum_reward, positivity_value=10000):
        '''
        Class-Function: Uses the mean-value theorem to calculate a new SECG at the end of each episode, instead of minimizing the maximum peak, this function maximizes the sum of rewards.
        
        Args:
            sum_reward (float): Takes in the sum of rewards after each episode
            positivity_value (float): Should be a large number, which is added to the sum of rewards to make sure that negative rewards will be transformed into positive ones
            
        Returns:
            float: New global SECG, after calculation with the mean-value theorem
        '''
        print('')
        print('sum_reward:', sum_reward)
        sum_reward += positivity_value
        
        # If sum_reward is greater than prev_sum_reward, SECG must become greater:
        if sum_reward < self.prev_sum_reward:
            neu_global_zielverbrauch = self.global_zielverbrauch + (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        # If sum_reward is smaller than prev_sum_reward, SECG must become smaller:
        if sum_reward >= self.prev_sum_reward:
            neu_global_zielverbrauch = self.global_zielverbrauch - (abs(self.global_zielverbrauch - self.vorher_global_zielverbrauch) / 2)

        self.vorher_global_zielverbrauch = self.global_zielverbrauch
        self.global_zielverbrauch = neu_global_zielverbrauch
        self.prev_sum_reward = sum_reward
        return neu_global_zielverbrauch


    def find_optimum_for_perfect_pred(self):
        '''Class-Funtion: Prepares the SECG for each step, used when ``HEURISTIC_TYPE=Perfekt-Pred'``. Used before iterating through each step.'''
        
        print('Calculating optimal actions for perfect predictions...')

        # Setup iteration:
        power_to_shave = 0
        zielnetzverbrauch = []
        global_value = self.global_zielverbrauch

        power_dem_arr = self.df['norm_total_power'].to_numpy().copy()
        power_dem_arr *= self.env.__dict__['max_power_dem']
        print(power_dem_arr)

        # Iterate backwards through each step:
        for power_dem_step in power_dem_arr[::-1]:
            
            # Calculate power that should be provided by the batteries:
            if power_dem_step > global_value:
                power_to_shave += power_dem_step - global_value
                zielnetzverbrauch.append(global_value)
            
            # Calculate power to load the batteries:
            else:# power_dem_step <= global_value:
                if (power_to_shave + power_dem_step) > global_value:
                    power_to_shave -= (global_value - power_dem_step)
                    zielnetzverbrauch.append(global_value)
                else:
                    zielnetzverbrauch.append(power_to_shave + power_dem_step)
                    power_to_shave = 0

        self.heuristic_zielnetzverbrauch = zielnetzverbrauch[::-1]
        self.current_step = 0
        return power_to_shave


    def find_solution_for_imperfect_pred(self,LSTM_column):
        '''Funtion to prepare the SECG for each step, used when ``HEURISTIC_TYPE='LSTM-Pred'``. Used before iterating through each step.

        Args:
            LSTM_column (string): Name of the column in the passed dataframe `df`, this column has to contain the predictions you want to use.
        '''
        print('Calculating optimal actions for imperfect predictions...')

        # Setup iteration:
        power_to_shave = 0
        zielnetzverbrauch = []
        power_dem_arr = self.env.__dict__['df'][LSTM_column].to_numpy().copy()
        power_dem_arr *= self.env.__dict__['max_power_dem']
        print(power_dem_arr)
        global_value = self.global_zielverbrauch

        # Iterate backwards through each step:
        for power_dem_step in power_dem_arr[::-1]:
            
            # Calculate power that should be provided by the batteries:
            if power_dem_step > global_value:
                power_to_shave += power_dem_step - global_value
                zielnetzverbrauch.append(global_value)
            
            # Calculate power to load the batteries:
            else:# power_dem_step <= global_value:
                if (power_to_shave + power_dem_step) > global_value:
                    power_to_shave -= (global_value - power_dem_step)
                    zielnetzverbrauch.append(global_value)
                else:
                    zielnetzverbrauch.append(power_to_shave + power_dem_step)
                    power_to_shave = 0
        
        self.heuristic_zielnetzverbrauch = zielnetzverbrauch[::-1]
        self.current_step = 0
        return power_to_shave

    def find_practical_solution(self, LSTM_column):
        '''
        Funtion to prepare the SECG for each step, used when ``HEURISTIC_TYPE='Practical'``. Used before iterating through each step.

        Args:
            LSTM_column (string): Name of the column in the passed dataframe `df`, this column has to contain the predictions you want to use.
        '''
        print('Calculating practical actions for imperfect predictions...')

        # Setup iteration:
        power_to_shave = 0
        zielnetzverbrauch = []
        power_dem_arr = self.env.__dict__['power_dem_arr']
        sequence_pred = self.env.__dict__['df'][LSTM_column].to_numpy()
        sequence_pred *= self.env.__dict__['max_power_dem']
        global_value = self.global_zielverbrauch

        # Iterate trhough each step (forwards):
        for i in range(len(sequence_pred)):
            power_dem_step     = power_dem_arr[i]
            sequence_pred_step = sequence_pred[i]

            # Set SECG to maximum:
            if power_dem_step > global_value:
                zielnetzverbrauch.append(global_value)
            
            # Determine the SECG based on the prediction:
            else:
                if sequence_pred_step > global_value:
                    zielnetzverbrauch.append(global_value)
                else:
                    zielnetzverbrauch.append(power_dem_step)
        
        self.heuristic_zielnetzverbrauch = zielnetzverbrauch[::-1]
        self.current_step = 0

    def act(self, SMS_PRIO = 0):
        '''Function, in which the heuristic decides an action, based on HEURISTIC_TYPE previously set.
        
        Args:
            SMS_PRIO (float): Is either set to 0 or 1:
                
                0 sets the priority of SMS to true
                
                1 sets the priority of SMS to false

        Returns:
            SECG, SMS_PRIO (tuple):
            
            SECG (float): should energy-consumption from the grid at step
            
            SMS_PRIO (float): priority of SMS at step
        '''
        if self.HEURISTIC_TYPE == 'Single-Value' or self.HEURISTIC_TYPE == 'Single-Value-Reward':
            return [self.global_zielverbrauch, SMS_PRIO] # 0 bedeutet dass SMS-Priority aktiviert wird.
        
        elif any(self.HEURISTIC_TYPE in s for s in ['Perfekt-Pred','LSTM-Pred','Practical']):
            #self.HEURISTIC_TYPE == 'Perfekt-Pred' or self.HEURISTIC_TYPE =='LSTM-Pred':
            action = [self.heuristic_zielnetzverbrauch[self.current_step], SMS_PRIO]
            self.current_step += 1
            return action
        else:
            raise Exception("HEURISTIC_TYPE not understood. HEURISTIC_TYPE must be: 'Single-Value', 'Single-Value-Reward', 'Perfekt-Pred', 'LSTM-Pred'")



    def printer(self, i, max_i):
        '''
        Helper function to print helpful information about the mean-value process at the process bar
        
        Args:
            i (int): Current iteration step
            max_i (int): The number of all iterations
        '''
        if self.HEURISTIC_TYPE == 'Single-Value':
            cm.print_progress("Target Demand - {}, Progress".format(self.global_zielverbrauch), i, max_i)
        elif self.HEURISTIC_TYPE == 'Single-Value-Reward':
            cm.print_progress("Target Demand - {}, Sum Reward - {}, Progress".format(self.global_zielverbrauch, self.prev_sum_reward), i, max_i)
        else:
            cm.print_progress('Progress',i , max_i)


    def calculate(self, epochs=1, LSTM_column='normal'):
        '''
        Main function that tests the chosen heursitic

        Args:
            epochs (int): Number of epochs. Note that this should be one for all heuristics except ``HEURISTIC_TYPE='Single-Value'`` and ``HEURISTIC_TYPE='Single-Value-Reward'``.
            LSTM_column (string): Name of the column in the passed dataframe `df`, this column has to contain the predictions you want to use.
        '''
        # Check if the heuristic needs preparation:
        power_to_shave = None
        if self.HEURISTIC_TYPE == 'Perfekt-Pred':
            power_to_shave = self.find_optimum_for_perfect_pred()
        elif self.HEURISTIC_TYPE == 'LSTM-Pred':
            power_to_shave = self.find_solution_for_imperfect_pred(LSTM_column)
        elif self.HEURISTIC_TYPE == 'Practical':
            self.find_practical_solution(LSTM_column)

        print('Testing',self.HEURISTIC_TYPE,'with a treshold of',self.global_zielverbrauch,'for',epochs,'Epochs')
        #print('Sum of initial battery charge:',power_to_shave)
        
        # Iterate Testing:
        for e in range(epochs):

            print('\nepoch:', e)

            cur_state = self.env.reset()
            self.env.set_soc_and_current_state(SoC=power_to_shave)

            epoch_len = len(self.df)
            for step in range(epoch_len):

                action                               = self.act(SMS_PRIO=0)# 0 heißt Prio ist aktiv
                new_state, reward, done, max_peak, _ = self.env.step(action)        
                self.printer(step, epoch_len)
                if done:
                    break
            
            # Calculate new SECG if necessary for chosen heuristic:
            if self.HEURISTIC_TYPE == 'Single-Value':
                neu_global_zielverbrauch = self.global_single_value_for_max_peak(max_peak)
            elif self.HEURISTIC_TYPE == 'Single-Value-Reward':
                neu_global_zielverbrauch = self.global_single_value_for_reward(self.env.__dict__['reward_maker'].__dict__['sum_cost_saving'])

            print('cost savings:',self.env.__dict__['reward_maker'].__dict__['sum_cost_saving'])


        # Return new SECG if necessary for chosen heuristic:
        if self.HEURISTIC_TYPE == 'Single-Value' or self.HEURISTIC_TYPE == 'Single-Value-Reward':
            return neu_global_zielverbrauch, self.env.__dict__['reward_maker'].__dict__['sum_cost_saving']


