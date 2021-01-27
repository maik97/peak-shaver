import pandas as pd
import numpy as np

from collections import deque


class reward_maker():
    '''
    Class for reward calculations. 

    Args:
        LOGGER (object): Logs scalars to tensorboard without tensor ops, see :class:`logger.Logger`
        COST_TYPE (string): Mode by which costs are calculated. Use'exact_costs', 'yearly_costs' or 'max_peak_focus'.
        R_TYPE (string): Mode by which rewards are calulated. Use 'costs_focus', 'positive' or 'savings_focus' (use 'yearly_costs' as COST_TYPE when using this mode)
        R_HORIZON (string): Mode that determines the range of steps to calculate the reward. Use 'single_step' (calculates reward at each step seperatly), 'episode' (calculates the reward for complete dataset), or an integer for multi-step (Number of steps for multi-step rewards).
        M_STRATEGY (string): Use None when R_HORIZON is set to 'single_step'. For multi-step rewards use 'sum_to_terminal', 'average_to_neighbour' or 'recurrent_to_Terminal'.
        cost_per_kwh (float): Cost of 1 kwh in €
        LION_Anschaffungs_Preis (float): Cost of one lithium-ion battery in €
        LION_max_Ladezyklen (int): Number of maximum charging cycles of the lithium-ion battery
        SMS_Anschaffungs_Preis (float): Cost of one flywheel storage unit in €
        SMS_max_Nutzungsjahre (int): Number of years a flywheel storage can be used
        Leistungspreis (float): Cost of maximum peak per year calculated by €/kw 
        focus_peak_multiplier (float): Factor by which the peak-costs are multiplied, used when COST_TYPE is set to 'max_peak_focus'
        logging_list (list):  Logs cost with :class:`logger.Logger`. Possible strings in list are 'exact_costs','costs_focus','single_step','sum_exact_costs','sum_costs_focus','sum_single_step'.
        deactivate_SMS (bool): Can be used to deactivate the flying wheel when set to `True`
        deactivate_LION (bool): Can be used to deactivate the lithium-ion battery when set to `True`
    '''
    def __init__(
            self, LOGGER,
            COST_TYPE               = 'exact_costs',     # 'yearly_costs', 'max_peak_focus'
            R_TYPE                  = 'costs_focus',     # 'positive','savings_focus' -> für 'saving_focus' müsste 'yearly_costs' benutzt werden
            R_HORIZON               = 'single_step',     # 'episode', 'single_step', integer for multi-step
            M_STRATEGY              = 'sum_to_terminal', # 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
            cost_per_kwh            = 0.07,  # in €
            LION_Anschaffungs_Preis = 32000, # in €
            LION_max_Ladezyklen     = 1000,
            SMS_Anschaffungs_Preis  = 10000, # in €
            SMS_max_Nutzungsjahre   = 20,    # in Jahre
            Leistungspreis          = 90,    # in €
            focus_peak_multiplier   = 4,     # multiplier for max_peak costs
            logging_list            = [None], #['exact_costs','costs_focus','single_step','sum_exact_costs','sum_costs_focus','sum_single_step','step_reward']
            deactivate_SMS          = False,
            deactivate_LION         = False,
            ):

        self.LOGGER                  = LOGGER
        self.COST_TYPE               = COST_TYPE
        self.R_TYPE                  = R_TYPE
        self.R_HORIZON               = R_HORIZON
        self.M_STRATEGY              = M_STRATEGY
        self.cost_per_kwh            = cost_per_kwh
        self.LION_Anschaffungs_Preis = LION_Anschaffungs_Preis
        self.LION_max_Ladezyklen     = LION_max_Ladezyklen
        self.SMS_Anschaffungs_Preis  = SMS_Anschaffungs_Preis
        self.SMS_max_Nutzungsjahre   = SMS_max_Nutzungsjahre
        self.Leistungspreis          = Leistungspreis
        self.focus_peak_multiplier   = focus_peak_multiplier
        self.logging_list            = logging_list
        self.deactivate_SMS          = deactivate_SMS
        self.deactivate_LION         = deactivate_LION

        if self.R_TYPE == 'savings_focus' and self.COST_TYPE != 'yearly_costs':
            print('COST_TYPE is', self.COST_TYPE, "but should be 'yearly_costs', when R_TYPE is 'savings_focus'")
            print("Changing: COST_TYPE is now 'yearly_costs' for accurate results")
            self.COST_TYPE = 'yearly_costs'

        # Init Logging
        


    def pass_env(self, PERIODEN_DAUER, steps_per_episode, max_power_dem, mean_power_dem, sum_power_dem, max_rolling_power_dem):
        '''
        Takes Init Parameter from GYM environment for the HIPE dataset, initiates all other necessary parameters and calculates the costs without peak shaving.
        
        Args:
            PERIODEN_DAUER (int):
            steps_per_episode (int):
            max_power_dem (float):
            mean_power_dem (float):
            sum_power_dem (float):
        '''
        self.PERIODEN_DAUER    = PERIODEN_DAUER
        self.steps_per_episode = steps_per_episode
        self.max_power_dem     = max_power_dem
        self.mean_power_dem    = mean_power_dem
        self.sum_power_dem     = sum_power_dem
        self.max_rolling_power_dem = max_rolling_power_dem

        # Init Parameter Zeit:
        self.minutes_per_year  = 525600
        self.steps_per_year    = self.minutes_per_year  / self.PERIODEN_DAUER

        self.minutes_per_week  = 10080
        self.steps_per_week    = self.minutes_per_week / self.PERIODEN_DAUER
        
        self.minutes_per_day   = 1440
        self.steps_per_day     = self.minutes_per_day  / self.PERIODEN_DAUER

        self.auf_jahr_rechnen  = self.steps_per_year   / self.steps_per_episode
        
        # Init Parameter Kosten:
        cost_power, cost_peak  = self.get_sum_of_usual_costs()
        self.sum_usual_costs   = cost_power + cost_peak
        print('\nJahres-Kosten ohne Peak-Shaving:')
        print('Stromkosten:',  round(cost_power,2))
        print('Peak-Kosten:',  round(cost_peak,2))
        print('Gesamtkosten:', round(self.sum_usual_costs,2),'\n')

        self.power_dem_list = []
        self.peak_diff_list = []
        self.cost_peak_test = []
        self.cost_strom_test =[]

    def get_sum_of_usual_costs(self):
        '''
        Calculation of costs without peak shaving
        '''
        cost_power = self.sum_power_dem * self.PERIODEN_DAUER * (self.cost_per_kwh/60) * self.auf_jahr_rechnen
        #c * steps   #kw * steps           #min pro step         #c / (kw * min)

        cost_peak  = self.max_rolling_power_dem * self.Leistungspreis 
        #c * steps   #kw                  # c / kw * jahr
        return cost_power, cost_peak


    def get_reward_range(self, x=100):
        '''
        Sets reward range for a GYM environmet

        Args:
            x (int): Integer x sets range to (-x,x)
        '''
        MIN_REWARD = -x
        MAX_REWARD = x
        return (MIN_REWARD, MAX_REWARD)


    def pass_state(self, done, day_max_peak, week_max_peak, episode_max_peak,
                   power_dem, LION_nutzung, sum_steps, step_counter_episode):
        '''
        Passes necessary variables of a state
        '''

        self.done = done

        if self.episode_max_peak < episode_max_peak:
            self.max_peak_diff    = episode_max_peak - self.episode_max_peak
            #print(self.max_peak_diff, self.episode_max_peak, episode_max_peak)   
            self.episode_max_peak = episode_max_peak
        else:
            self.max_peak_diff = 0


        self.power_dem        = power_dem
        self.LION_nutzung     = LION_nutzung

        self.sum_steps = sum_steps
        self.step_counter_episode = step_counter_episode


    def reset(self):
        '''
        Resets initiation, used when GYM environment is reset.
        '''
        self.episode_max_peak = 0

        # Reset Episode Sums
        self.sum_step_reward  = 0
        self.sum_exact_costs  = 0
        self.sum_yearly_costs = 0
        self.sum_cost_saving  = 0

        # Reset memory of Multi-Step-Horizon:
        if self.R_HORIZON == 'episode':
            self.memory = deque(maxlen=self.steps_per_episode)
            self.reward_list = np.zeros((self.steps_per_episode))
        elif self.R_HORIZON != 'single_step':
            try:
                self.memory = deque(maxlen=int(self.R_HORIZON))
                self.reward_list = np.zeros((self.steps_per_episode))
                #return self.R_HORIZON
            
            except:
                raise Exception("Failed to initialized memory for Multi-Step-Horizon: R_HORIZON must be an integer.",
                                "R_HORIZON was set to:", self.R_HORIZON,
                                "Alternatively use 'single_step' or 'episode', instead of an integer")

    def asfjulfsda(self):
        cost_power = self.sum_power_dem * self.PERIODEN_DAUER * (self.cost_per_kwh/60) * self.auf_jahr_rechnen
        #c * steps   #kw * steps           #min pro step         #c / (kw * min)

        cost_peak  = self.max_power_dem * self.Leistungspreis 
        #c * steps   #kw                  # c / kw * jahr

    def cost_function(self, interval_sum_power_dem, sum_LION_nutzung, max_peak_diff, observed_period):
        '''
        Calculates the cost at a step when peak shaving is used
        '''

        cost_power = interval_sum_power_dem * self.PERIODEN_DAUER * (self.cost_per_kwh/60)
        #c * steps   #kw * steps              #min pro step            #c / (kw * min)

        if self.deactivate_LION == False:
            cost_LION  = sum_LION_nutzung * (self.LION_Anschaffungs_Preis / self.LION_max_Ladezyklen)       # -> Abschreibung über Nutzung
            #c * steps   #nutzung * steps   #c                              #max_nutzung
        else:
            cost_LION = 0
        
        if self.deactivate_SMS == False:
            cost_SMS   = (self.SMS_Anschaffungs_Preis / (self.SMS_max_Nutzungsjahre*self.steps_per_year)) #* observed_period       # -> Abschreibung über Zeit
            #c * steps   #c                             #max_nutzung in jahre         #steps / (steps*jahr)
        else:
            cost_SMS = 0
        
        cost_peak = max_peak_diff * self.Leistungspreis #* (self.steps_per_episode /self.steps_per_year)
        #c * steps   #kw        # c / kw * jahr       #steps / (steps*jahr)

        self.power_dem_list.append(interval_sum_power_dem)
        self.peak_diff_list.append(max_peak_diff)
        self.cost_peak_test.append(cost_peak)
        self.cost_strom_test.append(cost_power)
        if self.done == True:
            print(len(self.power_dem_list))
            print(sum(self.power_dem_list))
            print(sum(self.peak_diff_list)*self.Leistungspreis)
            print(np.mean(self.power_dem_list))
            print(sum(self.cost_peak_test))
            print(sum(self.cost_strom_test)*self.auf_jahr_rechnen)
            exit()

        return cost_power, cost_LION, cost_SMS, cost_peak

    def get_reward(self):
        '''
        Function to get the reward based on the R_HORIZON mode
        '''
        if self.R_HORIZON == 'single_step':
            step_reward = self.get_step_reward()
            self.reward_LOGGER(step_reward, self.sum_steps)
            return step_reward
        elif self.R_HORIZON == 'episode':
            return self.episode_step_rewards()
        elif isinstance(self.R_HORIZON,int) == True:
            return self.multi_step_rewards()
        else:
            raise Exception("R_HORIZON not understood. Please use: 'single-step', 'episode-step' or an integer for multi-step.")

    def get_step_reward(self):
        '''
        Function to get a single-step reward based COST_TYPE and R_TYPE modes
        '''

        
        cost_power, cost_LION, cost_SMS, cost_peak = self.cost_function(
                                                                    interval_sum_power_dem = self.power_dem,
                                                                    sum_LION_nutzung       = self.LION_nutzung,
                                                                    max_peak_diff          = self.max_peak_diff,
                                                                    observed_period        = 1 # step
                                                                    )


        exact_costs = (cost_power + cost_LION + cost_SMS + cost_peak)
        yearly_costs = (cost_power + cost_LION + cost_SMS) * self.auf_jahr_rechnen + cost_peak

        if self.COST_TYPE == 'exact_costs':
            costs = exact_costs
        elif self.COST_TYPE == 'yearly_costs':
            costs = yearly_costs
        elif self.COST_TYPE == 'max_peak_focus':
            costs = (cost_power + cost_LION + cost_SMS + (self.focus_peak_multiplier * cost_peak) )
        else:
            raise Exception("COST_TYPE not understood. Please use: 'exact_costs', 'yearly_costs', 'max_peak_focus'")


        cost_saving = (self.sum_usual_costs / self.steps_per_episode) - yearly_costs
        
        if self.R_TYPE == 'costs_focus':
            step_reward = - costs
        elif self.R_TYPE == 'savings_focus':
            step_reward = cost_saving
        elif self.R_TYPE == 'positive':
            step_reward = 100 - costs  
        else:
            raise Exception("R_TYPE not understood. Please use: 'costs_focus', 'savings_focus'")


        self.cost_LOGGER(exact_costs, yearly_costs, cost_saving)

        return step_reward


    def multi_step_rewards(self):
        '''
        Calculates a multi-step reward based on M_STRATEGY mode
        '''
        step_reward = self.get_step_reward()

        self.memory.append([step_reward, self.sum_steps, self.step_counter_episode])

        
        if self.M_STRATEGY == 'sum_to_terminal':
            self.sum_to_terminal()
        elif self.M_STRATEGY == 'average_to_neighbour':
            self.average_to_neighbour()
        elif self.M_STRATEGY == 'recurrent_to_Terminal':
            self.recurrent_to_Terminal()
        else:
            raise Exception("M_STRATEGY not understood. Please use: 'sum_to_terminal-step', 'average_to_neighbour', 'recurrent_to_Terminal'.")

        return None

    def get_multi_step_reward_list(self,step_counter_episode):
        #print(self.reward_list)
        return self.reward_list[step_counter_episode]

    def sum_to_terminal(self):
        '''
        Calculates a multi-step reward based on the sum-to-terminal strategy from the paper...
        '''
        if len(self.memory) >= self.R_HORIZON and self.done == False:
            sum_step_reward = 0
            for mem_step in self.memory:
                step_reward, sum_steps, step_counter_episode = mem_step
                sum_step_reward += step_reward
            step_reward, sum_steps, step_counter_episode = self.memory[0]
            #print(step_counter_episode)
            self.reward_list[step_counter_episode-1] = sum_step_reward
            #print('step_counter_episode-1',step_counter_episode-1)
            #print(step_reward)
            #print(sum_step_reward)
            self.reward_LOGGER(sum_step_reward, sum_steps)

        if len(self.memory) >= self.R_HORIZON and self.done == True:
            for i in range(len(self.memory)):
                sum_step_reward = 0
                for j in range(len(self.memory)-i):
                    step_reward, sum_steps, step_counter_episode = self.memory[i+j]
                    sum_step_reward += step_reward
                step_reward, sum_steps, step_counter_episode = self.memory[i]
                self.reward_list[step_counter_episode-1] = sum_step_reward
                self.reward_LOGGER(sum_step_reward, sum_steps)

    def average_to_neighbour(self):
        '''
        Calculates a multi-step reward based on the average-to-neighbour strategy from the paper...
        Not implemented yet!
        '''
        raise Exception('average_to_neighbour is not implemented yet')


    def recurrent_to_Terminal(self):
        '''
        Calculates a multi-step reward based on the recurrent-to-Terminal strategy from the paper...
        Not implemented yet!
        '''
        raise Exception('recurrent_to_Terminal is not implemented yet')


    def cost_LOGGER(self, exact_costs, yearly_costs, cost_saving):
        self.sum_exact_costs  += exact_costs
        self.sum_yearly_costs += yearly_costs
        self.sum_cost_saving  += cost_saving

        if any(string in ['exact_costs'] for string in self.logging_list):
            self.LOGGER.log_scalar('step_exact_costs:',  exact_costs,  self.sum_steps, self.done)
        if any(string in ['yearly_costs'] for string in self.logging_list):
            self.LOGGER.log_scalar('step_yearly_costs:', yearly_costs, self.sum_steps, self.done)
        if any(string in ['cost_saving'] for string in self.logging_list):
            self.LOGGER.log_scalar('step_cost_saving:',  cost_saving,  self.sum_steps, self.done)

        if any(string in ['sum_exact_costs'] for string in self.logging_list):
            self.LOGGER.log_scalar('sum_exact_costs',  self.sum_exact_costs,  self.sum_steps, self.done)
        if any(string in ['sum_yearly_costs'] for string in self.logging_list):
            self.LOGGER.log_scalar('sum_yearly_costs', self.sum_yearly_costs, self.sum_steps, self.done)
        if any(string in ['sum_cost_saving'] for string in self.logging_list):
            self.LOGGER.log_scalar('sum_cost_saving',  self.sum_cost_saving,  self.sum_steps, self.done)


    def reward_LOGGER(self, step_reward, sum_steps):
        self.sum_step_reward  += step_reward
        self.step_reward       = step_reward

        if any(string in ['step_reward'] for string in self.logging_list):
            self.LOGGER.log_scalar('step_reward:',  step_reward,  sum_steps, self.done)
            self.LOGGER.log_scalar('sum_reward:',  self.sum_step_reward, sum_steps, self.done)


    def get_log(self):
        return self.sum_cost_saving, self.sum_step_reward, self.step_reward, self.sum_steps







        




















