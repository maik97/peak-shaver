import random
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot  as plt

from collections import deque

import logger

class reward_maker():

    def __init__(
            self,
            COST_TYPE               = 'exact_costs', # 'yearly_costs', 'max_peak_focus'
            R_TYPE                  = 'costs_focus', # 'positive','savings_focus' -> für 'saving_focus' müsste 'yearly_costs' benutzt werden
            R_HORIZON               = 'single_step', # 'episode', '...', integer for multi-step
            M_STRATEGY              = None,          # 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
            cost_per_kwh            = 0.07,  # in €
            LION_Anschaffungs_Preis = 32000, # in €
            LION_max_Ladezyklen     = 1000,
            SMS_Anschaffungs_Preis  = 10000, # in €
            SMS_max_Nutzungsjahre   = 20,    # in Jahre
            Leistungspreis          = 90,    # in €
            focus_peak_multiplier   = 4,     # multiplier for max_peak costs
            logging_list            = ['exact_costs','costs_focus','single_step','sum_exact_costs','sum_costs_focus','sum_single_step']
            ):

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

        if self.R_TYPE == 'savings_focus' and self.COST_TYPE != 'yearly_costs':
            print('COST_TYPE is', self.COST_TYPE, "but should be 'yearly_costs', when R_TYPE is 'savings_focus'")
            print("Changing: COST_TYPE is now 'yearly_costs' for accurate results")
            self.COST_TYPE = 'yearly_costs'


    def pass_env(self, df, NAME, DATENSATZ_PATH, PERIODEN_DAUER, steps_per_episode, max_power_dem, mean_power_dem, sum_power_dem):
        
        # Init Parameter aus Env:
        self.df                = df
        self.NAME              = NAME
        self.DATENSATZ_PATH    = DATENSATZ_PATH
        self.PERIODEN_DAUER    = PERIODEN_DAUER
        self.steps_per_episode = steps_per_episode
        self.max_power_dem     = max_power_dem
        self.mean_power_dem    = mean_power_dem
        self.sum_power_dem     = sum_power_dem

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
        print('Stromkosten:',  cost_power)
        print('Peak-Kosten:',  cost_peak)
        print('Gesamtkosten:', self.sum_usual_costs,'\n')

        # Init Logging
        self.LOGGER            = logger.Logger(DATENSATZ_PATH+'LOGS/reward_logging/rewards_'+NAME)

    def get_sum_of_usual_costs(self):
        cost_power = self.sum_power_dem * self.PERIODEN_DAUER * (self.cost_per_kwh/60) * self.auf_jahr_rechnen
        #c * steps   #kw * steps           #min pro step         #c / (kw * min)

        cost_peak  = self.max_power_dem * self.Leistungspreis 
        #c * steps   #kw                  # c / kw * jahr
        return cost_power, cost_peak


    def get_reward_range(self):
        MIN_REWARD = -100
        MAX_REWARD = 100
        return (MIN_REWARD, MAX_REWARD)


    def pass_state(
                self,
                new_obs,
                action,
                done,

                day_max_peak,
                week_max_peak,
                episode_max_peak,

                power_dem,
                ziel_netzverbrauch,

                SMS_loss,
                LION_nutzung,
                SMS_SoC,
                LION_SoC,

                day_counter,
                week_counter,
                episode_counter,

                sum_steps,
                step_counter_episode,
                ):

        self.done = done

        if self.episode_max_peak < episode_max_peak:
            self.max_peak_diff    = episode_max_peak - self.episode_max_peak
            self.episode_max_peak = episode_max_peak        

        self.power_dem        = power_dem
        self.LION_nutzung     = LION_nutzung

        self.sum_steps = sum_steps
        self.step_counter_episode = step_counter_episode

    def _reset(self):
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
                print('Failed to initialized memory for Multi-Step-Horizon: R_HORIZON must be an integer')
                print('R_HORIZON was set to:', self.R_HORIZON)
                print("Alternatively use 'single_step' or 'episode', instead of an integer")
                exit()
            


    def cost_function(self, sum_power_dem, sum_LION_nutzung, max_peak, observed_period):
        '''
        PERIODEN_DAUER
        cost_per_kwh
        LION_Anschaffungs_Preis
        LION_max_Ladezyklen
        SMS_Anschaffungs_Preis
        SMS_max_Nutzungsjahre
        Leistungspreis
        '''
        cost_power = sum_power_dem * self.PERIODEN_DAUER * (self.cost_per_kwh/60)
        #c * steps   #kw * steps     #min pro step            #c / (kw * min)

        cost_LION  = sum_LION_nutzung * (self.LION_Anschaffungs_Preis / self.LION_max_Ladezyklen)       # -> Abschreibung über Nutzung
        #c * steps   #nutzung * steps   #c                              #max_nutzung

        cost_SMS   = (self.SMS_Anschaffungs_Preis / self.SMS_max_Nutzungsjahre) * observed_period       # -> Abschreibung über Zeit
        #c * steps   #c                             #max_nutzung in jahre         #steps / (steps*jahr)

        cost_peak  = max_peak * self.Leistungspreis * observed_period
        #c * steps   #kw        # c / kw * jahr       #steps / (steps*jahr)

        return cost_power, cost_LION, cost_SMS, cost_peak  

    def get_reward(self):
        if self.R_HORIZON == 'single_step':
            step_reward = self.get_step_reward()
            self.reward_LOGGER(step_reward, self.sum_steps)
            return step_reward
        elif self.R_HORIZON == 'episode':
            return self.episode_step_rewards()
        elif isinstance(self.R_HORIZON,int) == True:
            return self.multi_step_rewards()
        else:
            print("Error: R_HORIZON not understood. Please use: 'single-step', 'episode-step' or an integer for multi-step.")
            exit()

    def get_step_reward(self):
        '''
        max_peak_diff = max_peak_neu - max_peak_alt
        steps_per_year = (525 600 Minuten pro Jahr / 5 min) = 105120 steps
        step_anteil_jahr = (5 Minuten / 525 600 Minuten pro Jahr) = 0.00000951293
        episode_anteil_jahr = (self.steps_per_episode/ 105120 steps)
        auf_jahr_rechnen = (105120 steps/self.steps_per_episode)
        '''

        
        cost_power, cost_LION, cost_SMS, cost_peak = self.cost_function(
                                                                    sum_power_dem    = self.power_dem,
                                                                    sum_LION_nutzung = self.LION_nutzung,
                                                                    max_peak         = self.max_peak_diff,
                                                                    observed_period  = 1 / self.steps_per_episode
                                                                    )


        exact_costs = (cost_power + cost_LION + cost_SMS + cost_peak)
        yearly_costs = exact_costs * self.auf_jahr_rechnen

        if self.COST_TYPE == 'exact_costs':
            costs = exact_costs
        elif self.COST_TYPE == 'yearly_costs':
            costs = yearly_costs
        elif self.COST_TYPE == 'max_peak_focus':
            costs = (cost_power + cost_LION + cost_SMS + (self.focus_peak_multiplier * cost_peak) ) * self.auf_jahr_rechnen
        else:
            print("Error: COST_TYPE not understood. Please use: 'exact_costs', 'yearly_costs', 'max_peak_focus'")
            exit()

        cost_saving = (self.sum_usual_costs / self.steps_per_episode) - yearly_costs
        
        if self.R_TYPE == 'costs_focus':
            step_reward = - costs
        elif self.R_TYPE == 'savings_focus':
            step_reward = cost_saving
        elif self.R_TYPE == 'positive':
            step_reward = 100 - costs  
        else:
            print("Error: R_TYPE not understood. Please use: 'costs_focus', 'savings_focus'")
            exit()

        self.cost_LOGGER(exact_costs, yearly_costs, cost_saving)

        return step_reward


    def multi_step_rewards(self):
        step_reward = self.get_step_reward()

        self.memory.append([step_reward, self.sum_steps, self.step_counter_episode])

        
        if self.M_STRATEGY == 'sum_to_terminal':
            self.sum_to_terminal()
        elif self.M_STRATEGY == 'average_to_neighbour':
            self.average_to_neighbour()
        elif self.M_STRATEGY == 'recurrent_to_Terminal':
            self.recurrent_to_Terminal()
        else:
            print("Error: M_STRATEGY not understood. Please use: 'sum_to_terminal-step', 'average_to_neighbour', 'recurrent_to_Terminal'.")
            exit()
        

        return None

    def get_multi_step_reward_list(self,step_counter_episode):
        #print(self.reward_list)
        return self.reward_list[step_counter_episode]

    def sum_to_terminal(self):
        '''
        reward_list
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



    def cost_LOGGER(self, exact_costs, yearly_costs, cost_saving):
        self.sum_exact_costs  += exact_costs
        self.sum_yearly_costs += yearly_costs
        self.sum_cost_saving  += cost_saving

        #if 'exact_costs' == any(self.logging_list):
        #self.LOGGER.log_scalar('Reward-Maker - exact_costs:',  exact_costs,  self.sum_steps)
        #if 'yearly_costs' == any(self.logging_list):
        #self.LOGGER.log_scalar('Reward-Maker - yearly_costs:', yearly_costs, self.sum_steps)
        #if 'cost_saving' == any(self.logging_list):
        #self.LOGGER.log_scalar('Reward-Maker - cost_saving:',  cost_saving,  self.sum_steps)

        #if 'sum_exact_costs' == any(self.logging_list):
        #self.LOGGER.log_scalar('Sum_Episode - exact_costs:',  self.sum_exact_costs,  self.sum_steps)
        #if 'sum_yearly_costs' == any(self.logging_list):
        #self.LOGGER.log_scalar('Sum_Episode - yearly_costs:', self.sum_yearly_costs, self.sum_steps)
        #if 'sum_cost_saving' == any(self.logging_list):
        
        ###self.LOGGER.log_scalar('Sum_Episode - cost_saving:',  self.sum_cost_saving,  self.sum_steps)


    def reward_LOGGER(self, step_reward, sum_steps):
        self.sum_step_reward  += step_reward
        self.step_reward       = step_reward
        #self.LOGGER.log_scalar('Reward-Maker - step_reward:',  step_reward,  sum_steps)
        #self.LOGGER.log_scalar('Sum_Episode - step_reward:',  self.sum_step_reward, sum_steps)

    def get_sum_reward(self):
        return self.sum_step_reward

    def get_log(self):
        return self.sum_cost_saving, self.sum_step_reward, self.step_reward, self.sum_steps





def main():
    import wahrsager
    import schaffer
    print('Teste Initialisierungen')
    test_reward_maker = reward_maker(
                            COST_TYPE               = 'exact_costs', # 'yearly_costs', 'max_peak_focus'
                            R_TYPE                  = 'costs_focus', # 'savings_focus'
                            R_HORIZON               = 'single_step', # 'multi_step', 'episode', '...'
                            cost_per_kwh            = 0.07,  # in €
                            LION_Anschaffungs_Preis = 32000, # in €
                            LION_max_Ladezyklen     = 1000,
                            SMS_Anschaffungs_Preis  = 10000, # in €
                            SMS_max_Nutzungsjahre   = 15,    # in Jahre
                            Leistungspreis          = 90,    # in €
                            focus_peak_multiplier   = 4      # multiplier for max_peak costs
                            )






        




















