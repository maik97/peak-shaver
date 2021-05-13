import random
import pandas as pd
import numpy as np

import gym
from gym import spaces

from collections import deque

'''
############   TO-DO  ############

- extra Reward-Modes-List = ['episode-sum','discounted-reward','soll-ist-diff']
- sum-cost-saving-per-episode
- Tabelle rendern lassen
- heursitik verbessern
- temp logs!
'''

class common_env(gym.Env):
    '''GYM environment to simulate the HIPE-dataset

    Args:
        df (dataframe): Pandas dataframe of the normalized HIPE-Dataset
        power_dem_df (array): Array of the local power demand, not normalized 
        input_list (list): List of strings, represents the columns of the dataframe that will be used as inputs. Possible column-names are: 'norm_total_power' and 'max_pred_seq'
        max_SMS_SoC (float): Maximum Capacity of the flying wheel in kwh
        max_LION_SoC (float): Maximum Capacity of the lithium-ion battery in kwh
        PERIODEN_DAUER (integer): Period-lenght of one step in minutes
        ACTION_TYPE (string): Sets the actions as discrete or continuous inputs. Use  'discrete' or 'contin'.
        reward_maker (object): Takes the used reward_maker object
        AGENT_TYPE (string): Inititates the environment to inputs from either heurtistic, the RL-Agents or from ``stable-baselines``. Use 'heuristic', 'normal' or 'standart_gym'.
        max_ziel (float): Defines the maximum possible SECG (in kw), will only be used when AGENT_TYPE is set to 'normal'.
        min_ziel (float): Defines the minimal possible SECG (in kw), will only be used when AGENT_TYPE is set to 'normal'.

    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, reward_maker, df, power_dem_df, input_list,
            max_SMS_SoC          = 12,
            max_LION_SoC         = 54,
            LION_max_entladung   = 50,
            SMS_max_entladung    = 100,
            SMS_entladerate      = 0.72,
            LION_entladerate     = 0.00008,
            PERIODEN_DAUER       = 5,
            ACTION_TYPE          = 'discrete', #'contin'
            OBS_TYPE             = 'discrete', 
            AGENT_TYPE           = 'normal', # 'heuristic'
            discrete_space       = 22,
            # set max_ziel = 1 and min_ziel = 0, if you want to pass zielnetzverbrauch as value
            max_ziel             = 50,
            min_ziel             = 25,
            measure_intervall    = 15,
            val_split            = 0,
            ):
        
        super(common_env, self).__init__()


        # Parameter Datensatz und Logging:
        self.og_df = df
        self.og_power_dem_df = power_dem_df
        print(len(self.og_df))
        print(len(self.og_power_dem_df))
        if val_split != 0:
            self.df = df[:-int(val_split*len(df))]
            self.power_dem_df = power_dem_df[:-int(val_split*len(df))]
        else:
            self.df = df
            self.power_dem_df = power_dem_df
        print('')
        print('Length of train data:',len(self.df))
        print('Length of val data:', len(self.og_df) - len(self.df))
        print(len(self.power_dem_df))
        print('')

        self.input_list            = input_list
        self.AGENT_TYPE            = AGENT_TYPE
        self.netzverbrauch_deque   = deque(maxlen=int(measure_intervall/PERIODEN_DAUER))
        self.measure_intervall     = measure_intervall

        # Parameter Environment:
        self.max_SMS_SoC           = max_SMS_SoC  * 60 # umrechnung von kwh in kwmin
        self.max_LION_SoC          = max_LION_SoC * 60 # umrechnung von kwh in kwmin
        self.LION_max_entladung    = LION_max_entladung #50 #kw
        self.SMS_max_entladung     = SMS_max_entladung #100 #kw
        self.SMS_entladerate       = (SMS_entladerate/(24*60))*PERIODEN_DAUER#*self.max_SMS_SoC
        self.LION_entladerate      = (LION_entladerate/(24*60))*PERIODEN_DAUER#*self.max_LION_SoC
        print(self.SMS_entladerate)
        print(self.LION_entladerate)
        self.PERIODEN_DAUER        = PERIODEN_DAUER    # in min

        # Parameter Agent:
        self.OBS_TYPE              = OBS_TYPE
        self.ACTION_TYPE           = ACTION_TYPE
        self.input_dim             = len(self.input_list) + 2 # jeweils plus SoC für beide Akkus
        self.discrete_space        = discrete_space

        if AGENT_TYPE == 'normal' or AGENT_TYPE == 'standart_gym':
            self.max_ziel          = max_ziel
            self.min_ziel          = min_ziel
        elif AGENT_TYPE == 'heuristic':
            self.max_ziel          = 1
            self.min_ziel          = 0
        else:
            raise Exception("AGENT_TYPE not understood. AGENT_TYPE must be: 'normal', 'heuristic or 'standart_gym''")

        # Init Agent:
        if self.ACTION_TYPE == 'discrete':
            self.action_space      = spaces.Discrete(discrete_space) # A ∈ [0,1]
        elif self.ACTION_TYPE == 'contin':
            self.action_space      = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float16)
        else:
            raise Exception("ACTION_TYPE not understood. ACTION_TYPE must be: 'discrete', 'contin'")

        if self.OBS_TYPE == 'discrete':
            self.num_discrete_obs  = discrete_space
            self.observation_space = spaces.Box(low = 0, high=self.num_discrete_obs, shape=(self.input_dim ,1), dtype=np.float16)
        elif self.OBS_TYPE == 'contin':
            self.num_discrete_obs  = 1
            self.observation_space = spaces.Box(low = 0, high=1, shape=(self.input_dim ,1), dtype=np.float16)
        else:
            raise Exception("OBS_TYPE not understood. OBS_TYPE must be: 'discrete', 'contin'")

        # Init Rewards
        self.reward_maker          = reward_maker
        self.reward_range          = self.reward_maker.get_reward_range() #als (MIN_REWARD, MAX_REWARD)
        
        self.deactivate_SMS        = self.reward_maker.__dict__['deactivate_SMS']
        self.deactivate_LION       = self.reward_maker.__dict__['deactivate_LION']
        
        self.LOGGER                = self.reward_maker.__dict__['LOGGER']
        self.NAME                  = self.LOGGER.__dict__['NAME']
        self.D_PATH                = self.LOGGER.__dict__['D_PATH']

        # Init Step-Counter:
        self.sum_steps             = 0
        self.step_counter_day      = 0
        self.step_counter_week     = 0
        self.step_counter_episode  = 0

        # Init other Counter
        self.day_counter           = 0
        self.week_counter          = 0
        self.episode_counter       = 0
        
        # Init Max-Peaks:
        self.day_max_peak          = 0
        self.week_max_peak         = 0
        self.episode_max_peak      = 0

        self.steps_per_week        = 10080 / self.PERIODEN_DAUER # min_pro_woche/Periodendauer
        self.steps_per_day         = 1440  / self.PERIODEN_DAUER # min_pro_trag/Periodendauer

        self.calc_parameter_dataset()

        #Stromkosten: 28554.73
        #Peak-Kosten: 8539.1
        #Gesamtkosten: 37093.83

    def calc_parameter_dataset(self):
        self.power_dem_arr         = self.power_dem_df.to_numpy().copy()
        self.rolling_power_dem     = self.power_dem_df.rolling(int(self.measure_intervall/self.PERIODEN_DAUER)).mean().fillna(0).to_numpy().copy()
        print(len(self.power_dem_arr))
        self.max_power_dem         = np.max(self.power_dem_arr)
        self.max_rolling_power_dem = np.max(self.rolling_power_dem)
        self.mean_power_dem        = np.mean(self.power_dem_arr)
        self.sum_power_dem         = np.sum(self.df['norm_total_power']*self.max_power_dem)
        self.steps_per_episode     = len(self.df['norm_total_power'])
        print('\nMaximum Power-Demand:', round(self.max_power_dem,2))
        print('Maximum Power-Demand (15min):', round(self.max_rolling_power_dem,2))
        print('Mean Power-Demand:',      round(self.mean_power_dem,2))
        print('Sum of Power-Demand',     round(self.sum_power_dem,2))
        print('Steps pro Episode:',      round(self.steps_per_episode,2))
        print(self.sum_power_dem-np.sum(self.power_dem_arr))
        print(np.max(self.df['norm_total_power']*self.max_power_dem))
        #print(self.power_dem_df-self.df['norm_total_power']*self.max_power_dem)
        print(self.power_dem_df)
        print(self.df['norm_total_power']*self.max_power_dem)
        self.reward_maker.pass_env(self.PERIODEN_DAUER, self.steps_per_episode, self.max_power_dem, self.mean_power_dem, self.sum_power_dem, self.max_rolling_power_dem)


    def use_only_val_data(self):
        print('Using only val-data')
        self.df = self.og_df[int(val_split*len(self.og_df)):]
        self.power_dem_df = self.og_power_dem_df[int(val_split*len(self.og_df)):]
        self.calc_parameter_dataset()



    def use_all_data(self):
        print('Using training and val-data')
        self.df = self.og_df
        self.power_dem_df = self.og_power_dem_df
        self.calc_parameter_dataset()


    def check_max_peak(self, past_max_peak):
        '''Class-Function that checks if there is a new maximum peak for a defined time-period.

        Args:
            past_max_peak (float): The previous maximum peak.
        Returns
            new_max_peak (float): Returns either the previous maximum peak or a new maximum peak
        '''
        möglicher_peak = np.mean(self.netzverbrauch_deque)

        if möglicher_peak > past_max_peak:
            #print(möglicher_peak)
            #print(self.netzverbrauch_deque)
            return möglicher_peak
        else:
            return past_max_peak

    def check_all_max_peak(self):
        '''Class-Function that will check at each step if there is a new maximum peak over a defined period of time.
        Checks for a day, a week and for the whole episode.

        Returns:
            day_max_peak, week_max_peak, episode_max_peak (tuple):

            day_max_peak (float): Maximum daily peak that was recorded so far.

            week_max_peak (float): Maximum weekly peak that was recorded so far.

            episode_max_peak (float): Maximum peak of the episode that was recorded so far.
        '''
        self.netzverbrauch_deque.append(self.netzverbrauch)
        self.day_max_peak     = self.check_max_peak(self.day_max_peak)
        self.week_max_peak    = self.check_max_peak(self.week_max_peak)
        self.episode_max_peak = self.check_max_peak(self.episode_max_peak)
        return self.day_max_peak, self.week_max_peak, self.episode_max_peak

    def step_counter(self):
        '''Class-Function that counts the steps in different timeframes.
        (sum of steps over all episodes, current step of the dataset, daily, weekly, step of the episode)
        It also passes the maximum peaks from the different recording timefrime to a logging function, which can be traced in Tensorboard.
        '''
        self.sum_steps            += 1
        self.current_step         += 1
        self.step_counter_day     += 1
        self.step_counter_week    += 1
        self.step_counter_episode += 1

        # Falls neuer Step nach dem Ende der Datenpunkte, gehe wieder zum Anfang:
        if self.current_step > len(self.df['norm_total_power']) - 1:
            self.current_step = 0

        # Überprüfe, ob Ende der Episode erreicht wurde:
        if self.step_counter_episode >= self.steps_per_episode:
            self.LOGGER.log_scalar('4.3 max-peak-epoch', self.episode_max_peak, self.episode_counter, True)
            self.episode_counter     += 1
            self.step_counter_episode = 0
            self.episode_max_peak     = 0
            done = True
        else:
            done = False

        # Neue Woche
        if self.step_counter_week >= self.steps_per_week or done == True:
            self.LOGGER.log_scalar('4.2 max-peak-week', self.week_max_peak, self.week_counter, done)
            self.week_counter     += 1
            self.step_counter_week = 0
            self.week_max_peak     = 0

        # Neuer Tag
        if self.step_counter_day >= self.steps_per_day or done == True:
            self.LOGGER.log_scalar('4.1 max-peak-day', self.day_max_peak, self.day_counter, done)
            self.day_counter     += 1
            self.step_counter_day = 0
            self.day_max_peak     = 0
        
        return done

    
    def next_observation(self):
        '''
        Makes an observation from the dataset. For discrete observation ACTION_TYPE must be set to 'discrete'.
        
        Returns:
            obs (integer or array):

            obs (integer): returns discrete observations, when ACTION_TYPE is set to 'discrete'

            obs (array): returns an continious observations, when ACTION_TYPE is set to 'contin'
        '''
        obs = np.array([
            [(self.SMS_SoC  / self.max_SMS_SoC)  * self.num_discrete_obs], # SMS-Ladestand
            [(self.LION_SoC / self.max_LION_SoC) * self.num_discrete_obs], # LION-Ladestand
            ])
        for input_column in self.input_list:
            obs = np.append(obs,[self.df[input_column][self.current_step] * self.num_discrete_obs])

        if self.num_discrete_obs > 1:
            obs = obs.astype(int)-1
        else:
            obs = obs.reshape(self.input_dim,1)
        
        #print(obs)
        return obs

    
    def get_discrete_outputs(self, action):
        '''
        Determines the SECG and if SMS priority will be used from an discrete action space
        
        Returns:
            SECG, SMS_priority (tuple):

            SECG (float): the energy consumption (in kw) that should come from the grit

            SMS_priority (bool): determines if the flywheel should be charged first

        '''
        # Einteilung Aktionen in SMS Laden und Nicht-Laden
        if self.deactivate_SMS == False and self.deactivate_LION == False:
            act_multplier = 1/(int(self.discrete_space/2) - 1) # should be 0.1 for discrete_space=22
            if action <= int(self.discrete_space/2) - 1:
                SMS_priority = True
            else:
                SMS_priority = False
                action -= int(self.discrete_space/2)
        else:
            act_multplier = 1/(int(self.discrete_space - 1))
            if self.deactivate_SMS == False:
                SMS_priority = True
            else:
                SMS_priority = False

        # Berechnung Ziel-Netzverbrauch
        return (self.max_ziel-self.min_ziel) * action * act_multplier + self.min_ziel, SMS_priority

    
    def get_contin_outputs(self, action):
        '''
        Determines the SECG and if SMS priority will be used from an continious action space
        
        Returns:
            SECG, SMS_priority (tuple):

            SECG (float): the energy consumption (in kw) that should come from the grit

            SMS_priority (bool): determines if the flywheel should be charged first

        '''
        if self.deactivate_SMS == False and self.deactivate_LION == False:
            if action[1] < 0.5:
                SMS_priority = True
            else:
                SMS_priority = False
        else:
            if self.deactivate_SMS == False:
                SMS_priority = True
            else:
                SMS_priority = False
        return (self.max_ziel-self.min_ziel) * action[0] + self.min_ziel, SMS_priority
    
    
    def take_action(self, action):
        '''
        Agent takes action based on ACTION_TYPE and updates the simulation of the batteries.

        Args:
            action (integer or array): integer when action space is discrete and array when action space is continious
        '''
        
        # Diskrete oder Stetige Outputs?
        if   self.ACTION_TYPE == 'discrete':
            self.ziel_netzverbrauch, SMS_priority = self.get_discrete_outputs(action)
        elif self.ACTION_TYPE == 'contin':
            self.ziel_netzverbrauch, SMS_priority = self.get_contin_outputs(action)

        # Aktueller Energie-Bedarf der Maschinen:
        power_dem = self.df['norm_total_power'][self.current_step] * self.max_power_dem

        # LADEN:
        if  self.ziel_netzverbrauch > power_dem:
            if SMS_priority == True:
                lademenge_SMS = self.SMS_laden(self.ziel_netzverbrauch - power_dem)
            else:
                lademenge_SMS = 0
            if self.deactivate_LION == False:
                lademenge_LION = self.LION_laden(self.ziel_netzverbrauch - lademenge_SMS - power_dem)
            else:
                lademenge_LION = self.LION_laden(0)
            self.netzverbrauch = power_dem + lademenge_SMS + lademenge_LION
        
        # Entladen:
        elif self.ziel_netzverbrauch < power_dem:  
            entlademenge_SMS = self.SMS_entladen(power_dem - self.ziel_netzverbrauch)
            entlademenge_LION = self.LION_entladen(power_dem - entlademenge_SMS - self.ziel_netzverbrauch)
            self.netzverbrauch = power_dem - entlademenge_SMS - entlademenge_LION
           
        # Falls weder geladen noch entladen wird:
        else:
            self.netzverbrauch = self.ziel_netzverbrauch
            self.LION_nutzung = 0
            # Berechne Verlust SMS pro Step:
        
        self.p_loss = self.SMS_verlust()
        lion_verlust = self.LION_verlust()


    def step(self, action):
        '''
        Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for calling reset() to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            obs, reward, done, info (tuple):
            
            obs (float): agent's observation of the current environment
            
            reward (float): amount of reward returned after previous action
            
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results

            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
            
            step_counter_episode (integer): counted step of the episode, only returned when ACTION_TYPE is set to 'discrete'
            
            episode_max_peak (float): maximum peak of the episode so far, only returned when ACTION_TYPE is set to 'heuristic'
        '''
        
        # Führe einen Step in der Environment aus:
        self.take_action(action)

        # Checke ob neuer Peak existiert und behalte die zusätzlich Daten lokal(für rewards):
        day_max_peak, week_max_peak, episode_max_peak = self.check_all_max_peak()

        # Setze neuen Step (hier könnten globale max_peaks auf null gesetzt werden):
        done   = self.step_counter()

        # Nächste Inputs:
        obs    = self.next_observation()

        # Gebe Varibeln an reward_maker
        self.reward_maker.pass_state(done, day_max_peak, week_max_peak, episode_max_peak,
            power_dem            = self.netzverbrauch,
            LION_nutzung         = self.LION_nutzung,
            sum_steps            = self.sum_steps,
            step_counter_episode = self.step_counter_episode)
        
        # Hole Reward: (eventuell gleich None, falls z.B. Multi-Step-Reward)
        reward = self.reward_maker.get_reward()

        # Logging pro Step
        self.step_LOGGER(action, done)

        if self.AGENT_TYPE == 'normal':
            if self.ACTION_TYPE == 'discrete':
                return obs, reward, done, self.step_counter_episode, {}
            elif self.ACTION_TYPE == 'contin':
                return obs, reward, done, {}
        elif self.AGENT_TYPE == 'heuristic':
            return obs, reward, done, episode_max_peak, {}
        elif self.AGENT_TYPE == 'standart_gym':
            return obs, reward, done, {}


    def get_multi_step_reward(self, step_counter_episode):
        ''' returns a list of multi step rewards from the reward_maker

        Args:
            step_counter_episode (integer): counted step of the episode
        '''
        return self.reward_maker.get_multi_step_reward_list(step_counter_episode)


    def reset(self):
        '''Resets the environment to an initial state and returns an initial observation.
        Randomizes both state of charges and the current step of the dataset to begin from.

        Returns 
            obs (object): the initial observation
        '''
        # Wähle zufälligen Step als Start aus:
        self.current_step = random.randint(0, self.steps_per_episode - 1)
        
        # Zufällige Akku-SoC zum Start:
        self.SMS_SoC  = random.uniform(0.5*self.max_SMS_SoC,  self.max_SMS_SoC)
        self.LION_SoC = random.uniform(0.5*self.max_LION_SoC, self.max_LION_SoC)

        # Reset Reward-Maker
        self.reward_maker.reset()

        return self.next_observation()


    def set_soc_and_current_state(self,SoC_full=True,SoC=None):
        '''
        Used by heuristic agents, sets the current step of the dataset to 0 and can set the battery charges to full.

        Args:
            SoC_full (bool): Sets the charge of both batteries to full when set to True
        '''
        self.current_step = 0
        if SoC_full == True:
            self.SMS_SoC  = self.max_SMS_SoC
            self.LION_SoC = self.max_LION_SoC
        else:
            self.SMS_SoC  = random.uniform(0.5*self.max_SMS_SoC,  self.max_SMS_SoC)
            self.LION_SoC = random.uniform(0.5*self.max_LION_SoC, self.max_LION_SoC)

        if SoC != None:
            self.LION_SoC = min(SoC*self.PERIODEN_DAUER,self.max_LION_SoC)
            self.SMS_SoC = min(SoC*self.PERIODEN_DAUER-self.LION_SoC,self.max_SMS_SoC)

        print('Initial Charge SMS',self.SMS_SoC)
        print('Initial Charge LION',self.LION_SoC)

        # Reset Reward-Maker
        self.reward_maker.reset()

        return self.next_observation()


    def SMS_laden(self, lade_menge):
        '''
        Used to simulate charging of the flying wheel

        Args:
            lade_menge (float): amount to be charged in kw

        Returns:
            energieverbrauch (float): real amount that could be charged in kw
        '''
        # Berechne Akku-Ladung in kwmin
        lade_menge *= self.PERIODEN_DAUER

        # wenn neuer ladastand nicht das maximum überschreiten würde:
        if self.SMS_SoC + lade_menge <= self.max_SMS_SoC:
            self.SMS_SoC += lade_menge
            energievebrauch = lade_menge / self.PERIODEN_DAUER

        # wenn neuer ladestand zu groß wäre:
        else:
            energievebrauch = (self.max_SMS_SoC - self.SMS_SoC) / self.PERIODEN_DAUER
            self.SMS_SoC = self.max_SMS_SoC

        return energievebrauch

    
    def SMS_entladen(self, akku_entladung):
        '''
        Used to simulate discharging of the flying wheel

        Args:
            lade_menge (float): amount to be decharged in kw

        Returns:
            energieverbrauch (float): real amount that could be discharged in kw
        '''
        akku_entladung *= self.PERIODEN_DAUER # kw in kwt
        #print(akku_entladung)
        # wenn ladestand groß genug und erforderliche Leistung nicht über 250
        if self.SMS_SoC > akku_entladung and akku_entladung < self.SMS_max_entladung*self.PERIODEN_DAUER: #max kw, bzw 250kwt
            self.SMS_SoC -= akku_entladung
            tatsächliche_akku_entladung = akku_entladung / self.PERIODEN_DAUER
        
        #wenn ladestand groß genug, aber geforderete leistung nicht innerhalb der zeit erbracht werden kann:
        elif self.SMS_SoC > akku_entladung and akku_entladung > self.SMS_max_entladung*self.PERIODEN_DAUER:
            self.SMS_SoC -= self.SMS_max_entladung
            tatsächliche_akku_entladung = self.SMS_max_entladung / self.PERIODEN_DAUER
        
        # wenn ladestand nicht ausreichst:
        elif self.SMS_SoC < akku_entladung:
            tatsächliche_akku_entladung = self.SMS_SoC / self.PERIODEN_DAUER
            self.SMS_SoC = 0
        
        else: # akku_entladung == 0:
            tatsächliche_akku_entladung = 0

        return tatsächliche_akku_entladung
        '''
        # Berechne Akku Entladung in kwmin
        akku_entladung *= self.PERIODEN_DAUER # kw in kwt

        # wenn ladestand groß genug und erforderliche Leistung nicht über 250
        if self.SMS_SoC > akku_entladung:
            self.SMS_SoC -= akku_entladung
            tatsächliche_akku_entladung = akku_entladung / self.PERIODEN_DAUER
        
        # wenn ladestand nicht ausreichst:
        elif self.SMS_SoC < akku_entladung:
            tatsächliche_akku_entladung = self.SMS_SoC / self.PERIODEN_DAUER
            self.SMS_SoC = 0

        return tatsächliche_akku_entladung
        '''
    
    def LION_laden(self, lade_menge):
        '''
        Used to simulate charging of the lithium-ion battery

        Args:
            lade_menge (float): amount to be charged in kw

        Returns:
            energieverbrauch (float): real amount that could be charged in kw
        '''
        # Berechne Akku-Ladung in kwmin
        lade_menge *= self.PERIODEN_DAUER

        # wenn neuer ladastand nicht das maximum überschreiten würde:
        if self.LION_SoC + lade_menge <= self.max_LION_SoC:
            self.LION_SoC += lade_menge
            energievebrauch = lade_menge / self.PERIODEN_DAUER

        # wenn neuer ladestand zu groß wäre:
        else:
            energievebrauch = (self.max_LION_SoC - self.LION_SoC) / self.PERIODEN_DAUER
            self.LION_SoC = self.max_LION_SoC

        self.LION_nutzung = energievebrauch / self.max_LION_SoC * 0.5

        return energievebrauch

    
    def LION_entladen(self, akku_entladung):
        '''
        Used to simulate discharging of the lithium-ion battery

        Args:
            lade_menge (float): amount to be decharged in kw

        Returns:
            energieverbrauch (float): real amount that could be discharged in kw
        '''
        # Berechne Akku Entladung in kwmin
        akku_entladung *= self.PERIODEN_DAUER # kw in kwt
        #print(akku_entladung)
        # wenn ladestand groß genug und erforderliche Leistung nicht über 250
        if self.LION_SoC > akku_entladung and akku_entladung < self.LION_max_entladung*self.PERIODEN_DAUER: #max kw, bzw 250kwt
            self.LION_SoC -= akku_entladung
            tatsächliche_akku_entladung = akku_entladung / self.PERIODEN_DAUER
        
        #wenn ladestand groß genug, aber geforderete leistung nicht innerhalb der zeit erbracht werden kann:
        elif self.LION_SoC > akku_entladung and akku_entladung > self.LION_max_entladung*self.PERIODEN_DAUER:
            self.LION_SoC -= self.LION_max_entladung
            tatsächliche_akku_entladung = self.LION_max_entladung / self.PERIODEN_DAUER
        
        # wenn ladestand nicht ausreichst:
        elif self.LION_SoC < akku_entladung:
            tatsächliche_akku_entladung = self.LION_SoC / self.PERIODEN_DAUER
            self.LION_SoC = 0
        
        else: # akku_entladung == 0:
            tatsächliche_akku_entladung = 0

        # Berechen Abnutzung (in Zyklen):
        self.LION_nutzung = tatsächliche_akku_entladung / self.max_LION_SoC * 0.5

        return tatsächliche_akku_entladung

    
    def SMS_verlust(self):
        ''' Calculates the loss of the flying wheel

        Returns:
            SoC_loss (float): Charge loss in kw
        '''
        '''
        if self.SMS_SoC >= self.max_SMS_SoC*self.SMS_entladerate:
            self.SMS_SoC -= self.max_SMS_SoC*self.SMS_entladerate 
            return self.max_SMS_SoC*self.SMS_entladerate 
        else:
            self.SMS_SoC = 0
            return 0
        '''
        self.SMS_SoC -= self.SMS_SoC*self.SMS_entladerate 
        return self.SMS_SoC*self.SMS_entladerate 


    def LION_verlust(self):
        ''' Calculates the loss of the flying wheel

        Returns:
            SoC_loss (float): Charge loss in kw
        '''
        '''
        if self.LION_SoC >= self.max_LION_SoC*self.LION_entladerate:
            self.LION_SoC -= self.max_LION_SoC*self.LION_entladerate 
            return self.max_LION_SoC*self.LION_entladerate 
        else:
            self.LION_SoC = 0
            return 0
        '''
        self.LION_SoC -= self.LION_SoC*self.LION_entladerate 
        return self.LION_SoC*self.LION_entladerate 
   
    
    def step_LOGGER(self, action, done):
        '''
        Logs data for tensorboard

        Args:
            action (integer or array): integer when action space is discrete and array when action space is continious
        '''
        if self.ACTION_TYPE == 'discrete':
            self.LOGGER.log_scalar('1.1 discrete-action', action, self.sum_steps, done)
        elif self.ACTION_TYPE == 'contin':
            self.LOGGER.log_scalar('1.2 continuous-action', action[0], self.sum_steps, done)
            self.LOGGER.log_scalar('1.3 Prio-SMS', action[1], self.sum_steps, done)

        self.LOGGER.log_scalar('2.1 real-GEC', self.netzverbrauch, self.sum_steps, done)
        self.LOGGER.log_scalar('2.2 target-GEC', self.ziel_netzverbrauch, self.sum_steps, done)
        self.LOGGER.log_scalar('3.1 SoC-SMS', self.SMS_SoC/60, self.sum_steps, done)
        self.LOGGER.log_scalar('3.2 SoC-LION', self.LION_SoC/60, self.sum_steps, done)

        '''
        try:
            sum_cost_saving, sum_step_reward, step_reward, sum_steps = self.reward_maker.get_log()
            self.LOGGER.log_scalar('Reward-Maker - step_reward:',  step_reward,  sum_steps)
            self.LOGGER.log_scalar('Sum_Episode - step_reward:',  sum_step_reward, sum_steps)
            self.LOGGER.log_scalar('Sum_Episode - cost_saving:',  sum_cost_saving, sum_steps)
        except:
            bla = 1
        '''
        if done == True:
            sum_cost_saving, sum_step_reward, step_reward, sum_steps = self.reward_maker.get_log()
            self.LOGGER.log_scalar('5.1 sum_savings_epoch',  sum_cost_saving, self.episode_counter, done)
            self.LOGGER.log_scalar('5.2 sum_reward_epoch',  sum_step_reward, self.episode_counter, done)



        #self.LOGGER.log_scalar('Epsilon:', epsilon, self.sum_steps)
        #self.LOGGER.log_scalar('Step: Differenz Netzverbrauch', self.diff_netzverbrauch, self.sum_steps)
        
        #self.LOGGER.log_scalar('Reward (Step):', reward, self.sum_steps)
        #self.LOGGER.log_scalar('Reward (Episode):', self.sum_episode_reward, self.sum_steps)
        #self.LOGGER.log_scalar('Reward (Day):', self.sum_day_reward, self.sum_steps)

    
    #def render(self, mode='human', close=False):















