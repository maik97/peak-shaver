'''
Example of an RL-Agent that uses the basic Q-Table.
'''
import numpy as np
from datetime import datetime
from collections import deque

from main.common_func import max_seq, mean_seq, training, testing
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.logger import Logger
from main.reward_maker import reward_maker
from main.common_env import common_env

# Import the Q-Table agent: 
from main.agent_q_table import Q_Learner

# Naming the agent and setting up the directory path:
now    = datetime.now()
NAME   = 'Q_Table'+now.strftime("_%d-%m-%Y_%H-%M-%S")
D_PATH = '_small_d/'

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
power_dem_df = main_dataset.load_total_power()

# Load the LSTM input dataset:
lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

# Making predictions:
normal_predictions = wahrsager(lstm_dataset, power_dem_df, TYPE='NORMAL').pred()[:-12]
seq_predictions    = wahrsager(lstm_dataset, power_dem_df, TYPE='SEQ', num_outputs=12).pred()

# Adding the predictions to the dataset:
df            = df[24:-12]
df['normal']  = normal_predictions
df['seq-max'] = max_seq(seq_predictions)


# Number of warm-up steps:
num_warmup_steps = 100
# Train every x number of steps:
update_num       = 1000
# Number of epochs and steps:
epochs           = 1000
epochs_len       = len(df)
max_steps        = epochs*epochs_len
# Horizon for Multi-Step-Rewards and/or LSTM-Implementation:
horizon = 0

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
    Leistungspreis          = 102)

# Setup common_env
env = common_env(
    reward_maker   = r_maker,
    df             = df,
    power_dem_df   = power_dem_df,
    input_list     = ['norm_total_power','normal','seq-max'],
    max_SMS_SoC    = 12,
    max_LION_SoC   = 54,
    PERIODEN_DAUER = 5,
    ACTION_TYPE    = 'discrete',
    OBS_TYPE       = 'discrete',
    discrete_space = 22)

# Setup agent:
agent = Q_Learner(
    env            = env,
    memory_len     = update_num,
    # Training-Parameter:
    gamma          = 0.85,
    epsilon        = 0.8,
    epsilon_min    = 0.1,
    epsilon_decay  = 0.999996,
    lr             = 0.5,
    tau            = 0.125,
    # jede Dimension jeweils ∈ [0,0.05,...,1]
    Q_table        = np.zeros((22,22,22,22,22,22)))


training(agent, epochs, update_num, num_warmup_steps)
testing(agent)

