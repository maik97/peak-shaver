'''
Example of an RL-Agent that uses Dualing Deep Q-Networks.
'''
from datetime import datetime
from collections import deque

from main.common_func import max_seq, mean_seq
from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.logger import Logger
from main.reward_maker import reward_maker
from main.common_env import common_env

# Import the DQN agent: 
from main.agent_deep_q import DQN


# Naming the agent and setting up the directory path:
now    = datetime.now()
NAME   = 'DQN'+now.strftime("_%d-%m-%Y_%H-%M-%S")
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
power_dem_arr = main_dataset.load_total_power()

# Load the LSTM input dataset:
lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

# Making predictions:
normal_predictions = wahrsager(lstm_dataset, power_dem_arr, TYPE='NORMAL').pred()[:-12]
seq_predictions    = wahrsager(lstm_dataset, power_dem_arr, TYPE='SEQ', num_outputs=12).pred()

# Adding the predictions to the dataset:
df            = df[24:-12]
df['normal']  = normal_predictions
df['seq-max'] = max_seq(seq_predictions)


# Number of warm-up steps:
num_warmup_steps = 100
# Train every x number of steps:
update_num       = 50
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
    M_STRATEGY              = 'single_step',
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
    power_dem_arr  = power_dem_arr,
    input_list     = ['norm_total_power','normal','seq-max'],
    max_SMS_SoC    = 12,
    max_LION_SoC   = 54,
    PERIODEN_DAUER = 5,
    ACTION_TYPE    = 'discrete',
    OBS_TYPE       = 'contin',
    discrete_space = 22)


# Setup Agent:
Agent = DQN(
    env            = env,
    memory         = deque(maxlen=(horizon+update_num)),
    max_steps      = max_steps,
    gamma          = 0.85,
    epsilon        = 0.8,
    epsilon_min    = 0.1,
    epsilon_decay  = 0.999996,
    lr             = 0.5,
    tau            = 0.125,
    model_lr       = 0.5,
    activation     = 'relu',
    loss           = 'mean_squared_error')


print('Warmup-Steps per Episode:', num_warmup_steps)
print('Training for',epochs,'Epochs')

for e in range(epochs):
    
    cur_state = env.reset()
    update_counter   = 0
    warmup_counter   = 0

    while warmup_counter < num_warmup_steps:
        action, epsilon                 = Agent.act(cur_state)
        new_state, reward, done, sce, _ = env.step(action, epsilon, random_mode=True)
        new_state                       = new_state
        Agent.remember(cur_state, action, reward, new_state, done, sce)

        cur_state = new_state
        warmup_counter += 1

    for step in range(epochs_len):

        action, epsilon                 = Agent.act(cur_state)
        new_state, reward, done, sce, _ = env.step(action, epsilon)
        new_state                       = new_state
        Agent.remember(cur_state, action, reward, new_state, done, sce)
        
        cur_state = new_state
        
        if done == False:
            batch_size = update_num
        else:
            batch_size = update_num + horizon

        update_counter += 1
        if update_counter == update_num or done == True:
            Agent.replay(batch_size)
            update_counter = 0

        if done:
            break

    if e % 100 == 0:
        Agent.save_agent(NAME, e)




