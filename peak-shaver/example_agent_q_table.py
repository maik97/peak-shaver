'''
Example of an RL-Agent that uses the basic Q-Table.
'''
from datetime import datetime
from collections import deque

from main.schaffer import mainDataset, lstmInputDataset
from main.wahrsager import wahrsager
from main.common_func import max_seq, mean_seq
from main.common_env import common_env
from main.reward_maker import reward_maker

# Import the Q-Table agent: 
from main.agent_q_table import Q_Learner

# Naming the agent and setting up the directory path:
now    = datetime.now()
NAME   = 'Q_Table'+now.strftime("_%d-%m-%Y_%H:%M:%S")
D_PATH = '_BIG_D/'

# Load the dataset:
main_dataset = mainDataset(
    D_PATH=D_PATH,
    period_string_min='5min',
    full_dataset=True)

# Normalized dataframe:
df = main_dataset.make_input_df(
    drop_main_terminal=False,
    use_time_diff=True,
    day_diff='holiday-weekend')

# Sum of the power demand dataframe (nor normalized):
power_dem_arr = main_dataset.load_total_power()

# Load the LSTM input dataset:
lstm_dataset = lstmInputDataset(main_dataset,df)

# Making predictions:
normal_predictions = wahrsager(lstm_dataset, TYPE='NORMAL').pred()[:-12]
seq_predictions    = wahrsager(lstm_dataset, TYPE='SEQ', num_outputs=12).pred()

# Adding the predictions to the dataset:
df            = df[24:-12]
df['normal']  = normal_predictions
df['seq-max'] = max_seq(seq_predictions)


# Number of warm-up steps:
num_warmup_steps = 100
# Train every x number of steps:
update_num       = 500
# Number of epochs and steps:
epochs           = 1000
epochs_len       = len(df)
# Horizon for Multi-Step-Rewards and/or LSTM-Implementation:
horizon = 0

# Setup reward_maker
r_maker = reward_maker(
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
    df                   = df,
    power_dem_arr        = power_dem_arr,
    input_list           = ['norm_total_power','max_pred_seq'],
    DATENSATZ_PATH       = DATENSATZ_PATH,
    NAME                 = NAME,
    max_SMS_SoC          = 12,
    max_LION_SoC         = 54,
    PERIODEN_DAUER       = 5,
    ACTION_TYPE          = 'discrete',
    OBS_TYPE             = 'discrete',
    num_discrete_obs     = 21,
    num_discrete_actions = 22,
    reward_maker         = r_maker)

# Setup agent:
Agent = Q_Learner(
    env            = env,
    memory         = deque(maxlen=(update_num+horizon)),
    # Training-Parameter:
    gamma          = 0.85,
    epsilon        = 0.8,
    epsilon_min    = 0.1,
    epsilon_decay  = 0.999996,
    lr             = 0.5,
    tau            = 0.125,
    # jede Dimension jeweils ∈ [0,0.05,...,1]
    Q_table        = np.zeros((22,22,22,22,22)))


print('Warmup-Steps per Episode:', num_warmup_steps)
print('Training for',epochs,'Epochs')

for e in range(epochs):

    cur_state = env.reset()
    update_counter   = 0
    warmup_counter   = 0

    while warmup_counter < num_warmup_steps:
        action, epsilon                 = Agent.act(cur_state)
        new_state, reward, done, sce, _ = env.step(action, epsilon)
        new_state                       = new_state
        Agent.remember(cur_state, action, reward, new_state, done, sce)

        cur_state = new_state
        warmup_counter += 1

    for step in range(epochs_len):

        action, epsilon                 = Agent.act(cur_state)
        new_state, reward, done, sce, _ = env.step(action, epsilon)
        new_state                       = new_state
        Agent.remember(cur_state, action, reward, new_state, done, step_counter_episode)
        
        cur_state = new_state
        
        if done == False:
            training_num = update_num
        else:
            training_num = update_num + R_HORIZON

        update_counter += 1
        if update_counter == update_num or done == True:
            Agent.replay(training_num)
            update_counter = 0

        if done:
            break

    if e % 10 == 0:
        Agent.save_agent(NAME, DATENSATZ_PATH, e)


