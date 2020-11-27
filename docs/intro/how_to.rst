.. _how_to:

How to use the agents:
======================

This sections aims to provide a tutorial how to set up an agent. For this purpose the ``agent_q_table`` will be used. You can look up the examples for the other agents to see the differences, but the basic structure is for all agents (except PPO2) pretty much the same. The code will be explained step by step and put together at the end.

Importing the dependencies:

.. code-block:: python

    from datetime import datetime
    from collections import deque

    from main.schaffer import mainDataset, lstmInputDataset
    from main.wahrsager import wahrsager
    from main.common_func import max_seq, mean_seq
    from main.common_env import common_env
    from main.reward_maker import reward_maker

    # Import the Q-Table agent: 
    from main.agent_q_table import Q_Learner

The last line determine which agent we use. In this case we want to create an agent that uses a Q-Table, which will limit the input and action-space to discrete values. This is important for setting up the environment later.

Setting up parameters and loading the dataset:

.. code-block:: python

    # Naming the agent and setting up the directory path:
    now    = datetime.now()
    NAME   = 'Q_Table'+now.strftime("_%d-%m-%Y_%H:%M:%S")
    D_PATH = '_BIG_D/'

    # Load the dataset:
    main_dataset = mainDataset(D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True)
    # Normalized dataframe:
    df = main_dataset.make_input_df(drop_main_terminal=False, use_time_diff=True, day_diff='holiday-weekend')
    # Sum of the power demand dataframe (nor normalized):
    power_dem_arr = main_dataset.load_total_power()

- importance of D_PATH and full_dataset
- why normalized and unnormalized sum

Making LSTM-Predictions:

.. code-block:: python
    
    # Load the LSTM input dataset:
    lstm_dataset = lstmInputDataset(D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True,
                                            num_past_periods=12, drop_main_terminal=False, use_time_diff=True,
                                            day_diff='holiday-weekend')
    # Making predictions:
    normal_predictions = wahrsager(lstm_dataset, TYPE=, NAME=).pred()[:-12]
    seq_predictions    = wahrsager(lstm_dataset, TYPE=, NAME=, num_outputs=12).pred()

    # Adding the predictions to the dataset:
    df            = df[24:-12]
    df['normal']  = normal_predictions
    df['seq-max'] = max_seq(seq_predictions)

- warum df[24:-12], etc
- beispiel um normale labels und sequence labels zu verwenden

Initilizing parameters for the iteration:

.. code-block:: python
    
    # Initilisiere Parameter für Target-Network
    update_num       = 1
    update_counter   = 0

    # Inititialsiere Epoch-Parameter:
    epochs           = 1000
    epochs_len       = len(df)

    num_warmup_steps = 100
    warmup_counter   = 0

    R_HORIZON = 0

Setting up the ``reward_maker``:

.. code-block:: python
    
    r_maker = reward_maker(
        COST_TYPE               = 'exact_costs',     # 'yearly_costs', 'max_peak_focus'
        R_TYPE                  = 'savings_focus',   #'costs_focus', 'savings_focus'
        M_STRATEGY              = 'single_step',              # None, 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
        R_HORIZON               = None,         # 'episode', 'single_step', integer for multi-step
        cost_per_kwh            = 0.2255,  # in €
        LION_Anschaffungs_Preis = 34100,   # in €
        LION_max_Ladezyklen     = 1000,
        SMS_Anschaffungs_Preis  = 115000/3,# in €
        SMS_max_Nutzungsjahre   = 20,      # in Jahre
        Leistungspreis          = 102,     # in €
        focus_peak_multiplier   = 4        # multiplier for max_peak costs
        )

Setting up the ``common_env``:

.. code-block:: python
    
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
        num_discrete_obs     = 21,
        num_discrete_actions = 22,
        #action_space         = spaces.Discrete(22), # A ∈ [0,1]
        #observation_space    = spaces.Box(low=0, high=21, shape=(4,1), dtype=np.float16),
        reward_maker         = r_maker)

Setting up the ``agent_q_table``:

.. code-block:: python
    
    Agent = Q_Learner(
        env            = env,
        #memory         = deque(maxlen=(update_num)),
        memory         = deque(maxlen=(R_HORIZON+update_num)),

        gamma          = 0.85,
        epsilon        = 0.8,
        epsilon_min    = 0.1,
        epsilon_decay  = 0.999996,
        lr             = 0.5,
        tau            = 0.125,

        Q_table        = np.zeros((22,22,22,22,22)), # jede Dimension jeweils ∈ [0,0.05,...,1]
        )

Iterating through epochs:

.. code-block:: python
    
    for e in range(epochs):
        cur_state = env.reset()

        while warmup_counter < num_warmup_steps:
            ...
            warmup_counter += 1

        for s in range(epochs_len):
            ...

            if done == False:
            index_len = update_num
            else:
                index_len = update_num + R_HORIZON

            update_counter += 1
            if update_counter == update_num or done == True:
                Agent.replay(index_len)
                update_counter = 0

            if done:
                break

        if e % 10 == 0:
            Agent.save_agent(NAME, DATENSATZ_PATH, e)

'...':

.. code-block:: python
    
    # For every step (normal and warm-up):
    action, epsilon            = Agent.act(cur_state)
    new_state, reward, done, step_counter_episode, _ = env.step(action, epsilon)
    new_state                  = new_state.reshape(len(cur_state),1).tolist()            
    Agent.remember(cur_state, action, reward, new_state, done, step_counter_episode)
    cur_state                  = new_state



Full code:

.. code-block:: python
    
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
    main_dataset = mainDataset(D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True)
    # Normalized dataframe:
    df = main_dataset.make_input_df(drop_main_terminal=False, use_time_diff=True, day_diff='holiday-weekend')
    # Sum of the power demand dataframe (nor normalized):
    power_dem_arr = main_dataset.load_total_power()

    # Load the LSTM input dataset:
    lstm_dataset = lstmInputDataset(D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True,
                                            num_past_periods=12, drop_main_terminal=False, use_time_diff=True,
                                            day_diff='holiday-weekend')
    # Making predictions:
    normal_predictions = wahrsager(lstm_dataset, TYPE=, NAME=).pred()[:-12]
    seq_predictions    = wahrsager(lstm_dataset, TYPE=, NAME=, num_outputs=12).pred()

    # Adding the predictions to the dataset:
    df            = df[24:-12]
    df['normal']  = normal_predictions
    df['seq-max'] = max_seq(seq_predictions)

    # Initilisiere Parameter für Target-Network
    update_num       = 1
    update_counter   = 0

    # Inititialsiere Epoch-Parameter:
    epochs           = 1000
    epochs_len       = len(df)

    num_warmup_steps = 100
    warmup_counter   = 0

    R_HORIZON = 0


    r_maker = reward_maker(
        COST_TYPE               = 'exact_costs',     # 'yearly_costs', 'max_peak_focus'
        R_TYPE                  = 'savings_focus',   #'costs_focus', 'savings_focus'
        M_STRATEGY              = 'single_step',              # None, 'sum_to_terminal', 'average_to_neighbour', 'recurrent_to_Terminal'
        R_HORIZON               = None,         # 'episode', 'single_step', integer for multi-step
        cost_per_kwh            = 0.2255,  # in €
        LION_Anschaffungs_Preis = 34100,   # in €
        LION_max_Ladezyklen     = 1000,
        SMS_Anschaffungs_Preis  = 115000/3,# in €
        SMS_max_Nutzungsjahre   = 20,      # in Jahre
        Leistungspreis          = 102,     # in €
        focus_peak_multiplier   = 4        # multiplier for max_peak costs
        )


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
        num_discrete_obs     = 21,
        num_discrete_actions = 22,
        #action_space         = spaces.Discrete(22), # A ∈ [0,1]
        #observation_space    = spaces.Box(low=0, high=21, shape=(4,1), dtype=np.float16),
        reward_maker         = r_maker)

    Agent = Q_Learner(
        env            = env,
        #memory         = deque(maxlen=(update_num)),
        memory         = deque(maxlen=(R_HORIZON+update_num)),

        gamma          = 0.85,
        epsilon        = 0.8,
        epsilon_min    = 0.1,
        epsilon_decay  = 0.999996,
        lr             = 0.5,
        tau            = 0.125,

        Q_table        = np.zeros((22,22,22,22,22)), # jede Dimension jeweils ∈ [0,0.05,...,1]
        )

    for e in range(epochs):
        cur_state = env.reset()

        while warmup_counter < num_warmup_steps:
            action, epsilon            = Agent.act(cur_state)
            new_state, reward, done, step_counter_episode, _ = env.step(action, epsilon)
            new_state                  = new_state.reshape(len(cur_state),1).tolist()            
            Agent.remember(cur_state, action, reward, new_state, done, step_counter_episode)
            cur_state                  = new_state

            warmup_counter += 1

        for s in range(epochs_len):
            action, epsilon            = Agent.act(cur_state)
            new_state, reward, done, step_counter_episode, _ = env.step(action, epsilon)
            new_state                  = new_state.reshape(len(cur_state),1).tolist()            
            Agent.remember(cur_state, action, reward, new_state, done, step_counter_episode)
            cur_state                  = new_state

            if done == False:
            index_len = update_num
            else:
                index_len = update_num + R_HORIZON

            update_counter += 1
            if update_counter == update_num or done == True:
                Agent.replay(index_len)
                update_counter = 0

            if done:
                break

        if e % 10 == 0:
            Agent.save_agent(NAME, DATENSATZ_PATH, e)