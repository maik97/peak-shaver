.. _how_to:

How to use the agents 
=====================

Not updated yet!
Look up Getting Started and Examples, since those provide the correct informations for the current build.

This sections aims to provide a tutorial how to set up an agent. For this purpose the ``agent_q_table`` will be used. You can look up the examples for the other agents to see the differences, but the basic structure is for all agents pretty much the same. In order to use agents from other RL-Libriaris you need to make some minor changes, which will be explained at the end. If you didn't read the Getting Started Guide, we recommend to check this out first.

Setting up common settings
**************************
- filepath
- codesnippet: paramter and wahrsager


Importing the dependencies
**************************

.. code-block:: python
    
    # Datetime will be used to crate a timestamp when naming a agent-run
    from datetime import datetime

    # Normal imports to set up an agent:
    from common_settings import dataset_and_logger
    from main.common_func import training, testing
    from main.reward_maker import reward_maker
    from main.common_env import common_env

    # Import the Q-Table agent: 
    from main.agent_q_table import Q_Learner

The last line determines the agent we want to use. In this case we want to import an agent that uses a Q-Table. This will limit the input and action-space to discrete values, which will be important for setting up the environment later.

Setting up parameters and loading the dataset
*********************************************

.. code-block:: python

    # Naming the agent and setting up the directory path:
    now    = datetime.now()
    NAME   = 'Q_Table'+now.strftime("_%d-%m-%Y_%H:%M:%S")
    D_PATH = '_BIG_D/'

    # Load the dataset:
    main_dataset = mainDataset(D_PATH='_BIG_D/', period_min=5, full_dataset=True)
    # Normalized dataframe:
    df = main_dataset.make_input_df(drop_main_terminal=False, use_time_diff=True, day_diff='holiday-weekend')
    # Sum of the power demand dataframe (nor normalized):
    power_dem_arr = main_dataset.load_total_power()

- importance of D_PATH and full_dataset
- why normalized and unnormalized sum

Making LSTM-Predictions:

- Remove, since this is in common settings

.. code-block:: python
    
    # Load the LSTM input dataset:
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

    # Making predictions:
    normal_predictions = wahrsager(lstm_dataset, TYPE=, NAME=).pred()[:-12]
    seq_predictions    = wahrsager(lstm_dataset, TYPE=, NAME=, num_outputs=12).pred()

    # Adding the predictions to the dataset:
    df            = df[24:-12]
    df['normal']  = normal_predictions
    df['seq-max'] = max_seq(seq_predictions)

- warum df[24:-12], etc
- beispiel um normale labels und sequence labels zu verwenden

Initilizing parameters for the iteration
****************************************

.. code-block:: python
    
    # Number of warm-up steps:
    num_warmup_steps = 100
    # Train every x number of steps:
    update_num       = 50
    # Number of epochs and steps:
    epochs           = 1

Setting up the ``reward_maker``
*******************************

.. code-block:: python
    
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
        Leistungspreis          = 102)

Setting up the ``common_env``
*****************************

.. code-block:: python
    
    # Setup common_env
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
        # DQN inputs can be conti and must be discrete:
        ACTION_TYPE    = 'discrete',
        OBS_TYPE       = 'contin',
        # Set number of discrete values:
        discrete_space = 22,
        # Size of validation data:
        val_split      = 0.1)

Setting up the ``agent_q_table``
********************************

.. code-block:: python
    
    # Setup Agent:
    agent = DQN(
        env            = env,
        memory_len     = update_num,
        # Training parameter:
        gamma          = 0.85,
        epsilon        = 0.8,
        epsilon_min    = 0.1,
        epsilon_decay  = 0.999996,
        lr             = 0.5,
        tau            = 0.125,
        activation     = 'relu',
        loss           = 'mean_squared_error',
        hidden_size    = 518)

Training process
****************

Agents from other RL-Libraries
******************************
