.. _getting_started:

Getting Started Guide
=====================

``peak-shaver`` aims to provide the tools to explore different approaches of reinforcement learning within a simulation of the `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . The module for the simulation ``common_env`` is made as a ``gym`` environment, which provides a common API for a wide range of different RL-libraries (for example ``stable-baseline`` which is also used as part of the study project). You can also create your own Agents following the ``gym`` guide-lines. But note that ``common_env`` requires some extra methods (look up the :ref:`module <common_env_doc>`) which will also be explained in this guide. For example is ``reward_maker`` used to specify the kind of reward the agent will receive.

Installation and Dependencies
*****************************

You can download the zip file from the `github repository <https://github.com/maik97/peak-shaver>`_ (alternatively just clone the project to your own github) or run the command below if you have `git <https://git-scm.com/downloads>`_ installed. Since we want to encourage improvements of the project there won't be a pip available.

.. code-block:: console
   
    $ git clone git://github.com/maik97/peak-shaver.git

Make sure to have these libraries with the right versions installed:

- matplotlib==3.3.3
- numpy==1.19.4
- pandas==0.25.3
- stable-baselines==2.10.1
- tensorboard==1.9.0
- tensorflow==1.9.0
- gym==0.17.3
- tqdm
- h5py

If you dont know how to install those properly look up `pip <https://pip.pypa.io/en/stable/>`_ . You can also install all dependedencies at once via the requirements.txt found in the github repository.

The dataset to simulate the can be downloaded here: `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . There are two different versions, one is the complete dataset over three months, the smaller one is just the first week.

Folder Structure
****************
Set up these folders, if you want to follow the examples provided, and put (both) unzipped datasets in the folder dataset. /peak-shaver-master is the downloded github folder.

| peak-shaver-master
| ├── peak-shaver
| │   ├── dataset
| │   │   ├── hipe_cleaned_v1.0.1_geq_2017-10-23_lt_2017-10-30
| │   │   └── hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01
| │   ├── _BIG_D
| │   ├── _small_d
| │   ├── [Put here any of your own code]
| │   └── ...
| └── ...

When following the examples you should be in the directory /peak-shaver, this is also the place you would put your own code.

Data Preparation
****************
The data preparation will be executed automaticaly when you first run ``wahrsager`` or any of the agents (provided you didn't do it manually). But it is recommended to create the preparetions seperately with ``schaffer`` since this can take up some time and you have the freedom to set up some parameters to your liking.

First all the necassary functions to transform the dataset are explained seperatly. You can run these step by step or just run the last one in which case all the previous steps will be run automaticly.

- ``load_geglättet_df()`` will take the dataset and smoth the data to a specific timeframe
- ``load_norm_data()`` will take the dataset from ``load_geglättet_df()`` and will first add differentiation between weekday, then add the activation time of each machine and lastly normalize the data. It also has a dataset where the sum of power consumption isn't normalized (Note that this function is deprecated, thus you dont need to run it)
- ``load_only_norm_data()`` has the same functionality as ``load_norm_data()``, except the extra dataset where the sum of power consumption isn't normalized
- ``load_total_power()`` has the same functionality as ``load_norm_data()``, except it returns only the extra dataset where the sum of power consumption isn't normalized
- ``alle_inputs()`` will take the datasets of ``load_only_norm_data()`` and merge those in a single one 
- ``alle_inputs_neu()`` is an extra funtion that makes a differentiation between work day and holiday (might be useful for predictions)

Recommended way to run the necessary functions:

.. code-block:: python
    
    import main.schaffer as schaffer

    schaffer.global_var(_NAME='', _VERSION='', _DATENSATZ_PATH ='_BIG_D/', _großer_datensatz = True, _zeitintervall = '5min')
    
    # If you want to check that everthing works fine, run those rather step by step:
    schaffer.load_geglättet_df()
    schaffer.load_only_norm_data()
    schaffer.load_total_power()
    schaffer.alle_inputs()
    schaffer.alle_inputs_neu()

If you want to know more about possible parameters for the ``schaffer`` functions check out the :ref:`module page <schaffer_doc>`.

Making Predictions
******************
Following the same princible above (time consumption, more freedom to set up) it is also recommended to make the predictions seperately, although this will also be done automaticly provided you didn't do it manually. 

With the module ``wahrsager`` you can train a LSTM that aims to predict the future power consumption. It's possible to modify the ``main`` function and run ``wahrsager`` directly. You can also create your own python code following this example:

.. code-block:: python
    
    ''' Example code to train a LSTM using the wahrsager module'''
    from main.wahrsager import wahrsager, max_seq, mean_seq, try_training_on_gpu()

    # Check if GPU is available:
    try_training_on_gpu()

    # Predictions (and training) with different approaches:
    prediction_mean           = wahrsager(PLOT_MODE=True, TYPE='MEAN').train()
    prediction_max            = wahrsager(PLOT_MODE=True, TYPE='MAX').train()
    prediction_normal         = wahrsager(PLOT_MODE=True, TYPE='NORMAL').train()
    prediction_max_label_seq  = wahrsager(PLOT_MODE=True, TYPE='MAX_LABEL_SEQ').train()
    prediction_mean_label_seq = wahrsager(PLOT_MODE=True, TYPE='MEAN_LABEL_SEQ').train()

    prediction_seq      = wahrsager(PLOT_MODE=True, TYPE='SEQ', num_outputs=12).train()
    max_prediction_seq  = max_seq(prediction_seq)
    mean_prediction_seq = mean_seq(prediction_seq)

The ``train()`` function is used to train a LSTM-model and will return predictions after the training is complete. You can use ``pred()`` instead of ``train()`` once you have run the training for the first time (This will be used by the agents). You can find the saved models in either _BIG_D/LSTM-models/ or _small_d/LSTM-models/.

There are different approaches to modify the input-dataset, which can be set with ``TYPE=...``. Below are explanations of the variables from the code snippet which are returns from a LSTM with a different ``TYPE``.

- ``prediction_mean`` with ``TYPE='MEAN'``: Predictions of the dataset modified with a rolling mean
- ``prediction_max`` with ``TYPE='MAX'``: Predictions of the dataset modified with a rolling max
- ``prediction_normal`` with ``TYPE='NORMAL'``: Predictions of the unmodified dataset
- ``prediction_max_label_seq`` with ``TYPE='MAX_LABEL_SEQ'``: Predictions where just the label data is modified with a rolling max
- ``prediction_mean_label_seq`` with ``TYPE='MEAN_LABEL_SEQ'``: Predictions where just the label data is modified with a rolling mean
- ``prediction_seq`` with ``TYPE='SEQ'``: Sequence-Predictions of the unmodified dataset, each sequence can be transformed to the mean or max value with ``max_seq(prediction_seq)`` or ``mean_seq(prediction_seq)``

All these different approaches will have similiar results, but can be used to optimize the predictions furthermore. If you want to tune the parameters, look up the ``wahrsager`` class :ref:`here <wahrsager_doc>` (change timeframe, LSTM size, ...). Note that for every new timeframe a seperate dataset will be created.

Set ``PLOT_MODE=True`` if you want to see a graph of the predictions compared to the actual data. You also can find the saved graphs in either _BIG_D/LSTM-graphs/ or _small_d/LSTM-graphs/. An example graph is provided below:

- hier kommt beispiel graph

Explanation of a Basic RL-Agent
*******************************

In this section a basic RL-Agent that uses a gym environment will be explained. All agents are build in a similar structure, thus this section aims to provide a basic understanding. The differences will be explained for each agent in the Examples section. Note that all the code provided in this section is pseudo-code.

Assuming you have understood the basics of RL-Learning, the first thing to explain is the general structure of a RL-Agent class:

.. code-block:: python
    
    class Q_Learner:
        
        def __init__(self, env, memory, gamma, epsilon, epsilon_min, epsilon_decay, lr, tau, Q_table):
        ...

        def act(self, state):
        ...

        def remember(self, state, action, reward, new_state, done, ...):
        ...

        def replay(self, ...):
        ...

        def save_agent(self, NAME, DATENSATZ_PATH, e):
        ...

- ``__init__()`` is all about parameter tuning. Note that in this case we have a parameter called Q_table (This will be different for each type of RL-Agent).
- ``act()`` is the function in which the agent decides on its actions based on the state. This is also the place where the greedy function will be applied.
- ``remember()`` is necessary to save the all the necessary information for the learning process, since we dont want to update the Q-values every single step.
- ``replay()`` is where the Q-function is applied and the learning process takes place, with the help of the memory from the ``remember()`` function.
- ``save_agent()`` is used to make a backup of the agent. This should be used every x steps (x should be big, because the total steps can go into millions), since you dont want to make a backup every step. Note that each backup takes time as well as space on your device.

The full code of the basic RL-Agent can be checked out on `Github <https://github.com/maik97/peak-shaver/blob/main/peak-shaver/main/agent_q_table.py>`_ .

The next thing to understand is the basic structure of a ``gym`` environment:

.. https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

.. code-block:: python
    
    import gym
    from gym import spaces

    class CustomEnv(gym.Env):
      """Custom Environment that follows gym interface"""
      metadata = {'render.modes': ['human']}

      def __init__(self, arg1, arg2, ...):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

      def step(self, action):
        # Execute one time step within the environment
        ...
      def reset(self):
        # Reset the state of the environment to an initial state
        ...
      def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

      def more_functions_to_simulate_the_data(...):
        # In the case of peak shaving the batteries need to be simulated
        ...

      ...

When put together in order to iterate over each step it should look something like this:

.. code-block:: python

    from gym_env import CustomEnv
    from agent import Q_Learner
    from schaffer import dataset

    env = CustomEnv(dataset,...)
    Agent = Q_Learner(...)

    # naming the model:
    NAME = 'basic_agent'
    # using the big dataset:
    DATENSATZ_PATH = '_BIG_D'

    # number of epochs:
    epochs = x
    # every y steps the agent will learn
    update_num = y

    for e in range(epochs):
        '''
        you can add here some functionality for warm-up steps
        (basically the same as below without learning)
        '''
        cur_state = env.reset()

        update_counter = 0
        for step in range(len(dataset)):

            action, epsilon            = Agent.act(cur_state)
            new_state, reward, done, _ = env.step(action, ...)
            Agent.remember(cur_state, action, reward, new_state, done, ...)
            cur_state                  = new_state

            update_counter += 1
            if update_counter == update_num or done == True:
                Agent.replay(...)
                update_counter = 0

            if done:
                break

        if e % 10 == 0:
            Agent.save_agent(NAME, DATENSATZ_PATH, e)

Note that all the provided pseudo-codes are more complex when implementet.