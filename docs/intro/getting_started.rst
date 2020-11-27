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

Note that ``tensorflow 1.9.0`` is an older version and only works with ``python 3.6``. The code of ``logger`` needs to be updated in order to be compatible with of ``tensorflow 2.x.x``.

If you dont know how to install those properly look up `pip <https://pip.pypa.io/en/stable/>`_ . You can also install all dependedencies at once via the requirements.txt found in the github repository.

The dataset to simulate the can be downloaded here: `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . There are two different versions, one is the complete dataset over three months, the smaller one is just the first week.

Folder Structure
****************

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

- ``peak-shaver-master`` is the downloded github folder.
- ``peak-shaver`` is where the actual package is located. When following the examples or if you want to create your own code you should be in this directory.
- ``dataset``: put in (both) unzipped HIPE-datasets.
- ``_BIG_D`` (big dataset) and ``_small_d`` (small dataset): this is where datasets, models, statistics and logs will be saved. Note that those folders will be created by setting the parameter `D_PATH` and therfore can be named differently. More on this in the next section.

Data Preparation
****************
The data preparation will be executed automaticaly when you first run ``wahrsager`` or any of the agents (provided you didn't do it manually). But it is recommended to create the preparations separately with ``schaffer`` since this can take up some time and you have the freedom to set up some parameters to your liking.

Create the basic dataset:

.. code-block:: python
    
    from main.schaffer import mainDataset

    main_dataset_creator = mainDataset(D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True)

    # Run this first, since this can take up a lot of time:
    main_dataset_creator.smoothed_df()
    
    # These don't take up a lot of time to run, 
    # but you can run those beforhand to check if everything is setup properly:
    main_dataset_creator.load_total_power()
    main_dataset_creator.normalized_df()
    main_dataset_creator.norm_activation_time_df()

- :meth:`schaffer.mainDataset.smoothed_df` will take the dataset and smooth the data to a specific time-frame.
- :meth:`schaffer.mainDataset.load_total_power` will take the table from ``smoothed_df`` and calculates the (not normalized) sum of the power requirements.
- :meth:`schaffer.mainDataset.normalized_df` will take the table from ``smoothed_df`` and normalize the data
- :meth:`schaffer.mainDataset.norm_activation_time_df` will take the table from ``smoothed_df`` and calculate the normalized activation times of the machines.

In this tutorial we seperate the big and small datasets, by setting ``D_PATH=_BIG_D`` for the big one and ``D_PATH=_BIG_D`` for the small one. Dont forget to set ``full_dataset=False`` if you want to use the small dataset. ``period_string_min`` can be set to `xmin` where x are the minutes one period should be.

Create an input-dataset:

.. code-block:: python
    
    from main.schaffer import lstmInputDataset

    lstm_dataset_creator = lstmInputDataset(D_PATH='_BIG_D/', period_string_min='5min', full_dataset=True,
                                            num_past_periods=12, drop_main_terminal=False, use_time_diff=True,
                                            day_diff='holiday-weekend')

    # If you want to check that everything works fine, run those rather step by step:
    lstm_dataset_creator.rolling_mean_training_data()
    lstm_dataset_creator.rolling_max_training_data()
    lstm_dataset_creator.normal_training_data()
    lstm_dataset_creator.sequence_training_data(num_seq_periods=12)

- :meth:`schaffer.lstmInputDataset.rolling_mean_training_data`creates an input-dataset that was transformed with a `rolling mean` operation
- :meth:`schaffer.lstmInputDataset.rolling_max_training_data` creates an input-dataset that was transformed with a `rolling max` operation
- :meth:`schaffer.lstmInputDataset.normal_training_data` creates a normale input-dataset.
- :meth:`schaffer.lstmInputDataset.normal_training_data` creates an input-dataset with sequence-labels the size of ``num_seq_periods``.

Make sure to use the same parameters in ``lstmInputDataset`` that you used in ``mainDataset``


Making Predictions
******************
Following the same principle above (time consumption, more freedom to set up) it is also recommended to make the predictions seperately, although this will also be done automatically provided you didn't do it manually. 

With the module ``wahrsager`` you can train a LSTM that aims to predict the future power consumption. It's possible to modify the ``main`` function and run ``wahrsager`` directly. You can also create your own python code following this example:

.. code-block:: python
    
    ''' Example code to train a LSTM using the wahrsager module'''
    from main.wahrsager import wahrsager
    from main.common_func import max_seq, mean_seq

    # Predictions (and training) with different approaches:
    prediction_mean           = wahrsager(PLOTTING=True, TYPE='MEAN').train()
    prediction_max            = wahrsager(PLOTTING=True, TYPE='MAX').train()
    prediction_normal         = wahrsager(PLOTTING=True, TYPE='NORMAL').train()
    prediction_max_label_seq  = wahrsager(PLOTTING=True, TYPE='MAX_LABEL_SEQ').train()
    prediction_mean_label_seq = wahrsager(PLOTTING=True, TYPE='MEAN_LABEL_SEQ').train()

    prediction_seq      = wahrsager(PLOTTING=True, TYPE='SEQ', num_outputs=12).train()
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

All these different approaches will have similar results, but can be used to optimize the predictions furthermore. If you want to tune the parameters, look up the ``wahrsager`` class :ref:`here <wahrsager_doc>` (change time-frame, LSTM size, ...). Note that for every new time-frame a separate dataset will be created.

Set ``PLOTTING=True`` if you want to see a graph of the predictions compared to the actual data. You also can find the saved graphs in either _BIG_D/LSTM-graphs/ or _small_d/LSTM-graphs/. An example graph is provided below:

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
- ``remember()`` is necessary to save the all the necessary information for the learning process, since we don't want to update the Q-values every single step.
- ``replay()`` is where the Q-function is applied and the learning process takes place, with the help of the memory from the ``remember()`` function.
- ``save_agent()`` is used to make a backup of the agent. This should be used every x steps (x should be big, because the total steps can go into millions), since you don't want to make a backup every step. Note that each backup takes time as well as space on your device.

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