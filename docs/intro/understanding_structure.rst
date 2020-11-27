.. _understanding_structure:

Understanding the Structure
===========================



Understanding the agents structure
**********************************

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

Understanding the environment structure
***************************************

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

Putting together Agent and Environment
**************************************

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