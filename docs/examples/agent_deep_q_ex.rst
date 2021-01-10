.. _agent_deep_q_ex:
.. Studienprojekt documentation master file, created by
   sphinx-quickstart on Fri Nov 20 14:43:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Q-Learning
===============
A deep Q-Learning agent uses a neural network instead of a q-table for the learning process. Because of the neural network the inputs and outputs must be discrete. This has to be set while initiating the class :class:`common_env.common_env`:

.. code-block:: python
	
	env = common_env(
		...
		# DQN inputs can be conti and must be discrete:
		ACTION_TYPE    = 'discrete',
		OBS_TYPE       = 'contin',
		# Set number of discrete values:
		discrete_space = 22)

There are also more options to configure the neural network available, when initiating :class:`agent_deep_q.DQN`

.. code-block:: python
	
	agent = DQN(
		...
		activation     = 'relu',
		loss           = 'mean_squared_error',
		hidden_size    = 518)

Note that you can additionally use multi-step-rewards and/ or an LSTM integretaion. Those are described in :ref:`Deep Q-Learning with Multi-Step-Learning <agent_multi_step_ex>` and :ref:`Deep Q-Learning with LSTM integration <agent_lstm_q_ex>. 

Full Code:
**********

.. literalinclude:: ../docs_snippets/example_dqn.py