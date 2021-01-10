.. _agent_multi_step_ex:
.. Studienprojekt documentation master file, created by
   sphinx-quickstart on Fri Nov 20 14:43:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Q-Learning with Multi-Step-Learning
========================================

This agent has the same structure as :ref:`Deep Q-Learning<agent_deep_q_ex>`, but in order to implement multi-step-rewards the class :class:`reward_maker.reward_maker` needs to include following initilization:

.. code-block:: python
	
	r_maker = reward_maker(
		...
		# R_HORIZON is now an int for the periodes of the reward horizon:
		R_HORIZON  = 12,
		# Additional the multi-step strategy must be set:
		M_STRATEGY = 'sum_to_terminal')

Note that you can additionally use an LSTM integretaion as described in :ref:`Deep Q-Learning with LSTM integration <agent_lstm_q_ex>. 

Full Code:
**********

.. literalinclude:: ../docs_snippets/example_dqn_multistep.py
