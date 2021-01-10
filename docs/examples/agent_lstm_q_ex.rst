.. _agent_lstm_q_ex_ex:
.. Studienprojekt documentation master file, created by
   sphinx-quickstart on Fri Nov 20 14:43:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Q-Learning with LSTM integration
=====================================

This agent has the same structure as :ref:`Deep Q-Learning<agent_deep_q_ex>`, but in order to implement an LSTM-network the class :class:`agent_deep_q.DQN` needs to include following initilization:

.. code-block:: python

	agent = DQN(
		...
		# Model type must be set to 'lstm' now
		model_type     = 'lstm',
		# LSTM size can be set:
		lstm_size      = 128)

Note that you can additionally use multi-step-rewards as described in :ref:`Deep Q-Learning with Multi-Step-Learning <agent_multi_step_ex>`.

Full Code:
**********

.. literalinclude:: ../docs_snippets/example_dqn_lstm.py
