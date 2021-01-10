.. _agent_q_table_ex:
.. Studienprojekt documentation master file, created by
   sphinx-quickstart on Fri Nov 20 14:43:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Basic Q-Learning
================

A basic Q-Learning agent uses a Q-Table for the learning process. Because of the Q-Table the inputs and outputs must be discrete. This has to be set while initiating the class :class:`common_env.common_env`:

.. code-block:: python

	env = common_env(
		...
		ACTION_TYPE    = 'discrete',
		OBS_TYPE       = 'discrete',
		discrete_space = 22)

Note that you can additionally use multi-step-rewards as described in :ref:`Deep Q-Learning with Multi-Step-Learning <agent_multi_step_ex>`.

Full Code:
**********

.. literalinclude:: ../docs_snippets/example_q_table.py

