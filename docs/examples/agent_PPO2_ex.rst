.. _agent_PPO2_ex:
.. Studienprojekt documentation master file, created by
   sphinx-quickstart on Fri Nov 20 14:43:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Proximal Policy Optimization
============================

This agent implements the PPO2-agent from ```stable-baselines``. Inputs and outputs for PPOs can be continious. This has to be set while initiating the class :class:`common_env.common_env`. Additionaly you have to define here, that you are using an agent for standart ``gym`` environments:

.. code-block:: python

	env = common_env(
		...
		# PPO can use conti values:
		ACTION_TYPE = 'contin',
		OBS_TYPE    = 'contin',
		# Tells the environment to make standart GYM outputs, 
		# so agents from stable-baselines (or any other standart module) can be used
		AGENT_TYPE  = 'standart_gym')

Since its not possible to use the multi-step-rewards from :class:`reward_maker.reward_maker`, you have to use this standart initilization:

.. code-block:: python
	
	r_maker = reward_maker(
		...
		# Agents from stable base-lines cant use multi-step rewards from our code
		# So R_HOTIZON can only be 'single-step'
		R_HORIZON               = 'single_step')

Full Code:
**********

.. literalinclude:: ../docs_snippets/example_PPO2.py


