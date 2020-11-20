.. Studienprojekt documentation master file, created by
   sphinx-quickstart on Fri Nov 20 14:43:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Peak Shaving: Leveling the power consumption of a production system through reinforcement learning
==================================================================================================

This documentation provides an explanation for the code as part of the study project: Nivellierung der Leistungsaufnahme eines Produktionssystems durch Reinforcement Learning (link). Note that the pdf is only available in german. Further details in english can be read here: About the project(link).

Information about the installation, first steps and a general example are found in the Getting Started Guide(link).

We also provide detailed information and to some extend explanations of all the classes and functions used in the different modules in the section: Module Documentation. Look in the Example section if you want to see how to use the modules.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   about_project
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Module Documentations:

   modules/schaffer_doc
   modules/wahrsager_doc
   modules/common_env_doc
   modules/reward_maker_doc
   modules/logger_doc
   modules/agent_heuristic_doc
   modules/agent_q_table_doc
   modules/agent_deep_q_doc
   modules/agent_lstm_q_doc
   modules/agent_PPO2_doc

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/agent_heuristic_ex
   examples/agent_q_table_ex
   examples/agent_deep_q_ex
   examples/agent_lstm_q_ex
   examples/agent_PPO2_ex


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
