Peak Shaving: Leveling the power consumption of a production system through reinforcement learning
==================================================================================================

This documentation provides an explanation for the code as part of the study project: `Nivellierung der Leistungsaufnahme eines Produktionssystems durch Reinforcement Learning <https://www.stackoverflow.com>`_. Note that the pdf is only available in german. Further details in english can be read here: :ref:`About the project <about_project>`. All abbreviations that will be used in the documentation will be explained there too.

Information about the installation, first steps and a general example are found in the :ref:`Getting Started Guide <getting_started>`.

We also provide detailed information of all the classes and functions used by the different modules in the section: Module Documentation. Look in the Example section if you want to see how to use the different RL-aproaches we explored within the study project.

`string`
``peak-shaver``
:class:`testModule.testClass`

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   intro/about_project
   intro/getting_started
   intro/tensorboard_doc

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
