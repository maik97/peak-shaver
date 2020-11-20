.. _getting_started:

Getting Started Guide
=====================

``peak-shaver`` aims to provide the tools to explore different approaches of reinforcement learning within a simulation of the `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . The module for the simulation ``common_env`` is made as a ``gym`` environment, which provides a common API for a wide range of different RL-libraries (for example ``stable-baseline`` which is also used as part of the study project). You can also create your own Agents following the ``gym`` guide-lines. Note that ``common_env`` requires some extra methods (look up the :ref:`module <common_env_doc>`) which will also be explained in this guide. For example is ``reward_maker`` used to specify the kind of reward the agent will receive.

Installation and Dependencies
*****************************

You can download the zip file from the `github repository <https://github.com/maik97/peak-shaver>`_ (alternatively just clone the project to your own github) or run this command if you have `git <https://git-scm.com/downloads>`_ installed.

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

Folder Structure
****************
- wie ordner mit daten heißen muss

Data Preparation
****************
- schaffer explanation

Making Predictions
******************
- example lstm predictions with specific timeframe

Basic RL-Agent with in-depth explanation
***************************************
- im gegensatz zu examples wird hier genau der aufbau erklärt (tut style)

