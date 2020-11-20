.. _getting_started:

Getting Started Guide
=====================

``peak-shaver`` aims to provide the tools to explore different approaches of reinforcement learning within a simulation of the `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . The module for the simulation ``common_env`` is made as a ``gym`` environment, which provides a common API for a wide range of different RL-libraries (for example ``stable-baseline`` which is also used as part of the study project). You can also create your own Agents following the ``gym`` guide-lines. But note that ``common_env`` requires some extra methods (look up the :ref:`module <common_env_doc>`) which will also be explained in this guide. For example is ``reward_maker`` used to specify the kind of reward the agent will receive.

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

The dataset to simulate the can be downloaded here: `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . There are two different versions, one is the complete dataset over three months, the smaller one is just the first week.

Folder Structure
****************
Set up these folders, if you want to follow the examples provided, and put (both) unzipped datasets in the folder dataset.

| peak-shaver-master
| ├── peak-shaver
| │   ├── dataset
| │   │   ├── hipe_cleaned_v1.0.1_geq_2017-10-23_lt_2017-10-30
| │   │   └── hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01
| │   ├── _BIG_D
| │   ├── _small_d
| │   └── ...
| └── ...

Data Preparation
****************
The data preparation will be executed automaticaly when you first run ``wahrsager`` or any of the agents (provided you didn#t do it manually). But it is recommended to create the preparetions seperately with ``schaffer`` since this can take up some time and you have the freedome to set up some parameters to your liking.
- schaffer explanation

Making Predictions
******************
Following the same princible above (time consumption, more freedom to set up) it is also recommended to make the predictions seperately, although this will also be done automaticly providid you didn't do it manually. 
- example lstm predictions with specific timeframe

Basic RL-Agent with in-depth explanation
***************************************
- im gegensatz zu examples wird hier genau der aufbau erklärt (tut style)

