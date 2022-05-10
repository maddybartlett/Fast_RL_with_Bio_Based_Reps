# Fast_RL_with_Bio_Based_Reps
Companion repository to ICCM paper "Fast Online Reinforcement Learning with Biologically-Based State Representations".

Contributors: Dr. M. Bartlett, Dr. T. Stewart & Dr. J. Orchard
Affiliation: University of Waterloo, ON, Canada

Repository to accompany Bartlett, Stewart & Orchard (2022) "Fast Online Reinforcement Learning with Biologically-Based State Representations" ICCM Paper (LINK).

## Requirements:

You will need to have Jupyter Notebook installed in order to run these scripts. Recommended to install [Anaconda](https://www.anaconda.com/products/individual). 

* Python 3.5+
* OpenAI Gym
* Gym MiniGrid
* PyTry
* Nengo
* NNI
* Numpy
* Pandas
* Pickle
* Matplotlib
* Plotly.express
* Pathlib
* Sys
* OS

## Scripts and How-To Guide:

Download repository as a zip file or clone repository by clicking the "Code" button on the top right. <br>

### Run the Network

You can run the network from the command line using ac_network_run.py - in the terminal, navigate to the directory containing these scripts and run the command "python ac_network_run.py". 

Running this command without any changes to the file will result in you first being prompted to provide a directory for saving the data and figures. The AC network will then try to solve the MiniGrid task using the TD($\lambda$) learning rule and Spatial Semantic Pointers (SSPs) to represent the state. The preset parameter values were identified as being one of the best performing parameter sets for this configuration. 
A plot showing the total reward received in each of the 500 learning trials will be presented once the AC network has finished, and the plot will also be saved as ep_rewards.pdf in the directory you provided. 

### Experiment

The python files named "config..." and "exp..." are the scripts needed to re-run the NNI experiments which perform a search of the hyperparameter space for 8 configurations of the AC network. 
These configurations vary in terms of the learning rule used (TD(0) vs. TD($\lambda$)) and the method for representing the state information (baseline vs. one hot vs. random SSPs vs. grid cells). 
In order to replicate the experiment reported in Bartlett, Stewart & Orchard (2022), you simply need to run each "config...." file from the command line. For example: 

```
nnictl --config config_MGTD0Baseline_long.yml
```

This will start the NNI parameter search experiment which will run for 12 hours. <br>
Once the experiment starts running, a localhost url will be provided in the terminal which you can go to in order to monitor the progress of the experiment. 

** Important: you must change the data_dir in each "exp..." file before running these NNI experiments. Change to a unique data directory where you would like the data from the experiment to be saved. ** We recommend using the following format for naming the data_dir: 

Path('../DATA_FOLDER/'TASK_RULE_REP), 

where:

* DATA_FOLDER = the parent folder where you want everything to be saved. Set this to whatever you want
* TASK = the task being learned:
    * 'MG' for MiniGrid
* RULE = the rule used:
    * 'TD0' for TD(0)
    * 'TDLam' for TD($\lambda$)
* REP = the representation used:
    * 'Baseline' for one hot with no neurons 
    * '1H' for one hot
    * 'SSP' for SSP
    * 'Grid' for grid cells

### Data Preprocessing:

Before running the analysis you will need to run a pre-processing script on the data in order to convert it from text files into pandas dataframes. You can do this either from the command line or in a Jupyter notebook.

The relevant scripts for this contain two main functions. 
The *txt_to_dataFrame* function will retrieve the text files saved by the NNI experiments and convert them to pandas data frames. 
Then the *add_roll_mean* function will add a column to the data frame that contains the rolling average reward gained for each run (the average reward over each 100 learning trials). 

#### Command Line:

To run the pre-processing from the command line, simply run:

'''
python preprocess_data.py
'''

You will be prompted to enter the path to the relevant data folder.

#### Notebook: 

Alternatively, you can step through the pre-process by running Preprocess_Data.ipynb from a Jupyter notebook. 

You will need to manually set the destination folder where you would like the data to be saved in cell 4:

'''python
folder = '.\\data\\main_experiment'
'''

### Analysis:

#### Command Line:

#### Notebook: 
