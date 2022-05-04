## Script for creating pandas data frames from the data produced by running Actor_Critic_Network_Run ##

## import dependencies
import sys
sys.path.insert(0, './network')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytry
from pathlib import Path
import pickle

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

## Function for fetching the data and saving it as a pandas data frame
def get_data(data_folder, destination_folder, task, rule, rep, param):
    ## Lists containing the names for the tasks, rules and representations methods
    TASKS = ['MG', 'MC']
    REPS = {'1H' : 'One Hot', 'SSP': 'SSP', 'Grid' : 'GridSSP'}
    RULES = {'TD0': 'TD(0)', 'TDLambda': 'TD(Lambda)'}
    
    ## Make sure the function's input parameters are correct
    if task not in TASKS:
        raise ValueError(task, 'not found in', TASKS)
    elif rule not in RULES:
        raise ValueError(rule, 'not found in', REPS.keys())
    elif rep not in REPS:
        raise ValueError(rep, 'not found in', RULES.keys())
        
    # Load the data files
    data_folder = Path(data_folder+task+'_'+rule+'_'+rep+'_'+param)
    data = pd.DataFrame(pytry.read(data_folder))
    print('data loaded')
    
    # Save the data
    data.to_pickle(destination_folder+task+'_'+rule+'_'+rep+'_'+param+'_data')
    print('data saved to:', destination_folder+task+'_'+rule+'_'+rep+'_'+param+'_data')
    
## Run the preprocessing ##
data_folder = input("Enter path to data folder ")
print('data folder:', data_folder)

destination_folder = input("Enter path to folder where new data frames will be saved ")
print('new data frames will be saved to:', destination_folder)

task = input("Type the task initials (MG/MC) ")
rep = input("Type the representation used (1H/SSP/Grid) ")
rule = input("Type the rule used (TD0/TDLambda) ")
param = input("Type the parameter that was tested (512dims/1024dims, 5000neurons/6000neurons etc.) ")

get_data(data_folder, destination_folder, task, rule, rep, param)