## Analysis of the NNI experiment data ##

## Import dependencies
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

## Helper Functions

# Function to calculate number of trials needed to solve problem in each experiment 
# (i.e. how many trials needed to reach goal rolling mean)

def trials_to_goal(data, goal):
    ## create list for storing trial indices
    goal_reached = []
    
    ## for each value in roll_mean, record the index of values > goal
    for i in range(len(data)):
        a=[index for index,val in enumerate(data['roll_mean'][i]) if val > goal]
        
        ## if experiment never reached goal, put max number of runs in list
        if len(a) < 1:
            goal_reached.append(2000)
        ## otherwise, add the index to the list for plotting
        else:
            goal_reached.append(a[0])
        
    ## return list 
    return goal_reached

# Function for creating a new dataframe with just the 'best' 2% of experiments. 
# I.e. the 2% of experiments that reached the goal rolling mean in the fewest learning trials.

def get_best(dataframe, params_list):
    ## Set a threshold for getting the quickest experiments
    top_2per = int(len(dataframe)*0.02)
    
    ## Generate new dataframe containing only the experiments that achieved the goal rolling mean 
    ## quicker than the threshold number of trials
    best = dataframe.nsmallest(top_2per, ['goal_reached'])
    best = best.reset_index()
        
    ## return new data frame
    return best

## Function for creating a dataframe containing data from all of the experiments 
## which used the 'best' performing parameter value combinations

def get_all_best_params(main_df, param_sets_df, param_list):
    ## for each parameter that was tested
    for p in param_list:
        
        ## initialise new data frame with the alpha parameter
        if p == 'alpha ':
            ## get minimum and maximum values of alpha
            max_v = param_sets_df[p].max()
            min_v = param_sets_df[p].min()
            
            ## find all instances of alpha values in this range being tested and create a new dataframe
            in_range = main_df.loc[(main_df[p] <= max_v) & 
                                   (main_df['alpha '] >= min_v)]
        
        ## for all other parameters
        else:
            ## get minimum and maximum values of the parameter
            max_v = param_sets_df[p].max()
            min_v = param_sets_df[p].min()
            
            ## restrict the data frame to instances where the tested value for this parameter were inside this range
            in_range = in_range.loc[(in_range[p] <= max_v) & 
                                    (in_range[p] >= min_v)]
        
        ## return new data frame
        return in_range
    
## Function for retrieving the reward schedules for each of the 'best' experiments
def get_best_reward_schedules(dataframe):
    
    ## remove square brackets from the data contained in the 'episodes ' column
    dataframe['episodes '] = dataframe['episodes '].str.replace('[','')
    dataframe['episodes '] = dataframe['episodes '].str.replace(']','')
    
    ## for each row in the data frame
    for i in range (len(dataframe)):
        ## data contained in each row of this column is currently a single string.
        ## create individual strings for each numerical value by separating after each comma
        dataframe['episodes '][i] = dataframe['episodes '][i].split(',')
        
        ## convert all values from strings to floats
        dataframe['episodes '][i] = pd.to_numeric(dataframe["episodes "][i], downcast="float")
        
    ## return data frame 
    return dataframe






## GET MAIN FIGURES FOR PAPER ##

folder = input("Enter path to data folder ")
data_folder = Path(folder)
print('path to folder:', data_folder)

## Load data frames
print('Loading data')
## TD(0)
mg_0_base =  pd.read_pickle(data_folder/'MG_TD0_Baseline_long_dataframe') # baseline condition
mg_0_1h =  pd.read_pickle(data_folder/'MG_TD0_1H_long_dataframe')         # 1 hot condition
mg_0_ssp =  pd.read_pickle(data_folder/'MG_TD0_SSP_long_dataframe')       # SSP condition
mg_0_grid =  pd.read_pickle(data_folder/'MG_TD0_Grid_long_dataframe')     # grid cell condition

## TD(lambda)
mg_lam_base =  pd.read_pickle(data_folder/'MG_TDLambda_Baseline_long_dataframe') # baseline condition
mg_lam_1h =  pd.read_pickle(data_folder/'MG_TDLambda_1H_long_dataframe')         # 1 hot condition
mg_lam_ssp =  pd.read_pickle(data_folder/'MG_TDLambda_SSP_long_dataframe')       # SSP condition
mg_lam_grid =  pd.read_pickle(data_folder/'MG_TDLambda_Grid_long_dataframe')     # grid cell condition


## Add a column showing how many learning trials it took to reach the 
## goal of a rolling average reward of 0.95 over the last 100 trials.
## Set goal value
goal = 0.95

## TD(0)
## generate list of how quickly the goal was reached in each experiment
goal_reached = trials_to_goal(mg_0_base, goal)
## add this list to dataframe as new column
mg_0_base['goal_reached'] = goal_reached

## Repeat for each representation ##
## 1 hot
goal_reached = trials_to_goal(mg_0_1h, goal)
mg_0_1h['goal_reached'] = goal_reached

## SSP
goal_reached = trials_to_goal(mg_0_ssp, goal)
mg_0_ssp['goal_reached'] = goal_reached

## grid cell
goal_reached = trials_to_goal(mg_0_grid, goal)
mg_0_grid['goal_reached'] = goal_reached


## TD(lambda)
## baseline
goal_reached = trials_to_goal(mg_lam_base, goal)
mg_lam_base['goal_reached'] = goal_reached

## Repeat for each representation
## 1 hot
goal_reached = trials_to_goal(mg_lam_1h, goal)
mg_lam_1h['goal_reached'] = goal_reached

## SSP
goal_reached = trials_to_goal(mg_lam_ssp, goal)
mg_lam_ssp['goal_reached'] = goal_reached

## grid cell
goal_reached = trials_to_goal(mg_lam_grid, goal)
mg_lam_grid['goal_reached'] = goal_reached


## TD(0)
## Convert values in dataframe from object to numeric
mg_0_base["alpha "] = pd.to_numeric(mg_0_base["alpha "], downcast="float")
mg_0_1h["alpha "] = pd.to_numeric(mg_0_1h["alpha "], downcast="float")
mg_0_ssp["alpha "] = pd.to_numeric(mg_0_ssp["alpha "], downcast="float")
mg_0_grid["alpha "] = pd.to_numeric(mg_0_grid["alpha "], downcast="float")

mg_0_base["beta "] = pd.to_numeric(mg_0_base["beta "], downcast="float")
mg_0_1h["beta "] = pd.to_numeric(mg_0_1h["beta "], downcast="float")
mg_0_ssp["beta "] = pd.to_numeric(mg_0_ssp["beta "], downcast="float")
mg_0_grid["beta "] = pd.to_numeric(mg_0_grid["beta "], downcast="float")

mg_0_base["gamma "] = pd.to_numeric(mg_0_base["gamma "], downcast="float")
mg_0_1h["gamma "] = pd.to_numeric(mg_0_1h["gamma "], downcast="float")
mg_0_ssp["gamma "] = pd.to_numeric(mg_0_ssp["gamma "], downcast="float")
mg_0_grid["gamma "] = pd.to_numeric(mg_0_grid["gamma "], downcast="float")

mg_0_1h["n_neurons "] = pd.to_numeric(mg_0_1h["n_neurons "], downcast="float")
mg_0_ssp["n_neurons "] = pd.to_numeric(mg_0_ssp["n_neurons "], downcast="float")
mg_0_grid["n_neurons "] = pd.to_numeric(mg_0_grid["n_neurons "], downcast="float")

mg_0_1h["sparsity "] = pd.to_numeric(mg_0_1h["sparsity "], downcast="float")
mg_0_ssp["sparsity "] = pd.to_numeric(mg_0_ssp["sparsity "], downcast="float")
mg_0_grid["sparsity "] = pd.to_numeric(mg_0_grid["sparsity "], downcast="float")

mg_0_ssp["dims "] = pd.to_numeric(mg_0_ssp["dims "], downcast="float")

mg_0_base["goal_reached"] = pd.to_numeric(mg_0_base["goal_reached"], downcast="float")
mg_0_1h["goal_reached"] = pd.to_numeric(mg_0_1h["goal_reached"], downcast="float")
mg_0_ssp["goal_reached"] = pd.to_numeric(mg_0_ssp["goal_reached"], downcast="float")
mg_0_grid["goal_reached"] = pd.to_numeric(mg_0_grid["goal_reached"], downcast="float")

## TD(lambda)
mg_lam_base["alpha "] = pd.to_numeric(mg_lam_base["alpha "], downcast="float")
mg_lam_1h["alpha "] = pd.to_numeric(mg_lam_1h["alpha "], downcast="float")
mg_lam_ssp["alpha "] = pd.to_numeric(mg_lam_ssp["alpha "], downcast="float")
mg_lam_grid["alpha "] = pd.to_numeric(mg_lam_grid["alpha "], downcast="float")

mg_lam_base["beta "] = pd.to_numeric(mg_lam_base["beta "], downcast="float")
mg_lam_1h["beta "] = pd.to_numeric(mg_lam_1h["beta "], downcast="float")
mg_lam_ssp["beta "] = pd.to_numeric(mg_lam_ssp["beta "], downcast="float")
mg_lam_grid["beta "] = pd.to_numeric(mg_lam_grid["beta "], downcast="float")

mg_lam_base["gamma "] = pd.to_numeric(mg_lam_base["gamma "], downcast="float")
mg_lam_1h["gamma "] = pd.to_numeric(mg_lam_1h["gamma "], downcast="float")
mg_lam_ssp["gamma "] = pd.to_numeric(mg_lam_ssp["gamma "], downcast="float")
mg_lam_grid["gamma "] = pd.to_numeric(mg_lam_grid["gamma "], downcast="float")

mg_lam_base["lambd "] = pd.to_numeric(mg_lam_base["lambd "], downcast="float")
mg_lam_1h["lambd "] = pd.to_numeric(mg_lam_1h["lambd "], downcast="float")
mg_lam_ssp["lambd "] = pd.to_numeric(mg_lam_ssp["lambd "], downcast="float")
mg_lam_grid["lambd "] = pd.to_numeric(mg_lam_grid["lambd "], downcast="float")

mg_lam_1h["n_neurons "] = pd.to_numeric(mg_lam_1h["n_neurons "], downcast="float")
mg_lam_ssp["n_neurons "] = pd.to_numeric(mg_lam_ssp["n_neurons "], downcast="float")
mg_lam_grid["n_neurons "] = pd.to_numeric(mg_lam_grid["n_neurons "], downcast="float")

mg_lam_1h["sparsity "] = pd.to_numeric(mg_lam_1h["sparsity "], downcast="float")
mg_lam_ssp["sparsity "] = pd.to_numeric(mg_lam_ssp["sparsity "], downcast="float")
mg_lam_grid["sparsity "] = pd.to_numeric(mg_lam_grid["sparsity "], downcast="float")

mg_lam_ssp["dims "] = pd.to_numeric(mg_lam_ssp["dims "], downcast="float")

mg_lam_base["goal_reached"] = pd.to_numeric(mg_lam_base["goal_reached"], downcast="float")
mg_lam_1h["goal_reached"] = pd.to_numeric(mg_lam_1h["goal_reached"], downcast="float")
mg_lam_ssp["goal_reached"] = pd.to_numeric(mg_lam_ssp["goal_reached"], downcast="float")
mg_lam_grid["goal_reached"] = pd.to_numeric(mg_lam_grid["goal_reached"], downcast="float")




## Create dataframes containing the 'best' 2% of nni experiments 
print('Getting best 2%')
## TD(0)
mg_0_base_best = get_best(mg_0_base, ['alpha ', 'beta ', 'gamma '])
mg_0_1h_best = get_best(mg_0_1h, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])
mg_0_ssp_best = get_best(mg_0_ssp, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'dims '])
mg_0_grid_best = get_best(mg_0_grid, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])

param_sets_base = mg_0_base_best[['index', 'alpha ','beta ','gamma ', 'goal_reached']]
param_sets_1h = mg_0_1h_best[['index', 'alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'goal_reached']]
param_sets_ssp = mg_0_ssp_best[['index', 'alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'dims ', 'goal_reached']]
param_sets_grid = mg_0_grid_best[['index', 'alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'goal_reached']]

## TD(lambda)
mg_lam_base_best = get_best(mg_lam_base, ['alpha ', 'beta ', 'gamma '])
mg_lam_1h_best = get_best(mg_lam_1h, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])
mg_lam_ssp_best = get_best(mg_lam_ssp, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'dims '])
mg_lam_grid_best = get_best(mg_lam_grid, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])

param_sets_lam_base = mg_lam_base_best[['index', 'alpha ','beta ','gamma ', 'goal_reached']]
param_sets_lam_1h = mg_lam_1h_best[['index', 'alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'goal_reached']]
param_sets_lam_ssp = mg_lam_ssp_best[['index', 'alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'dims ', 'goal_reached']]
param_sets_lam_grid = mg_lam_grid_best[['index', 'alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'goal_reached']]



## Create data frames containing experiments that tested value combinations 
## that were inside the ranges used in the top 2% experiments
## TD(0)
in_range_base = get_all_best_params(mg_0_base, param_sets_base, ['alpha ', 'beta ', 'gamma '])
in_range_1h = get_all_best_params(mg_0_1h, param_sets_1h, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])
in_range_ssp = get_all_best_params(mg_0_ssp, param_sets_ssp, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'dims '])
in_range_grid = get_all_best_params(mg_0_grid, param_sets_grid, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])

## TD(lambda)
in_range_lam_base = get_all_best_params(mg_lam_base, param_sets_lam_base, ['alpha ', 'beta ', 'gamma '])
in_range_lam_1h = get_all_best_params(mg_lam_1h, param_sets_lam_1h, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])
in_range_lam_ssp = get_all_best_params(mg_lam_ssp, param_sets_lam_ssp, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity ', 'dims '])
in_range_lam_grid = get_all_best_params(mg_lam_grid, param_sets_lam_grid, ['alpha ', 'beta ', 'gamma ', 'n_neurons ', 'sparsity '])

## CREATE PLOTS ##
print('Plotting optimization curves')
## Optimization Curves
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24,10), sharey=True)
fig.suptitle('Optimization Curves', fontsize=24)
fig.text(0.49, 0.91, "TD(0)", fontsize = 24)

ax1.scatter(mg_0_base.index, mg_0_base['goal_reached'], color = 'white', edgecolors = 'black')
ax1.set_title('Baseline', fontsize=22)
ax1.set_ylabel('No. Trials to reach goal reward (log scale)', fontsize=20)
ax1.yaxis.set_label_coords(-.15, -.1)
ax1.set_yscale('log')
ax1.tick_params(labelsize = 18)

ax2.scatter(mg_0_1h.index, mg_0_1h['goal_reached'], color = 'white', edgecolors = 'black')
ax2.set_title('1 Hot', fontsize=22)
ax2.set_yscale('log')
ax2.tick_params(labelsize = 18)

ax3.scatter(mg_0_ssp.index, mg_0_ssp['goal_reached'], color = 'white', edgecolors = 'black')
ax3.set_title('SSP', fontsize=22)
ax3.set_yscale('log')
ax3.tick_params(labelsize = 18)

ax4.scatter(mg_0_grid.index, mg_0_grid['goal_reached'], color = 'white', edgecolors = 'black')
ax4.set_title('Grid Cells', fontsize=22)
ax4.set_yscale('log')
ax4.tick_params(labelsize = 18)

fig.text(0.47, 0.48, "TD(Lambda)", fontsize = 24)

ax5.scatter(mg_lam_base.index, mg_lam_base['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax5.set_title('Baseline', fontsize=22)
ax5.set_yscale('log')
ax5.tick_params(labelsize = 18)

ax6.scatter(mg_lam_1h.index, mg_lam_1h['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax6.set_title('1 Hot', fontsize=22)
ax6.set_yscale('log')
ax6.tick_params(labelsize = 18)

ax7.scatter(mg_lam_ssp.index, mg_lam_ssp['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax7.set_title('SSP', fontsize=22)
ax7.set_xlabel('Experiment Index', fontsize=20)
ax7.xaxis.set_label_coords(-.11, -.15)
ax7.set_yscale('log')
ax7.tick_params(labelsize = 18)

ax8.scatter(mg_lam_grid.index, mg_lam_grid['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax8.set_title('Grid Cells', fontsize=22)
ax8.set_yscale('log')
ax8.tick_params(labelsize = 18)

plt.subplots_adjust(hspace=0.4)

## Save figure
print('Saving optimization curves in: ./figures/nni_combined_optim.pdf')
fig.savefig("./figures/nni_combined_optim.pdf", bbox_inches="tight")

## Best Parameter Plots
print('Plotting Best Parameter Experiments')
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24,10), sharey=True)
fig.suptitle('Performance with Best Parameters', fontsize=24)
fig.text(0.49, 0.91, "TD(0)", fontsize = 24)

ax1.scatter(in_range_base.index, in_range_base['goal_reached'], color = 'white', edgecolors = 'black')
ax1.set_title('Baseline', fontsize=22)
ax1.set_ylabel('No. Trials to reach goal reward (log scale)', fontsize=20)
ax1.yaxis.set_label_coords(-.2, -.15)
ax1.tick_params(labelsize = 18)
#ax1.set_yscale('log')

ax2.scatter(in_range_1h.index, in_range_1h['goal_reached'], color = 'white', edgecolors = 'black')
ax2.set_title('1 Hot', fontsize=22)
ax2.tick_params(labelsize = 18)
#ax2.set_yscale('log')

ax3.scatter(in_range_ssp.index, in_range_ssp['goal_reached'], color = 'white', edgecolors = 'black')
ax3.set_title('SSP', fontsize=22)
#ax3.set_xlabel('Experiment index', fontsize=14)
ax3.tick_params(labelsize = 18)
#ax3.set_yscale('log')

ax4.scatter(in_range_grid.index, in_range_grid['goal_reached'], color = 'white', edgecolors = 'black')
ax4.set_title('Grid Cells', fontsize=22)
#ax4.set_xlabel('Experiment index', fontsize=14)
ax4.tick_params(labelsize = 18)
#ax4.set_yscale('log')

fig.text(0.47, 0.48, "TD(Lambda)", fontsize = 24)

ax5.scatter(in_range_lam_base.index, in_range_lam_base['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax5.set_title('Baseline', fontsize=22)
ax5.tick_params(labelsize=18)
#ax5.set_yscale('log')

ax6.scatter(in_range_lam_1h.index, in_range_lam_1h['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax6.set_title('1 Hot', fontsize=22)
ax6.tick_params(labelsize=18)
#ax6.set_yscale('log')

ax7.scatter(in_range_lam_ssp.index, in_range_lam_ssp['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax7.set_title('SSP', fontsize=22)
ax7.set_xlabel('Experiment Index', fontsize=20)
ax7.xaxis.set_label_coords(-.11, -.15)
ax7.tick_params(labelsize=18)
#ax7.set_yscale('log')

ax8.scatter(in_range_lam_grid.index, in_range_lam_grid['goal_reached'], color = 'white', edgecolors = 'dimgrey')
ax8.set_title('Grid Cells', fontsize=22)
#ax8.set_xlabel('Experiment index', fontsize=14)
ax8.tick_params(labelsize=18)
#ax4.set_yscale('log')

plt.subplots_adjust(hspace=0.4)

## Save figure
print('Saving best parameter plots in: ./figures/nni_combined_bestparams.pdf')
fig.savefig("./figures/nni_combined_bestparams.pdf", bbox_inches="tight")