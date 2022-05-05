## Script for Getting the Plots for the TD( 𝜆 ) SSP Exploration Experiments ##

## import dependencies
# Set path
import sys
sys.path.insert(0, './network')

# General data manipulation packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Packages for saving and accessing files
import pytry
from pathlib import Path
import pickle

## Function to calculate number of trials needed to solve problem in each experiment 
## (i.e. how many trials needed to reach goal rolling mean)
def trials_to_goal(data, goal):
    ## create list for storing trial indices
    goal_reached = []
    
    ## for each value in roll_mean, record the index of values > goal
    for i in range(len(data)):
        try:
            a=[index for index,val in enumerate(data['roll_mean'][i]) if val > goal]
        except ValueError:
            a=[index for index,val in enumerate(data['roll_mean'][i][0]) if val > goal]
        
        ## if experiment never reached goal, put max number of runs in list
        if len(a) < 1:
            goal_reached.append(2000)
        ## otherwise, add the index to the list for plotting
        else:
            goal_reached.append(a[0])
        
    ## return list 
    return goal_reached

## function taken from seaborn for calculating confidence intervals
## https://github.com/mwaskom/seaborn/blob/master/seaborn/utils.py
def get_ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)

## Functions for converting confidence intervals to error bars
def ci_to_errsize(cis, heights):
    """Convert intervals to error arguments relative to plot heights.
    Parameters
    ----------
    cis : 2 x n sequence
        sequence of confidence interval limits
    heights : n sequence
        sequence of plot heights
    Returns
    -------
    errsize : 2 x n array
        sequence of error size relative to height values in correct
        format as argument for plt.bar
    """
    cis = np.atleast_2d(cis).reshape(2, -1)
    heights = np.atleast_1d(heights)
    errsize = []
    for i, (low, high) in enumerate(np.transpose(cis)):
        h = heights[i]
        elow = h - low
        ehigh = high - h
        errsize.append([elow, ehigh])

    errsize = np.asarray(errsize).T
    return errsize


## GET MAIN FIGURES FOR PAPER ##

folder = input("Enter path to data folder ")
data_folder = Path(folder)
print('path to folder:', data_folder)

folder = input("Enter path to figures folder ")
fig_folder = Path(folder)
print('path to folder:', fig_folder)

## Load data
mg_lam_ssp = pd.read_pickle(data_folder/'MG_TDLambda_SSP_long_dataframe')

## Add column with number of trials to reach goal rolling mean 
goal = 0.95
goal_reached = trials_to_goal(mg_lam_ssp, goal)
mg_lam_ssp['goal_reached'] = goal_reached

## Set all values to floats
mg_lam_ssp["alpha "] = pd.to_numeric(mg_lam_ssp["alpha "], downcast="float")
mg_lam_ssp["beta "] = pd.to_numeric(mg_lam_ssp["beta "], downcast="float")
mg_lam_ssp["gamma "] = pd.to_numeric(mg_lam_ssp["gamma "], downcast="float")
mg_lam_ssp["lambd "] = pd.to_numeric(mg_lam_ssp["lambd "], downcast="float")
mg_lam_ssp["n_neurons "] = pd.to_numeric(mg_lam_ssp["n_neurons "], downcast="float")
mg_lam_ssp["sparsity "] = pd.to_numeric(mg_lam_ssp["sparsity "], downcast="float")
mg_lam_ssp["dims "] = pd.to_numeric(mg_lam_ssp["dims "], downcast="float")
mg_lam_ssp["goal_reached"] = pd.to_numeric(mg_lam_ssp["goal_reached"], downcast="float")

## Plot Parallel Coordinates Plot
print('Plotting parallel coordinates plot')

fig = px.parallel_coordinates(mg_lam_ssp, color="goal_reached", 
                              labels={"alpha ": "Alpha",
                                "beta ": "Beta", "gamma ": "Gamma", "lambd ": "Lambda",
                                "n_neurons ": "Neurons", "sparsity ": "Sparsity", "dims ": "Dimensions",
                                "goal_reached": "N Trials"},
                             color_continuous_scale=px.colors.sequential.Aggrnyl,
                             width=1000, height=400)

fig.update_layout(
    xaxis_title="Hyperparameter",
    yaxis_title="Hyperparameter Value",
    legend_title="",
    font=dict(
        family="Arial, monospace",
        size=16
    )
)

fig.show()

## Save figure
print('Saving parallel coordinates plot in: ' + folder + '/SSP_lam_param_vals.pdf')
fig.write_image(fig_folder/"SSP_lam_param_vals.pdf")





## Dimensions Experiments ##
## load data
orig_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_orig_data')
dim512_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_512dims_data')
dim1024_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_1024dims_data')

## Add column with number of trials to reach goal rolling mean 
goal_reached = trials_to_goal(orig_data, goal)
orig_data['goal_reached'] = goal_reached

goal_reached = trials_to_goal(dim512_data, goal)
dim512_data['goal_reached'] = goal_reached

goal_reached = trials_to_goal(dim1024_data, goal)
dim1024_data['goal_reached'] = goal_reached

## Collect dimensions and trials to goal data into single data frame
dim_data = pd.DataFrame(columns = ['dims', 'goal_reached'])
dim_data['dims'] = pd.concat([orig_data['dims'], dim512_data['dims'], dim1024_data['dims']], axis=0)
dim_data['goal_reached'] = pd.concat([orig_data['goal_reached'], dim512_data['goal_reached'], dim1024_data['goal_reached']], axis=0)

## Calculate means and error bars
#get list of mean trials to goal for each dims experiment
means = pd.DataFrame(dim_data.groupby(['dims'])['goal_reached'].mean())
means_lst = list(means['goal_reached'])

#get confidence intervals
ci = [get_ci(dim_data[dim_data['dims'] == 256]['goal_reached'])]
ci.append(get_ci(dim_data[dim_data['dims'] == 512]['goal_reached']))
ci.append(get_ci(dim_data[dim_data['dims'] == 1024]['goal_reached']))

#convert confidence intervals to errors
errors = ci_to_errsize(ci, means_lst)

## Plot means and error bars
print('Plotting means and error bars for dimensions exploration.')

x_ticks = ['256', '512', '1024']
plt.errorbar(x_ticks, means_lst, yerr=errors, elinewidth=3, color = 'black', marker = 'o', markersize = 7)
plt.xlabel('N Dimensions', fontsize = 14)
plt.ylabel('N Trials to Goal', fontsize = 14)
#plt.title('Mean Trials to Goal Rolling Mean Reward', fontsize = 16)
plt.tick_params(labelsize = 12)

## Save figure
print('Saving plot in: ' + folder + '/SSPdims_means.pdf')
plt.savefig(fig_folder/"SSPdims_means.pdf", bbox_inches="tight")





## Neurons Experiments ##
## load data
n5000_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_5000neurons_data')
n6000_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_6000neurons_data')
n7000_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_7000neurons_data')
n8000_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_8000neurons_data')
n9000_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_9000neurons_data')
n10000_data = pd.read_pickle(data_folder/'MG_TDLambda_SSP_10000neurons_data')

## using above function, generate list of how quickly the goal was reached in each experiment
goal_reached = trials_to_goal(n5000_data, goal)
n5000_data['goal_reached'] = goal_reached

## repeat 
goal_reached = trials_to_goal(n6000_data, goal)
n6000_data['goal_reached'] = goal_reached

## repeat 
goal_reached = trials_to_goal(n7000_data, goal)
n7000_data['goal_reached'] = goal_reached

## repeat 
goal_reached = trials_to_goal(n8000_data, goal)
n8000_data['goal_reached'] = goal_reached

## repeat 
goal_reached = trials_to_goal(n9000_data, goal)
n9000_data['goal_reached'] = goal_reached

## repeat 
goal_reached = trials_to_goal(n10000_data, goal)
n10000_data['goal_reached'] = goal_reached

## Collect neuron and trials to goal data into single data frame
neuron_data = pd.DataFrame(columns = ['n_neurons', 'goal_reached'])
neuron_data['n_neurons'] = pd.concat([orig_data['n_neurons'], n5000_data['n_neurons'], n6000_data['n_neurons'], n7000_data['n_neurons'], 
                                  n8000_data['n_neurons'], n9000_data['n_neurons'], n10000_data['n_neurons']], axis=0)
neuron_data['goal_reached'] = pd.concat([orig_data['goal_reached'], n5000_data['goal_reached'], n6000_data['goal_reached'], n7000_data['goal_reached'], 
                                  n8000_data['goal_reached'], n9000_data['goal_reached'], n10000_data['goal_reached']], axis=0)

## Calculate means and error bars
#get list of mean trials to goal for each dims experiment
means = pd.DataFrame(neuron_data.groupby(['n_neurons'])['goal_reached'].mean())
means_lst = list(means['goal_reached'])

#get confidence intervals
ci = [get_ci(neuron_data[neuron_data['n_neurons'] == 4117]['goal_reached'])]
ci.append(get_ci(neuron_data[neuron_data['n_neurons'] == 5000]['goal_reached']))
ci.append(get_ci(neuron_data[neuron_data['n_neurons'] == 6000]['goal_reached']))
ci.append(get_ci(neuron_data[neuron_data['n_neurons'] == 7000]['goal_reached']))
ci.append(get_ci(neuron_data[neuron_data['n_neurons'] == 8000]['goal_reached']))
ci.append(get_ci(neuron_data[neuron_data['n_neurons'] == 9000]['goal_reached']))
ci.append(get_ci(neuron_data[neuron_data['n_neurons'] == 10000]['goal_reached']))

#convert confidence intervals to errors
errors = ci_to_errsize(ci, means_lst)

## Plot means and error bars
x_ticks = ['4117', '5000', '6000', '7000', '8000', '9000', '10000']
plt.errorbar(x_ticks, means_lst, yerr=errors, elinewidth=3, color = 'black', marker = 'o', markersize = 7)
plt.xlabel('N Neurons', fontsize = 14)
plt.ylabel('N Trials to Goal', fontsize = 14)
#plt.title('Mean Trials to Goal Rolling Mean Reward', fontsize = 16)
plt.tick_params(labelsize = 12)

## Save figure
print('Saving plot in: ' + folder + '/SSPneurons_means.pdf')
plt.savefig(fig_folder/"SSPneurons_means.pdf", bbox_inches="tight")